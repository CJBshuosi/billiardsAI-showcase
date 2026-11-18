const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u


function createRankingWorkerHandler({
  verifyToken, pipeline, snapshotPublisher, replayWorker, reviewResolutionService,
  settlementQueue, expiryService, readScope, readSeason, now,
  rankingScopeId, developmentResetService,
  reportError = () => {},
}) {
  async function publish() {
    const [scope, season] = await Promise.all([readScope(), readSeason()])
    return snapshotPublisher.publish({ scope, season, asOf: now() })
  }

  async function processJob(matchId) {
    const job = matchId
      ? await settlementQueue.claim(matchId)
      : await settlementQueue.claimNext(rankingScopeId)
    if (!job) return { settlement: { status: 'idle' }, snapshot: null }
    try {
      const settlement = await pipeline.process({ matchId: job.match_id })
      const snapshot = settlement.status === 'settled' ? await publish() : null
      await settlementQueue.finish(job, settlement)
      return { settlement, snapshot }
    } catch (error) {
      await settlementQueue.retry(job, error)
      throw error
    }
  }

  return async event => {
    if (!event || event.schema_version !== '1.0' || !verifyToken(event.token)) {
      return { ok: false, error: { code: 'WORKER_UNAUTHORIZED' } }
    }
    try {
      if (event.action === 'process.match' && ID_PATTERN.test(event.match_id || '')) {
        return { ok: true, data: await processJob(event.match_id) }
      }
      if (event.action === 'process.next') {
        return { ok: true, data: await processJob(null) }
      }
      if (event.action === 'publish.scope') {
        return { ok: true, data: await publish() }
      }
      if (event.action === 'replay.next') {
        const replay = await replayWorker.runNext({ rankingScopeId })
        const snapshot = replay.status === 'completed' ? await publish() : null
        return { ok: true, data: { replay, snapshot } }
      }
      if (event.action === 'review.resolve' && ID_PATTERN.test(event.match_id || '')) {
        const resolution = await reviewResolutionService.resolve({
          matchId: event.match_id, reviewerId: event.reviewer_id,
          eventType: event.resolution_type, payload: event.payload,
        })
        const snapshot = resolution.settlement && resolution.settlement.status === 'settled'
          ? await publish() : null
        await settlementQueue.finishByMatch(event.match_id, resolution.settlement)
        return { ok: true, data: { resolution, snapshot } }
      }
      if (event.action === 'expire.due') {
        return { ok: true, data: await expiryService.expireDue() }
      }
      if (event.action === 'development.reset' && ID_PATTERN.test(event.reset_id || '')) {
        const scope = await readScope()
        return {
          ok: true,
          data: await developmentResetService.reset({ scope, resetId: event.reset_id }),
        }
      }
      return { ok: false, error: { code: 'WORKER_REQUEST_INVALID' } }
    } catch (error) {
      const diagnostic = {
        code: error && error.code || 'WORKER_INTERNAL_ERROR',
        message: String(error && error.message || ''),
      }
      reportError(diagnostic)
      return { ok: false, error: { code: diagnostic.code } }
    }
  }
}


module.exports = { createRankingWorkerHandler }
