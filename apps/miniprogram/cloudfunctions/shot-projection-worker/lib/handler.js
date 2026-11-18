function createShotProjectionWorkerHandler({ repo, verifyToken }) {
  return async event => {
    if (!event || typeof event !== 'object' || Array.isArray(event)
        || typeof event.token !== 'string' || !verifyToken(event.token)) {
      return { ok: false, error: { code: 'SHOT_PROJECTION_UNAUTHORIZED' } }
    }
    if (event.action !== 'run_pending'
        || !Object.hasOwn(event, 'action') || !Object.hasOwn(event, 'token')
        || Object.keys(event).length !== 2) {
      return { ok: false, error: { code: 'SHOT_PROJECTION_REQUEST_INVALID' } }
    }
    try {
      const candidate = await repo.findPending()
      if (!candidate) return { ok: true, data: { outcome: 'idle', projected_count: 0 } }
      const result = await repo.projectBatch({ jobId: candidate.job_id })
      return { ok: true, data: {
        outcome: result.outcome, job_id: result.job_id,
        projected_count: result.projected_count, next_sequence: result.next_sequence,
      } }
    } catch (_) {
      return { ok: false, error: { code: 'SHOT_PROJECTION_UNAVAILABLE' } }
    }
  }
}

module.exports = { createShotProjectionWorkerHandler }
