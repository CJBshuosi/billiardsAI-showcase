const { createRankingWorkerHandler } = require('../../cloudfunctions/ranking-worker/lib/handler')


function dependencies(settlement = { status: 'settled', rating_event_id: 'rating-1' }) {
  const calls = []
  return {
    calls,
    rankingScopeId: 'zhoukou-urban-dev',
    settlementQueue: {
      async claim(matchId) { calls.push(['claim', matchId]); return { match_id: matchId } },
      async claimNext(scopeId) { calls.push(['claim-next', scopeId]); return { match_id: 'match-next' } },
      async finish(job, settlement) { calls.push(['finish', job.match_id, settlement.status]) },
      async retry(job) { calls.push(['retry', job.match_id]) },
      async finishByMatch(matchId, settlement) { calls.push(['finish-by-match', matchId, settlement && settlement.status]) },
    },
    verifyToken: token => token === 'secret',
    pipeline: { async process(input) { calls.push(['process', input]); return settlement } },
    snapshotPublisher: { async publish(input) { calls.push(['publish', input]); return { status: 'published', snapshot_version: 2 } } },
    replayWorker: { async runNext(input) { calls.push(['replay', input]); return { status: 'completed', generation_id: 'g2' } } },
    reviewResolutionService: { async resolve(input) { calls.push(['resolve', input]); return { review: { status: 'released' }, settlement: { status: 'settled' } } } },
    expiryService: { async expireDue() { calls.push(['expire']); return { expired_results: 1, expired_challenges: 1 } } },
    developmentResetService: {
      async reset(input) { calls.push(['reset', input]); return { reset_id: input.resetId, season_id: 'dev-s2' } },
    },
    async readScope() { return { ranking_scope_id: 'zhoukou-urban-dev', status: 'approved', publication_status: 'open' } },
    async readSeason() { return { season_id: 's1', starts_at: '2026-09-01T00:00:00.000Z', ends_at: '2026-10-31T00:00:00.000Z' } },
    now: () => '2026-09-02T12:00:00.000Z',
  }
}


test('authenticated match processing publishes a new snapshot only after settlement', async () => {
  const deps = dependencies()
  const result = await createRankingWorkerHandler(deps)({
    schema_version: '1.0', token: 'secret', action: 'process.match', match_id: 'match-1',
  })
  expect(result).toMatchObject({ ok: true, data: { settlement: { status: 'settled' }, snapshot: { status: 'published' } } })
  expect(deps.calls.map(item => item[0])).toEqual(['claim', 'process', 'publish', 'finish'])
})


test('review holds do not publish snapshots and invalid worker tokens call nothing', async () => {
  const held = dependencies({ status: 'review_pending' })
  await createRankingWorkerHandler(held)({
    schema_version: '1.0', token: 'secret', action: 'process.match', match_id: 'match-1',
  })
  expect(held.calls.map(item => item[0])).toEqual(['claim', 'process', 'finish'])

  const denied = dependencies()
  const result = await createRankingWorkerHandler(denied)({
    schema_version: '1.0', token: 'wrong', action: 'publish.scope',
  })
  expect(result).toEqual({ ok: false, error: { code: 'WORKER_UNAUTHORIZED' } })
  expect(denied.calls).toEqual([])
})

test('completed replay publishes replacement snapshots through the same publication gate', async () => {
  const deps = dependencies()
  const result = await createRankingWorkerHandler(deps)({
    schema_version: '1.0', token: 'secret', action: 'replay.next',
  })
  expect(result).toMatchObject({ ok: true, data: { replay: { status: 'completed' }, snapshot: { status: 'published' } } })
  expect(deps.calls.map(item => item[0])).toEqual(['replay', 'publish'])
  expect(deps.calls[0]).toEqual(['replay', { rankingScopeId: 'zhoukou-urban-dev' }])
})

test('review release is audited then publishes only when ordinary settlement succeeds', async () => {
  const deps = dependencies()
  const result = await createRankingWorkerHandler(deps)({
    schema_version: '1.0', token: 'secret', action: 'review.resolve',
    match_id: 'match-1', reviewer_id: 'reviewer-1', resolution_type: 'review_released',
    payload: { resolution_code: 'evidence_confirmed' },
  })
  expect(result.ok).toBe(true)
  expect(deps.calls.map(item => item[0])).toEqual(['resolve', 'publish', 'finish-by-match'])
})


test('scheduled processing claims the next durable settlement job', async () => {
  const deps = dependencies()
  const result = await createRankingWorkerHandler(deps)({
    schema_version: '1.0', token: 'secret', action: 'process.next',
  })
  expect(result).toMatchObject({ ok: true, data: { settlement: { status: 'settled' } } })
  expect(deps.calls.map(item => item[0])).toEqual([
    'claim-next', 'process', 'publish', 'finish',
  ])
  expect(deps.calls[0]).toEqual(['claim-next', 'zhoukou-urban-dev'])
})


test('worker reports a sanitized internal failure without exposing it to callers', async () => {
  const deps = dependencies()
  const reported = []
  deps.reportError = error => reported.push(error)
  deps.settlementQueue.claimNext = async () => {
    const error = new Error('database query failed')
    error.code = 'DATABASE_QUERY_FAILED'
    throw error
  }

  const result = await createRankingWorkerHandler(deps)({
    schema_version: '1.0', token: 'secret', action: 'process.next',
  })

  expect(result).toEqual({ ok: false, error: { code: 'DATABASE_QUERY_FAILED' } })
  expect(reported).toEqual([{
    code: 'DATABASE_QUERY_FAILED', message: 'database query failed',
  }])
})


test('authenticated development reset uses the server scope and explicit reset id', async () => {
  const deps = dependencies()
  const result = await createRankingWorkerHandler(deps)({
    schema_version: '1.0', token: 'secret', action: 'development.reset',
    reset_id: 'acceptance-round-2',
  })

  expect(result).toEqual({
    ok: true, data: { reset_id: 'acceptance-round-2', season_id: 'dev-s2' },
  })
  expect(deps.calls).toEqual([['reset', {
    resetId: 'acceptance-round-2',
    scope: expect.objectContaining({ ranking_scope_id: 'zhoukou-urban-dev' }),
  }]])
})
