const { createRankingSettlementPipeline } = require('../../cloudfunctions/shared/ranking-settlement-pipeline')


const BASE_PROJECTION = {
  match_id: 'match-001', session_id: 'session-001', ranking_scope_id: 'zhoukou-urban',
  venue_id: 'venue-001', table_id: 'table-001', status: 'fact_package_ready',
  participants: { red: 'user-red', blue: 'user-blue' },
  settings: { mode: 'observation', ranking_intent: true, format: 'single', race_target: null },
  started_at: '2026-09-01T11:00:00.000Z',
  result: { result_revision: 1, winner_role: 'red', red_score: 1, blue_score: 0 },
}


const AUTHORITY = {
  configuration_version: 'zhoukou-ranking-v1',
  scope: {
    ranking_scope_id: 'zhoukou-urban', status: 'approved', mode: 'formal', is_test: false,
  },
  venue: { venue_id: 'venue-001', ranking_scope_id: 'zhoukou-urban', status: 'approved' },
  table: { table_id: 'table-001', venue_id: 'venue-001', status: 'approved' },
  session: {
    session_id: 'session-001', table_id: 'table-001', source_kind: 'live_capture',
    evidence_source_kind: 'usb_live',
    environment: 'production', device_deployment_status: 'approved',
    started_at: '2026-09-01T10:59:00.000Z',
    role_bindings: {
      red: { user_id: 'user-red', confirmed_at: '2026-09-01T10:58:00.000Z' },
      blue: { user_id: 'user-blue', confirmed_at: '2026-09-01T10:58:30.000Z' },
    },
  },
}


function dependencies({ projection = BASE_PROJECTION, hardConflicts = [], softSignals = [] } = {}) {
  const calls = { decisions: [], risks: [], reviews: [], settlements: [] }
  const match = {
    projection: structuredClone(projection),
    events: [
      { event_type: 'match_created', occurred_at: '2026-09-01T10:50:00.000Z' },
      { event_type: 'result_confirmed', occurred_at: '2026-09-01T12:00:00.000Z' },
      { event_type: 'result_confirmed', occurred_at: '2026-09-01T12:01:00.000Z' },
    ],
  }
  return {
    calls,
    matchFactsRepository: { async getMatch() { return structuredClone(match) } },
    authorityRepository: { async readForMatch() { return structuredClone(AUTHORITY) } },
    qualificationRepository: { async save(decision) { calls.decisions.push(structuredClone(decision)) } },
    riskRepository: {
      async collectSignals() { return { hardConflicts, softSignals } },
      async save(assessment) { calls.risks.push(structuredClone(assessment)) },
    },
    reviewRepository: { async open(input) { calls.reviews.push(structuredClone(input)) } },
    ratingRepository: {
      async settleEligibleMatch(input) {
        calls.settlements.push(structuredClone(input))
        return { rating_event_id: 'rating-001', scope_sequence: 1 }
      },
    },
  }
}


test('confirmed production facts flow through qualification and risk into one rating event', async () => {
  const deps = dependencies({ softSignals: ['repeated_opponent'] })
  const pipeline = createRankingSettlementPipeline(deps)

  const result = await pipeline.process({ matchId: 'match-001' })

  expect(result).toEqual({ status: 'settled', rating_event_id: 'rating-001', scope_sequence: 1 })
  expect(deps.calls.decisions[0]).toMatchObject({ status: 'eligible', reason_codes: ['ELIGIBLE'] })
  expect(deps.calls.risks[0]).toMatchObject({ status: 'clear_with_signals' })
  expect(deps.calls.settlements[0].match).toEqual({
    schema_version: '1.0', match_id: 'match-001', result_revision: 1,
    ranking_scope_id: 'zhoukou-urban', is_test: false,
    red_user_id: 'user-red', blue_user_id: 'user-blue',
    winner_role: 'red', format: 'single', started_at: '2026-09-01T11:00:00.000Z',
    settled_at: '2026-09-01T12:01:00.000Z',
  })
})


test('development qualification carries test identity into rating settlement', async () => {
  const projection = {
    ...BASE_PROJECTION,
    ranking_scope_id: 'zhoukou-urban-dev',
  }
  const deps = dependencies({ projection })
  deps.authorityRepository.readForMatch = async () => ({
    ...structuredClone(AUTHORITY),
    scope: {
      ranking_scope_id: 'zhoukou-urban-dev', status: 'approved',
      mode: 'development_acceptance', is_test: true,
    },
    venue: {
      ...AUTHORITY.venue, ranking_scope_id: 'zhoukou-urban-dev',
    },
    session: { ...structuredClone(AUTHORITY.session), environment: 'development' },
  })

  await createRankingSettlementPipeline(deps).process({ matchId: 'match-001' })

  expect(deps.calls.settlements[0]).toMatchObject({
    qualification: { is_test: true },
    match: { ranking_scope_id: 'zhoukou-urban-dev', is_test: true },
  })
})


test('a match that is not formally eligible stops before risk and rating', async () => {
  const deps = dependencies({ projection: { ...BASE_PROJECTION, status: 'confirmation_expired' } })
  const result = await createRankingSettlementPipeline(deps).process({ matchId: 'match-001' })

  expect(result).toEqual({ status: 'not_ranked', reason_codes: ['RESULT_CONFIRMATION_EXPIRED'] })
  expect(deps.calls.risks).toHaveLength(0)
  expect(deps.calls.settlements).toHaveLength(0)
})


test.each([
  [['duplicate_match_result'], [], 'blocked'],
  [[], ['implausible_duration', 'location_device_switch'], 'review_hold'],
])('hard conflicts and multi-signal holds never reach rating', async (hardConflicts, softSignals, status) => {
  const deps = dependencies({ hardConflicts, softSignals })
  const result = await createRankingSettlementPipeline(deps).process({ matchId: 'match-001' })

  expect(result.status).toBe(status === 'blocked' ? 'not_ranked' : 'review_pending')
  expect(deps.calls.settlements).toHaveLength(0)
  expect(deps.calls.reviews).toHaveLength(status === 'review_hold' ? 1 : 0)
})


test('missing or malformed persisted facts fail closed', async () => {
  const deps = dependencies()
  deps.matchFactsRepository.getMatch = async () => null
  await expect(createRankingSettlementPipeline(deps).process({ matchId: 'match-001' }))
    .rejects.toMatchObject({ code: 'RANKING_PIPELINE_MATCH_NOT_FOUND' })
  expect(deps.calls.settlements).toHaveLength(0)
})

test('a reviewer release allows a previously held match through the ordinary rating formula', async () => {
  const deps = dependencies({ softSignals: ['implausible_duration', 'location_device_switch'] })
  deps.reviewRepository.getState = async () => ({ status: 'released', settlement_allowed: true })
  const result = await createRankingSettlementPipeline(deps).process({ matchId: 'match-001' })
  expect(result.status).toBe('settled')
  expect(deps.calls.settlements).toHaveLength(1)
  expect(deps.calls.reviews).toHaveLength(0)
})
