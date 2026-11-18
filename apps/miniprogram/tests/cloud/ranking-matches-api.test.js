const { createRankingMatchApi } = require('../../cloudfunctions/client-api/lib/ranking-matches')


function match(status = 'awaiting_settings_confirmations') {
  return {
    projection: {
      match_id: 'match-1', session_id: 'session-1', status, latest_sequence: 2,
      ranking_scope_id: 'zhoukou-urban-dev',
      participants: { red: 'red', blue: 'blue' },
      settings: { settings_revision: 1, mode: 'observation', format: 'single', race_target: null, first_breaker: 'red', ranking_intent: true },
      settings_confirmations: ['red'], result: null, result_confirmations: [],
      result_confirmation_deadline_at: null,
    }, events: [],
  }
}


function dependencies() {
  const calls = []
  return {
    calls,
    matchRepository: {
      async getBySession() { return match() },
      async getMatch() { return match('awaiting_result_confirmations') },
      async listPendingForUser() { return [match('awaiting_result_confirmations')] },
    },
    fieldRepository: { async getAuthorizationContext() { return {
      session_id: 'session-1', table_id: 'table-1', joined: true, started: false,
      red_user_id: 'red', blue_user_id: 'blue',
    } } },
    rankingRepository: {
      async getScope() { return {
        ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
        mode: 'development_acceptance', is_test: true,
      } },
      async getTable() { return { table_id: 'table-1', venue_id: 'venue-1', status: 'approved' } },
      async getVenue() { return { venue_id: 'venue-1', ranking_scope_id: 'zhoukou-urban', status: 'approved' } },
    },
    outcomeRepository: {
      async read() { return { qualification: null, review: null, consumption: null, ratingEvent: null } },
    },
    setupService: {
      async proposeSettings(input) { calls.push(['propose', input]); return match().projection },
      async confirmSettings(input) { calls.push(['confirm-settings', input]); return match('ready').projection },
    },
    resultService: {
      async confirm(input) { calls.push(['confirm-result', input]); return match('fact_package_ready').projection },
      async dispute(input) { calls.push(['dispute-result', input]); return match('disputed').projection },
    },
  }
}


test('settings proposal builds ranking authority from server records, not client venue claims', async () => {
  const deps = dependencies(); const api = createRankingMatchApi(deps)
  const result = await api.proposeSettings({
    sessionId: 'session-1', actorUserId: 'red', mode: 'observation', format: 'single',
    raceTarget: null, firstBreaker: 'red', rankingIntent: true,
  })
  expect(result).toMatchObject({ status: 'awaiting_settings_confirmations', actor_role: 'red' })
  expect(deps.calls[0][1].context).toEqual({
    session_id: 'session-1', table_id: 'table-1', venue_id: 'venue-1',
    ranking_scope_id: 'zhoukou-urban', red_user_id: 'red', blue_user_id: 'blue',
  })
})


test('pending results are participant-only and do not expose an estimated rating change', async () => {
  const deps = dependencies(); const api = createRankingMatchApi(deps)
  const pending = await api.listPendingResults({ actorUserId: 'red' })
  expect(pending[0]).toMatchObject({ match_id: 'match-1', actor_role: 'red', status: 'awaiting_result_confirmations' })
  expect(JSON.stringify(pending)).not.toContain('rating_delta')
  await api.confirmResult({ matchId: 'match-1', actorUserId: 'red' })
  expect(deps.calls.at(-1)).toEqual(['confirm-result', { matchId: 'match-1', actorUserId: 'red' }])
})


test('outsiders cannot read or confirm a match result', async () => {
  const api = createRankingMatchApi(dependencies())
  await expect(api.get({ sessionId: 'session-1', actorUserId: 'outsider' }))
    .rejects.toMatchObject({ code: 'RANKING_MATCH_FORBIDDEN' })
  await expect(api.confirmResult({ matchId: 'match-1', actorUserId: 'outsider' }))
    .rejects.toMatchObject({ code: 'RANKING_MATCH_FORBIDDEN' })
})


test('match read exposes a participant-safe persistent settlement outcome', async () => {
  const deps = dependencies()
  const stored = match('fact_package_ready')
  stored.projection.result = {
    result_revision: 1, winner_role: 'red', red_score: 1, blue_score: 0,
  }
  deps.matchRepository.getBySession = async () => stored
  deps.outcomeRepository.read = async () => ({
    qualification: { status: 'eligible' }, review: { status: 'review_hold', appeals: [] },
    consumption: null, ratingEvent: null,
  })
  const api = createRankingMatchApi(deps)

  const result = await api.get({ sessionId: 'session-1', actorUserId: 'red' })

  expect(result.ranking_outcome).toEqual({
    status: 'review_pending', can_appeal: true, actor_appealed: false,
  })
  expect(JSON.stringify(result.ranking_outcome)).not.toContain('reason_codes')
  expect(result.ranking_scope).toEqual({
    ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
    mode: 'development_acceptance', is_test: true,
  })
})
