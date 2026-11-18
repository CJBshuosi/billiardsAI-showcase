const { createRankingSnapshotPublisher } = require('../../cloudfunctions/shared/ranking-snapshot-publisher')
const { createSeason } = require('../../cloudfunctions/shared/ranking-snapshot')
const { initialSkillState } = require('../../cloudfunctions/shared/rating-engine')


const AS_OF = '2026-09-02T12:00:00.000Z'


function placed(userId, rating) {
  return {
    ...initialSkillState(userId), rating, deviation: 100, rated_matches: 5,
    distinct_opponent_ids: ['a', 'b', 'c'], last_rated_at: AS_OF,
  }
}


test('publisher creates long-term and 60-day season snapshots from real ledger state only', async () => {
  const published = []
  const repo = {
    async readPlayers() { return [
      { user_id: 'u1', public_profile: { is_public: true, nickname: '球友甲' }, long_term: placed('u1', 1300) },
      { user_id: 'private', public_profile: { is_public: false, nickname: '不公开' }, long_term: placed('private', 1500) },
    ] },
    async readRatingEvents() { return [] },
    async publish(value) { published.push(value); return { snapshot_set_id: value.snapshot_set_id, snapshot_version: 7 } },
  }
  const season = createSeason({ seasonId: 's1', startsAt: '2026-09-01T00:00:00.000Z' })
  const result = await createRankingSnapshotPublisher({ repository: repo }).publish({
    scope: {
      ranking_scope_id: 'zhoukou-urban', status: 'approved', publication_status: 'open',
      mode: 'formal', is_test: false,
    },
    season, asOf: AS_OF,
  })
  expect(result).toMatchObject({ status: 'published', snapshot_version: 7 })
  expect(published[0].long_term.entries).toEqual([
    expect.objectContaining({ user_id: 'u1', display_rating: 1300 }),
  ])
  expect(published[0].season.entries).toEqual([])
  expect(JSON.stringify(published[0])).not.toContain('不公开')
  expect(published[0]).toMatchObject({ is_test: false })
  expect(published[0].long_term).toMatchObject({ is_test: false })
  expect(published[0].season).toMatchObject({ is_test: false })
})


test('development publisher reads and writes only test-classified ranking data', async () => {
  const calls = []
  const repository = {
    async readPlayers(scopeId, isTest) { calls.push(['players', scopeId, isTest]); return [] },
    async readRatingEvents(scopeId, isTest) { calls.push(['events', scopeId, isTest]); return [] },
    async publish(payload) {
      calls.push(['publish', payload.ranking_scope_id, payload.is_test])
      return { snapshot_set_id: payload.snapshot_set_id, snapshot_version: 1 }
    },
  }
  const season = createSeason({ seasonId: 'dev-s1', startsAt: '2026-09-01T00:00:00.000Z' })

  await createRankingSnapshotPublisher({ repository }).publish({
    scope: {
      ranking_scope_id: 'zhoukou-urban-dev', status: 'approved', publication_status: 'open',
      mode: 'development_acceptance', is_test: true,
    },
    season: { ...season, is_test: true }, asOf: AS_OF,
  })

  expect(calls).toEqual([
    ['players', 'zhoukou-urban-dev', true],
    ['events', 'zhoukou-urban-dev', true],
    ['publish', 'zhoukou-urban-dev', true],
  ])
})


test('closed publication gate performs no reads and creates no empty official board', async () => {
  const repository = { readPlayers: jest.fn(), readRatingEvents: jest.fn(), publish: jest.fn() }
  const result = await createRankingSnapshotPublisher({ repository }).publish({
    scope: {
      ranking_scope_id: 'zhoukou-urban', status: 'approved', publication_status: 'closed',
      mode: 'formal', is_test: false,
    },
    season: createSeason({ seasonId: 's1', startsAt: '2026-09-01T00:00:00.000Z' }),
    asOf: AS_OF,
  })
  expect(result).toEqual({ status: 'not_published', reason_code: 'PUBLICATION_GATE_CLOSED' })
  expect(repository.readPlayers).not.toHaveBeenCalled()
})
