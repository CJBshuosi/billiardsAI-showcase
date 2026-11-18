const {
  createCloudRankingRepository,
  getCityRanking,
  getPublicRankingPlayer,
  loadRuntimeRankingScope,
} = require('../../cloudfunctions/client-api/lib/ranking')
const { initialSkillState } = require('../../cloudfunctions/shared/rating-engine')


function repository(override = {}) {
  return {
    async getLatestSnapshot() { return null },
    async getSkillState(_scope, userId) {
      return {
        ...initialSkillState(userId), rating: 1188.4, deviation: 120, rated_matches: 2,
        distinct_opponent_ids: ['a', 'b'], last_rated_at: '2026-08-31T12:00:00.000Z',
      }
    },
    async getPublicProfile(userId) {
      return { user_id: userId, nickname: '周口球友', is_public: true, accept_challenges: true }
    },
    ...override,
  }
}


const developmentScope = {
  ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
  status: 'approved', publication_status: 'open',
  mode: 'development_acceptance', is_test: true,
}


test('runtime ranking scope is loaded from server storage and validated for development', async () => {
  const reads = []
  const result = await loadRuntimeRankingScope({
    environment: 'development',
    repo: repository({
      async getScope(scopeId) { reads.push(scopeId); return structuredClone(developmentScope) },
    }),
  })

  expect(reads).toEqual(['zhoukou-urban-dev'])
  expect(result).toEqual(developmentScope)
})


test('runtime ranking scope rejects a formal document in development', async () => {
  await expect(loadRuntimeRankingScope({
    environment: 'development',
    repo: repository({
      async getScope() {
        return {
          ranking_scope_id: 'zhoukou-urban', display_name: '周口市区',
          mode: 'formal', is_test: false,
        }
      },
    }),
  })).rejects.toThrow('RANKING_ENVIRONMENT_SCOPE_MISMATCH')
})


test('development board returns authoritative test scope metadata', async () => {
  const result = await getCityRanking({
    userId: 'me', rankingScopeId: 'zhoukou-urban-dev', rankingScope: developmentScope,
    kind: 'long_term', repo: repository(),
  })

  expect(result.ranking_scope).toEqual({
    ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
    mode: 'development_acceptance', is_test: true,
  })
})


test('cloud ranking repository exposes approved-table lookup used by match setup', async () => {
  const reads = []
  const repo = createCloudRankingRepository({
    collection(name) {
      return {
        doc(id) {
          reads.push({ name, id })
          return { async get() { return { data: { _id: id, table_id: 'table-1', status: 'approved' } } } }
        },
      }
    },
  })

  await expect(repo.getTable('table-1')).resolves.toEqual({ table_id: 'table-1', status: 'approved' })
  expect(reads).toEqual([{ name: 'ranking_tables', id: expect.stringMatching(/^[a-f0-9]{64}$/u) }])
})


test('missing published snapshot returns an honest unavailable state without invented entries', async () => {
  const result = await getCityRanking({
    userId: 'me', rankingScopeId: 'zhoukou-urban', kind: 'long_term', repo: repository(),
  })
  expect(result).toEqual({
    availability: 'unavailable', reason_code: 'RANKING_NOT_PUBLISHED',
    ranking_scope_id: 'zhoukou-urban', kind: 'long_term', entries: [], my_position: null,
    my_status: { status: 'unplaced', rated_matches: 2, distinct_opponents: 2, required_matches: 3, required_opponents: 3 },
  })
  expect(JSON.stringify(result)).not.toContain('1188')
})


test('published board and my position come from the same immutable snapshot', async () => {
  const snapshot = {
    snapshot_id: 'snapshot-9', snapshot_version: 9, ranking_scope_id: 'zhoukou-urban',
    kind: 'long_term', season_id: null, as_of: '2026-09-01T12:00:00.000Z', total_entries: 2,
    entries: [
      { rank: 1, user_id: 'other', nickname: '第一名', display_rating: 1300, recent_sample_limited: false },
      { rank: 2, user_id: 'me', nickname: '我的昵称', display_rating: 1200, recent_sample_limited: false },
    ],
  }
  const result = await getCityRanking({
    userId: 'me', rankingScopeId: 'zhoukou-urban', kind: 'long_term',
    repo: repository({ async getLatestSnapshot() { return structuredClone(snapshot) } }),
  })
  expect(result).toMatchObject({
    availability: 'published', snapshot_id: 'snapshot-9', snapshot_version: 9,
    entries: [{ user_id: 'other' }, { user_id: 'me' }],
    my_position: { rank: 2, display_rating: 1200 },
    my_status: { status: 'placed' },
  })
  expect(Object.values(result).some(value => typeof value === 'function')).toBe(false)
})


test('unplaced public player details never expose hidden rating', async () => {
  const result = await getPublicRankingPlayer({
    targetUserId: 'target', rankingScopeId: 'zhoukou-urban', repo: repository(),
  })
  expect(result).toEqual({
    user_id: 'target', nickname: '周口球友', accept_challenges: true,
    ranking_status: 'unplaced', rated_matches: 2, distinct_opponents: 2,
    required_matches: 3, required_opponents: 3,
  })
  expect(result).not.toHaveProperty('display_rating')
})


test('development player details carry the same authoritative test scope metadata', async () => {
  const result = await getPublicRankingPlayer({
    targetUserId: 'target', rankingScopeId: 'zhoukou-urban-dev',
    rankingScope: developmentScope, repo: repository(),
  })

  expect(result.ranking_scope).toEqual({
    ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
    mode: 'development_acceptance', is_test: true,
  })
})


test('private player profile is unavailable', async () => {
  await expect(getPublicRankingPlayer({
    targetUserId: 'private', rankingScopeId: 'zhoukou-urban',
    repo: repository({ async getPublicProfile() { return { is_public: false } } }),
  })).rejects.toMatchObject({ code: 'RANKING_PLAYER_UNAVAILABLE' })
})


test('player details can be pinned to the exact published snapshot opened from the board', async () => {
  const reads = []
  const result = await getPublicRankingPlayer({
    targetUserId: 'target', rankingScopeId: 'zhoukou-urban', snapshotId: 'snapshot-9',
    repo: repository({
      async getSkillState() {
        return {
          ...initialSkillState('target'), rating: 1222, deviation: 90, rated_matches: 4,
          distinct_opponent_ids: ['a', 'b', 'c'],
        }
      },
      async getSnapshotById(snapshotId) {
        reads.push(snapshotId)
        return {
          snapshot_id: 'snapshot-9', snapshot_version: 9,
          ranking_scope_id: 'zhoukou-urban', kind: 'long_term', status: 'published',
          entries: [],
        }
      },
      async getSnapshotEntry(snapshotId, userId) {
        expect([snapshotId, userId]).toEqual(['snapshot-9', 'target'])
        return { rank: 3, user_id: 'target', display_rating: 1222 }
      },
      async getLatestSnapshot() { throw new Error('must use pinned snapshot') },
    }),
  })

  expect(reads).toEqual(['snapshot-9'])
  expect(result).toMatchObject({ snapshot_id: 'snapshot-9', snapshot_version: 9, rank: 3 })
})


test('paged board resolves my position from the same snapshot even when I am off-page', async () => {
  const result = await getCityRanking({
    userId: 'me', rankingScopeId: 'zhoukou-urban', kind: 'long_term', page: 1, pageSize: 1,
    repo: repository({
      async getLatestSnapshot() {
        return {
          snapshot_id: 'snapshot-20', snapshot_version: 20,
          ranking_scope_id: 'zhoukou-urban', kind: 'long_term',
          as_of: '2026-09-01T12:00:00.000Z', total_entries: 2,
          entries: [{ rank: 1, user_id: 'other', nickname: '第一名', display_rating: 1300 }],
        }
      },
      async getSnapshotEntry(snapshotId, userId) {
        expect([snapshotId, userId]).toEqual(['snapshot-20', 'me'])
        return { rank: 2, user_id: 'me', nickname: '本人', display_rating: 1200 }
      },
    }),
  })

  expect(result).toMatchObject({
    page: 1, page_size: 1, has_more: true,
    my_position: { rank: 2, user_id: 'me', display_rating: 1200 },
  })
})
