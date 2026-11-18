const {
  createDevelopmentRankingResetService,
} = require('../../cloudfunctions/shared/development-ranking-reset-service')
const {
  createDevelopmentRankingAdminRepository,
} = require('../../cloudfunctions/shared/development-ranking-admin-repository')


function scope() {
  return {
    ranking_scope_id: 'zhoukou-urban-dev', mode: 'development_acceptance', is_test: true,
  }
}


function repository() {
  const commits = []
  return {
    commits,
    async getReset() { return null },
    async readState() {
      return {
        season: {
          season_id: 'dev-s1', ranking_scope_id: 'zhoukou-urban-dev',
          is_test: true, status: 'active',
        },
        head: {
          ranking_scope_id: 'zhoukou-urban-dev', is_test: true,
          active_generation_id: 'generation-1', next_sequence: 5,
        },
      }
    },
    async commitReset(value) { commits.push(structuredClone(value)); return value.result },
  }
}


test('reset closes the old season and starts a fresh exact 60-day test generation', async () => {
  const repo = repository()
  const service = createDevelopmentRankingResetService({
    repository: repo,
    now: () => '2026-09-02T00:00:00.000Z',
  })

  const result = await service.reset({ scope: scope(), resetId: 'acceptance-round-2' })

  expect(result).toMatchObject({
    reset_id: 'acceptance-round-2', ranking_scope_id: 'zhoukou-urban-dev', is_test: true,
    previous_season_id: 'dev-s1', starts_at: '2026-09-02T00:00:00.000Z',
    ends_at: '2026-11-01T00:00:00.000Z',
  })
  expect(repo.commits[0]).toMatchObject({
    previousSeason: { season_id: 'dev-s1', status: 'closed' },
    nextSeason: { status: 'active', is_test: true },
    nextGeneration: { status: 'active', is_test: true },
    nextHead: { next_sequence: 1, is_test: true },
  })
})


test('reset is idempotent for one reset id', async () => {
  const repo = repository()
  repo.getReset = async () => ({
    reset_id: 'same-reset', result: { reset_id: 'same-reset', season_id: 'already-created' },
  })
  const service = createDevelopmentRankingResetService({ repository: repo, now: jest.fn() })

  await expect(service.reset({ scope: scope(), resetId: 'same-reset' }))
    .resolves.toEqual({ reset_id: 'same-reset', season_id: 'already-created' })
  expect(repo.commits).toEqual([])
})


test.each([
  { ranking_scope_id: 'zhoukou-urban', mode: 'formal', is_test: false },
  { ranking_scope_id: 'zhoukou-urban-dev', mode: 'development_acceptance', is_test: false },
])('reset refuses formal or unmarked scope %#', async invalidScope => {
  const service = createDevelopmentRankingResetService({
    repository: repository(), now: () => '2026-09-02T00:00:00.000Z',
  })
  await expect(service.reset({ scope: invalidScope, resetId: 'reset-1' }))
    .rejects.toMatchObject({ code: 'RANKING_TEST_RESET_FORBIDDEN' })
})


test('reset refuses a head or active season with conflicting test identity', async () => {
  const repo = repository()
  const original = repo.readState
  repo.readState = async () => {
    const state = await original()
    state.head.is_test = false
    return state
  }
  const service = createDevelopmentRankingResetService({
    repository: repo, now: () => '2026-09-02T00:00:00.000Z',
  })
  await expect(service.reset({ scope: scope(), resetId: 'reset-1' }))
    .rejects.toMatchObject({ code: 'RANKING_TEST_GENERATION_INVALID' })
})


test('cloud reset repository commits all reset documents atomically and preserves audit history', async () => {
  const names = [
    'ranking_test_resets', 'ranking_seasons', 'rating_scope_heads', 'rating_generations',
  ]
  const maps = Object.fromEntries(names.map(name => [name, new Map()]))
  const collection = name => ({
    doc(id) {
      return {
        async get() { return { data: maps[name].get(id) || null } },
        async set({ data }) { maps[name].set(id, { _id: id, ...structuredClone(data) }) },
        async update({ data }) {
          maps[name].set(id, { ...maps[name].get(id), ...structuredClone(data) })
        },
      }
    },
  })
  const db = { collection, async runTransaction(callback) { return callback({ collection }) } }
  const repo = createDevelopmentRankingAdminRepository(db)
  const input = {
    resetId: 'reset-2', expectedSeasonId: 'dev-s1', expectedGenerationId: 'generation-1',
    previousSeason: {
      season_id: 'dev-s1', ranking_scope_id: 'zhoukou-urban-dev',
      is_test: true, status: 'closed',
    },
    nextSeason: {
      season_id: 'dev-s2', ranking_scope_id: 'zhoukou-urban-dev',
      is_test: true, status: 'active',
    },
    nextGeneration: {
      generation_id: 'generation-2', ranking_scope_id: 'zhoukou-urban-dev',
      is_test: true, status: 'active',
    },
    nextHead: {
      ranking_scope_id: 'zhoukou-urban-dev', is_test: true,
      active_generation_id: 'generation-2', next_sequence: 1,
    },
    result: { reset_id: 'reset-2', season_id: 'dev-s2' },
  }
  const crypto = require('crypto')
  const id = value => crypto.createHash('sha256').update(value).digest('hex')
  maps.ranking_seasons.set(id('dev-s1'), {
    _id: id('dev-s1'), season_id: 'dev-s1', ranking_scope_id: 'zhoukou-urban-dev',
    is_test: true, status: 'active',
  })
  maps.rating_scope_heads.set(id('zhoukou-urban-dev'), {
    _id: id('zhoukou-urban-dev'), ranking_scope_id: 'zhoukou-urban-dev',
    is_test: true, active_generation_id: 'generation-1', next_sequence: 5,
  })

  await expect(repo.commitReset(input)).resolves.toEqual(input.result)
  expect([...maps.ranking_seasons.values()]).toEqual(expect.arrayContaining([
    expect.objectContaining({ season_id: 'dev-s1', status: 'closed' }),
    expect.objectContaining({ season_id: 'dev-s2', status: 'active' }),
  ]))
  expect([...maps.rating_generations.values()][0]).toMatchObject({ generation_id: 'generation-2' })
  expect([...maps.ranking_test_resets.values()][0]).toMatchObject({
    reset_id: 'reset-2', result: input.result,
  })
})
