const crypto = require('crypto')

const { ALGORITHM_VERSION } = require('./rating-engine')


const SIXTY_DAYS_MS = 60 * 24 * 60 * 60 * 1000
const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u


class DevelopmentRankingResetError extends Error {
  constructor(code) {
    super(code)
    this.name = 'DevelopmentRankingResetError'
    this.code = code
  }
}


function fail(code) {
  throw new DevelopmentRankingResetError(code)
}


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


function createDevelopmentRankingResetService({ repository, now }) {
  return {
    async reset({ scope, resetId }) {
      if (!scope || scope.ranking_scope_id !== 'zhoukou-urban-dev'
          || scope.mode !== 'development_acceptance' || scope.is_test !== true
          || !ID_PATTERN.test(resetId || '')) {
        fail('RANKING_TEST_RESET_FORBIDDEN')
      }
      const existing = await repository.getReset(resetId)
      if (existing) return structuredClone(existing.result)
      const state = await repository.readState(scope.ranking_scope_id)
      if (!state || !state.season || !state.head
          || state.season.status !== 'active' || state.season.is_test !== true
          || state.head.is_test !== true || !state.head.active_generation_id) {
        fail('RANKING_TEST_GENERATION_INVALID')
      }
      const startsAt = now()
      const startsTime = Date.parse(startsAt)
      if (!Number.isFinite(startsTime)) fail('RANKING_TEST_RESET_FORBIDDEN')
      const endsAt = new Date(startsTime + SIXTY_DAYS_MS).toISOString()
      const seasonId = `dev-${digest(`${scope.ranking_scope_id}:${resetId}:season`).slice(0, 24)}`
      const generationId = digest(`${scope.ranking_scope_id}:${resetId}:${ALGORITHM_VERSION}`)
      const result = {
        reset_id: resetId, ranking_scope_id: scope.ranking_scope_id, is_test: true,
        previous_season_id: state.season.season_id, season_id: seasonId,
        generation_id: generationId, starts_at: startsAt, ends_at: endsAt,
      }
      return repository.commitReset({
        resetId,
        expectedSeasonId: state.season.season_id,
        expectedGenerationId: state.head.active_generation_id,
        previousSeason: {
          ...structuredClone(state.season), status: 'closed',
          ends_at: startsAt, closed_at: startsAt,
        },
        nextSeason: {
          schema_version: '1.0', season_id: seasonId,
          ranking_scope_id: scope.ranking_scope_id, is_test: true,
          status: 'active', starts_at: startsAt, ends_at: endsAt,
        },
        nextGeneration: {
          schema_version: '1.0', generation_id: generationId,
          ranking_scope_id: scope.ranking_scope_id, is_test: true,
          algorithm_version: ALGORITHM_VERSION, status: 'active', created_at: startsAt,
        },
        nextHead: {
          ...structuredClone(state.head), active_generation_id: generationId,
          next_sequence: 1, is_test: true, updated_at: startsAt,
        },
        result,
      })
    },
  }
}


module.exports = {
  DevelopmentRankingResetError,
  createDevelopmentRankingResetService,
}
