const crypto = require('crypto')

const { createRankingSnapshot } = require('./ranking-snapshot')
const { replaySeasonRatings } = require('./season-rating')


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


function createRankingSnapshotPublisher({ repository }) {
  return {
    async publish({ scope, season, asOf }) {
      if (!scope || scope.status !== 'approved' || scope.publication_status !== 'open') {
        return { status: 'not_published', reason_code: 'PUBLICATION_GATE_CLOSED' }
      }
      const [players, ratingEvents] = await Promise.all([
        repository.readPlayers(scope.ranking_scope_id, scope.is_test),
        repository.readRatingEvents(scope.ranking_scope_id, scope.is_test),
      ])
      const seasonStates = replaySeasonRatings({ season, ratingEvents })
      const snapshotPlayers = players.map(player => ({
        ...structuredClone(player), season: seasonStates[player.user_id] || null,
      }))
      const snapshotSetId = digest(`${scope.ranking_scope_id}:${season.season_id}:${asOf}`)
      const longTerm = createRankingSnapshot({
        snapshotId: `${snapshotSetId}-long`, rankingScopeId: scope.ranking_scope_id,
        kind: 'long_term', asOf, players: snapshotPlayers,
      })
      const seasonSnapshot = createRankingSnapshot({
        snapshotId: `${snapshotSetId}-season`, rankingScopeId: scope.ranking_scope_id,
        kind: 'season', season, asOf, players: snapshotPlayers,
      })
      const stored = await repository.publish({
        schema_version: '1.0', snapshot_set_id: snapshotSetId,
        ranking_scope_id: scope.ranking_scope_id, is_test: scope.is_test,
        season_id: season.season_id, as_of: asOf,
        long_term: { ...longTerm, is_test: scope.is_test },
        season: { ...seasonSnapshot, is_test: scope.is_test },
      })
      return {
        status: 'published', snapshot_set_id: stored.snapshot_set_id,
        snapshot_version: stored.snapshot_version,
      }
    },
  }
}


module.exports = { createRankingSnapshotPublisher }
