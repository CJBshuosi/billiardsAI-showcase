const { placementProgress } = require('./rating-engine')


const DAY_MS = 24 * 60 * 60 * 1000


function canonicalTime(value) {
  if (typeof value !== 'string' || !Number.isFinite(Date.parse(value))) throw new Error('RANKING_TIME_INVALID')
  return Date.parse(value)
}


function createSeason({ seasonId, startsAt }) {
  if (typeof seasonId !== 'string' || seasonId.length === 0) throw new Error('SEASON_INVALID')
  const start = canonicalTime(startsAt)
  return {
    schema_version: '1.0',
    season_id: seasonId,
    starts_at: startsAt,
    ends_at: new Date(start + 60 * DAY_MS).toISOString(),
    duration_days: 60,
  }
}


function inheritSeasonSkill({ season, longTerm }) {
  return {
    schema_version: '1.0',
    algorithm_version: longTerm.algorithm_version,
    season_id: season.season_id,
    user_id: longTerm.user_id,
    rating: longTerm.rating,
    deviation: longTerm.deviation,
    rated_matches: 0,
    distinct_opponent_ids: [],
    last_rated_at: null,
    inherited_from_rating: longTerm.rating,
    inherited_from_deviation: longTerm.deviation,
  }
}


function createRankingSnapshot({
  snapshotId, rankingScopeId, kind, season = null, asOf, players,
}) {
  if (!['long_term', 'season'].includes(kind) || !Array.isArray(players)) {
    throw new Error('RANKING_SNAPSHOT_INVALID')
  }
  const asOfMs = canonicalTime(asOf)
  const candidates = players.filter(player => {
    if (!player.public_profile || player.public_profile.is_public !== true) return false
    if (!placementProgress(player.long_term).is_placed) return false
    return kind !== 'season' || (player.season
      && player.season.season_id === season.season_id
      && player.season.rated_matches >= 1)
  }).map(player => {
    const skill = kind === 'season' ? player.season : player.long_term
    const lastRated = skill.last_rated_at || player.long_term.last_rated_at
    return {
      user_id: player.user_id,
      nickname: player.public_profile.nickname,
      display_rating: Math.round(skill.rating),
      recent_sample_limited: validRecentSample(lastRated, asOfMs),
    }
  }).sort((left, right) => (
    right.display_rating - left.display_rating || left.user_id.localeCompare(right.user_id)
  ))

  let previousRating = null
  let previousRank = 0
  const entries = candidates.map((candidate, index) => {
    const rank = candidate.display_rating === previousRating ? previousRank : index + 1
    previousRating = candidate.display_rating
    previousRank = rank
    return { rank, ...candidate }
  })
  return {
    schema_version: '1.0',
    snapshot_id: snapshotId,
    snapshot_version: 1,
    ranking_scope_id: rankingScopeId,
    kind,
    season_id: kind === 'season' ? season.season_id : null,
    as_of: asOf,
    total_entries: entries.length,
    entries,
  }
}


function validRecentSample(lastRatedAt, asOfMs) {
  if (typeof lastRatedAt !== 'string' || !Number.isFinite(Date.parse(lastRatedAt))) return true
  return asOfMs - Date.parse(lastRatedAt) >= 90 * DAY_MS
}


module.exports = { createRankingSnapshot, createSeason, inheritSeasonSkill }
