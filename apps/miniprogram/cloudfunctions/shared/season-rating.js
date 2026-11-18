const { updateMatchRatings } = require('./rating-engine')


function seasonBase(state, season) {
  return {
    ...structuredClone(state),
    season_id: season.season_id,
    rated_matches: 0,
    distinct_opponent_ids: [],
    last_rated_at: null,
    inherited_from_rating: state.rating,
    inherited_from_deviation: state.deviation,
  }
}


function replaySeasonRatings({ season, ratingEvents }) {
  if (!season || !Array.isArray(ratingEvents)
      || !Number.isFinite(Date.parse(season.starts_at))
      || !Number.isFinite(Date.parse(season.ends_at))) {
    throw new Error('SEASON_RATING_INVALID')
  }
  const startsAt = Date.parse(season.starts_at)
  const endsAt = Date.parse(season.ends_at)
  const events = structuredClone(ratingEvents)
    .filter(event => event.status === 'active'
      && Date.parse(event.started_at) >= startsAt
      && Date.parse(event.started_at) < endsAt)
    .sort((left, right) => left.scope_sequence - right.scope_sequence)
  const states = {}
  for (const event of events) {
    const red = states[event.red_user_id] || seasonBase(event.before.red, season)
    const blue = states[event.blue_user_id] || seasonBase(event.before.blue, season)
    const update = updateMatchRatings({
      red, blue, winnerRole: event.winner_role,
      format: event.format, settledAt: event.settled_at,
    })
    states[event.red_user_id] = update.after.red
    states[event.blue_user_id] = update.after.blue
  }
  return states
}


module.exports = { replaySeasonRatings }
