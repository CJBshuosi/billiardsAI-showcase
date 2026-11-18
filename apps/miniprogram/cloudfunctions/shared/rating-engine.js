const ALGORITHM_VERSION = 'glicko-zhoukou-v1-candidate'
const INITIAL_RATING = 1000
const INITIAL_DEVIATION = 350
const MIN_DEVIATION = 50
const RACE_INFORMATION_WEIGHT = 1.25
const Q = Math.log(10) / 400


class RatingEngineError extends Error {
  constructor(code) {
    super(code)
    this.name = 'RatingEngineError'
    this.code = code
  }
}


function invalid() {
  throw new RatingEngineError('RATING_MATCH_INVALID')
}


function initialSkillState(userId) {
  if (typeof userId !== 'string' || userId.length === 0) invalid()
  return {
    schema_version: '1.0',
    algorithm_version: ALGORITHM_VERSION,
    user_id: userId,
    rating: INITIAL_RATING,
    deviation: INITIAL_DEVIATION,
    rated_matches: 0,
    distinct_opponent_ids: [],
    last_rated_at: null,
  }
}


function validState(state) {
  return state && typeof state.user_id === 'string' && state.user_id.length > 0
    && state.algorithm_version === ALGORITHM_VERSION
    && Number.isFinite(state.rating)
    && Number.isFinite(state.deviation)
    && state.deviation >= MIN_DEVIATION && state.deviation <= INITIAL_DEVIATION
    && Number.isInteger(state.rated_matches) && state.rated_matches >= 0
    && Array.isArray(state.distinct_opponent_ids)
}


function g(deviation) {
  return 1 / Math.sqrt(1 + (3 * Q * Q * deviation * deviation) / (Math.PI * Math.PI))
}


function expectedScore(player, opponent) {
  return 1 / (1 + 10 ** (-(g(opponent.deviation) * (player.rating - opponent.rating)) / 400))
}


function addOpponent(state, opponentId) {
  return [...new Set([...state.distinct_opponent_ids, opponentId])].sort()
}


function updateOne(player, opponent, score, expected, format, settledAt) {
  const opponentImpact = g(opponent.deviation)
  const information = Q * Q * opponentImpact * opponentImpact * expected * (1 - expected)
  const standardVariance = 1 / ((1 / (player.deviation * player.deviation)) + information)
  const informationWeight = format === 'race' ? RACE_INFORMATION_WEIGHT : 1
  const postDeviation = Math.sqrt(1 / (
    (1 / (player.deviation * player.deviation)) + informationWeight * information
  ))
  return {
    ...structuredClone(player),
    rating: player.rating + Q * standardVariance * opponentImpact * (score - expected),
    deviation: Math.max(MIN_DEVIATION, Math.min(INITIAL_DEVIATION, postDeviation)),
    rated_matches: player.rated_matches + 1,
    distinct_opponent_ids: addOpponent(player, opponent.user_id),
    last_rated_at: settledAt,
  }
}


function updateMatchRatings({ red, blue, winnerRole, format, settledAt }) {
  if (!validState(red) || !validState(blue) || red.user_id === blue.user_id
      || !['red', 'blue'].includes(winnerRole) || !['single', 'race'].includes(format)
      || typeof settledAt !== 'string' || !Number.isFinite(Date.parse(settledAt))) invalid()
  const before = { red: structuredClone(red), blue: structuredClone(blue) }
  const expected = {
    red: expectedScore(before.red, before.blue),
    blue: expectedScore(before.blue, before.red),
  }
  const score = winnerRole === 'red' ? { red: 1, blue: 0 } : { red: 0, blue: 1 }
  return {
    schema_version: '1.0',
    algorithm_version: ALGORITHM_VERSION,
    before,
    expected,
    outcome: score,
    after: {
      red: updateOne(before.red, before.blue, score.red, expected.red, format, settledAt),
      blue: updateOne(before.blue, before.red, score.blue, expected.blue, format, settledAt),
    },
  }
}


function placementProgress(state) {
  if (!validState(state)) invalid()
  const distinctOpponents = new Set(state.distinct_opponent_ids).size
  return {
    is_placed: state.rated_matches >= 3 && distinctOpponents >= 3,
    rated_matches: state.rated_matches,
    distinct_opponents: distinctOpponents,
    matches_needed: Math.max(0, 3 - state.rated_matches),
    distinct_opponents_needed: Math.max(0, 3 - distinctOpponents),
  }
}


module.exports = {
  ALGORITHM_VERSION,
  INITIAL_DEVIATION,
  INITIAL_RATING,
  MIN_DEVIATION,
  RACE_INFORMATION_WEIGHT,
  RatingEngineError,
  initialSkillState,
  placementProgress,
  updateMatchRatings,
}
