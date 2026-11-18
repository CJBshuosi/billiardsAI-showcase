const crypto = require('crypto')

const {
  ALGORITHM_VERSION,
  initialSkillState,
  updateMatchRatings,
} = require('./rating-engine')


class RatingLedgerError extends Error {
  constructor(code) {
    super(code)
    this.name = 'RatingLedgerError'
    this.code = code
  }
}


function fail(code) {
  throw new RatingLedgerError(code)
}


function ratingEventId(match) {
  return crypto.createHash('sha256')
    .update(`${ALGORITHM_VERSION}:${match.ranking_scope_id}:${match.match_id}:${match.result_revision}`, 'utf8')
    .digest('hex')
}


function validateMatch(match) {
  if (!match || match.schema_version !== '1.0'
      || typeof match.match_id !== 'string' || match.match_id.length === 0
      || !Number.isInteger(match.result_revision) || match.result_revision < 1
      || typeof match.ranking_scope_id !== 'string' || match.ranking_scope_id.length === 0
      || !Number.isInteger(match.scope_sequence) || match.scope_sequence < 1
      || typeof match.red_user_id !== 'string' || typeof match.blue_user_id !== 'string'
      || match.red_user_id === match.blue_user_id
      || !['red', 'blue'].includes(match.winner_role)
      || !['single', 'race'].includes(match.format)
      || !['active', 'revoked'].includes(match.status)
      || typeof match.started_at !== 'string' || !Number.isFinite(Date.parse(match.started_at))
      || typeof match.settled_at !== 'string' || !Number.isFinite(Date.parse(match.settled_at))) {
    fail('RATING_LEDGER_INVALID')
  }
}


function replayRatingLedger(inputMatches) {
  if (!Array.isArray(inputMatches)) fail('RATING_LEDGER_INVALID')
  const matches = structuredClone(inputMatches).sort((left, right) => left.scope_sequence - right.scope_sequence)
  const sequences = new Set()
  const matchIds = new Set()
  let rankingScopeId = null
  for (const match of matches) {
    validateMatch(match)
    if (sequences.has(match.scope_sequence)) fail('RATING_LEDGER_SEQUENCE_CONFLICT')
    if (matchIds.has(match.match_id)) fail('RATING_LEDGER_MATCH_CONFLICT')
    if (rankingScopeId !== null && rankingScopeId !== match.ranking_scope_id) fail('RATING_LEDGER_INVALID')
    sequences.add(match.scope_sequence)
    matchIds.add(match.match_id)
    rankingScopeId = match.ranking_scope_id
  }

  const states = {}
  const events = []
  for (const match of matches) {
    if (match.status === 'revoked') continue
    const red = states[match.red_user_id] || initialSkillState(match.red_user_id)
    const blue = states[match.blue_user_id] || initialSkillState(match.blue_user_id)
    const update = updateMatchRatings({
      red,
      blue,
      winnerRole: match.winner_role,
      format: match.format,
      settledAt: match.settled_at,
    })
    states[match.red_user_id] = update.after.red
    states[match.blue_user_id] = update.after.blue
    events.push({
      schema_version: '1.0',
      rating_event_id: ratingEventId(match),
      algorithm_version: ALGORITHM_VERSION,
      ranking_scope_id: match.ranking_scope_id,
      scope_sequence: match.scope_sequence,
      match_id: match.match_id,
      result_revision: match.result_revision,
      started_at: match.started_at,
      settled_at: match.settled_at,
      format: match.format,
      winner_role: match.winner_role,
      red_user_id: match.red_user_id,
      blue_user_id: match.blue_user_id,
      before: update.before,
      expected: update.expected,
      outcome: update.outcome,
      after: update.after,
      status: 'active',
    })
  }
  return {
    schema_version: '1.0',
    algorithm_version: ALGORITHM_VERSION,
    ranking_scope_id: rankingScopeId,
    events,
    states,
  }
}


module.exports = {
  RatingLedgerError,
  ratingEventId,
  replayRatingLedger,
}
