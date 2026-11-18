const crypto = require('crypto')


function fail(code) {
  const error = new Error(code)
  error.name = 'RatingReplayError'
  error.code = code
  throw error
}


function buildRatingReplayRequest({ reviewState, ratingEvent }) {
  if (!reviewState || reviewState.replay_required !== true || !reviewState.resolution) {
    fail('RATING_REPLAY_NOT_REQUIRED')
  }
  const actionByEvent = { result_revoked: 'revoke', result_corrected: 'correct' }
  const action = actionByEvent[reviewState.resolution.event_type]
  if (!action || !ratingEvent || !Number.isInteger(ratingEvent.scope_sequence)) {
    fail('RATING_REPLAY_INVALID')
  }
  const replayRequestId = crypto.createHash('sha256').update([
    ratingEvent.generation_id,
    ratingEvent.ranking_scope_id,
    ratingEvent.scope_sequence,
    ratingEvent.match_id,
    reviewState.resolution.event_type,
    reviewState.resolution.resolved_at,
  ].join(':'), 'utf8').digest('hex')
  const request = {
    schema_version: '1.0',
    replay_request_id: replayRequestId,
    ranking_scope_id: ratingEvent.ranking_scope_id,
    source_generation_id: ratingEvent.generation_id,
    from_scope_sequence: ratingEvent.scope_sequence,
    affected_match_id: ratingEvent.match_id,
    affected_result_revision: ratingEvent.result_revision,
    action,
    resolution_code: reviewState.resolution.resolution_code,
    requested_at: reviewState.resolution.resolved_at,
    status: 'pending',
  }
  if (action === 'correct') {
    request.corrected_result = {
      winner_role: reviewState.resolution.winner_role,
      red_score: reviewState.resolution.red_score,
      blue_score: reviewState.resolution.blue_score,
    }
  }
  return request
}


module.exports = { buildRatingReplayRequest }
