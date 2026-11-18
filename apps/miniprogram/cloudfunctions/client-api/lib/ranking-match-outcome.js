const crypto = require('crypto')

const { placementProgress } = require('../../shared/rating-engine')
const { ClientApiError } = require('./errors')


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


async function optional(ref) {
  try {
    const result = await ref.get()
    return result && result.data ? result.data : null
  } catch (error) {
    if (String(error && error.message).includes('does not exist')) return null
    throw error
  }
}


function clean(value) {
  if (!value) return null
  const { _id: _documentId, ...document } = value
  return structuredClone(document)
}


function actorRole(projection, actorUserId) {
  if (projection && projection.participants && projection.participants.red === actorUserId) return 'red'
  if (projection && projection.participants && projection.participants.blue === actorUserId) return 'blue'
  throw new ClientApiError('RANKING_MATCH_FORBIDDEN')
}


function buildParticipantOutcome({
  projection, actorUserId, qualification, review, consumption, ratingEvent,
}) {
  const role = actorRole(projection, actorUserId)
  if (['awaiting_result_confirmations', 'disputed', 'confirmation_expired'].includes(projection.status)) {
    return { status: projection.status === 'awaiting_result_confirmations' ? 'awaiting_confirmation' : projection.status }
  }
  if (projection.status === 'revoked') return { status: 'revoked' }
  if (review) {
    if (review.status === 'review_hold') {
      return {
        status: 'review_pending', can_appeal: true,
        actor_appealed: Array.isArray(review.appeals)
          && review.appeals.some(appeal => appeal.actor_id === actorUserId),
      }
    }
    if (['not_ranked', 'corrected', 'revoked'].includes(review.status)) {
      const result = { status: review.status, can_appeal: true,
        actor_appealed: Array.isArray(review.appeals)
          && review.appeals.some(appeal => appeal.actor_id === actorUserId) }
      if (review.status === 'corrected' && review.resolution) {
        result.corrected_result = {
          winner_role: review.resolution.winner_role,
          red_score: review.resolution.red_score,
          blue_score: review.resolution.blue_score,
        }
      }
      return result
    }
  }
  if (qualification && qualification.status !== 'eligible') {
    return { status: qualification.status === 'revoked' ? 'revoked' : 'not_ranked' }
  }
  if (!consumption || consumption.status !== 'active' || !ratingEvent || ratingEvent.status !== 'active') {
    return { status: 'processing' }
  }
  const before = ratingEvent.before && ratingEvent.before[role]
  const after = ratingEvent.after && ratingEvent.after[role]
  if (!before || !after) throw new ClientApiError('RANKING_MATCH_OUTCOME_INVALID')
  const progress = placementProgress(after)
  if (!progress.is_placed) {
    return {
      status: 'settled', placement_status: 'unplaced',
      rated_matches: progress.rated_matches,
      distinct_opponents: progress.distinct_opponents,
      required_matches: 3, required_opponents: 3,
    }
  }
  return {
    status: 'settled', placement_status: 'placed',
    display_rating: Math.round(after.rating),
    rating_delta: Math.round(after.rating) - Math.round(before.rating),
  }
}


function createCloudRankingMatchOutcomeRepository(db) {
  return {
    async read(projection) {
      if (!projection || !projection.result) {
        return { qualification: null, review: null, consumption: null, ratingEvent: null }
      }
      const revision = projection.result.result_revision
      const scopeId = projection.ranking_scope_id
      const matchId = projection.match_id
      const [qualification, reviewDocument, consumption] = await Promise.all([
        optional(db.collection('match_qualification_decisions').doc(digest(`${matchId}:${revision}`))),
        optional(db.collection('match_review_states').doc(digest(`${matchId}:${revision}`))),
        optional(db.collection('rating_match_consumptions').doc(digest(`${scopeId}:${matchId}`))),
      ])
      const ratingEvent = consumption && consumption.rating_event_id
        ? await optional(db.collection('rating_events').doc(consumption.rating_event_id)) : null
      return {
        qualification: clean(qualification),
        review: reviewDocument ? structuredClone(reviewDocument.state) : null,
        consumption: clean(consumption), ratingEvent: clean(ratingEvent),
      }
    },
  }
}


module.exports = { buildParticipantOutcome, createCloudRankingMatchOutcomeRepository }
