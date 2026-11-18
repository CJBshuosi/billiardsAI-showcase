const { buildRatingReplayRequest } = require('./rating-replay-request')


const RESOLUTION_TYPES = new Set([
  'review_released', 'review_not_ranked', 'result_corrected', 'result_revoked',
])


function createReviewResolutionService({
  reviewRepository, matchRepository, ratingEventRepository, pipeline,
}) {
  return {
    async resolve({ matchId, reviewerId, eventType, payload }) {
      if (typeof matchId !== 'string' || !/^reviewer-[A-Za-z0-9._-]+$/u.test(reviewerId || '')
          || !RESOLUTION_TYPES.has(eventType) || !payload || typeof payload !== 'object') {
        throw new Error('REVIEW_RESOLUTION_INVALID')
      }
      const match = await matchRepository.getMatch(matchId)
      if (!match || !match.projection.result) throw new Error('REVIEW_RESOLUTION_MATCH_NOT_FOUND')
      const review = await reviewRepository.appendEvent({
        matchId, resultRevision: match.projection.result.result_revision,
        eventType, actorId: reviewerId, payload,
      })
      if (eventType === 'review_released') {
        return { review, settlement: await pipeline.process({ matchId }), replay_request: null }
      }
      if (!['result_corrected', 'result_revoked'].includes(eventType)) {
        return { review, settlement: null, replay_request: null }
      }
      const ratingEvent = await ratingEventRepository.findActive({
        rankingScopeId: match.projection.ranking_scope_id, matchId,
      })
      if (!ratingEvent) return { review, settlement: null, replay_request: null }
      const request = buildRatingReplayRequest({ reviewState: review, ratingEvent })
      return {
        review, settlement: null,
        replay_request: await ratingEventRepository.saveReplayRequest(request),
      }
    },
  }
}


module.exports = { createReviewResolutionService }
