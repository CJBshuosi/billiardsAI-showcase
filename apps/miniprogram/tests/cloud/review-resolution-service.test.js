const { createReviewResolutionService } = require('../../cloudfunctions/shared/review-resolution-service')


function reviewState(eventType) {
  return {
    status: eventType === 'review_released' ? 'released' : 'revoked',
    settlement_allowed: eventType === 'review_released', replay_required: eventType === 'result_revoked',
    resolution: { event_type: eventType, resolution_code: 'reviewed', resolved_at: '2026-09-02T12:00:00.000Z' },
  }
}


test('review release re-enters the ordinary pipeline instead of editing points', async () => {
  const calls = []
  const service = createReviewResolutionService({
    reviewRepository: { async appendEvent() { return reviewState('review_released') } },
    matchRepository: { async getMatch() { return { projection: { result: { result_revision: 1 } } } } },
    ratingEventRepository: {},
    pipeline: { async process(input) { calls.push(input); return { status: 'settled' } } },
  })
  const result = await service.resolve({
    matchId: 'match-1', reviewerId: 'reviewer-1', eventType: 'review_released',
    payload: { resolution_code: 'reviewed' },
  })
  expect(result).toMatchObject({ review: { status: 'released' }, settlement: { status: 'settled' } })
  expect(calls).toEqual([{ matchId: 'match-1' }])
})


test('post-settlement revocation creates a replay request and never accepts a score delta', async () => {
  const saved = []
  const service = createReviewResolutionService({
    reviewRepository: { async appendEvent() { return reviewState('result_revoked') } },
    matchRepository: { async getMatch() { return { projection: {
      result: { result_revision: 1 }, ranking_scope_id: 'zhoukou-urban',
    } } } },
    ratingEventRepository: {
      async findActive() { return { generation_id: 'g1', ranking_scope_id: 'zhoukou-urban', scope_sequence: 2, match_id: 'match-1', result_revision: 1 } },
      async saveReplayRequest(value) { saved.push(value); return value },
    },
    pipeline: { async process() { throw new Error('must not settle directly') } },
  })
  const result = await service.resolve({
    matchId: 'match-1', reviewerId: 'reviewer-1', eventType: 'result_revoked',
    payload: { resolution_code: 'reviewed' },
  })
  expect(result.replay_request.action).toBe('revoke')
  expect(JSON.stringify(saved)).not.toContain('rating_delta')
})
