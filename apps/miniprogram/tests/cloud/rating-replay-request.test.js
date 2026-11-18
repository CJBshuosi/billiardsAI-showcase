const { buildRatingReplayRequest } = require('../../cloudfunctions/shared/rating-replay-request')


const LEDGER_EVENT = {
  generation_id: 'generation-1', ranking_scope_id: 'zhoukou-urban',
  scope_sequence: 12, match_id: 'match-001', result_revision: 1,
}


test.each([
  [
    {
      status: 'revoked', replay_required: true,
      resolution: {
        event_type: 'result_revoked', resolution_code: 'identity_conflict_unresolved',
        reviewer_id: 'reviewer-1', resolved_at: '2026-09-01T13:00:00.000Z',
      },
    },
    'revoke',
  ],
  [
    {
      status: 'corrected', replay_required: true,
      resolution: {
        event_type: 'result_corrected', resolution_code: 'confirmed_score_correction',
        winner_role: 'blue', red_score: 0, blue_score: 1,
        reviewer_id: 'reviewer-1', resolved_at: '2026-09-01T13:00:00.000Z',
      },
    },
    'correct',
  ],
])('%s review creates a replay request from the affected sequence', (reviewState, action) => {
  const request = buildRatingReplayRequest({ reviewState, ratingEvent: LEDGER_EVENT })

  expect(request).toMatchObject({
    schema_version: '1.0', ranking_scope_id: 'zhoukou-urban',
    source_generation_id: 'generation-1', from_scope_sequence: 12,
    affected_match_id: 'match-001', action, status: 'pending',
  })
  expect(request.replay_request_id).toMatch(/^[a-f0-9]{64}$/u)
  expect(JSON.stringify(request)).not.toContain('rating_delta')
})


test('release and not-ranked pre-settlement outcomes cannot request ledger mutation', () => {
  for (const reviewState of [
    { status: 'released', replay_required: false, resolution: { event_type: 'review_released' } },
    { status: 'not_ranked', replay_required: false, resolution: { event_type: 'review_not_ranked' } },
  ]) {
    expect(() => buildRatingReplayRequest({ reviewState, ratingEvent: LEDGER_EVENT }))
      .toThrow('RATING_REPLAY_NOT_REQUIRED')
  }
})
