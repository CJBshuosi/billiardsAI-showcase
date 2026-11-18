const { ReviewEventError, reduceReviewEvents } = require('../../cloudfunctions/shared/review-events')


function event(sequence, type, actorId, payload) {
  return {
    schema_version: '1.0', event_id: `review-event-${sequence}`, match_id: 'match-001',
    sequence, event_type: type, actor_id: actorId,
    occurred_at: new Date(Date.parse('2026-09-01T12:00:00.000Z') + sequence * 1000).toISOString(),
    payload,
  }
}


test('review hold can be released without requesting rating replay', () => {
  const state = reduceReviewEvents([
    event(1, 'review_opened', 'system', { reason_codes: ['implausible_duration', 'repeated_opponent'] }),
    event(2, 'review_released', 'reviewer-1', { resolution_code: 'evidence_sufficient' }),
  ])

  expect(state).toMatchObject({
    status: 'released', replay_required: false, settlement_allowed: true, audit_count: 2,
  })
})


test.each([
  ['review_not_ranked', { resolution_code: 'insufficient_match_evidence' }, 'not_ranked', false],
  [
    'result_corrected',
    { resolution_code: 'confirmed_score_correction', winner_role: 'blue', red_score: 0, blue_score: 1 },
    'corrected',
    true,
  ],
  ['result_revoked', { resolution_code: 'identity_conflict_unresolved' }, 'revoked', true],
])('%s creates a structured resolution without direct score mutation', (
  eventType, payload, status, replayRequired,
) => {
  const state = reduceReviewEvents([
    event(1, 'review_opened', 'system', { reason_codes: ['wristband_identity_conflict', 'location_device_switch'] }),
    event(2, eventType, 'reviewer-1', payload),
  ])

  expect(state).toMatchObject({ status, replay_required: replayRequired, settlement_allowed: false })
  expect(state.resolution).toMatchObject({ event_type: eventType, resolution_code: payload.resolution_code })
})


test('a participant appeal is attached to one concrete match and preserves neutral status', () => {
  const state = reduceReviewEvents([
    event(1, 'review_opened', 'system', { reason_codes: ['independent_substitution_reports', 'location_device_switch'] }),
    event(2, 'appeal_submitted', 'user-red', { reason_code: 'identity_evidence_available' }),
  ])

  expect(state).toMatchObject({ status: 'review_hold', replay_required: false })
  expect(state.appeals).toEqual([{
    actor_id: 'user-red', reason_code: 'identity_evidence_available',
    submitted_at: '2026-09-01T12:00:02.000Z',
  }])
})


test('terminal resolution rejects a second reviewer outcome', () => {
  const events = [
    event(1, 'review_opened', 'system', { reason_codes: ['implausible_duration', 'repeated_trace_similarity'] }),
    event(2, 'result_revoked', 'reviewer-1', { resolution_code: 'match_evidence_invalid' }),
    event(3, 'review_released', 'reviewer-2', { resolution_code: 'override' }),
  ]

  expect(() => reduceReviewEvents(events)).toThrow(expect.objectContaining({
    name: 'ReviewEventError', code: 'REVIEW_EVENT_CONFLICT',
  }))
})


test('closed schemas reject an attempted direct rating edit', () => {
  const directEdit = event(1, 'review_opened', 'system', { reason_codes: ['implausible_duration'] })
  directEdit.payload.rating_delta = 500

  expect(() => reduceReviewEvents([directEdit])).toThrow(expect.objectContaining({
    code: 'REVIEW_EVENT_INVALID',
  }))
  expect(new ReviewEventError('REVIEW_EVENT_INVALID')).toMatchObject({
    message: 'REVIEW_EVENT_INVALID', code: 'REVIEW_EVENT_INVALID',
  })
})
