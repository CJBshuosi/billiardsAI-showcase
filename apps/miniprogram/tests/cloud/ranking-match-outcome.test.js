const {
  buildParticipantOutcome,
} = require('../../cloudfunctions/client-api/lib/ranking-match-outcome')
const { initialSkillState } = require('../../cloudfunctions/shared/rating-engine')


function projection(status = 'fact_package_ready') {
  return {
    match_id: 'match-1', ranking_scope_id: 'zhoukou-urban', status,
    participants: { red: 'user-red', blue: 'user-blue' },
    result: { result_revision: 1, winner_role: 'red', red_score: 1, blue_score: 0 },
  }
}


test('review hold is neutral, appealable and never exposes risk reasons', () => {
  const result = buildParticipantOutcome({
    projection: projection(), actorUserId: 'user-red',
    qualification: { status: 'eligible' },
    review: {
      status: 'review_hold', appeals: [{ actor_id: 'user-blue' }],
      reason_codes: ['internal_device_signal'],
    },
    consumption: null, ratingEvent: null,
  })

  expect(result).toEqual({ status: 'review_pending', can_appeal: true, actor_appealed: false })
  expect(JSON.stringify(result)).not.toContain('internal_device_signal')
})


test('settled unplaced player sees progress but not hidden rating or delta', () => {
  const result = buildParticipantOutcome({
    projection: projection(), actorUserId: 'user-red',
    qualification: { status: 'eligible' }, review: null,
    consumption: { status: 'active', rating_event_id: 'event-1' },
    ratingEvent: {
      status: 'active', before: { red: initialSkillState('user-red') },
      after: { red: {
        ...initialSkillState('user-red'), rating: 1032, deviation: 180,
        rated_matches: 2, distinct_opponent_ids: ['a', 'b'],
      } },
    },
  })

  expect(result).toEqual({
    status: 'settled', placement_status: 'unplaced', rated_matches: 2,
    distinct_opponents: 2, required_matches: 3, required_opponents: 3,
  })
  expect(result).not.toHaveProperty('display_rating')
  expect(result).not.toHaveProperty('rating_delta')
})


test('settled placed player sees only the official ledger change', () => {
  const result = buildParticipantOutcome({
    projection: projection(), actorUserId: 'user-blue',
    qualification: { status: 'eligible' }, review: null,
    consumption: { status: 'active', rating_event_id: 'event-1' },
    ratingEvent: {
      status: 'active', before: { blue: {
        ...initialSkillState('user-blue'), rating: 1211.4, deviation: 100,
        rated_matches: 7, distinct_opponent_ids: ['a', 'b', 'c'],
      } },
      after: {
        blue: {
          ...initialSkillState('user-blue'), rating: 1198.2, deviation: 95,
          rated_matches: 8, distinct_opponent_ids: ['a', 'b', 'c'],
        },
      },
    },
  })

  expect(result).toEqual({
    status: 'settled', placement_status: 'placed', display_rating: 1198, rating_delta: -13,
  })
})


test.each([
  ['disputed', 'disputed'],
  ['confirmation_expired', 'confirmation_expired'],
])('facts status %s is reported before any settlement lookup', (factsStatus, expected) => {
  expect(buildParticipantOutcome({
    projection: projection(factsStatus), actorUserId: 'user-red',
    qualification: null, review: null, consumption: null, ratingEvent: null,
  })).toEqual({ status: expected })
})


test('a reviewed correction exposes the corrected public score without exposing reviewer metadata', () => {
  const result = buildParticipantOutcome({
    projection: projection(), actorUserId: 'user-red', qualification: { status: 'eligible' },
    review: {
      status: 'corrected', appeals: [],
      resolution: {
        event_type: 'result_corrected', resolution_code: 'confirmed_score_correction',
        winner_role: 'blue', red_score: 0, blue_score: 1,
        reviewer_id: 'reviewer-private', resolved_at: '2026-09-01T12:00:00.000Z',
      },
    },
    consumption: null, ratingEvent: null,
  })

  expect(result).toEqual({
    status: 'corrected', can_appeal: true, actor_appealed: false,
    corrected_result: { winner_role: 'blue', red_score: 0, blue_score: 1 },
  })
  expect(JSON.stringify(result)).not.toContain('reviewer-private')
})
