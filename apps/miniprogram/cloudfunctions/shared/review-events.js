const EVENT_KEYS = [
  'actor_id', 'event_id', 'event_type', 'match_id', 'occurred_at', 'payload',
  'schema_version', 'sequence',
]

const PAYLOAD_KEYS = {
  review_opened: ['reason_codes'],
  review_released: ['resolution_code'],
  review_not_ranked: ['resolution_code'],
  result_corrected: ['blue_score', 'red_score', 'resolution_code', 'winner_role'],
  result_revoked: ['resolution_code'],
  appeal_submitted: ['reason_code'],
}

const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u


class ReviewEventError extends Error {
  constructor(code) {
    super(code)
    this.name = 'ReviewEventError'
    this.code = code
  }
}


function fail(code) {
  throw new ReviewEventError(code)
}


function exactKeys(value, keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const actual = Object.keys(value).sort()
  return actual.length === keys.length && actual.every((key, index) => key === keys[index])
}


function validCode(value) {
  return typeof value === 'string' && ID_PATTERN.test(value)
}


function validateEvent(item, index, firstMatchId, previousTime, eventIds) {
  if (!exactKeys(item, EVENT_KEYS) || item.schema_version !== '1.0'
      || !validCode(item.event_id) || !validCode(item.match_id) || !validCode(item.actor_id)
      || !Object.hasOwn(PAYLOAD_KEYS, item.event_type)
      || !exactKeys(item.payload, PAYLOAD_KEYS[item.event_type])
      || item.sequence !== index + 1 || item.match_id !== firstMatchId
      || eventIds.has(item.event_id)) fail('REVIEW_EVENT_INVALID')
  const milliseconds = Date.parse(item.occurred_at)
  if (!Number.isFinite(milliseconds) || milliseconds < previousTime) fail('REVIEW_EVENT_INVALID')
  return milliseconds
}


function validatePayload(item) {
  const { payload } = item
  if (item.event_type === 'review_opened') {
    if (!Array.isArray(payload.reason_codes) || payload.reason_codes.length === 0
        || payload.reason_codes.some(code => !validCode(code))) fail('REVIEW_EVENT_INVALID')
  } else if (item.event_type === 'result_corrected') {
    if (!validCode(payload.resolution_code) || !['red', 'blue'].includes(payload.winner_role)
        || !Number.isInteger(payload.red_score) || payload.red_score < 0
        || !Number.isInteger(payload.blue_score) || payload.blue_score < 0
        || (payload.winner_role === 'red' && payload.red_score <= payload.blue_score)
        || (payload.winner_role === 'blue' && payload.blue_score <= payload.red_score)) {
      fail('REVIEW_EVENT_INVALID')
    }
  } else {
    const code = payload.resolution_code || payload.reason_code
    if (!validCode(code)) fail('REVIEW_EVENT_INVALID')
  }
}


function reduceReviewEvents(events) {
  if (!Array.isArray(events) || events.length === 0) fail('REVIEW_EVENT_INVALID')
  const matchId = events[0] && events[0].match_id
  const eventIds = new Set()
  let previousTime = -Infinity
  const state = {
    match_id: matchId,
    status: 'clear',
    settlement_allowed: true,
    replay_required: false,
    reason_codes: [],
    resolution: null,
    appeals: [],
    audit_count: 0,
  }

  for (let index = 0; index < events.length; index += 1) {
    const item = events[index]
    previousTime = validateEvent(item, index, matchId, previousTime, eventIds)
    eventIds.add(item.event_id)
    validatePayload(item)
    if (index === 0 && item.event_type !== 'review_opened') fail('REVIEW_EVENT_INVALID')
    if (index > 0 && item.event_type === 'review_opened') fail('REVIEW_EVENT_CONFLICT')
    if (state.resolution && item.event_type !== 'appeal_submitted') fail('REVIEW_EVENT_CONFLICT')

    if (item.event_type === 'review_opened') {
      if (item.actor_id !== 'system') fail('REVIEW_EVENT_CONFLICT')
      state.status = 'review_hold'
      state.settlement_allowed = false
      state.reason_codes = [...new Set(item.payload.reason_codes)].sort()
    } else if (item.event_type === 'appeal_submitted') {
      if (item.actor_id === 'system' || item.actor_id.startsWith('reviewer-')) {
        fail('REVIEW_EVENT_CONFLICT')
      }
      state.appeals.push({
        actor_id: item.actor_id,
        reason_code: item.payload.reason_code,
        submitted_at: item.occurred_at,
      })
    } else {
      if (!item.actor_id.startsWith('reviewer-')) fail('REVIEW_EVENT_CONFLICT')
      const statusByType = {
        review_released: 'released',
        review_not_ranked: 'not_ranked',
        result_corrected: 'corrected',
        result_revoked: 'revoked',
      }
      state.status = statusByType[item.event_type]
      state.settlement_allowed = item.event_type === 'review_released'
      state.replay_required = ['result_corrected', 'result_revoked'].includes(item.event_type)
      state.resolution = {
        event_type: item.event_type,
        ...structuredClone(item.payload),
        reviewer_id: item.actor_id,
        resolved_at: item.occurred_at,
      }
    }
    state.audit_count += 1
  }
  return state
}


module.exports = { ReviewEventError, reduceReviewEvents }
