const crypto = require('crypto')


const DISPUTE_REASONS = [
  'winner_mismatch',
  'score_mismatch',
  'not_completed',
  'identity_issue',
  'other',
]


class ResultConfirmationError extends Error {
  constructor(code) {
    super(code)
    this.name = 'ResultConfirmationError'
    this.code = code
  }
}


function fail(code) {
  throw new ResultConfirmationError(code)
}


function eventId(action, matchId, resultRevision, role = 'system') {
  const digest = crypto.createHash('sha256')
    .update(`${action}:${matchId}:${resultRevision}:${role}`, 'utf8')
    .digest('hex')
  return `result-${action}-${digest.slice(0, 48)}`
}


function participantRole(projection, actorUserId) {
  if (projection.participants.red === actorUserId) return 'red'
  if (projection.participants.blue === actorUserId) return 'blue'
  fail('RESULT_CONFIRMATION_UNAUTHORIZED')
}


function pendingResult(projection) {
  if (projection.status !== 'awaiting_result_confirmations'
      || !projection.result || !projection.result_confirmation_deadline_at) {
    fail('RESULT_CONFIRMATION_NOT_PENDING')
  }
  return projection.result
}


function buildEvent({ action, match, actorUserId, eventType, payload, occurredAt, role }) {
  return {
    schema_version: '1.0',
    event_id: eventId(action, match.projection.match_id, payload.result_revision, role),
    match_id: match.projection.match_id,
    sequence: match.projection.latest_sequence + 1,
    event_type: eventType,
    actor_user_id: actorUserId,
    occurred_at: occurredAt,
    payload,
  }
}


function createResultConfirmationService({ repository, now }) {
  async function requireMatch(matchId) {
    const match = await repository.getMatch(matchId)
    if (!match) fail('RESULT_CONFIRMATION_NOT_FOUND')
    return match
  }

  return {
    async confirm({ matchId, actorUserId }) {
      const match = await requireMatch(matchId)
      const role = participantRole(match.projection, actorUserId)
      const existing = match.events.find(item => (
        item.event_type === 'result_confirmed'
        && item.actor_user_id === actorUserId
        && item.payload.role === role
      ))
      if (existing) return structuredClone(match.projection)

      const result = pendingResult(match.projection)
      const occurredAt = now()
      if (Date.parse(occurredAt) >= Date.parse(match.projection.result_confirmation_deadline_at)) {
        fail('RESULT_CONFIRMATION_EXPIRED')
      }
      return repository.appendEvent(buildEvent({
        action: 'confirmed',
        match,
        actorUserId,
        eventType: 'result_confirmed',
        payload: { result_revision: result.result_revision, role },
        occurredAt,
        role,
      }))
    },

    async dispute({ matchId, actorUserId, reason }) {
      if (!DISPUTE_REASONS.includes(reason)) fail('RESULT_CONFIRMATION_INVALID_REASON')
      const match = await requireMatch(matchId)
      const role = participantRole(match.projection, actorUserId)
      const existing = match.events.find(item => (
        item.event_type === 'result_disputed'
        && item.actor_user_id === actorUserId
        && item.payload.role === role
        && item.payload.reason === reason
      ))
      if (existing) return structuredClone(match.projection)

      const result = pendingResult(match.projection)
      if (match.projection.result_confirmations.includes(role)) {
        fail('RESULT_CONFIRMATION_NOT_PENDING')
      }
      const occurredAt = now()
      if (Date.parse(occurredAt) >= Date.parse(match.projection.result_confirmation_deadline_at)) {
        fail('RESULT_CONFIRMATION_EXPIRED')
      }
      return repository.appendEvent(buildEvent({
        action: `disputed-${reason}`,
        match,
        actorUserId,
        eventType: 'result_disputed',
        payload: { result_revision: result.result_revision, role, reason },
        occurredAt,
        role,
      }))
    },

    async expire({ matchId }) {
      const match = await requireMatch(matchId)
      const existing = match.events.find(item => item.event_type === 'result_confirmation_expired')
      if (existing) return structuredClone(match.projection)

      const result = pendingResult(match.projection)
      const occurredAt = now()
      if (Date.parse(occurredAt) < Date.parse(match.projection.result_confirmation_deadline_at)) {
        fail('RESULT_CONFIRMATION_TOO_EARLY')
      }
      return repository.appendEvent(buildEvent({
        action: 'expired',
        match,
        actorUserId: 'system',
        eventType: 'result_confirmation_expired',
        payload: { result_revision: result.result_revision },
        occurredAt,
        role: 'system',
      }))
    },
  }
}


module.exports = {
  DISPUTE_REASONS,
  ResultConfirmationError,
  createResultConfirmationService,
}
