const crypto = require('crypto')


class MatchSetupError extends Error {
  constructor(code) {
    super(code)
    this.name = 'MatchSetupError'
    this.code = code
  }
}


const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u


function fail(code) {
  throw new MatchSetupError(code)
}


function eventId(action, sessionId, revision, actor = 'system') {
  const value = crypto.createHash('sha256')
    .update(`${action}:${sessionId}:${revision}:${actor}`, 'utf8').digest('hex').slice(0, 48)
  return `match-${action}-${value}`
}


function matchId(sessionId) {
  return `match-${crypto.createHash('sha256').update(sessionId, 'utf8').digest('hex').slice(0, 48)}`
}


function participantRole(projection, actorUserId) {
  if (projection.participants.red === actorUserId) return 'red'
  if (projection.participants.blue === actorUserId) return 'blue'
  fail('MATCH_SETUP_UNAUTHORIZED')
}


function validateContext(context, actorUserId) {
  const values = context && [
    context.session_id, context.ranking_scope_id, context.venue_id, context.table_id,
    context.red_user_id, context.blue_user_id, actorUserId,
  ]
  if (!values || values.some(value => !ID_PATTERN.test(value || ''))
      || context.red_user_id === context.blue_user_id
      || ![context.red_user_id, context.blue_user_id].includes(actorUserId)) {
    fail('MATCH_SETUP_INVALID')
  }
}


function validateSettings({ mode, format, raceTarget, firstBreaker, rankingIntent }) {
  if (!['observation', 'assistance'].includes(mode)
      || !['single', 'race'].includes(format)
      || !['red', 'blue'].includes(firstBreaker)
      || typeof rankingIntent !== 'boolean'
      || (mode === 'assistance' && rankingIntent)
      || (format === 'single' && raceTarget !== null)
      || (format === 'race' && (!Number.isInteger(raceTarget) || raceTarget < 2 || raceTarget > 99))) {
    fail('MATCH_SETUP_INVALID')
  }
}


function createMatchSetupService({ repository, now }) {
  if (!repository || typeof repository.getBySession !== 'function'
      || typeof repository.appendEvent !== 'function' || typeof now !== 'function') {
    fail('MATCH_SETUP_CONFIGURATION_INVALID')
  }

  async function requireMatch(sessionId) {
    const match = await repository.getBySession(sessionId)
    if (!match) fail('MATCH_SETUP_NOT_FOUND')
    return match
  }

  return {
    async proposeSettings(input) {
      validateContext(input.context, input.actorUserId)
      validateSettings(input)
      let match = await repository.getBySession(input.context.session_id)
      if (!match) {
        const id = matchId(input.context.session_id)
        await repository.appendEvent({
          schema_version: '1.0', event_id: eventId('created', input.context.session_id, 1),
          match_id: id, sequence: 1, event_type: 'match_created', actor_user_id: 'system',
          occurred_at: now(), payload: {
            session_id: input.context.session_id,
            ranking_scope_id: input.context.ranking_scope_id,
            venue_id: input.context.venue_id,
            table_id: input.context.table_id,
            red_user_id: input.context.red_user_id,
            blue_user_id: input.context.blue_user_id,
          },
        })
        match = await requireMatch(input.context.session_id)
      }
      participantRole(match.projection, input.actorUserId)
      const revision = match.projection.settings ? match.projection.settings.settings_revision + 1 : 1
      return repository.appendEvent({
        schema_version: '1.0',
        event_id: eventId('settings', input.context.session_id, revision, input.actorUserId),
        match_id: match.projection.match_id, sequence: match.projection.latest_sequence + 1,
        event_type: 'settings_proposed', actor_user_id: input.actorUserId, occurred_at: now(),
        payload: {
          settings_revision: revision, mode: input.mode, format: input.format,
          race_target: input.raceTarget, first_breaker: input.firstBreaker,
          ranking_intent: input.rankingIntent,
        },
      })
    },

    async confirmSettings({ sessionId, actorUserId }) {
      const match = await requireMatch(sessionId)
      const role = participantRole(match.projection, actorUserId)
      if (match.projection.settings_confirmations.includes(role)) return structuredClone(match.projection)
      if (match.projection.status !== 'awaiting_settings_confirmations') fail('MATCH_SETUP_NOT_CONFIRMABLE')
      const revision = match.projection.settings.settings_revision
      return repository.appendEvent({
        schema_version: '1.0', event_id: eventId('settings-confirmed', sessionId, revision, role),
        match_id: match.projection.match_id, sequence: match.projection.latest_sequence + 1,
        event_type: 'settings_confirmed', actor_user_id: actorUserId, occurred_at: now(),
        payload: { settings_revision: revision, role },
      })
    },

    async startFromDevice({ sessionId }) {
      const match = await requireMatch(sessionId)
      if (match.projection.status === 'in_progress') return structuredClone(match.projection)
      if (match.projection.status !== 'ready') fail('MATCH_SETUP_NOT_READY')
      const revision = match.projection.settings.settings_revision
      return repository.appendEvent({
        schema_version: '1.0', event_id: eventId('started', sessionId, revision),
        match_id: match.projection.match_id, sequence: match.projection.latest_sequence + 1,
        event_type: 'match_started', actor_user_id: 'system', occurred_at: now(),
        payload: { settings_revision: revision },
      })
    },

    async proposeResultFromDevice({ sessionId, winnerRole, redScore, blueScore, finishReason }) {
      const match = await requireMatch(sessionId)
      if (match.projection.status !== 'in_progress') fail('MATCH_RESULT_NOT_READY')
      return repository.appendEvent({
        schema_version: '1.0', event_id: eventId('result', sessionId, 1),
        match_id: match.projection.match_id, sequence: match.projection.latest_sequence + 1,
        event_type: 'result_proposed', actor_user_id: 'system', occurred_at: now(),
        payload: {
          result_revision: 1, winner_role: winnerRole,
          red_score: redScore, blue_score: blueScore, finish_reason: finishReason,
        },
      })
    },
  }
}


module.exports = { MatchSetupError, createMatchSetupService, matchId }
