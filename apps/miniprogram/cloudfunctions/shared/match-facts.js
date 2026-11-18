class MatchFactsError extends Error {
  constructor(code) {
    super(code)
    this.name = 'MatchFactsError'
    this.code = code
  }
}


function invalid() {
  throw new MatchFactsError('MATCH_FACTS_INVALID')
}


function conflict() {
  throw new MatchFactsError('MATCH_FACTS_CONFLICT')
}


const COMMON_EVENT_KEYS = [
  'actor_user_id',
  'event_id',
  'event_type',
  'match_id',
  'occurred_at',
  'payload',
  'schema_version',
  'sequence',
]

const PAYLOAD_KEYS = {
  match_created: ['blue_user_id', 'ranking_scope_id', 'red_user_id', 'session_id', 'table_id', 'venue_id'],
  settings_proposed: ['first_breaker', 'format', 'mode', 'race_target', 'ranking_intent', 'settings_revision'],
  settings_confirmed: ['role', 'settings_revision'],
  match_started: ['settings_revision'],
  result_proposed: ['blue_score', 'finish_reason', 'red_score', 'result_revision', 'winner_role'],
  result_confirmed: ['result_revision', 'role'],
  result_disputed: ['reason', 'result_revision', 'role'],
  result_confirmation_expired: ['result_revision'],
  match_interrupted: ['reason'],
}

const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u


function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}


function hasExactKeys(value, expectedKeys) {
  if (!isRecord(value)) return false
  const actualKeys = Object.keys(value).sort()
  return actualKeys.length === expectedKeys.length
    && actualKeys.every((key, index) => key === expectedKeys[index])
}


function isValidId(value) {
  return typeof value === 'string' && ID_PATTERN.test(value)
}


function timestamp(value) {
  if (typeof value !== 'string') invalid()
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-]\d{2}:\d{2})$/u.exec(value)
  if (!match) invalid()
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, zone] = match
  const year = Number(yearText)
  const month = Number(monthText)
  const day = Number(dayText)
  const hour = Number(hourText)
  const minute = Number(minuteText)
  const second = Number(secondText)
  const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate()
  if (month < 1 || month > 12 || day < 1 || day > daysInMonth
      || hour > 23 || minute > 59 || second > 59) invalid()
  if (zone !== 'Z') {
    const [offsetHour, offsetMinute] = zone.slice(1).split(':').map(Number)
    if (offsetHour > 23 || offsetMinute > 59) invalid()
  }
  const milliseconds = Date.parse(value)
  if (!Number.isFinite(milliseconds)) invalid()
  return milliseconds
}


function validateCreated(payload) {
  for (const key of PAYLOAD_KEYS.match_created) {
    if (!isValidId(payload[key])) invalid()
  }
  if (payload.red_user_id === payload.blue_user_id) invalid()
}


function validateEventEnvelope(currentEvent, index, stream) {
  if (!hasExactKeys(currentEvent, COMMON_EVENT_KEYS)) invalid()
  if (currentEvent.schema_version !== '1.0') invalid()
  if (!isValidId(currentEvent.event_id) || !isValidId(currentEvent.match_id)) invalid()
  if (!isValidId(currentEvent.actor_user_id)) invalid()
  if (!Object.hasOwn(PAYLOAD_KEYS, currentEvent.event_type)) invalid()
  if (!hasExactKeys(currentEvent.payload, PAYLOAD_KEYS[currentEvent.event_type])) invalid()
  if (currentEvent.sequence !== index + 1) invalid()
  if (index === 0 && currentEvent.event_type !== 'match_created') invalid()
  if (index > 0 && currentEvent.event_type === 'match_created') invalid()
  if (index > 0 && currentEvent.match_id !== stream[0].match_id) invalid()
  return timestamp(currentEvent.occurred_at)
}


function validateSettings(payload) {
  if (!Number.isInteger(payload.settings_revision) || payload.settings_revision < 1) invalid()
  if (!['observation', 'assistance'].includes(payload.mode)) invalid()
  if (!['single', 'race'].includes(payload.format)) invalid()
  if (!['red', 'blue'].includes(payload.first_breaker)) invalid()
  if (typeof payload.ranking_intent !== 'boolean') invalid()
  if (payload.mode === 'assistance' && payload.ranking_intent) invalid()
  if (payload.format === 'single' && payload.race_target !== null) invalid()
  if (payload.format === 'race'
      && (!Number.isInteger(payload.race_target) || payload.race_target < 2 || payload.race_target > 99)) {
    invalid()
  }
}


function validateResult(payload, settings) {
  if (payload.result_revision !== 1) invalid()
  if (!['red', 'blue'].includes(payload.winner_role)) invalid()
  if (!Number.isInteger(payload.red_score) || payload.red_score < 0) invalid()
  if (!Number.isInteger(payload.blue_score) || payload.blue_score < 0) invalid()
  if (!['normal', 'concession'].includes(payload.finish_reason)) invalid()

  const winnerScore = payload.winner_role === 'red' ? payload.red_score : payload.blue_score
  const loserScore = payload.winner_role === 'red' ? payload.blue_score : payload.red_score

  if (payload.finish_reason === 'concession') {
    const finishTarget = settings.format === 'single' ? 1 : settings.race_target
    if (winnerScore < loserScore || winnerScore >= finishTarget || loserScore >= finishTarget) invalid()
    return
  }

  if (settings.format === 'single') {
    const isRedWin = payload.winner_role === 'red' && payload.red_score === 1 && payload.blue_score === 0
    const isBlueWin = payload.winner_role === 'blue' && payload.red_score === 0 && payload.blue_score === 1
    if (!isRedWin && !isBlueWin) invalid()
    return
  }

  if (winnerScore !== settings.race_target || loserScore >= settings.race_target) invalid()
}


function reduceMatchEvents(events) {
  if (!Array.isArray(events) || events.length === 0) invalid()

  const eventIds = new Set()
  let previousTime = -Infinity
  for (let index = 0; index < events.length; index += 1) {
    const currentTime = validateEventEnvelope(events[index], index, events)
    if (eventIds.has(events[index].event_id) || currentTime < previousTime) invalid()
    eventIds.add(events[index].event_id)
    previousTime = currentTime
  }

  let state = null

  for (const currentEvent of events) {
    const { event_type: eventType, payload } = currentEvent

    if (eventType === 'match_created') {
      if (currentEvent.actor_user_id !== 'system') conflict()
      validateCreated(payload)
      state = {
        match_id: currentEvent.match_id,
        session_id: payload.session_id,
        ranking_scope_id: payload.ranking_scope_id,
        venue_id: payload.venue_id,
        table_id: payload.table_id,
        status: 'awaiting_settings',
        event_count: 1,
        latest_sequence: currentEvent.sequence,
        participants: { red: payload.red_user_id, blue: payload.blue_user_id },
        settings: null,
        settings_confirmations: [],
        started_at: null,
        result: null,
        result_confirmations: [],
        result_proposed_at: null,
        result_confirmation_deadline_at: null,
        dispute: null,
        confirmation_expired_at: null,
        interruption: null,
      }
      continue
    }

    if (['fact_package_ready', 'interrupted', 'disputed', 'confirmation_expired'].includes(state.status)) conflict()
    const participantIds = Object.values(state.participants)
    if (currentEvent.actor_user_id !== 'system' && !participantIds.includes(currentEvent.actor_user_id)) conflict()

    if (eventType === 'settings_proposed') {
      if (currentEvent.actor_user_id === 'system') conflict()
      if (!['awaiting_settings', 'awaiting_settings_confirmations', 'ready'].includes(state.status)) conflict()
      validateSettings(payload)
      const expectedRevision = state.settings === null ? 1 : state.settings.settings_revision + 1
      if (payload.settings_revision !== expectedRevision) conflict()
      state.settings = { ...payload }
      state.settings_confirmations = []
      state.status = 'awaiting_settings_confirmations'
    } else if (eventType === 'settings_confirmed') {
      if (state.status !== 'awaiting_settings_confirmations') conflict()
      if (payload.settings_revision !== state.settings.settings_revision) conflict()
      if (!['red', 'blue'].includes(payload.role)) invalid()
      if (state.settings_confirmations.includes(payload.role)) conflict()
      if (currentEvent.actor_user_id !== state.participants[payload.role]) conflict()
      state.settings_confirmations = [...state.settings_confirmations, payload.role].sort()
      if (state.settings_confirmations.length === 2) state.status = 'ready'
    } else if (eventType === 'match_started') {
      if (currentEvent.actor_user_id !== 'system') conflict()
      if (state.status !== 'ready') conflict()
      if (payload.settings_revision !== state.settings.settings_revision) conflict()
      state.started_at = currentEvent.occurred_at
      state.status = 'in_progress'
    } else if (eventType === 'result_proposed') {
      if (state.status !== 'in_progress') conflict()
      validateResult(payload, state.settings)
      state.result = { ...payload }
      state.result_confirmations = []
      state.result_proposed_at = currentEvent.occurred_at
      state.result_confirmation_deadline_at = new Date(
        timestamp(currentEvent.occurred_at) + 24 * 60 * 60 * 1000,
      ).toISOString()
      state.dispute = null
      state.confirmation_expired_at = null
      state.status = 'awaiting_result_confirmations'
    } else if (eventType === 'result_confirmed') {
      if (state.status !== 'awaiting_result_confirmations') conflict()
      if (payload.result_revision !== state.result.result_revision) conflict()
      if (!['red', 'blue'].includes(payload.role)) invalid()
      if (state.result_confirmations.includes(payload.role)) conflict()
      if (currentEvent.actor_user_id !== state.participants[payload.role]) conflict()
      if (timestamp(currentEvent.occurred_at) >= timestamp(state.result_confirmation_deadline_at)) conflict()
      state.result_confirmations = [...state.result_confirmations, payload.role].sort()
      if (state.result_confirmations.length === 2) state.status = 'fact_package_ready'
    } else if (eventType === 'result_disputed') {
      if (state.status !== 'awaiting_result_confirmations') conflict()
      if (payload.result_revision !== state.result.result_revision) conflict()
      if (!['red', 'blue'].includes(payload.role)) invalid()
      if (!['winner_mismatch', 'score_mismatch', 'not_completed', 'identity_issue', 'other']
        .includes(payload.reason)) invalid()
      if (currentEvent.actor_user_id !== state.participants[payload.role]) conflict()
      if (state.result_confirmations.includes(payload.role)) conflict()
      if (timestamp(currentEvent.occurred_at) >= timestamp(state.result_confirmation_deadline_at)) conflict()
      state.dispute = { ...payload, disputed_at: currentEvent.occurred_at }
      state.status = 'disputed'
    } else if (eventType === 'result_confirmation_expired') {
      if (state.status !== 'awaiting_result_confirmations') conflict()
      if (currentEvent.actor_user_id !== 'system') conflict()
      if (payload.result_revision !== state.result.result_revision) conflict()
      if (timestamp(currentEvent.occurred_at) < timestamp(state.result_confirmation_deadline_at)) conflict()
      state.confirmation_expired_at = currentEvent.occurred_at
      state.status = 'confirmation_expired'
    } else if (eventType === 'match_interrupted') {
      if (!['ready', 'in_progress'].includes(state.status)) conflict()
      if (!['technical', 'dispute', 'no_start_evidence', 'participant_changed'].includes(payload.reason)) invalid()
      state.interruption = { ...payload }
      state.status = 'interrupted'
    }

    state.event_count += 1
    state.latest_sequence = currentEvent.sequence
  }

  return state
}


module.exports = {
  MatchFactsError,
  reduceMatchEvents,
}
