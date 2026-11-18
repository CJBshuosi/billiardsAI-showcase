const { MatchFactsError, reduceMatchEvents } = require('../../cloudfunctions/shared/match-facts')


const BASE_TIME_MS = Date.parse('2026-09-01T00:00:00.000Z')


function event(sequence, eventType, actorUserId, payload) {
  return {
    schema_version: '1.0',
    event_id: `event-${sequence}`,
    match_id: 'match-001',
    sequence,
    event_type: eventType,
    actor_user_id: actorUserId,
    occurred_at: new Date(BASE_TIME_MS + sequence * 1000).toISOString(),
    payload,
  }
}


function completeSingleEvents() {
  return [
    event(1, 'match_created', 'system', {
      session_id: 'session-001',
      ranking_scope_id: 'zhoukou-urban',
      venue_id: 'venue-001',
      table_id: 'table-001',
      red_user_id: 'user-red',
      blue_user_id: 'user-blue',
    }),
    event(2, 'settings_proposed', 'user-red', {
      settings_revision: 1,
      mode: 'observation',
      format: 'single',
      race_target: null,
      first_breaker: 'red',
      ranking_intent: true,
    }),
    event(3, 'settings_confirmed', 'user-red', { settings_revision: 1, role: 'red' }),
    event(4, 'settings_confirmed', 'user-blue', { settings_revision: 1, role: 'blue' }),
    event(5, 'match_started', 'system', { settings_revision: 1 }),
    event(6, 'result_proposed', 'system', {
      result_revision: 1,
      winner_role: 'red',
      red_score: 1,
      blue_score: 0,
      finish_reason: 'normal',
    }),
    event(7, 'result_confirmed', 'user-red', { result_revision: 1, role: 'red' }),
    event(8, 'result_confirmed', 'user-blue', { result_revision: 1, role: 'blue' }),
  ]
}


test('a mutually confirmed single match produces a fact package', () => {
  const state = reduceMatchEvents(completeSingleEvents())

  expect(state).toMatchObject({
    match_id: 'match-001',
    session_id: 'session-001',
    ranking_scope_id: 'zhoukou-urban',
    venue_id: 'venue-001',
    table_id: 'table-001',
    status: 'fact_package_ready',
    event_count: 8,
    latest_sequence: 8,
    participants: { red: 'user-red', blue: 'user-blue' },
    settings: { settings_revision: 1, format: 'single', ranking_intent: true },
    result: { result_revision: 1, winner_role: 'red', red_score: 1, blue_score: 0 },
    result_confirmations: ['blue', 'red'],
  })
})


function expectContractError(events, code) {
  expect(() => reduceMatchEvents(events)).toThrow(expect.objectContaining({
    name: 'MatchFactsError',
    code,
  }))
}


test('assistance mode cannot declare ranking intent', () => {
  const events = completeSingleEvents().slice(0, 2)
  events[1].payload.mode = 'assistance'

  expectContractError(events, 'MATCH_FACTS_INVALID')
})


test.each([
  ['single', 2],
  ['race', 1],
  ['race', 100],
])('%s settings reject race target %p', (format, raceTarget) => {
  const events = completeSingleEvents().slice(0, 2)
  events[1].payload.format = format
  events[1].payload.race_target = raceTarget

  expectContractError(events, 'MATCH_FACTS_INVALID')
})


test('a newer settings revision clears prior confirmations', () => {
  const events = completeSingleEvents().slice(0, 4)
  events.push(event(5, 'settings_proposed', 'user-blue', {
    settings_revision: 2,
    mode: 'observation',
    format: 'race',
    race_target: 5,
    first_breaker: 'blue',
    ranking_intent: true,
  }))

  expect(reduceMatchEvents(events)).toMatchObject({
    status: 'awaiting_settings_confirmations',
    settings: { settings_revision: 2, format: 'race', race_target: 5 },
    settings_confirmations: [],
  })
})


test('confirming a stale settings revision conflicts with current settings', () => {
  const events = completeSingleEvents().slice(0, 4)
  events.push(event(5, 'settings_proposed', 'user-blue', {
    settings_revision: 2,
    mode: 'observation',
    format: 'race',
    race_target: 5,
    first_breaker: 'blue',
    ranking_intent: true,
  }))
  events.push(event(6, 'settings_confirmed', 'user-red', { settings_revision: 1, role: 'red' }))

  expectContractError(events, 'MATCH_FACTS_CONFLICT')
})


test('a match cannot start before both players confirm the latest settings', () => {
  const events = completeSingleEvents().slice(0, 3)
  events.push(event(4, 'match_started', 'system', { settings_revision: 1 }))

  expectContractError(events, 'MATCH_FACTS_CONFLICT')
})


test('settings are frozen after the match starts', () => {
  const events = completeSingleEvents().slice(0, 5)
  events.push(event(6, 'settings_proposed', 'user-blue', {
    settings_revision: 2,
    mode: 'observation',
    format: 'race',
    race_target: 5,
    first_breaker: 'blue',
    ranking_intent: true,
  }))

  expectContractError(events, 'MATCH_FACTS_CONFLICT')
})


test('MatchFactsError exposes only the stable contract code', () => {
  const error = new MatchFactsError('MATCH_FACTS_INVALID')

  expect(error).toMatchObject({ name: 'MatchFactsError', message: 'MATCH_FACTS_INVALID', code: 'MATCH_FACTS_INVALID' })
})


test.each([
  [2, 0, 'red'],
  [0, 2, 'blue'],
  [1, 0, 'blue'],
])('normal single rejects score %i:%i with winner %s', (redScore, blueScore, winnerRole) => {
  const events = completeSingleEvents().slice(0, 6)
  events[5].payload.red_score = redScore
  events[5].payload.blue_score = blueScore
  events[5].payload.winner_role = winnerRole

  expectContractError(events, 'MATCH_FACTS_INVALID')
})


function startedRaceEvents() {
  const events = completeSingleEvents().slice(0, 5)
  events[1].payload.format = 'race'
  events[1].payload.race_target = 5
  return events
}


test('normal race requires the declared winner to reach the target first', () => {
  const valid = startedRaceEvents()
  valid.push(event(6, 'result_proposed', 'system', {
    result_revision: 1,
    winner_role: 'blue',
    red_score: 3,
    blue_score: 5,
    finish_reason: 'normal',
  }))
  valid.push(event(7, 'result_confirmed', 'user-blue', { result_revision: 1, role: 'blue' }))
  valid.push(event(8, 'result_confirmed', 'user-red', { result_revision: 1, role: 'red' }))

  expect(reduceMatchEvents(valid)).toMatchObject({
    status: 'fact_package_ready',
    result: { winner_role: 'blue', red_score: 3, blue_score: 5 },
  })

  const winnerBelowTarget = startedRaceEvents()
  winnerBelowTarget.push(event(6, 'result_proposed', 'system', {
    result_revision: 1,
    winner_role: 'red',
    red_score: 4,
    blue_score: 3,
    finish_reason: 'normal',
  }))
  expectContractError(winnerBelowTarget, 'MATCH_FACTS_INVALID')

  const loserAtTarget = startedRaceEvents()
  loserAtTarget.push(event(6, 'result_proposed', 'system', {
    result_revision: 1,
    winner_role: 'red',
    red_score: 5,
    blue_score: 5,
    finish_reason: 'normal',
  }))
  expectContractError(loserAtTarget, 'MATCH_FACTS_INVALID')
})


test('concession permits a consistent pre-target race score', () => {
  const events = startedRaceEvents()
  events.push(event(6, 'result_proposed', 'user-blue', {
    result_revision: 1,
    winner_role: 'red',
    red_score: 3,
    blue_score: 2,
    finish_reason: 'concession',
  }))

  expect(reduceMatchEvents(events)).toMatchObject({
    status: 'awaiting_result_confirmations',
    result: { winner_role: 'red', red_score: 3, blue_score: 2, finish_reason: 'concession' },
  })
})


test('single concession can end a started rack before either player scores', () => {
  const events = completeSingleEvents().slice(0, 5)
  events.push(event(6, 'result_proposed', 'user-blue', {
    result_revision: 1,
    winner_role: 'red',
    red_score: 0,
    blue_score: 0,
    finish_reason: 'concession',
  }))

  expect(reduceMatchEvents(events)).toMatchObject({
    status: 'awaiting_result_confirmations',
    result: { winner_role: 'red', red_score: 0, blue_score: 0, finish_reason: 'concession' },
  })
})


test('concession scores must remain below the configured finish target', () => {
  const events = startedRaceEvents()
  events.push(event(6, 'result_proposed', 'user-blue', {
    result_revision: 1,
    winner_role: 'red',
    red_score: 5,
    blue_score: 4,
    finish_reason: 'concession',
  }))

  expectContractError(events, 'MATCH_FACTS_INVALID')
})


test('a result cannot be proposed before the match starts', () => {
  const events = completeSingleEvents().slice(0, 4)
  events.push({ ...completeSingleEvents()[5], sequence: 5, event_id: 'event-5' })

  expectContractError(events, 'MATCH_FACTS_CONFLICT')
})


test('result confirmations must target the current result revision and be unique per role', () => {
  const stale = completeSingleEvents().slice(0, 7)
  stale[6].payload.result_revision = 2
  expectContractError(stale, 'MATCH_FACTS_CONFLICT')

  const duplicate = completeSingleEvents().slice(0, 7)
  duplicate.push(event(8, 'result_confirmed', 'user-red', { result_revision: 1, role: 'red' }))
  expectContractError(duplicate, 'MATCH_FACTS_CONFLICT')
})


test.each([
  ['ready', 4],
  ['in_progress', 5],
])('a match can be interrupted from %s without producing a winner', (_status, eventCount) => {
  const events = completeSingleEvents().slice(0, eventCount)
  events.push(event(eventCount + 1, 'match_interrupted', 'system', { reason: 'technical' }))

  expect(reduceMatchEvents(events)).toMatchObject({
    status: 'interrupted',
    result: null,
    result_confirmations: [],
    interruption: { reason: 'technical' },
  })
})


test('terminal matches reject later events', () => {
  const completed = completeSingleEvents()
  completed.push(event(9, 'match_interrupted', 'system', { reason: 'dispute' }))
  expectContractError(completed, 'MATCH_FACTS_CONFLICT')

  const interrupted = completeSingleEvents().slice(0, 5)
  interrupted.push(event(6, 'match_interrupted', 'system', { reason: 'technical' }))
  interrupted.push(event(7, 'result_proposed', 'system', completeSingleEvents()[5].payload))
  expectContractError(interrupted, 'MATCH_FACTS_CONFLICT')
})


test('event IDs must be unique and sequences must start at one without gaps', () => {
  const duplicateId = completeSingleEvents().slice(0, 3)
  duplicateId[2].event_id = duplicateId[1].event_id
  expectContractError(duplicateId, 'MATCH_FACTS_INVALID')

  const startsAtTwo = completeSingleEvents().slice(0, 1)
  startsAtTwo[0].sequence = 2
  expectContractError(startsAtTwo, 'MATCH_FACTS_INVALID')

  const missingSequence = completeSingleEvents().slice(0, 3)
  missingSequence[2].sequence = 4
  expectContractError(missingSequence, 'MATCH_FACTS_INVALID')
})


test('one event stream cannot cross matches or move backward in time', () => {
  const crossMatch = completeSingleEvents().slice(0, 3)
  crossMatch[2].match_id = 'match-002'
  expectContractError(crossMatch, 'MATCH_FACTS_INVALID')

  const reversedTime = completeSingleEvents().slice(0, 3)
  reversedTime[2].occurred_at = '2026-08-31T23:59:59.000Z'
  expectContractError(reversedTime, 'MATCH_FACTS_INVALID')
})


test('events and their payloads use closed schemas', () => {
  const extraEventField = completeSingleEvents().slice(0, 1)
  extraEventField[0].trusted = true
  expectContractError(extraEventField, 'MATCH_FACTS_INVALID')

  const extraPayloadField = completeSingleEvents().slice(0, 2)
  extraPayloadField[1].payload.score_hint = 7
  expectContractError(extraPayloadField, 'MATCH_FACTS_INVALID')
})


test.each([
  '',
  'bad actor',
  'x'.repeat(129),
])('invalid actor ID %p is rejected', (actorUserId) => {
  const events = completeSingleEvents().slice(0, 2)
  events[1].actor_user_id = actorUserId
  expectContractError(events, 'MATCH_FACTS_INVALID')
})


test('participant identities and confirmation roles cannot be mismatched', () => {
  const sameUser = completeSingleEvents().slice(0, 1)
  sameUser[0].payload.blue_user_id = sameUser[0].payload.red_user_id
  expectContractError(sameUser, 'MATCH_FACTS_INVALID')

  const wrongConfirmer = completeSingleEvents().slice(0, 3)
  wrongConfirmer[2].actor_user_id = 'user-blue'
  expectContractError(wrongConfirmer, 'MATCH_FACTS_CONFLICT')
})


test('timestamps must be valid ISO instants with an explicit timezone', () => {
  for (const occurredAt of ['not-a-date', '2026-09-01T00:00:00', '2026-02-30T00:00:00.000Z']) {
    const events = completeSingleEvents().slice(0, 1)
    events[0].occurred_at = occurredAt
    expectContractError(events, 'MATCH_FACTS_INVALID')
  }
})


test('reducing an event stream does not mutate the input or retain payload references', () => {
  const events = completeSingleEvents()
  const before = JSON.parse(JSON.stringify(events))

  const state = reduceMatchEvents(events)
  state.settings.format = 'race'
  state.result.red_score = 99

  expect(events).toEqual(before)
})


test('a proposed result opens an exact 24-hour confirmation window', () => {
  const state = reduceMatchEvents(completeSingleEvents().slice(0, 6))

  expect(state).toMatchObject({
    status: 'awaiting_result_confirmations',
    result_proposed_at: '2026-09-01T00:00:06.000Z',
    result_confirmation_deadline_at: '2026-09-02T00:00:06.000Z',
    dispute: null,
    confirmation_expired_at: null,
  })
})


test('player confirmation is allowed before but not at the deadline', () => {
  const beforeDeadline = completeSingleEvents().slice(0, 7)
  beforeDeadline[6].occurred_at = '2026-09-02T00:00:05.999Z'
  expect(reduceMatchEvents(beforeDeadline)).toMatchObject({
    status: 'awaiting_result_confirmations',
    result_confirmations: ['red'],
  })

  const atDeadline = completeSingleEvents().slice(0, 7)
  atDeadline[6].occurred_at = '2026-09-02T00:00:06.000Z'
  expectContractError(atDeadline, 'MATCH_FACTS_CONFLICT')
})


test('a participant can dispute the current result before the deadline', () => {
  const events = completeSingleEvents().slice(0, 6)
  events.push(event(7, 'result_disputed', 'user-blue', {
    result_revision: 1,
    role: 'blue',
    reason: 'score_mismatch',
  }))

  expect(reduceMatchEvents(events)).toMatchObject({
    status: 'disputed',
    result_confirmations: [],
    dispute: {
      result_revision: 1,
      role: 'blue',
      reason: 'score_mismatch',
      disputed_at: '2026-09-01T00:00:07.000Z',
    },
  })
})


test('result dispute enforces actor role and the confirmation deadline', () => {
  const wrongActor = completeSingleEvents().slice(0, 6)
  wrongActor.push(event(7, 'result_disputed', 'user-red', {
    result_revision: 1,
    role: 'blue',
    reason: 'winner_mismatch',
  }))
  expectContractError(wrongActor, 'MATCH_FACTS_CONFLICT')

  const expired = completeSingleEvents().slice(0, 6)
  const dispute = event(7, 'result_disputed', 'user-blue', {
    result_revision: 1,
    role: 'blue',
    reason: 'other',
  })
  dispute.occurred_at = '2026-09-02T00:00:06.000Z'
  expired.push(dispute)
  expectContractError(expired, 'MATCH_FACTS_CONFLICT')
})


test('system expiry is rejected early and accepted at the deadline', () => {
  const early = completeSingleEvents().slice(0, 6)
  early.push(event(7, 'result_confirmation_expired', 'system', { result_revision: 1 }))
  expectContractError(early, 'MATCH_FACTS_CONFLICT')

  const due = completeSingleEvents().slice(0, 6)
  const expiry = event(7, 'result_confirmation_expired', 'system', { result_revision: 1 })
  expiry.occurred_at = '2026-09-02T00:00:06.000Z'
  due.push(expiry)
  expect(reduceMatchEvents(due)).toMatchObject({
    status: 'confirmation_expired',
    confirmation_expired_at: '2026-09-02T00:00:06.000Z',
  })
})


test.each([
  ['result_disputed', 'disputed'],
  ['result_confirmation_expired', 'confirmation_expired'],
])('%s produces a terminal %s state', (terminalEventType) => {
  const events = completeSingleEvents().slice(0, 6)
  const terminal = terminalEventType === 'result_disputed'
    ? event(7, terminalEventType, 'user-blue', {
      result_revision: 1, role: 'blue', reason: 'other',
    })
    : event(7, terminalEventType, 'system', { result_revision: 1 })
  if (terminalEventType === 'result_confirmation_expired') {
    terminal.occurred_at = '2026-09-02T00:00:06.000Z'
  }
  events.push(terminal)
  events.push({
    ...event(8, 'result_confirmed', 'user-red', { result_revision: 1, role: 'red' }),
    occurred_at: '2026-09-02T00:00:07.000Z',
  })

  expectContractError(events, 'MATCH_FACTS_CONFLICT')
})
