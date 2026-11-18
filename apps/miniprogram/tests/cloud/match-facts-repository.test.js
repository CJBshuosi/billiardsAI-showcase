const {
  createMatchFactsRepository,
  eventDocumentId,
  matchDocumentId,
} = require('../../cloudfunctions/shared/match-facts-repository')
const { reduceMatchEvents } = require('../../cloudfunctions/shared/match-facts')


const RECORDED_AT = '2026-09-01T08:00:00.000Z'


function createdEvent() {
  return {
    schema_version: '1.0',
    event_id: 'event-1',
    match_id: 'match-001',
    sequence: 1,
    event_type: 'match_created',
    actor_user_id: 'system',
    occurred_at: '2026-09-01T07:59:00.000Z',
    payload: {
      session_id: 'session-001',
      ranking_scope_id: 'zhoukou-urban',
      venue_id: 'venue-001',
      table_id: 'table-001',
      red_user_id: 'user-red',
      blue_user_id: 'user-blue',
    },
  }
}


function event(sequence, eventType, actorUserId, payload) {
  return {
    schema_version: '1.0',
    event_id: `event-${sequence}`,
    match_id: 'match-001',
    sequence,
    event_type: eventType,
    actor_user_id: actorUserId,
    occurred_at: new Date(Date.parse('2026-09-01T07:59:00.000Z') + (sequence - 1) * 1000).toISOString(),
    payload,
  }
}


function completeSingleEvents() {
  return [
    createdEvent(),
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


function fakeDatabase() {
  const maps = { matches: new Map(), match_events: new Map() }
  const writes = []
  let transactions = 0
  const collection = name => ({
    where(criteria) {
      const matches = [...maps[name].values()].filter(value => (
        Object.entries(criteria).every(([key, expected]) => value[key] === expected)
      ))
      return { limit() { return this }, async get() { return { data: structuredClone(matches) } } }
    },
    doc(id) {
      return {
        async get() {
          const value = maps[name].get(id)
          return { data: value ? structuredClone(value) : null }
        },
        async set({ data }) {
          writes.push({ operation: 'set', collection: name, id, data: structuredClone(data) })
          maps[name].set(id, { _id: id, ...structuredClone(data) })
        },
      }
    },
  })
  return {
    maps,
    writes,
    transactions: () => transactions,
    collection,
    async runTransaction(callback) {
      transactions += 1
      return callback({ collection })
    },
  }
}


test('first event atomically creates the append-only event and rebuildable match projection', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })

  const projection = await repository.appendEvent(createdEvent())

  expect(db.transactions()).toBe(1)
  expect(db.writes).toHaveLength(2)
  expect(eventDocumentId('event-1')).toBe('ce36863f51b6baf9d16397ffb3e9af506b284a816f72d487e55943c1fd974d6d')
  expect(matchDocumentId('match-001')).toBe('9d6b06a0b3d83904d424b95ca9f96d08849b26a6358f854494b8051ee5aa904a')

  expect(db.maps.match_events.get(eventDocumentId('event-1'))).toMatchObject({
    ...createdEvent(),
    recorded_at: RECORDED_AT,
  })
  expect(db.maps.matches.get(matchDocumentId('match-001'))).toMatchObject({
    schema_version: '1.0',
    match_id: 'match-001',
    session_id: 'session-001',
    event_ids: ['event-1'],
    latest_sequence: 1,
    created_at: RECORDED_AT,
    updated_at: RECORDED_AT,
    projection: { status: 'awaiting_settings', event_count: 1 },
  })
  expect(projection).toMatchObject({ status: 'awaiting_settings', event_count: 1 })
  expect(db.maps.matches.get(matchDocumentId('match-001')).projection)
    .toEqual(reduceMatchEvents([createdEvent()]))
  const { _id, recorded_at, ...rehydratedEvent } = db.maps.match_events.get(eventDocumentId('event-1'))
  expect({ _id, recorded_at }).toEqual({
    _id: eventDocumentId('event-1'),
    recorded_at: RECORDED_AT,
  })
  expect(rehydratedEvent).toEqual(createdEvent())

  const stored = await repository.getMatch('match-001')
  expect(db.transactions()).toBe(2)
  expect(stored).toEqual({ projection, events: [createdEvent()] })
  expect(Object.hasOwn(stored.events[0], 'recorded_at')).toBe(false)
  expect(Object.hasOwn(stored.events[0], '_id')).toBe(false)
  await expect(repository.getBySession('session-001')).resolves.toEqual(stored)
  expect(db.maps.matches.get(matchDocumentId('match-001'))).toMatchObject({
    red_user_id: 'user-red', blue_user_id: 'user-blue', status: 'awaiting_settings',
  })
})


test('ordered appends replay a complete single match to one fact package', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  let projection

  for (const matchEvent of completeSingleEvents()) {
    projection = await repository.appendEvent(matchEvent)
  }

  expect(projection).toMatchObject({
    status: 'fact_package_ready',
    event_count: 8,
    latest_sequence: 8,
    result_confirmations: ['blue', 'red'],
  })
  expect(db.transactions()).toBe(8)
  expect(db.maps.match_events.size).toBe(8)
  expect(db.maps.matches.size).toBe(1)
  expect(db.writes).toHaveLength(16)
  expect(db.maps.matches.get(matchDocumentId('match-001'))).toMatchObject({
    event_ids: completeSingleEvents().map(item => item.event_id),
    latest_sequence: 8,
    updated_at: RECORDED_AT,
  })
})

test('pending result lookup uses indexed participant fields and returns replayed facts', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  for (const matchEvent of completeSingleEvents().slice(0, 6)) await repository.appendEvent(matchEvent)

  await expect(repository.listPendingForUser('user-red')).resolves.toEqual([
    expect.objectContaining({ projection: expect.objectContaining({
      match_id: 'match-001', status: 'awaiting_result_confirmations',
    }) }),
  ])
  await expect(repository.listPendingForUser('outsider')).resolves.toEqual([])
})


test('retrying an identical event is write-free and preserves its first recorded time', async () => {
  const db = fakeDatabase()
  let authorityNow = RECORDED_AT
  const repository = createMatchFactsRepository(db, { now: () => authorityNow })
  const firstProjection = await repository.appendEvent(createdEvent())
  authorityNow = '2026-09-01T09:00:00.000Z'
  const writesBeforeRetry = db.writes.length

  const retriedProjection = await repository.appendEvent(structuredClone(createdEvent()))

  expect(retriedProjection).toEqual(firstProjection)
  expect(db.writes).toHaveLength(writesBeforeRetry)
  expect(db.maps.match_events.get(eventDocumentId('event-1')).recorded_at).toBe(RECORDED_AT)
})


test('retrying an old event after later appends returns the latest projection', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  for (const matchEvent of completeSingleEvents().slice(0, 5)) {
    await repository.appendEvent(matchEvent)
  }
  const writesBeforeRetry = db.writes.length

  const projection = await repository.appendEvent(createdEvent())

  expect(projection).toMatchObject({ status: 'in_progress', event_count: 5, latest_sequence: 5 })
  expect(db.writes).toHaveLength(writesBeforeRetry)
})


test('reusing an event ID with different content is an idempotency conflict', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  await repository.appendEvent(createdEvent())
  const changed = createdEvent()
  changed.payload.table_id = 'table-002'
  const writesBeforeConflict = db.writes.length

  await expect(repository.appendEvent(changed)).rejects.toMatchObject({
    name: 'MatchFactsError',
    code: 'MATCH_FACTS_CONFLICT',
  })
  expect(db.writes).toHaveLength(writesBeforeConflict)
})


test('an existing event without its match projection fails closed', async () => {
  const db = fakeDatabase()
  const orphan = createdEvent()
  db.maps.match_events.set(eventDocumentId(orphan.event_id), {
    _id: eventDocumentId(orphan.event_id),
    ...structuredClone(orphan),
    recorded_at: RECORDED_AT,
  })
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })

  await expect(repository.appendEvent(orphan)).rejects.toMatchObject({
    code: 'MATCH_FACTS_CONFLICT',
  })
  expect(db.writes).toHaveLength(0)
})


test('a missing historical event blocks both reads and later appends', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  await repository.appendEvent(createdEvent())
  db.maps.match_events.delete(eventDocumentId('event-1'))
  const writesBeforeConflict = db.writes.length

  await expect(repository.getMatch('match-001')).rejects.toMatchObject({ code: 'MATCH_FACTS_CONFLICT' })
  await expect(repository.appendEvent(completeSingleEvents()[1]))
    .rejects.toMatchObject({ code: 'MATCH_FACTS_CONFLICT' })
  expect(db.writes).toHaveLength(writesBeforeConflict)
})


test('a saved projection that differs from replay blocks later events', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  await repository.appendEvent(createdEvent())
  const stored = db.maps.matches.get(matchDocumentId('match-001'))
  stored.projection.status = 'fact_package_ready'
  const writesBeforeConflict = db.writes.length

  await expect(repository.appendEvent(completeSingleEvents()[1]))
    .rejects.toMatchObject({ code: 'MATCH_FACTS_CONFLICT' })
  expect(db.writes).toHaveLength(writesBeforeConflict)
})


test.each([
  ['match_id', 'match-002'],
  ['latest_sequence', 99],
])('mismatched match document field %s fails closed', async (field, value) => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  await repository.appendEvent(createdEvent())
  db.maps.matches.get(matchDocumentId('match-001'))[field] = value
  const writesBeforeConflict = db.writes.length

  await expect(repository.appendEvent(completeSingleEvents()[1]))
    .rejects.toMatchObject({ code: 'MATCH_FACTS_CONFLICT' })
  expect(db.writes).toHaveLength(writesBeforeConflict)
})


test('a different first event cannot replace an existing match', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  await repository.appendEvent(createdEvent())
  const replacement = createdEvent()
  replacement.event_id = 'replacement-event'
  replacement.payload.table_id = 'table-002'
  const writesBeforeConflict = db.writes.length

  await expect(repository.appendEvent(replacement)).rejects.toMatchObject({
    code: 'MATCH_FACTS_CONFLICT',
  })
  expect(db.writes).toHaveLength(writesBeforeConflict)
})


test('invalid authority time is rejected before a transaction or write', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => '2026-09-01T08:00:00' })

  await expect(repository.appendEvent(createdEvent())).rejects.toMatchObject({
    code: 'MATCH_FACTS_INVALID',
  })
  expect(db.transactions()).toBe(0)
  expect(db.writes).toHaveLength(0)
})


test('repository cloning cannot erase an undefined extra field before domain validation', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  const candidate = createdEvent()
  candidate.trusted = undefined

  await expect(repository.appendEvent(candidate)).rejects.toMatchObject({
    code: 'MATCH_FACTS_INVALID',
  })
  expect(db.writes).toHaveLength(0)
})


test('repository inputs and returned values do not retain mutable database references', async () => {
  const db = fakeDatabase()
  const repository = createMatchFactsRepository(db, { now: () => RECORDED_AT })
  const input = createdEvent()
  const inputBefore = structuredClone(input)
  const projection = await repository.appendEvent(input)
  expect(input).toEqual(inputBefore)
  projection.status = 'tampered-client-copy'
  input.payload.table_id = 'tampered-input'

  const firstRead = await repository.getMatch('match-001')
  firstRead.projection.status = 'tampered-read-copy'
  firstRead.events[0].payload.table_id = 'tampered-read-event'
  const secondRead = await repository.getMatch('match-001')

  expect(secondRead.projection.status).toBe('awaiting_settings')
  expect(secondRead.events[0].payload.table_id).toBe('table-001')
  await expect(repository.getMatch('match-does-not-exist')).resolves.toBeNull()
})
