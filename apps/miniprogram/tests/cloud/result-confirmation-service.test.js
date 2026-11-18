const { createMatchFactsRepository } = require('../../cloudfunctions/shared/match-facts-repository')
const {
  ResultConfirmationError,
  createResultConfirmationService,
} = require('../../cloudfunctions/shared/result-confirmation-service')


function event(sequence, eventType, actorUserId, payload) {
  return {
    schema_version: '1.0',
    event_id: `setup-event-${sequence}`,
    match_id: 'match-001',
    sequence,
    event_type: eventType,
    actor_user_id: actorUserId,
    occurred_at: new Date(Date.parse('2026-09-01T00:00:00.000Z') + sequence * 1000).toISOString(),
    payload,
  }
}


function pendingResultEvents() {
  return [
    event(1, 'match_created', 'system', {
      session_id: 'session-001', ranking_scope_id: 'zhoukou-urban',
      venue_id: 'venue-001', table_id: 'table-001',
      red_user_id: 'user-red', blue_user_id: 'user-blue',
    }),
    event(2, 'settings_proposed', 'user-red', {
      settings_revision: 1, mode: 'observation', format: 'single', race_target: null,
      first_breaker: 'red', ranking_intent: true,
    }),
    event(3, 'settings_confirmed', 'user-red', { settings_revision: 1, role: 'red' }),
    event(4, 'settings_confirmed', 'user-blue', { settings_revision: 1, role: 'blue' }),
    event(5, 'match_started', 'system', { settings_revision: 1 }),
    event(6, 'result_proposed', 'system', {
      result_revision: 1, winner_role: 'red', red_score: 1, blue_score: 0,
      finish_reason: 'normal',
    }),
  ]
}


function fakeDatabase() {
  const maps = { matches: new Map(), match_events: new Map() }
  const collection = name => ({
    doc(id) {
      return {
        async get() {
          const value = maps[name].get(id)
          return { data: value ? structuredClone(value) : null }
        },
        async set({ data }) { maps[name].set(id, { _id: id, ...structuredClone(data) }) },
      }
    },
  })
  return {
    maps,
    collection,
    async runTransaction(callback) { return callback({ collection }) },
  }
}


async function fixture() {
  const db = fakeDatabase()
  let currentTime = '2026-09-01T00:01:00.000Z'
  const now = () => currentTime
  const repository = createMatchFactsRepository(db, { now })
  for (const matchEvent of pendingResultEvents()) await repository.appendEvent(matchEvent)
  const service = createResultConfirmationService({ repository, now })
  return { db, repository, service, setNow: value => { currentTime = value } }
}


test('a participant confirms the result using the role bound in the match projection', async () => {
  const { repository, service } = await fixture()

  const projection = await service.confirm({ matchId: 'match-001', actorUserId: 'user-red' })

  expect(projection).toMatchObject({
    status: 'awaiting_result_confirmations',
    result_confirmations: ['red'],
    latest_sequence: 7,
  })
  const stored = await repository.getMatch('match-001')
  expect(stored.events[6]).toMatchObject({
    event_type: 'result_confirmed',
    actor_user_id: 'user-red',
    payload: { result_revision: 1, role: 'red' },
  })
})


test('repeating a confirmation is idempotent before and after the other player confirms', async () => {
  const { db, service } = await fixture()
  const first = await service.confirm({ matchId: 'match-001', actorUserId: 'user-red' })
  const eventCountAfterFirst = db.maps.match_events.size

  const retry = await service.confirm({ matchId: 'match-001', actorUserId: 'user-red' })
  expect(retry).toEqual(first)
  expect(db.maps.match_events.size).toBe(eventCountAfterFirst)

  const ready = await service.confirm({ matchId: 'match-001', actorUserId: 'user-blue' })
  expect(ready.status).toBe('fact_package_ready')
  const completedRetry = await service.confirm({ matchId: 'match-001', actorUserId: 'user-red' })
  expect(completedRetry).toEqual(ready)
  expect(db.maps.match_events.size).toBe(8)
})


test('an outsider cannot confirm or dispute a match result', async () => {
  const { service } = await fixture()

  await expect(service.confirm({ matchId: 'match-001', actorUserId: 'user-outsider' }))
    .rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_UNAUTHORIZED' })
  await expect(service.dispute({
    matchId: 'match-001', actorUserId: 'user-outsider', reason: 'score_mismatch',
  })).rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_UNAUTHORIZED' })
})


test('a participant dispute is terminal and safely retryable', async () => {
  const { db, service } = await fixture()

  const disputed = await service.dispute({
    matchId: 'match-001', actorUserId: 'user-blue', reason: 'winner_mismatch',
  })
  expect(disputed).toMatchObject({
    status: 'disputed',
    dispute: { role: 'blue', reason: 'winner_mismatch' },
  })
  const eventCount = db.maps.match_events.size
  await expect(service.dispute({
    matchId: 'match-001', actorUserId: 'user-blue', reason: 'winner_mismatch',
  })).resolves.toEqual(disputed)
  expect(db.maps.match_events.size).toBe(eventCount)
  await expect(service.confirm({ matchId: 'match-001', actorUserId: 'user-red' }))
    .rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_NOT_PENDING' })
})


test('a participant cannot dispute the same result after confirming it', async () => {
  const { service } = await fixture()
  await service.confirm({ matchId: 'match-001', actorUserId: 'user-red' })

  await expect(service.dispute({
    matchId: 'match-001', actorUserId: 'user-red', reason: 'score_mismatch',
  })).rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_NOT_PENDING' })
})


test('participant actions expire at the exact deadline', async () => {
  const { service, setNow } = await fixture()
  setNow('2026-09-02T00:00:06.000Z')

  await expect(service.confirm({ matchId: 'match-001', actorUserId: 'user-red' }))
    .rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_EXPIRED' })
  await expect(service.dispute({
    matchId: 'match-001', actorUserId: 'user-blue', reason: 'other',
  })).rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_EXPIRED' })
})


test('system expiry is blocked early and becomes an explicit terminal fact when due', async () => {
  const { db, service, setNow } = await fixture()

  await expect(service.expire({ matchId: 'match-001' }))
    .rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_TOO_EARLY' })
  expect(db.maps.match_events.size).toBe(6)

  setNow('2026-09-02T00:00:06.000Z')
  const expired = await service.expire({ matchId: 'match-001' })
  expect(expired).toMatchObject({
    status: 'confirmation_expired',
    confirmation_expired_at: '2026-09-02T00:00:06.000Z',
  })
  const retry = await service.expire({ matchId: 'match-001' })
  expect(retry).toEqual(expired)
  expect(db.maps.match_events.size).toBe(7)
})


test('missing matches, non-pending states and invalid reasons expose stable service errors', async () => {
  const { repository, service } = await fixture()
  await expect(service.confirm({ matchId: 'missing-match', actorUserId: 'user-red' }))
    .rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_NOT_FOUND' })
  await expect(service.dispute({
    matchId: 'match-001', actorUserId: 'user-red', reason: 'free-text-accusation',
  })).rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_INVALID_REASON' })

  const nonPendingDb = fakeDatabase()
  const nonPendingRepository = createMatchFactsRepository(nonPendingDb, {
    now: () => '2026-09-01T00:01:00.000Z',
  })
  for (const matchEvent of pendingResultEvents().slice(0, 5)) {
    await nonPendingRepository.appendEvent(matchEvent)
  }
  const nonPendingService = createResultConfirmationService({
    repository: nonPendingRepository,
    now: () => '2026-09-01T00:01:00.000Z',
  })
  await expect(nonPendingService.confirm({ matchId: 'match-001', actorUserId: 'user-red' }))
    .rejects.toMatchObject({ code: 'RESULT_CONFIRMATION_NOT_PENDING' })

  expect(new ResultConfirmationError('RESULT_CONFIRMATION_NOT_FOUND')).toMatchObject({
    name: 'ResultConfirmationError',
    message: 'RESULT_CONFIRMATION_NOT_FOUND',
    code: 'RESULT_CONFIRMATION_NOT_FOUND',
  })
  await expect(repository.getMatch('match-001')).resolves.toBeTruthy()
})
