const { createMatchSetupService } = require('../../cloudfunctions/shared/match-setup-service')
const { reduceMatchEvents } = require('../../cloudfunctions/shared/match-facts')


const TIMES = [
  '2026-09-01T10:00:00.000Z', '2026-09-01T10:00:01.000Z',
  '2026-09-01T10:00:02.000Z', '2026-09-01T10:00:03.000Z',
  '2026-09-01T10:00:04.000Z', '2026-09-01T10:00:05.000Z',
]


function repository() {
  let events = []
  return {
    events: () => structuredClone(events),
    async getBySession() {
      return events.length ? { events: structuredClone(events), projection: reduceMatchEvents(events) } : null
    },
    async getMatch() {
      return events.length ? { events: structuredClone(events), projection: reduceMatchEvents(events) } : null
    },
    async appendEvent(event) {
      if (!events.some(item => item.event_id === event.event_id)) events.push(structuredClone(event))
      return reduceMatchEvents(events)
    },
  }
}


function context() {
  return {
    session_id: 'session-1', ranking_scope_id: 'zhoukou-urban', venue_id: 'venue-1',
    table_id: 'table-1', red_user_id: 'red', blue_user_id: 'blue',
  }
}


test('participants propose and confirm one settings version before device start', async () => {
  const repo = repository(); let index = 0
  const service = createMatchSetupService({ repository: repo, now: () => TIMES[index++] })
  const proposed = await service.proposeSettings({
    context: context(), actorUserId: 'red', mode: 'observation', format: 'race',
    raceTarget: 3, firstBreaker: 'blue', rankingIntent: true,
  })
  expect(proposed).toMatchObject({ status: 'awaiting_settings_confirmations', settings: {
    settings_revision: 1, mode: 'observation', format: 'race', race_target: 3,
    first_breaker: 'blue', ranking_intent: true,
  } })
  await service.confirmSettings({ sessionId: 'session-1', actorUserId: 'red' })
  const ready = await service.confirmSettings({ sessionId: 'session-1', actorUserId: 'blue' })
  expect(ready.status).toBe('ready')
  const started = await service.startFromDevice({ sessionId: 'session-1' })
  expect(started).toMatchObject({ status: 'in_progress', started_at: TIMES[4] })
})


test('assistance cannot request ranking and ordinary end is not a result proposal', async () => {
  const repo = repository(); let index = 0
  const service = createMatchSetupService({ repository: repo, now: () => TIMES[index++] })
  await expect(service.proposeSettings({
    context: context(), actorUserId: 'red', mode: 'assistance', format: 'single',
    raceTarget: null, firstBreaker: 'red', rankingIntent: true,
  })).rejects.toMatchObject({ code: 'MATCH_SETUP_INVALID' })
  expect(repo.events()).toHaveLength(0)
  expect(service).not.toHaveProperty('endSessionAsResult')
})


test('device can propose a legal candidate result only after start', async () => {
  const repo = repository(); let index = 0
  const service = createMatchSetupService({ repository: repo, now: () => TIMES[index++] })
  await service.proposeSettings({
    context: context(), actorUserId: 'red', mode: 'observation', format: 'single',
    raceTarget: null, firstBreaker: 'red', rankingIntent: true,
  })
  await service.confirmSettings({ sessionId: 'session-1', actorUserId: 'red' })
  await service.confirmSettings({ sessionId: 'session-1', actorUserId: 'blue' })
  await service.startFromDevice({ sessionId: 'session-1' })
  const result = await service.proposeResultFromDevice({
    sessionId: 'session-1', winnerRole: 'red', redScore: 1, blueScore: 0,
    finishReason: 'normal',
  })
  expect(result).toMatchObject({
    status: 'awaiting_result_confirmations', result: { winner_role: 'red', result_revision: 1 },
  })
})
