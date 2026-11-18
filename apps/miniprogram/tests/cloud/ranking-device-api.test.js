const {
  completeRankingStart,
  proposeDeviceResult,
} = require('../../cloudfunctions/sync-api/lib/ranking-device')


test('applied breaker command records the formal match start exactly through the setup service', async () => {
  const calls = []
  const service = { async startFromDevice(input) { calls.push(input); return { status: 'in_progress' } } }
  await expect(completeRankingStart({
    command: { status: 'applied', command_type: 'set_breaker', session_id: 'session-1' },
    matchSetupService: service,
  })).resolves.toEqual({ status: 'in_progress' })
  await expect(completeRankingStart({
    command: { status: 'applied', command_type: 'continue_turn', session_id: 'session-1' },
    matchSetupService: service,
  })).resolves.toBeNull()
  expect(calls).toEqual([{ sessionId: 'session-1' }])
})


test('authenticated table device can propose a result only for its own started match', async () => {
  const calls = []
  const result = await proposeDeviceResult({
    tableId: 'table-1', sessionId: 'session-1', winnerRole: 'blue', redScore: 1, blueScore: 3,
    finishReason: 'normal',
    matchRepository: { async getBySession() { return { projection: { table_id: 'table-1', status: 'in_progress' } } } },
    matchSetupService: { async proposeResultFromDevice(input) { calls.push(input); return { status: 'awaiting_result_confirmations' } } },
  })
  expect(result.status).toBe('awaiting_result_confirmations')
  expect(calls[0]).toMatchObject({ sessionId: 'session-1', winnerRole: 'blue', redScore: 1, blueScore: 3 })
})


test('retrying the same device result after the first write is idempotent', async () => {
  const projection = {
    session_id: 'session-1',
    table_id: 'table-1',
    status: 'awaiting_result_confirmations',
    result: {
      result_revision: 1,
      winner_role: 'red',
      red_score: 1,
      blue_score: 0,
      finish_reason: 'normal',
    },
  }
  const matchSetupService = {
    async proposeResultFromDevice() {
      throw new Error('an idempotent retry must not append a second result event')
    },
  }

  await expect(proposeDeviceResult({
    tableId: 'table-1', sessionId: 'session-1', winnerRole: 'red',
    redScore: 1, blueScore: 0, finishReason: 'normal',
    matchRepository: { async getBySession() { return { projection } } },
    matchSetupService,
  })).resolves.toEqual(projection)
})
