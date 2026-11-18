const { processConfirmedMatch } = require('../../cloudfunctions/client-api/lib/ranking-worker-client')


test('fact package dispatches internal ranking work while delivery failure stays honestly pending', async () => {
  const calls = []
  const settled = await processConfirmedMatch({
    match: { match_id: 'match-1', status: 'fact_package_ready' }, token: 'secret',
    enqueueMatch: async matchId => calls.push(['enqueue', matchId]),
    callWorker: async input => { calls.push(input); return { result: { ok: true, data: { settlement: { status: 'settled' } } } } },
  })
  expect(settled).toEqual({ settlement_status: 'settled' })
  expect(calls[0]).toEqual(['enqueue', 'match-1'])
  expect(calls[1].data).toMatchObject({ action: 'process.match', match_id: 'match-1' })

  await expect(processConfirmedMatch({
    match: { match_id: 'match-2', status: 'fact_package_ready' }, token: 'secret',
    enqueueMatch: async () => {},
    callWorker: async () => { throw new Error('temporary') },
  })).resolves.toEqual({ settlement_status: 'pending_processing' })
})
