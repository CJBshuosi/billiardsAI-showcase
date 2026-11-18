const {
  createRankingSettlementQueue,
} = require('../../cloudfunctions/shared/ranking-settlement-queue')


function fakeDatabase() {
  const documents = new Map()
  const collection = () => ({
    doc(id) { return {
      async get() { return { data: documents.get(id) || null } },
      async set({ data }) { documents.set(id, { _id: id, ...structuredClone(data) }) },
      async update({ data }) { documents.set(id, { ...documents.get(id), ...structuredClone(data) }) },
    } },
    where(query) { return {
      orderBy() { return this }, limit() { return this },
      async get() {
        return { data: [...documents.values()].filter(item => Object.entries(query)
          .every(([key, value]) => item[key] === value)).slice(0, 1) }
      },
    } },
  })
  return { documents, collection, async runTransaction(callback) { return callback({ collection }) } }
}


test('settlement queue persists, claims and completes one match idempotently', async () => {
  const db = fakeDatabase()
  const queue = createRankingSettlementQueue(db, { now: () => '2026-09-01T12:00:00.000Z' })
  await queue.enqueue({ matchId: 'match-1', rankingScopeId: 'zhoukou-urban' })
  await queue.enqueue({ matchId: 'match-1', rankingScopeId: 'zhoukou-urban' })
  const job = await queue.claim('match-1')
  expect(job).toMatchObject({ match_id: 'match-1', status: 'processing' })
  await queue.finish(job, { status: 'settled', rating_event_id: 'rating-1' })
  expect(await queue.claim('match-1')).toBeNull()
  expect([...db.documents.values()][0]).toMatchObject({
    status: 'completed', settlement_status: 'settled', rating_event_id: 'rating-1',
  })
})


test('temporary worker failure returns the claimed job to pending', async () => {
  const db = fakeDatabase()
  const queue = createRankingSettlementQueue(db, { now: () => '2026-09-01T12:00:00.000Z' })
  await queue.enqueue({ matchId: 'match-2', rankingScopeId: 'zhoukou-urban' })
  const job = await queue.claim('match-2')
  await queue.retry(job, new Error('temporary'))
  expect(await queue.claimNext('zhoukou-urban')).toMatchObject({
    match_id: 'match-2', status: 'processing', attempt_count: 2,
  })
})
