const { createMatchReviewRepository } = require('../../cloudfunctions/shared/match-review-repository')


function fakeDatabase() {
  const maps = { match_review_states: new Map(), match_review_events: new Map() }
  const collection = name => ({ doc(id) { return {
    async get() { return { data: maps[name].get(id) || null } },
    async set({ data }) { maps[name].set(id, { _id: id, ...structuredClone(data) }) },
  } } })
  return { maps, collection, async runTransaction(callback) { return callback({ collection }) } }
}


test('review hold is append-only, auditable and idempotent', async () => {
  const db = fakeDatabase()
  const repo = createMatchReviewRepository(db, { now: () => '2026-09-01T12:00:00.000Z' })
  const input = { match_id: 'match-1', result_revision: 1, reason_codes: ['implausible_duration', 'repeated_opponent'] }
  const first = await repo.open(input)
  const retry = await repo.open(structuredClone(input))
  expect(retry).toEqual(first)
  expect(first).toMatchObject({ status: 'review_hold', settlement_allowed: false, audit_count: 1 })
  expect(db.maps.match_review_events.size).toBe(1)
  expect(db.maps.match_review_states.size).toBe(1)
})

test('participant appeal appends an audit event without directly changing score or rating', async () => {
  const db = fakeDatabase()
  const repo = createMatchReviewRepository(db, { now: () => '2026-09-01T12:00:00.000Z' })
  await repo.open({ match_id: 'match-1', result_revision: 1, reason_codes: ['implausible_duration', 'repeated_opponent'] })
  const state = await repo.appendEvent({
    matchId: 'match-1', resultRevision: 1, eventType: 'appeal_submitted',
    actorId: 'user-red', payload: { reason_code: 'result_is_valid' },
  })
  expect(state).toMatchObject({ status: 'review_hold', audit_count: 2, appeals: [{ actor_id: 'user-red' }] })
  expect(JSON.stringify(state)).not.toContain('rating_delta')
  expect(db.maps.match_review_events.size).toBe(2)
})
