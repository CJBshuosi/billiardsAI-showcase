const { createRatingLedgerRepository } = require('../../cloudfunctions/shared/rating-ledger-repository')


const NOW = '2026-09-01T12:00:00.000Z'


function eligible(matchId = 'match-001') {
  return {
    schema_version: '1.0', match_id: matchId, result_revision: 1,
    ranking_scope_id: 'zhoukou-urban', is_test: false,
    configuration_version: 'zhoukou-ranking-v1',
    status: 'eligible', reason_codes: ['ELIGIBLE'],
  }
}


function ratedMatch(matchId = 'match-001', override = {}) {
  return {
    schema_version: '1.0', match_id: matchId, result_revision: 1,
    ranking_scope_id: 'zhoukou-urban', is_test: false,
    red_user_id: 'user-red', blue_user_id: 'user-blue',
    winner_role: 'red', format: 'single', started_at: '2026-09-01T11:00:00.000Z',
    settled_at: NOW, ...override,
  }
}


function fakeDatabase() {
  const names = [
    'rating_scope_heads', 'rating_generations', 'rating_events',
    'rating_match_consumptions', 'player_skill_states',
  ]
  const maps = Object.fromEntries(names.map(name => [name, new Map()]))
  const writes = []
  let transactions = 0
  const collection = name => ({
    doc(id) { return {
      async get() {
        const value = maps[name].get(id)
        return { data: value ? structuredClone(value) : null }
      },
      async set({ data }) {
        writes.push({ collection: name, id, data: structuredClone(data) })
        maps[name].set(id, { _id: id, ...structuredClone(data) })
      },
    } },
  })
  return {
    maps, writes, collection, transactions: () => transactions,
    async runTransaction(callback) {
      transactions += 1
      return callback({ collection })
    },
  }
}


test('first eligible match atomically creates one ledger event and two skill states', async () => {
  const db = fakeDatabase()
  const repository = createRatingLedgerRepository(db, { now: () => NOW })

  const event = await repository.settleEligibleMatch({
    qualification: eligible(), match: ratedMatch(),
  })

  expect(db.transactions()).toBe(1)
  expect(event).toMatchObject({
    scope_sequence: 1, match_id: 'match-001', result_revision: 1,
    algorithm_version: 'glicko-zhoukou-v1-candidate',
    before: { red: { rating: 1000 }, blue: { rating: 1000 } },
    status: 'active',
  })
  expect(db.maps.rating_events.size).toBe(1)
  expect(db.maps.rating_match_consumptions.size).toBe(1)
  expect(db.maps.player_skill_states.size).toBe(2)
  expect([...db.maps.rating_scope_heads.values()][0]).toMatchObject({ next_sequence: 2 })
  expect([...db.maps.rating_generations.values()][0]).toMatchObject({
    status: 'active', algorithm_version: 'glicko-zhoukou-v1-candidate', is_test: false,
  })
  for (const collection of [
    'rating_scope_heads', 'rating_events', 'rating_match_consumptions', 'player_skill_states',
  ]) {
    expect([...db.maps[collection].values()].every(document => document.is_test === false)).toBe(true)
  }
})


test('an existing generation cannot switch between formal and test data', async () => {
  const db = fakeDatabase()
  const repository = createRatingLedgerRepository(db, { now: () => NOW })
  await repository.settleEligibleMatch({ qualification: eligible(), match: ratedMatch() })

  await expect(repository.settleEligibleMatch({
    qualification: { ...eligible('match-002'), is_test: true },
    match: { ...ratedMatch('match-002'), is_test: true },
  })).rejects.toMatchObject({ code: 'RANKING_TEST_GENERATION_INVALID' })
})


test('duplicate consumption returns the original event without writes', async () => {
  const db = fakeDatabase()
  const repository = createRatingLedgerRepository(db, { now: () => NOW })
  const first = await repository.settleEligibleMatch({ qualification: eligible(), match: ratedMatch() })
  const writesBeforeRetry = db.writes.length

  const retry = await repository.settleEligibleMatch({
    qualification: structuredClone(eligible()), match: structuredClone(ratedMatch()),
  })

  expect(retry).toEqual(first)
  expect(db.writes).toHaveLength(writesBeforeRetry)
})


test('later matches use a server-assigned sequence and current simultaneous pre-states', async () => {
  const db = fakeDatabase()
  const repository = createRatingLedgerRepository(db, { now: () => NOW })
  const first = await repository.settleEligibleMatch({ qualification: eligible(), match: ratedMatch() })
  const secondMatch = ratedMatch('match-002', {
    red_user_id: 'user-red', blue_user_id: 'user-third', winner_role: 'blue',
    settled_at: '2026-09-01T12:01:00.000Z',
  })
  const second = await repository.settleEligibleMatch({
    qualification: eligible('match-002'), match: secondMatch,
  })

  expect(second.scope_sequence).toBe(2)
  expect(second.before.red).toEqual(first.after.red)
  expect(second.before.blue.rating).toBe(1000)
  expect(second.after.red.rating).toBeLessThan(second.before.red.rating)
})


test('non-eligible decisions and changed duplicate results fail closed', async () => {
  const db = fakeDatabase()
  const repository = createRatingLedgerRepository(db, { now: () => NOW })
  const pending = { ...eligible(), status: 'pending_confirmation', reason_codes: ['RESULT_NOT_CONFIRMED'] }
  await expect(repository.settleEligibleMatch({ qualification: pending, match: ratedMatch() }))
    .rejects.toMatchObject({ code: 'RATING_SETTLEMENT_NOT_ELIGIBLE' })
  expect(db.transactions()).toBe(0)

  await repository.settleEligibleMatch({ qualification: eligible(), match: ratedMatch() })
  await expect(repository.settleEligibleMatch({
    qualification: eligible(), match: ratedMatch('match-001', { winner_role: 'blue' }),
  })).rejects.toMatchObject({ code: 'RATING_SETTLEMENT_CONFLICT' })
})
