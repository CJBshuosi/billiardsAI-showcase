const {
  RatingLedgerError,
  replayRatingLedger,
} = require('../../cloudfunctions/shared/rating-ledger')


function match(sequence, red, blue, winnerRole, override = {}) {
  return {
    schema_version: '1.0',
    match_id: `match-${sequence}`,
    result_revision: 1,
    ranking_scope_id: 'zhoukou-urban',
    scope_sequence: sequence,
    started_at: new Date(Date.parse('2026-09-01T11:00:00.000Z') + sequence * 1000).toISOString(),
    settled_at: new Date(Date.parse('2026-09-01T12:00:00.000Z') + sequence * 1000).toISOString(),
    red_user_id: red,
    blue_user_id: blue,
    winner_role: winnerRole,
    format: 'single',
    status: 'active',
    ...override,
  }
}


test('eligible matches replay into immutable ordered ledger events and player states', () => {
  const result = replayRatingLedger([
    match(1, 'user-a', 'user-b', 'red'),
    match(2, 'user-a', 'user-c', 'red'),
    match(3, 'user-a', 'user-d', 'red'),
  ])

  expect(result.events).toHaveLength(3)
  expect(result.events[0]).toMatchObject({
    scope_sequence: 1,
    match_id: 'match-1',
    algorithm_version: 'glicko-zhoukou-v1-candidate',
    before: { red: { rating: 1000 }, blue: { rating: 1000 } },
  })
  expect(result.events[2].after.red.rated_matches).toBe(3)
  expect(result.states['user-a'].distinct_opponent_ids).toEqual(['user-b', 'user-c', 'user-d'])
  expect(result.states['user-a'].rating).toBeGreaterThan(1000)
})


test('replay sorts by scope sequence and is deterministic', () => {
  const ordered = [
    match(1, 'user-a', 'user-b', 'red'),
    match(2, 'user-b', 'user-c', 'blue'),
    match(3, 'user-a', 'user-c', 'blue'),
  ]

  const first = replayRatingLedger([ordered[2], ordered[0], ordered[1]])
  const second = replayRatingLedger(structuredClone(ordered))

  expect(first).toEqual(second)
  expect(first.events.map(item => item.scope_sequence)).toEqual([1, 2, 3])
})


test('revocation recomputes later matches instead of mechanically reversing points', () => {
  const originalMatches = [
    match(1, 'user-a', 'user-b', 'red'),
    match(2, 'user-a', 'user-c', 'red'),
    match(3, 'user-b', 'user-c', 'blue'),
  ]
  const original = replayRatingLedger(originalMatches)
  const revoked = replayRatingLedger([
    { ...originalMatches[0], status: 'revoked' },
    originalMatches[1],
    originalMatches[2],
  ])
  const rebuiltWithoutFirst = replayRatingLedger([originalMatches[1], originalMatches[2]])

  expect(revoked).toEqual(rebuiltWithoutFirst)
  expect(revoked.events.map(item => item.match_id)).toEqual(['match-2', 'match-3'])
  expect(revoked.states['user-a'].rating).not.toBe(original.states['user-a'].rating)
})


test('challenge origin and score margin are not rating inputs', () => {
  const ordinary = match(1, 'user-a', 'user-b', 'red', { match_origin: 'ordinary', red_score: 1, blue_score: 0 })
  const challenge = match(1, 'user-a', 'user-b', 'red', { match_origin: 'challenge', red_score: 5, blue_score: 0 })

  expect(replayRatingLedger([challenge])).toEqual(replayRatingLedger([ordinary]))
})


test.each([
  [
    [match(1, 'a', 'b', 'red'), match(1, 'c', 'd', 'red', { match_id: 'match-other' })],
    'RATING_LEDGER_SEQUENCE_CONFLICT',
  ],
  [
    [match(1, 'a', 'b', 'red'), match(2, 'c', 'd', 'red', { match_id: 'match-1' })],
    'RATING_LEDGER_MATCH_CONFLICT',
  ],
  [
    [match(1, 'a', 'b', 'red', { status: 'pending' })],
    'RATING_LEDGER_INVALID',
  ],
])('invalid ledger input fails closed', (matches, code) => {
  expect(() => replayRatingLedger(matches)).toThrow(expect.objectContaining({
    name: 'RatingLedgerError', code,
  }))
  expect(new RatingLedgerError(code)).toMatchObject({ message: code, code })
})


test('ledger replay does not mutate match inputs', () => {
  const matches = [match(1, 'a', 'b', 'red'), match(2, 'b', 'c', 'blue')]
  const before = structuredClone(matches)

  replayRatingLedger(matches)

  expect(matches).toEqual(before)
})
