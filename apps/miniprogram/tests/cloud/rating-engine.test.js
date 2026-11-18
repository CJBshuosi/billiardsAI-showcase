const {
  ALGORITHM_VERSION,
  RatingEngineError,
  initialSkillState,
  placementProgress,
  updateMatchRatings,
} = require('../../cloudfunctions/shared/rating-engine')


const SETTLED_AT = '2026-09-01T12:00:00.000Z'


test('new players start at a hidden 1000 with high uncertainty', () => {
  expect(initialSkillState('user-red')).toEqual({
    schema_version: '1.0',
    algorithm_version: 'glicko-zhoukou-v1-candidate',
    user_id: 'user-red',
    rating: 1000,
    deviation: 350,
    rated_matches: 0,
    distinct_opponent_ids: [],
    last_rated_at: null,
  })
  expect(ALGORITHM_VERSION).toBe('glicko-zhoukou-v1-candidate')
})


test('equal players move in the correct direction from the same pre-match state', () => {
  const red = initialSkillState('user-red')
  const blue = initialSkillState('user-blue')
  const before = structuredClone({ red, blue })

  const result = updateMatchRatings({ red, blue, winnerRole: 'red', format: 'single', settledAt: SETTLED_AT })

  expect(result.expected.red).toBeCloseTo(0.5, 12)
  expect(result.expected.blue).toBeCloseTo(0.5, 12)
  expect(result.after.red.rating).toBeGreaterThan(1000)
  expect(result.after.blue.rating).toBeLessThan(1000)
  expect(result.after.red.rating - 1000).toBeCloseTo(1000 - result.after.blue.rating, 10)
  expect(result.after.red.rated_matches).toBe(1)
  expect(result.after.red.distinct_opponent_ids).toEqual(['user-blue'])
  expect(result.after.blue.distinct_opponent_ids).toEqual(['user-red'])
  expect({ red, blue }).toEqual(before)
})


test('an upset moves ratings more than the expected strong-player win', () => {
  const strong = { ...initialSkillState('strong'), rating: 1300, deviation: 100 }
  const weak = { ...initialSkillState('weak'), rating: 700, deviation: 100 }

  const expected = updateMatchRatings({
    red: strong, blue: weak, winnerRole: 'red', format: 'single', settledAt: SETTLED_AT,
  })
  const upset = updateMatchRatings({
    red: strong, blue: weak, winnerRole: 'blue', format: 'single', settledAt: SETTLED_AT,
  })

  expect(expected.expected.red).toBeGreaterThan(0.9)
  expect(expected.after.red.rating - strong.rating)
    .toBeLessThan(Math.abs(upset.after.red.rating - strong.rating))
  expect(upset.after.blue.rating - weak.rating)
    .toBeGreaterThan(Math.abs(expected.after.blue.rating - weak.rating))
})


test('a high-uncertainty newcomer moves faster than a stable player', () => {
  const opponent = { ...initialSkillState('opponent'), deviation: 100 }
  const newcomer = initialSkillState('newcomer')
  const stable = { ...initialSkillState('stable'), deviation: 60, rated_matches: 30 }

  const newcomerWin = updateMatchRatings({
    red: newcomer, blue: opponent, winnerRole: 'red', format: 'single', settledAt: SETTLED_AT,
  })
  const stableWin = updateMatchRatings({
    red: stable, blue: opponent, winnerRole: 'red', format: 'single', settledAt: SETTLED_AT,
  })

  expect(newcomerWin.after.red.rating - newcomer.rating)
    .toBeGreaterThan(stableWin.after.red.rating - stable.rating)
})


test('race format changes confidence only, not the rating delta', () => {
  const red = { ...initialSkillState('red'), deviation: 180 }
  const blue = { ...initialSkillState('blue'), deviation: 180 }
  const single = updateMatchRatings({ red, blue, winnerRole: 'red', format: 'single', settledAt: SETTLED_AT })
  const race = updateMatchRatings({ red, blue, winnerRole: 'red', format: 'race', settledAt: SETTLED_AT })

  expect(race.after.red.rating).toBeCloseTo(single.after.red.rating, 12)
  expect(race.after.blue.rating).toBeCloseTo(single.after.blue.rating, 12)
  expect(race.after.red.deviation).toBeLessThan(single.after.red.deviation)
  expect(race.after.blue.deviation).toBeLessThan(single.after.blue.deviation)
})


test('placement requires three rated matches against three distinct opponents', () => {
  const base = initialSkillState('user-main')
  expect(placementProgress(base)).toEqual({
    is_placed: false, rated_matches: 0, distinct_opponents: 0,
    matches_needed: 3, distinct_opponents_needed: 3,
  })

  const twoOpponents = {
    ...base, rated_matches: 10, distinct_opponent_ids: ['opponent-a', 'opponent-b'],
  }
  expect(placementProgress(twoOpponents)).toMatchObject({
    is_placed: false, matches_needed: 0, distinct_opponents_needed: 1,
  })

  const placed = {
    ...base, rated_matches: 3,
    distinct_opponent_ids: ['opponent-c', 'opponent-a', 'opponent-b'],
  }
  expect(placementProgress(placed)).toEqual({
    is_placed: true, rated_matches: 3, distinct_opponents: 3,
    matches_needed: 0, distinct_opponents_needed: 0,
  })
})


test('A becomes placed only after A-B A-C and A-D, while a repeat opponent stays distinct once', () => {
  let playerA = initialSkillState('user-a')
  const opponents = ['user-b', 'user-c', 'user-d']

  opponents.forEach((opponentId, index) => {
    const result = updateMatchRatings({
      red: playerA,
      blue: initialSkillState(opponentId),
      winnerRole: 'red',
      format: 'single',
      settledAt: `2026-09-01T12:0${index}:00.000Z`,
    })
    playerA = result.after.red
    expect(placementProgress(playerA).is_placed).toBe(index === 2)
  })

  expect(placementProgress(playerA)).toMatchObject({
    is_placed: true,
    rated_matches: 3,
    distinct_opponents: 3,
  })

  playerA = updateMatchRatings({
    red: playerA,
    blue: initialSkillState('user-b'),
    winnerRole: 'blue',
    format: 'single',
    settledAt: '2026-09-01T12:03:00.000Z',
  }).after.red
  expect(placementProgress(playerA)).toMatchObject({
    is_placed: true,
    rated_matches: 4,
    distinct_opponents: 3,
  })
})


test.each([
  [{ winnerRole: 'green', format: 'single' }, 'RATING_MATCH_INVALID'],
  [{ winnerRole: 'red', format: 'best-of-three' }, 'RATING_MATCH_INVALID'],
])('invalid match input fails closed', (override, code) => {
  expect(() => updateMatchRatings({
    red: initialSkillState('red'), blue: initialSkillState('blue'),
    settledAt: SETTLED_AT, ...override,
  })).toThrow(expect.objectContaining({ name: 'RatingEngineError', code }))
  expect(new RatingEngineError(code)).toMatchObject({ message: code, code })
})
