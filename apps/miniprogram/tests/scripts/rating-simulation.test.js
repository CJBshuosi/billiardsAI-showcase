const { simulate } = require('../../scripts/simulate-city-rating')


test('candidate simulation stays synthetic and demonstrates approved ranking behavior', () => {
  const result = simulate()

  expect(result.synthetic_only).toBe(true)
  expect(result.equal_first_race.red.rating).toBe(result.equal_first_single.red.rating)
  expect(result.equal_first_race.red.deviation).toBeLessThan(result.equal_first_single.red.deviation)
  expect(result.weak_upset.weak.rating - 700)
    .toBeGreaterThan(result.strong_expected_win.strong.rating - 1300)
  expect(result.ten_repeated_wins.red_deltas[9]).toBeLessThan(result.ten_repeated_wins.red_deltas[1])
  expect(result.placement_three_distinct.progress.is_placed).toBe(true)
  expect(result.placement_ten_two_opponents.progress).toMatchObject({
    is_placed: false,
    distinct_opponents_needed: 1,
  })
})
