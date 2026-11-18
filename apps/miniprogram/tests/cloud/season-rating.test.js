const { replaySeasonRatings } = require('../../cloudfunctions/shared/season-rating')
const { createSeason } = require('../../cloudfunctions/shared/ranking-snapshot')
const { initialSkillState } = require('../../cloudfunctions/shared/rating-engine')


function event(sequence, settledAt, winnerRole = 'red', startedAt = settledAt) {
  const red = { ...initialSkillState('red'), rating: 1200, deviation: 100, rated_matches: 10, distinct_opponent_ids: ['x', 'y', 'z'] }
  const blue = { ...initialSkillState('blue'), rating: 1100, deviation: 120, rated_matches: 8, distinct_opponent_ids: ['x', 'y', 'z'] }
  return {
    status: 'active', scope_sequence: sequence, started_at: startedAt, settled_at: settledAt,
    red_user_id: 'red', blue_user_id: 'blue', winner_role: winnerRole, format: 'single',
    before: { red, blue },
  }
}


test('season ratings inherit long-term strength but reset season samples', () => {
  const season = createSeason({ seasonId: 's1', startsAt: '2026-09-01T00:00:00.000Z' })
  const result = replaySeasonRatings({
    season,
    ratingEvents: [event(1, '2026-08-31T23:59:59.999Z'), event(2, '2026-09-02T12:00:00.000Z')],
  })
  expect(result.red).toMatchObject({
    season_id: 's1', inherited_from_rating: 1200, rated_matches: 1,
    distinct_opponent_ids: ['blue'],
  })
  expect(result.red.rating).toBeGreaterThan(1200)
  expect(result.blue.rating).toBeLessThan(1100)
})


test('revoked and out-of-window events never enter the season', () => {
  const season = createSeason({ seasonId: 's1', startsAt: '2026-09-01T00:00:00.000Z' })
  const revoked = { ...event(1, '2026-09-02T12:00:00.000Z'), status: 'revoked' }
  const atEnd = event(2, season.ends_at)
  expect(replaySeasonRatings({ season, ratingEvents: [revoked, atEnd] })).toEqual({})
})


test('season membership follows formal match start time, not later confirmation time', () => {
  const season = createSeason({ seasonId: 's1', startsAt: '2026-09-01T00:00:00.000Z' })
  const startedBefore = event(
    1, '2026-09-01T00:05:00.000Z', 'red', '2026-08-31T23:55:00.000Z',
  )
  const startedInside = event(
    2, season.ends_at, 'blue', '2026-10-30T23:55:00.000Z',
  )

  const result = replaySeasonRatings({ season, ratingEvents: [startedBefore, startedInside] })

  expect(result.red.rated_matches).toBe(1)
  expect(result.red.rating).toBeLessThan(1200)
})
