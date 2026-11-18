const {
  ALGORITHM_VERSION,
  initialSkillState,
  placementProgress,
  updateMatchRatings,
} = require('../cloudfunctions/shared/rating-engine')


const SETTLED_AT = '2026-09-01T12:00:00.000Z'


function rounded(state) {
  return {
    rating: Number(state.rating.toFixed(2)),
    deviation: Number(state.deviation.toFixed(2)),
    matches: state.rated_matches,
    opponents: state.distinct_opponent_ids.length,
  }
}


function play(red, blue, winnerRole, format = 'single') {
  return updateMatchRatings({ red, blue, winnerRole, format, settledAt: SETTLED_AT }).after
}


function repeatedWins(count) {
  let red = initialSkillState('repeat-red')
  let blue = initialSkillState('repeat-blue')
  const redDeltas = []
  for (let index = 0; index < count; index += 1) {
    const before = red.rating
    ;({ red, blue } = play(red, blue, 'red'))
    redDeltas.push(Number((red.rating - before).toFixed(2)))
  }
  return { red: rounded(red), blue: rounded(blue), red_deltas: redDeltas }
}


function placementScenario(opponentIds) {
  let player = initialSkillState('placement-player')
  for (const opponentId of opponentIds) {
    const opponent = initialSkillState(opponentId)
    player = play(player, opponent, 'red').red
  }
  return { state: rounded(player), progress: placementProgress(player) }
}


function simulate() {
  const equalRed = initialSkillState('equal-red')
  const equalBlue = initialSkillState('equal-blue')
  const single = play(equalRed, equalBlue, 'red', 'single')
  const race = play(equalRed, equalBlue, 'red', 'race')
  const strong = { ...initialSkillState('strong'), rating: 1300, deviation: 100 }
  const weak = { ...initialSkillState('weak'), rating: 700, deviation: 100 }
  const expected = play(strong, weak, 'red')
  const upset = play(strong, weak, 'blue')

  return {
    algorithm_version: ALGORITHM_VERSION,
    synthetic_only: true,
    equal_first_single: { red: rounded(single.red), blue: rounded(single.blue) },
    equal_first_race: { red: rounded(race.red), blue: rounded(race.blue) },
    strong_expected_win: { strong: rounded(expected.red), weak: rounded(expected.blue) },
    weak_upset: { strong: rounded(upset.red), weak: rounded(upset.blue) },
    ten_repeated_wins: repeatedWins(10),
    placement_three_distinct: placementScenario(['opponent-a', 'opponent-b', 'opponent-c']),
    placement_ten_two_opponents: placementScenario([
      'opponent-a', 'opponent-b', 'opponent-a', 'opponent-b', 'opponent-a',
      'opponent-b', 'opponent-a', 'opponent-b', 'opponent-a', 'opponent-b',
    ]),
  }
}


if (require.main === module) process.stdout.write(`${JSON.stringify(simulate(), null, 2)}\n`)


module.exports = { simulate }
