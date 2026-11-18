const { submitMatchReport, submitReviewAppeal } = require('../../cloudfunctions/client-api/lib/ranking-reports')


function match() {
  return { projection: { match_id: 'match-1', participants: { red: 'red', blue: 'blue' }, result: { result_revision: 1 } } }
}


test('participant can report a specific match and appeal its review hold', async () => {
  const calls = []
  const matchRepository = { async getMatch() { return match() } }
  await submitMatchReport({
    matchId: 'match-1', actorUserId: 'red', reasonCode: 'suspected_substitution',
    matchRepository, reportRepository: { async save(value) { calls.push(['report', value]); return value } },
  })
  await submitReviewAppeal({
    matchId: 'match-1', actorUserId: 'blue', reasonCode: 'result_is_valid',
    matchRepository,
    reviewRepository: { async appendEvent(value) { calls.push(['appeal', value]); return { status: 'review_hold' } } },
  })
  expect(calls[0][1]).toMatchObject({ match_id: 'match-1', reporter_user_id: 'red' })
  expect(calls[1][1]).toMatchObject({ eventType: 'appeal_submitted', actorId: 'blue' })
})


test('outsiders cannot report or appeal another match', async () => {
  await expect(submitMatchReport({
    matchId: 'match-1', actorUserId: 'outsider', reasonCode: 'suspected_substitution',
    matchRepository: { async getMatch() { return match() } }, reportRepository: {},
  })).rejects.toMatchObject({ code: 'RANKING_MATCH_FORBIDDEN' })
})


test('a report cannot be stored without a concrete result revision', async () => {
  await expect(submitMatchReport({
    matchId: 'match-1', actorUserId: 'red', reasonCode: 'result_incorrect',
    matchRepository: { async getMatch() { return {
      projection: { match_id: 'match-1', participants: { red: 'red', blue: 'blue' }, result: null },
    } } },
    reportRepository: { async save() { throw new Error('must not save') } },
  })).rejects.toMatchObject({ code: 'RANKING_REPORT_INVALID' })
})
