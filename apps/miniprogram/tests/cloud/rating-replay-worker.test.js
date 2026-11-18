const { createRatingReplayWorker } = require('../../cloudfunctions/shared/rating-replay-worker')


function match(sequence, winnerRole = 'red') {
  return {
    schema_version: '1.0', match_id: `match-${sequence}`, result_revision: 1,
    ranking_scope_id: 'zhoukou-urban', scope_sequence: sequence,
    red_user_id: 'red', blue_user_id: `blue-${sequence}`, winner_role: winnerRole,
    format: 'single', status: 'active', started_at: `2026-09-01T11:00:0${sequence}.000Z`,
    settled_at: `2026-09-01T12:00:0${sequence}.000Z`,
  }
}


test.each([
  ['revoke', null, ['match-2']],
  ['correct', { winner_role: 'blue', red_score: 0, blue_score: 1 }, ['match-1', 'match-2']],
])('replay %s rebuilds a new immutable generation in original sequence order', async (action, correctedResult, expectedIds) => {
  const commits = []
  const request = {
    replay_request_id: `request-${action}`, ranking_scope_id: 'zhoukou-urban',
    source_generation_id: 'generation-1', affected_match_id: 'match-1',
    action, corrected_result: correctedResult, status: 'pending',
  }
  const worker = createRatingReplayWorker({ repository: {
    async claimNext() { return structuredClone(request) },
    async loadSourceMatches() { return [match(1), match(2)] },
    async commit(value) { commits.push(value); return { generation_id: value.generation_id } },
    async fail() { throw new Error('must not fail') },
  } })
  const result = await worker.runNext({ rankingScopeId: 'zhoukou-urban' })
  expect(result.status).toBe('completed')
  expect(commits[0].replay.events.map(item => item.match_id)).toEqual(expectedIds)
  expect(commits[0].generation_id).toMatch(/^[a-f0-9]{64}$/u)
  if (action === 'correct') expect(commits[0].replay.events[0].winner_role).toBe('blue')
})
