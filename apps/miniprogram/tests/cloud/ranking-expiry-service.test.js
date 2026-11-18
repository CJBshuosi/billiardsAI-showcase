const { createRankingExpiryService } = require('../../cloudfunctions/shared/ranking-expiry-service')


test('expiry service expires due result confirmations and invitations without inferring losses', async () => {
  const calls = []
  const service = createRankingExpiryService({
    repository: {
      async listDueResultMatches() { return ['match-1'] },
      async listDueChallenges() { return ['challenge-1'] },
    },
    resultConfirmationService: { async expire(input) { calls.push(['result', input]); return { status: 'confirmation_expired' } } },
    challengeService: { async expire(input) { calls.push(['challenge', input]); return { status: 'expired' } } },
  })
  const result = await service.expireDue()
  expect(result).toEqual({ expired_results: 1, expired_challenges: 1 })
  expect(calls).toEqual([
    ['result', { matchId: 'match-1' }], ['challenge', { challengeId: 'challenge-1' }],
  ])
  expect(JSON.stringify(result)).not.toContain('winner')
})
