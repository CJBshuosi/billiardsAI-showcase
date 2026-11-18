function createRankingExpiryService({
  repository, resultConfirmationService, challengeService,
}) {
  return {
    async expireDue() {
      const [matches, challenges] = await Promise.all([
        repository.listDueResultMatches(), repository.listDueChallenges(),
      ])
      for (const matchId of matches) await resultConfirmationService.expire({ matchId })
      for (const challengeId of challenges) await challengeService.expire({ challengeId })
      return { expired_results: matches.length, expired_challenges: challenges.length }
    },
  }
}


function createCloudRankingExpiryRepository(db, { now }) {
  return {
    async listDueResultMatches() {
      const result = await db.collection('matches').where({
        status: 'awaiting_result_confirmations',
        result_confirmation_deadline_at: db.command.lte(now()),
      }).limit(100).get()
      return (result.data || []).map(item => item.match_id)
    },
    async listDueChallenges() {
      const result = await db.collection('challenges').where({
        status: 'pending', expires_at: db.command.lte(now()),
      }).limit(100).get()
      return (result.data || []).map(item => item.challenge_id)
    },
  }
}


module.exports = { createCloudRankingExpiryRepository, createRankingExpiryService }
