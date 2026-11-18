const crypto = require('crypto')

const { ALGORITHM_VERSION } = require('./rating-engine')
const { replayRatingLedger } = require('./rating-ledger')


function createRatingReplayWorker({ repository }) {
  return {
    async runNext({ rankingScopeId }) {
      const request = await repository.claimNext(rankingScopeId)
      if (!request) return { status: 'idle' }
      try {
        const matches = await repository.loadSourceMatches(request)
        const affected = matches.find(item => item.match_id === request.affected_match_id)
        if (!affected) throw new Error('RATING_REPLAY_AFFECTED_MATCH_MISSING')
        const replayMatches = matches.map(item => {
          if (item.match_id !== request.affected_match_id) return item
          if (request.action === 'revoke') return { ...item, status: 'revoked' }
          if (request.action === 'correct' && request.corrected_result) {
            return {
              ...item, result_revision: item.result_revision + 1,
              winner_role: request.corrected_result.winner_role, status: 'active',
            }
          }
          throw new Error('RATING_REPLAY_REQUEST_INVALID')
        })
        const replay = replayRatingLedger(replayMatches)
        const generationId = crypto.createHash('sha256').update([
          request.source_generation_id, request.replay_request_id, ALGORITHM_VERSION,
        ].join(':'), 'utf8').digest('hex')
        await repository.commit({ request, generation_id: generationId, replay, matches: replayMatches })
        return { status: 'completed', replay_request_id: request.replay_request_id, generation_id: generationId }
      } catch (error) {
        await repository.fail(request, error)
        throw error
      }
    },
  }
}


module.exports = { createRatingReplayWorker }
