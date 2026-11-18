const { ClientApiError } = require('./errors')
const { buildParticipantOutcome } = require('./ranking-match-outcome')


const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u


function roleFor(projection, actorUserId) {
  if (projection.participants.red === actorUserId) return 'red'
  if (projection.participants.blue === actorUserId) return 'blue'
  throw new ClientApiError('RANKING_MATCH_FORBIDDEN')
}


function publicMatch(projection, actorUserId) {
  const role = roleFor(projection, actorUserId)
  return {
    schema_version: '1.0', match_id: projection.match_id, session_id: projection.session_id,
    status: projection.status, actor_role: role,
    settings: projection.settings ? structuredClone(projection.settings) : null,
    settings_confirmations: structuredClone(projection.settings_confirmations || []),
    result: projection.result ? structuredClone(projection.result) : null,
    result_confirmations: structuredClone(projection.result_confirmations || []),
    actor_result_confirmed: (projection.result_confirmations || []).includes(role),
    result_confirmation_deadline_at: projection.result_confirmation_deadline_at || null,
    dispute: projection.dispute ? { reason: projection.dispute.reason } : null,
  }
}

function publicRankingScope(scope) {
  if (!scope) return null
  return {
    ranking_scope_id: scope.ranking_scope_id,
    display_name: scope.display_name,
    mode: scope.mode,
    is_test: scope.is_test === true,
  }
}


function createRankingMatchApi({
  matchRepository, fieldRepository, rankingRepository, outcomeRepository, setupService, resultService,
}) {
  async function requireByMatch(matchId, actorUserId) {
    if (!ID_PATTERN.test(matchId || '') || !ID_PATTERN.test(actorUserId || '')) {
      throw new ClientApiError('RANKING_MATCH_INVALID')
    }
    const match = await matchRepository.getMatch(matchId)
    if (!match) throw new ClientApiError('RANKING_MATCH_NOT_FOUND')
    roleFor(match.projection, actorUserId)
    return match
  }

  return {
    async get({ sessionId, actorUserId }) {
      if (!ID_PATTERN.test(sessionId || '') || !ID_PATTERN.test(actorUserId || '')) {
        throw new ClientApiError('RANKING_MATCH_INVALID')
      }
      const match = await matchRepository.getBySession(sessionId)
      if (!match) return { status: 'not_configured', session_id: sessionId }
      const result = publicMatch(match.projection, actorUserId)
      const scope = match.projection.ranking_scope_id && rankingRepository.getScope
        ? await rankingRepository.getScope(match.projection.ranking_scope_id) : null
      const scopedResult = { ...result, ranking_scope: publicRankingScope(scope) }
      if (!match.projection.result) return scopedResult
      const outcome = await outcomeRepository.read(match.projection)
      return {
        ...scopedResult,
        ranking_outcome: buildParticipantOutcome({
          projection: match.projection, actorUserId, ...outcome,
        }),
      }
    },

    async proposeSettings(input) {
      const field = await fieldRepository.getAuthorizationContext({
        sessionId: input.sessionId, actorUserId: input.actorUserId,
      })
      if (!field || field.joined !== true || field.started
          || !field.red_user_id || !field.blue_user_id
          || ![field.red_user_id, field.blue_user_id].includes(input.actorUserId)) {
        throw new ClientApiError('RANKING_MATCH_FORBIDDEN')
      }
      const table = await rankingRepository.getTable(field.table_id)
      const venue = table && await rankingRepository.getVenue(table.venue_id)
      if (!table || table.status !== 'approved' || !venue || venue.status !== 'approved') {
        throw new ClientApiError('RANKING_MATCH_VENUE_UNAVAILABLE')
      }
      const projection = await setupService.proposeSettings({
        context: {
          session_id: field.session_id, table_id: field.table_id,
          venue_id: venue.venue_id, ranking_scope_id: venue.ranking_scope_id,
          red_user_id: field.red_user_id, blue_user_id: field.blue_user_id,
        },
        actorUserId: input.actorUserId, mode: input.mode, format: input.format,
        raceTarget: input.raceTarget, firstBreaker: input.firstBreaker,
        rankingIntent: input.rankingIntent,
      })
      return publicMatch(projection, input.actorUserId)
    },

    async confirmSettings({ sessionId, actorUserId }) {
      const match = await matchRepository.getBySession(sessionId)
      if (!match) throw new ClientApiError('RANKING_MATCH_NOT_FOUND')
      roleFor(match.projection, actorUserId)
      const projection = await setupService.confirmSettings({ sessionId, actorUserId })
      return publicMatch(projection, actorUserId)
    },

    async listPendingResults({ actorUserId }) {
      if (!ID_PATTERN.test(actorUserId || '')) throw new ClientApiError('RANKING_MATCH_INVALID')
      const matches = await matchRepository.listPendingForUser(actorUserId)
      return matches.map(match => publicMatch(match.projection, actorUserId))
    },

    async confirmResult({ matchId, actorUserId }) {
      await requireByMatch(matchId, actorUserId)
      const projection = await resultService.confirm({ matchId, actorUserId })
      return publicMatch(projection, actorUserId)
    },

    async disputeResult({ matchId, actorUserId, reason }) {
      await requireByMatch(matchId, actorUserId)
      const projection = await resultService.dispute({ matchId, actorUserId, reason })
      return publicMatch(projection, actorUserId)
    },
  }
}


module.exports = { createRankingMatchApi, publicMatch, publicRankingScope }
