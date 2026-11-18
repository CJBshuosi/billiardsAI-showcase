const { evaluateMatchEligibility } = require('./match-qualification')
const { evaluateMatchRisk } = require('./match-risk')


class RankingPipelineError extends Error {
  constructor(code) {
    super(code)
    this.name = 'RankingPipelineError'
    this.code = code
  }
}


function fail(code) {
  throw new RankingPipelineError(code)
}


function completionTime(match) {
  const confirmations = match.events.filter(event => event.event_type === 'result_confirmed')
  if (confirmations.length !== 2) fail('RANKING_PIPELINE_FACTS_INVALID')
  const times = confirmations.map(event => Date.parse(event.occurred_at))
  if (times.some(time => !Number.isFinite(time))) fail('RANKING_PIPELINE_FACTS_INVALID')
  return new Date(Math.max(...times)).toISOString()
}


function ratingMatch(match, isTest) {
  const { projection } = match
  if (!projection || !projection.result || !projection.settings || !projection.participants) {
    fail('RANKING_PIPELINE_FACTS_INVALID')
  }
  return {
    schema_version: '1.0',
    match_id: projection.match_id,
    result_revision: projection.result.result_revision,
    ranking_scope_id: projection.ranking_scope_id,
    is_test: isTest,
    red_user_id: projection.participants.red,
    blue_user_id: projection.participants.blue,
    winner_role: projection.result.winner_role,
    format: projection.settings.format,
    started_at: projection.started_at,
    settled_at: completionTime(match),
  }
}


function requireDependency(value, method) {
  if (!value || typeof value[method] !== 'function') fail('RANKING_PIPELINE_CONFIGURATION_INVALID')
}


function createRankingSettlementPipeline({
  matchFactsRepository,
  authorityRepository,
  qualificationRepository,
  riskRepository,
  reviewRepository,
  ratingRepository,
}) {
  requireDependency(matchFactsRepository, 'getMatch')
  requireDependency(authorityRepository, 'readForMatch')
  requireDependency(qualificationRepository, 'save')
  requireDependency(riskRepository, 'collectSignals')
  requireDependency(riskRepository, 'save')
  requireDependency(reviewRepository, 'open')
  requireDependency(ratingRepository, 'settleEligibleMatch')

  return {
    async process({ matchId }) {
      if (typeof matchId !== 'string' || matchId.length === 0) fail('RANKING_PIPELINE_INVALID')
      const match = await matchFactsRepository.getMatch(matchId)
      if (!match || !match.projection || match.projection.match_id !== matchId) {
        fail('RANKING_PIPELINE_MATCH_NOT_FOUND')
      }

      const authority = await authorityRepository.readForMatch(match.projection)
      const qualification = evaluateMatchEligibility({ projection: match.projection, authority })
      await qualificationRepository.save(qualification)
      if (qualification.status !== 'eligible') {
        return { status: qualification.status, reason_codes: qualification.reason_codes }
      }

      const signals = await riskRepository.collectSignals({ match, authority })
      const risk = evaluateMatchRisk(signals)
      const assessment = {
        schema_version: '1.0', match_id: matchId,
        result_revision: match.projection.result.result_revision,
        ranking_scope_id: match.projection.ranking_scope_id,
        ...risk,
      }
      await riskRepository.save(assessment)
      if (risk.status === 'blocked') {
        return { status: 'not_ranked', reason_codes: risk.hard_conflicts }
      }
      if (risk.status === 'review_hold') {
        const reviewState = typeof reviewRepository.getState === 'function'
          ? await reviewRepository.getState(matchId, match.projection.result.result_revision) : null
        if (reviewState && reviewState.status === 'released'
            && reviewState.settlement_allowed === true) {
          const event = await ratingRepository.settleEligibleMatch({
            qualification,
            match: ratingMatch(match, qualification.is_test),
          })
          return {
            status: 'settled', rating_event_id: event.rating_event_id,
            scope_sequence: event.scope_sequence,
          }
        }
        if (reviewState && ['not_ranked', 'corrected', 'revoked'].includes(reviewState.status)) {
          return { status: 'not_ranked', reason_codes: ['REVIEW_NOT_RELEASED'] }
        }
        await reviewRepository.open({
          match_id: matchId,
          result_revision: match.projection.result.result_revision,
          reason_codes: risk.soft_signals,
        })
        return { status: 'review_pending', reason_codes: risk.soft_signals }
      }

      const event = await ratingRepository.settleEligibleMatch({
        qualification,
        match: ratingMatch(match, qualification.is_test),
      })
      return {
        status: 'settled',
        rating_event_id: event.rating_event_id,
        scope_sequence: event.scope_sequence,
      }
    },
  }
}


module.exports = {
  RankingPipelineError,
  createRankingSettlementPipeline,
  ratingMatch,
}
