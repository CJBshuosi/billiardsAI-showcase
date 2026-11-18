const crypto = require('crypto')

const { ClientApiError } = require('./errors')


const REPORT_REASONS = new Set([
  'suspected_substitution', 'identity_mismatch', 'result_incorrect', 'match_not_completed', 'other',
])
const APPEAL_REASONS = new Set(['result_is_valid', 'identity_evidence_available', 'other'])


function participant(match, userId) {
  return match && match.projection && match.projection.participants
    && Object.values(match.projection.participants).includes(userId)
}


async function submitMatchReport({
  matchId, actorUserId, reasonCode, matchRepository, reportRepository,
}) {
  const match = await matchRepository.getMatch(matchId)
  if (!participant(match, actorUserId)) throw new ClientApiError('RANKING_MATCH_FORBIDDEN')
  if (!REPORT_REASONS.has(reasonCode) || !match.projection.result
      || !Number.isInteger(match.projection.result.result_revision)) {
    throw new ClientApiError('RANKING_REPORT_INVALID')
  }
  return reportRepository.save({
    schema_version: '1.0', match_id: matchId,
    result_revision: match.projection.result.result_revision,
    reporter_user_id: actorUserId, reason_code: reasonCode,
  })
}


async function submitReviewAppeal({
  matchId, actorUserId, reasonCode, matchRepository, reviewRepository,
}) {
  const match = await matchRepository.getMatch(matchId)
  if (!participant(match, actorUserId)) throw new ClientApiError('RANKING_MATCH_FORBIDDEN')
  if (!match.projection.result || !APPEAL_REASONS.has(reasonCode)) {
    throw new ClientApiError('RANKING_APPEAL_INVALID')
  }
  return reviewRepository.appendEvent({
    matchId, resultRevision: match.projection.result.result_revision,
    eventType: 'appeal_submitted', actorId: actorUserId,
    payload: { reason_code: reasonCode },
  })
}


function createCloudMatchReportRepository(db, { now }) {
  return {
    async save(report) {
      const id = crypto.createHash('sha256')
        .update(`${report.match_id}:${report.reporter_user_id}`, 'utf8').digest('hex')
      return db.runTransaction(async transaction => {
        const ref = transaction.collection('user_match_reports').doc(id)
        let existing = null
        try { existing = (await ref.get()).data || null } catch (_error) { existing = null }
        if (existing) {
          if (existing.reason_code !== report.reason_code) {
            throw new ClientApiError('RANKING_REPORT_ALREADY_SUBMITTED')
          }
          const { _id: _documentId, ...stored } = existing
          return stored
        }
        const stored = { ...report, submitted_at: now() }
        await ref.set({ data: stored })
        return structuredClone(stored)
      })
    },
  }
}


module.exports = {
  createCloudMatchReportRepository,
  submitMatchReport,
  submitReviewAppeal,
}
