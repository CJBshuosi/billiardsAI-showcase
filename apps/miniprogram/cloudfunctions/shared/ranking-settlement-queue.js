const crypto = require('crypto')


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


async function optional(ref) {
  try {
    const result = await ref.get()
    return result && result.data ? result.data : null
  } catch (error) {
    if (String(error && error.message).includes('does not exist')) return null
    throw error
  }
}


function clean(value) {
  if (!value) return null
  const { _id: _documentId, ...document } = value
  return structuredClone(document)
}


function createRankingSettlementQueue(db, { now }) {
  const repository = {
    async enqueue({ matchId, rankingScopeId }) {
      const id = digest(matchId)
      return db.runTransaction(async transaction => {
        const ref = transaction.collection('ranking_settlement_jobs').doc(id)
        const existing = await optional(ref)
        if (existing) {
          if (existing.match_id !== matchId || existing.ranking_scope_id !== rankingScopeId) {
            throw new Error('RANKING_SETTLEMENT_JOB_CONFLICT')
          }
          return clean(existing)
        }
        const timestamp = now()
        const job = {
          schema_version: '1.0', match_id: matchId, ranking_scope_id: rankingScopeId,
          status: 'pending', attempt_count: 0, created_at: timestamp, updated_at: timestamp,
        }
        await ref.set({ data: job })
        return structuredClone(job)
      })
    },

    async claim(matchId) {
      return db.runTransaction(async transaction => {
        const ref = transaction.collection('ranking_settlement_jobs').doc(digest(matchId))
        const job = await optional(ref)
        if (!job || job.status !== 'pending') return null
        const claimed = {
          ...clean(job), status: 'processing', attempt_count: (job.attempt_count || 0) + 1,
          claimed_at: now(), updated_at: now(),
        }
        await ref.set({ data: claimed })
        return structuredClone(claimed)
      })
    },

    async claimNext(rankingScopeId) {
      const result = await db.collection('ranking_settlement_jobs')
        .where({ ranking_scope_id: rankingScopeId, status: 'pending' })
        .orderBy('created_at', 'asc').limit(1).get()
      const candidate = result.data && result.data[0]
      return candidate ? repository.claim(candidate.match_id) : null
    },

    async finish(job, settlement) {
      const status = settlement.status === 'review_pending' ? 'review_pending' : 'completed'
      await db.collection('ranking_settlement_jobs').doc(digest(job.match_id)).update({ data: {
        status, settlement_status: settlement.status,
        rating_event_id: settlement.rating_event_id || null,
        completed_at: now(), updated_at: now(),
      } })
    },

    async finishByMatch(matchId, settlement) {
      const ref = db.collection('ranking_settlement_jobs').doc(digest(matchId))
      const job = await optional(ref)
      if (!job) return
      await ref.update({ data: {
        status: settlement && settlement.status === 'review_pending' ? 'review_pending' : 'completed',
        settlement_status: settlement && settlement.status || 'not_ranked',
        rating_event_id: settlement && settlement.rating_event_id || null,
        completed_at: now(), updated_at: now(),
      } })
    },

    async retry(job, error) {
      await db.collection('ranking_settlement_jobs').doc(digest(job.match_id)).update({ data: {
        status: 'pending', failure_code: error && (error.code || error.message) || 'UNKNOWN',
        updated_at: now(),
      } })
    },
  }
  return repository
}


module.exports = { createRankingSettlementQueue }
