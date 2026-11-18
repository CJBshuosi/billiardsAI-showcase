const {
  claimDeletionJob,
  prepareDeletionJob,
} = require('./deletion')


function withoutId(value) {
  const { _id: _documentId, ...rest } = value
  return rest
}


async function optional(ref) {
  try { return (await ref.get()).data || null } catch (error) {
    if (String(error && error.message).includes('does not exist')) return null
    throw error
  }
}


async function authorityFor(scope, clip) {
  const [favorites, evidence] = await Promise.all([
    scope.collection('clip_favorites').where({ clip_id: clip.clip_id }).limit(100).get(),
    scope.collection('clip_evidence_holds').where({ clip_id: clip.clip_id }).limit(100).get(),
  ])
  return {
    clip,
    favorites: favorites.data || [],
    evidenceHolds: evidence.data || [],
    pendingJobs: [],
  }
}


function createCloudRetentionRepository(db) {
  return {
    async listCandidateAuthorities({ limit }) {
      const result = await db.collection('clips').where({
        source_kind: 'synthetic_retention_test',
        upload_status: 'published',
        retention_state: 'active',
      }).limit(Math.min(Math.max(limit || 1, 1), 100)).get()
      return Promise.all((result.data || []).map(clip => authorityFor(db, clip)))
    },

    async createPending(proposed) {
      return db.runTransaction(async (transaction) => {
        const existing = await optional(
          transaction.collection('media_deletion_jobs').doc(proposed.job_id),
        )
        if (existing) return withoutId(existing)
        const clipRef = transaction.collection('clips').doc(proposed.clip_id)
        const clip = await optional(clipRef)
        if (!clip || clip.retention_state !== 'active'
            || clip.source_kind !== 'synthetic_retention_test') {
          throw new Error('retention authority changed')
        }
        const authority = await authorityFor(transaction, clip)
        let rebuilt
        try { rebuilt = prepareDeletionJob({ ...authority, now: proposed.created_at }) } catch (_error) {
          throw new Error('retention authority changed')
        }
        if (!rebuilt || rebuilt.job_id !== proposed.job_id) {
          throw new Error('retention authority changed')
        }
        await clipRef.update({ data: {
          retention_state: 'deletion_pending', deletion_job_id: proposed.job_id,
        } })
        await transaction.collection('media_deletion_jobs').doc(proposed.job_id)
          .set({ data: withoutId(rebuilt) })
        return withoutId(rebuilt)
      })
    },

    async claim(job, options) {
      return db.runTransaction(async (transaction) => {
        const ref = transaction.collection('media_deletion_jobs').doc(job.job_id)
        const current = await optional(ref)
        if (!current) throw new Error('deletion job does not exist')
        const claimed = claimDeletionJob(current, options)
        await ref.set({ data: withoutId(claimed) })
        return withoutId(claimed)
      })
    },

    async saveCloudDeletion(job) {
      return db.runTransaction(async (transaction) => {
        const ref = transaction.collection('media_deletion_jobs').doc(job.job_id)
        const current = await optional(ref)
        if (!current || current.status !== 'claimed' || current.claim_token !== job.claim_token
            || current.revision !== job.revision || job.status !== 'awaiting_edge'
            || job.cloud_deleted !== true) throw new Error('deletion claim changed')
        await ref.set({ data: withoutId(job) })
        return withoutId(job)
      })
    },

    async markFailed(job) {
      if (!job || !job.job_id) return
      const ref = db.collection('media_deletion_jobs').doc(job.job_id)
      if (!await optional(ref)) return
      await ref.update({
        data: { status: 'failed', last_error_code: 'RETENTION_DELETE_RETRY' },
      })
    },
  }
}


module.exports = { createCloudRetentionRepository }
