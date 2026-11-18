const { SyncApiError } = require('./auth')
const { applyEdgeReceipt } = require('../../media-retention/lib/deletion')


const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u
const HASH_PATTERN = /^[a-f0-9]{64}$/u


async function claimNextEdgeDeletion({ deviceId, tableId, now, repo }) {
  if (!ID_PATTERN.test(deviceId || '') || !ID_PATTERN.test(tableId || '')) {
    throw new SyncApiError('SYNC_INVALID')
  }
  const job = await repo.claimNext({ deviceId, tableId, now })
  if (!job) return null
  if (job.device_id !== deviceId || job.table_id !== tableId || job.status !== 'awaiting_edge'
      || !HASH_PATTERN.test(job.content_sha256 || '') || !HASH_PATTERN.test(job.claim_token || '')) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  return {
    schema_version: '1.0',
    job_id: job.job_id,
    clip_id: job.clip_id,
    content_sha256: job.content_sha256,
    claim_token: job.claim_token,
  }
}


async function completeEdgeDeletion({
  jobId, deviceId, tableId, claimToken, clipId, outcome, now, repo,
}) {
  if (!ID_PATTERN.test(jobId || '') || !ID_PATTERN.test(deviceId || '')
      || !ID_PATTERN.test(tableId || '') || !ID_PATTERN.test(clipId || '')
      || !HASH_PATTERN.test(claimToken || '')
      || !['deleted', 'already_absent'].includes(outcome)) {
    throw new SyncApiError('SYNC_INVALID')
  }
  return repo.complete({
    jobId, deviceId, tableId, claimToken, clipId, outcome, now,
  })
}


function withoutId(value) {
  const { _id: _documentId, ...rest } = value
  return rest
}


function createCloudMediaDeletionRepository(db) {
  return {
    async claimNext({ deviceId, tableId }) {
      const result = await db.collection('media_deletion_jobs').where({
        device_id: deviceId, table_id: tableId, status: 'awaiting_edge',
      }).orderBy('created_at', 'asc').limit(1).get()
      return result.data && result.data[0] ? result.data[0] : null
    },
    async complete({ jobId, deviceId, tableId, claimToken, clipId, outcome, now }) {
      return db.runTransaction(async (transaction) => {
        const jobRef = transaction.collection('media_deletion_jobs').doc(jobId)
        const result = await jobRef.get()
        const job = result.data
        if (!job || job.device_id !== deviceId || job.table_id !== tableId
            || job.clip_id !== clipId) throw new SyncApiError('SYNC_CONFLICT')
        if (job.claim_token !== claimToken || job.cloud_deleted !== true) {
          throw new SyncApiError('SYNC_CONFLICT')
        }
        if (job.status === 'completed') return withoutId(job)
        let completed
        try {
          completed = applyEdgeReceipt(job, {
            claimToken, receipt: { clip_id: clipId, device_id: deviceId, outcome }, now,
          })
        } catch (_error) { throw new SyncApiError('SYNC_CONFLICT') }
        const clipRef = transaction.collection('clips').doc(clipId)
        const clipResult = await clipRef.get()
        if (!clipResult.data || clipResult.data._device_id !== deviceId
            || clipResult.data.table_id !== tableId) throw new SyncApiError('SYNC_CONFLICT')
        await clipRef.update({ data: {
          deleted_at: now,
          storage_file_id: null,
          storage_objects: [],
          retention_state: 'released',
          deletion_job_id: jobId,
        } })
        await jobRef.set({ data: withoutId(completed) })
        return withoutId(completed)
      })
    },
  }
}


module.exports = {
  claimNextEdgeDeletion,
  completeEdgeDeletion,
  createCloudMediaDeletionRepository,
}
