const crypto = require('crypto')

const { timestamp } = require('../../shared/retention-policy')
const { buildRetentionInventory } = require('./inventory')


function executionAllowed({ env, clip }) {
  return Boolean(env && env.MEDIA_RETENTION_EXECUTION_ENABLED === 'synthetic-only'
    && clip && !Object.hasOwn(clip, 'retention_policy')
    && clip.source_kind === 'synthetic_retention_test')
}


function digest(value, length = 64) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex').slice(0, length)
}


function buildDeletionJob({ clip, now }) {
  // This job authority is ordinary expiry only. User deletion needs a separate
  // authority and must never be inferred from an indefinite policy or cache.
  if (clip && Object.hasOwn(clip, 'retention_policy')) {
    throw new Error('clip retention authority is invalid')
  }
  const objectIds = Array.isArray(clip && clip.storage_objects)
    ? clip.storage_objects.map(object => object && object.file_id) : []
  if (!clip || typeof clip.clip_id !== 'string' || !clip.clip_id
      || typeof clip._device_id !== 'string' || !clip._device_id
      || typeof clip.content_sha256 !== 'string' || !/^[a-f0-9]{64}$/u.test(clip.content_sha256)
      || !Array.isArray(clip.storage_objects) || !clip.storage_objects.length
      || new Set(objectIds).size !== objectIds.length
      || clip.storage_objects.some(object => !object || !['published', 'staging', 'derivative'].includes(object.kind)
        || typeof object.file_id !== 'string' || !object.file_id.startsWith('cloud://'))) {
    throw new Error('clip storage lineage is invalid')
  }
  return {
    schema_version: '1.0',
    job_id: `delete-${digest(clip.clip_id, 32)}`,
    clip_id: clip.clip_id,
    session_id: clip.session_id,
    device_id: clip._device_id,
    table_id: clip.table_id,
    content_sha256: clip.content_sha256,
    status: 'pending',
    revision: 0,
    playback_blocked: true,
    cloud_objects: clip.storage_objects.map(({ kind, file_id: fileId }) => ({ kind, file_id: fileId })),
    created_at: now,
    updated_at: now,
  }
}


function prepareDeletionJob({ clip, favorites, evidenceHolds, pendingJobs, now }) {
  const inventory = buildRetentionInventory({
    clips: [clip], favorites, evidenceHolds, pendingJobs, now,
  })
  if (inventory.invalid.count) throw new Error('clip retention authority is invalid')
  if (inventory.due.count !== 1) return null
  return buildDeletionJob({ clip, now })
}


function claimDeletionJob(job, { workerId, now, leaseSeconds }) {
  timestamp(now)
  if (!workerId || !Number.isInteger(leaseSeconds) || leaseSeconds < 1) {
    throw new Error('deletion claim is invalid')
  }
  const expired = job.status === 'claimed' && timestamp(job.lease_expires_at) <= timestamp(now)
  if (!['pending', 'failed'].includes(job.status) && !expired) {
    throw new Error('deletion job is not claimable')
  }
  const revision = Number(job.revision || 0) + 1
  const leaseExpiresAt = new Date(timestamp(now) + leaseSeconds * 1000).toISOString()
  return {
    ...job,
    status: 'claimed',
    revision,
    claimed_by: workerId,
    claim_token: digest(`${job.job_id}:${revision}:${workerId}:${now}`),
    claimed_at: now,
    lease_expires_at: leaseExpiresAt,
    updated_at: now,
  }
}


function requireToken(job, claimToken) {
  if (!claimToken || claimToken !== job.claim_token) throw new Error('deletion claim token is invalid')
}


function applyCloudDeletion(job, { claimToken, now, deletedCount }) {
  if (job.status === 'awaiting_edge' || job.status === 'completed') return job
  requireToken(job, claimToken)
  if (job.status !== 'claimed' || deletedCount !== job.cloud_objects.length) {
    throw new Error('cloud deletion evidence is incomplete')
  }
  return {
    ...job,
    status: 'awaiting_edge',
    cloud_deleted: true,
    cloud_deleted_at: now,
    updated_at: now,
  }
}


function applyEdgeReceipt(job, { claimToken, receipt, now }) {
  requireToken(job, claimToken)
  if (job.status === 'completed') return job
  if (job.status !== 'awaiting_edge' || job.cloud_deleted !== true
      || !receipt || receipt.clip_id !== job.clip_id
      || receipt.device_id !== job.device_id
      || !['deleted', 'already_absent'].includes(receipt.outcome)) {
    throw new Error('edge receipt does not match deletion job')
  }
  return {
    ...job,
    status: 'completed',
    edge_deleted: true,
    edge_outcome: receipt.outcome,
    edge_deleted_at: now,
    completed_at: now,
    updated_at: now,
  }
}


async function executeClaimedCloudDeletion({ job, clip, env, storage, now }) {
  if (!executionAllowed({ env, clip })) throw new Error('physical deletion execution is disabled')
  if (!job || job.status !== 'claimed' || job.clip_id !== clip.clip_id
      || !storage || typeof storage.deleteObjects !== 'function') {
    throw new Error('cloud deletion request is invalid')
  }
  const fileIds = job.cloud_objects.map(object => object.file_id)
  const evidence = await storage.deleteObjects(fileIds)
  if (!evidence || evidence.deleted !== true || evidence.count !== fileIds.length) {
    throw new Error('cloud deletion evidence is incomplete')
  }
  return applyCloudDeletion(job, {
    claimToken: job.claim_token, now, deletedCount: evidence.count,
  })
}


module.exports = {
  applyCloudDeletion,
  applyEdgeReceipt,
  buildDeletionJob,
  claimDeletionJob,
  executeClaimedCloudDeletion,
  executionAllowed,
  prepareDeletionJob,
}
