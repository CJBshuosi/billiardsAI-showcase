const { enqueuePendingOwnerDeviceDeletion } = require('./owner-device-deletion-dispatch')
const { canonicalJson, documentId, isImmutablePrivateClipOwner, validOwnership, validSavedSelection } = require('./shot-ownership')
const { makeStorageIdentity, createCloudMediaStorageAuthority, MediaStorageAuthorityError } = require('./media-storage-authority')
const ID = /^[A-Za-z0-9._-]{1,128}$/u

class OwnerClipDeletionError extends Error {
  constructor(code) { super(code); this.name = 'OwnerClipDeletionError'; this.code = code }
}
async function optional(ref) {
  try { return (await ref.get()).data ?? null } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error?.message || '')) return null
    throw error
  }
}
// Presence is the fence. Even malformed server records must never grant access.
async function getOwnerClipTombstone(databaseOrTransaction, clipId) {
  return optional(databaseOrTransaction.collection('owner_clip_tombstones').doc(clipId))
}
function validTime(value) {
  return typeof value === 'string' && value.endsWith('Z') && Number.isFinite(Date.parse(value))
}
function deletionJobId(clipId) { return `owner-delete-${documentId(clipId).slice(0, 32)}` }
function validReceipt(receipt, clipId) {
  return receipt && receipt.schema_version === '1.0' && receipt.clip_id === clipId
    && ID.test(receipt.request_id || '') && receipt.deletion_job_id === deletionJobId(clipId)
    && receipt.status === 'deletion_requested' && validTime(receipt.requested_at)
    && Object.keys(receipt).length === 6
}
function storageInventory(clip) {
  const inventory = [{ kind: 'published', file_id: clip.storage_file_id }]
  if (clip.storage_objects !== undefined) {
    if (!Array.isArray(clip.storage_objects)) throw new OwnerClipDeletionError('OWNER_DELETE_UNAVAILABLE')
    inventory.push(...clip.storage_objects)
  }
  const files = new Map()
  for (const object of inventory) {
    if (!object || !['published', 'staging', 'derivative'].includes(object.kind)
        || typeof object.file_id !== 'string' || !object.file_id.startsWith('cloud://')
        || object.file_id.length <= 8 || /[\u0000-\u001f]/u.test(object.file_id)) {
      throw new OwnerClipDeletionError('OWNER_DELETE_UNAVAILABLE')
    }
    if (!files.has(object.file_id)) files.set(object.file_id, { kind: object.kind, file_id: object.file_id })
  }
  return [...files.values()]
}
// Internal only: a future adapter must resolve userId from trusted authentication.
async function requestOwnerClipDeletion({ clipId, requestId, userId, repo, now } = {}) {
  if (![clipId, requestId, userId].every(value => typeof value === 'string' && ID.test(value))
      || !validTime(now) || !repo || typeof repo.request !== 'function') {
    throw new OwnerClipDeletionError('OWNER_DELETE_INVALID')
  }
  return repo.request({ clipId, requestId, userId, now })
}
function createCloudOwnerDeleteRepository(db) {
  const storageAuthority = createCloudMediaStorageAuthority(db)
  return { async request({ clipId, requestId, userId, now }) {
    return db.runTransaction(async transaction => {
      const requestKey = documentId(`${userId}:${requestId}`)
      const requestRef = transaction.collection('owner_clip_delete_requests').doc(requestKey)
      const old = await optional(requestRef)
      if (old !== null && (old.user_id !== userId || old.request_id !== requestId || old.clip_id !== clipId)) {
        throw new OwnerClipDeletionError('OWNER_DELETE_CONFLICT')
      }
      const tombstone = await getOwnerClipTombstone(transaction, clipId)
      if (tombstone !== null) {
        if (tombstone.schema_version !== '1.0' || tombstone.clip_id !== clipId || tombstone.owner_user_id !== userId
            || !validOwnership(tombstone) || !validSavedSelection(tombstone)
            || !ID.test(tombstone.device_id || '') || !ID.test(tombstone.table_id || '')
            || !validReceipt(tombstone.receipt, clipId)) throw new OwnerClipDeletionError('OWNER_DELETE_UNAVAILABLE')
        if (old !== null) {
          if (!validReceipt(old.receipt, clipId)
              || canonicalJson(old.receipt) !== canonicalJson(tombstone.receipt)) {
            throw new OwnerClipDeletionError('OWNER_DELETE_CONFLICT')
          }
          const job = await optional(transaction.collection('owner_clip_deletion_jobs').doc(tombstone.receipt.deletion_job_id))
          if (job && job.device_id === tombstone.device_id && job.table_id === tombstone.table_id && job.clip_id === clipId) await enqueuePendingOwnerDeviceDeletion(transaction, { job })
          return old.receipt
        }
        const job = await optional(transaction.collection('owner_clip_deletion_jobs').doc(tombstone.receipt.deletion_job_id))
        if (job && job.device_id === tombstone.device_id && job.table_id === tombstone.table_id && job.clip_id === clipId) await enqueuePendingOwnerDeviceDeletion(transaction, { job })
        await requestRef.set({ data: { schema_version: '1.0', user_id: userId,
          request_id: requestId, clip_id: clipId, receipt: tombstone.receipt } })
        return tombstone.receipt
      }
      if (old !== null) throw new OwnerClipDeletionError('OWNER_DELETE_UNAVAILABLE')
      const clipRef = transaction.collection('clips').doc(clipId)
      const clip = await optional(clipRef)
      const projection = clip && await optional(transaction.collection('field_shot_ownership')
        .doc(documentId(`${clip.session_id}:${clip.shot_id}`)))
      if (!clip || clip.clip_id !== clipId || !isImmutablePrivateClipOwner(clip, userId, projection)
          || !/^[0-9a-f]{64}$/u.test(clip.content_sha256 || '')) {
        throw new OwnerClipDeletionError('OWNER_DELETE_UNAVAILABLE')
      }
      const jobId = deletionJobId(clipId)
      const jobRef = transaction.collection('owner_clip_deletion_jobs').doc(jobId)
      if (await optional(jobRef) !== null) throw new OwnerClipDeletionError('OWNER_DELETE_CONFLICT')
      const receipt = { schema_version: '1.0', clip_id: clipId, request_id: requestId,
        deletion_job_id: jobId, status: 'deletion_requested', requested_at: now }
      let authoritySummary
      try {
        const identity = makeStorageIdentity(clip, { tableId: clip.table_id, deviceId: clip._device_id, kind: 'saved_private' })
        authoritySummary = await storageAuthority.closeForDeletion(transaction, identity)
      } catch (error) {
        if (error instanceof MediaStorageAuthorityError) throw new OwnerClipDeletionError('OWNER_DELETE_UNAVAILABLE')
        throw error
      }
      const frozen = {
        clip_id: clipId, owner_user_id: userId, actor_role: clip.actor_role,
        session_id: clip.session_id, shot_id: clip.shot_id, device_id: clip._device_id,
        table_id: clip.table_id, artifact_id: clip.artifact_id, source_job_id: clip.job_id,
        content_sha256: clip.content_sha256, byte_size: clip.byte_size,
        ownership_receipt: clip.ownership_receipt, ownership_receipt_sha256: clip.ownership_receipt_sha256,
        save_request_id: clip.save_request_id, save_receipt_sha256: clip.save_receipt_sha256,
        postgame_manifest_sha256: clip.postgame_manifest_sha256,
        storage_objects: storageInventory(clip),
        storage_authority: authoritySummary,
        playback_authorized_until: clip.playback_authorized_until || null,
      }
      await transaction.collection('owner_clip_tombstones').doc(clipId).set({ data: {
        schema_version: '1.0', ...frozen, receipt, requested_at: now,
      } })
      await requestRef.set({ data: { schema_version: '1.0', user_id: userId,
        request_id: requestId, clip_id: clipId, receipt } })
      const job = { schema_version: '1.0', ...frozen,
        job_id: jobId, deletion_job_id: jobId, job_kind: 'owner_delete_v1', status: 'pending_cloud',
        requested_at: now, receipt }
      await jobRef.set({ data: job })
      await enqueuePendingOwnerDeviceDeletion(transaction, { job })
      await clipRef.update({ data: { retention_state: 'deletion_requested' } })
      return receipt
    })
  } }
}
module.exports = { OwnerClipDeletionError, requestOwnerClipDeletion, createCloudOwnerDeleteRepository, getOwnerClipTombstone }
