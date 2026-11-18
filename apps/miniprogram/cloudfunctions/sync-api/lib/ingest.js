const crypto = require('crypto')
const { contentDigest, validateSessionFacts } = require('../../shared/session-facts')
const { enqueuePublishedShots, ShotProjectionError } = require('../../shared/shot-projection')
const { getOwnerClipTombstone } = require('../../shared/owner-delete')

const { SyncApiError } = require('./auth')
const { isManagedSessionId, verifyManagedShotOwnership } = require('../../shared/managed-command-authority')
const { digest: ownershipDigest, documentId, validOwnership, validSavedSelection, ownershipProjection, matchesProjection } = require('../../shared/shot-ownership')
const { ordinaryExpiry, SAVED_UNTIL_DELETED_POLICY } = require('../../shared/retention-policy')
const { readManagedMediaBirth, makeStorageAdmission, validStorageAdmission } = require('../../shared/managed-media-admission')
const { makeStorageIdentity, createCloudMediaStorageAuthority } = require('../../shared/media-storage-authority')
const { makeManagedUploadAttempt, validManagedUploadAttempt } = require('../../shared/managed-upload-attempt')
const { validUploadReceipt } = require('../../shared/upload-terminal-receipts')
const { MANAGED_UPLOAD_REQUEST_BINDING_CONTRACT, validManagedUploadRequest, managedUploadRequestId, validManagedUploadRequestBinding } = require('../../shared/managed-upload-request')


const HASH_PATTERN = /^[0-9a-f]{64}$/u
const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u
const MAX_SAVED_CLIP_BYTES = 256 * 1024 * 1024

// Existing shared/models/miniprogram.py V1 fields; review authority is not a
// field-sync concern. Reject extension attempts instead of storing trusted-looking data.
function onlyKeys(value, keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).some(key => !keys.includes(key))) throw new SyncApiError('SYNC_INVALID')
}

// Private publication fence, not a new edge revision contract. Changed V1 facts
// have no review authority and therefore cannot overwrite an existing session.
async function publicationState(transaction, job) {
  if (job.status !== 'completed') assertGenericMediaAllowed(job.envelope)
  for (const clip of job.envelope.media) {
    if (await getOwnerClipTombstone(transaction, clip.clip_id) !== null) throw new SyncApiError('SYNC_CONFLICT')
  }
  const live = await getOptionalDocument(transaction.collection('field_live_sessions').doc(job.session_id))
  if (!live.data || live.data.session_id !== job.session_id
      || !['open', 'ended'].includes(live.data.status)
      || live.data.table_id !== job.table_id || !ID_PATTERN.test(job.table_id || '')
      || job.envelope.session.session_id !== job.session_id) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  // Existing server-approved role bindings for this ended session, never the
  // current table roster or an edge-supplied subset. Legacy missing authority
  // stays pending; submitted participants cannot certify themselves.
  const members = []
  for (const role of ['red', 'blue']) {
    const id = crypto.createHash('sha256').update(`${job.session_id}:${role}`).digest('hex')
    const { data } = await getOptionalDocument(transaction.collection('session_role_bindings').doc(id))
    if (data) {
      if (data.session_id !== job.session_id || data.role !== role
          || !ID_PATTERN.test(data.user_id || '') || !data.confirmed_at
          || (live.data[`${role}_user_id`] != null && live.data[`${role}_user_id`] !== data.user_id)) {
        throw new SyncApiError('SYNC_CONFLICT')
      }
      members.push(data)
    } else if (live.data[`${role}_user_id`]) throw new SyncApiError('SYNC_CONFLICT')
  }
  const submitted = job.envelope.session.participants
  if (!members.length || submitted.length !== members.length
      || new Set(members.map(p => p.user_id)).size !== members.length
      || members.some(p => !submitted.some(s => s.role === p.role && s.user_id === p.user_id))) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  const existing = await getOptionalDocument(transaction.collection('sessions').doc(job.session_id))
  const fence = {
    table_id: job.table_id,
    device_id: job._device_id,
    content_sha256: contentDigest(job.envelope),
  }
  if (existing.data && (existing.data._sync_publication?.table_id !== fence.table_id
      || existing.data._sync_publication?.device_id !== fence.device_id
      || existing.data._sync_publication?.content_sha256 !== fence.content_sha256)) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  for (const clip of job.envelope.media) {
    const old = await getOptionalDocument(transaction.collection('clips').doc(clip.clip_id))
    if (old.data && (!existing.data || old.data.session_id !== job.session_id
        || old.data.shot_id !== clip.shot_id || old.data.artifact_id !== clip.artifact_id
        || old.data.content_sha256 !== clip.content_sha256)) {
      throw new SyncApiError('SYNC_CONFLICT')
    }
  }
  // Edge persists its end/outbox before cloud command completion. Valid but
  // still-open sessions must remain unpublished and retry through the existing
  // SYNC_PARTIAL/503 path, not become permanently blocked by a 409. All table,
  // member and lineage conflicts above remain permanent failures.
  if (live.data.status === 'open') throw new SyncApiError('SYNC_PARTIAL')
  return { existing: existing.data, fence }
}


async function getOptionalDocument(reference) {
  try {
    return await reference.get()
  } catch (error) {
    const message = error && error.message ? error.message : String(error)
    if (/document with _id .+ does not exist/iu.test(message)) return { data: null }
    throw error
  }
}


function withoutDocumentId(document) {
  if (!document || typeof document !== 'object') return document
  const { _id: _documentId, ...data } = document
  return data
}


function validateJobInput({ envelope, idempotencyKey, manifestHash, tableId }) {
  if (!envelope || envelope.schema_version !== '1.0' || !envelope.session
      || !ID_PATTERN.test(envelope.session.session_id || '')
      || !Array.isArray(envelope.media)
      || !ID_PATTERN.test(idempotencyKey || '')
      || !HASH_PATTERN.test(manifestHash || '')
      || !ID_PATTERN.test(tableId || '')) {
    throw new SyncApiError('SYNC_INVALID')
  }
  const { session } = envelope
  const factsOnlyUsb = envelope.media.length === 0 && session.evidence_source_kind === 'usb_live'
  onlyKeys(envelope, ['schema_version', 'session', 'media'])
  let shots
  try { shots = validateSessionFacts(session, { factsOnlyUsb }) } catch (_error) {
    throw new SyncApiError('SYNC_INVALID')
  }
  const clipIds = new Set()
  const sourceKinds = new Set()
  for (const clip of envelope.media) {
    onlyKeys(clip, ['schema_version', 'clip_id', 'session_id', 'shot_id', 'artifact_id',
      'relative_path', 'source_kind', 'start_ms', 'end_ms', 'duration_ms', 'content_sha256',
      'byte_size', 'media_type'])
    if (!clip || !ID_PATTERN.test(clip.clip_id || '')
        || clip.session_id !== session.session_id || clip.artifact_id !== session.artifact_id
        || !shots.has(clip.shot_id)
        || !HASH_PATTERN.test(clip.content_sha256 || '')
        || !Number.isInteger(clip.byte_size) || clip.byte_size < 1
        || clip.media_type !== 'video/mp4'
        || !['development_replay', 'usb_live'].includes(clip.source_kind)
        || typeof clip.relative_path !== 'string'
        || clip.relative_path.includes('..') || clip.relative_path.includes('\\')
        || clip.relative_path.startsWith('/') || clipIds.has(clip.clip_id)) {
      throw new SyncApiError('SYNC_INVALID')
    }
    clipIds.add(clip.clip_id)
    sourceKinds.add(clip.source_kind)
  }
  if (sourceKinds.size > 1) throw new SyncApiError('SYNC_INVALID')
  const replayEvidence = sourceKinds.has('development_replay')
  const replayLimitation = session.limitations.includes('development_replay_media')
  const declaredReplayEvidence = session.evidence_source_kind === 'development_replay'
  if ((sourceKinds.size === 1 && replayEvidence !== replayLimitation)
      || declaredReplayEvidence !== replayLimitation) {
    throw new SyncApiError('SYNC_INVALID')
  }
  if (sourceKinds.size && !sourceKinds.has(session.evidence_source_kind)) {
    throw new SyncApiError('SYNC_INVALID')
  }
}

// Product media enters only through the postgame saved-receipt route. Keep
// historical completed jobs readable and explicit nonmanaged replay usable.
function assertGenericMediaAllowed(envelope) {
  if (envelope.media.length && (isManagedSessionId(envelope.session.session_id)
      || envelope.media.some(clip => clip.source_kind === 'usb_live'))) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
}

async function createSyncJob({
  envelope, idempotencyKey, manifestHash, tableId, deviceId, repo, idFactory, now,
}) {
  validateJobInput({ envelope, idempotencyKey, manifestHash, tableId })
  if (!ID_PATTERN.test(deviceId || '')) throw new SyncApiError('SYNC_UNAUTHORIZED')
  const existing = await repo.getByIdempotencyKey(idempotencyKey)
  if (existing) {
    if (existing.manifest_hash !== manifestHash || existing.table_id !== tableId
        || existing._device_id !== deviceId
        || contentDigest(existing.envelope) !== contentDigest(envelope)) {
      throw new SyncApiError('SYNC_CONFLICT')
    }
    if (existing.status !== 'completed') assertGenericMediaAllowed(envelope)
    await repo.validatePublication(existing)
    return existing
  }
  assertGenericMediaAllowed(envelope)
  return repo.create({
    schema_version: '1.0',
    job_id: idFactory(),
    idempotency_key: idempotencyKey,
    manifest_hash: manifestHash,
    table_id: tableId,
    _device_id: deviceId,
    session_id: envelope.session.session_id,
    status: envelope.media.length ? 'awaiting_media' : 'ready_to_complete',
    envelope,
    created_at: now,
  })
}


async function authorizeMedia({ jobId, clipId, tableId, deviceId, repo, storage }) {
  const job = await repo.get(jobId)
  if (!job || job.status === 'completed' || job.table_id !== tableId || job._device_id !== deviceId
      || !ID_PATTERN.test(deviceId || '')) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  const clip = job.envelope.media.find((item) => item.clip_id === clipId)
  if (!clip) throw new SyncApiError('SYNC_INVALID')
  assertGenericMediaAllowed(job.envelope)
  await repo.validatePublication(job)
  return storage.authorizeUpload({
    object_key: `sessions/${job.table_id}/${job._device_id}/${job.session_id}/${job.job_id}/${clip.clip_id}.mp4`,
    byte_size: clip.byte_size,
    content_sha256: clip.content_sha256,
    media_type: clip.media_type,
  })
}


async function completeSyncJob({ jobId, tableId, deviceId, repo, storage, now }) {
  const job = await repo.get(jobId)
  if (!job || job.table_id !== tableId || job._device_id !== deviceId
      || !ID_PATTERN.test(deviceId || '')) throw new SyncApiError('SYNC_INVALID')
  await repo.validatePublication(job)
  if (job.status === 'completed') return job
  assertGenericMediaAllowed(job.envelope)
  const storageFiles = {}
  for (const clip of job.envelope.media) {
    const suffix = `${job.table_id}/${job._device_id}/${job.session_id}/${job.job_id}/${clip.clip_id}.mp4`
    const objectKey = `sessions/${suffix}`
    const verification = await storage.verifyObject({
      object_key: objectKey,
      publish_object_key: `published/${suffix}`,
      byte_size: clip.byte_size,
      content_sha256: clip.content_sha256,
      media_type: clip.media_type,
    })
    const verified = verification && verification.verified === true && verification.file_id
    if (!verified) throw new SyncApiError('SYNC_PARTIAL')
    storageFiles[clip.clip_id] = {
      file_id: verification.file_id,
      staging_file_id: verification.staging_file_id || null,
    }
  }
  return repo.markCompleted(jobId, now, storageFiles)
}


function createCloudSyncRepository(db) {
  return {
    async getByIdempotencyKey(key) {
      const id = crypto.createHash('sha256').update(key, 'utf8').digest('hex')
      const result = await getOptionalDocument(db.collection('sync_idempotency').doc(id))
      if (!result.data) return null
      const job = await db.collection('sync_jobs').doc(result.data.job_id).get()
      return job.data || null
    },
    async create(job) {
      const id = crypto.createHash('sha256')
        .update(job.idempotency_key, 'utf8')
        .digest('hex')
      return db.runTransaction(async (transaction) => {
        const keyRef = transaction.collection('sync_idempotency').doc(id)
        const existing = await getOptionalDocument(keyRef)
        if (existing.data) {
          const oldJob = await transaction.collection('sync_jobs').doc(existing.data.job_id).get()
          if (!oldJob.data || oldJob.data.manifest_hash !== job.manifest_hash
              || oldJob.data._device_id !== job._device_id
              || oldJob.data.table_id !== job.table_id
              || contentDigest(oldJob.data.envelope) !== contentDigest(job.envelope)) {
            throw new SyncApiError('SYNC_CONFLICT')
          }
          await publicationState(transaction, oldJob.data)
          return oldJob.data
        }
        assertGenericMediaAllowed(job.envelope)
        await publicationState(transaction, job)
        await transaction.collection('sync_jobs').doc(job.job_id).set({ data: job })
        await keyRef.set({
          data: {
            schema_version: '1.0',
            idempotency_key_digest: id,
            job_id: job.job_id,
          },
        })
        return job
      })
    },
    async get(jobId) {
      const result = await db.collection('sync_jobs').doc(jobId).get()
      return result.data || null
    },
    async validatePublication(job) {
      return db.runTransaction(transaction => publicationState(transaction, job))
    },
    async markCompleted(jobId, completedAt, storageFiles) {
      return db.runTransaction(async (transaction) => {
        const ref = transaction.collection('sync_jobs').doc(jobId)
        const result = await ref.get()
        const job = result.data
        if (!job) throw new SyncApiError('SYNC_INVALID')
        const publication = await publicationState(transaction, job)
        if (job.status === 'completed') return job
        const completed = {
          ...withoutDocumentId(job), status: 'completed', completed_at: completedAt,
        }
        if (publication.existing) {
          // No writes to facts/ACL/media on an exact replay under another key.
          await ref.set({ data: completed })
          return completed
        }
        const liveRef = transaction.collection('field_live_sessions').doc(job.session_id)
        const liveResult = await liveRef.get()
        const session = {
          ...job.envelope.session,
          _sync_publication: publication.fence,
          sync_status: 'published',
          published_at: completedAt,
        }
        try {
          session._shot_projection_job_id = await enqueuePublishedShots(transaction, session)
        } catch (error) {
          if (error instanceof ShotProjectionError) throw new SyncApiError('SYNC_CONFLICT')
          throw error
        }
        await transaction.collection('sessions').doc(job.session_id).set({ data: session })
        for (const participant of session.participants) {
          const participantId = crypto.createHash('sha256')
            .update(`${job.session_id}:${participant.user_id}`)
            .digest('hex')
          await transaction.collection('session_participants').doc(participantId).set({
            data: {
              ...participant,
              session_id: job.session_id,
              created_at_utc: session.created_at_utc,
            },
          })
        }
        for (const clip of job.envelope.media) {
          const shot = session.shots.find((item) => item.shot_id === clip.shot_id)
          const ordinaryExpiresAt = ordinaryExpiry(session.ended_at_utc)
          await transaction.collection('clips').doc(clip.clip_id).set({
            data: {
              ...clip,
              sequence: shot.sequence,
              upload_status: 'published',
              consent_status: 'participants',
              deleted_at: null,
              table_id: job.table_id,
              _device_id: job._device_id,
              storage_file_id: typeof storageFiles[clip.clip_id] === 'string'
                ? storageFiles[clip.clip_id] : storageFiles[clip.clip_id].file_id,
              storage_objects: (() => {
                const lineage = typeof storageFiles[clip.clip_id] === 'string'
                  ? { file_id: storageFiles[clip.clip_id] } : storageFiles[clip.clip_id]
                return [
                  { kind: 'published', file_id: lineage.file_id },
                  lineage.staging_file_id
                    ? { kind: 'staging', file_id: lineage.staging_file_id } : null,
                ].filter(Boolean)
              })(),
              published_at: completedAt,
              ordinary_expires_at: ordinaryExpiresAt,
              retention_expires_at: ordinaryExpiresAt,
              retention_state: 'active',
            },
          })
        }
        if (liveResult.data) {
          await liveRef.update({ data: { status: 'ended', updated_at: completedAt } })
        }
        await ref.set({ data: completed })
        return completed
      })
    },
  }
}


const SAVED_CLIP_FIELDS = [
  'schema_version', 'publication_kind', 'clip_id', 'session_id', 'shot_id',
  'job_id', 'artifact_id', 'postgame_manifest_sha256', 'source_kind',
  'save_request_id', 'save_receipt_sha256',
  'start_ms', 'end_ms', 'duration_ms', 'content_sha256', 'byte_size',
  'content_type', 'owner_user_id', 'actor_role', 'ownership_receipt', 'ownership_receipt_sha256',
]

// The signed descriptor is an authenticated kiosk assertion. The immutable
// save receipt binds the user's large-screen selection to that assertion;
// cloud session facts independently bind session/shot/job/source/timing. A
// compromised authenticated kiosk is outside the session-facts trust boundary.

function savedClipIdentity(descriptor) {
  return `clip-${crypto.createHash('sha256').update(JSON.stringify([
    descriptor.session_id, descriptor.shot_id, descriptor.job_id, descriptor.artifact_id,
  ])).digest('hex')}`
}

function validateSavedClipInput({ descriptor, idempotencyKey, tableId, deviceId }) {
  onlyKeys(descriptor, SAVED_CLIP_FIELDS)
  if (Object.keys(descriptor).length !== SAVED_CLIP_FIELDS.length
      || descriptor.schema_version !== '2.0'
      || !validOwnership(descriptor)
      || !validSavedSelection(descriptor)
      || descriptor.publication_kind !== 'saved_postgame_clip'
      || !ID_PATTERN.test(descriptor.clip_id || '')
      || descriptor.clip_id !== savedClipIdentity(descriptor)
      || descriptor.clip_id !== idempotencyKey
      || !ID_PATTERN.test(descriptor.session_id || '')
      || !ID_PATTERN.test(descriptor.shot_id || '')
      || !ID_PATTERN.test(descriptor.job_id || '')
      || !HASH_PATTERN.test(descriptor.artifact_id || '')
      || !HASH_PATTERN.test(descriptor.postgame_manifest_sha256 || '')
      || !ID_PATTERN.test(descriptor.save_request_id || '')
      || !HASH_PATTERN.test(descriptor.save_receipt_sha256 || '')
      || descriptor.source_kind !== 'usb_live'
      || !Number.isSafeInteger(descriptor.start_ms) || descriptor.start_ms < 0
      || !Number.isSafeInteger(descriptor.end_ms) || descriptor.end_ms <= descriptor.start_ms
      || !Number.isSafeInteger(descriptor.duration_ms) || descriptor.duration_ms > 300000
      || descriptor.duration_ms !== descriptor.end_ms - descriptor.start_ms
      || !HASH_PATTERN.test(descriptor.content_sha256 || '')
      || !Number.isSafeInteger(descriptor.byte_size) || descriptor.byte_size < 1
      || descriptor.byte_size > MAX_SAVED_CLIP_BYTES
      || descriptor.content_type !== 'video/mp4'
      || !ID_PATTERN.test(tableId || '') || !ID_PATTERN.test(deviceId || '')) {
    throw new SyncApiError('SYNC_INVALID')
  }
}

function validateSavedClipJobRecord(job) {
  validateSavedClipInput({
    descriptor: job && job.descriptor,
    idempotencyKey: job && job.idempotency_key,
    tableId: job && job.table_id,
    deviceId: job && job._device_id,
  })
  if (job.clip_id !== job.descriptor.clip_id
      || job.session_id !== job.descriptor.session_id
      || job.shot_id !== job.descriptor.shot_id
      || job.content_sha256 !== job.descriptor.content_sha256
      || job.descriptor_sha256 !== contentDigest(job.descriptor)
      || job.object_key !== `saved-clips/${job.table_id}/${job._device_id}/${job.session_id}/${job.clip_id}.mp4`) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
}

function publicSavedClipJob(job) {
  return {
    job_id: job.job_id,
    status: job.status,
    clip_id: job.clip_id,
    session_id: job.session_id,
    shot_id: job.shot_id,
    content_sha256: job.content_sha256,
    publication_kind: 'saved_postgame_clip',
  }
}

function safeSavedUploadAuthorization(value) {
  try {
    const parsed = new URL(value.url)
    const headers = value.headers
    if (value.method !== 'PUT' || parsed.protocol !== 'https:' || !parsed.hostname
        || parsed.username || parsed.password || parsed.hash
        || !headers || Array.isArray(headers) || typeof headers !== 'object'
        || Object.keys(headers).length > 32
        || Object.entries(headers).some(([key, headerValue]) => typeof key !== 'string'
          || typeof headerValue !== 'string' || !key || key.length > 128
          || headerValue.length > 4096 || /[\r\n]/u.test(key) || /[\r\n]/u.test(headerValue))) {
      throw new Error('unsafe authorization')
    }
    return value
  } catch (_error) {
    throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
  }
}

async function createSavedClipJob({
  descriptor, idempotencyKey, tableId, deviceId, repo, idFactory, now,
}) {
  if (!ID_PATTERN.test(idempotencyKey || '') || !ID_PATTERN.test(tableId || '')
      || !ID_PATTERN.test(deviceId || '') || !descriptor
      || typeof descriptor !== 'object' || Array.isArray(descriptor)) {
    throw new SyncApiError('SYNC_INVALID')
  }
  const digest = contentDigest(descriptor)
  const existing = await repo.getByIdempotencyKey(idempotencyKey)
  if (existing) {
    if (existing.descriptor_sha256 !== digest || existing.table_id !== tableId
        || existing._device_id !== deviceId
        || contentDigest(existing.descriptor) !== digest) {
      throw new SyncApiError('SYNC_CONFLICT')
    }
    validateSavedClipJobRecord(existing)
    await repo.validatePublication(existing)
    return publicSavedClipJob(existing)
  }
  validateSavedClipInput({ descriptor, idempotencyKey, tableId, deviceId })
  const jobId = idFactory()
  if (!ID_PATTERN.test(jobId || '')) throw new SyncApiError('SYNC_INVALID')
  const job = await repo.create({
    schema_version: '1.0',
    job_id: jobId,
    idempotency_key: idempotencyKey,
    descriptor_sha256: digest,
    table_id: tableId,
    _device_id: deviceId,
    clip_id: descriptor.clip_id,
    session_id: descriptor.session_id,
    shot_id: descriptor.shot_id,
    content_sha256: descriptor.content_sha256,
    object_key: `saved-clips/${tableId}/${deviceId}/${descriptor.session_id}/${descriptor.clip_id}.mp4`,
    status: 'pending',
    descriptor,
    created_at: now,
  })
  return publicSavedClipJob(job)
}

function requireTrackedPorts(repo, storage, providerPort, repositoryPorts) {
  if (typeof storage?.[providerPort] !== 'function'
      || repositoryPorts.some(port => typeof repo?.[port] !== 'function')) {
    throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
  }
}

async function trackedProviderResult(call, repo, ticket) {
  // Provider exceptions can follow a write invocation. They never establish
  // no-write; persist only the fixed union and exact reserved observation.
  let result
  try { result = await call() } catch (_error) {
    result = { outcome: 'unknown', reason: 'provider_error' }
  }
  const outcome = result?.outcome
  if (!['external_issued', 'server_succeeded', 'no_write', 'unknown'].includes(outcome)) {
    result = { outcome: 'unknown', reason: 'provider_error' }
  }
  const fileId = result.file_id
  const objects = typeof fileId === 'string' && fileId.startsWith('cloud://')
    ? [{ kind: ticket.target.kind, cloud_path: ticket.target.cloud_path, file_id: fileId }] : []
  await admissionCall(() => repo.recordTrackedResult({ ticket, outcome: result.outcome, objects,
    reason: result.outcome === 'no_write' ? 'not_started' : result.outcome === 'unknown' ? 'provider_error' : null }))
  return result
}

async function authorizeSavedClip({ jobId, tableId, deviceId, uploadRequest, repo, storage }) {
  const requested = uploadRequest !== undefined
  let begun
  const reservation = ticket => ({ upload_request: { ...uploadRequest }, upload_attempt: makeManagedUploadAttempt(ticket), request_status: 'reserved' })
  if (requested) {
    if (!validManagedUploadRequest(uploadRequest)) throw new SyncApiError('SYNC_INVALID')
    if (typeof repo?.lookupTrackedUploadRequest !== 'function') throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
    const found = await admissionCall(() => repo.lookupTrackedUploadRequest({ jobId, tableId, deviceId, uploadRequest }))
    if (found) return reservation(found.ticket)
    requireTrackedPorts(repo, storage, 'authorizeRegisteredUpload', ['beginTrackedUploadRequest', 'recordTrackedResult'])
    // Begin performs its own duplicate check before mutable publication checks.
    // A lookup miss is only a hint; another caller may have issued/closed since.
    begun = await admissionCall(() => repo.beginTrackedUploadRequest({ jobId, tableId, deviceId, uploadRequest }))
    if (!begun.fresh) return reservation(begun.ticket)
  }
  const job = requested ? begun.job : await repo.get(jobId)
  if (!job || job.table_id !== tableId || job._device_id !== deviceId
      || !ID_PATTERN.test(deviceId || '') || job.status === 'completed') {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  validateSavedClipJobRecord(job)
  if (!requested) await repo.validatePublication(job)
  if (Object.hasOwn(job, 'storage_admission')) {
    if (!requested) {
      requireTrackedPorts(repo, storage, 'authorizeRegisteredUpload', ['beginTrackedUpload', 'recordTrackedResult'])
      begun = await repo.beginTrackedUpload({ jobId, tableId, deviceId })
    }
    const { ticket, job: current } = begun
    const result = await trackedProviderResult(() => storage.authorizeRegisteredUpload({
      object_key: ticket.target.cloud_path,
      byte_size: current.descriptor.byte_size,
      content_sha256: current.descriptor.content_sha256,
      media_type: current.descriptor.content_type,
    }), repo, ticket)
    if (result.outcome !== 'external_issued') throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
    const authorization = safeSavedUploadAuthorization(result.authorization)
    // This reads the actual current job and provenance after result persistence.
    // A stale pre-provider job snapshot cannot authorize credential delivery.
    await repo.validatePublication(current, { requireIncomplete: true })
    return { method: authorization.method, url: authorization.url, headers: authorization.headers,
      expires_in_seconds: authorization.expires_in_seconds, upload_attempt: makeManagedUploadAttempt(ticket),
      ...(requested ? { upload_request: { ...uploadRequest }, request_status: 'issued' } : {}) }
  }
  assertLegacySavedWriter(job)
  const authorization = safeSavedUploadAuthorization(await storage.authorizeUpload({
    object_key: job.object_key,
    byte_size: job.descriptor.byte_size,
    content_sha256: job.descriptor.content_sha256,
    media_type: job.descriptor.content_type,
  }))
  await repo.markUploading(jobId)
  return authorization
}

async function completeSavedClipJob({ jobId, tableId, deviceId, uploadAttempt, repo, storage, now }) {
  if (uploadAttempt !== undefined && !validManagedUploadAttempt(uploadAttempt)) throw new SyncApiError('SYNC_INVALID')
  const job = await repo.get(jobId)
  if (!job || job.table_id !== tableId || job._device_id !== deviceId
      || !ID_PATTERN.test(deviceId || '')) throw new SyncApiError('SYNC_CONFLICT')
  validateSavedClipJobRecord(job)
  const validatedJob = await repo.validatePublication(job)
  // Historical repository ports validate and return void. The actual repository
  // returns its current row, which is authoritative over the initial snapshot.
  const currentJob = validatedJob === undefined ? job : validatedJob
  if (currentJob.status === 'completed') return publicSavedClipJob(currentJob)
  if (Object.hasOwn(job, 'storage_admission')) {
    // Missing capability deliberately precedes requiring an absent attempt:
    // existing pending deployments retain retryable 503 without writes.
    requireTrackedPorts(repo, storage, 'copyRegisteredObject', ['beginTrackedCopy', 'recordTrackedResult', 'publishTrackedCopy'])
    // The repository checks completion before validating the attempt in the
    // same current-job transaction, so a concurrent winner remains read-only.
    const begun = await repo.beginTrackedCopy({ jobId, tableId, deviceId, uploadAttempt })
    if (begun.completed === true) return publicSavedClipJob(begun.job)
    const { ticket, input, job: current } = begun
    const result = await trackedProviderResult(() => storage.copyRegisteredObject({
      staging_file_id: input.file_id,
      publish_object_key: ticket.target.cloud_path,
      byte_size: current.descriptor.byte_size,
      content_sha256: current.descriptor.content_sha256,
      media_type: current.descriptor.content_type,
    }), repo, ticket)
    if (result.outcome !== 'server_succeeded') {
      throw new SyncApiError(result.outcome === 'no_write' && result.permanent === true
        ? 'SYNC_CONFLICT' : 'SYNC_STORAGE_UNAVAILABLE')
    }
    return publicSavedClipJob(await repo.publishTrackedCopy({ jobId, tableId, deviceId, copyTicket: ticket, completedAt: now }))
  }
  assertLegacySavedWriter(job)
  const verification = await storage.verifyObject({
    object_key: job.object_key,
    publish_object_key: `published/${job.object_key}`,
    byte_size: job.descriptor.byte_size,
    content_sha256: job.descriptor.content_sha256,
    media_type: job.descriptor.content_type,
  })
  if (verification && verification.permanent === true) throw new SyncApiError('SYNC_CONFLICT')
  if (!verification || verification.verified !== true
      || typeof verification.file_id !== 'string' || !verification.file_id.startsWith('cloud://')
      || (verification.staging_file_id != null
        && (typeof verification.staging_file_id !== 'string' || !verification.staging_file_id.startsWith('cloud://')))) {
    throw new SyncApiError('SYNC_PARTIAL')
  }
  return publicSavedClipJob(await repo.markCompleted(jobId, now, {
    file_id: verification.file_id, staging_file_id: verification.staging_file_id || null,
  }))
}

async function discoverManagedHistory(db, job) {
  if (!job || !isManagedSessionId(job.session_id)) return null
  validateSavedClipJobRecord(job)
  const receipt = job.descriptor.ownership_receipt
  const result = await admissionCall(() => db.collection('session_role_binding_history').where({
    session_id: job.session_id, role: receipt.actor_role,
    resulting_revision: db.command.lte(receipt.match_revision),
  }).orderBy('resulting_revision', 'desc').limit(1).get())
  const candidate = result.data && result.data[0]
  // Discovery is only an ID hint. Neither query absence nor a returned row proves ownership.
  return candidate && Number.isSafeInteger(candidate.resulting_revision)
    ? documentId(`${job.session_id}:${receipt.actor_role}:${candidate.resulting_revision}`) : null
}

async function savedClipPublicationState(transaction, job, { db, createProjection = false, candidateHistoryId = null } = {}) {
  validateSavedClipJobRecord(job)
  if (await getOwnerClipTombstone(transaction, job.clip_id) !== null) throw new SyncApiError('SYNC_CONFLICT')
  const sessionResult = await getOptionalDocument(
    transaction.collection('sessions').doc(job.session_id),
  )
  const session = sessionResult.data
  if (!session || session.status !== 'ended' || session.sync_status !== 'published') {
    throw new SyncApiError('SYNC_PARTIAL')
  }
  const publication = session._sync_publication
  const participants = Array.isArray(session.participants) ? session.participants : []
  const participantUsers = new Set(participants.map(item => item && item.user_id))
  const participantRoles = new Set(participants.map(item => item && item.role))
  if (session.session_id !== job.session_id || !publication || publication.table_id !== job.table_id
      || publication.device_id !== job._device_id
      || session.evidence_source_kind !== 'usb_live'
      || session.postgame_manifest_sha256 !== job.descriptor.postgame_manifest_sha256
      || !Array.isArray(session.participants) || !participants.length
      || participants.length > 2
      || participantUsers.size !== participants.length
      || participantRoles.size !== participants.length
      || participants.some(participant => !participant
        || !ID_PATTERN.test(participant.user_id || '')
        || !['red', 'blue'].includes(participant.role))) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  const descriptor = job.descriptor
  const shot = Array.isArray(session.shots)
    ? session.shots.find(item => item && item.shot_id === descriptor.shot_id) : null
  if (!shot || !Number.isSafeInteger(shot.sequence) || shot.sequence < 1
      || !Number.isSafeInteger(shot.started_ms) || !Number.isSafeInteger(shot.settled_ms)
      || shot.settled_ms < shot.started_ms
      || descriptor.start_ms > shot.started_ms || descriptor.end_ms < shot.settled_ms
      || shot.live_job_id !== descriptor.job_id
      || descriptor.clip_id !== savedClipIdentity(descriptor)) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  const receipt = descriptor.ownership_receipt
  if (receipt.sequence !== shot.sequence || receipt.source_started_ms !== shot.started_ms
      || receipt.actor_role !== shot.actor_role || receipt.source_artifact_id !== session.artifact_id) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  if (isManagedSessionId(job.session_id)) {
    if (!await verifyManagedShotOwnership(transaction, { receipt, job, candidateHistoryId })) {
      throw new SyncApiError('SYNC_CONFLICT')
    }
  } else {
    const historyResult = await transaction.collection('session_role_binding_history').where({
      session_id: job.session_id, role: receipt.actor_role,
      resulting_revision: db.command.lte(receipt.match_revision),
    }).orderBy('resulting_revision', 'desc').limit(1).get()
    const history = historyResult.data && historyResult.data[0]
    if (!history) throw new SyncApiError('SYNC_PARTIAL')
    if (history.session_id !== job.session_id || history.role !== receipt.actor_role
        || !Number.isSafeInteger(history.resulting_revision) || history.resulting_revision < 0
        || history.resulting_revision > receipt.match_revision
        || history.user_id !== receipt.actor_user_id || history.table_id !== job.table_id
        || history.device_id !== job._device_id || !ID_PATTERN.test(history.command_id || '')) {
      throw new SyncApiError('SYNC_CONFLICT')
    }
    const keyedHistory = await getOptionalDocument(transaction.collection('session_role_binding_history')
      .doc(documentId(`${job.session_id}:${history.role}:${history.resulting_revision}`)))
    if (!keyedHistory.data || ownershipDigest(withoutDocumentId(keyedHistory.data))
        !== ownershipDigest(withoutDocumentId(history))) throw new SyncApiError('SYNC_CONFLICT')
    const commandResult = await getOptionalDocument(transaction.collection('field_session_commands').doc(history.command_id))
    const command = commandResult.data
    if (!command) throw new SyncApiError('SYNC_PARTIAL')
    if (command.command_id !== history.command_id || command.status !== 'applied'
        || command.session_id !== job.session_id || command.table_id !== job.table_id
        || command.claimed_by_device_id !== job._device_id
        || command.resulting_revision !== history.resulting_revision
        || command.command_type !== 'claim_role' || command.actor_user_id !== history.user_id
        || command.validated_payload?.role !== history.role
        || (command.applied_role != null && command.applied_role !== history.role)) {
      throw new SyncApiError('SYNC_CONFLICT')
    }
  }
  const existing = await getOptionalDocument(
    transaction.collection('clips').doc(descriptor.clip_id),
  )
  const projectionRef = transaction.collection('field_shot_ownership')
    .doc(documentId(`${job.session_id}:${job.shot_id}`))
  const projection = await getOptionalDocument(projectionRef)
  if (projection.data) {
    if (!matchesProjection({ ...descriptor, table_id: job.table_id, _device_id: job._device_id }, projection.data)) {
      throw new SyncApiError('SYNC_CONFLICT')
    }
  } else if (createProjection && !existing.data) {
    await projectionRef.set({ data: ownershipProjection(descriptor, job.table_id, job._device_id) })
  } else throw new SyncApiError('SYNC_CONFLICT')
  if (existing.data && (existing.data.schema_version !== '2.0'
      || existing.data.privacy_policy !== 'shot_owner_only_v1'
      || !matchesProjection(existing.data, projection.data || ownershipProjection(descriptor, job.table_id, job._device_id))
      || existing.data.save_request_id !== descriptor.save_request_id
      || existing.data.save_receipt_sha256 !== descriptor.save_receipt_sha256
      || existing.data.publication_kind !== 'saved_postgame_clip'
      || existing.data.session_id !== descriptor.session_id
      || existing.data.shot_id !== descriptor.shot_id
      || existing.data.job_id !== descriptor.job_id
      || existing.data.artifact_id !== descriptor.artifact_id
      || existing.data.postgame_manifest_sha256 !== descriptor.postgame_manifest_sha256
      || existing.data.source_kind !== descriptor.source_kind
      || existing.data.start_ms !== descriptor.start_ms
      || existing.data.end_ms !== descriptor.end_ms
      || existing.data.duration_ms !== descriptor.duration_ms
      || existing.data.content_sha256 !== descriptor.content_sha256
      || existing.data.byte_size !== descriptor.byte_size
      || existing.data.content_type !== descriptor.content_type
      || existing.data.media_type !== descriptor.content_type
      || existing.data.table_id !== job.table_id
      || existing.data._device_id !== job._device_id
      || existing.data.upload_status !== 'published'
      || typeof existing.data.storage_file_id !== 'string'
      || !existing.data.storage_file_id.startsWith('cloud://'))) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  if (job.status === 'completed' && !existing.data) throw new SyncApiError('SYNC_CONFLICT')
  return { session, shot, existingClip: existing.data }
}

function assertLegacySavedWriter(job) {
  // Admission metadata never grants permission to use legacy object_key.
  if (Object.hasOwn(job, 'storage_admission')) throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
}

async function admissionCall(work) {
  try { return await work() } catch (error) {
    if (error instanceof SyncApiError) throw error
    throw new SyncApiError(['MEDIA_ADMISSION_CONFLICT', 'STORAGE_AUTHORITY_CONFLICT'].includes(error?.code)
      ? 'SYNC_CONFLICT' : 'SYNC_STORAGE_UNAVAILABLE')
  }
}

function createCloudSavedClipRepository(db) {
  const authority = createCloudMediaStorageAuthority(db)
  const idempotencyDigest = value => crypto.createHash('sha256').update(value).digest('hex')
  async function storageContext(transaction, job, { fresh = false } = {}) {
    return admissionCall(async () => {
      if (Object.hasOwn(job, 'storage_admission')) {
        const identity = makeStorageIdentity(job.descriptor, { tableId: job.table_id, deviceId: job._device_id, kind: 'saved_private' })
        const marker = job.storage_admission
        if (!validStorageAdmission(marker, identity, marker?.epoch)) throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
        const root = await authority.getAdmission(transaction, identity)
        if (!validStorageAdmission(marker, identity, root.epoch)) throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
        return { identity, root }
      }
      if (!isManagedSessionId(job.session_id)) return null
      const birth = await readManagedMediaBirth(transaction, {
        sessionId: job.session_id, tableId: job.table_id, deviceId: job._device_id,
      })
      if (birth === null) return null
      if (!fresh) throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      return { identity: makeStorageIdentity(job.descriptor, { tableId: job.table_id, deviceId: job._device_id, kind: 'saved_private' }) }
    })
  }
  async function checkedPublication(transaction, job, options) {
    await storageContext(transaction, job)
    return savedClipPublicationState(transaction, job, { db, ...options })
  }
  async function writeSavedPublication(transaction, job, state, storageFile, completedAt, storageObjects = null) {
    if (!state.existingClip) {
      const descriptor = job.descriptor
      await transaction.collection('clips').doc(descriptor.clip_id).set({ data: {
        ...descriptor,
        media_type: descriptor.content_type,
        sequence: state.shot.sequence,
        upload_status: 'published',
        consent_status: 'owner',
        privacy_policy: 'shot_owner_only_v1',
        deleted_at: null,
        table_id: job.table_id,
        _device_id: job._device_id,
        storage_file_id: storageFile.file_id,
        storage_objects: storageObjects || [
          { kind: 'published', file_id: storageFile.file_id },
          storageFile.staging_file_id
            ? { kind: 'staging', file_id: storageFile.staging_file_id } : null,
        ].filter(Boolean),
        published_at: completedAt,
        retention_policy: SAVED_UNTIL_DELETED_POLICY,
        ordinary_expires_at: null,
        retention_expires_at: null,
        retention_state: 'active',
      } })
    }
    const completed = { ...withoutDocumentId(job), status: 'completed', completed_at: completedAt }
    await transaction.collection('saved_clip_sync_jobs').doc(job.job_id).set({ data: completed })
    return completed
  }
  async function trackedTransaction({ jobId, tableId, deviceId }, work, { beforeRead, rejectUntracked = false } = {}) {
    let candidateHistoryId; let discoveryError
    try {
      const discovered = (await getOptionalDocument(db.collection('saved_clip_sync_jobs').doc(jobId))).data
      candidateHistoryId = await discoverManagedHistory(db, discovered)
    } catch (error) {
      if (!beforeRead) throw error
      // History discovery is a mutable hint, never a prerequisite for permanent
      // reservation recovery. Fresh publication still requires it to succeed.
      discoveryError = error
    }
    return admissionCall(() => db.runTransaction(async transaction => {
      if (beforeRead) {
        const recovered = await beforeRead(transaction)
        if (recovered !== null) return recovered
      }
      if (discoveryError) throw discoveryError
      const job = (await getOptionalDocument(transaction.collection('saved_clip_sync_jobs').doc(jobId))).data
      if (!job || job.job_id !== jobId || job.table_id !== tableId || job._device_id !== deviceId
          || !ID_PATTERN.test(deviceId || '') || !['pending', 'uploading', 'completed'].includes(job.status)) {
        throw new SyncApiError('SYNC_CONFLICT')
      }
      validateSavedClipJobRecord(job)
      if (!Object.hasOwn(job, 'storage_admission')) {
        if (rejectUntracked && await storageContext(transaction, job) === null) throw new SyncApiError('SYNC_INVALID')
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      const context = await storageContext(transaction, job)
      const state = await savedClipPublicationState(transaction, job, { db, candidateHistoryId })
      return work(transaction, job, context, state)
    }))
  }
  function validateRequestScope({ jobId, tableId, deviceId, uploadRequest }) {
    if (!validManagedUploadRequest(uploadRequest)) throw new SyncApiError('SYNC_INVALID')
    if (![jobId, tableId, deviceId].every(value => typeof value === 'string' && ID_PATTERN.test(value))) throw new SyncApiError('SYNC_CONFLICT')
  }
  const requestRef = (transaction, scope) => transaction.collection('managed_upload_requests').doc(managedUploadRequestId(scope))
  async function readRequestBinding(transaction, scope) {
    const record = (await getOptionalDocument(requestRef(transaction, scope))).data
    if (record === null || record === undefined) return null
    if (!validManagedUploadRequestBinding(record)) throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
    if (record.request_id !== scope.uploadRequest.request_id || record.job_id !== scope.jobId
        || record.table_id !== scope.tableId || record.device_id !== scope.deviceId) throw new SyncApiError('SYNC_CONFLICT')
    return record
  }
  async function lookupRequest(transaction, scope) {
    const record = await readRequestBinding(transaction, scope)
    if (record === null) return null
    const ticket = await authority.getUploadOperation(transaction, { identity: record.identity, ingestJobId: record.job_id,
      uploadAttempt: makeManagedUploadAttempt(record.ticket) })
    if (ownershipDigest(ticket) !== ownershipDigest(record.ticket) || ticket.identity_sha256 !== record.identity_sha256) throw new SyncApiError('SYNC_CONFLICT')
    return { ticket }
  }
  return {
    async getByIdempotencyKey(key) {
      const row = await getOptionalDocument(
        db.collection('saved_clip_sync_idempotency').doc(idempotencyDigest(key)),
      )
      if (!row.data) return null
      return (await getOptionalDocument(
        db.collection('saved_clip_sync_jobs').doc(row.data.job_id),
      )).data || null
    },
    async create(job) {
      const candidateHistoryId = await discoverManagedHistory(db, job)
      return db.runTransaction(async transaction => {
        const keyRef = transaction.collection('saved_clip_sync_idempotency')
          .doc(idempotencyDigest(job.idempotency_key))
        const existingKey = await getOptionalDocument(keyRef)
        if (existingKey.data) {
          const old = await getOptionalDocument(
            transaction.collection('saved_clip_sync_jobs').doc(existingKey.data.job_id),
          )
          if (!old.data || old.data.descriptor_sha256 !== job.descriptor_sha256
              || old.data.table_id !== job.table_id || old.data._device_id !== job._device_id) {
            throw new SyncApiError('SYNC_CONFLICT')
          }
          await checkedPublication(transaction, old.data, { candidateHistoryId })
          return old.data
        }
        // A new job's marker is exclusively derived here. Direct repository
        // callers cannot supply it, even after idempotency compaction.
        if (Object.hasOwn(job, 'storage_admission')) throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
        const context = await storageContext(transaction, job, { fresh: true })
        const state = await savedClipPublicationState(transaction, job, { db, createProjection: true, candidateHistoryId })
        let candidate = state.existingClip
          ? { ...job, status: 'completed', completed_at: state.existingClip.published_at }
          : job
        if (context) {
          const root = await admissionCall(() => authority.enroll(transaction, context.identity))
          if (root.status === 'legacy_untracked') throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
          const marker = await admissionCall(() => makeStorageAdmission(context.identity, root))
          candidate = { ...candidate, storage_admission: marker }
        }
        await transaction.collection('saved_clip_sync_jobs').doc(job.job_id)
          .set({ data: candidate })
        await keyRef.set({ data: {
          schema_version: '1.0',
          idempotency_key_digest: idempotencyDigest(job.idempotency_key),
          job_id: job.job_id,
        } })
        return candidate
      })
    },
    async get(jobId) {
      return (await getOptionalDocument(db.collection('saved_clip_sync_jobs').doc(jobId))).data || null
    },
    async validatePublication(job, { requireIncomplete = false } = {}) {
      const candidateHistoryId = await discoverManagedHistory(db, job)
      return admissionCall(() => db.runTransaction(async transaction => {
        const current = (await getOptionalDocument(transaction.collection('saved_clip_sync_jobs').doc(job.job_id))).data
        if (!current || current.job_id !== job.job_id || current.table_id !== job.table_id
            || current._device_id !== job._device_id || current.descriptor_sha256 !== job.descriptor_sha256
            || contentDigest(current.descriptor) !== job.descriptor_sha256
            || contentDigest(current.storage_admission ?? null) !== contentDigest(job.storage_admission ?? null)
            || (requireIncomplete && !['pending', 'uploading'].includes(current.status))) {
          throw new SyncApiError('SYNC_CONFLICT')
        }
        await checkedPublication(transaction, current, { candidateHistoryId })
        return current
      }))
    },
    async recordTrackedUploadReceipt({ jobId, tableId, deviceId, receipt }) {
      if (!validUploadReceipt(receipt)) throw new SyncApiError('SYNC_INVALID')
      const scope = { jobId, tableId, deviceId, uploadRequest: receipt.upload_request }
      validateRequestScope(scope)
      return admissionCall(() => db.runTransaction(async transaction => {
        const requestBinding = await readRequestBinding(transaction, scope)
        if (requestBinding === null) throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
        return authority.recordUploadReceipt(transaction, { requestBinding, receipt })
      }))
    },
    async lookupTrackedUploadRequest(scope) {
      validateRequestScope(scope)
      return admissionCall(() => db.runTransaction(transaction => lookupRequest(transaction, scope)))
    },
    async beginTrackedUploadRequest(scope) {
      validateRequestScope(scope)
      return trackedTransaction(scope, async (transaction, job, context) => {
        if (job.status === 'completed') throw new SyncApiError('SYNC_CONFLICT')
        const ticket = await authority.beginOperation(transaction, { identity: context.identity,
          ingestJobId: job.job_id, kind: 'upload_authorization' })
        await requestRef(transaction, scope).set({ data: {
          contract: MANAGED_UPLOAD_REQUEST_BINDING_CONTRACT, request_id: scope.uploadRequest.request_id,
          table_id: scope.tableId, device_id: scope.deviceId, job_id: job.job_id,
          descriptor_sha256: job.descriptor_sha256, identity: context.identity,
          identity_sha256: ticket.identity_sha256, ticket,
        } })
        const current = { ...withoutDocumentId(job), status: 'uploading' }
        await transaction.collection('saved_clip_sync_jobs').doc(job.job_id).set({ data: current })
        return { fresh: true, ticket, job: current }
      }, { rejectUntracked: true, beforeRead: async transaction => {
        const existing = await lookupRequest(transaction, scope)
        return existing ? { fresh: false, ticket: existing.ticket } : null
      } })
    },
    async beginTrackedUpload(scope) {
      return trackedTransaction(scope, async (transaction, job, context) => {
        if (job.status === 'completed') throw new SyncApiError('SYNC_CONFLICT')
        const ticket = await authority.beginOperation(transaction, { identity: context.identity,
          ingestJobId: job.job_id, kind: 'upload_authorization' })
        const current = { ...withoutDocumentId(job), status: 'uploading' }
        await transaction.collection('saved_clip_sync_jobs').doc(job.job_id).set({ data: current })
        return { ticket, job: current }
      })
    },
    async beginTrackedCopy({ uploadAttempt, ...scope }) {
      if (uploadAttempt !== undefined && !validManagedUploadAttempt(uploadAttempt)) throw new SyncApiError('SYNC_INVALID')
      return trackedTransaction(scope, async (transaction, job, context) => {
        // Documented no-provider race/replay branch; no attempt is required and
        // no operation is allocated once actual publication is complete.
        if (job.status === 'completed') return { completed: true, job }
        if (!validManagedUploadAttempt(uploadAttempt)) throw new SyncApiError('SYNC_INVALID')
        const begun = await authority.beginCopyForUpload(transaction, { identity: context.identity,
          ingestJobId: job.job_id, uploadAttempt })
        return { ...begun, job }
      })
    },
    async recordTrackedResult(result) {
      // Intentionally no admission/open-publication gate: reserved operations
      // retain observations even after the owner has closed the authority.
      return admissionCall(() => authority.recordOperationResult(result))
    },
    async publishTrackedCopy({ copyTicket, completedAt, ...scope }) {
      return trackedTransaction(scope, async (transaction, job, context, state) => {
        if (job.status === 'completed') return job
        const storageFile = await authority.registerCopyReferences(transaction, {
          identity: context.identity, ingestJobId: job.job_id, copyTicket,
        })
        return writeSavedPublication(transaction, job, state, storageFile, completedAt, storageFile.objects)
      })
    },
    async markUploading(jobId) {
      const discoveredJob = (await getOptionalDocument(db.collection('saved_clip_sync_jobs').doc(jobId))).data
      const candidateHistoryId = await discoverManagedHistory(db, discoveredJob)
      return db.runTransaction(async transaction => {
        const ref = transaction.collection('saved_clip_sync_jobs').doc(jobId)
        const row = await getOptionalDocument(ref)
        if (!row.data) throw new SyncApiError('SYNC_CONFLICT')
        await checkedPublication(transaction, row.data, { candidateHistoryId })
        if (row.data.status === 'completed') return row.data
        assertLegacySavedWriter(row.data)
        const next = { ...withoutDocumentId(row.data), status: 'uploading' }
        await ref.set({ data: next })
        return next
      })
    },
    async markCompleted(jobId, completedAt, storageFile) {
      const discoveredJob = (await getOptionalDocument(db.collection('saved_clip_sync_jobs').doc(jobId))).data
      const candidateHistoryId = await discoverManagedHistory(db, discoveredJob)
      return db.runTransaction(async transaction => {
        const ref = transaction.collection('saved_clip_sync_jobs').doc(jobId)
        const row = await getOptionalDocument(ref)
        const job = row.data
        if (!job) throw new SyncApiError('SYNC_CONFLICT')
        const state = await checkedPublication(transaction, job, { candidateHistoryId })
        if (job.status === 'completed') return job
        assertLegacySavedWriter(job)
        return writeSavedPublication(transaction, job, state, storageFile, completedAt)
      })
    },
  }
}


module.exports = {
  authorizeMedia,
  authorizeSavedClip,
  completeSavedClipJob,
  completeSyncJob,
  createCloudSavedClipRepository,
  createCloudSyncRepository,
  createSavedClipJob,
  createSyncJob,
  publicSavedClipJob,
  validateSavedClipInput,
}
