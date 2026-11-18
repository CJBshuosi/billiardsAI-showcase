const crypto = require('crypto')
const { sessionId, now } = require('../helpers/managed-media-admission-fixture')
const { digest, documentId } = require('../../cloudfunctions/shared/shot-ownership')
const { createSavedClipJob, createCloudSavedClipRepository, authorizeSavedClip, completeSavedClipJob,
  createSyncJob, createCloudSyncRepository, authorizeMedia, completeSyncJob } = require('../../cloudfunctions/sync-api/lib/ingest')

const { setup } = require('../helpers/managed-saved-flow-fixture')

function last(db) { return db.transactionAttempts.at(-1) }
function noWrites(db) { expect(last(db).calls.every(call => call.method === 'get')).toBe(true) }
const storage = () => ({ authorizeUpload: jest.fn(), verifyObject: jest.fn() })

test('actual marked saved publication creates private admission atomically and exact replay is immutable', async () => {
  const s = await setup(); const response = await createSavedClipJob(s.input)
  const raw = s.db.rows.saved_clip_sync_jobs[response.job_id]
  expect(raw.storage_admission).toBeDefined()
  const root = s.db.rows.clip_storage_authorities[s.descriptor.clip_id]
  expect(raw.storage_admission).toEqual({ contract: 'managed_clip_admission_v1', storage_contract: 'clip_storage_v2',
    identity_sha256: digest(s.identity), epoch: root.epoch })
  expect(response.storage_admission).toBeUndefined()
  expect(raw.descriptor).toEqual(s.descriptor)
  expect(last(s.db).calls.filter(c => c.method === 'set').map(c => c.collection).sort())
    .toEqual(['clip_storage_authorities', 'field_shot_ownership', 'saved_clip_sync_idempotency', 'saved_clip_sync_jobs'])
  const before = structuredClone(s.db.rows)
  expect(await createSavedClipJob(s.input)).toEqual(response); noWrites(s.db)
  expect(s.db.rows).toEqual(before)
  expect(last(s.db).calls.filter(c => c.collection === 'media_storage_operations')).toHaveLength(0)
})

test('tracked jobs refuse all legacy provider ports and direct mutations with retryable pending state', async () => {
  const s = await setup(); const job = await createSavedClipJob(s.input); const provider = storage()
  const args = { ...s.scope, jobId: job.job_id, repo: s.repo, storage: provider, now }
  for (const operation of [() => authorizeSavedClip(args), () => completeSavedClipJob(args),
    () => s.repo.markUploading(job.job_id), () => s.repo.markCompleted(job.job_id, now, { file_id: 'cloud://forbidden' })]) {
    await expect(operation()).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  }
  expect(provider.authorizeUpload).not.toHaveBeenCalled(); expect(provider.verifyObject).not.toHaveBeenCalled()
  expect(s.db.rows.saved_clip_sync_jobs[job.job_id].status).toBe('pending')
  expect(Object.keys(s.db.rows.clips || {})).toHaveLength(0)
})

test.each(['missing', 'unmarked', 'extra', 'epoch', 'foreign-root', 'missing-root', 'closed'])('tracked %s context never falls back', async state => {
  const s = await setup(); const job = await createSavedClipJob(s.input)
  const raw = s.db.rows.saved_clip_sync_jobs[job.job_id]
  if (state === 'missing') delete s.db.rows.managed_session_births[sessionId]
  if (state === 'unmarked') delete s.db.rows.managed_session_births[sessionId].media_admission_contract
  if (state === 'extra') raw.storage_admission.token = 'secret'
  if (state === 'epoch') raw.storage_admission.epoch = crypto.randomUUID()
  if (state === 'foreign-root') s.db.rows.clip_storage_authorities[job.clip_id].identity.byte_size++
  if (state === 'missing-root') delete s.db.rows.clip_storage_authorities[job.clip_id]
  if (state === 'closed') await s.db.runTransaction(tx => s.authority.closeForDeletion(tx, s.identity))
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: state === 'closed' ? 'SYNC_CONFLICT' : 'SYNC_STORAGE_UNAVAILABLE' })
  noWrites(s.db)
})

test('marked birth refuses an old unmarked job without promoting it', async () => {
  const s = await setup(); const job = await createSavedClipJob(s.input)
  delete s.db.rows.saved_clip_sync_jobs[job.job_id].storage_admission
  const before = structuredClone(s.db.rows)
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before)
})

test.each([{ token: 'secret' }, { media_admission_contract: 'unknown' }, { table_id: 'foreign' }])('corrupt or foreign birth %p refuses without legacy fallback', async change => {
  const s = await setup(); Object.assign(s.db.rows.managed_session_births[sessionId], change)
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: change.table_id ? 'SYNC_CONFLICT' : 'SYNC_STORAGE_UNAVAILABLE' })
  expect(Object.keys(s.db.rows.saved_clip_sync_jobs || {})).toHaveLength(0)
})

test('device descriptor cannot inject the private server admission marker', async () => {
  const s = await setup()
  await expect(createSavedClipJob({ ...s.input, descriptor: { ...s.descriptor, storage_admission: {} } }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
})

test('strict unmarked managed saved receipt retains explicit legacy behavior', async () => {
  const s = await setup({ marked: false }); const job = await createSavedClipJob(s.input)
  expect(s.db.rows.saved_clip_sync_jobs[job.job_id]).not.toHaveProperty('storage_admission')
  expect(Object.keys(s.db.rows.clip_storage_authorities || {})).toHaveLength(0)
  await expect(authorizeSavedClip({ ...s.scope, jobId: job.job_id, repo: s.repo,
    storage: { authorizeUpload: async () => ({ method: 'PUT', url: 'https://upload.invalid', headers: {} }) },
  })).resolves.toMatchObject({ method: 'PUT' })
})

test('concurrent saved creates converge and a subsequent owner tombstone fences every retry', async () => {
  const s = await setup()
  const jobs = await Promise.all([createSavedClipJob(s.input), createSavedClipJob({ ...s.input, idFactory: () => 'other-job' })])
  expect(jobs[0]).toEqual(jobs[1]); expect(Object.keys(s.db.rows.clip_storage_authorities)).toHaveLength(1)
  await s.db.collection('owner_clip_tombstones').doc(s.descriptor.clip_id).set({ data: {} })
  const before = structuredClone(s.db.rows)
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before)
})

test.each([true, false])('owner closure and fresh create serialize without late revival, closeFirst=%s', async closeFirst => {
  let armed = false; let signalCreateReady; let releaseCreate
  const createReady = new Promise(resolve => { signalCreateReady = resolve })
  const createCommitGate = new Promise(resolve => { releaseCreate = resolve })
  const order = []
  let rootAtCreateCommit = null; let rootAtCloseEntry = null
  const s = await setup({ dbOptions: { async beforeCommit({ attempt, rows }) {
    if (!armed) return
    if (attempt.calls.some(call => call.collection === 'saved_clip_sync_jobs' && call.method === 'set')) {
      order.push('create-ready')
      rootAtCreateCommit = structuredClone(rows.clip_storage_authorities[s.descriptor.clip_id])
      signalCreateReady()
      await createCommitGate
    }
    if (attempt.calls.some(call => call.collection === 'owner_clip_tombstones' && call.method === 'set')) {
      order.push('close-ready')
    }
  } } })
  armed = true
  const close = () => s.db.runTransaction(async tx => {
    order.push('close-enter')
    rootAtCloseEntry = structuredClone(s.db.rows.clip_storage_authorities?.[s.descriptor.clip_id] ?? null)
    const result = await s.authority.closeForDeletion(tx, s.identity)
    await tx.collection('owner_clip_tombstones').doc(s.descriptor.clip_id).set({ data: { clip_id: s.descriptor.clip_id } })
    return result
  })
  let created; let closed; let closeEnteredWhileCreateHeld = false
  if (closeFirst) {
    closed = close()
    created = createSavedClipJob(s.input)
  } else {
    created = createSavedClipJob(s.input)
    // Reach the actual publication transaction's commit boundary before queueing
    // closure: merely invoking the async creator first does not establish order.
    await Promise.race([createReady, created.then(() => { throw new Error('create bypassed commit barrier') })])
    closed = close()
    closeEnteredWhileCreateHeld = order.includes('close-enter')
  }
  releaseCreate()
  const [createResult, closeResult] = await Promise.allSettled([created, closed])
  expect(closeResult.status).toBe('fulfilled')
  const root = s.db.rows.clip_storage_authorities?.[s.descriptor.clip_id]
  if (closeFirst) {
    expect(order).toEqual(['close-enter', 'close-ready'])
    expect(createResult).toMatchObject({ status: 'rejected', reason: { code: 'SYNC_CONFLICT' } })
    expect(closeResult.value).toMatchObject({ status: 'legacy_untracked', reasons: ['missing_authority'] })
    expect(rootAtCloseEntry).toBeNull(); expect(rootAtCreateCommit).toBeNull()
    expect(root).toBeUndefined()
    expect(Object.keys(s.db.rows.saved_clip_sync_jobs || {})).toHaveLength(0)
  } else {
    expect(order).toEqual(['create-ready', 'close-enter', 'close-ready'])
    expect(closeEnteredWhileCreateHeld).toBe(false)
    expect(createResult).toMatchObject({ status: 'fulfilled', value: { job_id: 'saved-job', status: 'pending' } })
    expect(rootAtCreateCommit).toMatchObject({ admission: 'open', high_water: 0 })
    expect(rootAtCloseEntry).toEqual(rootAtCreateCommit)
    expect(closeResult.value).toMatchObject({ status: 'tracked_closed', epoch: rootAtCreateCommit.epoch })
    expect(root).toEqual({ ...rootAtCreateCommit, admission: 'closed', revision: rootAtCreateCommit.revision + 1 })
    expect(s.db.rows.saved_clip_sync_jobs['saved-job'].storage_admission.epoch).toBe(root.epoch)
  }
  const before = structuredClone(s.db.rows)
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before)
  for (const attempt of s.db.transactionAttempts) expect(attempt.count).toBeLessThanOrEqual(90)
})

test('a valid foreign root is a conflict rather than retryable corruption', async () => {
  const s = await setup(); const job = await createSavedClipJob(s.input)
  const root = s.db.rows.clip_storage_authorities[job.clip_id]
  root.identity.byte_size++; root.identity_sha256 = digest(root.identity)
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' }); noWrites(s.db)
})

test('unavailable private provenance is a fixed retryable error without raw DB details', async () => {
  const s = await setup()
  const original = s.db.runTransaction.bind(s.db)
  s.db.runTransaction = work => original(tx => work({ collection(name) {
    if (name !== 'managed_session_births') return tx.collection(name)
    return { doc() { return { async get() { throw new Error('private backend credential detail') } } } }
  } }))
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE', message: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(Object.keys(s.db.rows.saved_clip_sync_jobs || {})).toHaveLength(0)
})

test('completed tracked historical replay stays read-only after current admission and publication checks', async () => {
  const s = await setup(); const job = await createSavedClipJob(s.input)
  // Explicit historical state fixture; the pending production writer cannot
  // manufacture this state or publish an object in this increment.
  s.db.rows.saved_clip_sync_jobs[job.job_id].status = 'completed'
  s.db.rows.clips[job.clip_id] = { ...s.descriptor, table_id: 'table-1', _device_id: 'device-1',
    privacy_policy: 'shot_owner_only_v1', media_type: 'video/mp4', upload_status: 'published', storage_file_id: 'cloud://historical/clip',
  }
  const before = structuredClone(s.db.rows); const provider = storage()
  await expect(completeSavedClipJob({ ...s.scope, jobId: job.job_id, repo: s.repo, storage: provider, now }))
    .resolves.toMatchObject({ status: 'completed' })
  expect(await createSavedClipJob(s.input)).toMatchObject({ status: 'completed' })
  expect(s.db.rows).toEqual(before)
  expect(provider.authorizeUpload).not.toHaveBeenCalled(); expect(provider.verifyObject).not.toHaveBeenCalled()
})

test('actual composed N64/A64 compaction replay validates full inventory within 90 document calls', async () => {
  let retried = false
  const s = await setup({ dbOptions: { retryConflicts: 1, beforeCommit({ rows }) {
    if (!retried && rows.saved_clip_sync_jobs?.['recreated-job']) {
      retried = true; throw Object.assign(new Error('synthetic callback conflict'), { code: 'SYNTHETIC_CONFLICT' })
    }
  } } }); const job = await createSavedClipJob(s.input)
  const freshCount = last(s.db).count
  const epoch = s.db.rows.clip_storage_authorities[job.clip_id].epoch
  for (let index = 0; index < 64; index++) {
    const ticket = await s.db.runTransaction(tx => s.authority.beginOperation(tx, {
      identity: s.identity, ingestJobId: job.job_id, kind: 'upload_authorization',
    }))
    await s.authority.recordOperationResult({ ticket, outcome: 'external_issued', objects: [{
      kind: 'staging', cloud_path: ticket.target.cloud_path, file_id: `cloud://synthetic/${index}.mp4`,
    }], reason: null })
  }
  const retained = structuredClone(s.db.rows.clip_storage_authorities[job.clip_id])
  delete s.db.rows.saved_clip_sync_jobs[job.job_id]
  delete s.db.rows.saved_clip_sync_idempotency[documentId(job.clip_id)]
  expect(await createSavedClipJob({ ...s.input, idFactory: () => 'recreated-job' })).toMatchObject({ status: 'pending' })
  expect(s.db.rows.clip_storage_authorities[job.clip_id]).toEqual(retained)
  expect(s.db.rows.saved_clip_sync_jobs['recreated-job'].storage_admission.epoch).toBe(epoch)
  expect(last(s.db).count).toBeLessThanOrEqual(90)
  expect(s.db.transactionAttempts.slice(-2).map(attempt => attempt.count)).toEqual([83, 83])
  expect(last(s.db).calls.filter(c => c.collection === 'media_storage_operations')).toHaveLength(4)
  expect(last(s.db).calls.filter(c => c.collection === 'media_storage_objects')).toHaveLength(64)
  expect(last(s.db).calls.some(c => c.collection === 'session_command_authority')).toBe(true)
  expect(last(s.db).calls.some(c => c.collection === 'session_role_binding_history')).toBe(true)
  const replayCount = last(s.db).count
  await createSavedClipJob(s.input); noWrites(s.db)
  const retryCount = last(s.db).count
  delete s.db.rows.saved_clip_sync_jobs['recreated-job']; delete s.db.rows.saved_clip_sync_idempotency[documentId(job.clip_id)]
  delete s.db.rows.media_storage_operations[Object.keys(s.db.rows.media_storage_operations)[0]]
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect({ freshCount, replayCount, retryCount, incompleteCount: last(s.db).count })
    .toEqual({ freshCount: 19, replayCount: 83, retryCount: 12, incompleteCount: 65 })
})

function ordinaryEnvelope(s, kind, managed) {
  const envelope = structuredClone(s.facts)
  envelope.session.session_id = managed ? sessionId : 'ordinary-session'
  envelope.session.evidence_source_kind = kind
  envelope.session.limitations = kind === 'development_replay' ? ['development_replay_media'] : []
  envelope.media = [{ clip_id: 'ordinary-clip', session_id: envelope.session.session_id,
    shot_id: envelope.session.shots[0].shot_id, artifact_id: envelope.session.artifact_id,
    relative_path: 'clip.mp4', source_kind: kind, content_sha256: 'a'.repeat(64), byte_size: 100, media_type: 'video/mp4' }]
  return envelope
}

test.each([['usb_live', false], ['usb_live', true], ['development_replay', true]])('generic %s managed=%s refuses creation and old incomplete writer ports before provider', async (kind, managed) => {
  const s = await setup(); const envelope = ordinaryEnvelope(s, kind, managed)
  const id = envelope.session.session_id
  delete s.db.rows.sessions[id]
  s.db.rows.field_live_sessions[id] = { ...s.db.rows.field_live_sessions[sessionId], session_id: id }
  s.db.rows.session_role_bindings[documentId(`${id}:red`)] = {
    ...s.db.rows.session_role_bindings[documentId(`${sessionId}:red`)], session_id: id,
  }
  const repo = createCloudSyncRepository(s.db)
  const args = { ...s.scope, repo, envelope, idempotencyKey: 'ordinary', manifestHash: 'f'.repeat(64), idFactory: () => 'ordinary-job', now }
  await expect(createSyncJob(args)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  const raw = { job_id: 'old-job', table_id: 'table-1', _device_id: 'device-1', session_id: envelope.session.session_id,
    idempotency_key: 'ordinary', manifest_hash: 'f'.repeat(64), status: 'awaiting_media', envelope }
  await expect(repo.create(raw)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  s.db.rows.sync_jobs['old-job'] = raw
  const provider = storage(); const operation = { ...s.scope, jobId: 'old-job', repo, storage: provider, clipId: 'ordinary-clip', now }
  await expect(authorizeMedia(operation)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(completeSyncJob(operation)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(repo.markCompleted('old-job', now, {})).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(provider.authorizeUpload).not.toHaveBeenCalled(); expect(provider.verifyObject).not.toHaveBeenCalled()
})

test('generic create replay refuses an incomplete unsaved job before invoking publication port', async () => {
  const s = await setup(); const envelope = ordinaryEnvelope(s, 'usb_live', false)
  const raw = { table_id: 'table-1', _device_id: 'device-1', manifest_hash: 'f'.repeat(64), status: 'awaiting_media', envelope }
  const validatePublication = jest.fn()
  const repo = { getByIdempotencyKey: async () => raw, validatePublication }
  await expect(createSyncJob({ ...s.scope, envelope, idempotencyKey: 'ordinary', manifestHash: 'f'.repeat(64), repo }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(validatePublication).not.toHaveBeenCalled()
})

test('managed history discovery outage remains retryable without exposing raw private errors', async () => {
  const s = await setup()
  const collection = s.db.collection
  s.db.collection = name => name === 'session_role_binding_history'
    ? { where() { throw new Error('private query credential detail') } } : collection(name)
  await expect(createSavedClipJob(s.input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE', message: 'SYNC_STORAGE_UNAVAILABLE' })
})
