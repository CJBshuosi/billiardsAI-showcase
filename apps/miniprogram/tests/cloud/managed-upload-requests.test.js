const crypto = require('crypto')
const { setup } = require('../helpers/managed-saved-flow-fixture')
const { now } = require('../helpers/managed-media-admission-fixture')
const { createSavedClipJob, authorizeSavedClip, completeSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const { createHandler } = require('../../cloudfunctions/sync-api')
const request = () => ({ contract: 'managed_upload_request_v1', request_id: crypto.randomUUID() })
const args = s => ({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage: s.storage })
const operations = s => Object.values(s.db.rows.media_storage_operations || {}).flatMap(page => page.entries.map(entry => entry.operation))
const records = s => Object.values(s.db.rows.managed_upload_requests || {})
async function saved(options) {
  const s = await setup(options); s.job = await createSavedClipJob(s.input)
  s.uploads = []; s.storage = {
    async authorizeRegisteredUpload(input) { s.uploads.push(input); return {
      outcome: 'external_issued', file_id: `cloud://test/${input.object_key}`,
      authorization: { method: 'PUT', url: 'https://upload.invalid/?signature=SECRET', headers: { authorization: 'SECRET' }, expires_in_seconds: 300 },
    } },
    async copyRegisteredObject(input) { return { outcome: 'server_succeeded', file_id: `cloud://test/${input.publish_object_key}` } },
  }
  s.db.rows.managed_upload_requests ||= {}
  s.db.setStrictTransactions(true)
  return s
}
function barrier() {
  let entered; let release
  return { ready: new Promise(resolve => { entered = resolve }), gate: new Promise(resolve => { release = resolve }),
    release: () => release(), hold() { entered(); return this.gate } }
}
async function close(s) {
  await s.db.runTransaction(async tx => {
    await s.authority.closeForDeletion(tx, s.identity)
    await tx.collection('owner_clip_tombstones').doc(s.descriptor.clip_id).set({ data: { clip_id: s.descriptor.clip_id } })
  })
}
async function invariant(s) {
  const snapshot = await s.db.runTransaction(tx => s.authority.snapshotForDeletion(tx, s.descriptor.clip_id))
  expect(snapshot.erasable).toBe(false)
  for (const attempt of s.db.transactionAttempts) expect(attempt.count).toBeLessThanOrEqual(90)
}
function reserved(uploadRequest, grant) {
  return { upload_request: uploadRequest, upload_attempt: grant.upload_attempt, request_status: 'reserved' }
}
function signed(value, path) {
  const body = JSON.stringify(value); const timestamp = String(Math.floor(Date.parse(now) / 1000)); const nonce = crypto.randomUUID()
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
  const signature = crypto.createHmac('sha256', 'test-secret').update(canonicalSignatureInput({ method: 'POST', path, timestamp, nonce, bodyHash })).digest('hex')
  return { httpMethod: 'POST', path, body, headers: { 'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce, 'x-body-sha256': bodyHash, 'x-signature': signature } }
}
function handler(s) {
  return createHandler({ savedClipRepo: s.repo, storage: s.storage, nowIso: () => now,
    auth: { nowSeconds: () => Math.floor(Date.parse(now) / 1000), nonceRepo: { consume: async () => true },
      deviceRepo: { getActiveDevice: async () => ({ device_id: 'device-1', table_id: 'table-1', status: 'active', secret: 'test-secret' }) } } })
}

test('public request helper is strict and exact', () => {
  const { MANAGED_UPLOAD_REQUEST_CONTRACT, validManagedUploadRequest } = require('../../cloudfunctions/shared/managed-upload-request')
  expect(MANAGED_UPLOAD_REQUEST_CONTRACT).toBe('managed_upload_request_v1')
  expect(validManagedUploadRequest(request())).toBe(true)
  for (const value of [null, [], {}, { ...request(), extra: true }, { ...request(), request_id: crypto.randomUUID().toUpperCase() },
    { ...request(), request_id: '00000000-0000-1000-8000-000000000000' }, { ...request(), contract: 'other' }]) expect(validManagedUploadRequest(value)).toBe(false)
})

test.each([null, [], {}, { ...request(), extra: true }, { ...request(), request_id: crypto.randomUUID().toUpperCase() }])('malformed explicit request refuses before I/O: %p', async uploadRequest => {
  const s = await saved(); const before = structuredClone(s.db.rows)
  await expect(authorizeSavedClip({ ...args(s), uploadRequest })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  expect(s.db.rows).toEqual(before); expect(s.uploads).toHaveLength(0)
})

test('signed route accepts exact request, rejects invalid/unknown body, and preserves untracked {}', async () => {
  const s = await saved(); const run = handler(s); const path = `/v1/sync/saved-clips/${s.job.job_id}/media-authorization`
  for (const body of [{ upload_request: null }, { upload_request: request(), extra: 1 }, { arbitrary: 1 }, { upload_request: { ...request(), request_id: 'bad' } }]) {
    const before = structuredClone(s.db.rows); const response = await run(signed(body, path))
    expect(response.statusCode).toBe(400); expect(JSON.parse(response.body).error.code).toBe('SYNC_INVALID'); expect(s.db.rows).toEqual(before)
  }
  const uploadRequest = request(); const response = await run(signed({ upload_request: uploadRequest }, path))
  expect(response.statusCode).toBe(200); const grant = JSON.parse(response.body).data
  expect(grant).toMatchObject({ upload_request: uploadRequest, request_status: 'issued' })
  expect(JSON.parse((await run(signed({ upload_request: uploadRequest }, path))).body).data).toEqual(reserved(uploadRequest, grant))
  const legacy = JSON.parse((await run(signed({}, path))).body).data
  expect(legacy.upload_attempt.sequence).toBe(2); expect(legacy).not.toHaveProperty('upload_request'); expect(legacy).not.toHaveProperty('request_status')
  expect(records(s)).toHaveLength(1); expect(s.uploads).toHaveLength(2); await invariant(s)
})

test('legacy unmarked jobs reject explicit request rather than downgrade', async () => {
  const s = await saved({ marked: false }); s.storage.authorizeUpload = jest.fn()
  await expect(authorizeSavedClip({ ...args(s), uploadRequest: request() })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  expect(s.storage.authorizeUpload).not.toHaveBeenCalled(); expect(records(s)).toHaveLength(0)
})

test('concurrent duplicate during provider wait never grants credentials or allocates twice', async () => {
  const s = await saved(); const uploadRequest = request(); const io = barrier(); const real = s.storage.authorizeRegisteredUpload
  s.storage.authorizeRegisteredUpload = async input => { const result = await real(input); await io.hold(); return result }
  const pending = authorizeSavedClip({ ...args(s), uploadRequest }); await io.ready
  const duplicate = await authorizeSavedClip({ ...args(s), uploadRequest })
  expect(duplicate).toEqual({ upload_request: uploadRequest, upload_attempt: { contract: 'managed_upload_attempt_v1', operation_id: operations(s)[0].operation_id, sequence: 1 }, request_status: 'reserved' })
  expect(operations(s)[0].outcome).toBeNull(); expect(s.uploads).toHaveLength(1); io.release()
  const issued = await pending; expect(issued.request_status).toBe('issued'); expect(duplicate).toEqual(reserved(uploadRequest, issued))
  const next = await authorizeSavedClip({ ...args(s), uploadRequest: request() }); expect(next.upload_attempt.sequence).toBe(2)
  expect(records(s)).toHaveLength(2); expect(s.uploads).toHaveLength(2)
  for (const secret of ['SECRET', 'https://', 'headers', 'authorization']) expect(JSON.stringify(records(s))).not.toContain(secret)
  for (const privateField of ['identity', 'ticket', 'owner_user_id', 'table_id', 'device_id', 'descriptor_sha256']) expect(issued).not.toHaveProperty(privateField)
  expect(records(s)[0].descriptor_sha256).toBe(s.db.rows.saved_clip_sync_jobs[s.job.job_id].descriptor_sha256)
  await invariant(s)
})

test.each(['provider-throw', 'result-write-failure', 'creator-crash'])('lost issuance %s permanently returns reservation', async mode => {
  let failResult = false
  const s = await saved({ dbOptions: { failWrite(name) { if (failResult && name === 'media_storage_objects') throw new Error('RAW_SECRET') } } })
  const uploadRequest = request()
  if (mode === 'provider-throw') s.storage.authorizeRegisteredUpload = async input => { s.uploads.push(input); throw new Error('RAW_SECRET') }
  if (mode === 'result-write-failure') failResult = true
  if (mode === 'creator-crash') await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  else await expect(authorizeSavedClip({ ...args(s), uploadRequest })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  failResult = false; const before = structuredClone(s.db.rows)
  const replay = await authorizeSavedClip({ ...args(s), uploadRequest, storage: {} })
  expect(replay.request_status).toBe('reserved'); expect(s.db.rows).toEqual(before)
  expect(s.uploads).toHaveLength(mode === 'creator-crash' ? 0 : 1); expect(operations(s)).toHaveLength(1)
  expect(JSON.stringify(s.db.rows)).not.toContain('RAW_SECRET'); await invariant(s)
})

test('atomic begin handles concurrent lookup misses and forced reservation retries', async () => {
  let retry = true
  const s = await saved({ dbOptions: { retryConflicts: 1, beforeCommit({ attempt }) {
    if (retry && attempt.calls.some(call => call.collection === 'managed_upload_requests' && call.method === 'set')) {
      retry = false; const error = new Error('controlled conflict'); error.code = 'SYNTHETIC_CONFLICT'; throw error
    }
  } } })
  const uploadRequest = request(); let misses = 0; const io = barrier(); const actualLookup = s.repo.lookupTrackedUploadRequest
  s.repo.lookupTrackedUploadRequest = async input => { const found = await actualLookup(input); if (!found && ++misses === 1) await io.hold(); return found }
  const first = authorizeSavedClip({ ...args(s), uploadRequest }); await io.ready
  const second = await authorizeSavedClip({ ...args(s), uploadRequest }); io.release(); const delayed = await first
  expect(second.request_status).toBe('issued'); expect(delayed).toEqual(reserved(uploadRequest, second))
  expect(s.uploads).toHaveLength(1); expect(operations(s)).toHaveLength(1); expect(records(s)).toHaveLength(1)
  expect(s.db.transactionAttempts.some(attempt => attempt.index === 1)).toBe(true); await invariant(s)
})

test('permanent duplicate survives publication, actual owner deletion, and mutable job compaction', async () => {
  const { requestOwnerClipDeletion, createCloudOwnerDeleteRepository } = require('../../cloudfunctions/shared/owner-delete')
  const s = await saved(); const uploadRequest = request(); const grant = await authorizeSavedClip({ ...args(s), uploadRequest })
  await completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt, now })
  const originalDescriptor = structuredClone(s.descriptor); const originalOperations = structuredClone(operations(s))
  expect(await authorizeSavedClip({ ...args(s), uploadRequest })).toEqual(reserved(uploadRequest, grant))
  await requestOwnerClipDeletion({ clipId: s.descriptor.clip_id, requestId: 'owner-request', userId: 'user-a', repo: createCloudOwnerDeleteRepository(s.db), now })
  expect(await authorizeSavedClip({ ...args(s), uploadRequest })).toEqual(reserved(uploadRequest, grant))
  delete s.db.rows.saved_clip_sync_jobs[s.job.job_id]
  delete s.db.rows.saved_clip_sync_idempotency[Object.keys(s.db.rows.saved_clip_sync_idempotency)[0]]
  const before = structuredClone(s.db.rows)
  expect(await authorizeSavedClip({ ...args(s), uploadRequest, storage: {} })).toEqual(reserved(uploadRequest, grant))
  expect(await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })).toEqual({ fresh: false, ticket: records(s)[0].ticket })
  expect(s.db.rows).toEqual(before); expect(operations(s)).toEqual(originalOperations); expect(s.descriptor).toEqual(originalDescriptor)
  expect(s.uploads).toHaveLength(1); await invariant(s)
})

test('late provider result is retained after closure but fresh credentials are withheld', async () => {
  const s = await saved(); const uploadRequest = request(); const io = barrier(); const actual = s.storage.authorizeRegisteredUpload
  s.storage.authorizeRegisteredUpload = async input => { const result = await actual(input); await io.hold(); return result }
  const pending = authorizeSavedClip({ ...args(s), uploadRequest }); await io.ready; await close(s)
  expect((await authorizeSavedClip({ ...args(s), uploadRequest })).request_status).toBe('reserved')
  io.release(); await expect(pending).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(operations(s)[0].outcome).toBe('external_issued'); expect(s.uploads).toHaveLength(1); await invariant(s)
})

test.each(['job', 'table', 'device'])('immutable authenticated %s collision does not expose a foreign reservation', async scope => {
  const s = await saved(); const uploadRequest = request(); await authorizeSavedClip({ ...args(s), uploadRequest })
  const changed = scope === 'job' ? { jobId: 'foreign-job' } : scope === 'table' ? { tableId: 'foreign-table' } : { deviceId: 'foreign-device' }
  const before = structuredClone(s.db.rows)
  await expect(authorizeSavedClip({ ...args(s), uploadRequest, ...changed })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before); expect(s.uploads).toHaveLength(1)
})

test.each(['extra', 'descriptor', 'identity', 'ticket', 'root', 'page', 'alias', 'same-job'])('corrupt %s binding/dependency rejects read-only duplicate', async change => {
  const s = await saved(); const uploadRequest = request(); await authorizeSavedClip({ ...args(s), uploadRequest }); const record = records(s)[0]
  if (change === 'extra') record.url = 'SECRET'
  if (change === 'descriptor') record.descriptor_sha256 = 'bad'
  if (change === 'identity') record.identity.byte_size++
  if (change === 'ticket') record.ticket.epoch = crypto.randomUUID()
  if (change === 'root') delete s.db.rows.clip_storage_authorities[s.descriptor.clip_id]
  if (change === 'page') delete s.db.rows.media_storage_operations[Object.keys(s.db.rows.media_storage_operations)[0]]
  if (change === 'alias') delete s.db.rows.media_storage_objects[Object.keys(s.db.rows.media_storage_objects)[0]]
  if (change === 'same-job') operations(s)[0].ingest_job_id = 'foreign-job'
  const before = structuredClone(s.db.rows)
  await expect(authorizeSavedClip({ ...args(s), uploadRequest })).rejects.toMatchObject({ code: ['ticket', 'same-job'].includes(change) ? 'SYNC_CONFLICT' : 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before); expect(s.uploads).toHaveLength(1)
})

test('request map write failure rolls back reservation and mutable status', async () => {
  const s = await saved({ dbOptions: { failWrite(name) { if (name === 'managed_upload_requests') throw new Error('RAW_SECRET') } } }); const before = structuredClone(s.db.rows)
  await expect(authorizeSavedClip({ ...args(s), uploadRequest: request() })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before); expect(s.uploads).toHaveLength(0); expect(operations(s)).toHaveLength(0)
})

test('N63 append-to64 and N64/A64 permanent duplicate callbacks remain doc-only <=90 including retries', async () => {
  let retry = false; let retryDuplicate = false
  const s = await saved({ dbOptions: { retryConflicts: 1, beforeCommit({ attempt }) {
    if (retryDuplicate && attempt.calls.some(call => call.collection === 'managed_upload_requests' && call.method === 'get')) {
      retryDuplicate = false; const error = new Error('controlled duplicate conflict'); error.code = 'SYNTHETIC_CONFLICT'; throw error
    }
    if (retry && attempt.calls.some(call => call.collection === 'managed_upload_requests' && call.method === 'set')) {
      retry = false; const error = new Error('controlled conflict'); error.code = 'SYNTHETIC_CONFLICT'; throw error
    }
  } } })
  for (let index = 0; index < 63; index++) await authorizeSavedClip(args(s))
  const start = s.db.transactionAttempts.length; const uploadRequest = request(); retry = true
  const grant = await authorizeSavedClip({ ...args(s), uploadRequest })
  const before = structuredClone(s.db.rows); retryDuplicate = true
  expect(await authorizeSavedClip({ ...args(s), uploadRequest })).toEqual(reserved(uploadRequest, grant))
  expect(s.db.rows).toEqual(before)
  await close(s); delete s.db.rows.saved_clip_sync_jobs[s.job.job_id]
  const closed = structuredClone(s.db.rows); retryDuplicate = true
  expect(await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })).toEqual({ fresh: false, ticket: records(s)[0].ticket })
  expect(await authorizeSavedClip({ ...args(s), uploadRequest })).toEqual(reserved(uploadRequest, grant))
  expect(s.db.rows).toEqual(closed)
  expect(operations(s)).toHaveLength(64); expect(s.uploads).toHaveLength(64)
  expect(Object.keys(s.db.rows.media_storage_objects)).toHaveLength(64)
  console.info('REQUEST_N63_N64_A64_CALLBACKS', s.db.transactionAttempts.slice(start).map(attempt => attempt.count))
  await invariant(s)
}, 15000)

// The loser must reach atomic recovery even if the winner closes publication
// after the loser's first read-only lookup missed the permanent mapping.
test.each(['publication', 'close', 'compaction', 'corrupt-job'])('lookup-miss race recovers reservation after %s', async phase => {
  const s = await saved(); const uploadRequest = request(); const io = barrier(); let first = true
  const actual = s.repo.lookupTrackedUploadRequest
  s.repo.lookupTrackedUploadRequest = async input => { const found = await actual(input); if (first) { first = false; await io.hold() } return found }
  const loser = authorizeSavedClip({ ...args(s), uploadRequest }).then(value => ({ value }), error => ({ error }))
  await io.ready; const winner = await authorizeSavedClip({ ...args(s), uploadRequest })
  if (phase === 'publication') await completeSavedClipJob({ ...args(s), uploadAttempt: winner.upload_attempt, now })
  if (phase === 'close') await close(s)
  if (phase === 'compaction') delete s.db.rows.saved_clip_sync_jobs[s.job.job_id]
  if (phase === 'corrupt-job') s.db.rows.saved_clip_sync_jobs[s.job.job_id].descriptor.byte_size++
  const before = structuredClone(s.db.rows); io.release()
  expect(await loser).toEqual({ value: reserved(uploadRequest, winner) })
  expect(s.db.rows).toEqual(before); expect(s.uploads).toHaveLength(1); await invariant(s)
})

test('actual runtime metadata invocation is outside transactions and occurs once for a durable request', async () => {
  const { createRuntimeDependencies } = require('../../cloudfunctions/sync-api')
  const s = await saved(); const io = barrier(); let metadataCalls = 0; let depth = 0
  const runTransaction = s.db.runTransaction.bind(s.db)
  s.db.runTransaction = callback => runTransaction(async tx => { depth++; try { return await callback(tx) } finally { depth-- } })
  const sdk = {
    downloadFile() { throw new Error('legacy download forbidden') },
    getTempFileURL() { throw new Error('read URL forbidden') },
    async getUploadMetadata({ cloudPath }) {
    expect(depth).toBe(0); metadataCalls++; await io.hold()
    return { data: { fileId: `cloud://test/${cloudPath}`, cosFileId: 'cos', token: 'TOKEN', authorization: 'SECRET', url: 'https://upload.invalid' } }
  } }
  const dependencies = createRuntimeDependencies({ db: s.db, storageCloud: sdk, clock: () => Date.parse(now) })
  s.repo = dependencies.savedClipRepo; s.storage = dependencies.storage
  const uploadRequest = request(); const pending = authorizeSavedClip({ ...args(s), uploadRequest })
  await io.ready
  const duplicate = await authorizeSavedClip({ ...args(s), uploadRequest }); expect(metadataCalls).toBe(1)
  io.release(); const issued = await pending
  expect(duplicate).toEqual(reserved(uploadRequest, issued)); expect(issued.request_status).toBe('issued')
  expect(await authorizeSavedClip({ ...args(s), uploadRequest })).toEqual(duplicate)
  expect(metadataCalls).toBe(1); expect(operations(s)).toHaveLength(1); expect(records(s)).toHaveLength(1)
  for (const secret of ['SECRET', 'TOKEN', 'https://', 'headers', 'authorization']) expect(JSON.stringify(records(s))).not.toContain(secret)
  await invariant(s)
})

test('database _id metadata is accepted but copy tickets cannot substitute for a reserved upload', async () => {
  const { makeManagedUploadAttempt } = require('../../cloudfunctions/shared/managed-upload-attempt')
  const s = await saved(); const uploadRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest })
  const record = records(s)[0]; record._id = Object.keys(s.db.rows.managed_upload_requests)[0]
  expect(await authorizeSavedClip({ ...args(s), uploadRequest })).toEqual(reserved(uploadRequest, issued))
  const copy = await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt: issued.upload_attempt })
  record.ticket = copy.ticket; const before = structuredClone(s.db.rows)
  await expect(authorizeSavedClip({ ...args(s), uploadRequest })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(s.db.runTransaction(tx => s.authority.getUploadOperation(tx, {
    identity: s.identity, ingestJobId: s.job.job_id, uploadAttempt: makeManagedUploadAttempt(copy.ticket),
  }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  expect(s.db.rows).toEqual(before); expect(s.uploads).toHaveLength(1); await invariant(s)
})

test.each(['root', 'page', 'alias'])('fresh request rolls back before provider on corrupt existing %s inventory', async corrupt => {
  const s = await saved(); await authorizeSavedClip({ ...args(s), uploadRequest: request() })
  if (corrupt === 'root') s.db.rows.clip_storage_authorities[s.descriptor.clip_id].revision++
  if (corrupt === 'page') s.db.rows.media_storage_operations[Object.keys(s.db.rows.media_storage_operations)[0]].entries[0].object.file_id = 'cloud://foreign/object'
  if (corrupt === 'alias') delete s.db.rows.media_storage_objects[Object.keys(s.db.rows.media_storage_objects)[0]]
  const before = structuredClone(s.db.rows)
  await expect(authorizeSavedClip({ ...args(s), uploadRequest: request() })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before); expect(records(s)).toHaveLength(1); expect(s.uploads).toHaveLength(1); await invariant(s)
})
