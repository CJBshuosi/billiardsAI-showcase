const crypto = require('crypto')
const { Readable } = require('stream')
const { setup } = require('../helpers/managed-saved-flow-fixture')
const { now } = require('../helpers/managed-media-admission-fixture')
const { createSavedClipJob, authorizeSavedClip, completeSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const { makeStorageIdentity } = require('../../cloudfunctions/shared/media-storage-authority')
const { isPrivateClipOwner } = require('../../cloudfunctions/shared/shot-ownership')
const { createRuntimeDependencies } = require('../../cloudfunctions/sync-api')

// Only I/O is controlled. Owner/birth/admission, reservation, result persistence
// and publication decisions use the actual code and document-only transactions.
function provider() {
  const uploads = []; const copies = []
  return { uploads, copies,
    async authorizeRegisteredUpload(input) {
      uploads.push(input)
      return { outcome: 'external_issued', file_id: `cloud://test/${input.object_key}`,
        authorization: { method: 'PUT', url: 'https://upload.invalid/?signature=SECRET',
          headers: { authorization: 'SECRET', 'x-cos-security-token': 'TOKEN' }, expires_in_seconds: 300 } }
    },
    async copyRegisteredObject(input) {
      copies.push(input)
      return { outcome: 'server_succeeded', file_id: `cloud://test/${input.publish_object_key}` }
    },
    authorizeUpload() { throw new Error('legacy upload must not run') },
    verifyObject() { throw new Error('legacy verify must not run') },
  }
}
async function saved(options) {
  const s = await setup(options); const job = await createSavedClipJob(s.input)
  return { ...s, job, storage: provider() }
}
const args = s => ({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage: s.storage, now })
function operations(s) { return Object.values(s.db.rows.media_storage_operations || {}).flatMap(page => page.entries.map(entry => entry.operation)) }
function checkBudgets(s) { for (const attempt of s.db.transactionAttempts) expect(attempt.count).toBeLessThanOrEqual(90) }
async function close(s) {
  return s.db.runTransaction(async tx => {
    const closed = await s.authority.closeForDeletion(tx, s.identity)
    await tx.collection('owner_clip_tombstones').doc(s.descriptor.clip_id).set({ data: { clip_id: s.descriptor.clip_id } })
    return closed
  })
}
function barrier() {
  let entered; let release
  const ready = new Promise(resolve => { entered = resolve }); const gate = new Promise(resolve => { release = resolve })
  return { ready, release, async hold() { entered(); await gate } }
}

test('earlier uploaded grant selects its own staging object and completed replay is read-only', async () => {
  const s = await saved(); const original = JSON.stringify(s.descriptor)
  const root = structuredClone(s.db.rows.clip_storage_authorities[s.descriptor.clip_id])
  const first = await authorizeSavedClip(args(s)); const second = await authorizeSavedClip(args(s))
  expect(first.upload_attempt).toEqual({ contract: 'managed_upload_attempt_v1', operation_id: expect.stringMatching(/^[0-9a-f-]{36}$/u), sequence: 1 })
  expect(second.upload_attempt.sequence).toBe(2)
  expect(first).not.toHaveProperty('file_id'); expect(first).not.toHaveProperty('ticket')
  const complete = await completeSavedClipJob({ ...args(s), uploadAttempt: first.upload_attempt })
  expect(complete.status).toBe('completed')
  expect(s.storage.copies[0].staging_file_id).toBe(`cloud://test/${s.storage.uploads[0].object_key}`)
  const clip = s.db.rows.clips[s.descriptor.clip_id]
  expect(clip.storage_file_id).toBe(`cloud://test/${s.storage.copies[0].publish_object_key}`)
  expect(clip.storage_objects).toHaveLength(2)
  expect(JSON.stringify(s.db.rows.saved_clip_sync_jobs[s.job.job_id].descriptor)).toBe(original)
  expect(clip.ownership_receipt).toEqual(s.descriptor.ownership_receipt)
  expect(s.db.rows.clip_storage_authorities[s.descriptor.clip_id].epoch).toBe(root.epoch)
  const before = structuredClone(s.db.rows)
  expect(await completeSavedClipJob(args(s))).toEqual(complete)
  expect(await createSavedClipJob(s.input)).toEqual(complete)
  expect(await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id })).toEqual({ completed: true, job: s.db.rows.saved_clip_sync_jobs[s.job.job_id] })
  expect(s.db.rows).toEqual(before); expect(s.storage.copies).toHaveLength(1)
  const database = JSON.stringify(s.db.rows)
  for (const secret of ['SECRET', 'TOKEN', 'https://', 'upload_attempt']) expect(database).not.toContain(secret)
  expect(operations(s).map(op => op.outcome)).toEqual(['external_issued', 'external_issued', 'server_succeeded'])
  checkBudgets(s)
})

test.each([undefined, null, {}, { contract: 'managed_upload_attempt_v1', operation_id: crypto.randomUUID(), sequence: 1 },
  { contract: 'managed_upload_attempt_v1', operation_id: crypto.randomUUID(), sequence: '1' }])('invalid or foreign attempt refuses before copy allocation: %p', async attempt => {
  const s = await saved(); await authorizeSavedClip(args(s)); const before = structuredClone(s.db.rows)
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: attempt })).rejects.toMatchObject({ code: attempt && typeof attempt.sequence === 'number' ? 'SYNC_CONFLICT' : 'SYNC_INVALID' })
  expect(s.db.rows).toEqual(before); expect(s.storage.copies).toHaveLength(0)
})

test.each(['upload', 'copy'])('missing registered %s port remains retryable before allocation', async kind => {
  const s = await saved(); const before = structuredClone(s.db.rows)
  delete s.storage[kind === 'upload' ? 'authorizeRegisteredUpload' : 'copyRegisteredObject']
  await expect((kind === 'upload' ? authorizeSavedClip : completeSavedClipJob)(args(s))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before)
})

test.each(['upload', 'copy'])('late %s result survives owner closure and cannot deliver or publish', async kind => {
  const s = await saved(); const first = kind === 'copy' ? await authorizeSavedClip(args(s)) : null
  const io = barrier(); const port = kind === 'copy' ? 'copyRegisteredObject' : 'authorizeRegisteredUpload'
  const actual = s.storage[port]; s.storage[port] = async input => { const result = await actual(input); await io.hold(); return result }
  const pending = kind === 'copy' ? completeSavedClipJob({ ...args(s), uploadAttempt: first.upload_attempt }) : authorizeSavedClip(args(s))
  await io.ready; await close(s); io.release()
  await expect(pending).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(operations(s).at(-1).outcome).toBe(kind === 'copy' ? 'server_succeeded' : 'external_issued')
  expect(operations(s).at(-1).objects).toHaveLength(1)
  expect(Object.keys(s.db.rows.clips)).toHaveLength(0)
  const snapshot = await s.db.runTransaction(tx => s.authority.snapshotForDeletion(tx, s.descriptor.clip_id))
  expect(snapshot.erasable).toBe(false)
  await expect(authorizeSavedClip(args(s))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  checkBudgets(s)
})

test('two successful copies retain both observations while only winner publishes references', async () => {
  const s = await saved(); const grant = await authorizeSavedClip(args(s)); const io = barrier()
  const actual = s.storage.copyRegisteredObject
  s.storage.copyRegisteredObject = async input => { const result = await actual(input); if (s.storage.copies.length === 1) await io.hold(); return result }
  const first = completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })
  await io.ready
  const winner = await completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })
  const published = structuredClone(s.db.rows.clips[s.descriptor.clip_id]); io.release()
  expect(await first).toEqual(winner); expect(s.db.rows.clips[s.descriptor.clip_id]).toEqual(published)
  expect(operations(s).map(op => op.outcome)).toEqual(['external_issued', 'server_succeeded', 'server_succeeded'])
  const entries = Object.values(s.db.rows.media_storage_operations).flatMap(page => page.entries)
  expect(entries.map(entry => entry.object.references.length)).toEqual([1, 0, 1])
  checkBudgets(s)
})

test.each([
  [{ outcome: 'no_write', reason: 'not_started', permanent: true }, 'SYNC_CONFLICT'],
  [{ outcome: 'no_write', reason: 'not_started', permanent: false }, 'SYNC_STORAGE_UNAVAILABLE'],
  [{ outcome: 'unknown', reason: 'provider_error', file_id: 'cloud://observed/failed-copy' }, 'SYNC_STORAGE_UNAVAILABLE'],
])('copy result %p is durably observed before safe error %s', async (result, code) => {
  const s = await saved(); const grant = await authorizeSavedClip(args(s))
  s.storage.copyRegisteredObject = async () => result
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })).rejects.toMatchObject({ code })
  expect(operations(s).at(-1)).toMatchObject({ outcome: result.outcome, reason: result.reason })
  expect(operations(s).at(-1).objects).toHaveLength(result.file_id ? 1 : 0)
  expect(Object.keys(s.db.rows.clips)).toHaveLength(0)
})

test('production factory uses actual stream-backed provider and preserves default-off assembly', async () => {
  const s = await setup(); const bytes = Buffer.from('000000186674797069736f6d0000000069736f6d6d703432', 'hex')
  s.descriptor.byte_size = bytes.length; s.descriptor.content_sha256 = crypto.createHash('sha256').update(bytes).digest('hex')
  s.identity = makeStorageIdentity(s.descriptor, { ...s.scope, kind: 'saved_private' })
  const objects = new Map(); const reads = []
  let transactionDepth = 0
  const transact = s.db.runTransaction.bind(s.db)
  s.db.runTransaction = callback => transact(async tx => {
    transactionDepth++
    try { return await callback(tx) } finally { transactionDepth-- }
  })
  const sdk = {
    downloadFile() { throw new Error('legacy download forbidden') },
    async getUploadMetadata({ cloudPath }) { expect(transactionDepth).toBe(0); return { data: { fileId: `cloud://test/${cloudPath}`, cosFileId: 'cos', token: 'TOKEN', authorization: 'SECRET', url: 'https://upload.invalid' } } },
    async getTempFileURL({ fileList }) { expect(transactionDepth).toBe(0); reads.push(...fileList); return { fileList: fileList.map(fileID => ({ fileID, code: 'SUCCESS', tempFileURL: 'https://read.invalid' })) } },
    async uploadFile({ cloudPath, fileContent }) { expect(transactionDepth).toBe(0); expect(fileContent).toBeInstanceOf(require('fs').ReadStream); const chunks = []; for await (const chunk of fileContent) chunks.push(chunk); objects.set(`cloud://test/${cloudPath}`, Buffer.concat(chunks)); return { fileID: `cloud://test/${cloudPath}` } },
  }
  const deps = createRuntimeDependencies({ db: s.db, storageCloud: sdk, environment: { MANAGED_SESSIONS_ENABLED: 'true', MANAGED_MEDIA_ADMISSION_ENABLED: 'true' }, clock: () => Date.parse(now),
    registeredStorageOptions: { openDownload: async () => Object.assign(Readable.from([objects.get(reads.at(-1))]), { statusCode: 200, headers: {} }) } })
  const job = await createSavedClipJob({ ...s.input, repo: deps.savedClipRepo })
  const input = { ...s.scope, jobId: job.job_id, repo: deps.savedClipRepo, storage: deps.storage, now }
  const grant = await authorizeSavedClip(input)
  const staging = operations(s)[0].target.cloud_path; objects.set(`cloud://test/${staging}`, bytes)
  await expect(completeSavedClipJob({ ...input, uploadAttempt: grant.upload_attempt })).resolves.toMatchObject({ status: 'completed' })
  const clip = s.db.rows.clips[s.descriptor.clip_id]
  expect(objects.get(clip.storage_file_id)).toEqual(bytes)
  const projection = Object.values(s.db.rows.field_shot_ownership)[0]
  expect(isPrivateClipOwner(clip, 'user-a', projection)).toBe(true)
  expect(isPrivateClipOwner(clip, 'user-b', projection)).toBe(false)
  await expect(deps.joinAdmin.createManagedSession({ openRequestId: crypto.randomUUID(), ...s.scope })).rejects.toMatchObject({ code: 'SYNC_CONFIGURATION_REQUIRED' })
  const trusted = createRuntimeDependencies({ db: s.db, storageCloud: sdk, managedSessionsEnabled: true,
    managedMediaAdmissionEnabled: true, environment: {}, clock: () => Date.parse(now) })
  const born = await trusted.joinAdmin.createManagedSession({ openRequestId: crypto.randomUUID(), ...s.scope })
  expect(s.db.rows.managed_session_births[born.session_id].media_admission_contract).toBe('managed_clip_admission_v1')
  const database = JSON.stringify(s.db.rows)
  for (const secret of ['SECRET', 'TOKEN', 'https://', 'upload_attempt']) expect(database).not.toContain(secret)
  checkBudgets(s)
})

function signed(value, path) {
  const body = JSON.stringify(value); const timestamp = String(Math.floor(Date.parse(now) / 1000)); const nonce = crypto.randomUUID()
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
  const signature = crypto.createHmac('sha256', 'test-secret').update(canonicalSignatureInput({ method: 'POST', path, timestamp, nonce, bodyHash })).digest('hex')
  return { httpMethod: 'POST', path, body, headers: { 'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce, 'x-body-sha256': bodyHash, 'x-signature': signature } }
}
test('signed completion forwards only the exact attempt and rejects unknown or malformed completion fields', async () => {
  const s = await saved(); const { createHandler } = require('../../cloudfunctions/sync-api')
  const handler = createHandler({ savedClipRepo: s.repo, storage: s.storage, nowIso: () => now,
    auth: { nowSeconds: () => Math.floor(Date.parse(now) / 1000), nonceRepo: { consume: async () => true },
      deviceRepo: { getActiveDevice: async () => ({ device_id: 'device-1', table_id: 'table-1', status: 'active', secret: 'test-secret' }) } } })
  const grant = await authorizeSavedClip(args(s)); const path = `/v1/sync/saved-clips/${s.job.job_id}/complete`
  for (const body of [{ file_id: 'cloud://forged' }, { upload_attempt: grant.upload_attempt, extra: true }, { upload_attempt: null }]) {
    const before = structuredClone(s.db.rows); const result = await handler(signed(body, path))
    expect(result.statusCode).toBe(400); expect(JSON.parse(result.body).error.code).toBe('SYNC_INVALID'); expect(s.db.rows).toEqual(before)
  }
  const result = await handler(signed({ upload_attempt: grant.upload_attempt }, path))
  expect(result.statusCode).toBe(200); expect(JSON.parse(result.body).data.status).toBe('completed')
  expect((await handler(signed({}, path))).statusCode).toBe(200)
})

test.each(['upload', 'copy'])('N63 to N64 actual %s composite and full alias reference/retry callbacks stay within 90', async kind => {
  let retryPublication = false
  const s = await saved({ dbOptions: { retryConflicts: 1, beforeCommit({ attempt }) {
    if (retryPublication && attempt.calls.some(call => call.collection === 'clips' && call.method === 'set')) {
      retryPublication = false; const error = new Error('controlled transaction retry'); error.code = 'SYNTHETIC_CONFLICT'; throw error
    }
  } } })
  let grant
  for (let index = 0; index < 63; index++) { const issued = await authorizeSavedClip(args(s)); grant ||= issued }
  const start = s.db.transactionAttempts.length
  if (kind === 'upload') {
    await authorizeSavedClip(args(s))
    const before = structuredClone(s.db.rows)
    await expect(authorizeSavedClip(args(s))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
    expect(s.db.rows).toEqual(before)
    await expect(completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
    expect(s.storage.copies).toHaveLength(0)
  } else {
    retryPublication = true
    await completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })
    expect(s.db.rows.clip_storage_authorities[s.descriptor.clip_id].high_water).toBe(64)
    expect(Object.keys(s.db.rows.media_storage_objects)).toHaveLength(64)
    const publicationAttempts = s.db.transactionAttempts.slice(start).filter(attempt => attempt.calls.some(call => call.collection === 'clips' && call.method === 'set'))
    expect(publicationAttempts).toHaveLength(2)
    expect(publicationAttempts.map(attempt => attempt.count)).toEqual([87, 87])
    const before = structuredClone(s.db.rows)
    await completeSavedClipJob(args(s)); expect(s.db.rows).toEqual(before)
    expect(s.storage.copies).toHaveLength(1)
  }
  const callbacks = s.db.transactionAttempts.slice(start).map(attempt => attempt.count)
  console.info(`ACTUAL_${kind.toUpperCase()}_N63_N64_CALLBACKS`, callbacks)
  checkBudgets(s)
}, 15000)

test.each(['upload', 'copy'])('result persistence failure withholds %s success and retry reserves a new target', async kind => {
  let failResult = false
  const s = await saved({ dbOptions: { failWrite(name) {
    if (failResult && name === 'media_storage_objects') throw new Error('RAW_DB_CREDENTIAL_SECRET')
  } } })
  const grant = kind === 'copy' ? await authorizeSavedClip(args(s)) : null
  failResult = true
  const operation = () => kind === 'copy' ? completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt }) : authorizeSavedClip(args(s))
  await expect(operation()).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(operations(s).at(-1).outcome).toBeNull(); expect(Object.keys(s.db.rows.clips)).toHaveLength(0)
  failResult = false
  await operation()
  const inputs = kind === 'copy' ? s.storage.copies.map(input => input.publish_object_key) : s.storage.uploads.map(input => input.object_key)
  expect(new Set(inputs).size).toBe(2)
  expect(operations(s).filter(op => op.outcome === null)).toHaveLength(1)
  expect(JSON.stringify(s.db.rows)).not.toContain('RAW_DB_CREDENTIAL_SECRET')
  checkBudgets(s)
})

test('grant unknown observations survive malformed credentials and SDK throws without raw durable details', async () => {
  const s = await saved()
  s.storage.authorizeRegisteredUpload = async () => ({ outcome: 'unknown', reason: 'provider_error', file_id: 'cloud://observed/malformed-grant', authorization: { url: 'SECRET' } })
  await expect(authorizeSavedClip(args(s))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(operations(s)[0]).toMatchObject({ outcome: 'unknown', objects: [{ file_id: 'cloud://observed/malformed-grant' }] })
  s.storage.authorizeRegisteredUpload = async () => { throw new Error('SECRET') }
  await expect(authorizeSavedClip(args(s))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(operations(s)[1]).toMatchObject({ outcome: 'unknown', objects: [] })
  expect(JSON.stringify(s.db.rows)).not.toContain('SECRET')
})

test.each(['descriptor', 'marker', 'scope', 'status'])('credential delivery rereads actual current job after provider: %s changed', async change => {
  const s = await saved(); const io = barrier(); const actual = s.storage.authorizeRegisteredUpload
  s.storage.authorizeRegisteredUpload = async input => { const result = await actual(input); await io.hold(); return result }
  const pending = authorizeSavedClip(args(s)); await io.ready
  const job = s.db.rows.saved_clip_sync_jobs[s.job.job_id]
  if (change === 'descriptor') job.descriptor.byte_size++
  if (change === 'marker') job.storage_admission.epoch = crypto.randomUUID()
  if (change === 'scope') job._device_id = 'other-device'
  if (change === 'status') job.status = 'completed'
  io.release(); await expect(pending).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(operations(s)[0].outcome).toBe('external_issued')
  expect(Object.keys(s.db.rows.clips)).toHaveLength(0)
})

test('job compaction preserves authority and refuses correlation from the earlier ingest job', async () => {
  const s = await saved(); const grant = await authorizeSavedClip(args(s)); const root = structuredClone(s.db.rows.clip_storage_authorities[s.descriptor.clip_id])
  delete s.db.rows.saved_clip_sync_jobs[s.job.job_id]
  for (const key of Object.keys(s.db.rows.saved_clip_sync_idempotency)) delete s.db.rows.saved_clip_sync_idempotency[key]
  const recreated = await createSavedClipJob({ ...s.input, idFactory: () => 'recreated-job' }); s.job = recreated
  expect(s.db.rows.clip_storage_authorities[s.descriptor.clip_id]).toEqual(root)
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.storage.copies).toHaveLength(0)
  const before = structuredClone(s.db.rows)
  await expect(s.repo.beginTrackedUpload({ ...s.scope, deviceId: 'foreign', jobId: recreated.job_id })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before)
})

// Before first publication there is no user-visible clip to delete. These are
// INTERNAL common-root authority closure+tombstone races, not owner API proof.
test.each(['reservation', 'result-persistence', 'credential-delivery', 'publication'].flatMap(phase => [true, false].map(closeFirst => [phase, closeFirst])))(
  'internal closure at actual %s callback, closeFirst=%s', async (phase, closeFirst) => {
    const boundary = barrier(); const io = barrier(); const closeBoundary = barrier(); const order = []
    let armed = false
    const s = await saved({ dbOptions: { async beforeCommit({ attempt, rows }) {
      if (!armed) return
      const wrote = (collection, method = 'set') => attempt.calls.some(call => call.collection === collection && call.method === method)
      if (wrote('owner_clip_tombstones')) {
        order.push('internal-close-commit')
        if (closeFirst && phase === 'reservation') await closeBoundary.hold()
        return
      }
      const op = Object.values(rows.media_storage_operations || {}).flatMap(page => page.entries.map(entry => entry.operation)).at(-1)
      const isBoundary = phase === 'reservation' ? wrote('media_storage_operations') && op?.outcome === null
        : phase === 'result-persistence' ? wrote('media_storage_operations') && op?.outcome === 'external_issued'
          : phase === 'credential-delivery' ? closeFirst
            ? wrote('media_storage_operations') && op?.outcome === 'external_issued'
            : !attempt.calls.some(call => call.method !== 'get') && op?.outcome === 'external_issued'
            : wrote('clips')
      if (isBoundary) {
        order.push(`${phase}-commit`)
        if (!closeFirst || phase === 'credential-delivery') await boundary.hold()
      }
    } } })
    let grant
    if (phase === 'publication') grant = await authorizeSavedClip(args(s))
    const port = phase === 'publication' ? 'copyRegisteredObject' : 'authorizeRegisteredUpload'
    if (closeFirst && ['result-persistence', 'publication'].includes(phase)) {
      const actual = s.storage[port]
      s.storage[port] = async input => { const result = await actual(input); order.push('provider-result-held'); await io.hold(); return result }
    }
    armed = true
    const run = () => phase === 'publication'
      ? completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt }) : authorizeSavedClip(args(s))
    let pending; let closing
    if (closeFirst && phase === 'reservation') {
      closing = close(s); await closeBoundary.ready; pending = run().then(value => ({ value }), error => ({ error }))
      closeBoundary.release()
    } else {
      pending = run().then(value => ({ value }), error => ({ error }))
      const ready = closeFirst && ['result-persistence', 'publication'].includes(phase) ? io.ready : boundary.ready
      await Promise.race([ready, pending.then(() => { throw new Error(`bypassed actual ${phase} barrier`) })])
      closing = close(s)
      if (closeFirst && ['result-persistence', 'publication'].includes(phase)) { await closing; io.release() }
      else boundary.release()
    }
    const result = await pending; await closing
    if (!closeFirst && ['credential-delivery', 'publication'].includes(phase)) {
      expect(result.error).toBeUndefined()
      expect(result.value).toMatchObject(phase === 'publication' ? { status: 'completed' } : { upload_attempt: { sequence: 1 } })
      expect(order.indexOf(`${phase}-commit`)).toBeLessThan(order.indexOf('internal-close-commit'))
    } else {
      expect(result.error).toMatchObject({ code: 'SYNC_CONFLICT' })
    }
    if (phase === 'reservation' && closeFirst) {
      expect(operations(s)).toHaveLength(0); expect(s.storage.uploads).toHaveLength(0)
      expect(order).toEqual(['internal-close-commit'])
    } else {
      expect(operations(s).at(-1).outcome).toBe(phase === 'publication' ? 'server_succeeded' : 'external_issued')
      if (phase === 'result-persistence') {
        expect(order.filter(event => event.endsWith('commit'))).toEqual(closeFirst
          ? ['internal-close-commit', 'result-persistence-commit'] : ['result-persistence-commit', 'internal-close-commit'])
      }
    }
    expect(s.db.rows.clip_storage_authorities[s.descriptor.clip_id].admission).toBe('closed')
    expect(Object.keys(s.db.rows.clips)).toHaveLength(phase === 'publication' && !closeFirst ? 1 : 0)
    const before = structuredClone(s.db.rows)
    await expect(authorizeSavedClip(args(s))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(s.db.rows).toEqual(before); checkBudgets(s)
  },
)

test('actual owner deletion closes a normally published winner while a second real copy result remains pending', async () => {
  const { requestOwnerClipDeletion, createCloudOwnerDeleteRepository } = require('../../cloudfunctions/shared/owner-delete')
  const s = await setup(); const pendingResult = barrier()
  const bytes = Buffer.from('000000186674797069736f6d0000000069736f6d6d703432', 'hex')
  s.descriptor.byte_size = bytes.length
  s.descriptor.content_sha256 = crypto.createHash('sha256').update(bytes).digest('hex')
  s.identity = makeStorageIdentity(s.descriptor, { ...s.scope, kind: 'saved_private' })
  const objects = new Map(); const copies = []; let stagingId
  const sdk = {
    downloadFile() { throw new Error('legacy download forbidden') },
    async getUploadMetadata({ cloudPath }) { return { data: { fileId: `cloud://test/${cloudPath}`,
      cosFileId: 'cos', token: 'TOKEN', authorization: 'SECRET', url: 'https://upload.invalid' } } },
    async getTempFileURL({ fileList }) { return { fileList: fileList.map(fileID => ({ fileID, code: 'SUCCESS', tempFileURL: 'https://read.invalid' })) } },
    async uploadFile({ cloudPath, fileContent }) {
      expect(fileContent).toBeInstanceOf(require('fs').ReadStream)
      copies.push(cloudPath); const number = copies.length; const chunks = []
      for await (const chunk of fileContent) chunks.push(chunk)
      const copied = Buffer.concat(chunks); expect(copied).toEqual(bytes)
      objects.set(`cloud://test/${cloudPath}`, copied)
      if (number === 1) await pendingResult.hold()
      return { fileID: `cloud://test/${cloudPath}` }
    },
  }
  const dependencies = createRuntimeDependencies({ db: s.db, storageCloud: sdk, clock: () => Date.parse(now),
    registeredStorageOptions: { openDownload: async () => Object.assign(Readable.from([objects.get(stagingId)]), { statusCode: 200, headers: {} }) } })
  s.repo = dependencies.savedClipRepo; s.storage = dependencies.storage
  s.job = await createSavedClipJob({ ...s.input, repo: s.repo })
  const grant = await authorizeSavedClip(args(s))
  stagingId = operations(s)[0].objects[0].file_id; objects.set(stagingId, bytes)
  const late = completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt }).then(value => ({ value }), error => ({ error }))
  await pendingResult.ready
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })).resolves.toMatchObject({ status: 'completed' })
  const published = structuredClone(s.db.rows.clips[s.descriptor.clip_id])
  const deletionRepo = createCloudOwnerDeleteRepository(s.db)
  await expect(requestOwnerClipDeletion({ clipId: s.descriptor.clip_id, requestId: 'foreign-request', userId: 'user-b', repo: deletionRepo, now }))
    .rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
  // userId is the trusted authenticated identity seam of the real owner API.
  const receipt = await requestOwnerClipDeletion({ clipId: s.descriptor.clip_id, requestId: 'owner-request', userId: 'user-a', repo: deletionRepo, now })
  expect(receipt.status).toBe('deletion_requested')
  pendingResult.release()
  expect((await late).error).toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(operations(s).map(op => op.outcome)).toEqual(['external_issued', 'server_succeeded', 'server_succeeded'])
  expect(s.db.rows.clips[s.descriptor.clip_id].storage_file_id).toBe(published.storage_file_id)
  expect(s.db.rows.clips[s.descriptor.clip_id].retention_state).toBe('deletion_requested')
  expect(s.db.rows.owner_clip_tombstones[s.descriptor.clip_id].requested_at).toBe(now)
  expect(s.db.rows.clip_storage_authorities[s.descriptor.clip_id].admission).toBe('closed')
  const before = structuredClone(s.db.rows)
  await expect(authorizeSavedClip(args(s))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before); expect(copies).toHaveLength(2)
  const snapshot = await s.db.runTransaction(tx => s.authority.snapshotForDeletion(tx, s.descriptor.clip_id))
  expect(snapshot.erasable).toBe(false); checkBudgets(s)
})

test('completed-race copy branch rechecks actual job before requiring an attempt or calling provider', async () => {
  const s = await saved(); const grant = await authorizeSavedClip(args(s)); const beforeBegin = barrier()
  const racingRepo = { ...s.repo, async beginTrackedCopy(scope) { await beforeBegin.hold(); return s.repo.beginTrackedCopy(scope) } }
  const replay = completeSavedClipJob({ ...args(s), repo: racingRepo }).then(value => ({ value }), error => ({ error }))
  await Promise.race([beforeBegin.ready, replay.then(() => { throw new Error('completion did not reach actual-recheck boundary') })])
  const winner = await completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })
  const before = structuredClone(s.db.rows); beforeBegin.release()
  expect((await replay).value).toEqual(winner); expect(s.db.rows).toEqual(before)
  expect(s.storage.copies).toHaveLength(1)
})

test('validated completed race returns current job before checking a missing registered copy port', async () => {
  const s = await saved(); const grant = await authorizeSavedClip(args(s)); const beforeValidation = barrier()
  let validatedCurrent
  const racingRepo = { ...s.repo, async validatePublication(job) {
    expect(job.status).toBe('uploading')
    await beforeValidation.hold()
    validatedCurrent = await s.repo.validatePublication(job)
    return validatedCurrent
  } }
  const replay = completeSavedClipJob({ ...args(s), repo: racingRepo, storage: {} })
    .then(value => ({ value }), error => ({ error }))
  await Promise.race([beforeValidation.ready, replay.then(() => { throw new Error('replay bypassed actual validation boundary') })])
  const winner = await completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })
  const before = structuredClone(s.db.rows); beforeValidation.release()
  const result = await replay
  expect(validatedCurrent).toEqual(s.db.rows.saved_clip_sync_jobs[s.job.job_id])
  expect(validatedCurrent.status).toBe('completed')
  expect(result).toEqual({ value: winner })
  expect(s.db.rows).toEqual(before); expect(s.storage.copies).toHaveLength(1)
  expect(operations(s)).toHaveLength(2); checkBudgets(s)
})

test('legacy void-returning validation port preserves completed saved replay compatibility', async () => {
  const s = await saved({ marked: false })
  s.storage = { verifyObject: async () => ({ verified: true, file_id: 'cloud://legacy/saved' }) }
  const completed = await completeSavedClipJob(args(s))
  const legacyRepo = { ...s.repo, async validatePublication(job) { await s.repo.validatePublication(job) } }
  const before = structuredClone(s.db.rows)
  expect(await completeSavedClipJob({ ...args(s), repo: legacyRepo, storage: {} })).toEqual(completed)
  expect(s.db.rows).toEqual(before)
})

test('a completed status without an actual published clip cannot certify read-only replay', async () => {
  const s = await saved(); s.db.rows.saved_clip_sync_jobs[s.job.job_id].status = 'completed'
  const before = structuredClone(s.db.rows)
  await expect(completeSavedClipJob(args(s))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before); expect(s.storage.copies).toHaveLength(0)
})

test('publication write failure rolls back references, retains known copy, and retries with a fresh copy target', async () => {
  let failPublication = false
  const s = await saved({ dbOptions: { failWrite(name) { if (failPublication && name === 'clips') throw new Error('RAW_PUBLICATION_SECRET') } } })
  const grant = await authorizeSavedClip(args(s)); failPublication = true
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(operations(s).map(op => op.outcome)).toEqual(['external_issued', 'server_succeeded'])
  expect(Object.values(s.db.rows.media_storage_operations).flatMap(page => page.entries.map(entry => entry.object.references))).toEqual([[], []])
  expect(Object.keys(s.db.rows.clips)).toHaveLength(0)
  failPublication = false
  await completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })
  expect(s.storage.copies[0].publish_object_key).not.toBe(s.storage.copies[1].publish_object_key)
  expect(operations(s).map(op => op.outcome)).toEqual(['external_issued', 'server_succeeded', 'server_succeeded'])
  expect(JSON.stringify(s.db.rows)).not.toContain('RAW_PUBLICATION_SECRET'); checkBudgets(s)
})

test('a server-copy correlation cannot be used as an upload grant and corrupt full inventories roll back before I/O', async () => {
  const { makeManagedUploadAttempt } = require('../../cloudfunctions/shared/managed-upload-attempt')
  const s = await saved(); const grant = await authorizeSavedClip(args(s))
  const begun = await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt: grant.upload_attempt })
  const before = structuredClone(s.db.rows)
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: makeManagedUploadAttempt(begun.ticket) })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before); expect(s.storage.copies).toHaveLength(0)
  delete s.db.rows.media_storage_operations[Object.keys(s.db.rows.media_storage_operations)[0]]
  const corrupt = structuredClone(s.db.rows)
  await expect(authorizeSavedClip(args(s))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(corrupt); expect(s.storage.uploads).toHaveLength(1); expect(s.storage.copies).toHaveLength(0)
})

test.each([null, {}, [], { contract: 'managed_upload_attempt_v1', operation_id: crypto.randomUUID(), sequence: true }])('explicitly malformed present attempt always refuses service/repository replay and missing capability: %p', async uploadAttempt => {
  const s = await saved(); const grant = await authorizeSavedClip(args(s))
  delete s.storage.copyRegisteredObject
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  s.storage.copyRegisteredObject = provider().copyRegisteredObject
  await completeSavedClipJob({ ...args(s), uploadAttempt: grant.upload_attempt })
  const before = structuredClone(s.db.rows)
  delete s.storage.copyRegisteredObject
  await expect(completeSavedClipJob({ ...args(s), uploadAttempt })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  await expect(s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  await expect(completeSavedClipJob(args(s))).resolves.toMatchObject({ status: 'completed' })
  expect(s.db.rows).toEqual(before)
})
