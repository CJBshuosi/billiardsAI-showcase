const crypto = require('crypto')
const { setup } = require('../helpers/managed-saved-flow-fixture')
const { now } = require('../helpers/managed-media-admission-fixture')
const { createSavedClipJob, authorizeSavedClip, completeSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const { createHandler } = require('../../cloudfunctions/sync-api')
const { digest, documentId } = require('../../cloudfunctions/shared/shot-ownership')
const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
const request = () => ({ contract: 'managed_upload_request_v1', request_id: crypto.randomUUID() })
const args = s => ({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage: s.storage })
const binding = s => Object.values(s.db.rows.managed_upload_requests)[0]
const receiptPages = s => Object.values(s.db.rows.upload_terminal_receipts || {})
const receipt = (uploadRequest, uploadAttempt = null) => ({ contract: 'managed_upload_receipt_v1', upload_request: uploadRequest, upload_attempt: uploadAttempt,
  evidence: uploadAttempt ? { kind: 'http200', local_binding_sha256: 'a'.repeat(64), started_at_utc: '2024-02-29T23:59:59.123456Z', received_at_utc: '2024-02-29T23:59:59.123457Z', status_code: 200 }
    : { kind: 'retired_before_start', local_binding_sha256: 'a'.repeat(64) } })
async function saved(options) {
  const s = await setup(options); s.job = await createSavedClipJob(s.input)
  s.db.rows.managed_upload_requests ||= {}; s.db.rows.upload_terminal_receipts ||= {}; s.db.setStrictTransactions(true)
  s.storage = { authorizeRegisteredUpload: jest.fn(async input => ({ outcome: 'external_issued', file_id: `cloud://test/${input.object_key}`,
    authorization: { method: 'PUT', url: 'https://upload.invalid', headers: {}, expires_in_seconds: 300 } })),
  copyRegisteredObject: jest.fn(async input => ({ outcome: 'server_succeeded', file_id: `cloud://test/${input.publish_object_key}` })) }
  return s
}
const record = (s, value, changed = {}) => s.repo.recordTrackedUploadReceipt({ ...s.scope, jobId: s.job.job_id, receipt: value, ...changed })
function signed(value, path) {
  const body = JSON.stringify(value); const timestamp = String(Math.floor(Date.parse(now) / 1000)); const nonce = crypto.randomUUID()
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const signature = crypto.createHmac('sha256', 'test-secret').update(canonicalSignatureInput({ method: 'POST', path, timestamp, nonce, bodyHash })).digest('hex')
  return { httpMethod: 'POST', path, body, headers: { 'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce, 'x-body-sha256': bodyHash, 'x-signature': signature } }
}
function handler(s, scope = {}) {
  const nonces = new Set()
  return createHandler({ savedClipRepo: s.repo, storage: s.storage, nowIso: () => now,
    auth: { nowSeconds: () => Math.floor(Date.parse(now) / 1000), nonceRepo: { consume: async (_, nonce) => { if (nonces.has(nonce)) return false; nonces.add(nonce); return true } },
      deviceRepo: { getActiveDevice: async () => ({ device_id: 'device-1', table_id: 'table-1', status: 'active', secret: 'test-secret', ...scope }) } } })
}
async function invariant(s) {
  const snapshot = await s.db.runTransaction(tx => s.authority.snapshotForDeletion(tx, s.descriptor.clip_id))
  expect(snapshot.erasable).toBe(false)
  for (const attempt of s.db.transactionAttempts) {
    expect(attempt.count).toBeLessThanOrEqual(90)
    expect(attempt.calls.every(call => ['get', 'set', 'update', 'remove'].includes(call.method))).toBe(true)
  }
}
async function close(s) {
  await s.db.runTransaction(async tx => {
    await s.authority.closeForDeletion(tx, s.identity)
    await tx.collection('owner_clip_tombstones').doc(s.descriptor.clip_id).set({ data: { clip_id: s.descriptor.clip_id } })
  })
}

test('actual authenticated receipt route accepts retired evidence and grants no provider rights', async () => {
  const s = await saved(); const uploadRequest = request()
  await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  const value = receipt(uploadRequest); const before = structuredClone(s.db.rows)
  const result = await handler(s)(signed(value, `/v1/sync/saved-clips/${s.job.job_id}/upload-receipt`))
  expect(result.statusCode).toBe(200)
  expect(JSON.parse(result.body).data).toEqual({ contract: 'managed_upload_receipt_ack_v1', upload_request: uploadRequest, receipt_sha256: digest(value), received_at_utc: expect.any(String), erasable: false })
  const { upload_terminal_receipts, ...original } = s.db.rows
  const { upload_terminal_receipts: _, ...expected } = before
  expect(original).toEqual(expected); expect(receiptPages(s)).toHaveLength(1)
  expect(s.storage.authorizeRegisteredUpload).not.toHaveBeenCalled(); expect(s.storage.copyRegisteredObject).not.toHaveBeenCalled()
  await invariant(s)
})

test('repository accepts exact immutable receipt, replays original ACK and conflicts on changed evidence', async () => {
  const s = await saved(); const uploadRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest })
  const value = receipt(uploadRequest, issued.upload_attempt)
  const ack = await record(s, value); const before = structuredClone(s.db.rows); const start = s.db.transactionAttempts.length
  expect(await record(s, structuredClone(value))).toEqual(ack)
  expect(s.db.rows).toEqual(before); expect(s.db.transactionAttempts.slice(start).flatMap(a => a.calls).every(c => c.method === 'get')).toBe(true)
  await expect(record(s, { ...value, evidence: { ...value.evidence, local_binding_sha256: 'b'.repeat(64) } })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before); await invariant(s)
})

test('public receipt/ACK codec rejects extra fields, invalid dates and lossy microsecond intervals', () => {
  const { UPLOAD_RECEIPT_CONTRACT, UPLOAD_RECEIPT_ACK_CONTRACT, validUploadReceipt, validUploadReceiptAck } = require('../../cloudfunctions/shared/upload-terminal-receipts')
  expect(UPLOAD_RECEIPT_CONTRACT).toBe('managed_upload_receipt_v1'); expect(UPLOAD_RECEIPT_ACK_CONTRACT).toBe('managed_upload_receipt_ack_v1')
  const uploadRequest = request(); const attempt = { contract: 'managed_upload_attempt_v1', operation_id: crypto.randomUUID(), sequence: 1 }
  const success = receipt(uploadRequest, attempt); const retired = receipt(uploadRequest)
  expect(validUploadReceipt(success)).toBe(true); expect(validUploadReceipt(retired)).toBe(true)
  expect(validUploadReceipt({ ...retired, upload_attempt: attempt })).toBe(true)
  for (const value of [null, [], {}, { ...success, extra: true }, { ...success, contract: 'wrong' }, { ...success, upload_attempt: null },
    { ...success, upload_request: { ...uploadRequest, request_id: uploadRequest.request_id.toUpperCase() } },
    { ...success, upload_request: { ...uploadRequest, request_id: '00000000-0000-1000-8000-000000000000' } },
    { ...success, upload_request: { ...uploadRequest, extra: 1 } },
    { ...success, upload_attempt: { ...attempt, sequence: '1' } }, { ...success, upload_attempt: { ...attempt, sequence: 65 } },
    { ...success, upload_attempt: { ...attempt, operation_id: 'bad' } }, { ...success, upload_attempt: { ...attempt, extra: 1 } },
    { ...success, evidence: { ...success.evidence, kind: 'unknown' } }, { ...retired, evidence: { ...retired.evidence, started_at_utc: '2024-02-29T23:59:59.123456Z' } },
    ...[null, [], {}, { ...success.evidence, status_code: '200' }, { ...success.evidence, status_code: 201 }, { ...success.evidence, local_binding_sha256: 'A'.repeat(64) },
      { ...success.evidence, extra: 1 }, { ...success.evidence, received_at_utc: '2024-02-29T23:59:59.123455Z' },
      ...['2023-02-29T23:59:59.123456Z', '2024-04-31T23:59:59.123456Z', '2024-02-29T24:00:00.000000Z', '2024-02-29T23:59:60.123456Z', '2024-02-29T23:59:59.123Z', '2024-02-29T23:59:59.123456+00:00'].map(started_at_utc => ({ ...success.evidence, started_at_utc }))].map(evidence => ({ ...success, evidence }))]) {
    expect(validUploadReceipt(value)).toBe(false)
  }
  expect(validUploadReceipt({ ...success, evidence: { ...success.evidence, received_at_utc: success.evidence.started_at_utc } })).toBe(true)
  const ack = { contract: UPLOAD_RECEIPT_ACK_CONTRACT, upload_request: uploadRequest, receipt_sha256: digest(success), received_at_utc: '2024-02-29T23:59:59.123Z', erasable: false }
  expect(validUploadReceiptAck(ack)).toBe(true)
  for (const value of [null, [], {}, { ...ack, erasable: true }, { ...ack, erasable: 0 }, { ...ack, extra: 1 }, { ...ack, receipt_sha256: 'bad' },
    { ...ack, upload_request: { ...uploadRequest, extra: 1 } }, { ...ack, received_at_utc: '2023-02-29T23:59:59.123Z' }, { ...ack, received_at_utc: '2024-02-29T23:59:59.123456Z' }]) expect(validUploadReceiptAck(value)).toBe(false)
})

test('signed route enforces original body/HMAC, nonce, active device and table/device scope', async () => {
  const s = await saved(); const uploadRequest = request()
  await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  const value = receipt(uploadRequest); const path = `/v1/sync/saved-clips/${s.job.job_id}/upload-receipt`; const run = handler(s)
  const original = signed(value, path); const before = structuredClone(s.db.rows)
  const tampered = { ...original, body: JSON.stringify({ ...value, evidence: { ...value.evidence, local_binding_sha256: 'b'.repeat(64) } }) }
  expect(JSON.parse((await run(tampered)).body).error.code).toBe('SYNC_UNAUTHORIZED')
  const wrongSignature = signed(value, path); wrongSignature.headers['x-signature'] = '0'.repeat(64)
  expect(JSON.parse((await run(wrongSignature)).body).error.code).toBe('SYNC_UNAUTHORIZED')
  expect(JSON.parse((await handler(s, { status: 'disabled' })(signed(value, path))).body).error.code).toBe('SYNC_UNAUTHORIZED')
  for (const scope of [{ table_id: 'foreign-table' }, { device_id: 'foreign-device' }]) {
    expect(JSON.parse((await handler(s, scope)(signed(value, path))).body).error.code).toBe('SYNC_STORAGE_UNAVAILABLE')
  }
  const foreignJob = signed(value, '/v1/sync/saved-clips/foreign-job/upload-receipt')
  expect(JSON.parse((await run(foreignJob)).body).error.code).toBe('SYNC_CONFLICT')
  expect(s.db.rows).toEqual(before)
  expect((await run(original)).statusCode).toBe(200)
  const savedRows = structuredClone(s.db.rows)
  expect(JSON.parse((await run(original)).body).error.code).toBe('SYNC_REPLAYED')
  expect((await run(signed(value, path))).statusCode).toBe(200); expect(s.db.rows).toEqual(savedRows)
  await invariant(s)
})

test.each([null, {}, [], { ...receipt(request()), extra: true }, { ...receipt(request()), upload_attempt: undefined },
  { ...receipt(request()), evidence: { kind: 'started', local_binding_sha256: 'a'.repeat(64) } }])('invalid actual route/repository receipt is rejected before storage: %p', async value => {
  const s = await saved(); const before = structuredClone(s.db.rows); const count = s.db.transactionAttempts.length
  const result = await handler(s)(signed(value, `/v1/sync/saved-clips/${s.job.job_id}/upload-receipt`))
  expect(JSON.parse(result.body).error.code).toBe('SYNC_INVALID')
  await expect(record(s, value)).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  expect(s.db.rows).toEqual(before); expect(s.db.transactionAttempts).toHaveLength(count)
})

test('missing request mapping is retryable without allocating or consulting mutable jobs/admission', async () => {
  const s = await saved(); const value = receipt(request()); const before = structuredClone(s.db.rows); const start = s.db.transactionAttempts.length
  await expect(record(s, value)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before)
  expect(s.db.transactionAttempts.slice(start).flatMap(a => a.calls).map(c => [c.collection, c.method])).toEqual([['managed_upload_requests', 'get']])
  expect(s.storage.authorizeRegisteredUpload).not.toHaveBeenCalled(); await invariant(s)
})

test('null retirement remains immutable before late external issuance, and success requires external_issued', async () => {
  const s = await saved(); const uploadRequest = request()
  const { ticket } = await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  const attempt = { contract: 'managed_upload_attempt_v1', operation_id: ticket.operation_id, sequence: ticket.sequence }
  const value = receipt(uploadRequest); const success = receipt(uploadRequest, attempt)
  await expect(record(s, success)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  const ack = await record(s, value)
  await s.repo.recordTrackedResult({ ticket, outcome: 'external_issued', objects: [{ kind: 'staging', cloud_path: ticket.target.cloud_path, file_id: 'cloud://test/late' }], reason: null })
  expect(await record(s, value)).toEqual(ack)
  expect(receiptPages(s)[0].entries[0].receipt.upload_attempt).toBeNull()
  expect(receiptPages(s)[0].entries[0].request_binding.ticket).toEqual(ticket)
  await expect(record(s, success)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(record(s, { ...value, upload_attempt: attempt })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(s.storage.authorizeRegisteredUpload).not.toHaveBeenCalled(); await invariant(s)
})

test.each(['null', 'no_write', 'unknown'])('HTTP200 refuses original outcome %s while nonnull retirement still records evidence', async outcome => {
  const s = await saved(); const uploadRequest = request(); const { ticket } = await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  if (outcome !== 'null') await s.repo.recordTrackedResult({ ticket, outcome, objects: [], reason: outcome === 'no_write' ? 'not_started' : 'result_lost' })
  const attempt = { contract: 'managed_upload_attempt_v1', operation_id: ticket.operation_id, sequence: ticket.sequence }
  await expect(record(s, receipt(uploadRequest, attempt))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  const value = { ...receipt(uploadRequest), upload_attempt: attempt }
  const ack = await record(s, value); expect(ack.erasable).toBe(false); await invariant(s)
})

test.each(['attempt-id', 'attempt-sequence', 'ticket-epoch', 'ticket-target', 'same-job', 'identity-hash', 'scope', 'copy'])('foreign %s original binding/attempt conflicts without changing any records', async change => {
  const s = await saved(); const uploadRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest })
  let value = receipt(uploadRequest, issued.upload_attempt); const original = binding(s)
  if (change === 'attempt-id') value.upload_attempt.operation_id = crypto.randomUUID()
  if (change === 'attempt-sequence') value.upload_attempt.sequence = 2
  if (change === 'ticket-epoch') original.ticket.epoch = crypto.randomUUID()
  if (change === 'ticket-target') original.ticket.target.cloud_path = 'foreign'
  if (change === 'same-job') Object.values(s.db.rows.media_storage_operations)[0].entries[0].operation.ingest_job_id = 'foreign-job'
  if (change === 'identity-hash') original.ticket.identity_sha256 = 'f'.repeat(64)
  if (change === 'scope') original.table_id = 'foreign-table'
  if (change === 'copy') { const copy = await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt: issued.upload_attempt }); original.ticket = copy.ticket }
  const before = structuredClone(s.db.rows)
  await expect(record(s, value)).rejects.toMatchObject({ code: change === 'scope' ? 'SYNC_STORAGE_UNAVAILABLE' : 'SYNC_CONFLICT' })
  expect(s.db.rows).toEqual(before); expect(s.storage.authorizeRegisteredUpload).toHaveBeenCalledTimes(1); expect(s.storage.copyRegisteredObject).not.toHaveBeenCalled()
})

test.each(['map', 'root', 'operation-page', 'alias'])('corrupt or missing original %s dependency stays unavailable', async change => {
  const s = await saved(); const uploadRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest })
  if (change === 'map') binding(s).descriptor_sha256 = 'bad'
  if (change === 'root') s.db.rows.clip_storage_authorities[s.descriptor.clip_id].revision++
  if (change === 'operation-page') delete s.db.rows.media_storage_operations[Object.keys(s.db.rows.media_storage_operations)[0]]
  if (change === 'alias') delete s.db.rows.media_storage_objects[Object.keys(s.db.rows.media_storage_objects)[0]]
  const before = structuredClone(s.db.rows)
  await expect(record(s, receipt(uploadRequest, issued.upload_attempt))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before)
})

test('success and retirement replay survive publication, owner deletion, closed root and mutable compaction', async () => {
  const { requestOwnerClipDeletion, createCloudOwnerDeleteRepository } = require('../../cloudfunctions/shared/owner-delete')
  const s = await saved(); const uploadRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest })
  const value = receipt(uploadRequest, issued.upload_attempt); const ack = await record(s, value)
  await completeSavedClipJob({ ...args(s), uploadAttempt: issued.upload_attempt, now })
  await requestOwnerClipDeletion({ clipId: s.descriptor.clip_id, requestId: 'owner-request', userId: 'user-a', repo: createCloudOwnerDeleteRepository(s.db), now })
  delete s.db.rows.saved_clip_sync_jobs[s.job.job_id]; delete s.db.rows.saved_clip_sync_idempotency[documentId(s.descriptor.clip_id)]
  const before = structuredClone(s.db.rows); const start = s.db.transactionAttempts.length
  expect(await record(s, value)).toEqual(ack); expect(s.db.rows).toEqual(before)
  const calls = s.db.transactionAttempts.slice(start).flatMap(a => a.calls)
  expect(calls.every(c => ['managed_upload_requests', 'clip_storage_authorities', 'media_storage_operations', 'media_storage_objects', 'upload_terminal_receipts'].includes(c.collection) && c.method === 'get')).toBe(true)
  await invariant(s)
})

test('first late receipt works after root close, tombstone and corrupt/compacted mutable state', async () => {
  const s = await saved(); const uploadRequest = request(); await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  await close(s); delete s.db.rows.saved_clip_sync_jobs[s.job.job_id]; s.db.rows.clips[s.descriptor.clip_id] = { corrupt: true }
  const before = structuredClone(s.db.rows); const value = receipt(uploadRequest)
  expect((await record(s, value)).erasable).toBe(false)
  const { upload_terminal_receipts, ...original } = s.db.rows; const { upload_terminal_receipts: _, ...expected } = before
  expect(original).toEqual(expected); await invariant(s)
})

test.each(['contract', 'extra', 'clip', 'hash', 'epoch', 'index', 'length', 'all-null', 'tail', 'oversize', 'entry-extra', 'sequence', 'entry-binding-extra', 'entry-scope', 'entry-job', 'entry-hash', 'entry-ticket', 'entry-copy', 'receipt-hash', 'receipt-invalid', 'server-time', 'duplicate-request', 'duplicate-operation'])('corrupt existing %s evidence is unavailable, never reset', async change => {
  const s = await saved(); const firstRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest: firstRequest })
  await record(s, receipt(firstRequest, issued.upload_attempt)); const secondRequest = request(); await authorizeSavedClip({ ...args(s), uploadRequest: secondRequest })
  const page = receiptPages(s)[0]; const entry = page.entries[0]
  if (change === 'contract') page.contract = 'wrong'
  if (change === 'extra') page.extra = true
  if (change === 'clip') page.clip_id = 'foreign'
  if (change === 'hash') page.identity_sha256 = 'f'.repeat(64)
  if (change === 'epoch') page.epoch = crypto.randomUUID()
  if (change === 'index') page.page_index = 1
  if (change === 'length') page.entries.pop()
  if (change === 'all-null') page.entries.fill(null)
  if (change === 'tail') page.entries[2] = structuredClone(entry)
  if (change === 'oversize') { page.entries[0].request_binding.identity.owner_user_id = 'x'.repeat(128 * 1024) }
  if (change === 'entry-extra') entry.extra = true
  if (change === 'sequence') entry.sequence = 2
  if (change === 'entry-binding-extra') entry.request_binding._id = 'metadata-must-be-stripped'
  if (change === 'entry-scope') entry.request_binding.table_id = 'foreign'
  if (change === 'entry-job') entry.request_binding.job_id = 'foreign'
  if (change === 'entry-hash') entry.request_binding.identity_sha256 = 'f'.repeat(64)
  if (change === 'entry-ticket') entry.request_binding.ticket.epoch = crypto.randomUUID()
  if (change === 'entry-copy') { const copy = await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt: issued.upload_attempt }); entry.request_binding.ticket = copy.ticket }
  if (change === 'receipt-hash') entry.receipt_sha256 = 'f'.repeat(64)
  if (change === 'receipt-invalid') { entry.receipt.evidence.status_code = 201; entry.receipt_sha256 = digest(entry.receipt) }
  if (change === 'server-time') entry.received_at_utc = '2023-02-29T00:00:00.000Z'
  if (change === 'duplicate-request' || change === 'duplicate-operation') {
    const secondBinding = Object.values(s.db.rows.managed_upload_requests)[1]
    const duplicate = { sequence: 2, request_binding: structuredClone(secondBinding), receipt: receipt(secondRequest), receipt_sha256: '', received_at_utc: entry.received_at_utc }
    if (change === 'duplicate-request') { duplicate.request_binding.request_id = firstRequest.request_id; duplicate.receipt.upload_request = firstRequest }
    if (change === 'duplicate-operation') { duplicate.request_binding.ticket = structuredClone(entry.request_binding.ticket) }
    duplicate.receipt_sha256 = digest(duplicate.receipt); page.entries[1] = duplicate
  }
  const before = structuredClone(s.db.rows)
  await expect(record(s, receipt(secondRequest))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before)
})

test.each([1, 2, 3])('unused companion page %i is read and rejected even when outside authoritative range', async index => {
  const s = await saved(); const uploadRequest = request(); await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  const root = s.db.rows.clip_storage_authorities[s.descriptor.clip_id]
  s.db.rows.upload_terminal_receipts[`page-${documentId(s.descriptor.clip_id)}-${index}`] = { contract: 'managed_upload_receipt_v1', clip_id: s.descriptor.clip_id, identity_sha256: root.identity_sha256, epoch: root.epoch, page_index: index, entries: Array(16).fill(null) }
  const before = structuredClone(s.db.rows)
  await expect(record(s, receipt(uploadRequest))).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before)
  const calls = s.db.transactionAttempts.at(-1).calls.filter(c => c.collection === 'upload_terminal_receipts')
  expect(calls.map(c => c.id)).toEqual([0, 1, 2, 3].map(i => `page-${documentId(s.descriptor.clip_id)}-${i}`))
})

test('evidence page failure rolls back atomically, hides raw error and retry appends only evidence', async () => {
  let fail = true
  const s = await saved({ dbOptions: { failWrite(name) { if (fail && name === 'upload_terminal_receipts') throw new Error('RAW_SECRET') } } })
  const uploadRequest = request(); await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  const before = structuredClone(s.db.rows); const value = receipt(uploadRequest)
  await expect(record(s, value)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE', message: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(s.db.rows).toEqual(before); fail = false
  expect((await record(s, value)).erasable).toBe(false); expect(JSON.stringify(s.db.rows)).not.toContain('RAW_SECRET'); await invariant(s)
})

test('database metadata is stripped from captured binding and replay stays byte-identical', async () => {
  const s = await saved(); const uploadRequest = request(); await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  binding(s)._id = Object.keys(s.db.rows.managed_upload_requests)[0]
  const value = receipt(uploadRequest); const ack = await record(s, value)
  expect(receiptPages(s)[0].entries[0].request_binding).not.toHaveProperty('_id')
  receiptPages(s)[0]._id = Object.keys(s.db.rows.upload_terminal_receipts)[0]
  const before = structuredClone(s.db.rows); expect(await record(s, value)).toEqual(ack); expect(s.db.rows).toEqual(before); await invariant(s)
})

test('strict N64/A64 full evidence pages, forced submission/replay retries and compaction stay doc-only <=90', async () => {
  let retryWrite = false; let retryRead = false
  const s = await saved({ dbOptions: { retryConflicts: 1, beforeCommit({ attempt }) {
    if ((retryWrite && attempt.calls.some(c => c.collection === 'upload_terminal_receipts' && c.method === 'set'))
        || (retryRead && attempt.calls.some(c => c.collection === 'upload_terminal_receipts' && c.method === 'get'))) {
      retryWrite = false; retryRead = false; const error = new Error('controlled conflict'); error.code = 'SYNTHETIC_CONFLICT'; throw error
    }
  } } })
  const values = []
  for (let index = 0; index < 64; index++) { const uploadRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest }); values.push(receipt(uploadRequest, issued.upload_attempt)) }
  const original = structuredClone(s.db.rows); const start = s.db.transactionAttempts.length
  retryWrite = true; const firstAck = await record(s, values[0])
  for (const value of values.slice(1)) await record(s, value)
  expect(receiptPages(s)).toHaveLength(4); expect(receiptPages(s).every(p => p.entries.length === 16 && p.entries.every(Boolean))).toBe(true)
  const { upload_terminal_receipts, ...unchanged } = s.db.rows; const { upload_terminal_receipts: _, ...expected } = original
  expect(unchanged).toEqual(expected)
  const before = structuredClone(s.db.rows); retryRead = true; expect(await record(s, values[0])).toEqual(firstAck); expect(s.db.rows).toEqual(before)
  const submissionAttempts = s.db.transactionAttempts.slice(start)
  const writes = submissionAttempts.filter(a => a.calls.some(c => c.collection === 'upload_terminal_receipts' && c.method === 'set'))
  const replays = submissionAttempts.filter(a => a.calls.every(c => c.method === 'get'))
  expect(writes.some(a => a.index === 1)).toBe(true); expect(replays.some(a => a.index === 1)).toBe(true)
  expect(writes.every(a => a.calls.filter(c => c.method !== 'get').length === 1)).toBe(true)
  expect(submissionAttempts.every(a => a.calls.filter(c => c.collection === 'upload_terminal_receipts' && c.method === 'get').length === 4)).toBe(true)
  await close(s); delete s.db.rows.saved_clip_sync_jobs[s.job.job_id]; const closed = structuredClone(s.db.rows)
  retryRead = true; expect(await record(s, values[0])).toEqual(firstAck); expect(s.db.rows).toEqual(closed)
  console.info('TERMINAL_RECEIPT_N64_A64_CALLBACKS', { submission: [...new Set(writes.map(a => a.count))], replay: [...new Set(replays.map(a => a.count))], closedReplay: s.db.transactionAttempts.slice(-2).map(a => a.count), max: Math.max(...submissionAttempts.map(a => a.count)) })
  expect(s.storage.authorizeRegisteredUpload).toHaveBeenCalledTimes(64); expect(s.storage.copyRegisteredObject).not.toHaveBeenCalled(); await invariant(s)
}, 20000)

test('receipt strings are strict ASCII values with no trailing newline or hidden own keys', () => {
  const { validUploadReceipt, validUploadReceiptAck } = require('../../cloudfunctions/shared/upload-terminal-receipts')
  const value = receipt(request(), { contract: 'managed_upload_attempt_v1', operation_id: crypto.randomUUID(), sequence: 1 })
  for (const invalid of [
    { ...value, upload_request: { ...value.upload_request, request_id: `${value.upload_request.request_id}\n` } },
    { ...value, upload_attempt: { ...value.upload_attempt, operation_id: `${value.upload_attempt.operation_id}\n` } },
    { ...value, evidence: { ...value.evidence, local_binding_sha256: `${value.evidence.local_binding_sha256}\n` } },
    { ...value, evidence: { ...value.evidence, received_at_utc: `${value.evidence.received_at_utc}\n` } },
    { ...value, [Symbol('extra')]: true }, { ...value, upload_attempt: { ...value.upload_attempt, [Symbol('extra')]: true } },
    { ...value, evidence: { ...value.evidence, [Symbol('extra')]: true } },
  ]) expect(validUploadReceipt(invalid)).toBe(false)
  const ack = { contract: 'managed_upload_receipt_ack_v1', upload_request: value.upload_request, receipt_sha256: digest(value), received_at_utc: '2024-02-29T23:59:59.123Z', erasable: false }
  for (const invalid of [{ ...ack, receipt_sha256: `${ack.receipt_sha256}\n` }, { ...ack, received_at_utc: `${ack.received_at_utc}\n` },
    { ...ack, upload_request: { ...ack.upload_request, request_id: `${ack.upload_request.request_id}\n` } }]) expect(validUploadReceiptAck(invalid)).toBe(false)
})

test('concurrent sparse same-page submissions preserve both slots; holes remain unknown and never change authority outcomes', async () => {
  const s = await saved(); const requests = [request(), request(), request()]
  for (const uploadRequest of requests) await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  const originals = structuredClone(s.db.rows); const values = requests.map(uploadRequest => receipt(uploadRequest))
  const [first, third] = await Promise.all([record(s, values[0]), record(s, values[2])])
  expect(receiptPages(s)).toHaveLength(1)
  expect(receiptPages(s)[0].entries.map(e => e?.sequence ?? null)).toEqual([1, null, 3, ...Array(13).fill(null)])
  expect(await record(s, values[0])).toEqual(first); expect(await record(s, values[2])).toEqual(third)
  const snapshot = await s.db.runTransaction(tx => s.authority.snapshotForDeletion(tx, s.descriptor.clip_id))
  expect(snapshot.erasable).toBe(false); expect(snapshot.complete).toBe(false)
  expect(snapshot.operations.every(op => op.outcome === null)).toBe(true)
  expect(s.db.rows.clip_storage_authorities).toEqual(originals.clip_storage_authorities)
  expect(s.db.rows.media_storage_operations).toEqual(originals.media_storage_operations)
  await record(s, values[1]); expect(receiptPages(s)[0].entries[1].sequence).toBe(2)
  expect(s.storage.authorizeRegisteredUpload).not.toHaveBeenCalled(); await invariant(s)
})

test.each(['kind', 'start', 'received', 'binding'])('changed immutable receipt %s permanently conflicts', async change => {
  const s = await saved(); const uploadRequest = request(); const issued = await authorizeSavedClip({ ...args(s), uploadRequest }); const value = receipt(uploadRequest, issued.upload_attempt)
  await record(s, value); const changed = structuredClone(value)
  if (change === 'kind') changed.evidence = { kind: 'retired_before_start', local_binding_sha256: value.evidence.local_binding_sha256 }
  if (change === 'start') changed.evidence.started_at_utc = '2024-02-29T23:59:59.123455Z'
  if (change === 'received') changed.evidence.received_at_utc = '2024-02-29T23:59:59.123458Z'
  if (change === 'binding') changed.evidence.local_binding_sha256 = 'f'.repeat(64)
  const before = structuredClone(s.db.rows)
  await expect(record(s, changed)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' }); expect(s.db.rows).toEqual(before)
})
