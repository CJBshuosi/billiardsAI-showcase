const crypto = require('crypto')
const { setup } = require('../helpers/managed-saved-flow-fixture')
const { createSavedClipJob, authorizeSavedClip } = require('../../cloudfunctions/sync-api/lib/ingest')
const { digest, documentId } = require('../../cloudfunctions/shared/shot-ownership')
const receipts = require('../../cloudfunctions/shared/upload-terminal-receipts')
const assessment = input => require('../../cloudfunctions/shared/owner-cloud-erasure').assessApplicationWriters(input)
const request = () => ({ contract: 'managed_upload_request_v1', request_id: crypto.randomUUID() })
const receipt = (uploadRequest, uploadAttempt = null) => ({ contract: 'managed_upload_receipt_v1', upload_request: uploadRequest, upload_attempt: uploadAttempt,
  evidence: uploadAttempt ? { kind: 'http200', local_binding_sha256: 'a'.repeat(64), started_at_utc: '2024-02-29T23:59:59.123456Z', received_at_utc: '2024-02-29T23:59:59.123457Z', status_code: 200 }
    : { kind: 'retired_before_start', local_binding_sha256: 'a'.repeat(64) } })
async function saved(options) {
  const s = await setup(options); s.job = await createSavedClipJob(s.input)
  s.db.rows.managed_upload_requests ||= {}; s.db.rows.upload_terminal_receipts ||= {}; s.db.setStrictTransactions(true)
  s.storage = { authorizeRegisteredUpload: async input => ({ outcome: 'external_issued', file_id: `cloud://test/${input.object_key}`,
    authorization: { method: 'PUT', url: 'https://upload.invalid', headers: {}, expires_in_seconds: 300 } }) }
  return s
}
const record = (s, value) => s.repo.recordTrackedUploadReceipt({ ...s.scope, jobId: s.job.job_id, receipt: value })
async function upload(s, terminal = 'success') {
  const uploadRequest = request()
  const issued = await authorizeSavedClip({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage: s.storage, uploadRequest })
  if (terminal) await record(s, receipt(uploadRequest, terminal === 'success' ? issued.upload_attempt : null))
  return { uploadRequest, issued }
}
const close = s => s.db.runTransaction(tx => s.authority.closeForDeletion(tx, s.identity))
const read = s => s.db.runTransaction(tx => s.authority.readWriterClosureInputs(tx, { clipId: s.descriptor.clip_id }))
const operations = s => Object.values(s.db.rows.media_storage_operations).flatMap(page => page.entries.map(e => e.operation))
function catalog(s) {
  const root = s.db.rows.clip_storage_authorities[s.descriptor.clip_id]
  return { root, operations: operations(s), pages: Array.from({ length: 4 }, (_, i) => s.db.rows.upload_terminal_receipts[receipts.receiptPageId(root.identity.clip_id, i)] ?? null) }
}
// Adversarial callers can compute hashes too. Never let stale hashes alone be
// the reason a malformed original snapshot is rejected.
function rehash(input) {
  const pages = Array(4).fill(null)
  for (const entry of input.receipt_entries) {
    const i = Math.floor((entry.sequence - 1) / 16)
    pages[i] ||= { contract: 'managed_upload_receipt_v1', clip_id: input.root.identity.clip_id,
      identity_sha256: input.root.identity_sha256, epoch: input.root.epoch, page_index: i, entries: Array(16).fill(null) }
    pages[i].entries[(entry.sequence - 1) % 16] = entry
  }
  input.receipt_page_sha256 = pages.map(digest); input.birth_sha256 = digest(input.birth)
  input.fingerprint_sha256 = digest({ root: input.root, operations: input.operations, objects: input.objects,
    receipt_page_sha256: input.receipt_page_sha256, birth_sha256: input.birth_sha256 })
  return input
}
function assertReadOnly(s, start, before) {
  expect(s.db.rows).toEqual(before)
  for (const attempt of s.db.transactionAttempts.slice(start)) {
    expect(attempt.count).toBeLessThanOrEqual(90)
    expect(attempt.calls.every(c => c.method === 'get')).toBe(true)
    expect(attempt.calls.filter(c => c.collection === 'upload_terminal_receipts')).toHaveLength(4)
    expect(attempt.calls.filter(c => c.collection === 'managed_upload_requests')).toHaveLength(0)
  }
}

test('append refuses extra page array elements instead of silently discarding evidence', async () => {
  const s = await saved(); const { uploadRequest } = await upload(s, null); const input = catalog(s)
  const requestBinding = Object.values(s.db.rows.managed_upload_requests)[0]
  expect(() => receipts.prepareUploadReceipt({ ...input, pages: [...input.pages, null], requestBinding, receipt: receipt(uploadRequest), receivedAt: '2026-11-01T10:00:00.000Z' })).toThrow()
})

test('receipt append and replay preserve original ACK serialization for reversed request key order', async () => {
  const s = await saved(); const uploadRequest = { request_id: crypto.randomUUID(), contract: 'managed_upload_request_v1' }
  await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  const value = receipt(uploadRequest)
  const first = await record(s, value); const replay = await record(s, value)
  expect(JSON.stringify(replay)).toBe(JSON.stringify(first))
})

test('closed full inventory preserves successful, retired, never-started and unpublished copy targets', async () => {
  const s = await saved(); const first = await upload(s); await upload(s, 'retired')
  const never = await s.authority.beginOperation(s.db, { identity: s.identity, ingestJobId: s.job.job_id, kind: 'upload_authorization' })
  await s.authority.recordOperationResult({ ticket: never, outcome: 'no_write', objects: [], reason: 'not_started' })
  const { ticket: copy } = await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt: first.issued.upload_attempt })
  await s.repo.recordTrackedResult({ ticket: copy, outcome: 'server_succeeded', objects: [{ ...copy.target, file_id: 'cloud://test/copy' }].map(({ object_id, ...o }) => o), reason: null })
  await close(s)
  const before = structuredClone(s.db.rows); const start = s.db.transactionAttempts.length
  const input = await read(s); const result = assessment(input)
  expect(Object.keys(result).sort()).toEqual(['contract', 'identity', 'identity_sha256', 'epoch', 'high_water', 'authority_revision', 'inputs_sha256', 'status', 'targets', 'blockers', 'erasable'].sort())
  expect(result).toMatchObject({ contract: 'owner_application_writers_v1', status: 'closed', high_water: 4, erasable: false, blockers: [], inputs_sha256: input.fingerprint_sha256 })
  expect(result.targets.map(t => [t.sequence, t.writer_state, t.receipt_sha256 === null])).toEqual([[1, 'upload_succeeded', false], [2, 'upload_retired', false], [3, 'no_write', true], [4, 'copy_succeeded', true]])
  expect(Object.keys(result.targets[0]).sort()).toEqual(['sequence', 'operation_id', 'object_id', 'kind', 'cloud_path', 'file_id', 'writer_state', 'receipt_sha256'].sort())
  expect(result.targets[2].file_id).toBeNull()
  expect(result.targets.map(t => t.object_id)).toEqual(input.root.object_ids)
  assertReadOnly(s, start, before)
  const conservative = await s.db.runTransaction(tx => s.authority.snapshotForDeletion(tx, s.descriptor.clip_id))
  expect(conservative).toMatchObject({ complete: false, erasable: false })
})

test.each([null, 'unknown', 'no_write'])('retirement never conceals unresolved issuer outcome %p', async outcome => {
  const s = await saved(); const uploadRequest = request()
  const { ticket } = await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
  if (outcome) await s.repo.recordTrackedResult({ ticket, outcome, objects: [], reason: outcome === 'unknown' ? 'result_lost' : 'provider_error' })
  await record(s, receipt(uploadRequest)); await close(s)
  expect(assessment(await read(s))).toMatchObject({ status: 'blocked', targets: [{ writer_state: 'unresolved', receipt_sha256: null }], blockers: [{ sequence: 1, code: 'writer_unresolved' }], erasable: false })
})

test('missing receipt blocks and receipt-only append changes fingerprint without changing root revision; replay does not', async () => {
  const s = await saved(); const { uploadRequest, issued } = await upload(s, null); await close(s)
  const missing = await read(s)
  expect(assessment(missing)).toMatchObject({ status: 'blocked', blockers: [{ sequence: 1, code: 'upload_receipt_missing' }] })
  const value = receipt(uploadRequest, issued.upload_attempt); const ack = await record(s, value); const present = await read(s)
  expect(present.root.revision).toBe(missing.root.revision); expect(present.fingerprint_sha256).not.toBe(missing.fingerprint_sha256)
  expect(assessment(present).status).toBe('closed'); expect(await record(s, value)).toEqual(ack)
  expect((await read(s)).fingerprint_sha256).toBe(present.fingerprint_sha256)
})

test.each(['birth-missing', 'birth-unmarked', 'birth-corrupt', 'birth-scope', 'root-missing', 'open', 'root-revision', 'operation-missing', 'operation-tail', 'alias-missing', 'alias-corrupt', 'receipt-tail', 'receipt-binding', 'receipt-duplicate', 'no-write-http200'])('invalid original %s fails with authority convention and no writes', async change => {
  const s = await saved(); await upload(s); await upload(s); await close(s)
  const root = s.db.rows.clip_storage_authorities[s.descriptor.clip_id]
  const birth = s.db.rows.managed_session_births[s.identity.session_id]
  const page = Object.values(s.db.rows.media_storage_operations)[0]
  const evidence = Object.values(s.db.rows.upload_terminal_receipts)[0]
  if (change === 'birth-missing') delete s.db.rows.managed_session_births[s.identity.session_id]
  if (change === 'birth-unmarked') delete birth.media_admission_contract
  if (change === 'birth-corrupt') birth.opened_at = 'bad'
  if (change === 'birth-scope') birth.device_id = 'foreign'
  if (change === 'root-missing') delete s.db.rows.clip_storage_authorities[s.descriptor.clip_id]
  if (change === 'open') { root.admission = 'open'; root.revision-- }
  if (change === 'root-revision') root.revision++
  if (change === 'operation-missing') delete s.db.rows.media_storage_operations[Object.keys(s.db.rows.media_storage_operations)[0]]
  if (change === 'operation-tail') s.db.rows.media_storage_operations[`page-${documentId(s.descriptor.clip_id)}-3`] = structuredClone(page)
  if (change === 'alias-missing') delete s.db.rows.media_storage_objects[Object.keys(s.db.rows.media_storage_objects)[0]]
  if (change === 'alias-corrupt') Object.values(s.db.rows.media_storage_objects)[0].file_id = 'cloud://foreign'
  if (change === 'receipt-tail') evidence.entries[3] = structuredClone(evidence.entries[0])
  if (change === 'receipt-binding') evidence.entries[0].request_binding.device_id = 'foreign'
  if (change === 'receipt-duplicate') {
    evidence.entries[1].request_binding.request_id = evidence.entries[0].request_binding.request_id
    evidence.entries[1].receipt.upload_request = structuredClone(evidence.entries[0].receipt.upload_request)
    evidence.entries[1].receipt_sha256 = digest(evidence.entries[1].receipt)
  }
  if (change === 'no-write-http200') {
    Object.assign(page.entries[0].operation, { outcome: 'no_write', objects: [], reason: 'not_started' })
    Object.assign(page.entries[0].object, { file_id: null, revision: 1 })
  }
  const before = structuredClone(s.db.rows)
  await expect(read(s)).rejects.toMatchObject({ code: change === 'birth-scope' ? 'STORAGE_AUTHORITY_CONFLICT' : 'STORAGE_AUTHORITY_INCOMPLETE' })
  expect(s.db.rows).toEqual(before)
})

test('unmarked legacy saved media never acquires writer closure provenance', async () => {
  const s = await saved({ marked: false })
  await expect(read(s)).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})

test('catalog returns ordered detached entries and exactly four metadata-free page digests', async () => {
  const s = await saved(); await upload(s); await upload(s, 'retired')
  const input = catalog(s); const expected = input.pages.map(digest)
  const result = receipts.readUploadReceiptCatalog(input)
  expect(result.entries.map(e => e.sequence)).toEqual([1, 2]); expect(result.page_sha256).toEqual(expected)
  input.pages[0]._id = 'metadata'
  expect(receipts.readUploadReceiptCatalog(input)).toEqual(result)
  result.entries[0].receipt.evidence.kind = 'forged'
  expect(receipts.readUploadReceiptCatalog(input).entries[0].receipt.evidence.kind).toBe('http200')
})

test.each(['short', 'extra', 'sparse', 'sparse-entries'])('catalog rejects %s arrays at the pure boundary', async change => {
  const s = await saved(); await upload(s); const input = catalog(s)
  if (change === 'short') input.pages.pop()
  if (change === 'extra') input.pages.push(null)
  if (change === 'sparse') delete input.pages[3]
  if (change === 'sparse-entries') delete input.pages[0].entries[15]
  expect(() => receipts.readUploadReceiptCatalog(input)).toThrow(expect.objectContaining({ code: 'INCOMPLETE' }))
})

test('metadata is excluded, original late facts are bound, and returned snapshots are detached', async () => {
  const s = await saved(); await upload(s); await close(s)
  const original = await read(s)
  s.db.rows.clip_storage_authorities[s.descriptor.clip_id]._id = 'root-meta'
  s.db.rows.managed_session_births[s.identity.session_id]._id = 'birth-meta'
  for (const page of Object.values(s.db.rows.media_storage_operations)) page._id = 'page-meta'
  for (const alias of Object.values(s.db.rows.media_storage_objects)) alias._id = 'alias-meta'
  for (const page of Object.values(s.db.rows.upload_terminal_receipts)) page._id = 'receipt-meta'
  expect(await read(s)).toEqual(original)
  const output = assessment(original); output.identity.owner_user_id = 'other'; output.targets[0].file_id = 'changed'
  expect(assessment(original).identity.owner_user_id).toBe('user-a')
  original.operations[0].objects[0].file_id = 'changed'
  expect(() => assessment(original)).toThrow(expect.objectContaining({ code: 'OWNER_CLOUD_ERASURE_INCOMPLETE' }))
  const birth = s.db.rows.managed_session_births[s.identity.session_id]; birth.opened_at = '2026-11-01T09:59:00.000Z'
  expect((await read(s)).fingerprint_sha256).not.toBe(original.fingerprint_sha256)
})

test.each(['fingerprint', 'birth-digest', 'birth', 'page-digest', 'root', 'operation', 'object', 'entries', 'extra', 'null'])('pure assessment rejects forged %s despite self-supplied hashes', async change => {
  const s = await saved(); await upload(s); await close(s); let input = await read(s)
  if (change === 'fingerprint') input.fingerprint_sha256 = '0'.repeat(64)
  if (change === 'birth-digest') input.birth_sha256 = '0'.repeat(64)
  if (change === 'birth') { input.birth.device_id = 'other'; input.birth_sha256 = digest(input.birth) }
  if (change === 'page-digest') input.receipt_page_sha256[0] = '0'.repeat(64)
  if (change === 'root') input.root.revision++
  if (change === 'operation') input.operations[0].target.cloud_path = 'changed'
  if (change === 'object') input.objects[0].file_id = 'cloud://other'
  if (change === 'entries') input.receipt_entries[0].request_binding.device_id = 'other'
  if (change === 'extra') input.extra = true
  if (change === 'null') input = null
  let error
  try { assessment(input) } catch (value) { error = value }
  expect(error).toMatchObject({ name: 'OwnerCloudErasureError', code: expect.stringMatching(/^OWNER_CLOUD_ERASURE_(INVALID|INCOMPLETE)$/) })
})

test.each(['root-revision', 'root-open', 'identity', 'operation', 'object', 'receipt', 'birth', 'symbol-birth', 'symbol-operation', 'array-extra', 'sparse-operations'])('recomputed hashes cannot bless malformed %s', async change => {
  const s = await saved(); await upload(s); await close(s); const input = await read(s)
  if (change === 'root-revision') input.root.revision++
  if (change === 'root-open') { input.root.admission = 'open'; input.root.revision-- }
  if (change === 'identity') input.root.identity.owner_user_id = 'foreign'
  if (change === 'operation') input.operations[0].outcome = 'server_succeeded'
  if (change === 'object') input.objects[0].file_id = 'cloud://foreign'
  if (change === 'receipt') { input.receipt_entries[0].request_binding.device_id = 'foreign' }
  if (change === 'birth') input.birth.device_id = 'foreign'
  if (change === 'symbol-birth') input.birth[Symbol('extra')] = true
  if (change === 'symbol-operation') input.operations[0][Symbol('extra')] = true
  if (change === 'array-extra') input.operations.extra = true
  if (change === 'sparse-operations') delete input.operations[0]
  rehash(input)
  expect(() => assessment(input)).toThrow(expect.objectContaining({ code: 'OWNER_CLOUD_ERASURE_INCOMPLETE' }))
})

test.each(['root', 'operation', 'object', 'birth'])('coherent original %s mutation changes the full fingerprint', async change => {
  const s = await saved(); await upload(s); await close(s); const before = await read(s)
  const page = Object.values(s.db.rows.media_storage_operations)[0]
  if (change === 'root') s.db.rows.clip_storage_authorities[s.descriptor.clip_id].created_at = '2026-11-01T09:59:00.000Z'
  if (change === 'operation') page.entries[0].operation.created_at = '2026-11-01T09:59:00.000Z'
  if (change === 'object') page.entries[0].object.state = 'delete_reserved'
  if (change === 'birth') s.db.rows.managed_session_births[s.identity.session_id].opened_at = '2026-11-01T09:59:00.000Z'
  const after = await read(s)
  expect(after.fingerprint_sha256).not.toBe(before.fingerprint_sha256)
  expect(assessment(after).status).toBe('closed')
})

test('strict N64/A64 late copy, receipt pages and forced read retry fit every doc-only callback', async () => {
  let retryRead = false
  const s = await saved({ dbOptions: { retryConflicts: 1, beforeCommit({ attempt }) {
    if (retryRead && attempt.calls.some(c => c.collection === 'upload_terminal_receipts')) { retryRead = false; const error = new Error('controlled conflict'); error.code = 'SYNTHETIC_CONFLICT'; throw error }
  } } })
  const first = await upload(s)
  for (let i = 1; i < 63; i++) await upload(s, i % 2 ? 'retired' : 'success')
  const { ticket } = await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt: first.issued.upload_attempt })
  await close(s)
  const blocked = assessment(await read(s)); expect(blocked.targets).toHaveLength(64)
  expect(blocked.blockers).toEqual([{ sequence: 64, code: 'writer_unresolved' }])
  await s.repo.recordTrackedResult({ ticket, outcome: 'server_succeeded', objects: [{ kind: ticket.target.kind, cloud_path: ticket.target.cloud_path, file_id: 'cloud://test/late-copy' }], reason: null })
  const before = structuredClone(s.db.rows); const start = s.db.transactionAttempts.length; retryRead = true
  const input = await read(s); const result = assessment(input)
  expect(result.status).toBe('closed'); expect(result.targets).toHaveLength(64); expect(result.inputs_sha256).not.toBe(blocked.inputs_sha256)
  expect(input.receipt_entries).toHaveLength(63); expect(Object.keys(s.db.rows.media_storage_objects)).toHaveLength(64)
  expect(s.db.transactionAttempts.slice(start).map(a => a.index)).toEqual([0, 1]); assertReadOnly(s, start, before)
  console.info('OWNER_WRITER_N64_A64_CALLBACKS', s.db.transactionAttempts.slice(start).map(a => a.count))
}, 30000)
