const { managedMediaBirthFixture, sessionId, input } = require('../helpers/managed-media-admission-fixture')
const { privateClip } = require('../helpers/private-clips')
const { documentId } = require('../../cloudfunctions/shared/shot-ownership')
const { makeStorageIdentity, createCloudMediaStorageAuthority } = require('../../cloudfunctions/shared/media-storage-authority')
const { makeManagedUploadAttempt } = require('../../cloudfunctions/shared/managed-upload-attempt')
const pageId = (clip, index) => `page-${documentId(clip)}-${index}`
const clone = value => structuredClone(value)
const attempt = ticket => ({ contract: 'managed_upload_attempt_v1', operation_id: ticket.operation_id, sequence: ticket.sequence })
const observation = ticket => ({ kind: ticket.target.kind, cloud_path: ticket.target.cloud_path, file_id: `cloud://controlled/${ticket.operation_id}` })
const entry = (s, sequence) => s.db.rows.media_storage_operations[pageId(s.identity.clip_id, Math.floor((sequence - 1) / 16))].entries[(sequence - 1) % 16]
const count = s => s.db.transactionAttempts.at(-1).count

async function setup(options = {}) {
  const { db, admin } = managedMediaBirthFixture({}, options)
  await admin.createManagedSession(input)
  let next = 0; let providerCalls = 0
  db.storage = new Proxy({}, { get() { providerCalls++; throw new Error('provider forbidden in authority') } })
  const authority = createCloudMediaStorageAuthority(db, { clock: () => Date.parse('2026-11-01T11:00:00.000Z'),
    idFactory: () => `00000000-0000-4000-8000-${String(++next).padStart(12, '0')}` })
  const identity = makeStorageIdentity(privateClip({ session_id: sessionId }), { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })
  const tx = work => db.runTransaction(work)
  await tx(t => authority.enroll(t, identity))
  const begin = (change = {}) => tx(t => authority.beginOperation(t, { identity, ingestJobId: 'ingest-1', kind: 'upload_authorization', ...change }))
  const record = (ticket, outcome, objects = [], reason = null) => authority.recordOperationResult({ ticket, outcome, objects, reason })
  const copy = (ticket, change = {}) => tx(t => authority.beginCopyForUpload(t, { identity, ingestJobId: 'ingest-1', uploadAttempt: attempt(ticket), ...change }))
  const register = (copyTicket, change = {}) => tx(t => authority.registerCopyReferences(t, { identity, ingestJobId: 'ingest-1', copyTicket, ...change }))
  const close = () => tx(t => authority.closeForDeletion(t, identity))
  return { db, authority, identity, tx, begin, record, copy, register, close, providerCalls: () => providerCalls }
}
async function issued(s, change) { const ticket = await s.begin(change); await s.record(ticket, 'external_issued', [observation(ticket)]); return ticket }
async function copied(s) { const upload = await issued(s); const copy = await s.copy(upload); await s.record(copy.ticket, 'server_succeeded', [observation(copy.ticket)]); return { upload, ...copy } }
async function unchanged(s, work, code) {
  const before = clone(s.db.rows)
  await expect(work()).rejects.toMatchObject({ name: 'MediaStorageAuthorityError', ...(code ? { code: `STORAGE_AUTHORITY_${code}` } : {}) })
  expect(s.db.rows).toEqual(before); expect(s.providerCalls()).toBe(0)
}

test('correlates the earlier actually uploaded grant and atomically references its observed staging and copied output', async () => {
  const s = await setup(); const upload = await issued(s); await issued(s)
  expect(makeManagedUploadAttempt(upload)).toEqual({ contract: 'managed_upload_attempt_v1',
    operation_id: '00000000-0000-4000-8000-000000000002', sequence: 1 })
  const { ticket, input: selected } = await s.copy(upload)
  expect(selected).toEqual({ object_id: upload.target.object_id, ...observation(upload) })
  expect(ticket).toMatchObject({ sequence: 3, epoch: upload.epoch, identity_sha256: upload.identity_sha256,
    target: { kind: 'published' } })
  expect(entry(s, 3).operation).toMatchObject({ kind: 'server_copy', input_object_id: upload.target.object_id, ingest_job_id: 'ingest-1', outcome: null })
  await s.record(ticket, 'server_succeeded', [observation(ticket)])
  expect(await s.register(ticket)).toEqual({ file_id: observation(ticket).file_id, staging_file_id: observation(upload).file_id,
    objects: [observation(upload), observation(ticket)] })
  expect(entry(s, 1).object.references).toEqual([{ clip_id: s.identity.clip_id, identity_sha256: upload.identity_sha256 }])
  expect(entry(s, 3).object.references).toEqual(entry(s, 1).object.references)
  expect(entry(s, 2).object.references).toEqual([])
  const before = clone(s.db.rows); await s.register(ticket); expect(s.db.rows).toEqual(before)
  expect((await s.tx(t => s.authority.snapshotForDeletion(t, s.identity.clip_id))).reasons).toEqual(['unresolved_operation:1', 'unresolved_operation:2'])
  expect(s.providerCalls()).toBe(0)
})

test.each(['wrong-job', 'wrong-sequence', 'wrong-uuid', 'foreign-identity', 'foreign-clip', 'extra-argument', 'expanded-attempt', 'copy-as-upload'])(
  'copy correlation refuses %s without reserving a target', async bad => {
    const s = await setup(); const upload = await issued(s); const other = await issued(s)
    const change = {}
    if (bad === 'wrong-job') change.ingestJobId = 'ingest-other'
    if (bad === 'wrong-sequence') change.uploadAttempt = { ...attempt(upload), sequence: other.sequence }
    if (bad === 'wrong-uuid') change.uploadAttempt = { ...attempt(upload), operation_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa' }
    if (bad === 'foreign-identity') change.identity = { ...s.identity, byte_size: s.identity.byte_size + 1 }
    if (bad === 'foreign-clip') change.identity = { ...s.identity, clip_id: 'foreign' }
    if (bad === 'extra-argument') change.file_id = observation(other).file_id
    if (bad === 'expanded-attempt') change.uploadAttempt = { ...attempt(upload), file_id: observation(other).file_id }
    if (bad === 'copy-as-upload') change.uploadAttempt = attempt((await s.copy(upload)).ticket)
    await unchanged(s, () => s.copy(upload, change))
  })

test.each(['unobserved', 'unknown-empty', 'unknown-observed', 'no-write', 'inactive'])(
  'copy refuses %s upload evidence', async state => {
    const s = await setup(); const upload = await s.begin()
    if (state.startsWith('unknown')) await s.record(upload, 'unknown', state === 'unknown-observed' ? [observation(upload)] : [], 'provider_error')
    if (state === 'no-write') await s.record(upload, 'no_write', [], 'not_started')
    if (state === 'inactive') { await s.record(upload, 'external_issued', [observation(upload)]); entry(s, 1).object.state = 'deleted' }
    await unchanged(s, () => s.copy(upload))
  })

test.each(['wrong-job', 'wrong-uuid', 'wrong-sequence', 'epoch', 'hash', 'target', 'identity', 'upload-ticket', 'extra-ticket', 'extra-argument'])(
  'publication refuses %s instead of trusting caller-selected output', async bad => {
    const s = await setup(); const { upload, ticket } = await copied(s); const change = {}; let altered = clone(ticket)
    if (bad === 'wrong-job') change.ingestJobId = 'other'
    if (bad === 'wrong-uuid') altered.operation_id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    if (bad === 'wrong-sequence') altered.sequence = upload.sequence
    if (bad === 'epoch') altered.epoch = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    if (bad === 'hash') altered.identity_sha256 = 'f'.repeat(64)
    if (bad === 'target') altered.target = clone(upload.target)
    if (bad === 'identity') change.identity = { ...s.identity, device_id: 'other' }
    if (bad === 'upload-ticket') altered = upload
    if (bad === 'extra-ticket') altered.file_id = observation(ticket).file_id
    if (bad === 'extra-argument') change.file_id = observation(ticket).file_id
    await unchanged(s, () => s.register(altered, change))
  })

test.each(['unobserved', 'unknown-empty', 'unknown-observed', 'no-write', 'inactive-output', 'inactive-input', 'foreign-input-job', 'unknown-input'])(
  'publication refuses %s even with a reserved copy target', async state => {
    const s = await setup(); const upload = await issued(s); const { ticket } = await s.copy(upload)
    if (state.startsWith('unknown-') && state !== 'unknown-input') await s.record(ticket, 'unknown', state === 'unknown-observed' ? [observation(ticket)] : [], 'provider_error')
    else if (state === 'no-write') await s.record(ticket, 'no_write', [], 'not_started')
    else if (state !== 'unobserved') await s.record(ticket, 'server_succeeded', [observation(ticket)])
    if (state === 'inactive-output') entry(s, 2).object.state = 'delete_reserved'
    if (state === 'inactive-input') entry(s, 1).object.state = 'deleted'
    if (state === 'foreign-input-job') entry(s, 1).operation.ingest_job_id = 'foreign'
    if (state === 'unknown-input') { entry(s, 1).operation.outcome = 'unknown'; entry(s, 1).operation.reason = 'provider_error' }
    await unchanged(s, () => s.register(ticket))
  })

test.each(['missing-page', 'tail-page', 'missing-input-alias', 'foreign-input-alias', 'missing-output-alias', 'wrong-input', 'root-revision'])(
  'complete inventory rejects %s before copy or reference writes', async corruption => {
    const s = await setup(); const { upload, ticket } = await copied(s)
    if (corruption === 'missing-page') delete s.db.rows.media_storage_operations[pageId(s.identity.clip_id, 0)]
    if (corruption === 'tail-page') s.db.rows.media_storage_operations[pageId(s.identity.clip_id, 3)] = { entries: [] }
    if (corruption.includes('alias')) {
      const objectId = corruption === 'missing-output-alias' ? ticket.target.object_id : upload.target.object_id
      const key = Object.keys(s.db.rows.media_storage_objects).find(id => s.db.rows.media_storage_objects[id].object_id === objectId)
      if (corruption === 'foreign-input-alias') s.db.rows.media_storage_objects[key].epoch = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
      else delete s.db.rows.media_storage_objects[key]
    }
    if (corruption === 'wrong-input') entry(s, 2).operation.input_object_id = ticket.target.object_id
    if (corruption === 'root-revision') s.db.rows.clip_storage_authorities[s.identity.clip_id].revision++
    await unchanged(s, () => s.copy(upload), 'INCOMPLETE')
    await unchanged(s, () => s.register(ticket), 'INCOMPLETE')
  })

test.each(['close', 'tombstone'])('%s fences new copies and references but retains late recorded results', async fence => {
  const s = await setup(); const upload = await issued(s); const { ticket } = await s.copy(upload)
  if (fence === 'close') await s.close()
  else s.db.rows.owner_clip_tombstones = { [s.identity.clip_id]: { deleted_at: '2026-11-01T12:00:00.000Z' } }
  await unchanged(s, () => s.copy(upload), 'CONFLICT')
  await s.record(ticket, 'server_succeeded', [observation(ticket)])
  await unchanged(s, () => s.register(ticket), 'CONFLICT')
  expect(entry(s, 2).object.file_id).toBe(observation(ticket).file_id)
})

test.each(['copy-first', 'close-first'])('delete versus begin serializes %s on the root', async order => {
  const s = await setup(); const upload = await issued(s)
  const operations = order === 'copy-first' ? [s.copy(upload), s.close()] : [s.close(), s.copy(upload)]
  const outcomes = await Promise.allSettled(operations)
  if (order === 'copy-first') {
    expect(outcomes[0].value.ticket.sequence).toBe(2); expect(outcomes[1].value.high_water).toBe(2)
  } else {
    expect(outcomes[0].value.high_water).toBe(1); expect(outcomes[1].reason.code).toBe('STORAGE_AUTHORITY_CONFLICT')
  }
})

async function inventory63(options = {}) {
  const s = await setup(options); const uploads = []
  for (let i = 0; i < 63; i++) uploads.push(await issued(s))
  return { ...s, uploads }
}
test('N63/A63 appends correlated copy 64 once; N64/A64 references and exact replay stay within 90 calls', async () => {
  const s = await inventory63(); const upload = s.uploads[0]; const before = s.db.transactionAttempts.length
  const { ticket } = await s.copy(upload)
  expect(ticket.sequence).toBe(64); expect(count(s)).toBe(71)
  await s.record(ticket, 'server_succeeded', [observation(ticket)]); expect(count(s)).toBe(72)
  expect(await s.register(ticket)).toEqual({ file_id: observation(ticket).file_id, staging_file_id: observation(upload).file_id,
    objects: [observation(upload), observation(ticket)] }); expect(count(s)).toBe(73)
  const saved = clone(s.db.rows); await s.register(ticket); expect(s.db.rows).toEqual(saved); expect(count(s)).toBe(70)
  await unchanged(s, () => s.copy(upload), 'LIMIT'); expect(count(s)).toBe(70)
  for (const callback of s.db.transactionAttempts.slice(before)) {
    expect(callback.count).toBeLessThanOrEqual(90)
    expect(callback.calls.filter(c => c.collection === 'media_storage_operations' && c.method === 'get')).toHaveLength(4)
  }
})

test.each(['append', 'references'])('full-inventory %s root conflict retries discard all partial writes and stay within 90 per callback', async port => {
  let retry = false
  const s = await inventory63({ retryConflicts: 1, beforeCommit({ attempt: callback }) {
    if (retry && callback.index === 0) throw Object.assign(new Error('synthetic root conflict'), { code: 'SYNTHETIC_CONFLICT' })
  } })
  let ticket
  if (port === 'references') { ticket = (await s.copy(s.uploads[0])).ticket; await s.record(ticket, 'server_succeeded', [observation(ticket)]) }
  retry = true; const start = s.db.transactionAttempts.length
  if (port === 'append') ticket = (await s.copy(s.uploads[0])).ticket
  else await s.register(ticket)
  const callbacks = s.db.transactionAttempts.slice(start)
  expect(callbacks.map(c => c.count)).toEqual(port === 'append' ? [71, 71] : [73, 73])
  expect(s.db.rows.clip_storage_authorities[s.identity.clip_id].high_water).toBe(64)
  expect(entry(s, 64).operation.operation_id).toBe(ticket.operation_id)
  for (const callback of callbacks) {
    expect(callback.count).toBeLessThanOrEqual(90)
    expect(callback.calls).toContainEqual({ collection: 'clip_storage_authorities', id: s.identity.clip_id, method: 'update' })
    expect(callback.calls.filter(c => c.collection === 'media_storage_operations' && c.method === 'get')).toHaveLength(4)
  }
  if (port === 'references') {
    expect(entry(s, 1).object.references).toHaveLength(1); expect(entry(s, 64).object.references).toHaveLength(1)
  }
})

test.each(['media_storage_operations', 'clip_storage_authorities'])('composite reference failure in %s rolls back both references', async collection => {
  let fail = false
  const s = await setup({ failWrite(name) { if (fail && name === collection) throw new Error('reference write failed') } })
  const { ticket } = await copied(s); const before = clone(s.db.rows); fail = true
  await expect(s.register(ticket)).rejects.toThrow('reference write failed'); expect(s.db.rows).toEqual(before)
})
