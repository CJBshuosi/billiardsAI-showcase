const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { privateClip } = require('../helpers/private-clips')
const { documentId } = require('../../cloudfunctions/shared/shot-ownership')
const { sessionId, birthId, now } = require('../helpers/managed-media-admission-fixture')
let api = {}
try { api = require('../../cloudfunctions/shared/media-storage-authority') } catch (error) { if (error.code !== 'MODULE_NOT_FOUND') throw error }
const { makeStorageIdentity, createCloudMediaStorageAuthority } = api
const pageId = (clipId, index) => `page-${documentId(clipId)}-${index}`
function entry(s, sequence) { return s.db.rows.media_storage_operations[pageId(s.identity.clip_id, Math.floor((sequence - 1) / 16))].entries[(sequence - 1) % 16] }
const NOW = '2026-10-29T19:00:00.000Z'
function setup(seed = {}, options = {}) {
  // Isolated authority primitive boundary: trusted positive provenance is seeded
  // here; managed-media-admission.test exercises actual server allocation. This
  // fixture does not prove caller publication eligibility or ownership history.
  const birth = { contract: 'managed_session_birth_v1', session_id: sessionId, birth_id: birthId,
    table_id: 'table-1', device_id: 'device-1', opened_at: now, media_admission_contract: 'managed_clip_admission_v1' }
  const db = ownerDeleteDb({ managed_session_births: { [sessionId]: birth }, ...seed }, { strictTransactions: true, ...options })
  db.storage = new Proxy({}, { get() { throw new Error('storage must not be called') } })
  let next = 0
  const authority = createCloudMediaStorageAuthority(db, { clock: () => Date.parse(NOW), idFactory: () => `00000000-0000-4000-8000-${String(++next).padStart(12, '0')}` })
  const identity = makeStorageIdentity(privateClip({ session_id: sessionId }), { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })
  const tx = work => db.runTransaction(work)
  const enroll = () => tx(t => authority.enroll(t, identity))
  const begin = (change = {}) => tx(t => authority.beginOperation(t, { identity, ingestJobId: 'ingest-1', kind: 'upload_authorization', ...change }))
  const snapshot = () => tx(t => authority.snapshotForDeletion(t, identity.clip_id))
  return { db, authority, identity, tx, enroll, begin, snapshot }
}
function observation(ticket, file = `cloud://synthetic/${ticket.operation_id}`) { return { kind: ticket.target.kind, cloud_path: ticket.target.cloud_path, file_id: file } }
async function result(s, ticket, outcome, objects = [], reason = null) { return s.authority.recordOperationResult({ ticket, outcome, objects, reason }) }
test('canonical identity keeps analysis job separate and verifies private ownership', () => {
  const clip = privateClip(); const identity = makeStorageIdentity(clip, { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })
  expect(identity.source_job_id).toBe(clip.job_id); expect(Object.keys(identity)).toHaveLength(16)
  expect(() => makeStorageIdentity({ ...clip, owner_user_id: 'foreign' }, { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })).toThrow()
  expect(makeStorageIdentity({ ...clip, job_id: undefined }, { tableId: 'table-1', deviceId: 'device-1', kind: 'ordinary' })).toMatchObject({ source_job_id: null, owner_user_id: null, save_receipt_sha256: null })
})
test('fresh enrollment replays exactly and conflicting identity cannot replace it', async () => {
  const s = setup(); const first = await s.enroll()
  expect(first).toMatchObject({ contract: 'clip_storage_v2', high_water: 0, admission: 'open' }); expect(await s.enroll()).toEqual(first)
  await expect(s.tx(t => s.authority.enroll(t, { ...s.identity, byte_size: 2 }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
})
test.each(['clips', 'saved_clip_sync_jobs', 'sync_jobs'])('existing %s never automatically enrolls', async name => {
  const clip = privateClip(); const s = setup({ [name]: { [clip.clip_id]: { clip_id: clip.clip_id, session_id: clip.session_id, envelope: { media: [{ clip_id: clip.clip_id }] } } } })
  // Historical jobs have no positive admission-writer provenance.
  delete s.db.rows.managed_session_births[sessionId].media_admission_contract
  await expect(s.enroll()).resolves.toMatchObject({ status: 'legacy_untracked' }); expect(Object.keys(s.db.rows.clip_storage_authorities || {})).toHaveLength(0)
  await expect(s.begin()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
test('concurrent begins allocate contiguous operations and enumerate beyond one page', async () => {
  const s = setup(); await s.enroll(); const tickets = await Promise.all(Array.from({ length: 32 }, () => s.begin()))
  expect(tickets.map(t => t.sequence)).toEqual(Array.from({ length: 32 }, (_, i) => i + 1)); expect(new Set(tickets.map(t => t.operation_id)).size).toBe(32)
  for (const t of tickets) {
    expect(t.target.cloud_path).toBe(`managed-clips-v1/${documentId(s.identity.clip_id)}/staging/${t.operation_id}.mp4`)
    expect(await s.tx(tx => s.authority.getOperation(tx, t))).toMatchObject({ outcome: null, sequence: t.sequence })
  }
  const snapshot = await s.snapshot(); expect(snapshot.operations).toHaveLength(32); expect(snapshot.objects).toHaveLength(32); expect(snapshot.complete).toBe(false)
})
test('64 operation limit refuses atomically instead of truncating', async () => {
  const s = setup(); await s.enroll(); for (let i = 0; i < 64; i++) await s.begin(); const before = structuredClone(s.db.rows)
  await expect(s.begin()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_LIMIT' }); expect(s.db.rows).toEqual(before); expect((await s.snapshot()).operations).toHaveLength(64)
})
test('missing deterministic operation remains incomplete beyond sequence 25', async () => {
  const s = setup(); await s.enroll(); const tickets = []; for (let i = 0; i < 30; i++) tickets.push(await s.begin())
  delete entry(s, 27).operation
  expect(await s.snapshot()).toMatchObject({ complete: false, reasons: expect.arrayContaining(['invalid_entry:27']) })
  await expect(s.tx(tx => s.authority.getOperation(tx, tickets[26]))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
test.each(['external_issued', 'unknown'])('%s cannot be upgraded into closed evidence', async outcome => {
  const s = setup(); await s.enroll(); const t = await s.begin(); await result(s, t, outcome, outcome === 'external_issued' ? [observation(t)] : [], outcome === 'unknown' ? 'result_lost' : null)
  const muchLater = createCloudMediaStorageAuthority(s.db, { clock: () => Date.parse('2100-01-01T00:00:00.000Z') })
  expect(await s.tx(tx => muchLater.snapshotForDeletion(tx, s.identity.clip_id))).toMatchObject({ complete: false, erasable: false, reasons: expect.arrayContaining(['unresolved_operation:1']) })
  await expect(result(s, t, 'no_write')).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
})
test('exact terminal retry is idempotent and changed file cannot rewrite evidence', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin(); await result(s, t, 'external_issued', [observation(t)]); const before = structuredClone(s.db.rows)
  await result(s, t, 'external_issued', [observation(t)]); expect(s.db.rows).toEqual(before)
  await expect(result(s, t, 'external_issued', [observation(t, 'cloud://other')])).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' }); expect(s.db.rows).toEqual(before)
})
test('same opaque file cannot alias multiple keys; rejected result rolls back', async () => {
  const s = setup(); await s.enroll(); const a = await s.begin(); const b = await s.begin(); await result(s, a, 'external_issued', [observation(a, 'cloud://same')]); const before = structuredClone(s.db.rows)
  await expect(result(s, b, 'external_issued', [observation(b, 'cloud://same')])).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' }); expect(s.db.rows).toEqual(before)
})
test('unwritten input cannot be copied and no_write is complete ledger evidence only', async () => {
  const s = setup(); await s.enroll(); const staging = await s.begin(); await result(s, staging, 'no_write')
  await expect(s.begin({ kind: 'server_copy', inputObjectId: staging.target.object_id })).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
  expect(await s.snapshot()).toMatchObject({ complete: true, erasable: false })
})
test.each([{ byte_size: Number.MAX_SAFE_INTEGER + 1 }, { clip_id: '..' }, { table_id: 'bad\u0000' }, { content_type: 'video/other' }, { content_sha256: 'A'.repeat(64) }, { extra: 'token' }])('invalid exact identity fails closed: %p', async change => {
  const s = setup(); await expect(s.tx(t => s.authority.enroll(t, { ...s.identity, ...change }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INVALID' })
})
test('credential fields and freeform secret reasons cannot enter records', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin(); const before = structuredClone(s.db.rows)
  await expect(s.authority.recordOperationResult({ ticket: t, outcome: 'external_issued', objects: [{ ...observation(t), token: 'secret' }], reason: null })).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INVALID' })
  await expect(s.authority.recordOperationResult({ ticket: t, outcome: 'external_issued', objects: [observation(t)], token: 'secret' })).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INVALID' })
  await expect(result(s, t, 'unknown', [], 'signature=secret')).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INVALID' }); expect(s.db.rows).toEqual(before)
})
test('SDK top-level _id is valid metadata but nested unknown identity is rejected', async () => {
  const s = setup(); await s.enroll(); s.db.rows.clip_storage_authorities[s.identity.clip_id]._id = s.identity.clip_id
  await expect(s.begin()).resolves.toMatchObject({ sequence: 1 }); s.db.rows.clip_storage_authorities[s.identity.clip_id].identity.token = 'secret'
  await expect(s.begin()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
test.each([
  { provenance: 'marked', media: [], status: undefined },
  { provenance: 'marked', media: [{ clip_id: 'different-clip' }], status: undefined },
  { provenance: 'legacy', media: [{ clip_id: privateClip().clip_id }], status: 'legacy_untracked' },
])('legacy provenance refuses old same-clip media while marked facts/other-clip jobs coexist: %p', async ({ provenance, media, status }) => {
  const s = setup({ sync_jobs: { old: { session_id: sessionId, envelope: { media } } } })
  if (provenance === 'legacy') delete s.db.rows.managed_session_births[sessionId].media_admission_contract
  expect((await s.enroll()).status).toBe(status)
})
test('late unknown observation is monotonic after closure; unresolved outcome cannot become success', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin()
  await result(s, t, 'unknown', [], 'result_lost')
  await s.tx(tx => s.authority.closeForDeletion(tx, s.identity))
  await result(s, t, 'unknown', [observation(t)], 'result_lost')
  expect(await s.snapshot()).toMatchObject({ complete: false, objects: [expect.objectContaining({ file_id: observation(t).file_id })] })
  const before = structuredClone(s.db.rows)
  await expect(result(s, t, 'unknown', [], 'result_lost')).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  await expect(result(s, t, 'unknown', [observation(t, 'cloud://changed')], 'result_lost')).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  expect(s.db.rows).toEqual(before)
})
test('close and begin serialize in either order, frozen sequence never grows and old result is retained', async () => {
  const s = setup(); await s.enroll()
  const [ticket, closure] = await Promise.all([s.begin(), s.tx(tx => s.authority.closeForDeletion(tx, s.identity))])
  expect(closure).toMatchObject({ status: 'tracked_closed', high_water: 1, epoch: ticket.epoch })
  await result(s, ticket, 'external_issued', [observation(ticket)])
  await expect(s.begin()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  const s2 = setup(); await s2.enroll()
  const outcomes = await Promise.allSettled([s2.tx(tx => s2.authority.closeForDeletion(tx, s2.identity)), s2.begin()])
  expect(outcomes[0].value.high_water).toBe(0); expect(outcomes[1].reason.code).toBe('STORAGE_AUTHORITY_CONFLICT')
})
test('publication accepts registered objects once and refuses reserved, deleted, foreign or late references', async () => {
  const s = setup(); await s.enroll(); const staging = await s.begin(); await result(s, staging, 'external_issued', [observation(staging)])
  const copy = await s.begin({ kind: 'server_copy', inputObjectId: staging.target.object_id })
  const object = observation(copy); await result(s, copy, 'server_succeeded', [object])
  expect(copy.target.cloud_path).toBe(`published/managed-clips-v1/${documentId(s.identity.clip_id)}/${copy.operation_id}.mp4`)
  const publish = (objects = [object], identity = s.identity) => s.tx(tx => s.authority.registerReferences(tx, { identity, objects }))
  await publish(); const before = structuredClone(s.db.rows); await publish(); expect(s.db.rows).toEqual(before)
  await expect(publish([{ ...object, file_id: 'cloud://foreign' }])).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  for (const state of ['delete_reserved', 'deleted']) {
    entry(s, copy.sequence).object.state = state
    await expect(publish()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  }
  entry(s, copy.sequence).object.state = 'active'
  entry(s, copy.sequence).object.references.push({ clip_id: 'foreign', identity_sha256: s.identity.content_sha256 })
  expect(await s.snapshot()).toMatchObject({ complete: false, reasons: expect.arrayContaining(['invalid_object:2']) })
  await expect(publish()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
  entry(s, copy.sequence).object.references.pop()
  await s.tx(tx => s.authority.closeForDeletion(tx, s.identity))
  await expect(publish()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
})
test('foreign clip cannot reference another clips key or alias', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin(); await result(s, t, 'external_issued', [observation(t)])
  const foreign = { ...s.identity, clip_id: 'foreign' }; await s.tx(tx => s.authority.enroll(tx, foreign))
  await expect(s.tx(tx => s.authority.registerReferences(tx, { identity: foreign, objects: [observation(t)] }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
})
test.each(['media_storage_operations', 'clip_storage_authorities'])('begin write failure in %s rolls back allocation', async name => {
  let fail = false; const s = setup({}, { failWrite(collection) { if (fail && collection === name) throw new Error('write failed') } })
  await s.enroll(); const before = structuredClone(s.db.rows); fail = true
  await expect(s.begin()).rejects.toThrow('write failed'); expect(s.db.rows).toEqual(before)
})
test('missing aliases or corrupt operation revisions fail closed without reconstructing records', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin(); await result(s, t, 'external_issued', [observation(t)])
  delete s.db.rows.media_storage_objects[Object.entries(s.db.rows.media_storage_objects).find(([, v]) => v.record_kind === 'file')[0]]
  expect(await s.snapshot()).toMatchObject({ complete: false, reasons: expect.arrayContaining(['invalid_alias:1']) })
  await expect(result(s, t, 'external_issued', [observation(t)])).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
test.each(['root', 'object', 'operation'])('inconsistent %s revisions cannot certify a complete ledger', async type => {
  const s = setup(); await s.enroll(); const t = await s.begin(); await result(s, t, 'no_write')
  const row = type === 'root' ? s.db.rows.clip_storage_authorities[s.identity.clip_id] : type === 'object' ? entry(s, t.sequence).object : entry(s, t.sequence).operation
  row.revision += 1
  expect(await s.snapshot()).toMatchObject({ complete: false })
  await expect(s.begin()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
test('publication duplicate observations register one reference and a consistent revision', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin(); const o = observation(t); await result(s, t, 'external_issued', [o])
  await s.tx(tx => s.authority.registerReferences(tx, { identity: s.identity, objects: [o, o] }))
  expect(entry(s, t.sequence).object.references).toHaveLength(1)
  expect(entry(s, t.sequence).object.revision).toBe(3)
  // Remove the unresolved-writer reason only for this assertion; integrity must hold.
  expect((await s.snapshot()).reasons).toEqual(['unresolved_operation:1'])
})
test('complete enumeration includes all 32 terminal operations and objects', async () => {
  const s = setup(); await s.enroll()
  for (let i = 0; i < 32; i++) await result(s, await s.begin(), 'no_write')
  const snapshot = await s.snapshot(); expect(snapshot.complete).toBe(true); expect(snapshot.operations).toHaveLength(32); expect(snapshot.objects).toHaveLength(32)
})
test('deletion reports damaged ledger, closes admission and never backfills missing sequence', async () => {
  const s = setup(); await s.enroll(); await s.begin(); s.db.rows.media_storage_operations = {}
  const closure = await s.tx(tx => s.authority.closeForDeletion(tx, s.identity))
  expect(closure).toMatchObject({ status: 'legacy_authority_incomplete', high_water: 1, reasons: ['missing_page:0'] })
  expect(s.db.rows.clip_storage_authorities[s.identity.clip_id].admission).toBe('closed')
  expect(s.db.rows.media_storage_operations).toEqual({})
})
test.each(['media_storage_objects', 'media_storage_operations', 'clip_storage_authorities'])('result write failure in %s cannot deliver partial alias evidence', async name => {
  let fail = false; const s = setup({}, { failWrite(collection) { if (fail && collection === name) throw new Error('result rollback') } })
  await s.enroll(); const t = await s.begin(); const before = structuredClone(s.db.rows); fail = true
  await expect(result(s, t, 'external_issued', [observation(t)])).rejects.toThrow('result rollback'); expect(s.db.rows).toEqual(before)
})
test('object overflow, foreign alias and nested credential corruption remain incomplete', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin(); await result(s, t, 'external_issued', [observation(t)])
  const key = entry(s, t.sequence).object; const alias = Object.values(s.db.rows.media_storage_objects).find(o => o.record_kind === 'file')
  alias.clip_id = 'foreign'; expect((await s.snapshot()).reasons).toContain('invalid_alias:1'); alias.clip_id = s.identity.clip_id
  key.references = Array.from({ length: 65 }, () => ({ clip_id: s.identity.clip_id, identity_sha256: t.identity_sha256 }))
  expect((await s.snapshot()).reasons).toContain('invalid_object:1'); key.references = []
  entry(s, 1).operation.token = 'secret'
  expect((await s.snapshot()).reasons).toContain('invalid_operation:1')
})
test('getOperation refuses a valid ticket when its registry alias has disappeared', async () => {
  const s = setup(); await s.enroll(); const t = await s.begin(); await result(s, t, 'external_issued', [observation(t)])
  delete s.db.rows.media_storage_objects[Object.keys(s.db.rows.media_storage_objects).find(key => key.startsWith('file-'))]
  await expect(s.tx(tx => s.authority.getOperation(tx, t))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
test('reference port rejects unknown fields instead of silently accepting credential-bearing payloads', async () => {
  const s = setup(); await s.enroll()
  await expect(s.tx(tx => s.authority.registerReferences(tx, { identity: s.identity, objects: [], token: 'secret' }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INVALID' })
})

test.each(['content_sha256', 'ownership_receipt_sha256', 'save_receipt_sha256', 'postgame_manifest_sha256'])('array-valued %s cannot enter through factory or admission', async field => {
  const s = setup(); const clip = privateClip()
  const kind = field === 'content_sha256' ? 'ordinary' : 'saved_private'
  expect(() => makeStorageIdentity({ ...clip, [field]: [clip[field]] }, { tableId: 'table-1', deviceId: 'device-1', kind })).toThrow(expect.objectContaining({ code: 'STORAGE_AUTHORITY_INVALID' }))
  await expect(s.tx(tx => s.authority.enroll(tx, { ...s.identity, [field]: [s.identity[field]] }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INVALID' })
  expect(Object.keys(s.db.rows.clip_storage_authorities || {})).toHaveLength(0)
})
test.each(['content_sha256', 'ownership_receipt_sha256', 'save_receipt_sha256', 'postgame_manifest_sha256'])('persisted array-valued %s cannot certify canonical identity', async field => {
  const s = setup(); await s.enroll(); const root = s.db.rows.clip_storage_authorities[s.identity.clip_id]
  root.identity[field] = [root.identity[field]]
  root.identity_sha256 = require('../../cloudfunctions/shared/shot-ownership').digest(root.identity)
  expect(await s.snapshot()).toMatchObject({ complete: false, reasons: ['invalid_authority'] })
})
test('array-valued epoch cannot pass persisted UUID validation', async () => {
  const s = setup(); await s.enroll(); const root = s.db.rows.clip_storage_authorities[s.identity.clip_id]; root.epoch = [root.epoch]
  expect(await s.snapshot()).toMatchObject({ complete: false, reasons: ['invalid_authority'] })
})
test.each(['operation_id', 'epoch', 'identity_sha256'])('ticket %s must be a string before regex validation', async field => {
  const s = setup(); await s.enroll(); const t = await s.begin()
  await expect(s.tx(tx => s.authority.getOperation(tx, { ...t, [field]: [t[field]] }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INVALID' })
})
async function copyHistory() {
  const s = setup(); await s.enroll(); const staging = await s.begin(); await result(s, staging, 'external_issued', [observation(staging)])
  const second = await s.begin({ kind: 'server_copy', inputObjectId: staging.target.object_id }); await result(s, second, 'server_succeeded', [observation(second)])
  const third = await s.begin({ kind: 'server_copy', inputObjectId: staging.target.object_id }); await result(s, third, 'server_succeeded', [observation(third)])
  const operation = entry(s, 3).operation
  return { ...s, staging, second, third, operation }
}
test.each(['published', 'unenumerated'])('copy history cannot use %s input as a valid preceding staging object', async kind => {
  const s = await copyHistory()
  if (kind === 'published') s.operation.input_object_id = s.second.target.object_id
  else {
    const objectId = `key-${'f'.repeat(64)}`
    s.db.rows.media_storage_objects[objectId] = { contract: 'clip_storage_v2', record_kind: 'key', object_id: objectId, clip_id: s.identity.clip_id, identity_sha256: s.third.identity_sha256, epoch: s.third.epoch, sequence: 0, file_id: 'cloud://unregistered' }
    s.operation.input_object_id = objectId
  }
  await expect(s.tx(tx => s.authority.getOperation(tx, s.third))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
  expect((await s.snapshot()).reasons).toContain('invalid_input:3')
  expect(await s.tx(tx => s.authority.closeForDeletion(tx, s.identity))).toMatchObject({ status: 'legacy_authority_incomplete', reasons: expect.arrayContaining(['invalid_input:3']) })
})
test('copy input must have a validated file alias in the enumerated inventory', async () => {
  const s = await copyHistory()
  const aliasId = Object.keys(s.db.rows.media_storage_objects).find(id => id.startsWith('file-') && s.db.rows.media_storage_objects[id].object_id === s.staging.target.object_id)
  delete s.db.rows.media_storage_objects[aliasId]
  expect((await s.snapshot()).reasons).toEqual(expect.arrayContaining(['invalid_alias:1', 'invalid_input:2', 'invalid_input:3']))
})
test.each(['delete_reserved', 'deleted'])('valid historical copy input remains valid after becoming %s', async state => {
  const s = await copyHistory(); entry(s, s.staging.sequence).object.state = state
  await expect(s.tx(tx => s.authority.getOperation(tx, s.third))).resolves.toMatchObject({ sequence: 3 })
  expect((await s.snapshot()).reasons).toEqual(['unresolved_operation:1'])
  await expect(s.begin({ kind: 'server_copy', inputObjectId: s.staging.target.object_id })).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
  expect(await s.tx(tx => s.authority.closeForDeletion(tx, s.identity))).toMatchObject({ status: 'tracked_closed', reasons: [] })
})

async function fullInventory({ finalObserved = true, count = 64 } = {}) {
  const s = setup(); await s.enroll(); const tickets = []
  for (let i = 0; i < count; i++) {
    const ticket = await s.begin(); tickets.push(ticket)
    if (finalObserved || i < count - 1) await result(s, ticket, 'external_issued', [observation(ticket, `cloud://${String(i).padStart(4, '0')}${'x'.repeat(2036)}`)])
  }
  s.db.setStrictTransactions(true)
  return { ...s, tickets }
}
function lastCount(s) { return s.db.transactionAttempts.at(-1).count }
test.each(['snapshot', 'close', 'enroll', 'getAdmission', 'getOperation', 'bindingThenOperation', 'replay', 'references', 'bindingThenReferences', 'beginLimit'])('strict 64-entry %s preserves complete membership within 90 calls', async port => {
  const s = await fullInventory(); const t = s.tickets[63]; const observed = observation(t, `cloud://0063${'x'.repeat(2036)}`)
  if (port === 'snapshot') expect((await s.snapshot()).objects).toHaveLength(64)
  if (port === 'close') expect(await s.tx(tx => s.authority.closeForDeletion(tx, s.identity))).toMatchObject({ status: 'tracked_closed', high_water: 64 })
  if (port === 'enroll') expect((await s.enroll()).high_water).toBe(64)
  if (port === 'getAdmission') expect((await s.tx(tx => s.authority.getAdmission(tx, s.identity))).high_water).toBe(64)
  if (port === 'getOperation') expect((await s.tx(tx => s.authority.getOperation(tx, t))).sequence).toBe(64)
  if (port === 'bindingThenOperation') expect((await s.tx(async tx => {
    await s.authority.getAdmission(tx, s.identity)
    return s.authority.getOperation(tx, t)
  })).sequence).toBe(64)
  if (port === 'replay') await result(s, t, 'external_issued', [observed])
  if (['references', 'bindingThenReferences'].includes(port)) {
    const objects = s.tickets.map((ticket, i) => observation(ticket, `cloud://${String(i).padStart(4, '0')}${'x'.repeat(2036)}`))
    expect((await s.tx(async tx => {
      if (port === 'bindingThenReferences') await s.authority.getAdmission(tx, s.identity)
      return s.authority.registerReferences(tx, { identity: s.identity, objects })
    })).object_ids).toHaveLength(64)
  }
  if (port === 'beginLimit') await expect(s.begin()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_LIMIT' })
  expect(lastCount(s)).toBeLessThanOrEqual(90)
  expect(lastCount(s)).toBe({ snapshot: 69, close: 70, enroll: 71, getAdmission: 3, getOperation: 69, bindingThenOperation: 72, replay: 69, references: 75, bindingThenReferences: 78, beginLimit: 70 }[port])
})
test('strict result adds new alias at entry 64 and replays within 90 calls', async () => {
  const s = await fullInventory({ finalObserved: false }); const t = s.tickets[63]
  await result(s, t, 'external_issued', [observation(t)]); expect(lastCount(s)).toBe(72)
  await result(s, t, 'external_issued', [observation(t)]); expect(lastCount(s)).toBe(69)
})
test.each(['upload_authorization', 'server_copy'])('strict %s admission of entry 64 keeps complete inventory within 90 calls', async kind => {
  const s = await fullInventory({ count: 63 })
  const change = kind === 'server_copy' ? { kind, inputObjectId: s.tickets[0].target.object_id } : { kind }
  expect((await s.begin(change)).sequence).toBe(64); expect(lastCount(s)).toBe(71)
})
test('fresh positive enroll satisfies native doc-only transaction contract', async () => {
  const s = setup({}, { strictTransactions: true })
  await expect(s.enroll()).resolves.toMatchObject({ high_water: 0, admission: 'open' })
  expect(lastCount(s)).toBe(6)
})
test('strict helper counts repeated document calls, rolls back above 100, and leaves outside queries available', async () => {
  const db = ownerDeleteDb({ test: { a: { n: 1 } } }, { strictTransactions: true })
  await expect(db.runTransaction(async tx => {
    await tx.collection('test').doc('a').update({ data: { n: 2 } })
    for (let i = 0; i < 100; i++) await tx.collection('test').doc('a').get()
  })).rejects.toThrow('transaction operation limit exceeded')
  expect(db.transactionAttempts[0].count).toBe(101); expect(db.rows.test.a.n).toBe(1)
  expect((await db.collection('test').where({ n: 1 }).get()).data).toEqual([{ n: 1 }])
})
test.each(['where', 'get', 'orderBy', 'limit', 'aggregate', 'count'])('strict helper rejects transaction collection %s', async method => {
  const db = ownerDeleteDb({}, { strictTransactions: true })
  await expect(db.runTransaction(tx => tx.collection('test')[method]({}))).rejects.toThrow(`transaction query unsupported: ${method}`)
})

test.each(['missing-page', 'missing-entry', 'header', 'epoch', 'tail', 'duplicate', 'oversized', 'entry-field'])('persisted page %s corruption blocks all checked ports', async corruption => {
  const s = setup(); await s.enroll(); const a = await s.begin(); await s.begin()
  const page = s.db.rows.media_storage_operations[pageId(s.identity.clip_id, 0)]
  if (corruption === 'missing-page') delete s.db.rows.media_storage_operations[pageId(s.identity.clip_id, 0)]
  if (corruption === 'missing-entry') page.entries.pop()
  if (corruption === 'header') page.clip_id = 'foreign'
  if (corruption === 'epoch') page.epoch = 'ffffffff-ffff-4fff-8fff-ffffffffffff'
  if (corruption === 'tail') page.entries.push(structuredClone(page.entries[1]))
  if (corruption === 'duplicate') page.entries[1] = structuredClone(page.entries[0])
  if (corruption === 'oversized') page.entries[0].operation.created_at = '2026-10-29' + ' '.repeat(128 * 1024)
  if (corruption === 'entry-field') page.entries[0].token = 'secret'
  expect((await s.snapshot()).complete).toBe(false)
  await expect(s.tx(tx => s.authority.getOperation(tx, a))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
  await expect(s.begin()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
test('old v1 root cannot be read, enrolled or upgraded', async () => {
  const s = setup(); await s.enroll(); s.db.rows.clip_storage_authorities[s.identity.clip_id].contract = 'clip_storage_v1'
  const before = structuredClone(s.db.rows)
  expect(await s.snapshot()).toMatchObject({ complete: false, reasons: ['invalid_authority'] })
  await expect(s.enroll()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
  expect(s.db.rows).toEqual(before)
})
test('full 2048-character fileIDs fit four bounded pages with no individual key documents', async () => {
  const s = await fullInventory(); const snapshot = await s.snapshot()
  expect(snapshot.objects.every(o => o.file_id.length === 2048)).toBe(true)
  const pages = Object.values(s.db.rows.media_storage_operations)
  expect(pages).toHaveLength(4)
  expect(pages.every(p => p.entries.length === 16 && Buffer.byteLength(JSON.stringify(p), 'utf8') <= 128 * 1024)).toBe(true)
  expect(Object.keys(s.db.rows.media_storage_objects).every(id => id.startsWith('file-'))).toBe(true)
})
test('strict helper cannot commit a transaction that catches the call-limit exception', async () => {
  const db = ownerDeleteDb({ test: { a: { n: 1 } } }, { strictTransactions: true })
  await expect(db.runTransaction(async tx => {
    await tx.collection('test').doc('a').update({ data: { n: 2 } })
    try { for (let i = 0; i < 100; i++) await tx.collection('test').doc('a').get() } catch (_) {}
  })).rejects.toThrow('transaction operation limit exceeded')
  expect(db.rows.test.a.n).toBe(1)
})
test('every mutating port retains the common root conflict boundary and retries from a fresh attempt', async () => {
  let retry = false
  const s = setup({}, { strictTransactions: false, retryConflicts: 1, beforeCommit({ attempt }) {
    if (retry && attempt.index === 0) throw Object.assign(new Error('synthetic root conflict'), { code: 'SYNTHETIC_CONFLICT' })
  } })
  await s.enroll(); s.db.setStrictTransactions(true); retry = true
  const before = s.db.transactionAttempts.length; const t = await s.begin()
  const attempts = s.db.transactionAttempts.slice(before)
  expect(attempts.map(a => a.count)).toEqual([8, 8])
  for (const attempt of attempts) expect(attempt.calls).toContainEqual({ collection: 'clip_storage_authorities', id: s.identity.clip_id, method: 'update' })
  expect(t.sequence).toBe(1); expect(s.db.rows.clip_storage_authorities[s.identity.clip_id].high_water).toBe(1)
  expect(Object.values(s.db.rows.media_storage_operations)[0].entries).toHaveLength(1)
  retry = false
  await result(s, t, 'external_issued', [observation(t)])
  expect(s.db.transactionAttempts.at(-1).calls).toContainEqual({ collection: 'clip_storage_authorities', id: s.identity.clip_id, method: 'update' })
  await s.tx(tx => s.authority.registerReferences(tx, { identity: s.identity, objects: [observation(t)] }))
  expect(s.db.transactionAttempts.at(-1).calls).toContainEqual({ collection: 'clip_storage_authorities', id: s.identity.clip_id, method: 'update' })
  await s.tx(tx => s.authority.closeForDeletion(tx, s.identity))
  expect(s.db.transactionAttempts.at(-1).calls).toContainEqual({ collection: 'clip_storage_authorities', id: s.identity.clip_id, method: 'update' })
})
test('strict unknown enrichment after close remains unresolved at 64 entries', async () => {
  const s = await fullInventory({ finalObserved: false }); const t = s.tickets[63]
  await result(s, t, 'unknown', [], 'result_lost'); expect(lastCount(s)).toBe(70)
  await s.tx(tx => s.authority.closeForDeletion(tx, s.identity)); expect(lastCount(s)).toBe(69)
  await result(s, t, 'unknown', [observation(t)], 'result_lost'); expect(lastCount(s)).toBe(72)
  expect((await s.snapshot()).reasons).toContain('unresolved_operation:64'); expect(lastCount(s)).toBe(69)
})
test('unused fixed page slots cannot hide unexpected tail members', async () => {
  const s = setup(); await s.enroll(); await s.begin()
  s.db.rows.media_storage_operations[pageId(s.identity.clip_id, 1)] = { ...structuredClone(s.db.rows.media_storage_operations[pageId(s.identity.clip_id, 0)]), page_index: 1 }
  expect((await s.snapshot()).reasons).toContain('unexpected_page:1')
  await expect(s.enroll()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})
