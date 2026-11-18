const { managedMediaBirthFixture, sessionId, input } = require('../helpers/managed-media-admission-fixture')
const { privateClip } = require('../helpers/private-clips')
const { documentId, digest } = require('../../cloudfunctions/shared/shot-ownership')
const { makeStorageIdentity, createCloudMediaStorageAuthority } = require('../../cloudfunctions/shared/media-storage-authority')

// Actual trusted server allocation proves marker provenance. These authority
// primitive tests do not replace the caller's publication/ownership proof.
async function setup({ birth = 'marked', seed = {} } = {}) {
  const { db, admin } = managedMediaBirthFixture({ managedMediaAdmissionEnabled: birth === 'marked' })
  if (birth !== 'missing') await admin.createManagedSession(input)
  Object.assign(db.rows, seed)
  let next = 0
  const authority = createCloudMediaStorageAuthority(db, {
    clock: () => Date.parse('2026-11-01T11:00:00.000Z'),
    idFactory: () => `00000000-0000-4000-8000-${String(++next).padStart(12, '0')}`,
  })
  const identity = makeStorageIdentity(privateClip({ session_id: sessionId }), { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })
  const tx = work => db.runTransaction(work)
  return { db, authority, identity, tx, enroll: () => tx(t => authority.enroll(t, identity)),
    admission: () => tx(t => authority.getAdmission(t, identity)) }
}
function noWrites(db) { expect(db.transactionAttempts.at(-1).calls.every(call => call.method === 'get')).toBe(true) }
function budget(db) { for (const attempt of db.transactionAttempts) expect(attempt.count).toBeLessThanOrEqual(90) }

test('fresh marked saved admission uses deterministic documents only and exact replay preserves epoch', async () => {
  const s = await setup(); const root = await s.enroll()
  expect(root).toMatchObject({ contract: 'clip_storage_v2', identity: s.identity, high_water: 0, admission: 'open', revision: 1 })
  expect(s.db.transactionAttempts.at(-1).calls).toEqual(expect.arrayContaining([
    { collection: 'clips', id: s.identity.clip_id, method: 'get' },
    { collection: 'saved_clip_sync_idempotency', id: documentId(s.identity.clip_id), method: 'get' },
  ]))
  expect(await s.enroll()).toEqual(root); noWrites(s.db); budget(s.db)
})

test.each(['missing', 'unmarked'])('%s birth is untracked and cannot create reserved authority', async birth => {
  const s = await setup({ birth }); const value = await s.enroll()
  expect(value).toEqual({ status: 'legacy_untracked', contract: 'clip_storage_v2', identity_sha256: expect.stringMatching(/^[a-f0-9]{64}$/u) })
  expect(s.db.rows.clip_storage_authorities?.[s.identity.clip_id]).toBeUndefined(); noWrites(s.db)
})

test.each([
  ['unknown marker', { media_admission_contract: 'foreign' }, 'INCOMPLETE'],
  ['extra field', { token: 'private' }, 'INCOMPLETE'],
  ['foreign table', { table_id: 'foreign' }, 'CONFLICT'],
  ['foreign device', { device_id: 'foreign' }, 'CONFLICT'],
  ['foreign session', { session_id: 'managed-87654321-1234-4234-8234-123456789abc' }, 'INCOMPLETE'],
])('%s never falls back to legacy', async (_, change, code) => {
  const s = await setup(); Object.assign(s.db.rows.managed_session_births[sessionId], change)
  await expect(s.enroll()).rejects.toMatchObject({ name: 'MediaStorageAuthorityError', code: `STORAGE_AUTHORITY_${code}`, message: `STORAGE_AUTHORITY_${code}` })
  noWrites(s.db)
})

test.each(['clip', 'old job', 'missing pointed job', 'foreign pointed job', 'tracked job without authority', 'malformed pointer'])('%s cannot be adopted when authority is absent', async state => {
  const s = await setup(); const id = s.identity.clip_id
  if (state === 'clip') s.db.rows.clips = { [id]: privateClip({ session_id: sessionId }) }
  else {
    s.db.rows.saved_clip_sync_idempotency = { [documentId(id)]: state === 'malformed pointer' ? {} : { job_id: 'old' } }
    if (state !== 'missing pointed job') s.db.rows.saved_clip_sync_jobs = { old: { clip_id: state === 'foreign pointed job' ? 'other' : id, session_id: sessionId,
      ...(state === 'tracked job without authority' ? { storage_admission: { contract: 'managed_clip_admission_v1', storage_contract: 'clip_storage_v2', identity_sha256: digest(s.identity), epoch: '12345678-1234-4234-8234-123456789abc' } } : {}) } }
  }
  expect(await s.enroll()).toMatchObject({ status: 'legacy_untracked' }); noWrites(s.db)
  expect(s.db.rows.clip_storage_authorities?.[id]).toBeUndefined()
})

test('facts-only same-session and different-clip jobs do not block positive admission', async () => {
  const s = await setup({ seed: {
    sync_jobs: { facts: { session_id: sessionId, envelope: { media: [] } }, other: { session_id: sessionId, envelope: { media: [{ clip_id: 'other' }] } } },
    saved_clip_sync_jobs: { other: { clip_id: 'other' } },
    saved_clip_sync_idempotency: { [documentId('other')]: { job_id: 'other' } },
  } })
  expect(await s.enroll()).toMatchObject({ high_water: 0 }); budget(s.db)
})

test('ordinary and nonmanaged saved fresh attempts remain outside reserved authority', async () => {
  const s = await setup()
  for (const identity of [makeStorageIdentity(privateClip(), { tableId: 'table-1', deviceId: 'device-1', kind: 'ordinary' }),
    makeStorageIdentity(privateClip(), { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })]) {
    expect(await s.tx(t => s.authority.enroll(t, identity))).toMatchObject({ status: 'legacy_untracked' }); noWrites(s.db)
  }
})

test.each(['missing job', 'compacted idempotency'])('retained inventory replays with original epoch after %s', async state => {
  const s = await setup(); await s.enroll()
  await s.tx(t => s.authority.beginOperation(t, { identity: s.identity, ingestJobId: 'job', kind: 'upload_authorization' }))
  const root = structuredClone(s.db.rows.clip_storage_authorities[s.identity.clip_id])
  s.db.rows.clips = { [s.identity.clip_id]: privateClip({ session_id: sessionId }) }
  s.db.rows.saved_clip_sync_idempotency = state === 'missing job' ? { [documentId(s.identity.clip_id)]: { job_id: 'gone' } } : {}
  expect(await s.enroll()).toEqual(root); noWrites(s.db)
  delete s.db.rows.media_storage_operations[Object.keys(s.db.rows.media_storage_operations)[0]]
  await expect(s.enroll()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})

test.each(['missing', 'unmarked'])('old retained authority with %s birth is never promoted or rewritten', async state => {
  const s = await setup(); await s.enroll()
  if (state === 'missing') delete s.db.rows.managed_session_births[sessionId]
  else delete s.db.rows.managed_session_births[sessionId].media_admission_contract
  const before = structuredClone(s.db.rows)
  expect(await s.enroll()).toMatchObject({ status: 'legacy_untracked' }); expect(s.db.rows).toEqual(before); noWrites(s.db)
  expect(await s.tx(t => s.authority.snapshotForDeletion(t, s.identity.clip_id))).toMatchObject({ status: 'tracked', complete: true })
  const ticket = await s.tx(t => s.authority.beginOperation(t, { identity: s.identity, ingestJobId: 'retained-job', kind: 'upload_authorization' }))
  expect(await s.tx(t => s.authority.getOperation(t, ticket))).toMatchObject({ sequence: 1, outcome: null })
  await s.authority.recordOperationResult({ ticket, outcome: 'no_write', objects: [], reason: null })
  expect(await s.tx(t => s.authority.registerReferences(t, { identity: s.identity, objects: [] }))).toMatchObject({ object_ids: [] })
  expect(await s.tx(t => s.authority.closeForDeletion(t, s.identity))).toMatchObject({ status: 'tracked_closed', high_water: 1 })
})

test('factory exposes getAdmission for tracked binding reads', async () => {
  const s = await setup(); expect(typeof s.authority.getAdmission).toBe('function')
})

test('factory exposes a lightweight marked-root admission binding without inventory enumeration', async () => {
  const s = await setup(); const root = await s.enroll()
  expect(typeof s.authority.getAdmission).toBe('function')
  expect(await s.admission()).toEqual(root)
  expect(s.db.transactionAttempts.at(-1).calls.map(call => call.collection).sort()).toEqual(['clip_storage_authorities', 'managed_session_births', 'owner_clip_tombstones'])
  noWrites(s.db)
  await s.tx(t => s.authority.beginOperation(t, { identity: s.identity, ingestJobId: 'job', kind: 'upload_authorization' }))
  s.db.rows.media_storage_operations = {}
  expect(await s.admission()).toMatchObject({ high_water: 1 }) // Binding is explicitly not a completeness proof.
  await expect(s.enroll()).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_INCOMPLETE' })
})

test.each(['missing birth', 'unmarked birth', 'missing root', 'invalid root', 'closed', 'tombstone', 'identity drift'])('getAdmission refuses %s without writes or inventory reads', async state => {
  const s = await setup(); await s.enroll()
  if (state === 'missing birth') delete s.db.rows.managed_session_births[sessionId]
  if (state === 'unmarked birth') delete s.db.rows.managed_session_births[sessionId].media_admission_contract
  if (state === 'missing root') delete s.db.rows.clip_storage_authorities[s.identity.clip_id]
  if (state === 'invalid root') s.db.rows.clip_storage_authorities[s.identity.clip_id].contract = 'clip_storage_v1'
  if (state === 'closed') await s.tx(t => s.authority.closeForDeletion(t, s.identity))
  if (state === 'tombstone') s.db.rows.owner_clip_tombstones = { [s.identity.clip_id]: {} }
  const identity = state === 'identity drift' ? { ...s.identity, byte_size: 2 } : s.identity
  await expect(s.tx(t => s.authority.getAdmission(t, identity))).rejects.toMatchObject({ code: ['closed', 'tombstone', 'identity drift'].includes(state) ? 'STORAGE_AUTHORITY_CONFLICT' : 'STORAGE_AUTHORITY_INCOMPLETE' })
  noWrites(s.db)
  expect(s.db.transactionAttempts.at(-1).calls.some(call => ['media_storage_operations', 'media_storage_objects'].includes(call.collection))).toBe(false)
})

test.each(['closed', 'tombstone', 'identity drift'])('enroll refuses existing %s rather than reopening/resetting', async state => {
  const s = await setup(); await s.enroll()
  if (state === 'closed') await s.tx(t => s.authority.closeForDeletion(t, s.identity))
  if (state === 'tombstone') s.db.rows.owner_clip_tombstones = { [s.identity.clip_id]: {} }
  const before = structuredClone(s.db.rows)
  await expect(s.tx(t => s.authority.enroll(t, state === 'identity drift' ? { ...s.identity, byte_size: 2 } : s.identity))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  expect(s.db.rows).toEqual(before)
})

test.each(['enroll', 'getAdmission'])('%s maps unavailable birth reads to fixed retryable incompleteness without fallback', async port => {
  const s = await setup()
  await expect(s.tx(tx => s.authority[port]({ collection(name) {
    if (name !== 'managed_session_births') return tx.collection(name)
    return { doc() { return { async get() { throw new Error('private backend credential detail') } } } }
  } }, s.identity))).rejects.toMatchObject({ name: 'MediaStorageAuthorityError',
    code: 'STORAGE_AUTHORITY_INCOMPLETE', message: 'STORAGE_AUTHORITY_INCOMPLETE' })
  noWrites(s.db)
  expect(s.db.rows.clip_storage_authorities?.[s.identity.clip_id]).toBeUndefined()
})

test('concurrent fresh enrollment converges on the retained root without an epoch reset', async () => {
  const s = await setup(); const roots = await Promise.all([s.enroll(), s.enroll()])
  expect(roots[0]).toEqual(roots[1])
  expect(s.db.transactionAttempts.flatMap(attempt => attempt.calls).filter(call => call.collection === 'clip_storage_authorities' && call.method === 'set')).toHaveLength(1)
  budget(s.db)
})
