const { createJoinAdmin } = require('../../cloudfunctions/sync-api/lib/join-admin')
const { validManagedBirth, MANAGED_MEDIA_ADMISSION_CONTRACT } = require('../../cloudfunctions/shared/managed-command-authority')
const { managedMediaBirthFixture, sessionId, birthId, now, input } = require('../helpers/managed-media-admission-fixture')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')

const contract = 'managed_clip_admission_v1'
const expectedPublic = { schema_version: '1.0', session_id: sessionId, birth_id: birthId,
  birth_contract: 'managed_session_birth_v1', table_id: 'table-1', device_id: 'device-1', status: 'open' }
const legacyBirth = { contract: 'managed_session_birth_v1', session_id: sessionId, birth_id: birthId,
  table_id: 'table-1', device_id: 'device-1', opened_at: now }
const scope = { sessionId, tableId: 'table-1', deviceId: 'device-1' }

function budget(db) {
  for (const attempt of db.transactionAttempts) {
    expect(attempt.count).toBeLessThanOrEqual(25)
    expect(attempt.calls.every(call => ['get', 'set', 'update'].includes(call.method))).toBe(true)
  }
}

test('only trusted enabled new allocation marks birth and keeps public/head/live bytes unchanged', async () => {
  const marked = managedMediaBirthFixture()
  const legacy = managedMediaBirthFixture({ managedMediaAdmissionEnabled: false })
  expect(await marked.admin.createManagedSession(input)).toEqual(expectedPublic)
  expect(await legacy.admin.createManagedSession({ ...input, managedMediaAdmissionEnabled: true })).toEqual(expectedPublic)
  expect(marked.db.rows.managed_session_births[sessionId]).toEqual({ ...legacyBirth, media_admission_contract: contract })
  expect(legacy.db.rows.managed_session_births[sessionId]).toEqual(legacyBirth)
  expect(marked.db.transactionAttempts).toHaveLength(1)
  expect(marked.db.transactionAttempts[0].calls.map(call => call.collection)).not.toEqual(expect.arrayContaining(['clip_storage_authorities', 'media_storage_operations', 'media_storage_objects', 'saved_clip_sync_jobs']))
  const { media_admission_contract, ...unmarked } = marked.db.rows.managed_session_births[sessionId]
  expect(unmarked).toEqual(legacy.db.rows.managed_session_births[sessionId])
  for (const name of Object.keys(legacy.db.rows).filter(name => name !== 'managed_session_births')) {
    expect(marked.db.rows[name]).toEqual(legacy.db.rows[name])
  }
  expect(MANAGED_MEDIA_ADMISSION_CONTRACT).toBe(contract)
  budget(marked.db); budget(legacy.db)
})

test.each([false, true])('allocation replay never changes existing birth when original marking is %s', async enabled => {
  const { db, admin, calls } = managedMediaBirthFixture({ managedMediaAdmissionEnabled: enabled })
  await admin.createManagedSession(input)
  const before = JSON.stringify(db.rows)
  const replay = createJoinAdmin(db, { managedSessionsEnabled: true, managedMediaAdmissionEnabled: !enabled,
    idFactory() { throw new Error('replay must not allocate') }, nowIso() { throw new Error('replay must not read clock') } })
  expect(await replay.createManagedSession(input)).toEqual(expectedPublic)
  expect(JSON.stringify(db.rows)).toBe(before)
  expect(calls).toEqual({ ids: 1, clocks: 1 })
  expect(db.transactionAttempts[1].calls.every(call => call.method === 'get')).toBe(true)
  budget(db)
})

test('omitted trusted option stays disabled', async () => {
  const { db } = managedMediaBirthFixture()
  await createJoinAdmin(db, { managedSessionsEnabled: true, idFactory: () => birthId, nowIso: () => now }).createManagedSession(input)
  expect(db.rows.managed_session_births[sessionId]).toEqual(legacyBirth)
})

test.each([null, 0, 1, 'true', {}, [], true])('invalid constructor configuration %p fails before DB work', value => {
  const { db } = managedMediaBirthFixture()
  const before = JSON.stringify(db.rows)
  expect(() => createJoinAdmin(db, { managedMediaAdmissionEnabled: value })).toThrow(expect.objectContaining({ code: 'SYNC_CONFIGURATION_REQUIRED' }))
  expect(db.transactionAttempts).toHaveLength(0)
  expect(JSON.stringify(db.rows)).toBe(before)
})

test('strict parser accepts exactly old six fields or new seven known fields', () => {
  expect(validManagedBirth(legacyBirth)).toBe(true)
  expect(validManagedBirth({ ...legacyBirth, media_admission_contract: undefined })).toBe(false)
  expect(validManagedBirth({ ...legacyBirth, media_admission_contract: contract })).toBe(true)
  expect(validManagedBirth({ ...legacyBirth, media_admission_contract: contract, _id: sessionId })).toBe(true)
})

test.each([null, '', 'unknown', 1, true, [], {}])('malformed marker %p refuses allocation replay', async marker => {
  const { db, admin } = managedMediaBirthFixture()
  await admin.createManagedSession(input)
  db.rows.managed_session_births[sessionId].media_admission_contract = marker
  expect(validManagedBirth({ ...legacyBirth, media_admission_contract: marker })).toBe(false)
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test.each([{ unexpected: true }, { table_id: 'foreign' }, { device_id: 'foreign' }])('marked birth corruption %p refuses replay', async mutation => {
  const { db, admin } = managedMediaBirthFixture()
  await admin.createManagedSession(input)
  Object.assign(db.rows.managed_session_births[sessionId], mutation)
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

// Load the new API in the tests so preimplementation birth assertions still run.
const admission = () => require('../../cloudfunctions/shared/managed-media-admission')
const epoch = 'abcdefab-1234-4234-9234-123456789abc'
const identity = { publication_kind: 'saved_private', clip_id: 'clip-1', session_id: sessionId, shot_id: 'shot-1',
  source_job_id: 'job-1', artifact_id: 'artifact-1', table_id: 'table-1', device_id: 'device-1',
  content_sha256: 'a'.repeat(64), byte_size: 10, content_type: 'video/mp4', source_kind: 'usb_live',
  owner_user_id: 'owner-1', ownership_receipt_sha256: 'b'.repeat(64), save_receipt_sha256: 'c'.repeat(64), postgame_manifest_sha256: 'd'.repeat(64) }
// Independently computed SHA-256 over sorted canonical JSON for the above fixture.
const identityHash = 'dcd21c70dcd78deb8089ff79a10781bb58a7d660a63eed39a657cf1e4d27ae10'
const marker = { contract, storage_contract: 'clip_storage_v2', identity_sha256: identityHash, epoch }
const root = { contract: 'clip_storage_v2', identity, identity_sha256: identityHash, epoch }

test('pure metadata helper builds exact marker using the existing canonical digest', () => {
  const { makeStorageAdmission, validStorageAdmission, STORAGE_AUTHORITY_CONTRACT } = admission()
  expect(makeStorageAdmission(identity, root)).toEqual(marker)
  expect(validStorageAdmission(marker, Object.fromEntries(Object.entries(identity).reverse()), epoch)).toBe(true)
  expect(STORAGE_AUTHORITY_CONTRACT).toBe('clip_storage_v2')
})

test.each([
  null, [], {}, { ...marker, unexpected: true }, { ...marker, _id: 'x' },
  { ...marker, contract: 'unknown' }, { ...marker, storage_contract: 'unknown' },
  { ...marker, identity_sha256: identityHash.toUpperCase() }, { ...marker, identity_sha256: '0'.repeat(64) },
  { ...marker, epoch: epoch.toUpperCase() }, { ...marker, epoch: 'not-uuid' }, { ...marker, epoch: null },
])('pure metadata validator refuses malformed marker %p without coercion', bad => {
  expect(admission().validStorageAdmission(bad, identity, epoch)).toBe(false)
})

test('metadata validator binds the full identity and original lowercase UUID epoch', () => {
  const { validStorageAdmission } = admission()
  expect(validStorageAdmission(marker, { ...identity, owner_user_id: 'foreign' }, epoch)).toBe(false)
  expect(validStorageAdmission(marker, identity, birthId)).toBe(false)
  for (const bad of [null, [], 'identity']) expect(validStorageAdmission(marker, bad, epoch)).toBe(false)
})

test.each([
  null, {}, { ...root, contract: 'old' }, { ...root, identity_sha256: '0'.repeat(64) },
  { ...root, identity: { ...identity, clip_id: 'foreign' } }, { ...root, epoch: epoch.toUpperCase() },
])('metadata builder refuses invalid or conflicting root %p with fixed error', bad => {
  expect(() => admission().makeStorageAdmission(identity, bad)).toThrow(admission().ManagedMediaAdmissionError)
  expect(() => admission().makeStorageAdmission(identity, bad)).toThrow(/^MEDIA_ADMISSION_(INVALID|CONFLICT)$/u)
})

test.each(['missing', 'unmarked', 'marked'])('birth reader uses one deterministic doc read for %s provenance', async state => {
  const { db, admin } = managedMediaBirthFixture({ managedMediaAdmissionEnabled: state === 'marked' })
  if (state !== 'missing') await admin.createManagedSession(input)
  const before = JSON.stringify(db.rows)
  const result = await db.runTransaction(tx => admission().readManagedMediaBirth(tx, scope))
  expect(result).toEqual(state === 'marked' ? { ...legacyBirth, media_admission_contract: contract } : null)
  const attempt = db.transactionAttempts.at(-1)
  expect(attempt.calls).toEqual([{ collection: 'managed_session_births', method: 'get', id: sessionId }])
  // The synthetic collection accessor may create an empty collection on a missing read.
  expect(JSON.stringify(db.rows.managed_session_births?.[sessionId] ?? null)).toBe(JSON.stringify(state === 'missing' ? null : JSON.parse(before).managed_session_births[sessionId]))
})

test.each([
  { ...legacyBirth, media_admission_contract: 'unknown' }, { ...legacyBirth, unexpected: true },
  { ...legacyBirth, table_id: 'foreign' }, { ...legacyBirth, device_id: 'foreign' },
  { ...legacyBirth, session_id: `managed-${epoch}` }, { ...legacyBirth, opened_at: 'bad' },
  { ...legacyBirth, media_admission_contract: contract, unexpected: true },
  false, 0, '', [],
])('birth reader rejects malformed/foreign provenance %p rather than legacy fallback', async birth => {
  const db = ownerDeleteDb({ managed_session_births: { [sessionId]: birth } }, { strictTransactions: true })
  await expect(db.runTransaction(tx => admission().readManagedMediaBirth(tx, scope)))
    .rejects.toMatchObject({ name: 'ManagedMediaAdmissionError', message: expect.stringMatching(/^MEDIA_ADMISSION_(INVALID|CONFLICT)$/u), code: expect.stringMatching(/^MEDIA_ADMISSION_(INVALID|CONFLICT)$/u) })
})

test('birth reader handles native missing document but never swallows provider/DB errors', async () => {
  const { readManagedMediaBirth } = admission()
  const tx = message => ({ collection(name) {
    expect(name).toBe('managed_session_births')
    return { doc(id) { expect(id).toBe(sessionId); return { get() { throw new Error(message) } } } }
  } })
  expect(await readManagedMediaBirth(tx(`document with _id ${sessionId} does not exist`), scope)).toBeNull()
  await expect(readManagedMediaBirth(tx('private backend error'), scope)).rejects.toMatchObject({ name: 'ManagedMediaAdmissionError', message: 'MEDIA_ADMISSION_INVALID', code: 'MEDIA_ADMISSION_INVALID' })
})


test.each([false, undefined, 'true', 1, null])('media enablement requires exact managed enablement (%p)', managedSessionsEnabled => {
  expect(() => createJoinAdmin({}, { managedSessionsEnabled, managedMediaAdmissionEnabled: true }))
    .toThrow(expect.objectContaining({ code: 'SYNC_CONFIGURATION_REQUIRED' }))
})

test('marked birth rolls back atomically when a later allocation write fails', async () => {
  const { db, admin } = managedMediaBirthFixture({}, { failWrite(name) {
    if (name === 'table_join_codes') throw new Error('synthetic write failure')
  } })
  const before = JSON.stringify(db.rows)
  await expect(admin.createManagedSession(input)).rejects.toThrow('synthetic write failure')
  expect(JSON.stringify(db.rows)).toBe(before)
})

test.each([
  null, [], {}, { ...identity, extra: true }, { ...identity, publication_kind: 'ordinary' },
  { ...identity, session_id: 'unmanaged' }, { ...identity, table_id: [] }, { ...identity, device_id: 1 },
  { ...identity, byte_size: 0 }, { ...identity, byte_size: '10' }, { ...identity, content_type: 'audio/mp4' },
  { ...identity, owner_user_id: null }, { ...identity, content_sha256: 'wrong' },
])('pure metadata rejects malformed or nonmanaged saved identity %p', bad => {
  const { makeStorageAdmission, validStorageAdmission } = admission()
  expect(validStorageAdmission(marker, bad, epoch)).toBe(false)
  expect(() => makeStorageAdmission(bad, root)).toThrow(expect.objectContaining({ code: 'MEDIA_ADMISSION_INVALID' }))
})

test.each([
  { ...scope, sessionId: 'legacy-session' }, { ...scope, sessionId: null },
  { ...scope, tableId: null }, { ...scope, deviceId: [] },
])('birth reader rejects invalid lookup scope %p before a DB read', async bad => {
  const db = ownerDeleteDb({}, { strictTransactions: true })
  await expect(db.runTransaction(tx => admission().readManagedMediaBirth(tx, bad)))
    .rejects.toMatchObject({ code: 'MEDIA_ADMISSION_INVALID' })
  expect(db.transactionAttempts[0].calls).toEqual([])
})

test('managed birth UUID case compatibility is preserved independently of storage epoch casing', async () => {
  const upperBirthId = birthId.toUpperCase()
  const upperSessionId = `managed-${upperBirthId}`
  const { db, admin } = managedMediaBirthFixture({ idFactory: () => upperBirthId })
  await admin.createManagedSession(input)
  expect(await db.runTransaction(tx => admission().readManagedMediaBirth(tx, { ...scope, sessionId: upperSessionId })))
    .toEqual({ ...legacyBirth, session_id: upperSessionId, birth_id: upperBirthId, media_admission_contract: contract })
})
