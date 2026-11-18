const crypto = require('crypto')
const { privateClip } = require('../helpers/private-clips')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { managedMediaBirthFixture, sessionId, input } = require('../helpers/managed-media-admission-fixture')
const { documentId, ownershipProjection, digest, saveReceipt } = require('../../cloudfunctions/shared/shot-ownership')
const { requestOwnerClipDeletion, createCloudOwnerDeleteRepository, getOwnerClipTombstone } = require('../../cloudfunctions/shared/owner-delete')
const NOW = '2026-10-29T19:00:00.000Z'
function seed(change = {}) {
  const clip = privateClip({ storage_objects: [
    { kind: 'published', file_id: 'cloud://private/clip.mp4' },
    { kind: 'staging', file_id: 'cloud://staging/clip.mp4' },
    { kind: 'derivative', file_id: 'cloud://private/thumb.jpg' },
    { kind: 'published', file_id: 'cloud://private/clip.mp4' },
  ], ...change })
  Object.assign(clip, change)
  return { clips: { [clip.clip_id]: clip }, field_shot_ownership: {
    [documentId(`${clip.session_id}:${clip.shot_id}`)]: ownershipProjection(privateClip(), 'table-1', 'device-1'),
  } }
}
function request(db, change = {}) {
  return requestOwnerClipDeletion({ clipId: privateClip().clip_id, requestId: 'request-1',
    userId: 'user-a', repo: createCloudOwnerDeleteRepository(db), now: NOW, ...change })
}
test('internal deletion authority is available', () => { expect(typeof requestOwnerClipDeletion).toBe('function') })
test('owner freezes immutable identity and unique storage inventory without storage operations', async () => {
  const db = ownerDeleteDb(seed())
  db.storage = new Proxy({}, { get() { throw new Error('storage must not be used') } })
  const receipt = await request(db)
  const jobId = `owner-delete-${crypto.createHash('sha256').update(privateClip().clip_id).digest('hex').slice(0, 32)}`
  expect(receipt).toEqual({ schema_version: '1.0', clip_id: privateClip().clip_id,
    request_id: 'request-1', deletion_job_id: jobId, status: 'deletion_requested', requested_at: NOW })
  const job = db.rows.owner_clip_deletion_jobs[jobId]
  expect(job).toMatchObject({ job_kind: 'owner_delete_v1', status: 'pending_cloud',
    owner_user_id: 'user-a', actor_role: 'red', session_id: 'session-1', shot_id: 'shot-0001',
    device_id: 'device-1', table_id: 'table-1', content_sha256: 'c8c5af84ac765d911a9ab05bc9a19d15d0b1bc5cf0654eff4469ce536410654e',
    ownership_receipt_sha256: privateClip().ownership_receipt_sha256,
    save_receipt_sha256: privateClip().save_receipt_sha256 })
  expect(job.storage_objects).toEqual([
    { kind: 'published', file_id: 'cloud://private/clip.mp4' },
    { kind: 'staging', file_id: 'cloud://staging/clip.mp4' },
    { kind: 'derivative', file_id: 'cloud://private/thumb.jpg' },
  ])
  expect(db.rows.clips[receipt.clip_id].retention_state).toBe('deletion_requested')
  expect(await getOwnerClipTombstone(db, receipt.clip_id)).toMatchObject({ owner_user_id: 'user-a', receipt })
  expect(db.rows.field_shot_ownership).toEqual(seed().field_shot_ownership)
  expect(db.rows.clips[receipt.clip_id].save_receipt_sha256).toBe(privateClip().save_receipt_sha256)
})
test.each(['user-b', 'unknown'])('known clip deletion by %s cannot mutate authority', async userId => {
  const db = ownerDeleteDb(seed()); const before = structuredClone(db.rows)
  await expect(request(db, { userId })).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
  for (const [name, rows] of Object.entries(before)) expect(db.rows[name]).toEqual(rows)
  expect(Object.keys(db.rows.owner_clip_tombstones || {})).toHaveLength(0)
})
test.each([
  { owner_user_id: null }, { schema_version: '1.0' }, { privacy_policy: 'shared' },
  { ownership_receipt_sha256: '0'.repeat(64) }, { save_receipt_sha256: '0'.repeat(64) },
  { actor_role: 'blue' }, { storage_objects: [{ kind: 'published', file_id: 'https://untrusted.invalid' }] },
])('malformed private authority %p fails closed', async change => {
  await expect(request(ownerDeleteDb(seed(change)))).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
})
test.each([{ clipId: '' }, { requestId: '../escape' }, { userId: null }, { now: 'bad' }, { now: null }])(
  'invalid service argument %p cannot start transaction', async change => {
    const db = ownerDeleteDb(seed()); db.runTransaction = jest.fn()
    await expect(request(db, change)).rejects.toMatchObject({ code: 'OWNER_DELETE_INVALID' })
    expect(db.runTransaction).not.toHaveBeenCalled()
  },
)
test('canonical receipt survives repeated requests and metadata/job/projection compaction', async () => {
  const db = ownerDeleteDb(seed()); const first = await request(db)
  expect(await request(db, { now: '2027-01-01T00:00:00.000Z' })).toEqual(first)
  delete db.rows.clips[first.clip_id]; db.rows.field_shot_ownership = {}; db.rows.owner_clip_deletion_jobs = {}
  expect(await request(db, { requestId: 'request-2' })).toEqual(first)
  expect(await request(db, { requestId: 'request-2' })).toEqual(first)
  expect(Object.keys(db.rows.owner_clip_delete_requests)).toHaveLength(2)
  await expect(request(db, { clipId: 'other-clip' })).rejects.toMatchObject({ code: 'OWNER_DELETE_CONFLICT' })
  await expect(request(db, { userId: 'user-b' })).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
})
test.each(['owner_clip_tombstones', 'owner_clip_delete_requests', 'owner_clip_deletion_jobs', 'clips'])(
  'failure writing %s atomically rolls back deletion', async failed => {
    const db = ownerDeleteDb(seed(), { failWrite(name) { if (name === failed) throw new Error('injected write failure') } })
    await expect(request(db)).rejects.toThrow('injected write failure')
    expect(db.rows).toEqual(seed())
  },
)
test('pending original selection receipt authorizes deletion without final membership', async () => {
  const clip = privateClip()
  const pending = digest({ ...saveReceipt(clip), status: 'SAVE_PENDING', clip_id: null })
  await expect(request(ownerDeleteDb(seed({ save_receipt_sha256: pending })))).resolves.toHaveProperty('status', 'deletion_requested')
})
test('malformed tombstone blocks requests without reconstruction', async () => {
  const db = ownerDeleteDb({ ...seed(), owner_clip_tombstones: { [privateClip().clip_id]: {} } })
  await expect(request(db)).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
  expect(db.rows.owner_clip_tombstones[privateClip().clip_id]).toEqual({})
})

const { createCloudContentRepository, getClipPlayback, listClips, listClipsPage } = require('../../cloudfunctions/client-api/lib/content')
function playableDb() { return ownerDeleteDb(seed({ published_at: NOW, sequence: 1 })) }
function cloudSigner() {
  return { getTempFileURL: jest.fn(async () => ({ fileList: [{ status: 0, tempFileURL: 'https://private.invalid/clip' }] })) }
}
test('delete commits first: no playback grant or signer and stale active lists stay hidden', async () => {
  const db = playableDb(); const cloud = cloudSigner(); const repo = createCloudContentRepository(db, cloud, { clock: () => Date.parse(NOW) })
  await request(db)
  // Simulate stale/restored active metadata. Durable tombstone remains authoritative.
  db.rows.clips[privateClip().clip_id].retention_state = 'active'
  await expect(getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
  expect(db.rows.clips[privateClip().clip_id]).not.toHaveProperty('playback_authorized_until')
  await expect(listClips({ userId: 'user-a', repo })).resolves.toEqual([])
  await expect(listClipsPage({ userId: 'user-a', repo })).resolves.toMatchObject({ items: [], next_cursor: null })
})
test('committed playback grant precedes delete: signer uses granted snapshot, next grant denied', async () => {
  const db = playableDb(); const cloud = cloudSigner(); const repo = createCloudContentRepository(db, cloud, { clock: () => Date.parse(NOW) })
  let enter; let resume
  const entered = new Promise(resolve => { enter = resolve })
  const paused = new Promise(resolve => { resume = resolve })
  const original = repo.createPlaybackUrl
  repo.createPlaybackUrl = async (...args) => { enter(); await paused; return original(...args) }
  const playback = getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW })
  await entered
  expect(db.rows.clips[privateClip().clip_id].playback_authorized_until).toBe('2026-10-29T19:05:00.000Z')
  await request(db)
  expect(Object.values(db.rows.owner_clip_deletion_jobs)[0].playback_authorized_until).toBe('2026-10-29T19:05:00.000Z')
  resume()
  await expect(playback).resolves.toMatchObject({ clip: { clip_id: privateClip().clip_id }, expires_in_seconds: 300 })
  await expect(getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(cloud.getTempFileURL).toHaveBeenCalledTimes(1)
})
test('grant transaction preserves longer grant and rolls back failed grant before signing', async () => {
  const db = playableDb(); const cloud = cloudSigner(); const repo = createCloudContentRepository(db, cloud, { clock: () => Date.parse(NOW) })
  db.rows.clips[privateClip().clip_id].playback_authorized_until = '2026-10-29T19:10:00.000Z'
  await getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW })
  expect(db.rows.clips[privateClip().clip_id].playback_authorized_until).toBe('2026-10-29T19:10:00.000Z')
  const failed = ownerDeleteDb(seed(), { failWrite(name) { if (name === 'clips') throw new Error('grant write failed') } })
  const failedCloud = cloudSigner()
  await expect(getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo: createCloudContentRepository(failed, failedCloud), now: NOW }))
    .rejects.toThrow('grant write failed')
  expect(failedCloud.getTempFileURL).not.toHaveBeenCalled()
})
test('malformed tombstone presence blocks playback, old lists and paged lists', async () => {
  const db = playableDb(); db.rows.owner_clip_tombstones = { [privateClip().clip_id]: {} }
  const cloud = cloudSigner(); const repo = createCloudContentRepository(db, cloud, { clock: () => Date.parse(NOW) })
  await expect(getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  await expect(listClips({ userId: 'user-a', repo })).resolves.toEqual([])
  await expect(listClipsPage({ userId: 'user-a', repo })).resolves.toMatchObject({ items: [] })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
})
test('all tombstoned page still advances cursor to a later playable clip', async () => {
  const db = playableDb()
  db.rows.clips = {}; db.rows.owner_clip_tombstones = {}
  for (let i = 0; i < 26; i += 1) {
    const clipId = `page-${String(i).padStart(2, '0')}`
    db.rows.clips[clipId] = privateClip({ clip_id: clipId, published_at: NOW, sequence: 1 })
    if (i > 0) db.rows.owner_clip_tombstones[clipId] = {}
  }
  const repo = createCloudContentRepository(db, {})
  const first = await listClipsPage({ userId: 'user-a', repo })
  expect(first.items).toEqual([]); expect(first.next_cursor).toEqual(expect.any(String))
  const second = await listClipsPage({ userId: 'user-a', repo, cursor: first.next_cursor })
  expect(second.items.map(clip => clip.clip_id)).toEqual(['page-00'])
})
test('owner authority accepts clips.delete in the actual client handler', async () => {
  const previous = process.env.RANKING_RUNTIME_ENVIRONMENT
  process.env.RANKING_RUNTIME_ENVIRONMENT = 'development'
  jest.doMock('wx-server-sdk', () => ({ init() {}, DYNAMIC_CURRENT_ENV: 'synthetic', database: () => playableDb(), getWXContext: () => ({ OPENID: 'owner' }) }))
  jest.doMock('../../cloudfunctions/client-api/lib/identity', () => ({ resolveUserId: async () => ({ user_id: 'user-a' }) }))
  try {
    let main; jest.isolateModules(() => { main = require('../../cloudfunctions/client-api').main })
    await expect(main({ schema_version: '1.0', action: 'clips.delete', payload: { clip_id: privateClip().clip_id, request_id: 'request-1' } }))
      .resolves.toMatchObject({ ok: true, data: { clip_id: privateClip().clip_id, status: 'deletion_requested' } })
  } finally {
    if (previous === undefined) delete process.env.RANKING_RUNTIME_ENVIRONMENT
    else process.env.RANKING_RUNTIME_ENVIRONMENT = previous
    jest.dontMock('wx-server-sdk'); jest.dontMock('../../cloudfunctions/client-api/lib/identity') }
})

test('concurrent owner request IDs converge on one canonical receipt and pending job', async () => {
  const db = playableDb()
  const results = await Promise.all([request(db), request(db, { requestId: 'request-2' })])
  expect(results[1]).toEqual(results[0])
  expect(Object.keys(db.rows.owner_clip_deletion_jobs)).toHaveLength(1)
  expect(Object.keys(db.rows.owner_clip_delete_requests)).toHaveLength(2)
})
test('ownership projection mismatch cannot be replaced by a matching final participant', async () => {
  const db = playableDb()
  db.rows.session_participants = { owner: { session_id: 'session-1', user_id: 'user-a', role: 'red' } }
  Object.values(db.rows.field_shot_ownership)[0].device_id = 'foreign-device'
  await expect(request(db)).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
  expect(db.rows.owner_clip_tombstones).toBeUndefined()
})
test('valid but changed owner receipt does not overwrite the immutable shot projection', async () => {
  const db = playableDb(); const clip = db.rows.clips[privateClip().clip_id]
  clip.ownership_receipt.match_revision += 1
  clip.ownership_receipt_sha256 = digest(clip.ownership_receipt)
  clip.save_receipt_sha256 = digest(saveReceipt(clip))
  await expect(request(db)).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
})
test('compacting request mappings preserves canonical clip receipt and cannot revive jobs', async () => {
  const db = playableDb(); const first = await request(db)
  db.rows.owner_clip_delete_requests = {}; db.rows.owner_clip_deletion_jobs = {}; db.rows.clips = {}
  expect(await request(db)).toEqual(first)
  expect(db.rows.owner_clip_deletion_jobs).toEqual({})
})

test('damaged frozen ownership in a tombstone never becomes deletion replay authority', async () => {
  const db = playableDb(); await request(db)
  db.rows.owner_clip_tombstones[privateClip().clip_id].ownership_receipt_sha256 = '0'.repeat(64)
  await expect(request(db)).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
})

// Immutable deletion authority must survive loss of viewer eligibility.
test.each([
  { consent_status: 'revoked' },
  { retention_state: 'released' },
  { deleted_at: '2026-10-29T18:59:00.000Z' },
])('immutable owner can request deletion with unavailable playback %p', async change => {
  const db = ownerDeleteDb(seed(change)); const cloud = cloudSigner()
  const repo = createCloudContentRepository(db, cloud, { clock: () => Date.parse(NOW) })
  await expect(getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  await expect(listClips({ userId: 'user-a', repo })).resolves.toEqual([])
  await expect(request(db, { userId: 'user-b' })).rejects.toMatchObject({ code: 'OWNER_DELETE_UNAVAILABLE' })
  await expect(request(db)).resolves.toMatchObject({ status: 'deletion_requested' })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
  expect(Object.values(db.rows.owner_clip_deletion_jobs)[0]).toMatchObject({ owner_user_id: 'user-a', status: 'pending_cloud' })
})
function playbackClock() {
  let time = Date.parse(NOW)
  return { clock: () => time, advance(milliseconds) { time += milliseconds } }
}
function pauseBeforeSigner(repo) {
  let enter; let release
  const entered = new Promise(resolve => { enter = resolve })
  const paused = new Promise(resolve => { release = resolve })
  const original = repo.createPlaybackUrl
  repo.createPlaybackUrl = async (...args) => { enter(); await paused; return original(...args) }
  return { entered, release }
}
test('signer resumed after committed grant expiration cannot mint a URL after deletion', async () => {
  const db = playableDb(); const cloud = cloudSigner(); const time = playbackClock()
  const repo = createCloudContentRepository(db, cloud, time)
  const pause = pauseBeforeSigner(repo)
  const playback = getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW })
  await pause.entered
  await request(db)
  time.advance(300001); pause.release()
  await expect(playback).rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
})
test('prior grant resumed before expiry caps requested URL lifetime to remaining authority', async () => {
  const db = playableDb(); const cloud = cloudSigner(); const time = playbackClock()
  const repo = createCloudContentRepository(db, cloud, time)
  const pause = pauseBeforeSigner(repo)
  const playback = getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW })
  await pause.entered
  await request(db)
  time.advance(120250); pause.release()
  await expect(playback).resolves.toMatchObject({ expires_in_seconds: 179 })
  expect(cloud.getTempFileURL).toHaveBeenCalledWith({ fileList: [{ fileID: 'cloud://private/clip.mp4', maxAge: 179 }] })
})
test('provider response received after grant deadline is discarded', async () => {
  const db = playableDb(); const time = playbackClock()
  const cloud = { getTempFileURL: jest.fn(async () => {
    time.advance(300001)
    return { fileList: [{ status: 0, tempFileURL: 'https://private.invalid/late' }] }
  }) }
  const repo = createCloudContentRepository(db, cloud, time)
  await expect(getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(cloud.getTempFileURL).toHaveBeenCalledTimes(1)
})
test('provider delay reduces returned lifetime within committed authority', async () => {
  const db = playableDb(); const time = playbackClock()
  const cloud = { getTempFileURL: jest.fn(async () => {
    time.advance(100250)
    return { fileList: [{ status: 0, tempFileURL: 'https://private.invalid/delayed' }] }
  }) }
  const repo = createCloudContentRepository(db, cloud, time)
  await expect(getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: NOW }))
    .resolves.toMatchObject({ expires_in_seconds: 199 })
})
test('request-level time cannot extend a playback grant beyond trusted repository time', async () => {
  const db = playableDb(); const time = playbackClock(); const cloud = cloudSigner()
  const repo = createCloudContentRepository(db, cloud, time)
  await getClipPlayback({ clipId: privateClip().clip_id, userId: 'user-a', repo, now: '2099-01-01T00:00:00.000Z' })
  expect(db.rows.clips[privateClip().clip_id].playback_authorized_until).toBe('2026-10-29T19:05:00.000Z')
})

const { makeStorageIdentity, createCloudMediaStorageAuthority } = require('../../cloudfunctions/shared/media-storage-authority')
async function trackedDeletionDb(options, { pending = false } = {}) {
  // Actual trusted allocation supplies positive provenance; the separate frozen
  // owner projection below remains the owner deletion authorization boundary.
  const { db, admin } = managedMediaBirthFixture({}, options)
  await admin.createManagedSession(input)
  const clip = privateClip({ session_id: sessionId })
  if (pending) clip.save_receipt_sha256 = digest({ ...saveReceipt(clip), status: 'SAVE_PENDING', clip_id: null })
  const identity = makeStorageIdentity(clip, { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })
  const authority = createCloudMediaStorageAuthority(db)
  await db.runTransaction(tx => authority.enroll(tx, identity))
  const ticket = await db.runTransaction(tx => authority.beginOperation(tx, { identity, ingestJobId: 'fresh-ingest', kind: 'upload_authorization' }))
  const publication = seed({ session_id: sessionId, save_receipt_sha256: clip.save_receipt_sha256 })
  publication.field_shot_ownership = { [documentId(`${clip.session_id}:${clip.shot_id}`)]: ownershipProjection(clip, 'table-1', 'device-1') }
  Object.assign(db.rows, publication)
  return { db, identity, authority, ticket }
}
test('owner request closes tracked admission in its own transaction and preserves original receipt', async () => {
  const { db, authority, identity, ticket } = await trackedDeletionDb()
  const receipt = await request(db)
  const frozen = db.rows.owner_clip_tombstones[identity.clip_id].storage_authority
  expect(frozen).toMatchObject({ status: 'tracked_closed', contract: 'clip_storage_v2', high_water: 1, epoch: ticket.epoch })
  expect(Object.values(db.rows.owner_clip_deletion_jobs)[0].storage_authority).toEqual(frozen)
  await expect(db.runTransaction(tx => authority.beginOperation(tx, { identity, ingestJobId: 'late', kind: 'upload_authorization' }))).rejects.toMatchObject({ code: 'STORAGE_AUTHORITY_CONFLICT' })
  await authority.recordOperationResult({ ticket, outcome: 'external_issued', objects: [{ kind: 'staging', cloud_path: ticket.target.cloud_path, file_id: 'cloud://late/object' }], reason: null })
  expect(await request(db, { requestId: 'second-request' })).toEqual(receipt)
  expect(db.rows.owner_clip_tombstones[identity.clip_id].storage_authority).toEqual(frozen)
})
test('legacy missing authority remains untracked in tombstone and deletion job', async () => {
  const db = ownerDeleteDb(seed()); await request(db)
  expect(db.rows.owner_clip_tombstones[privateClip().clip_id].storage_authority).toMatchObject({ status: 'legacy_untracked' })
  expect(Object.values(db.rows.owner_clip_deletion_jobs)[0].storage_authority).toMatchObject({ status: 'legacy_untracked' })
  expect(Object.keys(db.rows.clip_storage_authorities || {})).toHaveLength(0)
})
test('owner transaction rolls admission closure back when later tombstone write fails', async () => {
  let fail = false
  const { db, identity } = await trackedDeletionDb({ failWrite(name) { if (fail && name === 'owner_clip_tombstones') throw new Error('close rollback') } })
  const before = structuredClone(db.rows); fail = true
  await expect(request(db)).rejects.toThrow('close rollback'); expect(db.rows).toEqual(before)
  expect(db.rows.clip_storage_authorities[identity.clip_id].admission).toBe('open')
})

test.each(['SAVED', 'SAVE_PENDING'])('strict owner %s receipt closes all 64 entries within 90 calls and replays immutable bytes', async receiptState => {
  const { db, authority, identity, ticket } = await trackedDeletionDb(undefined, { pending: receiptState === 'SAVE_PENDING' })
  const tickets = [ticket]
  for (let i = 1; i < 64; i++) tickets.push(await db.runTransaction(tx => authority.beginOperation(tx, { identity, ingestJobId: 'fresh-ingest', kind: 'upload_authorization' })))
  for (const t of tickets) await authority.recordOperationResult({ ticket: t, outcome: 'external_issued', objects: [{ kind: t.target.kind, cloud_path: t.target.cloud_path, file_id: `cloud://owner/${t.operation_id}` }] })
  db.setStrictTransactions(true)
  const receipt = await request(db); expect(db.transactionAttempts.at(-1).count).toBe(82)
  expect(db.transactionAttempts.at(-1).count).toBeLessThanOrEqual(90)
  expect(db.transactionAttempts.at(-1).calls.filter(c => ['owner_device_deletion_deliveries', 'owner_device_deletion_receipts'].includes(c.collection)).map(c => [c.collection, c.method])).toEqual([
    ['owner_device_deletion_deliveries', 'get'], ['owner_device_deletion_receipts', 'get'], ['owner_device_deletion_deliveries', 'set'],
  ])
  const bytes = JSON.stringify(receipt)
  expect(db.rows.owner_clip_tombstones[identity.clip_id].storage_authority.high_water).toBe(64)
  expect(db.rows.owner_clip_tombstones[identity.clip_id].save_receipt_sha256).toBe(identity.save_receipt_sha256)
  expect(JSON.stringify(await request(db))).toBe(bytes)
  expect(db.transactionAttempts.at(-1).count).toBe(4)
  expect(JSON.stringify(await request(db, { requestId: 'request-two' }))).toBe(bytes)
  expect(db.transactionAttempts.at(-1).count).toBe(5)
})
