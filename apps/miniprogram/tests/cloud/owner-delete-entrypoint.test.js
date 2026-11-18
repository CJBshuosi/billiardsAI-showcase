const crypto = require('crypto')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { digest } = require('../../cloudfunctions/shared/shot-ownership')
const fixture = require('../../../../tests/fixtures/synthetic/miniprogram/private-shot-clips-v2.json')
const {
  authorizeSavedClip, completeSavedClipJob, createCloudSavedClipRepository, createSavedClipJob,
} = require('../../cloudfunctions/sync-api/lib/ingest')
const {
  makeStorageIdentity, createCloudMediaStorageAuthority,
} = require('../../cloudfunctions/shared/media-storage-authority')

const NOW = '2026-10-30T10:00:00.000Z'
const RED = fixture.selections[0].descriptor.clip_id
const BLUE = fixture.selections[1].descriptor.clip_id
const hash = value => crypto.createHash('sha256').update(value).digest('hex')
let previousEnvironment

beforeEach(() => {
  previousEnvironment = process.env.RANKING_RUNTIME_ENVIRONMENT
  process.env.RANKING_RUNTIME_ENVIRONMENT = 'development'
  jest.useFakeTimers().setSystemTime(new Date(NOW))
})
afterEach(() => {
  jest.dontMock('wx-server-sdk')
  jest.useRealTimers()
  jest.restoreAllMocks()
  if (previousEnvironment === undefined) delete process.env.RANKING_RUNTIME_ENVIRONMENT
  else process.env.RANKING_RUNTIME_ENVIRONMENT = previousEnvironment
})

// External SDK/provider only: authentication, main, owner deletion, publication,
// and content repositories all execute their production implementations.
async function harness({ retainedRed = false, ...options } = {}) {
  const users = {}; const identities = {}
  for (const userId of ['user-a', 'user-b', 'user-c']) {
    users[userId] = { schema_version: '1.0', user_id: userId, status: 'active',
      profile: { nickname: null, avatar_file_id: null }, created_at: NOW, updated_at: NOW }
    const identityDigest = hash(`openid-${userId}`)
    identities[identityDigest] = { schema_version: '1.0', identity_digest: identityDigest,
      user_id: userId, created_at: NOW }
  }
  const db = ownerDeleteDb({ ...structuredClone(fixture.authority_seed), users,
    wechat_identities: identities,
    sessions: { 'session-private': structuredClone(fixture.published_session) },
    session_participants: {
      replacement: { session_id: 'session-private', user_id: 'user-c' },
      blue: { session_id: 'session-private', user_id: 'user-b' },
    },
  }, options)
  if (retainedRed) {
    const identity = makeStorageIdentity(fixture.selections[0].descriptor, {
      tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private',
    })
    // Retained-old-v2 fixture boundary: this historical unmarked product origin
    // cannot enroll fresh. Seed only its original empty root; real operation
    // APIs below generate and validate every page/object/alias for owner close.
    // This is retained authority coverage, not marked publication admission.
    const enrollment = await db.runTransaction(tx => createCloudMediaStorageAuthority(db).enroll(tx, identity))
    expect(enrollment).toMatchObject({ status: 'legacy_untracked' })
    expect(db.transactionAttempts.at(-1).calls.every(call => call.method === 'get')).toBe(true)
    db.rows.clip_storage_authorities = { [identity.clip_id]: { contract: 'clip_storage_v2',
      identity: structuredClone(identity), identity_sha256: digest(identity),
      epoch: '11111111-1111-4111-8111-111111111111', high_water: 0, object_ids: [],
      admission: 'open', revision: 1, created_at: NOW } }
  }
  const savedRepo = createCloudSavedClipRepository(db)
  const storage = { authorizeUpload: jest.fn(), verifyObject: jest.fn() }
  const jobs = {}
  for (const { descriptor } of fixture.selections) {
    const input = { descriptor, idempotencyKey: descriptor.clip_id, tableId: 'table-1',
      deviceId: 'device-1', repo: savedRepo, idFactory: () => `saved-${descriptor.actor_role}`, now: NOW }
    const job = await createSavedClipJob(input)
    storage.verifyObject.mockResolvedValueOnce({ verified: true,
      file_id: `cloud://private/${descriptor.actor_role}.mp4` })
    await completeSavedClipJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1',
      repo: savedRepo, storage, now: NOW })
    jobs[descriptor.clip_id] = { input, job: structuredClone(db.rows.saved_clip_sync_jobs[job.job_id]) }
  }
  storage.verifyObject.mockClear()
  let openid = 'openid-user-a'
  const sdk = {
    init() {}, DYNAMIC_CURRENT_ENV: 'synthetic', database: () => db,
    getWXContext: () => ({ OPENID: openid }),
    getTempFileURL: jest.fn(async () => ({ fileList: [{ status: 0, tempFileURL: 'https://private.invalid/clip' }] })),
    deleteFile: jest.fn(), uploadFile: jest.fn(), callFunction: jest.fn(),
  }
  jest.doMock('wx-server-sdk', () => sdk)
  let main
  jest.isolateModules(() => { main = require('../../cloudfunctions/client-api').main })
  const action = (name, payload) => main({ schema_version: '1.0', action: name, payload })
  return { db, sdk, storage, savedRepo, jobs, action,
    as(userId) { openid = userId == null ? undefined : `openid-${userId}` },
    remove(clipId = RED, requestId = 'delete-1') { return action('clips.delete', { clip_id: clipId, request_id: requestId }) },
  }
}
function noProvider(h) {
  for (const name of ['getTempFileURL', 'deleteFile', 'uploadFile', 'callFunction']) {
    expect(h.sdk[name]).not.toHaveBeenCalled()
  }
  expect(h.storage.authorizeUpload).not.toHaveBeenCalled()
  expect(h.storage.verifyObject).not.toHaveBeenCalled()
}
function noDeletion(h) {
  expect(Object.keys(h.db.rows.owner_clip_tombstones || {})).toHaveLength(0)
  expect(Object.keys(h.db.rows.owner_clip_delete_requests || {})).toHaveLength(0)
  expect(Object.keys(h.db.rows.owner_clip_deletion_jobs || {})).toHaveLength(0)
  expect(Object.values(h.db.rows.clips).map(clip => clip.retention_state)).toEqual(['active', 'active'])
  noProvider(h)
}

// Catches wrong authenticated identity, roster-based authorization, receipt
// expansion, client timestamps, and omission of the real deletion transaction.
test.each([['user-a', RED, 'red'], ['user-b', BLUE, 'blue']])(
  'authenticated immutable %s owner deletes only their %s clip', async (userId, clipId, role) => {
    const h = await harness(); h.as(userId); h.db.setStrictTransactions(true)
    const before = structuredClone(h.db.rows)
    const result = await h.remove(clipId)
    const receipt = { schema_version: '1.0', clip_id: clipId, request_id: 'delete-1',
      deletion_job_id: `owner-delete-${hash(clipId).slice(0, 32)}`,
      status: 'deletion_requested', requested_at: NOW }
    expect(result).toEqual({ ok: true, data: receipt })
    expect(h.db.rows.owner_clip_tombstones[clipId]).toMatchObject({ owner_user_id: userId,
      actor_role: role, receipt, save_receipt_sha256: before.clips[clipId].save_receipt_sha256 })
    expect(h.db.rows.owner_clip_deletion_jobs[receipt.deletion_job_id]).toMatchObject({
      owner_user_id: userId, status: 'pending_cloud', receipt, job_kind: 'owner_delete_v1' })
    expect(h.db.rows.clips[clipId].retention_state).toBe('deletion_requested')
    const otherId = clipId === RED ? BLUE : RED
    expect(h.db.rows.clips[otherId]).toEqual(before.clips[otherId])
    expect(h.db.rows.field_shot_ownership).toEqual(before.field_shot_ownership)
    expect(h.db.rows.saved_clip_sync_jobs).toEqual(before.saved_clip_sync_jobs)
    expect(h.db.transactionAttempts.at(-1).count).toBe(13)
    expect(h.db.rows.owner_device_deletion_deliveries[receipt.deletion_job_id]).toMatchObject({ status: 'waiting_cloud', device_id: 'device-1', table_id: 'table-1', clip_id: clipId, environment_id: null, task_sha256: null })
    noProvider(h)
  },
)

test.each([['user-a', BLUE], ['user-b', RED], ['user-c', RED], ['user-c', BLUE], ['user-a', 'unknown']])(
  '%s cannot delete known foreign/replaced/unknown clip %s', async (userId, clipId) => {
    const h = await harness(); h.as(userId)
    await expect(h.remove(clipId)).resolves.toEqual({ ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } })
    noDeletion(h)
  },
)
test('unauthenticated context cannot delete or trust a client identity', async () => {
  const h = await harness(); h.as(null)
  await expect(h.remove()).resolves.toEqual({ ok: false, error: { code: 'AUTH_REQUIRED' } })
  noDeletion(h)
})

const validPayload = { clip_id: RED, request_id: 'delete-1' }
test.each([
  undefined, null, [], [validPayload], 'delete', {}, { clip_id: RED }, { request_id: 'delete-1' },
  { ...validPayload, clip_id: '' }, { ...validPayload, clip_id: '../escape' },
  { ...validPayload, clip_id: 1 }, { ...validPayload, request_id: [] },
  { ...validPayload, request_id: 'x'.repeat(129) }, { ...validPayload, request_id: '../escape' },
  ...['owner_user_id', 'user_id', 'device_id', 'file_id', 'path', 'now', 'requested_at', 'schema_version']
    .map(key => ({ ...validPayload, [key]: 'injected' })),
])('exact body contract rejects %p without deletion or provider operations', async payload => {
  const h = await harness()
  await expect(h.action('clips.delete', payload)).resolves.toEqual({ ok: false, error: { code: 'REQUEST_INVALID' } })
  noDeletion(h)
})

test.each([
  { privacy_policy: 'shared' }, { owner_user_id: null },
  { ownership_receipt_sha256: '0'.repeat(64) }, { save_receipt_sha256: '0'.repeat(64) },
])('legacy or damaged authority %p remains unavailable', async change => {
  const h = await harness(); Object.assign(h.db.rows.clips[RED], change)
  await expect(h.remove()).resolves.toEqual({ ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } })
  noDeletion(h)
})
test.each([{ consent_status: 'revoked' }, { retention_state: 'released' }, { deleted_at: NOW }])(
  'immutable owner can delete despite revoked viewing %p', async change => {
    const h = await harness(); Object.assign(h.db.rows.clips[RED], change)
    await expect(h.action('clips.getPlayback', { clip_id: RED })).resolves.toEqual({ ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } })
    await expect(h.remove()).resolves.toMatchObject({ ok: true, data: { status: 'deletion_requested', requested_at: NOW } })
    noProvider(h)
  },
)

test('retries keep the original six-field receipt across time, compaction and new request IDs', async () => {
  const h = await harness()
  const first = await h.remove(); expect(first.ok).toBe(true)
  const bytes = JSON.stringify(first.data)
  const originalJob = structuredClone(h.db.rows.owner_clip_deletion_jobs)
  jest.setSystemTime(new Date('2030-01-01T00:00:00.000Z'))
  expect(JSON.stringify((await h.remove()).data)).toBe(bytes)
  expect(JSON.stringify((await h.remove(RED, 'delete-2')).data)).toBe(bytes)
  expect(h.db.rows.owner_clip_deletion_jobs).toEqual(originalJob)
  delete h.db.rows.clips[RED]; h.db.rows.field_shot_ownership = {}; h.db.rows.owner_clip_deletion_jobs = {}
  expect(JSON.stringify((await h.remove(RED, 'delete-3')).data)).toBe(bytes)
  expect(h.db.rows.owner_clip_deletion_jobs).toEqual({})
  await expect(h.remove(BLUE)).resolves.toEqual({ ok: false, error: { code: 'REQUEST_CONFLICT' } })
  h.as('user-b')
  await expect(h.remove(RED)).resolves.toEqual({ ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } })
  noProvider(h)
})

test('authenticated request closes 64 retained-old-v2 operations within the strict 90-call owner budget', async () => {
  const h = await harness({ retainedRed: true })
  const identity = makeStorageIdentity(h.db.rows.clips[RED], {
    tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private',
  })
  const authority = createCloudMediaStorageAuthority(h.db)
  for (let i = 0; i < 64; i++) {
    const ticket = await h.db.runTransaction(tx => authority.beginOperation(tx, {
      identity, ingestJobId: 'saved-red', kind: 'upload_authorization',
    }))
    await authority.recordOperationResult({ ticket, outcome: 'external_issued',
      objects: [{ kind: ticket.target.kind, cloud_path: ticket.target.cloud_path,
        file_id: `cloud://synthetic/${ticket.operation_id}` }] })
  }
  h.db.setStrictTransactions(true)
  const first = await h.remove(); expect(first.ok).toBe(true)
  const attempt = h.db.transactionAttempts.at(-1)
  expect(attempt.count).toBe(82)
  expect(attempt.calls.filter(c => ['owner_device_deletion_deliveries', 'owner_device_deletion_receipts'].includes(c.collection)).map(c => [c.collection, c.method])).toEqual([
    ['owner_device_deletion_deliveries', 'get'], ['owner_device_deletion_receipts', 'get'], ['owner_device_deletion_deliveries', 'set'],
  ])
  expect(attempt.count).toBeLessThanOrEqual(90)
  expect(h.db.rows.owner_clip_tombstones[RED].storage_authority).toMatchObject({
    status: 'tracked_closed', contract: 'clip_storage_v2', high_water: 64,
  })
  expect(h.db.rows.clip_storage_authorities[RED].admission).toBe('closed')
  expect(await h.remove(RED, 'delete-retry')).toEqual(first)
  expect(h.db.transactionAttempts.at(-1).count).toBe(5)
  noProvider(h)
})

test('accepted deletion fences real list, playback, saved retry and publication paths', async () => {
  const h = await harness()
  const before = structuredClone(h.db.rows)
  expect((await h.action('clips.listPage', {})).data.items.map(clip => clip.clip_id)).toEqual([RED])
  const receipt = await h.remove(); expect(receipt.ok).toBe(true)
  // Stale metadata cannot bypass durable tombstone authority.
  h.db.rows.clips[RED].retention_state = 'active'
  for (const userId of ['user-a', 'user-b', 'user-c']) {
    h.as(userId)
    const expected = userId === 'user-b' ? [BLUE] : []
    for (const payload of [{}, { session_id: 'session-private' }]) {
      expect((await h.action('clips.list', payload)).data.map(clip => clip.clip_id)).toEqual(expected)
      expect((await h.action('clips.listPage', payload)).data.items.map(clip => clip.clip_id)).toEqual(expected)
    }
    await expect(h.action('clips.getPlayback', { clip_id: RED })).resolves.toEqual({ ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } })
  }
  const { input, job } = h.jobs[RED]
  const target = { jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1',
    repo: h.savedRepo, storage: h.storage, now: NOW }
  await expect(createSavedClipJob(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(h.savedRepo.create(job)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(authorizeSavedClip(target)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(completeSavedClipJob(target)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(h.savedRepo.markUploading(job.job_id)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(h.savedRepo.markCompleted(job.job_id, NOW, { file_id: 'cloud://forbidden/recreated' }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  h.db.rows.saved_clip_sync_jobs = {}; h.db.rows.saved_clip_sync_idempotency = {}; delete h.db.rows.clips[RED]
  await expect(createSavedClipJob(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(h.db.rows.clips[RED]).toBeUndefined()
  expect(h.db.rows.clips[BLUE]).toEqual(before.clips[BLUE])
  expect(h.db.rows.owner_clip_tombstones[RED].receipt).toEqual(receipt.data)
  expect(h.db.rows.owner_clip_tombstones[RED].save_receipt_sha256).toBe(before.clips[RED].save_receipt_sha256)
  expect(h.db.rows.field_shot_ownership).toEqual(before.field_shot_ownership)
  noProvider(h)
})

test('temporary database failures return the existing safe internal error and roll back', async () => {
  let failing = false
  const h = await harness({ failWrite(name) {
    if (failing && name === 'owner_clip_tombstones') throw new Error('private SDK database/storage detail')
  } })
  failing = true
  jest.spyOn(console, 'error').mockImplementation(() => {})
  await expect(h.remove()).resolves.toEqual({ ok: false, error: { code: 'INTERNAL_ERROR' } })
  noDeletion(h)
})
test('favorites and manual saving remain unsupported through the authenticated main', async () => {
  const h = await harness()
  for (const name of ['favorites.list', 'favorites.set']) {
    await expect(h.action(name, { clip_id: RED })).resolves.toEqual({ ok: false, error: { code: 'ACTION_UNSUPPORTED' } })
  }
  await expect(h.action('clips.save', { clip_id: RED })).resolves.toEqual({ ok: false, error: { code: 'REQUEST_INVALID' } })
  noDeletion(h)
})
