const crypto = require('crypto')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { authorizeSavedClip, completeSavedClipJob, createCloudSavedClipRepository, createSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const fixture = require('../../../../tests/fixtures/synthetic/miniprogram/private-shot-clips-v2.json')
const NOW = '2026-10-30T10:00:00.000Z'
const tick = () => new Promise(setImmediate)
function page(name) {
  let definition; global.Page = value => { definition = value }
  jest.isolateModules(() => require(`../../miniprogram/pages/${name}/index`))
  return { ...definition, route: `pages/${name}/index`, data: { ...definition.data },
    setData(value, callback) { Object.assign(this.data, value); if (callback) callback() } }
}
let previousEnvironment
beforeEach(() => { previousEnvironment = process.env.RANKING_RUNTIME_ENVIRONMENT; process.env.RANKING_RUNTIME_ENVIRONMENT = 'development' })
afterEach(() => {
  jest.dontMock('wx-server-sdk'); delete global.wx; delete global.Page; delete global.getCurrentPages
  if (previousEnvironment === undefined) delete process.env.RANKING_RUNTIME_ENVIRONMENT
  else process.env.RANKING_RUNTIME_ENVIRONMENT = previousEnvironment
})
async function harness({ revokeViewing = false, lostAck = false, userId = 'user-a', unknown = false } = {}) {
  const db = ownerDeleteDb({ ...structuredClone(fixture.authority_seed), sessions: { 'session-private': structuredClone(fixture.published_session) } })
  const savedRepo = createCloudSavedClipRepository(db)
  const storageProvider = { verifyObject: jest.fn(async () => ({ verified: true, file_id: 'cloud://private/clip.mp4' })), authorizeUpload: jest.fn() }
  const saved = []
  for (const { descriptor } of fixture.selections) {
    const input = { descriptor, idempotencyKey: descriptor.clip_id, tableId: 'table-1', deviceId: 'device-1', repo: savedRepo,
      idFactory: () => `saved-${descriptor.actor_role}`, now: NOW }
    const job = await createSavedClipJob(input)
    await completeSavedClipJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo: savedRepo, storage: storageProvider, now: NOW })
    saved.push({ input, job })
  }
  storageProvider.verifyObject.mockClear()
  const clipId = unknown ? 'unknown-clip' : fixture.selections[0].descriptor.clip_id
  if (revokeViewing) db.rows.clips[clipId].consent_status = 'revoked'
  const hash = value => crypto.createHash('sha256').update(value).digest('hex')
  for (const id of ['user-a', 'user-b']) {
    db.rows.users ||= {}; db.rows.users[id] = { schema_version: '1.0', user_id: id, status: 'active', profile: { nickname: null, avatar_file_id: null }, created_at: NOW, updated_at: NOW }
    db.rows.wechat_identities ||= {}; db.rows.wechat_identities[hash(`openid-${id}`)] = { schema_version: '1.0', identity_digest: hash(`openid-${id}`), user_id: id, created_at: NOW }
  }
  const provider = { init() {}, DYNAMIC_CURRENT_ENV: 'synthetic', database: () => db,
    getWXContext: () => ({ OPENID: userId ? `openid-${userId}` : undefined }),
    getTempFileURL: jest.fn(async () => ({ fileList: [{ status: 0, tempFileURL: 'https://synthetic.invalid/private.mp4' }] })), deleteFile: jest.fn() }
  jest.doMock('wx-server-sdk', () => provider)
  let main; jest.isolateModules(() => { main = require('../../cloudfunctions/client-api').main })
  const storage = new Map(); const requests = []
  global.wx = { cloud: { callFunction: async ({ name, data }) => {
    expect(name).toBe('client-api'); requests.push(data)
    const result = await main(data)
    if (data.action === 'clips.delete' && lostAck) { lostAck = false; throw new Error('ACK lost after commit') }
    return { result }
  } }, getRandomValues: ({ success }) => success({ randomValues: new Uint8Array(16).fill(9).buffer }),
  getStorageSync: key => storage.get(key), setStorageSync: (key, value) => storage.set(key, value), removeStorageSync: key => storage.delete(key),
  createVideoContext: () => ({ stop() {} }), showModal: ({ success }) => success({ confirm: true }) }
  const list = page('clips'); list.onLoad({}); await tick()
  const detail = page('clip-detail'); global.getCurrentPages = () => [list, detail]
  detail.onLoad({ clip_id: clipId }); await tick()
  return { db, provider, storageProvider, storage, requests, clipId, detail, list, main, saved, savedRepo }
}

test.each([false, true])('detail through actual client API tombstones owner clip; unavailable playback=%s still permits deletion', async revokeViewing => {
  const h = await harness({ revokeViewing })
  expect(Boolean(h.detail.data.playbackUrl)).toBe(!revokeViewing)
  await h.detail.requestDeletion()
  expect(h.detail.data.deletionState).toBe('accepted')
  expect(h.detail.data.playbackUrl).toBe(''); expect(h.list.data.clips).toEqual([])
  expect(h.db.rows.clips[h.clipId].retention_state).toBe('deletion_requested')
  const tombstone = h.db.rows.owner_clip_tombstones[h.clipId]
  expect(tombstone.owner_user_id).toBe('user-a'); expect(tombstone.receipt.request_id).toBe('09090909090909090909090909090909')
  expect(Object.values(h.db.rows.owner_clip_deletion_jobs)[0].status).toBe('pending_cloud')
  expect(h.storage.size).toBe(0)
  await expect(h.main({ schema_version: '1.0', action: 'clips.getPlayback', payload: { clip_id: h.clipId } })).resolves.toMatchObject({ ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } })
  await expect(h.main({ schema_version: '1.0', action: 'clips.listPage', payload: {} })).resolves.toMatchObject({ ok: true, data: { items: [] } })
  await expect(createSavedClipJob(h.saved[0].input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  const target = { jobId: h.saved[0].job.job_id, tableId: 'table-1', deviceId: 'device-1', repo: h.savedRepo, storage: h.storageProvider, now: NOW }
  await expect(authorizeSavedClip(target)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(completeSavedClipJob(target)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(h.provider.deleteFile).not.toHaveBeenCalled(); expect(h.storageProvider.verifyObject).not.toHaveBeenCalled()
})

test('real committed deletion with lost ACK reopens using same stable intent and canonical six-field receipt', async () => {
  const h = await harness({ lostAck: true }); await h.detail.requestDeletion()
  expect(h.detail.data.deletionState).toBe('unknown'); expect(h.list.data.clips).toHaveLength(1)
  const firstReceipt = structuredClone(h.db.rows.owner_clip_tombstones[h.clipId].receipt)
  const retry = page('clip-detail'); retry.onLoad({ clip_id: h.clipId }); await tick()
  expect(retry.data.deletionState).toBe('unknown'); expect(retry.data.playbackUrl).toBe('')
  global.getCurrentPages = () => [h.list, retry]; await retry.requestDeletion()
  expect(retry.data.deletionState).toBe('accepted'); expect(h.list.data.clips).toEqual([])
  const deletes = h.requests.filter(r => r.action === 'clips.delete')
  expect(deletes).toHaveLength(2); expect(deletes[1].payload).toEqual(deletes[0].payload)
  expect(h.db.rows.owner_clip_tombstones[h.clipId].receipt).toEqual(firstReceipt)
  expect(Object.values(h.db.rows.owner_clip_deletion_jobs)).toHaveLength(1)
  expect(h.provider.deleteFile).not.toHaveBeenCalled()
})

test.each([{ userId: 'user-b' }, { userId: null }, { unknown: true }])('UI intent grants no authority: actual server rejects %p without cache invalidation', async options => {
  const h = await harness(options); const before = [...h.list.data.clips]
  await h.detail.requestDeletion()
  expect(h.detail.data.deletionState).toBe('unknown'); expect(h.list.data.clips).toEqual(before)
  expect(Object.keys(h.db.rows.owner_clip_tombstones || {})).toHaveLength(0)
  expect(Object.keys(h.db.rows.owner_clip_deletion_jobs || {})).toHaveLength(0)
  expect(h.storage.size).toBe(1); expect(h.provider.deleteFile).not.toHaveBeenCalled()
})

test('unreadable local intent after real committed deletion reauthorizes freshly and server tombstone denies playback', async () => {
  const h = await harness({ lostAck: true }); await h.detail.requestDeletion()
  const before = h.requests.length
  wx.getStorageSync = () => { throw new Error('optional local storage unavailable') }
  const reopened = page('clip-detail'); reopened.onLoad({ clip_id: h.clipId }); await tick()
  expect(h.requests.slice(before).map(request => request.action)).toEqual(['clips.getPlayback'])
  expect(reopened.data.playbackUrl).toBe(''); expect(reopened.data.authorized).toBe(false)
  expect(reopened.data.errorCode).toBe('MEDIA_NOT_AVAILABLE')
  await reopened.requestDeletion()
  expect(h.requests.slice(before).map(request => request.action)).toEqual(['clips.getPlayback'])
  expect(Object.keys(h.db.rows.owner_clip_deletion_jobs)).toHaveLength(1)
})
