const crypto = require('crypto')
const fs = require('fs')
const path = require('path')

const {
  authorizeSavedClip,
  completeSavedClipJob,
  createCloudSavedClipRepository,
  createSavedClipJob,
} = require('../../cloudfunctions/sync-api/lib/ingest')
const {
  createCloudContentRepository,
  getClipPlayback,
  listClips,
} = require('../../cloudfunctions/client-api/lib/content')


const { ownershipSeed } = require('../helpers/private-clips')
const { canonicalJson, digest } = require('../../cloudfunctions/shared/shot-ownership')
const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
const { createHandler } = require('../../cloudfunctions/sync-api')

const fixture = JSON.parse(fs.readFileSync(path.resolve(
  __dirname, '../../../../tests/fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json',
)))
const privateFixture = JSON.parse(fs.readFileSync(path.resolve(
  __dirname, '../../../../tests/fixtures/synthetic/miniprogram/private-shot-clips-v2.json',
)))


function createCloudDatabase(seed = {}) {
  seed = { ...ownershipSeed(), ...seed }
  const collections = new Map()
  for (const [name, records] of Object.entries(seed)) {
    collections.set(name, new Map(Object.entries(records)))
  }
  function collection(name) {
    if (!collections.has(name)) collections.set(name, new Map())
    const records = collections.get(name)
    let filters = {}
    let order = null
    let maximum = null
    return {
      doc(id) {
        return {
          async get() { return { data: records.get(id) || null } },
          async set({ data }) { records.set(id, data) },
          async update({ data }) { records.set(id, { ...records.get(id), ...data }) },
        }
      },
      where(value) { filters = { ...filters, ...value }; return this },
      orderBy(field, direction) { order = { field, direction }; return this },
      limit(value) { maximum = value; return this },
      async get() {
        let data = [...records.values()].filter(record =>
          Object.entries(filters).every(([key, value]) => value && typeof value === 'object' ? record[key] <= value.lte : record[key] === value))
        if (order) data.sort((left, right) => {
          const compared = typeof left[order.field] === 'number' ? left[order.field] - right[order.field]
            : String(left[order.field]).localeCompare(String(right[order.field]))
          return order.direction === 'desc' ? -compared : compared
        })
        if (maximum != null) data = data.slice(0, maximum)
        return { data }
      },
    }
  }
  return {
    command: { lte: value => ({ lte: value }) },
    collections,
    collection,
    async runTransaction(callback) {
      const snapshot = new Map([...collections].map(([name, records]) => [name,
        new Map([...records].map(([key, value]) => [key, JSON.parse(JSON.stringify(value))]))]))
      try { return await callback({ collection }) } catch (error) {
        collections.clear()
        for (const [name, records] of snapshot) collections.set(name, records)
        throw error
      }
    },
  }
}


test('canonical V2 saved descriptor stays private to historical owner years later', async () => {
  const descriptor = fixture.descriptor
  expect(canonicalJson(descriptor)).toBe(fixture.descriptor_json)
  expect(digest(descriptor)).toBe(fixture.descriptor_sha256)
  expect(canonicalJson(descriptor.ownership_receipt)).toBe(fixture.ownership_receipt_json)
  expect(digest(descriptor.ownership_receipt)).toBe(descriptor.ownership_receipt_sha256)
  expect(digest(fixture.save_receipt)).toBe(descriptor.save_receipt_sha256)
  const db = createCloudDatabase({
    sessions: { 'session-1': fixture.published_session },
    session_participants: {
      'participant-a': { session_id: 'session-1', user_id: 'user-a' },
      'participant-b': { session_id: 'session-1', user_id: 'user-b' },
    },
  })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob({
    descriptor,
    idempotencyKey: descriptor.clip_id,
    tableId: 'table-1',
    deviceId: 'device-1',
    repo,
    idFactory: () => 'saved-job-1',
    now: '2026-10-28T14:06:00.000Z',
  })
  const uploaded = Buffer.from(fixture.mp4_hex, 'hex')
  const storage = {
    authorizeUpload: jest.fn().mockResolvedValue({
      method: 'PUT', url: 'https://upload.invalid/saved-1', headers: { authorization: 'signed' },
    }),
    verifyObject: jest.fn().mockImplementation(async expected => ({
      verified: uploaded.length === expected.byte_size
        && crypto.createHash('sha256').update(uploaded).digest('hex') === expected.content_sha256,
      file_id: 'cloud://development/private/saved-1.mp4',
    })),
  }
  await authorizeSavedClip({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
  })
  await completeSavedClipJob({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
    now: '2026-11-01T14:07:00.000Z',
  })

  const cloud = { getTempFileURL: jest.fn().mockImplementation(async ({ fileList }) => ({
    fileList: [{
      status: 0,
      tempFileURL: `https://temporary.invalid/${encodeURIComponent(fileList[0].fileID)}`,
    }],
  })) }
  const contentRepo = createCloudContentRepository(db, cloud, { clock: () => Date.parse('2046-11-01T14:08:00.000Z') })
  const listed = await listClips({
    sessionId: 'session-1', userId: 'user-a', repo: contentRepo,
    now: '2046-11-01T14:08:00.000Z',
  })
  expect(listed).toEqual([expect.objectContaining({
    clip_id: descriptor.clip_id,
    shot_id: 'shot-0001',
    publication_kind: 'saved_postgame_clip',
    retention_policy: 'saved_until_deleted_v1', available_until: null,
  })])
  expect(JSON.stringify(listed)).not.toContain('shot-0002')
  await expect(getClipPlayback({
    clipId: descriptor.clip_id, userId: 'user-a', repo: contentRepo,
    now: '2046-11-01T14:08:00.000Z',
  })).resolves.toMatchObject({ url: expect.stringContaining('saved-1.mp4'), expires_in_seconds: 300 })
  await expect(listClips({ userId: 'outsider', repo: contentRepo })).resolves.toEqual([])
  await expect(getClipPlayback({
    clipId: descriptor.clip_id, userId: 'outsider', repo: contentRepo,
  })).rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  await expect(getClipPlayback({ clipId: descriptor.clip_id, userId: 'user-b', repo: contentRepo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(db.collections.get('clips').size).toBe(1)
  expect(cloud.getTempFileURL).toHaveBeenCalledTimes(1)
  expect(cloud.getTempFileURL).toHaveBeenCalledWith({ fileList: [{
    fileID: 'cloud://development/private/saved-1.mp4', maxAge: 300,
  }] })
  expect(db.collections.get('clips').get(descriptor.clip_id)).toMatchObject({
    ordinary_expires_at: null, retention_expires_at: null,
  })
})

// Catches final-roster recipients, rewriting the original pending selection,
// and returning/signing a known foreign clip through the actual action routes.
test('actual Python red-pending and blue-saved selections publish only to their start owners through signed routes', async () => {
  const independentCanonical = value => JSON.stringify((function sorted(item) {
    if (Array.isArray(item)) return item.map(sorted)
    if (item && typeof item === 'object') return Object.fromEntries(
      Object.keys(item).sort().map(key => [key, sorted(item[key])]),
    )
    return item
  })(value))
  const independentHash = value => crypto.createHash('sha256').update(value, 'utf8').digest('hex')
  expect(privateFixture.selections.map(value => value.save_receipt.status)).toEqual(['SAVE_PENDING', 'SAVED'])
  expect(privateFixture.selections[0].save_receipt.clip_id).toBeNull()
  expect(privateFixture.source_facts.map(value => value.actor_user_id)).toEqual(['user-a', 'user-b'])
  expect(privateFixture.published_session.participants.map(value => value.user_id)).toEqual(['user-c', 'user-b'])
  for (const selection of privateFixture.selections) {
    const { descriptor } = selection
    expect(independentCanonical(descriptor)).toBe(selection.descriptor_json)
    expect(independentHash(selection.descriptor_json)).toBe(selection.descriptor_sha256)
    expect(independentCanonical(descriptor.ownership_receipt)).toBe(selection.ownership_receipt_json)
    expect(independentHash(selection.ownership_receipt_json)).toBe(descriptor.ownership_receipt_sha256)
    expect(independentCanonical(selection.save_receipt)).toBe(selection.save_receipt_json)
    expect(independentHash(selection.save_receipt_json)).toBe(descriptor.save_receipt_sha256)
  }
  const db = createCloudDatabase({
    ...structuredClone(privateFixture.authority_seed),
    sessions: { 'session-private': structuredClone(privateFixture.published_session) },
    session_participants: {
      replacement: { session_id: 'session-private', user_id: 'user-c' },
      blue: { session_id: 'session-private', user_id: 'user-b' },
    },
  })
  const uploaded = Buffer.from(privateFixture.mp4_hex, 'hex')
  const storage = {
    authorizeUpload: jest.fn(async () => ({ method: 'PUT', url: 'https://upload.invalid/private', headers: {} })),
    verifyObject: jest.fn(async expected => ({
      verified: uploaded.length === expected.byte_size
        && independentHash(uploaded) === expected.content_sha256,
      file_id: `cloud://development/private/${expected.clip_id}.mp4`,
    })),
  }
  let nextId = 0
  function newHandler() {
    return createHandler({
      auth: {
        nowSeconds: () => 1793268000,
        deviceRepo: { getActiveDevice: async () => ({ device_id: 'device-1', table_id: 'table-1', status: 'active', secret: 'development-device-secret' }) },
        nonceRepo: { consume: async () => true },
      },
      savedClipRepo: createCloudSavedClipRepository(db), storage,
      idFactory: () => `private-saved-job-${++nextId}`,
      nowIso: () => '2026-10-29T10:01:00.000Z',
    })
  }
  async function signed(handler, value, path = '/v1/sync/saved-clips') {
    const body = JSON.stringify(value)
    const timestamp = '1793268000'; const nonce = crypto.randomUUID()
    const bodyHash = independentHash(body)
    const signature = crypto.createHmac('sha256', 'development-device-secret').update(canonicalSignatureInput({ method: 'POST', path, timestamp, nonce, bodyHash })).digest('hex')
    const response = await handler({ httpMethod: 'POST', path, body, headers: {
      'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce,
      'x-body-sha256': bodyHash, 'x-signature': signature, 'x-idempotency-key': value.clip_id || 'private-request',
    } })
    expect(response.statusCode).toBe(200)
    return JSON.parse(response.body).data
  }
  for (const { descriptor } of privateFixture.selections) {
    const handler = newHandler()
    const created = await signed(handler, descriptor)
    expect(await signed(handler, descriptor)).toEqual(created)
    await signed(handler, {}, `/v1/sync/saved-clips/${created.job_id}/media-authorization`)
    // Completion commits, but its ACK is deliberately discarded by the caller.
    await signed(handler, {}, `/v1/sync/saved-clips/${created.job_id}/complete`)
    const restarted = newHandler()
    const retry = await signed(restarted, descriptor)
    expect(retry).toMatchObject({ job_id: created.job_id, status: 'completed', clip_id: descriptor.clip_id })
    await signed(restarted, {}, `/v1/sync/saved-clips/${created.job_id}/complete`)
    expect(db.collections.get('saved_clip_sync_jobs').get(created.job_id).descriptor).toEqual(descriptor)
    expect(db.collections.get('clips').get(descriptor.clip_id)).toMatchObject({
      owner_user_id: descriptor.owner_user_id, actor_role: descriptor.actor_role,
      save_receipt_sha256: descriptor.save_receipt_sha256,
      retention_policy: 'saved_until_deleted_v1', ordinary_expires_at: null, retention_expires_at: null,
    })
  }
  expect(storage.authorizeUpload).toHaveBeenCalledTimes(2)
  expect(storage.verifyObject).toHaveBeenCalledTimes(2)
  expect(db.collections.get('clips').size).toBe(2)
  expect(db.collections.get('field_shot_ownership').size).toBe(2)

  let currentUser = 'user-a'
  const signer = jest.fn(async ({ fileList }) => ({ fileList: [{ status: 0, tempFileURL: `https://temporary.invalid/${encodeURIComponent(fileList[0].fileID)}` }] }))
  const previousEnvironment = process.env.RANKING_RUNTIME_ENVIRONMENT
  process.env.RANKING_RUNTIME_ENVIRONMENT = 'development'
  jest.doMock('wx-server-sdk', () => ({
    init() {}, DYNAMIC_CURRENT_ENV: 'synthetic', database: () => db,
    getWXContext: () => ({ OPENID: `openid-${currentUser}` }), getTempFileURL: signer,
  }))
  jest.doMock('../../cloudfunctions/client-api/lib/identity', () => ({ resolveUserId: async () => ({ user_id: currentUser }) }))
  try {
    let client
    jest.isolateModules(() => { client = require('../../cloudfunctions/client-api').main })
    const action = (name, payload = {}) => client({ schema_version: '1.0', action: name, payload })
    for (const [user, shots] of [['user-a', ['shot-0001']], ['user-b', ['shot-0002']], ['user-c', []], ['outsider', []]]) {
      currentUser = user
      for (const scope of [{}, { session_id: 'session-private' }]) {
        const page = await action('clips.listPage', { ...scope, owner_user_id: 'user-a' })
        expect(page.ok).toBe(true)
        expect(page.data.items.map(value => value.shot_id)).toEqual(shots)
        expect(page.data.next_cursor).toBeNull()
        expect(JSON.stringify(page.data)).not.toContain('owner_user_id')
        const legacy = await action('clips.list', scope)
        expect(legacy.data.map(value => value.shot_id)).toEqual(shots)
      }
      for (const { descriptor } of privateFixture.selections) {
        const before = signer.mock.calls.length
        const playback = await action('clips.getPlayback', { clip_id: descriptor.clip_id, owner_user_id: descriptor.owner_user_id })
        if (shots.includes(descriptor.shot_id)) {
          expect(playback.ok).toBe(true)
          expect(playback.data).toMatchObject({ url: expect.any(String), clip: { clip_id: descriptor.clip_id, shot_id: descriptor.shot_id, actor_role: descriptor.actor_role } })
          expect(playback.data.clip).not.toHaveProperty('owner_user_id')
          expect(playback.data.clip).not.toHaveProperty('ownership_receipt')
          expect(signer).toHaveBeenCalledTimes(before + 1)
        } else {
          expect(playback.ok).toBe(false)
          expect(playback.error.code).toBe('MEDIA_NOT_AVAILABLE')
          expect(signer).toHaveBeenCalledTimes(before)
        }
      }
    }
  } finally {
    jest.dontMock('wx-server-sdk')
    jest.dontMock('../../cloudfunctions/client-api/lib/identity')
    if (previousEnvironment === undefined) delete process.env.RANKING_RUNTIME_ENVIRONMENT
    else process.env.RANKING_RUNTIME_ENVIRONMENT = previousEnvironment
  }
})
