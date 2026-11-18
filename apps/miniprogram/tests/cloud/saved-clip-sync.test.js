const crypto = require('crypto')

const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
const { createHandler } = require('../../cloudfunctions/sync-api')
const {
  authorizeSavedClip,
  completeSavedClipJob,
  createCloudSavedClipRepository,
  createSavedClipJob,
  validateSavedClipInput,
} = require('../../cloudfunctions/sync-api/lib/ingest')


const SECRET = 'development-device-secret'
const NOW_SECONDS = 1788696000

const ANALYSIS_ARTIFACT_ID = 'd'.repeat(64)
const fixture = require('../../../../tests/fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json')
const SAVE_RECEIPT_SHA256 = fixture.descriptor.save_receipt_sha256
const { ownershipSeed, historyId } = require('../helpers/private-clips')

function clipId(
  sessionId = 'session-1', shotId = 'shot-0001', jobId = 'job-1',
  artifactId = ANALYSIS_ARTIFACT_ID,
) {
  const digest = crypto.createHash('sha256')
    .update(JSON.stringify([sessionId, shotId, jobId, artifactId]))
    .digest('hex')
  return `clip-${digest}`
}

function descriptor(change = {}) {
  return {
    ...structuredClone(fixture.descriptor),
    schema_version: '2.0',
    publication_kind: 'saved_postgame_clip',
    clip_id: clipId(),
    session_id: 'session-1',
    shot_id: 'shot-0001',
    job_id: 'job-1',
    artifact_id: ANALYSIS_ARTIFACT_ID,
    postgame_manifest_sha256: 'b'.repeat(64),
    save_request_id: 'save-request-1',
    save_receipt_sha256: SAVE_RECEIPT_SHA256,
    source_kind: 'usb_live',
    start_ms: 2500,
    end_ms: 5500,
    duration_ms: 3000,
    content_sha256: 'a'.repeat(64),
    byte_size: 1024,
    content_type: 'video/mp4',
    ...change,
  }
}

const PUBLISHED_SESSION = {
  schema_version: '1.0',
  session_id: 'session-1',
  artifact_id: 'field-artifact-1',
  status: 'ended',
  sync_status: 'published',
  ended_at_utc: '2026-08-26T14:05:00.000Z',
  postgame_manifest_sha256: 'b'.repeat(64),
  evidence_source_kind: 'usb_live',
  participants: [{ user_id: 'user-a', role: 'red', group: 'open' }],
  shots: [{
    schema_version: '1.0', shot_id: 'shot-0001', sequence: 1,
    started_ms: 3000, settled_ms: 5000, live_job_id: 'job-1', actor_role: 'red',
  }],
  _sync_publication: {
    table_id: 'table-1', device_id: 'device-1', content_sha256: 'c'.repeat(64),
  },
}

function createCloudDatabase(seed = {}, { failWrite } = {}) {
  seed = { ...ownershipSeed(), ...seed }
  const collections = new Map()
  for (const [name, records] of Object.entries(seed)) {
    collections.set(name, new Map(Object.entries(records)))
  }
  let transactions = Promise.resolve()
  function collection(name) {
    if (!collections.has(name)) collections.set(name, new Map())
    const records = collections.get(name)
    return {
      where(filters) {
        return { orderBy() { return this }, limit() { return this }, async get() {
          return { data: [...records.values()].filter(row => Object.entries(filters).every(([key, value]) =>
            value && typeof value === 'object' ? row[key] <= value.lte : row[key] === value))
            .sort((a, b) => b.resulting_revision - a.resulting_revision).slice(0, 1) }
        } }
      },
      doc(id) {
        return {
          async get() { return { data: records.get(id) || null } },
          async set({ data }) {
            if (failWrite) failWrite(name, id)
            records.set(id, data)
          },
          async update({ data }) {
            if (failWrite) failWrite(name, id)
            records.set(id, { ...records.get(id), ...data })
          },
        }
      },
    }
  }
  return {
    command: { lte: value => ({ lte: value }) },
    collections,
    collection,
    async runTransaction(callback) {
      const current = transactions.then(async () => {
        const snapshot = structuredClone(collections)
        try { return await callback({ collection }) } catch (error) {
          collections.clear()
          for (const [name, records] of snapshot) collections.set(name, records)
          throw error
        }
      })
      transactions = current.catch(() => {})
      return current
    },
  }
}

function input(repo, value = descriptor()) {
  return {
    descriptor: value,
    idempotencyKey: value.clip_id,
    tableId: 'table-1',
    deviceId: 'device-1',
    repo,
    idFactory: () => 'saved-job-1',
    now: '2026-08-26T14:06:00.000Z',
  }
}

test('fact-only published session accepts one saved clip and exposes nothing before verified completion', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  expect(job).toMatchObject({ job_id: 'saved-job-1', status: 'pending', clip_id: descriptor().clip_id })
  expect(db.collections.get('clips')?.size || 0).toBe(0)

  const storage = {
    authorizeUpload: jest.fn().mockResolvedValue({
      method: 'PUT', url: 'https://upload.invalid/private', headers: { authorization: 'signed' },
    }),
    verifyObject: jest.fn().mockResolvedValue({
      verified: true,
      file_id: 'cloud://development/private/saved.mp4',
      staging_file_id: 'cloud://development/staging/saved.mp4',
    }),
  }
  await authorizeSavedClip({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
  })
  expect(storage.authorizeUpload).toHaveBeenCalledWith(expect.objectContaining({
    object_key: `saved-clips/table-1/device-1/session-1/${descriptor().clip_id}.mp4`,
    byte_size: 1024, content_sha256: 'a'.repeat(64), media_type: 'video/mp4',
  }))
  expect(db.collections.get('clips')?.size || 0).toBe(0)

  await expect(completeSavedClipJob({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
    now: '2026-08-26T14:07:00.000Z',
  })).resolves.toMatchObject({ status: 'completed', clip_id: descriptor().clip_id })
  expect(storage.verifyObject).toHaveBeenCalledWith(expect.objectContaining({
    object_key: `saved-clips/table-1/device-1/session-1/${descriptor().clip_id}.mp4`,
    publish_object_key: `published/saved-clips/table-1/device-1/session-1/${descriptor().clip_id}.mp4`,
  }))
  expect(db.collections.get('clips').get(descriptor().clip_id)).toMatchObject({
    publication_kind: 'saved_postgame_clip',
    upload_status: 'published',
    consent_status: 'owner', privacy_policy: 'shot_owner_only_v1', owner_user_id: 'user-a', actor_role: 'red',
    retention_state: 'active',
    retention_policy: 'saved_until_deleted_v1',
    ordinary_expires_at: null,
    retention_expires_at: null,
    session_id: 'session-1', shot_id: 'shot-0001', artifact_id: ANALYSIS_ARTIFACT_ID,
    save_request_id: 'save-request-1', save_receipt_sha256: SAVE_RECEIPT_SHA256,
    storage_file_id: 'cloud://development/private/saved.mp4',
  })
  expect(db.collections.get('sessions').get('session-1')).toEqual(PUBLISHED_SESSION)
  expect(db.collections.get('shots')?.size || 0).toBe(0)
})

test('saved publication after the old 48 hour deadline owns its indefinite policy', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  await completeSavedClipJob({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo,
    storage: { verifyObject: async () => ({ verified: true, file_id: 'cloud://private/saved.mp4' }) },
    now: '2026-09-01T00:00:00.000Z',
  })
  expect(db.collections.get('clips').get(descriptor().clip_id)).toMatchObject({
    retention_policy: 'saved_until_deleted_v1', ordinary_expires_at: null,
    retention_expires_at: null, retention_state: 'active',
  })
  expect(() => validateSavedClipInput({
    descriptor: descriptor({ retention_policy: 'saved_until_deleted_v1' }),
    idempotencyKey: descriptor().clip_id, tableId: 'table-1', deviceId: 'device-1',
  })).toThrow(expect.objectContaining({ code: 'SYNC_INVALID' }))
})

test('saved clip create is exact-replay idempotent and changed lineage conflicts', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const first = await createSavedClipJob(input(repo))
  await expect(createSavedClipJob(input(repo))).resolves.toEqual(first)
  for (const changed of [
    descriptor({ content_sha256: 'd'.repeat(64) }),
    descriptor({ save_receipt_sha256: 'f'.repeat(64) }),
    descriptor({ byte_size: 1025 }),
  ]) {
    await expect(createSavedClipJob({ ...input(repo, changed), idempotencyKey: descriptor().clip_id }))
      .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  }
  expect(db.collections.get('saved_clip_sync_jobs').size).toBe(1)
})

test('saved clip descriptor enforces the fixed inclusive 256 MiB boundary', () => {
  const common = {
    idempotencyKey: descriptor().clip_id,
    tableId: 'table-1',
    deviceId: 'device-1',
  }
  expect(() => validateSavedClipInput({
    ...common, descriptor: descriptor({ byte_size: 256 * 1024 * 1024 }),
  })).not.toThrow()
  expect(() => validateSavedClipInput({
    ...common, descriptor: descriptor({ byte_size: 256 * 1024 * 1024 + 1 }),
  })).toThrow(expect.objectContaining({ code: 'SYNC_INVALID' }))
})

test.each([
  ['unsafe integer time range', {
    start_ms: Number.MAX_SAFE_INTEGER - 1000,
    end_ms: Number.MAX_SAFE_INTEGER + 1000,
    duration_ms: 2000,
  }],
  ['over five minute duration', { end_ms: 302501, duration_ms: 300001 }],
  ['non-digest artifact identity', { artifact_id: 'analysis-1' }],
])('saved clip descriptor rejects %s', (_label, change) => {
  const value = descriptor(change)
  expect(() => validateSavedClipInput({
    descriptor: value,
    idempotencyKey: value.clip_id,
    tableId: 'table-1',
    deviceId: 'device-1',
  })).toThrow(expect.objectContaining({ code: 'SYNC_INVALID' }))
})

test.each(['save_request_id', 'save_receipt_sha256'])(
  'signed saved clip descriptor requires persisted selection authority %s', field => {
    const value = descriptor()
    delete value[field]
    expect(() => validateSavedClipInput({
      descriptor: value,
      idempotencyKey: value.clip_id,
      tableId: 'table-1',
      deviceId: 'device-1',
    })).toThrow(expect.objectContaining({ code: 'SYNC_INVALID' }))
  },
)

test.each([
  ['missing published session', {}, 'SYNC_PARTIAL'],
  ['foreign device', { sessions: { 'session-1': {
    ...PUBLISHED_SESSION,
    _sync_publication: { ...PUBLISHED_SESSION._sync_publication, device_id: 'foreign' },
  } } }, 'SYNC_CONFLICT'],
  ['open session', { sessions: { 'session-1': { ...PUBLISHED_SESSION, status: 'open' } } }, 'SYNC_PARTIAL'],
])('%s fails closed before storing a saved job', async (_label, seed, code) => {
  const db = createCloudDatabase(seed)
  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db))))
    .rejects.toMatchObject({ code })
  expect(db.collections.get('saved_clip_sync_jobs')?.size || 0).toBe(0)
  expect(db.collections.get('clips')?.size || 0).toBe(0)
})

test.each([
  descriptor({ shot_id: 'shot-9999', clip_id: clipId('session-1', 'shot-9999') }),
  descriptor({ start_ms: 3500, duration_ms: 2000 }),
  descriptor({ source_kind: 'development_replay' }),
])('wrong shot, timing, source or clip lineage is rejected', async value => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db), value)))
    .rejects.toHaveProperty('code')
  expect(db.collections.get('saved_clip_sync_jobs')?.size || 0).toBe(0)
})

test('zero-shot USB facts publish but cannot authorize a saved clip', async () => {
  const session = {
    ...PUBLISHED_SESSION,
    shots: [],
  }
  const db = createCloudDatabase({ sessions: { 'session-1': session } })

  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db))))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('saved_clip_sync_jobs')?.size || 0).toBe(0)
  expect(db.collections.get('clips')?.size || 0).toBe(0)
})

test.each([
  ['postgame manifest', {
    ...PUBLISHED_SESSION, postgame_manifest_sha256: '0'.repeat(64),
  }],
  ['live job', {
    ...PUBLISHED_SESSION,
    shots: [{ ...PUBLISHED_SESSION.shots[0], live_job_id: 'job-other' }],
  }],
  ['explicit evidence source', {
    ...PUBLISHED_SESSION, evidence_source_kind: 'development_replay',
  }],
])('saved publication blocks mismatched cloud %s lineage', async (_label, published) => {
  const db = createCloudDatabase({ sessions: { 'session-1': published } })
  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db))))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('saved_clip_sync_jobs')?.size || 0).toBe(0)
})

test('missing or mismatched uploaded bytes never create a public clip', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  const storage = { verifyObject: jest.fn().mockResolvedValue({ verified: false }) }
  await expect(completeSavedClipJob({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
    now: '2026-08-26T14:07:00.000Z',
  })).rejects.toMatchObject({ code: 'SYNC_PARTIAL' })
  expect(db.collections.get('clips')?.size || 0).toBe(0)
  expect(db.collections.get('saved_clip_sync_jobs').get(job.job_id).status).not.toBe('completed')
})

test('permanent uploaded media mismatch is blocked instead of retried', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  const storage = {
    verifyObject: jest.fn().mockResolvedValue({ verified: false, permanent: true }),
  }
  await expect(completeSavedClipJob({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
    now: '2026-08-26T14:07:00.000Z',
  })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('clips')?.size || 0).toBe(0)
})

test('a pre-existing partial or conflicting clip document cannot be adopted as completed', async () => {
  const value = descriptor()
  const db = createCloudDatabase({
    sessions: { 'session-1': PUBLISHED_SESSION },
    clips: { [value.clip_id]: {
      ...value,
      media_type: 'video/mp4', table_id: 'table-1', _device_id: 'device-1',
      upload_status: 'uploading', storage_file_id: null,
    } },
  })
  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db))))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('saved_clip_sync_jobs')?.size || 0).toBe(0)
})

test('foreign device and guessed job IDs cannot authorize or complete a clip', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  const storage = { authorizeUpload: jest.fn(), verifyObject: jest.fn() }
  await expect(authorizeSavedClip({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'foreign-device', repo, storage,
  })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(authorizeSavedClip({
    jobId: 'saved-job-guessed', tableId: 'table-1', deviceId: 'device-1', repo, storage,
  })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(completeSavedClipJob({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'foreign-device', repo, storage,
  })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(storage.authorizeUpload).not.toHaveBeenCalled()
  expect(storage.verifyObject).not.toHaveBeenCalled()
})

test.each([
  { method: 'POST', url: 'https://upload.invalid', headers: {} },
  { method: 'PUT', url: 'http://upload.invalid', headers: {} },
  { method: 'PUT', url: 'https://upload.invalid/#secret', headers: {} },
  { method: 'PUT', url: 'https://upload.invalid', headers: { authorization: 'ok\r\nleak' } },
])('unsafe upload authorization is never returned to the edge', async authorization => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  const storage = { authorizeUpload: jest.fn().mockResolvedValue(authorization) }
  await expect(authorizeSavedClip({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
  })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(db.collections.get('saved_clip_sync_jobs').get(job.job_id).status).toBe('pending')
})

test('duplicate completion preserves later privacy and deletion state', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  const storage = { verifyObject: jest.fn().mockResolvedValue({
    verified: true, file_id: 'cloud://development/private/saved.mp4',
  }) }
  const request = {
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage,
    now: '2026-08-26T14:07:00.000Z',
  }
  await completeSavedClipJob(request)
  const clip = { ...db.collections.get('clips').get(descriptor().clip_id),
    consent_status: 'revoked', deleted_at: '2026-08-26T15:00:00.000Z',
    retention_state: 'released',
  }
  db.collections.get('clips').set(descriptor().clip_id, structuredClone(clip))
  await completeSavedClipJob({ ...request, now: 'later' })
  await createSavedClipJob(input(repo))
  db.collections.get('saved_clip_sync_idempotency').clear()
  await createSavedClipJob({ ...input(repo), idFactory: () => 'replacement-job' })
  expect(db.collections.get('clips').get(descriptor().clip_id)).toEqual(clip)
  expect(storage.verifyObject).toHaveBeenCalledTimes(1)
})

test('publication rollback cannot expose a partial saved clip', async () => {
  let fail = true
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } }, {
    failWrite(name) { if (fail && name === 'saved_clip_sync_jobs') throw new Error('injected') },
  })
  const repo = createCloudSavedClipRepository(db)
  await expect(createSavedClipJob(input(repo))).rejects.toThrow('injected')
  expect(db.collections.get('saved_clip_sync_jobs')?.size || 0).toBe(0)
  expect(db.collections.get('saved_clip_sync_idempotency')?.size || 0).toBe(0)
  fail = false
  await expect(createSavedClipJob(input(repo))).resolves.toHaveProperty('job_id')
})

function signedEvent(value, path = '/v1/sync/saved-clips') {
  const body = JSON.stringify(value)
  const timestamp = String(NOW_SECONDS)
  const nonce = `nonce-${crypto.randomUUID()}`
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const signature = crypto.createHmac('sha256', SECRET).update(canonicalSignatureInput({
    method: 'POST', path, timestamp, nonce, bodyHash,
  })).digest('hex')
  return {
    httpMethod: 'POST', path, body,
    headers: {
      'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce,
      'x-body-sha256': bodyHash, 'x-signature': signature,
      'x-idempotency-key': value.clip_id || descriptor().clip_id,
    },
  }
}

test('signed HTTP routes return acknowledgements bound to the saved clip identity', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const savedClipRepo = createCloudSavedClipRepository(db)
  const storage = {
    authorizeUpload: jest.fn().mockResolvedValue({ method: 'PUT', url: 'https://upload.invalid', headers: {} }),
    verifyObject: jest.fn().mockResolvedValue({ verified: true, file_id: 'cloud://development/private/saved.mp4' }),
  }
  const handler = createHandler({
    auth: {
      nowSeconds: () => NOW_SECONDS,
      deviceRepo: { getActiveDevice: async () => ({
        device_id: 'device-1', table_id: 'table-1', status: 'active', secret: SECRET,
      }) },
      nonceRepo: { consume: async () => true },
    },
    savedClipRepo, storage,
    idFactory: () => 'saved-job-1',
    nowIso: () => '2026-08-26T14:07:00.000Z',
  })
  const created = await handler(signedEvent(descriptor()))
  expect(created.statusCode).toBe(200)
  expect(JSON.parse(created.body).data).toEqual(expect.objectContaining({
    job_id: 'saved-job-1', status: 'pending', clip_id: descriptor().clip_id,
    session_id: 'session-1', shot_id: 'shot-0001', content_sha256: 'a'.repeat(64),
    publication_kind: 'saved_postgame_clip',
  }))
  const authorized = await handler(signedEvent(
    {}, '/v1/sync/saved-clips/saved-job-1/media-authorization',
  ))
  expect(authorized.statusCode).toBe(200)
  const completed = await handler(signedEvent(
    {}, '/v1/sync/saved-clips/saved-job-1/complete',
  ))
  expect(completed.statusCode).toBe(200)
  expect(JSON.parse(completed.body).data).toEqual(expect.objectContaining({
    job_id: 'saved-job-1', status: 'completed', clip_id: descriptor().clip_id,
    publication_kind: 'saved_postgame_clip',
  }))
})


test('exact Python V2 canonical receipt and descriptor fixture are accepted', () => {
  expect(() => validateSavedClipInput({ ...input(null, fixture.descriptor) })).not.toThrow()
  const { digest, canonicalJson } = require('../../cloudfunctions/shared/shot-ownership')
  expect(canonicalJson(fixture.descriptor)).toBe(fixture.descriptor_json)
  expect(digest(fixture.descriptor)).toBe(fixture.descriptor_sha256)
  expect(canonicalJson(fixture.descriptor.ownership_receipt)).toBe(fixture.ownership_receipt_json)
})

test.each([
  ['legacy V1', value => { value.schema_version = '1.0' }],
  ['unknown ownership field', value => { value.ownership_receipt.extra = true }],
  ['unsafe revision', value => { value.ownership_receipt.match_revision = Number.MAX_SAFE_INTEGER + 1 }],
  ['changed save receipt', value => { value.save_receipt_sha256 = '0'.repeat(64) }],
  ['changed receipt digest', value => { value.ownership_receipt_sha256 = '0'.repeat(64) }],
])('V2 rejects %s before publication', (_name, mutate) => {
  const value = descriptor(); mutate(value)
  expect(() => validateSavedClipInput(input(null, value)))
    .toThrow(expect.objectContaining({ code: 'SYNC_INVALID' }))
})

test.each(['create retry', 'authorize', 'complete', 'completed retry'])('%s revalidates immutable history and shot projection', async phase => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  const storage = { authorizeUpload: jest.fn(), verifyObject: jest.fn(async () => ({
    verified: true, file_id: 'cloud://private/saved.mp4',
  })) }
  if (phase === 'completed retry') await completeSavedClipJob({
    jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage, now: 'now',
  })
  const projection = [...db.collections.get('field_shot_ownership').values()][0]
  expect(projection.ownership_receipt).toEqual(descriptor().ownership_receipt)
  projection.owner_user_id = 'user-c'
  const action = phase === 'create retry' ? () => createSavedClipJob(input(repo))
    : phase === 'authorize' ? () => authorizeSavedClip({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage })
      : () => completeSavedClipJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage, now: 'now' })
  await expect(action()).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(storage.authorizeUpload).not.toHaveBeenCalled()
  expect(storage.verifyObject).toHaveBeenCalledTimes(phase === 'completed retry' ? 1 : 0)
})

test.each([
  ['missing history', db => db.collections.get('session_role_binding_history').clear(), 'SYNC_PARTIAL'],
  ['orphan history', db => db.collections.get('field_session_commands').clear(), 'SYNC_PARTIAL'],
  ['mismatched historical owner', db => { db.collections.get('session_role_binding_history').get(historyId()).user_id = 'user-c' }, 'SYNC_CONFLICT'],
  ['tampered applied command', db => { db.collections.get('field_session_commands').get('claim-red-1').actor_user_id = 'user-c' }, 'SYNC_CONFLICT'],
  ['mismatched shot role', db => { db.collections.get('sessions').get('session-1').shots[0].actor_role = 'blue' }, 'SYNC_CONFLICT'],
])('%s fails closed with the stable retry classification', async (_name, mutate, code) => {
  const db = createCloudDatabase({ sessions: { 'session-1': structuredClone(PUBLISHED_SESSION) } })
  mutate(db)
  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db)))).rejects.toMatchObject({ code })
})

test('latest historical role before shot start wins despite a later final owner', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': {
    ...PUBLISHED_SESSION, participants: [{ user_id: 'user-c', role: 'red' }],
  } } })
  db.collections.get('session_role_binding_history').set(historyId(2), {
    ...ownershipSeed().session_role_binding_history[historyId()],
    resulting_revision: 2, user_id: 'user-c', command_id: 'claim-red-2',
  })
  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db)))).resolves.toMatchObject({ status: 'pending' })
})

test('a later release at start revision supersedes an earlier claim and never falls back', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const value = descriptor()
  value.ownership_receipt.match_revision = 2
  const { digest } = require('../../cloudfunctions/shared/shot-ownership')
  value.ownership_receipt_sha256 = digest(value.ownership_receipt)
  value.save_receipt_sha256 = digest({ ...fixture.save_receipt, ownership_receipt_sha256: value.ownership_receipt_sha256 })
  db.collections.get('session_role_binding_history').set(historyId(2), {
    ...ownershipSeed().session_role_binding_history[historyId()], resulting_revision: 2,
    user_id: null, command_id: 'release-red-2',
  })
  db.collections.get('field_session_commands').set('release-red-2', {
    ...ownershipSeed().field_session_commands['claim-red-1'], command_id: 'release-red-2',
    command_type: 'release_role', validated_payload: {}, applied_role: 'red', resulting_revision: 2,
  })
  await expect(createSavedClipJob(input(createCloudSavedClipRepository(db), value)))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test('existing clip cannot rebuild a missing immutable projection on replacement job creation', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  await completeSavedClipJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo,
    storage: { verifyObject: async () => ({ verified: true, file_id: 'cloud://private/saved.mp4' }) }, now: 'now' })
  db.collections.get('saved_clip_sync_idempotency').clear()
  db.collections.get('field_shot_ownership').clear()
  await expect(createSavedClipJob(input(repo))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test.each(['create retry', 'authorize', 'complete', 'completed retry'])(
  '%s rejects a changed original applied command before any media operation', async phase => {
    const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
    const repo = createCloudSavedClipRepository(db)
    const job = await createSavedClipJob(input(repo))
    const storage = { authorizeUpload: jest.fn(), verifyObject: jest.fn(async () => ({
      verified: true, file_id: 'cloud://private/saved.mp4',
    })) }
    const request = { jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage, now: 'now' }
    if (phase === 'completed retry') await completeSavedClipJob(request)
    db.collections.get('field_session_commands').get('claim-red-1').validated_payload.role = 'blue'
    const action = phase === 'create retry' ? () => createSavedClipJob(input(repo))
      : phase === 'authorize' ? () => authorizeSavedClip(request) : () => completeSavedClipJob(request)
    await expect(action()).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(storage.authorizeUpload).not.toHaveBeenCalled()
    expect(storage.verifyObject).toHaveBeenCalledTimes(phase === 'completed retry' ? 1 : 0)
  },
)

test('completed retry rejects a changed clip owner instead of acknowledging the old job', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo))
  const request = { jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo,
    storage: { verifyObject: jest.fn(async () => ({ verified: true, file_id: 'cloud://private/saved.mp4' })) }, now: 'now' }
  await completeSavedClipJob(request)
  db.collections.get('clips').get(descriptor().clip_id).owner_user_id = 'user-b'
  await expect(completeSavedClipJob(request)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(request.storage.verifyObject).toHaveBeenCalledTimes(1)
})

test('missing historical authority retries successfully only after immutable history arrives', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION }, session_role_binding_history: {} })
  const repo = createCloudSavedClipRepository(db)
  await expect(createSavedClipJob(input(repo))).rejects.toMatchObject({ code: 'SYNC_PARTIAL' })
  expect(db.collections.get('field_shot_ownership')?.size || 0).toBe(0)
  db.collections.get('session_role_binding_history').set(historyId(), ownershipSeed().session_role_binding_history[historyId()])
  await expect(createSavedClipJob(input(repo))).resolves.toMatchObject({ status: 'pending' })
})

test('a second authenticated receipt for the same shot cannot replace its first immutable projection', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  await createSavedClipJob(input(repo))
  const original = structuredClone([...db.collections.get('field_shot_ownership').values()][0])
  const value = descriptor({ artifact_id: 'f'.repeat(64), clip_id: clipId('session-1', 'shot-0001', 'job-1', 'f'.repeat(64)), owner_user_id: 'user-c' })
  value.ownership_receipt.actor_user_id = 'user-c'
  value.ownership_receipt.match_revision = 2
  const { digest } = require('../../cloudfunctions/shared/shot-ownership')
  value.ownership_receipt_sha256 = digest(value.ownership_receipt)
  value.save_receipt_sha256 = digest({ ...fixture.save_receipt, clip_id: value.clip_id,
    owner_user_id: 'user-c', ownership_receipt_sha256: value.ownership_receipt_sha256 })
  db.collections.get('session_role_binding_history').set(historyId(2), {
    ...ownershipSeed().session_role_binding_history[historyId()], resulting_revision: 2,
    user_id: 'user-c', command_id: 'claim-red-2',
  })
  db.collections.get('field_session_commands').set('claim-red-2', {
    ...ownershipSeed().field_session_commands['claim-red-1'], command_id: 'claim-red-2',
    actor_user_id: 'user-c', resulting_revision: 2,
  })
  await expect(createSavedClipJob({ ...input(repo, value), idFactory: () => 'replacement' }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect([...db.collections.get('field_shot_ownership').values()]).toEqual([original])
})

test('original SAVE_PENDING V2 selection survives late generation through create authorize and complete', async () => {
  const { digest, validSavedSelection } = require('../../cloudfunctions/shared/shot-ownership')
  const pendingReceipt = { ...fixture.save_receipt, status: 'SAVE_PENDING', clip_id: null }
  const value = descriptor({ save_receipt_sha256: digest(pendingReceipt) })
  expect(validSavedSelection(value)).toBe(true)
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db)
  const job = await createSavedClipJob(input(repo, value))
  const storage = {
    authorizeUpload: async () => ({ method: 'PUT', url: 'https://upload.invalid', headers: {} }),
    verifyObject: async () => ({ verified: true, file_id: 'cloud://private/pending-selection.mp4' }),
  }
  await authorizeSavedClip({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage })
  await completeSavedClipJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage, now: 'now' })
  expect(db.collections.get('clips').get(value.clip_id).save_receipt_sha256).toBe(digest(pendingReceipt))
  await expect(createSavedClipJob(input(repo, value))).resolves.toMatchObject({ status: 'completed' })
  await expect(createSavedClipJob(input(repo, descriptor()))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test.each([
  { status: 'SAVE_PENDING' }, { status: 'SAVED', clip_id: null }, { status: 'UNKNOWN', clip_id: null },
  { status: 'SAVE_PENDING', clip_id: null, owner_user_id: 'user-b' },
  { status: 'SAVE_PENDING', clip_id: null, actor_role: 'blue' },
  { status: 'SAVE_PENDING', clip_id: null, ownership_receipt_sha256: '0'.repeat(64) },
])('inconsistent or tampered original selection %p is rejected', change => {
  const { digest, validSavedSelection } = require('../../cloudfunctions/shared/shot-ownership')
  const value = descriptor({ save_receipt_sha256: digest({ ...fixture.save_receipt, ...change }) })
  expect(validSavedSelection(value)).toBe(false)
  expect(() => validateSavedClipInput(input(null, value))).toThrow(expect.objectContaining({ code: 'SYNC_INVALID' }))
})

test.each(['pending', 'completed', 'compacted', 'malformed'])(
  'permanent tombstone fences saved %s create/authorize/complete and direct transaction retries', async state => {
    const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
    const repo = createCloudSavedClipRepository(db)
    const job = await createSavedClipJob(input(repo))
    if (state !== 'pending') await completeSavedClipJob({
      jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo,
      storage: { verifyObject: async () => ({ verified: true, file_id: 'cloud://private/saved.mp4' }) }, now: input(repo).now,
    })
    const raw = structuredClone(db.collections.get('saved_clip_sync_jobs').get(job.job_id))
    await db.collection('owner_clip_tombstones').doc(job.clip_id).set({ data: state === 'malformed' ? {} : { clip_id: job.clip_id } })
    if (state === 'compacted') {
      db.collections.get('saved_clip_sync_jobs').clear(); db.collections.get('saved_clip_sync_idempotency').clear(); db.collections.get('clips').clear()
    }
    const storage = { authorizeUpload: jest.fn(), verifyObject: jest.fn() }
    await expect(createSavedClipJob(input(repo))).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    await expect(repo.create(raw)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    if (state !== 'compacted') {
      await expect(authorizeSavedClip({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage }))
        .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
      await expect(completeSavedClipJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage, now: input(repo).now }))
        .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
      await expect(repo.markUploading(job.job_id)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
      await expect(repo.markCompleted(job.job_id, input(repo).now, { file_id: 'cloud://new/file' }))
        .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    }
    expect(storage.authorizeUpload).not.toHaveBeenCalled(); expect(storage.verifyObject).not.toHaveBeenCalled()
    expect(db.collections.get('clips').get(job.clip_id)?.storage_file_id).not.toBe('cloud://new/file')
  },
)
test('tombstone committed during saved storage verification prevents publication', async () => {
  const db = createCloudDatabase({ sessions: { 'session-1': PUBLISHED_SESSION } })
  const repo = createCloudSavedClipRepository(db); const job = await createSavedClipJob(input(repo))
  await expect(completeSavedClipJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, now: input(repo).now,
    storage: { async verifyObject() {
      await db.collection('owner_clip_tombstones').doc(job.clip_id).set({ data: {} })
      return { verified: true, file_id: 'cloud://late/copy' }
    } },
  })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('clips')?.size || 0).toBe(0)
})
