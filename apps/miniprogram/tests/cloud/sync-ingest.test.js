const {
  authorizeMedia,
  completeSyncJob,
  createCloudSyncRepository,
  createSyncJob,
} = require('../../cloudfunctions/sync-api/lib/ingest')
const { createCloudShotProjectionRepository } = require('../../cloudfunctions/shared/shot-projection')


function createRepository() {
  const jobs = new Map()
  return {
    async validatePublication() {}, // Explicit synthetic repository, not cloud authority.
    async getByIdempotencyKey(key) {
      return jobs.get(key) || null
    },
    async create(job) {
      jobs.set(job.idempotency_key, job)
      return job
    },
    async get(jobId) {
      return [...jobs.values()].find((job) => job.job_id === jobId) || null
    },
    async markCompleted(jobId, completedAt, storageFiles) {
      const job = [...jobs.values()].find((item) => item.job_id === jobId)
      job.status = 'completed'
      job.completed_at = completedAt
      job.storage_files = storageFiles
      return job
    },
  }
}


const ENVELOPE = {
  schema_version: '1.0',
  session: {
    schema_version: '1.0',
    session_id: 'session-1',
    artifact_id: 'artifact-1',
    status: 'ended',
    created_at_utc: '2026-08-26T14:00:00.000Z',
    ended_at_utc: '2026-08-26T14:05:00.000Z',
    postgame_manifest_sha256: 'b'.repeat(64),
    evidence_source_kind: 'development_replay',
    participants: [{ user_id: 'user-a', role: 'red', group: 'stripe' }],
    breaker: 'red',
    active_role_at_end: 'red',
    shots: [{ shot_id: 'shot-1', sequence: 1, live_job_id: 'job-1' }],
    limitations: ['development_replay_media'],
  },
  media: [{
    clip_id: 'clip-1',
    session_id: 'session-1',
    shot_id: 'shot-1',
    artifact_id: 'artifact-1',
    relative_path: 'clips/shot-1.mp4',
    source_kind: 'development_replay',
    content_sha256: 'a'.repeat(64),
    byte_size: 1024,
    media_type: 'video/mp4',
  }],
}


function createCloudDatabase(seed, {
  throwOnMissing = false,
  includeDocumentId = false,
  rejectDocumentIdWrites = false,
  failWrite,
} = {}) {
  const collections = new Map()
  // Fixture server-approved role records; production derives neither from the
  // submitted envelope nor from another session's current table roster.
  if (!Object.hasOwn(seed, 'session_role_bindings')) {
    seed = { ...seed, session_role_bindings: {} }
    for (const live of Object.values(seed.field_live_sessions || {})) {
      for (const role of ['red', 'blue']) {
        if (!live[`${role}_user_id`]) continue
        const id = require('crypto').createHash('sha256').update(`${live.session_id}:${role}`).digest('hex')
        seed.session_role_bindings[id] = { session_id: live.session_id, role,
          user_id: live[`${role}_user_id`], confirmed_at: '2026-08-31T00:00:00Z' }
      }
    }
  }
  let transactions = Promise.resolve()
  for (const [name, records] of Object.entries(seed)) {
    collections.set(name, new Map(Object.entries(records)))
  }
  function collection(name) {
    if (!collections.has(name)) collections.set(name, new Map())
    const records = collections.get(name)
    return {
      doc(id) {
        return {
          async get() {
            if (throwOnMissing && !records.has(id)) {
              throw new Error(`document with _id ${id} does not exist`)
            }
            const data = records.get(id) || null
            return {
              data: data && includeDocumentId ? { ...data, _id: id } : data,
            }
          },
          async set({ data }) {
            if (failWrite) failWrite(name, id)
            if (rejectDocumentIdWrites && Object.hasOwn(data, '_id')) {
              throw new Error('document.set:fail -501007 invalid parameters. 不能更新_id的值')
            }
            records.set(id, data)
          },
          async update({ data }) { records.set(id, { ...records.get(id), ...data }) },
        }
      },
    }
  }
  return {
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


test('first cloud sync creates a job when the idempotency document does not exist', async () => {
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } }, { throwOnMissing: true })
  const repo = createCloudSyncRepository(db)

  await expect(createSyncJob({
    envelope: { ...ENVELOPE, media: [] },
    idempotencyKey: 'sync-session-1',
    manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1',
    repo,
    idFactory: () => 'job-1',
    now: '2026-08-26T14:00:00.000Z',
  })).resolves.toMatchObject({
    job_id: 'job-1',
    status: 'ready_to_complete',
  })
})


test('same idempotency key with different manifest hash conflicts', async () => {
  const repo = createRepository()
  await createSyncJob({
    envelope: ENVELOPE,
    idempotencyKey: 'sync-session-1',
    manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1',
    repo,
    idFactory: () => 'job-1',
    now: '2026-08-26T14:00:00.000Z',
  })
  await expect(createSyncJob({
    envelope: ENVELOPE,
    idempotencyKey: 'sync-session-1',
    manifestHash: 'c'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1',
    repo,
    idFactory: () => 'job-2',
    now: '2026-08-26T14:01:00.000Z',
  })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})


test('invalid session end time is rejected before media upload can be authorized', async () => {
  const repo = createRepository()
  await expect(createSyncJob({
    envelope: { ...ENVELOPE, session: { ...ENVELOPE.session, ended_at_utc: 'not-a-date' } },
    idempotencyKey: 'sync-invalid-end', manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1', repo,
    idFactory: () => 'job-invalid', now: '2026-08-26T14:00:00.000Z',
  })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
})

test.each([
  ['missing explicit evidence source', session => { delete session.evidence_source_kind }],
  ['evidence source inconsistent with media', session => { session.evidence_source_kind = 'usb_live' }],
  ['missing frozen postgame manifest', session => {
    session.evidence_source_kind = 'usb_live'
    session.limitations = session.limitations.filter(
      limitation => limitation !== 'development_replay_media',
    )
    delete session.postgame_manifest_sha256
  }],
  ['missing per-shot live job identity', session => {
    session.evidence_source_kind = 'usb_live'
    session.limitations = session.limitations.filter(
      limitation => limitation !== 'development_replay_media',
    )
    delete session.shots[0].live_job_id
  }],
])('facts-only sync rejects %s', async (_label, mutate) => {
  const envelope = structuredClone(ENVELOPE)
  envelope.media = []
  mutate(envelope.session)
  await expect(createSyncJob({
    envelope,
    idempotencyKey: `sync-invalid-${_label.replaceAll(' ', '-')}`,
    manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1', repo: createRepository(),
    idFactory: () => 'job-invalid-lineage', now: '2026-08-26T14:00:00.000Z',
  })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
})

test('legacy nonempty replay media accepts absent saved-clip lineage', async () => {
  const envelope = structuredClone(ENVELOPE)
  delete envelope.session.postgame_manifest_sha256
  delete envelope.session.shots[0].live_job_id

  await expect(createSyncJob({
    envelope,
    idempotencyKey: 'sync-legacy-replay',
    manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1', repo: createRepository(),
    idFactory: () => 'job-legacy-replay', now: '2026-08-26T14:00:00.000Z',
  })).resolves.toMatchObject({
    job_id: 'job-legacy-replay', status: 'awaiting_media',
  })
})

test('zero-shot USB facts-only sync accepts an explicit source and frozen empty manifest', async () => {
  const envelope = structuredClone(ENVELOPE)
  envelope.media = []
  envelope.session.shots = []
  envelope.session.evidence_source_kind = 'usb_live'
  envelope.session.limitations = envelope.session.limitations.filter(
    limitation => limitation !== 'development_replay_media',
  )

  await expect(createSyncJob({
    envelope,
    idempotencyKey: 'sync-empty-usb',
    manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1', repo: createRepository(),
    idFactory: () => 'job-empty-usb', now: '2026-08-26T14:00:00.000Z',
  })).resolves.toMatchObject({
    job_id: 'job-empty-usb', status: 'ready_to_complete',
  })
})

test('a device cannot self-label uploaded media as a synthetic deletion fixture', async () => {
  const repo = createRepository()
  const envelope = structuredClone(ENVELOPE)
  envelope.media[0].source_kind = 'synthetic_retention_test'
  await expect(createSyncJob({
    envelope, idempotencyKey: 'sync-fake-synthetic', manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1', repo,
    idFactory: () => 'job-fake', now: '2026-08-26T14:00:00.000Z',
  })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
})


test.each(['demo', 'file_replay', 'recorded_replay'])(
  'new sync writes reject non-canonical source kind %s', async sourceKind => {
    const repo = createRepository()
    const envelope = structuredClone(ENVELOPE)
    envelope.media[0].source_kind = sourceKind
    await expect(createSyncJob({
      envelope, idempotencyKey: `sync-${sourceKind}`, manifestHash: 'b'.repeat(64),
      tableId: 'table-1', deviceId: 'device-1', repo,
      idFactory: () => `job-${sourceKind}`, now: '2026-08-26T14:00:00.000Z',
    })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  },
)


test('identical retry returns the original sync job', async () => {
  const repo = createRepository()
  const input = {
    envelope: ENVELOPE,
    idempotencyKey: 'sync-session-1',
    manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1',
    repo,
    idFactory: () => 'job-1',
    now: '2026-08-26T14:00:00.000Z',
  }
  const first = await createSyncJob(input)
  const second = await createSyncJob(input)
  expect(second).toEqual(first)
})


test('media authorization is scoped to declared bytes and completion verifies storage', async () => {
  const repo = createRepository()
  await createSyncJob({
    envelope: ENVELOPE,
    idempotencyKey: 'sync-session-1',
    manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1',
    repo,
    idFactory: () => 'job-1',
    now: '2026-08-26T14:00:00.000Z',
  })
  const storage = {
    authorizeUpload: jest.fn().mockResolvedValue({ method: 'PUT', url: 'https://upload.invalid', expires_in_seconds: 300 }),
    verifyObject: jest.fn().mockResolvedValue({
      verified: true,
      file_id: 'cloud://development/private/clip-1.mp4',
    }),
  }
  await expect(authorizeMedia({ jobId: 'job-1', clipId: 'clip-1', tableId: 'table-1', deviceId: 'device-1', repo, storage }))
    .resolves.toMatchObject({ method: 'PUT', expires_in_seconds: 300 })
  expect(storage.authorizeUpload).toHaveBeenCalledWith(expect.objectContaining({
    byte_size: 1024,
    content_sha256: 'a'.repeat(64),
  }))
  await expect(completeSyncJob({
    jobId: 'job-1', tableId: 'table-1', deviceId: 'device-1', repo, storage, now: '2026-08-26T14:05:00.000Z',
  })).resolves.toMatchObject({
    status: 'completed',
    storage_files: { 'clip-1': {
      file_id: 'cloud://development/private/clip-1.mp4', staging_file_id: null,
    } },
  })
})


test('cloud completion atomically publishes complete session participants clips and pending shot projection', async () => {
  const job = {
    table_id: 'table-1',
    _device_id: 'device-1',
    job_id: 'job-1',
    session_id: 'session-1',
    status: 'awaiting_media',
    envelope: ENVELOPE,
  }
  const db = createCloudDatabase({
    sync_jobs: { 'job-1': job },
    field_live_sessions: { 'session-1': LIVE },
  })
  const repo = createCloudSyncRepository(db)

  await expect(repo.markCompleted(
    'job-1',
    '2026-08-26T14:05:00.000Z',
    { 'clip-1': {
      file_id: 'cloud://development/private/clip-1.mp4',
      staging_file_id: 'cloud://development/staging/clip-1.mp4',
    } },
  )).resolves.toMatchObject({ status: 'completed' })

  expect(db.collections.get('sessions').get('session-1'))
    .toMatchObject({
      sync_status: 'published', status: 'ended',
      evidence_source_kind: 'development_replay',
    })
  expect([...db.collections.get('session_participants').values()])
    .toEqual([expect.objectContaining({ session_id: 'session-1', user_id: 'user-a' })])
  expect(db.collections.get('shots')?.size || 0).toBe(0)
  const cursor = [...db.collections.get('shot_projection_jobs').values()][0]
  expect(cursor).toMatchObject({ status: 'pending', next_sequence: 1, expected_shot_count: 1 })
  expect(db.collections.get('sessions').get('session-1').shots).toEqual(ENVELOPE.session.shots)
  await createCloudShotProjectionRepository(db).projectBatch({ jobId: cursor.job_id })
  expect([...db.collections.get('shots').values()])
    .toEqual([expect.objectContaining({ session_id: 'session-1', shot_id: 'shot-1' })])
  expect(db.collections.get('clips').get('clip-1')).toMatchObject({
    session_id: 'session-1',
    shot_id: 'shot-1',
    sequence: 1,
    upload_status: 'published',
    storage_file_id: 'cloud://development/private/clip-1.mp4',
    storage_objects: [
      { kind: 'published', file_id: 'cloud://development/private/clip-1.mp4' },
      { kind: 'staging', file_id: 'cloud://development/staging/clip-1.mp4' },
    ],
    ordinary_expires_at: '2026-08-28T14:05:00.000Z',
    retention_expires_at: '2026-08-28T14:05:00.000Z',
    retention_state: 'active',
  })
  expect(db.collections.get('field_live_sessions').get('session-1'))
    .toMatchObject({ status: 'ended' })
})


test('unsaved USB completion refuses while historical completed reads and replay limitations remain valid', async () => {
  const usb = structuredClone(ENVELOPE)
  usb.session.limitations = usb.session.limitations.filter(
    limitation => limitation !== 'development_replay_media',
  )
  usb.media[0].source_kind = 'usb_live'
  usb.session.evidence_source_kind = 'usb_live'
  const job = {
    table_id: 'table-1', _device_id: 'device-1', job_id: 'job-usb', session_id: 'session-1',
    status: 'awaiting_media', envelope: usb,
  }
  const db = createCloudDatabase({
    sync_jobs: { 'job-usb': job }, field_live_sessions: { 'session-1': LIVE },
  })
  const repo = createCloudSyncRepository(db)
  await expect(repo.markCompleted(
    'job-usb', '2026-08-26T14:05:00.000Z', FILES,
  )).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('sessions')?.size || 0).toBe(0)
  // Historical completed state is a read-only compatibility fixture.
  job.status = 'completed'
  db.collections.get('sync_jobs').set('job-usb', job)
  await db.collection('sessions').doc('session-1').set({ data: { ...usb.session,
    _sync_publication: { table_id: 'table-1', device_id: 'device-1',
      content_sha256: require('../../cloudfunctions/shared/session-facts').contentDigest(usb) },
  } })
  const before = structuredClone(db.collections)
  const storage = { authorizeUpload: jest.fn(), verifyObject: jest.fn() }
  await expect(completeSyncJob({ jobId: 'job-usb', tableId: 'table-1', deviceId: 'device-1', repo, storage }))
    .resolves.toMatchObject({ status: 'completed' })
  expect([...db.collections].filter(([, rows]) => rows.size)).toEqual([...before].filter(([, rows]) => rows.size))
  expect(storage.verifyObject).not.toHaveBeenCalled()

  const disguised = structuredClone(ENVELOPE)
  disguised.session.limitations = []
  await expect(createSyncJob({ ...input(disguised, 'disguised'), repo: createRepository() }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
})


test('cloud completion does not write the readonly document id back to the job', async () => {
  const factsOnly = structuredClone(ENVELOPE)
  factsOnly.media = []
  factsOnly.session.limitations = factsOnly.session.limitations.filter(
    limitation => limitation !== 'development_replay_media',
  )
  factsOnly.session.evidence_source_kind = 'usb_live'
  const job = {
    table_id: 'table-1',
    _device_id: 'device-1',
    job_id: 'job-1',
    session_id: 'session-1',
    status: 'ready_to_complete',
    envelope: factsOnly,
  }
  const db = createCloudDatabase({
    sync_jobs: { 'job-1': job },
    field_live_sessions: { 'session-1': LIVE },
  }, { includeDocumentId: true, rejectDocumentIdWrites: true })
  const repo = createCloudSyncRepository(db)

  await expect(repo.markCompleted(
    'job-1',
    '2026-08-26T14:05:00.000Z',
    {},
  )).resolves.toMatchObject({ status: 'completed' })

  expect(db.collections.get('sync_jobs').get('job-1')).not.toHaveProperty('_id')
  expect(db.collections.get('sessions').get('session-1'))
    .toMatchObject({ evidence_source_kind: 'usb_live' })
})

function input(envelope = structuredClone(ENVELOPE), key = 'sync-session-1') {
  return { envelope, idempotencyKey: key, manifestHash: 'b'.repeat(64), tableId: 'table-1', deviceId: 'device-1',
    idFactory: () => `job-${key}`, now: '2026-08-31T00:00:00.000Z' }
}

const LIVE = { session_id: 'session-1', table_id: 'table-1', status: 'ended',
  red_user_id: 'user-a', blue_user_id: null }
const FILES = { 'clip-1': 'cloud://private/clip-1.mp4' }

test('sync racing cloud end acknowledgement retries without publishing or losing its idempotency key', async () => {
  const { errorResponse } = require('../../cloudfunctions/sync-api')
  // The local end command has enqueued its outbox; cloud completion is delayed.
  const db = createCloudDatabase({ field_live_sessions: {
    'session-1': { ...LIVE, status: 'open' },
  } })
  const repo = createCloudSyncRepository(db)
  const payload = input({ ...structuredClone(ENVELOPE), media: [] })
  const error = await createSyncJob({ ...payload, repo }).catch(value => value)
  const response = errorResponse(error)
  expect(response.statusCode).toBe(503)
  expect(JSON.parse(response.body)).toEqual({ ok: false, error: { code: 'SYNC_PARTIAL' } })
  for (const name of ['sync_jobs', 'sessions', 'shots', 'clips']) {
    expect(db.collections.get(name)?.size || 0).toBe(0)
  }
  // The same request becomes eligible only after the server confirms closure.
  db.collections.get('field_live_sessions').set('session-1', { ...LIVE })
  const job = await createSyncJob({ ...payload, repo })
  await repo.markCompleted(job.job_id, '2026-08-31T00:01:00.000Z', {})
  const replay = await createSyncJob({ ...payload, repo })
  expect(replay.job_id).toBe(job.job_id)
  expect(replay.status).toBe('completed')
  expect(db.collections.get('sync_jobs').size).toBe(1)
  expect(db.collections.get('sessions').size).toBe(1)
  expect(db.collections.get('shots')?.size || 0).toBe(0)
  expect(db.collections.get('sessions').get('session-1').shots).toEqual(ENVELOPE.session.shots)
})

test.each([
  { ...LIVE, status: 'open', table_id: 'foreign-table' },
  { ...LIVE, status: 'open', red_user_id: 'another-user' },
  { ...LIVE, status: 'expired' },
])('transient closure handling does not turn authority conflicts into retries', async live => {
  const { errorResponse } = require('../../cloudfunctions/sync-api')
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': live } })
  const error = await createSyncJob({ ...input(), repo: createCloudSyncRepository(db) })
    .catch(value => value)
  expect(errorResponse(error).statusCode).toBe(409)
  expect(error.code).toBe('SYNC_CONFLICT')
  expect(db.collections.get('sync_jobs')?.size || 0).toBe(0)
})

test.each(['session_id', 'shot_id', 'artifact_id'])('rejects media with mismatched %s before storing a job', async key => {
  const payload = input()
  payload.envelope.media[0][key] = 'foreign'
  await expect(createSyncJob({ ...payload, repo: createRepository() }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
})

test('same key and claimed hash cannot conceal different actual facts', async () => {
  const repo = createRepository()
  await createSyncJob({ ...input(), repo })
  const changed = input()
  changed.envelope.session.artifact_id = 'other'
  changed.envelope.media[0].artifact_id = 'other'
  await expect(createSyncJob({ ...changed, repo })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test.each([
  { ...LIVE, table_id: 'foreign-table' },
  { ...LIVE, red_user_id: 'another-user' },
  null,
])('publication verifies persisted table and participant authority', async live => {
  const db = createCloudDatabase({ field_live_sessions: live ? { 'session-1': live } : {} })
  await expect(createSyncJob({ ...input(), repo: createCloudSyncRepository(db) }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('sync_jobs')?.size || 0).toBe(0)
})

test('new key replay preserves reviewed and privacy state without republishing facts', async () => {
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } })
  const repo = createCloudSyncRepository(db)
  const first = await createSyncJob({ ...input(), repo })
  await repo.markCompleted(first.job_id, '2026-08-31T00:00:00.000Z', FILES)
  const session = db.collections.get('sessions').get('session-1')
  session.reviewed_projection = { review_revision: 4 }
  const clip = db.collections.get('clips').get('clip-1')
  clip.consent_status = 'revoked'
  clip.deleted_at = 'deleted'
  const second = await createSyncJob({ ...input(undefined, 'another-key'), repo })
  await repo.markCompleted(second.job_id, '2026-08-31T00:01:00.000Z', FILES)
  expect(db.collections.get('sessions').get('session-1')).toEqual(session)
  expect(db.collections.get('clips').get('clip-1')).toEqual(clip)
  expect(db.collections.get('session_participants').size).toBe(1)
  expect(db.collections.get('shots')?.size || 0).toBe(0)
  expect(db.collections.get('sessions').get('session-1').shots).toEqual(ENVELOPE.session.shots)
})

test('different facts with a new key cannot masquerade as an authorized revision', async () => {
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } })
  const repo = createCloudSyncRepository(db)
  const first = await createSyncJob({ ...input(), repo })
  await repo.markCompleted(first.job_id, '2026-08-31T00:00:00.000Z', FILES)
  const changed = input(undefined, 'revision-key')
  changed.envelope.session.shots[0].sequence = 2
  await expect(createSyncJob({ ...changed, repo })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
  changed.envelope.session.shots[0].sequence = 1
  changed.envelope.session.ended_at_utc = '2026-08-27T14:05:00.000Z'
  await expect(createSyncJob({ ...changed, repo })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test('publication rollback and retry cannot leave duplicate partial rows', async () => {
  let fail = true
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } }, {
    failWrite(name) { if (fail && name === 'clips') throw new Error('injected storage failure') },
  })
  const repo = createCloudSyncRepository(db)
  const job = await createSyncJob({ ...input(), repo })
  await expect(repo.markCompleted(job.job_id, '2026-08-31T00:00:00.000Z', FILES)).rejects.toThrow('injected')
  expect(db.collections.get('sessions')?.size || 0).toBe(0)
  fail = false
  await repo.markCompleted(job.job_id, '2026-08-31T00:01:00.000Z', FILES)
  expect(db.collections.get('clips').size).toBe(1)
  expect(db.collections.get('shots')?.size || 0).toBe(0)
  expect(db.collections.get('sessions').get('session-1').shots).toEqual(ENVELOPE.session.shots)
})

test.each([
  'reviewed_projection', 'growth_projection', 'highlight', '_sync_publication',
  '_evidence_source_kind',
])('edge cannot publish %s authority', async key => {
  const payload = input()
  payload.envelope.session[key] = { approved: true }
  await expect(createSyncJob({ ...payload, repo: createRepository() }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
})

test('a second session cannot replace another session clip with the same global clip ID', async () => {
  const db = createCloudDatabase({ field_live_sessions: {
    'session-1': LIVE, 'session-2': { ...LIVE, session_id: 'session-2' },
  } })
  const repo = createCloudSyncRepository(db)
  const first = await createSyncJob({ ...input(), repo })
  await repo.markCompleted(first.job_id, '2026-08-31T00:00:00.000Z', FILES)
  const next = input(undefined, 'session-2')
  next.envelope.session.session_id = 'session-2'
  next.envelope.media[0].session_id = 'session-2'
  await expect(createSyncJob({ ...next, repo })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.collections.get('clips').get('clip-1').session_id).toBe('session-1')
})

test('concurrent duplicate jobs converge through the serializable publication boundary', async () => {
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } })
  const repo = createCloudSyncRepository(db)
  const jobs = await Promise.all(['a', 'b'].map(key => createSyncJob({ ...input(undefined, key), repo })))
  await Promise.all(jobs.map(job => repo.markCompleted(job.job_id, '2026-08-31T00:00:00.000Z', FILES)))
  expect(db.collections.get('sessions').size).toBe(1)
  expect(db.collections.get('shots')?.size || 0).toBe(0)
  expect(db.collections.get('sessions').get('session-1').shots).toEqual(ENVELOPE.session.shots)
  expect(db.collections.get('clips').size).toBe(1)
  expect([...db.collections.get('sync_jobs').values()].map(job => job.status)).toEqual(['completed', 'completed'])
})

test('publication requires the complete authoritative participant set', async () => {
  const db = createCloudDatabase({ field_live_sessions: {
    'session-1': { ...LIVE, blue_user_id: 'user-b' },
  } })
  const repo = createCloudSyncRepository(db)
  await expect(createSyncJob({ ...input(), repo })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  const full = input()
  full.envelope.session.participants.push({ user_id: 'user-b', role: 'blue', group: 'solid' })
  await expect(createSyncJob({ ...full, repo })).resolves.toHaveProperty('job_id')
})

test.each([false, true])('stale pending job cannot authorize media after publication (other session=%s)', async otherSession => {
  const db = createCloudDatabase({ field_live_sessions: {
    'session-1': LIVE, 'session-2': { ...LIVE, session_id: 'session-2' },
  } })
  const repo = createCloudSyncRepository(db)
  const first = await createSyncJob({ ...input(), repo })
  const next = input(undefined, 'competing')
  if (otherSession) {
    next.envelope.session.session_id = 'session-2'
    next.envelope.media[0].session_id = 'session-2'
  } else next.envelope.media[0].content_sha256 = 'c'.repeat(64)
  const second = await createSyncJob({ ...next, repo })
  const storage = { authorizeUpload: jest.fn().mockResolvedValue({}) }
  await authorizeMedia({ jobId: first.job_id, clipId: 'clip-1', tableId: 'table-1', deviceId: 'device-1', repo, storage })
  await authorizeMedia({ jobId: second.job_id, clipId: 'clip-1', tableId: 'table-1', deviceId: 'device-1', repo, storage })
  const paths = storage.authorizeUpload.mock.calls.map(([c]) => c.object_key)
  expect(paths[0]).not.toBe(paths[1])
  await repo.markCompleted(first.job_id, '2026-08-31T00:00:00.000Z', FILES)
  storage.authorizeUpload.mockClear()
  await expect(authorizeMedia({ jobId: second.job_id, clipId: 'clip-1', tableId: 'table-1', deviceId: 'device-1', repo, storage }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(storage.authorizeUpload).not.toHaveBeenCalled()
})

test('completion requests server-only publication distinct from every issued upload path', async () => {
  const repo = createCloudSyncRepository(createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } }))
  const job = await createSyncJob({ ...input(), repo })
  const storage = {
    authorizeUpload: jest.fn().mockResolvedValue({}),
    verifyObject: jest.fn().mockResolvedValue({ verified: true, file_id: 'cloud://published' }),
  }
  await authorizeMedia({ jobId: job.job_id, clipId: 'clip-1', tableId: 'table-1', deviceId: 'device-1', repo, storage })
  await completeSyncJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage, now: '2026-08-31T00:00:00.000Z' })
  const upload = storage.authorizeUpload.mock.calls[0][0].object_key
  const verification = storage.verifyObject.mock.calls[0][0]
  expect(verification.object_key).toBe(upload)
  expect(verification.publish_object_key).toMatch(/^published\//u)
  expect(verification.publish_object_key).not.toBe(upload)
})

test.each(['missing', 'open', 'wrong-session', 'duplicate', 'extra', 'empty', 'role'])('membership authority fails closed for %s', async mode => {
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } })
  const bindings = db.collections.get('session_role_bindings')
  const [id, binding] = [...bindings][0]
  const payload = input()
  if (mode === 'missing') bindings.clear()
  if (mode === 'open') db.collections.get('field_live_sessions').set('session-1', { ...LIVE, status: 'open' })
  if (mode === 'wrong-session') bindings.set(id, { ...binding, session_id: 'session-2' })
  if (mode === 'duplicate') payload.envelope.session.participants.push({ ...payload.envelope.session.participants[0] })
  if (mode === 'extra') payload.envelope.session.participants.push({ user_id: 'other', role: 'blue' })
  if (mode === 'empty') { bindings.clear(); payload.envelope.session.participants = [] }
  if (mode === 'role') bindings.set(id, { ...binding, role: 'blue' })
  await expect(createSyncJob({ ...payload, repo: createCloudSyncRepository(db) })).rejects.toHaveProperty('code')
  expect(db.collections.get('sessions')?.size || 0).toBe(0)
})

test('same-table foreign device cannot replay, authorize or complete another device job', async () => {
  const repo = createCloudSyncRepository(createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } }))
  const job = await createSyncJob({ ...input(), repo })
  const storage = { authorizeUpload: jest.fn(), verifyObject: jest.fn() }
  await expect(createSyncJob({ ...input(), deviceId: 'foreign-device', repo })).rejects.toHaveProperty('code')
  for (const operation of [authorizeMedia, completeSyncJob]) {
    await expect(operation({ jobId: job.job_id, clipId: 'clip-1', tableId: 'table-1',
      deviceId: 'foreign-device', repo, storage })).rejects.toHaveProperty('code')
  }
  expect(storage.authorizeUpload).not.toHaveBeenCalled()
  expect(storage.verifyObject).not.toHaveBeenCalled()
})

test('member change after upload authorization blocks finalize without installing ACL', async () => {
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } })
  const repo = createCloudSyncRepository(db)
  const job = await createSyncJob({ ...input(), repo })
  const bindings = db.collections.get('session_role_bindings')
  const [id, binding] = [...bindings][0]
  const storage = { authorizeUpload: jest.fn().mockResolvedValue({}), verifyObject: jest.fn() }
  await authorizeMedia({ jobId: job.job_id, clipId: 'clip-1', tableId: 'table-1', deviceId: 'device-1', repo, storage })
  bindings.set(id, { ...binding, user_id: 'replacement' })
  await expect(completeSyncJob({ jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(storage.verifyObject).not.toHaveBeenCalled()
  expect(db.collections.get('sessions')?.size || 0).toBe(0)
})

test('verified publication survives DB rollback/retry and delayed staging replacement without overwriting privacy', async () => {
  const crypto = require('crypto')
  const { createStoragePort, createCloudBaseStorageProvider } = require('../../cloudfunctions/sync-api/lib/storage')
  let fail = true
  const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } }, {
    failWrite(name) { if (fail && name === 'clips') throw new Error('DB unavailable') },
  })
  const repo = createCloudSyncRepository(db)
  const bytes = Buffer.concat([
    Buffer.from([0, 0, 0, 24]), Buffer.from('ftypisom'), Buffer.alloc(4),
    Buffer.from('isommp42synthetic-video'),
  ])
  let staged = Buffer.from(bytes)
  const published = new Map()
  const storage = createStoragePort(createCloudBaseStorageProvider({
    getUploadMetadata: async () => ({ data: { fileId: 'cloud://staging' } }),
    downloadFile: async () => ({ fileContent: staged }),
    uploadFile: async ({ cloudPath, fileContent }) => {
      staged = Buffer.from('replacement-after-verification')
      published.set(cloudPath, Buffer.from(fileContent))
      return { fileID: `cloud://${cloudPath}` }
    },
  }))
  const payload = input()
  Object.assign(payload.envelope.media[0], { byte_size: bytes.length,
    content_sha256: crypto.createHash('sha256').update(bytes).digest('hex') })
  const job = await createSyncJob({ ...payload, repo })
  const request = { jobId: job.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage, now: '2026-08-31T00:00:00.000Z' }
  await expect(completeSyncJob(request)).rejects.toThrow('DB unavailable')
  expect(db.collections.get('sessions')?.size || 0).toBe(0)
  expect([...published.values()]).toEqual([bytes]) // orphan, not playable
  fail = false
  await expect(completeSyncJob(request)).rejects.toMatchObject({ code: 'SYNC_PARTIAL' })
  staged = Buffer.from(bytes)
  await completeSyncJob(request)
  const clip = db.collections.get('clips').get('clip-1')
  expect(clip.storage_file_id).toMatch(/^cloud:\/\/published\//u)
  clip.consent_status = 'revoked'
  clip.deleted_at_utc = 'later'
  await Promise.all([completeSyncJob(request), completeSyncJob(request)])
  expect(db.collections.get('clips').get('clip-1')).toMatchObject({ consent_status: 'revoked', deleted_at_utc: 'later' })
  expect([...published.values()]).toEqual([bytes])
})

test.each(['fresh', 'completed', 'compacted'])(
  'generic media %s publication honors tombstone while facts-only sync remains usable', async state => {
    const db = createCloudDatabase({ field_live_sessions: { 'session-1': LIVE } })
    const repo = createCloudSyncRepository(db)
    const args = { envelope: ENVELOPE, idempotencyKey: 'delete-fence', manifestHash: 'b'.repeat(64), tableId: 'table-1',
      deviceId: 'device-1', repo, idFactory: () => 'delete-job', now: '2026-09-01T00:00:00.000Z' }
    let raw
    if (state !== 'fresh') {
      raw = await createSyncJob(args)
      await repo.markCompleted(raw.job_id, args.now, { 'clip-1': 'cloud://old/file' })
      raw = structuredClone(db.collections.get('sync_jobs').get(raw.job_id))
    }
    await db.collection('owner_clip_tombstones').doc('clip-1').set({ data: {} })
    if (state === 'compacted') {
      for (const name of ['clips', 'sync_jobs', 'sync_idempotency', 'sessions']) db.collections.get(name)?.clear()
    }
    await expect(createSyncJob(args)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    if (raw) {
      await expect(repo.create(raw)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
      if (state === 'completed') {
        await expect(completeSyncJob({ jobId: raw.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage: {}, now: args.now }))
          .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
        await expect(repo.markCompleted(raw.job_id, args.now, {})).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
      }
    }
    // A new facts-only session remains usable; an immutable cursor is never
    // removed or repurposed when old sync metadata is compacted.
    const factsEnvelope = structuredClone(ENVELOPE)
    factsEnvelope.session.session_id = 'facts-session'
    factsEnvelope.media = []
    await db.collection('field_live_sessions').doc('facts-session').set({ data: { ...LIVE, session_id: 'facts-session' } })
    const roleId = require('crypto').createHash('sha256').update('facts-session:red').digest('hex')
    await db.collection('session_role_bindings').doc(roleId).set({ data: { session_id: 'facts-session', role: 'red', user_id: 'user-a', confirmed_at: args.now } })
    const facts = await createSyncJob({ ...args, envelope: factsEnvelope,
      idempotencyKey: 'facts-after-delete', idFactory: () => 'facts-job' })
    await expect(completeSyncJob({ jobId: facts.job_id, tableId: 'table-1', deviceId: 'device-1', repo, storage: {}, now: args.now }))
      .resolves.toHaveProperty('status', 'completed')
  },
)
