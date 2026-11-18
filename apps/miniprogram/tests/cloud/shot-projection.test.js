const crypto = require('crypto')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { createCloudSyncRepository, createSyncJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const projection = () => require('../../cloudfunctions/shared/shot-projection')
const NOW = '2026-10-29T10:00:00.000Z'
const LATER = '2026-10-29T10:01:00.000Z'
const hash = value => crypto.createHash('sha256').update(value).digest('hex')
const jobId = `shot-projection-${hash('session-1')}`
const rowId = n => hash(`session-1:shot-${n}`)
function fixture(count = 33) {
  const session = { schema_version: '1.0', session_id: 'session-1', artifact_id: 'artifact-1', status: 'ended',
    created_at_utc: '2026-10-29T09:00:00.000Z', ended_at_utc: NOW,
    evidence_source_kind: 'usb_live', postgame_manifest_sha256: 'b'.repeat(64),
    participants: [{ user_id: 'user-a', role: 'red' }, { user_id: 'user-b', role: 'blue' }],
    breaker: 'red', active_role_at_end: 'blue', limitations: [],
    shots: Array.from({ length: count }, (_, i) => ({ shot_id: `shot-${i + 1}`, sequence: i + 1,
      live_job_id: `live-${i + 1}`, actor_role: i % 2 ? 'blue' : 'red', started_ms: i * 1000,
      settled_ms: i * 1000 + 500, limitations: [] })) }
  const bindings = Object.fromEntries(session.participants.map(p => [hash(`session-1:${p.role}`),
    { ...p, session_id: 'session-1', confirmed_at: NOW }]))
  return { session, seed: { field_live_sessions: { 'session-1': { session_id: 'session-1', table_id: 'table-1',
    status: 'ended', red_user_id: 'user-a', blue_user_id: 'user-b' } }, session_role_bindings: bindings } }
}
async function publish(count = 33, options = {}) {
  const { session, seed } = fixture(count)
  const db = ownerDeleteDb(seed, { strictTransactions: true, ...options })
  const repo = createCloudSyncRepository(db)
  const args = { envelope: { schema_version: '1.0', session, media: [] }, idempotencyKey: 'sync-1',
    manifestHash: 'b'.repeat(64), tableId: 'table-1', deviceId: 'device-1', idFactory: () => 'sync-job-1', now: NOW, repo }
  const job = await createSyncJob(args)
  await repo.markCompleted(job.job_id, NOW, {})
  return { db, args, repo, session }
}

test.each([90, 200])('atomic facts-only %i-shot publication stays within 25 doc calls with all facts', async count => {
  const { db, session } = await publish(count)
  expect(db.rows.sessions['session-1'].shots).toEqual(session.shots)
  expect(Object.keys(db.rows.session_participants)).toHaveLength(2)
  expect(db.rows.sync_jobs['sync-job-1'].status).toBe('completed')
  expect(db.rows.sessions['session-1']._shot_projection_job_id).toBe(jobId)
  expect(db.rows.shot_projection_jobs[jobId]).toMatchObject({ status: 'pending', next_sequence: 1, expected_shot_count: count })
  expect(Object.keys(db.rows.shots || {})).toHaveLength(0)
  expect(db.transactionAttempts.at(-1).count).toBeLessThanOrEqual(25)
})

test('16-shot batches survive restart and completed replay without changing source', async () => {
  const { db } = await publish()
  const source = structuredClone(db.rows.sessions)
  let repo = projection().createCloudShotProjectionRepository(db, { nowIso: () => LATER })
  expect(await repo.findPending()).toMatchObject({ job_id: jobId })
  expect(await repo.projectBatch({ jobId })).toEqual({ outcome: 'progress', job_id: jobId, projected_count: 16, next_sequence: 17 })
  repo = projection().createCloudShotProjectionRepository(db, { nowIso: () => LATER })
  expect(await repo.projectBatch({ jobId })).toMatchObject({ outcome: 'progress', projected_count: 16, next_sequence: 33 })
  expect(await repo.projectBatch({ jobId })).toMatchObject({ outcome: 'completed', projected_count: 1, next_sequence: 34 })
  expect(await repo.projectBatch({ jobId })).toMatchObject({ outcome: 'completed', projected_count: 0, next_sequence: 34 })
  expect(await repo.findPending()).toBeNull()
  expect(Object.keys(db.rows.shots)).toHaveLength(33)
  expect(db.rows.sessions).toEqual(source)
  expect(Math.max(...db.transactionAttempts.map(a => a.count))).toBeLessThanOrEqual(45)
  expect(db.rows.shots[rowId(1)]).toEqual({ ...source['session-1'].shots[0], session_id: 'session-1', _shot_projection: {
    contract: 'shot_projection_v1', session_id: 'session-1', publication_sha256: source['session-1']._sync_publication.content_sha256,
    source_sha256: 'f88ac7990ca7664376bb122ff958366299bb9a73194b63994c233150ad0148f3' } })
})

test('exact original session hash ignores SDK id and reviewed metadata', async () => {
  const { db } = await publish(1)
  expect(db.rows.shot_projection_jobs[jobId].source_session_sha256).toBe('79be728074558a348f6460bd632c61a2691d4b5ff52c747c4690fa78482515ec')
  db.rows.sessions['session-1']._id = 'session-1'
  db.rows.sessions['session-1'].reviewed_projection = { revision: 4 }
  db.rows.shot_projection_jobs[jobId]._id = jobId
  await expect(projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })).resolves.toMatchObject({ outcome: 'completed' })
})

test.each(['shot_projection_jobs', 'sessions', 'session_participants', 'sync_jobs'])('failure writing %s rolls back source publication', async collection => {
  const { session, seed } = fixture(90)
  const db = ownerDeleteDb(seed, { strictTransactions: true })
  const repo = createCloudSyncRepository(db)
  const job = await createSyncJob({ envelope: { schema_version: '1.0', session, media: [] }, idempotencyKey: 'sync-1', manifestHash: 'b'.repeat(64),
    tableId: 'table-1', deviceId: 'device-1', idFactory: () => 'sync-job-1', now: NOW, repo })
  const before = structuredClone(db.rows)
  const failing = ownerDeleteDb(before, { strictTransactions: true, failWrite(name) { if (name === collection) throw new Error('transient') } })
  await expect(createCloudSyncRepository(failing).markCompleted(job.job_id, NOW, {})).rejects.toThrow('transient')
  expect(failing.rows).toEqual(before)
})

test.each([
  ['next_sequence', 0], ['next_sequence', 1.5], ['next_sequence', 35], ['expected_shot_count', -1],
  ['publication_sha256', 'bad'], ['created_at', 'now'], ['status', 'completed'], ['blocked_reason', 'SDK secret'],
  ['extra', true], ['_id', 4], ['session_id', 4],
])('malformed cursor %s=%s fails closed without writes', async (field, value) => {
  const { db } = await publish()
  db.rows.shot_projection_jobs[jobId][field] = value
  const before = structuredClone(db.rows)
  await expect(projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })).rejects.toMatchObject({ code: 'SHOT_PROJECTION_INVALID' })
  expect(db.rows).toEqual(before)
})

test.each(['missing', 'facts', 'fence', 'sequence', 'pointer'])('changed source %s blocks without row writes', async kind => {
  const { db } = await publish()
  const source = db.rows.sessions['session-1']
  if (kind === 'missing') delete db.rows.sessions['session-1']
  if (kind === 'facts') source.shots[32].limitations.push('changed')
  if (kind === 'fence') source._sync_publication.device_id = 'foreign-device'
  if (kind === 'sequence') source.shots[32].sequence = 32
  if (kind === 'pointer') source._shot_projection_job_id = 'foreign'
  const before = structuredClone(db.rows.sessions)
  expect(await projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })).toMatchObject({ outcome: 'blocked', projected_count: 0, next_sequence: 1 })
  expect(db.rows.shot_projection_jobs[jobId].blocked_reason).toBe(kind === 'missing' ? 'source_unavailable' : 'source_conflict')
  expect(Object.keys(db.rows.shots || {})).toHaveLength(0)
  expect(db.rows.sessions).toEqual(before)
})

test.each(['facts', 'pointer', 'extra-fact'])('row preflight detects %s conflict before writes', async kind => {
  const { db } = await publish()
  const row = { ...db.rows.sessions['session-1'].shots[15], session_id: 'session-1', reviewed_projection: { revision: 5 } }
  if (kind === 'facts') row.started_ms++
  if (kind === 'pointer') row._shot_projection = { contract: 'shot_projection_v1', session_id: 'session-1', publication_sha256: 'a'.repeat(64), source_sha256: 'b'.repeat(64) }
  if (kind === 'extra-fact') row.power = { band: 'hard' }
  db.rows.shots = { [rowId(16)]: row }
  const rows = structuredClone(db.rows.shots)
  expect(await projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })).toMatchObject({ outcome: 'blocked', projected_count: 0 })
  expect(db.rows.shot_projection_jobs[jobId].blocked_reason).toBe('row_conflict')
  expect(db.rows.shots).toEqual(rows)
})

test('matching legacy row receives only pointer and preserves reviewed metadata', async () => {
  const { db } = await publish(1)
  const row = { ...db.rows.sessions['session-1'].shots[0], session_id: 'session-1', _id: rowId(1), reviewed_projection: { revision: 3 }, highlight: true }
  db.rows.shots = { [rowId(1)]: structuredClone(row) }
  await projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })
  expect(db.rows.shots[rowId(1)]).toMatchObject(row)
  expect(db.rows.shots[rowId(1)]._shot_projection).toBeDefined()
  expect(db.transactionAttempts.at(-1).calls.filter(c => c.collection === 'shots' && c.method !== 'get').map(c => c.method)).toEqual(['update'])
})

test.each(['shots', 'shot_projection_jobs'])('batch %s write failure rolls back and resumes', async collection => {
  const { db } = await publish()
  let fail = true
  const retryDb = ownerDeleteDb(db.rows, { strictTransactions: true, failWrite(name) { if (fail && name === collection) throw new Error('temporary database secret') } })
  const repo = projection().createCloudShotProjectionRepository(retryDb)
  const before = structuredClone(retryDb.rows)
  await expect(repo.projectBatch({ jobId })).rejects.toThrow('temporary database secret')
  expect(retryDb.rows).toEqual(before)
  fail = false
  expect(await repo.projectBatch({ jobId })).toMatchObject({ projected_count: 16, next_sequence: 17 })
})

test('concurrent batches and callback retry converge without duplicate or skipped rows', async () => {
  const { db } = await publish()
  let conflict = true
  const retryDb = ownerDeleteDb(db.rows, { strictTransactions: true, retryConflicts: 1, beforeCommit() {
    if (conflict) { conflict = false; throw Object.assign(new Error('retry'), { code: 'SYNTHETIC_CONFLICT' }) }
  } })
  const repo = projection().createCloudShotProjectionRepository(retryDb)
  const results = await Promise.all([repo.projectBatch({ jobId }), repo.projectBatch({ jobId }), repo.projectBatch({ jobId })])
  expect(results.map(r => r.next_sequence)).toEqual([17, 33, 34])
  expect(Object.keys(retryDb.rows.shots)).toHaveLength(33)
  expect(retryDb.transactionAttempts).toHaveLength(4)
  expect(Math.max(...retryDb.transactionAttempts.map(a => a.count))).toBeLessThanOrEqual(45)
})

test('empty publication is completed immediately and missing cursor is incomplete', async () => {
  const { db } = await publish(0)
  expect(db.rows.shot_projection_jobs[jobId]).toMatchObject({ status: 'completed', expected_shot_count: 0, next_sequence: 1 })
  const repo = projection().createCloudShotProjectionRepository(db)
  expect(await repo.findPending()).toBeNull()
  expect(await repo.projectBatch({ jobId })).toMatchObject({ outcome: 'completed', projected_count: 0 })
  delete db.rows.shot_projection_jobs[jobId]
  await expect(repo.projectBatch({ jobId })).rejects.toMatchObject({ code: 'SHOT_PROJECTION_INCOMPLETE' })
})

test('different-key replay preserves cursor progress; legacy publications never auto-backfill', async () => {
  const { db, repo, args } = await publish()
  await projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })
  const source = structuredClone(db.rows.sessions)
  const cursor = structuredClone(db.rows.shot_projection_jobs)
  const job = await createSyncJob({ ...args, idempotencyKey: 'sync-2', idFactory: () => 'sync-job-2' })
  await repo.markCompleted(job.job_id, LATER, {})
  expect(db.rows.sessions).toEqual(source)
  expect(db.rows.shot_projection_jobs).toEqual(cursor)
  delete db.rows.sessions['session-1']._shot_projection_job_id
  delete db.rows.shot_projection_jobs[jobId]
  const third = await createSyncJob({ ...args, idempotencyKey: 'sync-3', idFactory: () => 'sync-job-3' })
  await repo.markCompleted(third.job_id, LATER, {})
  expect(db.rows.shot_projection_jobs).toEqual({})
})

test('enqueue reuses an exact immutable identity and rejects overwrite without changing progress', async () => {
  const { db } = await publish()
  await projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })
  const before = structuredClone(db.rows)
  expect(await db.runTransaction(tx => projection().enqueuePublishedShots(tx, db.rows.sessions['session-1']))).toBe(jobId)
  expect(db.rows).toEqual(before)
  const changed = structuredClone(db.rows.sessions['session-1'])
  changed.shots[0].limitations.push('changed')
  await expect(db.runTransaction(tx => projection().enqueuePublishedShots(tx, changed)))
    .rejects.toMatchObject({ code: 'SHOT_PROJECTION_CONFLICT' })
  expect(db.rows).toEqual(before)
})

test('enqueue corruption maps to existing SYNC_CONFLICT and leaves the source unpublished', async () => {
  const { db, args } = await publish()
  const original = structuredClone(db.rows.shot_projection_jobs[jobId])
  const fresh = ownerDeleteDb({ field_live_sessions: db.rows.field_live_sessions,
    session_role_bindings: db.rows.session_role_bindings,
    shot_projection_jobs: { [jobId]: { ...original, next_sequence: 0 } } }, { strictTransactions: true })
  const repo = createCloudSyncRepository(fresh)
  const job = await createSyncJob({ ...args, repo })
  await expect(repo.markCompleted(job.job_id, NOW, {})).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(fresh.rows.sessions || {}).toEqual({})
  expect(fresh.rows.sync_jobs[job.job_id].status).toBe('ready_to_complete')
})

test('source publication callback conflict retries every atomic write including cursor', async () => {
  const { db } = await publish(200, { retryConflicts: 1, beforeCommit({ attempt }) {
    if (attempt.index === 0 && attempt.calls.some(c => c.collection === 'shot_projection_jobs' && c.method === 'set')) {
      throw Object.assign(new Error('retry source commit'), { code: 'SYNTHETIC_CONFLICT' })
    }
  } })
  expect(db.transactionAttempts.slice(-2).map(a => a.count)).toEqual([13, 13])
  expect(Object.keys(db.rows.sessions)).toHaveLength(1)
  expect(Object.keys(db.rows.shot_projection_jobs)).toHaveLength(1)
})

test('pending discovery orders oldest first with job ID tie-break and does no transaction', async () => {
  const { db } = await publish(1)
  db.rows.shot_projection_jobs.other = { ...db.rows.shot_projection_jobs[jobId], job_id: 'shot-projection-0000' }
  db.rows.shot_projection_jobs.older = { ...db.rows.shot_projection_jobs[jobId], job_id: 'shot-projection-ffff', updated_at: '2026-10-28T00:00:00.000Z' }
  const count = db.transactionAttempts.length
  const repo = projection().createCloudShotProjectionRepository(db)
  expect((await repo.findPending()).job_id).toBe('shot-projection-ffff')
  delete db.rows.shot_projection_jobs.older
  expect((await repo.findPending()).job_id).toBe('shot-projection-0000')
  expect(db.transactionAttempts).toHaveLength(count)
})

test.each([null, undefined, 0, 'bad'])('malformed project input %s is a typed invalid error', async input => {
  const { db } = await publish(1)
  await expect(projection().createCloudShotProjectionRepository(db).projectBatch(input))
    .rejects.toMatchObject({ code: 'SHOT_PROJECTION_INVALID' })
})

test('invalid source identity type blocks safely rather than leaking a hash-library error', async () => {
  const { db } = await publish(1)
  db.rows.sessions['session-1'].session_id = 123
  await expect(projection().createCloudShotProjectionRepository(db).projectBatch({ jobId }))
    .resolves.toMatchObject({ outcome: 'blocked', projected_count: 0 })
  expect(db.rows.shot_projection_jobs[jobId].blocked_reason).toBe('source_conflict')
})

test('partial row failure rolls back all earlier writes in the same batch', async () => {
  const { db } = await publish()
  const retryDb = ownerDeleteDb(db.rows, { strictTransactions: true, failWrite(name, id) {
    if (name === 'shots' && id === rowId(8)) throw new Error('row eight unavailable')
  } })
  const before = structuredClone(retryDb.rows)
  await expect(projection().createCloudShotProjectionRepository(retryDb).projectBatch({ jobId })).rejects.toThrow('row eight unavailable')
  expect(retryDb.rows).toEqual(before)
})

test('full public session and report/analysis readers are immediately available while normalized shots remain empty', async () => {
  const { db, session } = await publish(200)
  db.rows.field_live_sessions['session-1'].device_id = 'device-1'
  db.rows.field_live_sessions['session-1'].ended_at = NOW
  const { createCloudContentRepository, getSession } = require('../../cloudfunctions/client-api/lib/content')
  const content = createCloudContentRepository(db, Object.freeze({}))
  let result = await getSession({ sessionId: 'session-1', userId: 'user-a', repo: content })
  expect(result.shots).toHaveLength(200)
  expect(result.shots.at(-1)).toMatchObject({ shot_id: 'shot-200', sequence: 200 })
  expect(result).not.toHaveProperty('_shot_projection_job_id')
  expect(result).not.toHaveProperty('_sync_publication')
  const frozen = session.shots.map(s => s.shot_id)
  db.rows.clips = { 'saved-200': { clip_id: 'saved-200', session_id: 'session-1', shot_id: 'shot-200',
    artifact_id: session.artifact_id, upload_status: 'published' } }
  const { createCloudSessionAnalysisRepository, putSessionAnalysis } = require('../../cloudfunctions/sync-api/lib/session-analysis')
  await putSessionAnalysis({ sessionId: 'session-1', tableId: 'table-1', deviceId: 'device-1', idempotencyKey: 'analysis-1',
    repo: createCloudSessionAnalysisRepository(db), projection: { schema_version: '1.0', session_id: 'session-1', revision: 1,
      status: 'PARTIAL', total_shots: 200, available_shots: 1, retrying_shots: 199, report_available: true,
      deadline_at: LATER, updated_at: NOW, frozen_shots_digest: hash(JSON.stringify(frozen)), frozen_shot_ids: frozen,
      published_shots: [{ session_id: 'session-1', shot_id: 'shot-200' }],
      published_clips: [{ session_id: 'session-1', shot_id: 'shot-200', clip_id: 'saved-200' }] } })
  result = await getSession({ sessionId: 'session-1', userId: 'user-a', repo: content })
  expect(result.analysis).toMatchObject({ report_available: true, total_shots: 200 })
  expect(Object.keys(db.rows.shots || {})).toHaveLength(0)
  expect(db.rows.shot_projection_jobs[jobId].status).toBe('pending')
})

test.each(['missing', 'changed'])('completed replay revalidates %s source without rebuilding rows', async kind => {
  const { db } = await publish(1)
  const repo = projection().createCloudShotProjectionRepository(db)
  await repo.projectBatch({ jobId })
  const rows = structuredClone(db.rows.shots)
  if (kind === 'missing') delete db.rows.sessions['session-1']
  else db.rows.sessions['session-1'].shots[0].started_ms++
  expect(await repo.projectBatch({ jobId })).toEqual({ outcome: 'blocked', job_id: jobId, projected_count: 0, next_sequence: 2 })
  expect(db.rows.shot_projection_jobs[jobId]).toMatchObject({ status: 'blocked',
    blocked_reason: kind === 'missing' ? 'source_unavailable' : 'source_conflict' })
  expect(await repo.projectBatch({ jobId })).toMatchObject({ outcome: 'blocked', projected_count: 0 })
  expect(db.rows.shots).toEqual(rows)
})

test('valid completed replay only reads cursor and source and never repairs deleted normalized rows', async () => {
  const { db } = await publish(1)
  const repo = projection().createCloudShotProjectionRepository(db)
  await repo.projectBatch({ jobId })
  delete db.rows.shots[rowId(1)]
  const before = structuredClone(db.rows)
  expect(await repo.projectBatch({ jobId })).toMatchObject({ outcome: 'completed', projected_count: 0 })
  expect(db.rows).toEqual(before)
  expect(db.transactionAttempts.at(-1).count).toBe(2)
})

test('saved-clip source validation reads complete authoritative facts while projection remains pending', async () => {
  const saved = require('../../../../tests/fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json')
  const { ownershipSeed } = require('../helpers/private-clips')
  const { session, seed } = fixture(200)
  session.artifact_id = saved.published_session.artifact_id
  session.shots[0] = structuredClone(saved.published_session.shots[0])
  const db = ownerDeleteDb({ ...seed, ...ownershipSeed() }, { strictTransactions: true })
  const syncRepo = createCloudSyncRepository(db)
  const args = { envelope: { schema_version: '1.0', session, media: [] }, idempotencyKey: 'sync-saved',
    manifestHash: 'b'.repeat(64), tableId: 'table-1', deviceId: 'device-1', now: NOW, idFactory: () => 'source-job', repo: syncRepo }
  await createSyncJob(args)
  await syncRepo.markCompleted('source-job', NOW, {})
  const beforeSource = structuredClone(db.rows.sessions)
  // Existing latest-role saved-source queries have their own native acceptance
  // gate. Only that existing path uses the permissive adapter in this test.
  db.setStrictTransactions(false)
  const { createCloudSavedClipRepository, createSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
  const job = await createSavedClipJob({ descriptor: structuredClone(saved.descriptor),
    idempotencyKey: saved.descriptor.clip_id, tableId: 'table-1', deviceId: 'device-1', now: NOW,
    idFactory: () => 'saved-job', repo: createCloudSavedClipRepository(db) })
  expect(job).toMatchObject({ status: 'pending', clip_id: saved.descriptor.clip_id })
  expect(db.rows.sessions).toEqual(beforeSource)
  expect(db.rows.shot_projection_jobs[jobId]).toMatchObject({ status: 'pending', next_sequence: 1 })
  expect(Object.keys(db.rows.shots || {})).toHaveLength(0)
})

test.each([false, 0, ''])('falsy malformed cursor %s is never treated as absent or overwritten', async value => {
  const { db } = await publish(1)
  db.rows.shot_projection_jobs[jobId] = value
  const source = db.rows.sessions['session-1']
  await expect(projection().createCloudShotProjectionRepository(db).projectBatch({ jobId }))
    .rejects.toMatchObject({ code: 'SHOT_PROJECTION_INVALID' })
  await expect(db.runTransaction(tx => projection().enqueuePublishedShots(tx, source)))
    .rejects.toMatchObject({ code: 'SHOT_PROJECTION_INVALID' })
  expect(db.rows.shot_projection_jobs[jobId]).toBe(value)
})

test.each(['source', 'row'])('falsy malformed %s blocks without replacing it', async kind => {
  const { db } = await publish(1)
  if (kind === 'source') db.rows.sessions['session-1'] = false
  else db.rows.shots = { [rowId(1)]: false }
  expect(await projection().createCloudShotProjectionRepository(db).projectBatch({ jobId })).toMatchObject({ outcome: 'blocked' })
  expect(db.rows.shot_projection_jobs[jobId].blocked_reason).toBe(kind === 'source' ? 'source_conflict' : 'row_conflict')
  if (kind === 'source') expect(db.rows.sessions['session-1']).toBe(false)
  else expect(db.rows.shots[rowId(1)]).toBe(false)
})
