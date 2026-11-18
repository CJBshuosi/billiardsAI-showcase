const crypto = require('crypto')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { createCloudSyncRepository, createSyncJob } = require('../../cloudfunctions/sync-api/lib/ingest')

const TOKEN = 'synthetic-worker-secret'
const NOW = '2026-10-29T10:00:00.000Z'
const hash = value => crypto.createHash('sha256').update(value).digest('hex')
const JOB_ID = `shot-projection-${hash('session-1')}`
const request = () => ({ action: 'run_pending', token: TOKEN })
const handler = options => require('../../cloudfunctions/shot-projection-worker/lib/handler')
  .createShotProjectionWorkerHandler(options)

function countingRepo() {
  let calls = 0
  return { get calls() { return calls },
    async findPending() { calls++; return null },
    async projectBatch() { calls++; throw new Error('unexpected batch') } }
}

test.each([undefined, null, {}, { action: 'run_pending' }, { ...request(), token: 'wrong' }])(
  'unauthorized worker input %p cannot touch the repository', async input => {
    const repo = countingRepo()
    expect(await handler({ repo, verifyToken: token => token === TOKEN })(input))
      .toEqual({ ok: false, error: { code: 'SHOT_PROJECTION_UNAUTHORIZED' } })
    expect(repo.calls).toBe(0)
  },
)

test.each([
  { ...request(), action: 'other' }, { ...request(), session_id: 'foreign' },
  { ...request(), facts: [] }, { ...request(), limit: 200 }, { ...request(), jobId: JOB_ID },
  { ...request(), session: {} }, { ...request(), schema_version: '1.0' },
])('authenticated body scope injection %p cannot touch the repository', async input => {
  const repo = countingRepo()
  expect(await handler({ repo, verifyToken: token => token === TOKEN })(input))
    .toEqual({ ok: false, error: { code: 'SHOT_PROJECTION_REQUEST_INVALID' } })
  expect(repo.calls).toBe(0)
})

test('an authorized empty queue returns only idle counts', async () => {
  expect(await handler({ repo: countingRepo(), verifyToken: () => true })(request()))
    .toEqual({ ok: true, data: { outcome: 'idle', projected_count: 0 } })
})

test.each(['progress', 'completed', 'blocked'])('worker returns safe %s projection fields only', async outcome => {
  const repo = { async findPending() { return { job_id: JOB_ID, session: 'private facts', token: TOKEN } },
    async projectBatch(input) {
      expect(input).toEqual({ jobId: JOB_ID })
      return { outcome, job_id: JOB_ID, projected_count: outcome === 'blocked' ? 0 : 16,
        next_sequence: 17, token: TOKEN, facts: 'private facts', sdk: 'raw error' }
    } }
  expect(await handler({ repo, verifyToken: () => true })(request())).toEqual({ ok: true, data: {
    outcome, job_id: JOB_ID, projected_count: outcome === 'blocked' ? 0 : 16, next_sequence: 17,
  } })
})

test.each(['findPending', 'projectBatch'])('worker masks %s errors and supports a later retry', async method => {
  let fail = true
  const repo = { async findPending() {
    if (fail && method === 'findPending') throw Object.assign(new Error(TOKEN), { code: TOKEN })
    return { job_id: JOB_ID }
  }, async projectBatch() {
    if (fail) throw Object.assign(new Error(`SDK ${TOKEN}`), { code: TOKEN })
    return { outcome: 'completed', job_id: JOB_ID, projected_count: 1, next_sequence: 2 }
  } }
  const run = handler({ repo, verifyToken: () => true })
  expect(await run(request())).toEqual({ ok: false, error: { code: 'SHOT_PROJECTION_UNAVAILABLE' } })
  fail = false
  expect(await run(request())).toMatchObject({ ok: true, data: { outcome: 'completed' } })
})

async function publish(count) {
  const session = { schema_version: '1.0', session_id: 'session-1', artifact_id: 'artifact-1', status: 'ended',
    created_at_utc: '2026-10-29T09:00:00.000Z', ended_at_utc: NOW,
    evidence_source_kind: 'usb_live', postgame_manifest_sha256: 'b'.repeat(64),
    participants: [{ user_id: 'user-a', role: 'red' }, { user_id: 'user-b', role: 'blue' }],
    breaker: 'red', active_role_at_end: 'blue', limitations: [],
    shots: Array.from({ length: count }, (_, i) => ({ shot_id: `shot-${i + 1}`, sequence: i + 1,
      live_job_id: `live-${i + 1}`, actor_role: i % 2 ? 'blue' : 'red', started_ms: i * 1000,
      settled_ms: i * 1000 + 500, limitations: [] })) }
  const db = ownerDeleteDb({ field_live_sessions: { 'session-1': { session_id: 'session-1', table_id: 'table-1',
    status: 'ended', red_user_id: 'user-a', blue_user_id: 'user-b' } },
  session_role_bindings: Object.fromEntries(session.participants.map(p => [hash(`session-1:${p.role}`),
    { ...p, session_id: 'session-1', confirmed_at: NOW }])) }, { strictTransactions: true })
  const repo = createCloudSyncRepository(db)
  await createSyncJob({ envelope: { schema_version: '1.0', session, media: [] }, idempotencyKey: 'sync-1',
    manifestHash: 'b'.repeat(64), tableId: 'table-1', deviceId: 'device-1', idFactory: () => 'sync-job-1', now: NOW, repo })
  await repo.markCompleted('sync-job-1', NOW, {})
  return db
}

function reconstructMain(db) {
  jest.doMock('wx-server-sdk', () => ({
    DYNAMIC_CURRENT_ENV: 'synthetic-current-env',
    init(options) {
      if (!db) throw new Error('SDK initialized before authorization')
      expect(options).toEqual({ env: 'synthetic-current-env' })
    },
    database() { return db },
  }))
  return event => {
    let response
    jest.isolateModules(() => { response = require('../../cloudfunctions/index').main(event) })
    return response
  }
}

const originalEnv = { ...process.env }
afterEach(() => {
  process.env = { ...originalEnv }
  jest.dontMock('wx-server-sdk')
  jest.restoreAllMocks()
})

test.each([undefined, '', ' ', `${TOKEN}\n`, `${TOKEN}\r`, '{{env.SHOT_PROJECTION_WORKER_TOKEN}}', '${SHOT_PROJECTION_WORKER_TOKEN}', '<replace-me>']) (
  'real main rejects missing or template configured token %p before database access', async configured => {
    process.env.BILLIARDS_FUNCTION_ENTRY = 'shot-projection-worker'
    if (configured === undefined) delete process.env.SHOT_PROJECTION_WORKER_TOKEN
    else process.env.SHOT_PROJECTION_WORKER_TOKEN = configured
    const main = reconstructMain(null)
    expect(await main({ action: 'run_pending', token: configured || TOKEN }))
      .toEqual({ ok: false, error: { code: 'SHOT_PROJECTION_UNAUTHORIZED' } })
  },
)

test.each(['wrong', '', 4, null, `${TOKEN}\n`, `${TOKEN}suffix`])('real main rejects invalid candidate %p before SDK initialization', async token => {
  process.env.BILLIARDS_FUNCTION_ENTRY = 'shot-projection-worker'
  process.env.SHOT_PROJECTION_WORKER_TOKEN = TOKEN
  const main = reconstructMain(null)
  expect(await main({ action: 'run_pending', token }))
    .toEqual({ ok: false, error: { code: 'SHOT_PROJECTION_UNAUTHORIZED' } })
})

test.each([90, 200])('real main reconstructs and drains %i committed shots without another device request', async count => {
  const db = await publish(count)
  const before = structuredClone(db.rows)
  expect(before.sessions['session-1'].shots).toHaveLength(count)
  expect(Object.keys(before.shots || {})).toHaveLength(0)
  expect(db.transactionAttempts.at(-1).count).toBe(13)
  process.env.BILLIARDS_FUNCTION_ENTRY = 'shot-projection-worker'
  process.env.SHOT_PROJECTION_WORKER_TOKEN = TOKEN
  const attemptsBefore = db.transactionAttempts.length
  let total = 0
  for (let i = 0; i < Math.ceil(count / 16); i++) {
    // New module state for every scheduler invocation; the only durable state is DB.
    const response = await reconstructMain(db)(request())
    expect(response).toEqual({ ok: true, data: { outcome: i === Math.ceil(count / 16) - 1 ? 'completed' : 'progress',
      job_id: JOB_ID, projected_count: Math.min(16, count - i * 16), next_sequence: Math.min(count + 1, i * 16 + 17) } })
    total += response.data.projected_count
  }
  expect(total).toBe(count)
  expect(Object.keys(db.rows.shots)).toHaveLength(count)
  expect(db.rows.shot_projection_jobs[JOB_ID]).toMatchObject({ status: 'completed', next_sequence: count + 1 })
  expect(await reconstructMain(db)(request())).toEqual({ ok: true, data: { outcome: 'idle', projected_count: 0 } })
  for (const name of Object.keys(before).filter(name => name !== 'shot_projection_jobs' && name !== 'shots')) {
    expect(db.rows[name]).toEqual(before[name])
  }
  const attempts = db.transactionAttempts.slice(attemptsBefore)
  expect(attempts).toHaveLength(Math.ceil(count / 16))
  expect(Math.max(...attempts.map(attempt => attempt.count))).toBe(35)
  expect(attempts.every(attempt => attempt.count <= 45)).toBe(true)
  expect(attempts.flatMap(attempt => attempt.calls).filter(call => call.method !== 'get')
    .every(call => ['shots', 'shot_projection_jobs'].includes(call.collection))).toBe(true)
})

test('real main blocks corrupt source without changing authoritative state', async () => {
  const db = await publish(33)
  db.rows.sessions['session-1'].shots[32].started_ms++
  const before = structuredClone(db.rows)
  process.env.BILLIARDS_FUNCTION_ENTRY = 'shot-projection-worker'
  process.env.SHOT_PROJECTION_WORKER_TOKEN = TOKEN
  expect(await reconstructMain(db)(request())).toEqual({ ok: true, data: {
    outcome: 'blocked', job_id: JOB_ID, projected_count: 0, next_sequence: 1,
  } })
  for (const name of Object.keys(before).filter(name => name !== 'shot_projection_jobs' && name !== 'shots')) {
    expect(db.rows[name]).toEqual(before[name])
  }
  expect(Object.keys(db.rows.shots || {})).toHaveLength(0)
  expect(db.transactionAttempts.at(-1).count).toBe(3)
})

test('real main rolls back a partial batch error and recovers after process loss without device traffic', async () => {
  const published = await publish(33)
  const before = structuredClone(published.rows)
  let writes = 0
  let failing = true
  const db = ownerDeleteDb(before, { strictTransactions: true, failWrite(collection) {
    if (collection === 'shots' && failing && ++writes === 9) throw new Error(`SDK secret ${TOKEN}`)
  } })
  process.env.BILLIARDS_FUNCTION_ENTRY = 'shot-projection-worker'
  process.env.SHOT_PROJECTION_WORKER_TOKEN = TOKEN
  expect(await reconstructMain(db)(request())).toEqual({ ok: false, error: { code: 'SHOT_PROJECTION_UNAVAILABLE' } })
  expect(db.rows.shot_projection_jobs).toEqual(before.shot_projection_jobs)
  expect(Object.keys(db.rows.shots || {})).toHaveLength(0)
  failing = false
  for (let i = 0; i < 3; i++) await reconstructMain(db)(request())
  expect(db.rows.shot_projection_jobs[JOB_ID]).toMatchObject({ status: 'completed', next_sequence: 34 })
  expect(Object.keys(db.rows.shots)).toHaveLength(33)
  for (const name of Object.keys(before).filter(name => name !== 'shot_projection_jobs' && name !== 'shots')) {
    expect(db.rows[name]).toEqual(before[name])
  }
  expect(db.transactionAttempts.map(attempt => attempt.count)).toEqual([27, 35, 35, 5])
})
