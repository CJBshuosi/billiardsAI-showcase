const crypto = require('crypto')
const { EventEmitter } = require('events')
const { Readable } = require('stream')
const { setup } = require('../helpers/managed-saved-flow-fixture')
const { createSavedClipJob, authorizeSavedClip, completeSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const { createCloudOwnerDeleteRepository, requestOwnerClipDeletion } = require('../../cloudfunctions/shared/owner-delete')
const { digest } = require('../../cloudfunctions/shared/shot-ownership')
const erasure = require('../../cloudfunctions/shared/owner-cloud-erasure')
const { createOwnerErasureProvider } = require('../../cloudfunctions/sync-api/lib/owner-erasure-provider')
const NOW = Date.parse('2026-11-01T10:00:00.000Z')
const environmentId = 'env-test'; const bucket = 'media-123456'; const region = 'ap-guangzhou'
const file = key => `cloud://${environmentId}.${bucket}/${key}`
const uploadRequest = () => ({ contract: 'managed_upload_request_v1', request_id: crypto.randomUUID() })
const terminal = (request, attempt = null) => ({ contract: 'managed_upload_receipt_v1', upload_request: request, upload_attempt: attempt,
  evidence: attempt ? { kind: 'http200', local_binding_sha256: 'a'.repeat(64), started_at_utc: '2024-02-29T23:59:59.123456Z', received_at_utc: '2024-02-29T23:59:59.123457Z', status_code: 200 }
    : { kind: 'retired_before_start', local_binding_sha256: 'a'.repeat(64) } })
// Declared controlled native edge; no deployed acceptance evidence exists.
function controlledAdmission({ scope, assessment }) {
  return { contract: 'owner_erasure_native_admission_v1', scope_sha256: digest(scope), inputs_sha256: assessment.inputs_sha256,
    targets_sha256: digest(assessment.targets), environment_id: environmentId, bucket, region,
    provider_policy: 'cos_exact_target_absence_v1', evidence: { binding_sha256: '1'.repeat(64), writer_drain_sha256: '2'.repeat(64),
      visibility_sha256: '3'.repeat(64), configuration_fence_sha256: '4'.repeat(64), stale_request_idempotence_sha256: '5'.repeat(64) } }
}
async function fixture({ missing = false, orphan = 'server_succeeded', dbOptions = {}, full = false, foreign = false } = {}) {
  const s = await setup({ dbOptions }); s.now = NOW; s.job = await createSavedClipJob(s.input)
  s.storage = { authorizeRegisteredUpload: async input => ({ outcome: 'external_issued', file_id: foreign ? `cloud://foreign/${input.object_key}` : file(input.object_key),
    authorization: { method: 'PUT', url: 'https://upload.invalid', headers: {}, expires_in_seconds: 300 } }),
  copyRegisteredObject: async input => ({ outcome: 'server_succeeded', file_id: file(input.publish_object_key) }) }
  s.upload = async (state = 'success') => {
    const request = uploadRequest(); const grant = await authorizeSavedClip({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage: s.storage, uploadRequest: request })
    if (state) await s.repo.recordTrackedUploadReceipt({ ...s.scope, jobId: s.job.job_id, receipt: terminal(request, state === 'success' ? grant.upload_attempt : null) })
    return { request, grant }
  }
  s.first = await s.upload(missing ? null : 'success')
  if (full) { for (let i = 1; i < 62; i++) await s.upload(i % 2 ? 'retired' : 'success') }
  else await s.upload('retired')
  if (!full) {
    const ticket = await s.authority.beginOperation(s.db, { identity: s.identity, ingestJobId: s.job.job_id, kind: 'upload_authorization' })
    await s.authority.recordOperationResult({ ticket, outcome: 'no_write', objects: [], reason: 'not_started' })
  }
  const { ticket } = await s.repo.beginTrackedCopy({ ...s.scope, jobId: s.job.job_id, uploadAttempt: s.first.grant.upload_attempt }); s.orphan = ticket
  if (orphan) await s.repo.recordTrackedResult({ ticket, outcome: orphan, objects: orphan === 'server_succeeded' ? [{ kind: ticket.target.kind, cloud_path: ticket.target.cloud_path, file_id: file(ticket.target.cloud_path) }] : [], reason: orphan === 'unknown' ? 'result_lost' : null })
  await completeSavedClipJob({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage: s.storage, uploadAttempt: s.first.grant.upload_attempt, now: new Date(NOW).toISOString() })
  const receipt = await requestOwnerClipDeletion({ clipId: s.descriptor.clip_id, requestId: 'owner-request', userId: 'user-a', repo: createCloudOwnerDeleteRepository(s.db), now: new Date(NOW).toISOString() })
  s.jobId = receipt.deletion_job_id; s.original = structuredClone(s.db.rows)
  s.repository = options => erasure.createOwnerCloudErasureRepository(s.db, { authority: s.authority, clock: () => s.now, idFactory: crypto.randomUUID, nativeContractAdmission: controlledAdmission, ...options })
  return s
}
function controlledProvider(s, intercept, getCredentials = () => ({ SecretId: 'controlled-id', SecretKey: 'controlled-secret' })) {
  const calls = []
  const request = (options, onResponse) => {
    calls.push(options)
    const req = new EventEmitter(); req.destroy = () => {}; req.end = () => queueMicrotask(() => {
      const versioning = options.path.includes('versioning')
      const body = versioning ? '<VersioningConfiguration/>' : options.method === 'DELETE' ? '' : `<Error><Code>NoSuchKey</Code><Resource>${options.path.split('?')[0]}</Resource></Error>`
      const res = Readable.from([Buffer.from(body)]); res.statusCode = versioning ? 200 : options.method === 'DELETE' ? 204 : 404
      res.headers = {}; res.rawHeaders = []; if (intercept) intercept(options, res); onResponse(res)
    }); return req
  }
  return { calls, request, ...createOwnerErasureProvider({ environmentId, bucket, region, clock: () => s.now,
    getCredentials, request }) }
}
function worker(repository, provider) { return require('../../cloudfunctions/sync-api/lib/owner-cloud-erasure-worker').createOwnerCloudErasureWorker({ repository, provider }) }
async function drain(s, repository, provider, max = 30) {
  let outcome
  for (let i = 0; i < max; i++) { const before = provider.calls.length; outcome = await worker(repository, provider).runOnce({ jobId: s.jobId }); expect(provider.calls.length - before).toBeLessThanOrEqual(1); if (outcome.receipt) return outcome.receipt }
  throw new Error(`not complete: ${JSON.stringify(outcome)}`)
}
function unchanged(s) { for (const name of Object.keys(s.original)) expect(s.db.rows[name]).toEqual(s.original[name]) }

test('actual owner delete erases all reserved targets and replays one immutable receipt', async () => {
  const s = await fixture(); const repository = s.repository(); const provider = controlledProvider(s)
  const receipt = await drain(s, repository, provider)
  expect(receipt).toMatchObject({ contract: 'owner_cloud_erase_receipt_v1', deletion_job_id: s.jobId, environment_id: environmentId,
    scope: s.db.rows.owner_clip_tombstones[s.descriptor.clip_id], authority_epoch: s.db.rows.clip_storage_authorities[s.descriptor.clip_id].epoch })
  expect(receipt.targets.map(t => t.writer_state)).toEqual(['upload_succeeded', 'upload_retired', 'no_write', 'copy_succeeded', 'copy_succeeded'])
  expect(receipt.results.map(r => r.status)).toEqual(['absent', 'absent', 'excluded', 'absent', 'absent'])
  expect(provider.calls.filter(c => c.method === 'DELETE').map(c => c.path)).toEqual(receipt.targets.filter(t => t.writer_state !== 'no_write').map(t => `/${t.cloud_path}`))
  expect(receipt.receipt_sha256).toBe(digest(Object.fromEntries(Object.entries(receipt).filter(([k]) => k !== 'receipt_sha256'))))
  s.now += 100000
  expect((await worker(repository, provider).runOnce({ jobId: s.jobId })).receipt).toEqual(receipt)
  expect(provider.calls).toHaveLength(16); unchanged(s)
})
test.each([null, true, false, () => true, () => ({}), async () => ({})])('invalid native admission %p cannot execute', async admission => {
  const s = await fixture(); const provider = controlledProvider(s); const repository = s.repository({ nativeContractAdmission: admission })
  expect(await repository.claim({ jobId: s.jobId, leaseSeconds: 30 })).toBeNull()
  expect(await worker(repository, provider).runOnce({ jobId: s.jobId })).toMatchObject({ status: 'pending', receipt: null })
  expect(provider.calls).toHaveLength(0); unchanged(s)
})
test.each(['scope_sha256', 'inputs_sha256', 'targets_sha256', 'environment_id', 'bucket', 'provider_policy', 'evidence', 'extra', 'accessor'])('mismatched native %s denies', async field => {
  const s = await fixture(); const repository = s.repository({ nativeContractAdmission(input) {
    const record = controlledAdmission(input)
    if (field === 'evidence') record.evidence.visibility_sha256 = null
    else if (field === 'accessor') Object.defineProperty(record, 'region', { get() { throw new Error('must not invoke') }, enumerable: true })
    else record[field] = field.endsWith('sha256') ? '0'.repeat(64) : 'foreign'
    return record
  } })
  expect(await repository.claim({ jobId: s.jobId, leaseSeconds: 30 })).toBeNull()
})
test.each([{ missing: true }, { orphan: null }, { orphan: 'unknown' }, { foreign: true }])('unresolved or foreign writer %p denies', async options => {
  const s = await fixture(options); expect(await s.repository().claim({ jobId: s.jobId, leaseSeconds: 30 })).toBeNull(); unchanged(s)
})
test.each(['tombstone', 'ownership', 'save', 'scope', 'storage', 'summary'])('corrupt original %s denies', async change => {
  const s = await fixture(); const job = s.db.rows.owner_clip_deletion_jobs[s.jobId]; const tomb = s.db.rows.owner_clip_tombstones[s.descriptor.clip_id]
  if (change === 'tombstone') tomb.owner_user_id = 'other'
  if (change === 'ownership') { job.ownership_receipt.actor_role = 'blue'; tomb.ownership_receipt.actor_role = 'blue' }
  if (change === 'save') { job.save_request_id = 'other'; tomb.save_request_id = 'other' }
  if (change === 'scope') { job.device_id = 'other'; tomb.device_id = 'other' }
  if (change === 'storage') { job.storage_objects[0].file_id = 'cloud://foreign'; tomb.storage_objects[0].file_id = 'cloud://foreign' }
  if (change === 'summary') { job.storage_authority.epoch = crypto.randomUUID(); tomb.storage_authority.epoch = job.storage_authority.epoch }
  expect(await s.repository().claim({ jobId: s.jobId, leaseSeconds: 30 })).toBeNull()
})

function barrier() { let release; let enter; const ready = new Promise(r => { enter = r }); const gate = new Promise(r => { release = r }); return { ready, release, async hold() { enter(); return gate } } }
test('expired/replaced tokens cannot load, record, or finalize a claim', async () => {
  const s = await fixture(); const repository = s.repository(); const claim = await repository.claim({ jobId: s.jobId, leaseSeconds: 15 })
  expect(await repository.claim({ jobId: s.jobId, leaseSeconds: 30 })).toBeNull()
  const step = await repository.loadStep(claim); const result = await controlledProvider(s).step(step)
  expect(await repository.recordStep({ ...claim, token: crypto.randomUUID() }, result)).toBe(false)
  s.now += 15000
  expect(await repository.loadStep(claim)).toBeNull(); expect(await repository.recordStep(claim, result)).toBe(false); expect(await repository.finalize(claim)).toBeNull()
  const next = await repository.claim({ jobId: s.jobId, leaseSeconds: 30 }); expect(next.token).not.toBe(claim.token)
  expect(await repository.recordStep(claim, result)).toBe(false)
  expect(await repository.recordStep(next, result)).toBe(true)
  expect(await repository.recordStep(next, result)).toBe(false)
  unchanged(s)
})

test('crash after physical delete and before checkpoint repeats the exact durable target', async () => {
  const s = await fixture(); const repository = s.repository(); const provider = controlledProvider(s)
  await worker(repository, provider).runOnce({ jobId: s.jobId }) // versioning
  const claim = await repository.claim({ jobId: s.jobId, leaseSeconds: 30 }); const step = await repository.loadStep(claim)
  const result = await provider.step({ ...step, beforeRequest: async () => (await repository.loadStep(claim)) !== null })
  expect(result.checkpoint.phase).toBe('simple_probe') // simulated process death: no record
  s.now += 30001
  await worker(repository, provider).runOnce({ jobId: s.jobId })
  expect(provider.calls.slice(1, 3).map(c => [c.method, c.path])).toEqual([['DELETE', `/${step.target.cloud_path}`], ['DELETE', `/${step.target.cloud_path}`]])
  expect((await drain(s, repository, provider)).results).toHaveLength(5); unchanged(s)
})

test('native completion after lease takeover cannot checkpoint or finalize the old claim', async () => {
  const s = await fixture(); const repository = s.repository(); const provider = controlledProvider(s); const pause = barrier()
  const oldProvider = { step: async step => { const result = await provider.step(step); await pause.hold(); return result } }
  const old = worker(repository, oldProvider).runOnce({ jobId: s.jobId }); await pause.ready
  s.now += 30001
  await worker(repository, provider).runOnce({ jobId: s.jobId })
  const progress = structuredClone(s.db.rows.owner_cloud_erasure_executions[s.jobId])
  pause.release(); expect(await old).toEqual({ status: 'pending', receipt: null })
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId]).toEqual(progress)
  expect((await drain(s, repository, provider)).results).toHaveLength(5)
})

test.each(['lease-expiry', 'lease-takeover', 'receipt-only'])('beforeRequest refuses %s during lazy IAM', async change => {
  const s = await fixture(); const repository = s.repository(); const gate = barrier(); const provider = controlledProvider(s, null, async () => { await gate.hold(); return { SecretId: 'controlled', SecretKey: 'controlled' } })
  const run = worker(repository, provider).runOnce({ jobId: s.jobId }); await gate.ready
  if (change === 'receipt-only') Object.values(s.db.rows.upload_terminal_receipts)[0].entries[0].received_at_utc = '2026-11-01T10:00:01.000Z'
  else s.now += 30001
  if (change === 'lease-takeover') await repository.claim({ jobId: s.jobId, leaseSeconds: 30 })
  gate.release(); expect(await run).toEqual({ status: 'pending', receipt: null }); expect(provider.calls).toHaveLength(0)
})

test('receipt-only fingerprint changes require expired-claim reassessment and restart every proof', async () => {
  const s = await fixture(); const repository = s.repository(); const provider = controlledProvider(s)
  const claim = await repository.claim({ jobId: s.jobId, leaseSeconds: 30 }); const step = await repository.loadStep(claim)
  const result = await provider.step(step); const revision = s.db.rows.clip_storage_authorities[s.descriptor.clip_id].revision
  Object.values(s.db.rows.upload_terminal_receipts)[0].entries[0].received_at_utc = '2026-11-01T10:00:01.000Z'
  expect(s.db.rows.clip_storage_authorities[s.descriptor.clip_id].revision).toBe(revision)
  expect(await repository.recordStep(claim, result)).toBe(false)
  expect(await repository.claim({ jobId: s.jobId, leaseSeconds: 30 })).toBeNull()
  s.now += 30001
  const next = await repository.claim({ jobId: s.jobId, leaseSeconds: 30 }); expect(next.inputs_sha256).not.toBe(claim.inputs_sha256)
  expect((await repository.loadStep(next)).checkpoint).toBeNull()
  await repository.finalize(next)
  expect((await drain(s, repository, provider)).inputs_sha256).toBe(next.inputs_sha256)
})

test('late original alias/outcome remains required and all-target authority includes the unpublished copy', async () => {
  const s = await fixture({ orphan: null }); const repository = s.repository()
  expect(await repository.claim({ jobId: s.jobId, leaseSeconds: 30 })).toBeNull()
  const ticket = s.orphan
  await s.repo.recordTrackedResult({ ticket, outcome: 'server_succeeded', objects: [{ kind: ticket.target.kind, cloud_path: ticket.target.cloud_path, file_id: file(ticket.target.cloud_path) }], reason: null })
  const provider = controlledProvider(s); const receipt = await drain(s, repository, provider)
  expect(receipt.targets[3].cloud_path).toBe(ticket.target.cloud_path)
  expect(receipt.scope.storage_objects.some(r => r.file_id === file(ticket.target.cloud_path))).toBe(false)
  expect(provider.calls.filter(c => c.method === 'DELETE').map(c => c.path)).toContain(`/${ticket.target.cloud_path}`)
})

test('permission failure or partial absence never writes a final receipt', async () => {
  const s = await fixture(); const repository = s.repository(); const provider = controlledProvider(s, (options, res) => { if (options.method === 'DELETE') res.statusCode = 403 })
  for (let i = 0; i < 8; i++) expect(await worker(repository, provider).runOnce({ jobId: s.jobId })).toEqual({ status: 'pending', receipt: null })
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId].results).toEqual([])
  expect(Object.keys(s.db.rows.owner_cloud_erasure_receipts)).toHaveLength(0)
  const permitted = controlledProvider(s); for (let i = 0; i < 3; i++) await worker(repository, permitted).runOnce({ jobId: s.jobId })
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId].results).toHaveLength(1)
  expect(Object.keys(s.db.rows.owner_cloud_erasure_receipts)).toHaveLength(0); unchanged(s)
})

test.each(['target', 'transcript', 'checkpoint', 'mode', 'code', 'extra'])('malformed provider %s cannot commit progress or completion', async change => {
  const s = await fixture(); const repository = s.repository(); const claim = await repository.claim({ jobId: s.jobId, leaseSeconds: 30 })
  const result = await controlledProvider(s).step(await repository.loadStep(claim))
  if (change === 'target') result.evidence.target_sha256 = 'f'.repeat(64)
  if (change === 'transcript') result.evidence.transcript_sha256 = 'f'.repeat(64)
  if (change === 'checkpoint') result.checkpoint.versions = Array(17).fill('v')
  if (change === 'mode') result.evidence.versioning_mode = 'Enabled'
  if (change === 'code') result.evidence.code = 'private-provider-error'
  if (change === 'extra') result.evidence.secret = 'must not persist'
  const before = structuredClone(s.db.rows.owner_cloud_erasure_executions[s.jobId])
  expect(await repository.recordStep(claim, result)).toBe(false)
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId]).toEqual(before)
  expect(await repository.finalize(claim)).toBeNull()
})

test('factory default and booleans stay disabled; only bound internal composition constructs the executor', async () => {
  const { createRuntimeDependencies } = require('../../cloudfunctions/sync-api'); const s = await fixture(); let credentials = 0; let requests = 0
  const providerOptions = { environmentId, bucket, region, getCredentials: () => { credentials++; throw new Error('no credential activity during construction') }, request() { requests++; throw new Error('no native request') } }
  const config = { db: s.db, storageCloud: { getUploadMetadata() {}, downloadFile() {} }, environment: { OWNER_CLOUD_ERASURE_ENABLED: 'true', CLOUDBASE_ENV_ID: environmentId }, clock: () => s.now }
  for (const ownerCloudErasure of [undefined, null, true, { enabled: true }, { nativeContractAdmission: true, providerOptions }]) expect(createRuntimeDependencies({ ...config, ownerCloudErasure }).ownerCloudErasureWorker).toBeNull()
  const enabled = createRuntimeDependencies({ ...config, ownerCloudErasure: { nativeContractAdmission: controlledAdmission, providerOptions } })
  expect(typeof enabled.ownerCloudErasureWorker.runOnce).toBe('function'); expect(credentials).toBe(0); expect(requests).toBe(0)
  const mismatch = createRuntimeDependencies({ ...config, ownerCloudErasure: { nativeContractAdmission: controlledAdmission, providerOptions: { ...providerOptions, region: 'ap-shanghai' } } })
  expect(await mismatch.ownerCloudErasureWorker.runOnce({ jobId: s.jobId })).toEqual({ status: 'pending', receipt: null })
  expect(credentials).toBe(0); expect(requests).toBe(0)
})

test('N64/A64 and four receipt pages fit all repository callbacks including each forced retry', async () => {
  let retry = false
  const s = await fixture({ full: true, dbOptions: { retryConflicts: 1, beforeCommit({ attempt }) { if (retry && attempt.index === 0 && attempt.calls.some(c => c.collection === 'upload_terminal_receipts')) { const e = new Error('controlled conflict'); e.code = 'SYNTHETIC_CONFLICT'; throw e } } } })
  const repository = s.repository(); const provider = controlledProvider(s); const start = s.db.transactionAttempts.length; retry = true
  const receipt = await drain(s, repository, provider, 260)
  expect(receipt.targets).toHaveLength(64); expect(receipt.results.every(r => r.status === 'absent')).toBe(true)
  const attempts = s.db.transactionAttempts.slice(start)
  expect(attempts.some(a => a.index === 1)).toBe(true)
  for (const a of attempts) {
    expect(a.count).toBeLessThanOrEqual(90)
    expect(a.calls.filter(c => c.collection === 'media_storage_objects' && c.method === 'get')).toHaveLength(64)
    expect(a.calls.filter(c => c.collection === 'upload_terminal_receipts')).toHaveLength(4)
    expect(a.calls.filter(c => c.collection === 'managed_upload_requests')).toHaveLength(0)
    expect(a.calls.filter(c => c.method !== 'get').every(c => ['owner_cloud_erasure_executions', 'owner_cloud_erasure_receipts', 'owner_cloud_erasure_witnesses'].includes(c.collection))).toBe(true)
  }
  console.info('OWNER_ERASE_N64_A64_CALLBACKS', { attempts: attempts.length, min: Math.min(...attempts.map(a => a.count)), max: Math.max(...attempts.map(a => a.count)), retries: attempts.filter(a => a.index === 1).length, native: provider.calls.length })
  unchanged(s)
}, 30000)

test.each(['receipt-only', 'birth', 'scope', 'progress'])('finalization rereads complete original %s and cannot bless changed evidence', async change => {
  const s = await fixture(); const repository = s.repository(); const provider = controlledProvider(s); let lastClaim
  const paused = { ...repository, finalize: async claim => {
    if (s.db.rows.owner_cloud_erasure_executions[s.jobId]?.index === 5) { lastClaim = claim; return null }
    return repository.finalize(claim)
  } }
  for (let i = 0; i < 16; i++) await worker(paused, provider).runOnce({ jobId: s.jobId })
  expect(lastClaim).toBeDefined(); expect(Object.keys(s.db.rows.owner_cloud_erasure_receipts)).toHaveLength(0)
  if (change === 'receipt-only') Object.values(s.db.rows.upload_terminal_receipts)[0].entries[0].received_at_utc = '2026-11-01T10:00:01.000Z'
  if (change === 'birth') s.db.rows.managed_session_births[s.identity.session_id].opened_at = '2026-11-01T09:59:00.000Z'
  if (change === 'scope') s.db.rows.owner_clip_tombstones[s.descriptor.clip_id].actor_role = 'blue'
  if (change === 'progress') s.db.rows.owner_cloud_erasure_executions[s.jobId].results.pop()
  expect(await repository.finalize(lastClaim)).toBeNull()
  expect(Object.keys(s.db.rows.owner_cloud_erasure_receipts)).toHaveLength(0)
})

test('real internal dependency factory executes one bounded native step only after admission', async () => {
  const { createRuntimeDependencies } = require('../../cloudfunctions/sync-api'); const s = await fixture(); const native = controlledProvider(s)
  const deps = createRuntimeDependencies({ db: s.db, storageCloud: { getUploadMetadata() {}, downloadFile() {} }, clock: () => s.now, environment: { CLOUDBASE_ENV_ID: environmentId },
    ownerCloudErasure: { nativeContractAdmission: controlledAdmission, providerOptions: { environmentId, bucket, region,
      getCredentials: () => ({ SecretId: 'controlled-id', SecretKey: 'controlled-secret' }), request: native.request } } })
  expect(await deps.ownerCloudErasureWorker.runOnce({ jobId: s.jobId })).toEqual({ status: 'pending', receipt: null })
  expect(native.calls).toHaveLength(1)
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId].checkpoint).toMatchObject({ mode: 'NeverEnabled', phase: 'simple_delete' })
  unchanged(s)
})

test.each([
  ['repository', 'immediate', 2], ['repository', 'subsequent', 2],
  ['factory', 'immediate', 1], ['factory', 'subsequent', 1],
])('%s contains %s rejected Promise admission without awaiting it', async (boundary, timing, calls) => {
  const s = await fixture()
  // A separate real Node process owns the unhandled-rejection event. Jest's own
  // process listeners cannot mask it; all domain/repository/factory code is real.
  const program = `
    const { ownerDeleteDb } = require('./tests/helpers/owner-delete-db')
    const { createOwnerCloudErasureRepository } = require('./cloudfunctions/shared/owner-cloud-erasure')
    const { createOwnerCloudErasureWorker } = require('./cloudfunctions/sync-api/lib/owner-cloud-erasure-worker')
    const { createRuntimeDependencies } = require('./cloudfunctions/sync-api')
    const seed = JSON.parse(require('fs').readFileSync(0, 'utf8'))
    const db = ownerDeleteDb(seed.rows)
    const boundary = process.argv[1]; const timing = process.argv[2]
    let unhandled = 0; let requests = 0; let credentials = 0; let admissionCalls = 0
    const deferred = []
    process.on('unhandledRejection', () => { unhandled++ })
    const admission = () => {
      admissionCalls++
      if (timing === 'immediate') return Promise.reject(new Error('RAW_ADMISSION_MUST_NOT_ESCAPE'))
      return new Promise((resolve, reject) => { deferred.push(reject) })
    }
    ;(async () => {
      let claim = null; let outcome
      if (boundary === 'repository') {
        const repository = createOwnerCloudErasureRepository(db, { clock: () => ${NOW}, nativeContractAdmission: admission })
        claim = await repository.claim({ jobId: seed.jobId })
        outcome = await createOwnerCloudErasureWorker({ repository, provider: { step() { requests++; throw new Error('must not run') } } }).runOnce({ jobId: seed.jobId })
      } else {
        const deps = createRuntimeDependencies({ db, storageCloud: { getUploadMetadata() {}, downloadFile() {} }, clock: () => ${NOW}, environment: { CLOUDBASE_ENV_ID: 'env-test' },
          ownerCloudErasure: { nativeContractAdmission: admission, providerOptions: {
            environmentId: 'env-test', bucket: 'media-123456', region: 'ap-guangzhou',
            getCredentials() { credentials++; throw new Error('must not run') },
            request() { requests++; throw new Error('must not run') },
          } } })
        outcome = await deps.ownerCloudErasureWorker.runOnce({ jobId: seed.jobId })
      }
      // The direct claim/worker must return before a subsequent Promise settles.
      for (const reject of deferred) reject(new Error('RAW_ADMISSION_MUST_NOT_ESCAPE'))
      await new Promise(resolve => setImmediate(() => setImmediate(resolve)))
      process.stdout.write(JSON.stringify({ claim, outcome, admissionCalls, requests, credentials, unhandled,
        executions: Object.keys(db.rows.owner_cloud_erasure_executions || {}).length }))
    })().catch(() => { process.exitCode = 1 })
  `
  const { spawnSync } = require('child_process')
  const child = spawnSync(process.execPath, ['--unhandled-rejections=throw', '-e', program, boundary, timing], {
    cwd: require('path').resolve(__dirname, '../..'), env: process.env,
    input: JSON.stringify({ rows: s.db.rows, jobId: s.jobId }), encoding: 'utf8', timeout: 5000,
  })
  expect(child.error).toBeUndefined(); expect(child.status).toBe(0); expect(child.stderr).toBe('')
  expect(JSON.parse(child.stdout)).toEqual({ claim: null, outcome: { status: 'pending', receipt: null },
    admissionCalls: calls, requests: 0, credentials: 0, unhandled: 0, executions: 0 })
})
