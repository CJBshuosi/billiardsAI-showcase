const crypto = require('crypto')
const { spawnSync } = require('child_process')
const { EventEmitter } = require('events')
const { Readable } = require('stream')
const { canonicalJson, digest } = require('../../cloudfunctions/shared/shot-ownership')
const device = require('../../cloudfunctions/shared/owner-device-deletion')
const { fixture, provider, drain, admission, NOW, environmentId, bucket, region } = require('../helpers/owner-erasure-fixture')
function rehash(receipt) { const { receipt_sha256, ...body } = receipt; receipt.receipt_sha256 = digest(body) }

test('retained real owner proof issues exact Python task without current native admission', async () => {
  const s = await fixture(); const receipt = await drain(s)
  let admissions = 0
  const throwing = s.repository({ nativeContractAdmission() { admissions++; throw new Error('must not run') } })
  const start = s.db.transactionAttempts.length
  expect(await s.read({}, throwing)).toEqual(receipt)
  expect(await s.read()).toEqual(receipt)
  expect(s.db.transactionAttempts.slice(start).flatMap(a => a.calls).every(c => c.method === 'get')).toBe(true)
  expect(admissions).toBe(0)
  const issued = await s.ensure()
  expect(Object.keys(issued.task)).toHaveLength(19)
  expect(issued.task).toMatchObject({ contract: 'owner_delete_device_v1', deletion_job_id: s.jobId, environment_id: environmentId,
    device_id: 'device-1', table_id: 'table-1', owner_user_id: 'user-a', actor_role: 'red', cloud_erase_receipt_sha256: receipt.receipt_sha256 })
  const python = spawnSync(require('path').resolve(__dirname, '../../../../.venv/bin/python'), ['-c', 'import json,sys; from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask; t=OwnerDeleteDeviceTask.from_payload(json.load(sys.stdin)); print(json.dumps([t.canonical_json(),t.task_sha256]))'], {
    cwd: require('path').resolve(__dirname, '../../../..'), env: { ...process.env, PYTHONPATH: 'src' }, input: JSON.stringify(issued.task), encoding: 'utf8', timeout: 10000,
  })
  expect(python.stderr).toBe(''); expect(python.status).toBe(0)
  expect(JSON.parse(python.stdout)).toEqual([canonicalJson(issued.task), issued.task_sha256])
  // Literal independently produced by the real Python codec; the synthetic cloud
  // receipt binding is fixed so random storage operation IDs cannot mask drift.
  const fixed = device.makeOwnerDeleteDeviceTask({ ...receipt, receipt_sha256: 'f'.repeat(64) })
  const expectedBytes = '{"actor_role":"red","artifact_id":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","byte_size":24,"clip_id":"clip-e6dbede423620ea3909dc819216590005dd6b22536cf3ab5794fb2ec3d104bf9","cloud_erase_receipt_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","content_sha256":"c8c5af84ac765d911a9ab05bc9a19d15d0b1bc5cf0654eff4469ce536410654e","contract":"owner_delete_device_v1","deletion_job_id":"owner-delete-87e2375abcab3db16197143264e1b33a","device_id":"device-1","environment_id":"env-test","owner_user_id":"user-a","ownership_receipt_sha256":"299d9d7211167004991599ee131b2a4de560fb043dbafcc3c25ac80e71448716","postgame_manifest_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","save_receipt_sha256":"117b07b84a854d1d0f8a855cf277125d142ad3951fdaea27ed40b803a06c00ee","save_request_id":"save-request-1","session_id":"managed-12345678-1234-4234-8234-123456789abc","shot_id":"shot-0001","source_job_id":"job-1","table_id":"table-1"}'
  expect(canonicalJson(fixed)).toBe(expectedBytes)
  expect(digest(fixed)).toBe('6c2bdd853eb5ab8d6cd1cd4dbbae5ccc4d09ee9a2edb259c661c7623f78be930')
  const before = structuredClone(s.db.rows.owner_device_deletion_tasks)
  s.now++
  const replayStart = s.db.transactionAttempts.length
  expect(await s.ensure()).toEqual(issued)
  expect(s.db.rows.owner_device_deletion_tasks).toEqual(before)
  expect(s.db.transactionAttempts.slice(replayStart).flatMap(a => a.calls).every(c => c.method === 'get')).toBe(true)
  expect(await s.tasks().ensure({ jobId: s.jobId, ...s.scope, receipt })).toBeNull()
})

test.each(['tableId', 'deviceId', 'environmentId', 'jobId'])('retained proof denies foreign %s', async key => {
  const s = await fixture(); await drain(s)
  expect(await s.read({ [key]: 'foreign' })).toBeNull()
  if (key !== 'environmentId') expect(await s.ensure({ [key]: 'foreign' })).toBeNull()
  else expect(await device.createCloudOwnerDeviceTaskRepository(s.db, { environmentId: 'foreign' }).ensure({ jobId: s.jobId, ...s.scope })).toBeNull()
  expect(Object.keys(s.db.rows.owner_device_deletion_tasks || {})).toHaveLength(0)
})

test.each(['missing-receipt', 'missing-execution', 'incomplete-execution', 'execution-results', 'execution-admission', 'execution-hash', 'receipt-hash', 'receipt-result', 'receipt-native', 'scope', 'birth', 'catalog', 'fingerprint'])('retained proof rejects %s even with a recomputed receipt digest', async change => {
  const s = await fixture(); await drain(s)
  const receipt = s.db.rows.owner_cloud_erasure_receipts[s.jobId]
  const execution = s.db.rows.owner_cloud_erasure_executions[s.jobId]
  if (change === 'missing-receipt') delete s.db.rows.owner_cloud_erasure_receipts[s.jobId]
  if (change === 'missing-execution') delete s.db.rows.owner_cloud_erasure_executions[s.jobId]
  if (change === 'incomplete-execution') { execution.index--; execution.results.pop() }
  if (change === 'execution-results') execution.results[0].evidence.transcript_sha256 = 'f'.repeat(64)
  if (change === 'execution-admission') execution.admission.evidence.writer_drain_sha256 = 'f'.repeat(64)
  if (change === 'execution-hash') execution.assessment_sha256 = 'f'.repeat(64)
  if (change === 'receipt-hash') receipt.scope_sha256 = 'f'.repeat(64)
  if (change === 'receipt-result') receipt.results[0].status = 'pending'
  if (change === 'receipt-native') { receipt.native_admission.evidence.writer_drain_sha256 = null; execution.admission.evidence.writer_drain_sha256 = null }
  if (change === 'scope') s.db.rows.owner_clip_tombstones[s.descriptor.clip_id].owner_user_id = 'foreign'
  if (change === 'birth') s.db.rows.managed_session_births[s.identity.session_id].opened_at = '2026-11-01T09:59:00.000Z'
  if (change === 'catalog') Object.values(s.db.rows.upload_terminal_receipts)[0].entries[0].received_at_utc = '2026-11-01T10:00:01.000Z'
  if (change === 'fingerprint') { receipt.inputs_sha256 = 'f'.repeat(64); execution.inputs_sha256 = receipt.inputs_sha256 }
  rehash(receipt)
  expect(await s.read()).toBeNull(); expect(await s.ensure()).toBeNull()
  expect(Object.keys(s.db.rows.owner_device_deletion_tasks || {})).toHaveLength(0)
})

test('task codec rejects Python-invalid identities, schema and byte bounds', async () => {
  const s = await fixture(); const receipt = await drain(s); const task = device.makeOwnerDeleteDeviceTask(receipt)
  expect(device.validOwnerDeleteDeviceTask(task)).toBe(true)
  for (const patch of [{ byte_size: true }, { byte_size: 0 }, { byte_size: 268435457 }, { artifact_id: 'not-hash' }, { session_id: 'managed-not-a-uuid' }, { clip_id: 'clip-foreign' }, { deletion_job_id: 'owner-delete-foreign' }, { actor_role: 'spectator' }, { shot_id: 'foreign' }, { owner_user_id: '用户' }, { device_id: '..' }, { extra: true }]) {
    expect(device.validOwnerDeleteDeviceTask({ ...task, ...patch })).toBe(false)
    const bad = structuredClone(receipt); Object.assign(bad.scope, patch)
    if (Object.hasOwn(patch, 'deletion_job_id')) bad.deletion_job_id = patch.deletion_job_id
    if (!Object.hasOwn(patch, 'extra')) expect(() => device.makeOwnerDeleteDeviceTask(bad)).toThrow()
  }
  expect(device.validOwnerDeleteDeviceTask({ ...task, byte_size: 268435456 })).toBe(true)
})

test('changed retained receipt conflicts with original immutable task', async () => {
  const s = await fixture(); await drain(s); await s.ensure()
  const original = structuredClone(s.db.rows.owner_device_deletion_tasks)
  const receipt = s.db.rows.owner_cloud_erasure_receipts[s.jobId]; receipt.completed_at = '2026-11-01T10:00:01.000Z'; rehash(receipt)
  s.db.rows.owner_cloud_erasure_witnesses[s.jobId].receipt_sha256 = receipt.receipt_sha256
  await expect(s.ensure()).rejects.toMatchObject({ code: 'OWNER_DEVICE_DELETION_CONFLICT' })
  expect(s.db.rows.owner_device_deletion_tasks).toEqual(original)
})

test.each(['task-hash', 'scope', 'extra', 'timestamp'])('corrupt permanent issuance %s conflicts instead of rewriting', async change => {
  const s = await fixture(); await drain(s); await s.ensure()
  const row = s.db.rows.owner_device_deletion_tasks[s.jobId]
  if (change === 'task-hash') row.task_sha256 = 'f'.repeat(64)
  if (change === 'scope') row.device_id = 'foreign'
  if (change === 'extra') row.delivery_token = crypto.randomUUID()
  if (change === 'timestamp') row.issued_at_utc = 'invalid'
  const before = structuredClone(s.db.rows.owner_device_deletion_tasks)
  await expect(s.ensure()).rejects.toMatchObject({ code: 'OWNER_DEVICE_DELETION_CONFLICT' })
  expect(s.db.rows.owner_device_deletion_tasks).toEqual(before)
})

test.each(['missing', 'incomplete', 'results'])('callback replay denies %s retained execution before task issuance', async change => {
  const s = await fixture(); const receipt = await drain(s)
  const execution = s.db.rows.owner_cloud_erasure_executions[s.jobId]
  if (change === 'missing') delete s.db.rows.owner_cloud_erasure_executions[s.jobId]
  if (change === 'incomplete') { execution.index--; execution.results.pop() }
  if (change === 'results') execution.results[0].evidence.transcript_sha256 = 'f'.repeat(64)
  const repository = s.repository({ onFinalizedReceipt: s.tasks().issueVerified })
  expect(await repository.finalize({ contract: 'owner_cloud_erasure_replay_v1', job_id: s.jobId, receipt })).toBeNull()
  expect(Object.keys(s.db.rows.owner_device_deletion_tasks || {})).toHaveLength(0)
})

test('async trusted callback failure rolls back receipt and task while completed native targets remain recoverable', async () => {
  let failing = true; const s = await fixture(); const tasks = s.tasks(); let callbacks = 0
  const repository = s.repository({ async onFinalizedReceipt(tx, receipt) {
    callbacks++; await tasks.issueVerified(tx, receipt); await Promise.resolve()
    if (failing) throw new Error('controlled callback failure')
  } })
  await expect(drain(s, repository)).rejects.toThrow('controlled callback failure')
  expect(Object.keys(s.db.rows.owner_device_deletion_tasks || {})).toHaveLength(0)
  expect(Object.keys(s.db.rows.owner_cloud_erasure_receipts || {})).toHaveLength(0)
  expect(Object.keys(s.db.rows.owner_cloud_erasure_witnesses || {})).toHaveLength(0)
  const completed = structuredClone(s.db.rows.owner_cloud_erasure_executions[s.jobId])
  expect(completed.index).toBe(completed.targets.length)
  failing = false; s.now += 30001
  const noNative = { step() { throw new Error('completed targets must not run again') } }
  const receipt = await drain(s, repository, noNative)
  expect(receipt.results).toEqual(completed.results)
  expect(await s.ensure()).toMatchObject({ task: { cloud_erase_receipt_sha256: receipt.receipt_sha256 } })
  const before = structuredClone(s.db.rows.owner_device_deletion_tasks)
  await drain(s, repository, noNative)
  expect(callbacks).toBe(3); expect(s.db.rows.owner_device_deletion_tasks).toEqual(before)
})

test('actual factory uses trusted environment and atomically issues on configured completion; defaults remain disabled', async () => {
  const { createRuntimeDependencies } = require('../../cloudfunctions/sync-api'); const s = await fixture()
  let nativeCalls = 0
  const request = (options, respond) => {
    nativeCalls++; const req = new EventEmitter(); req.destroy = () => {}; req.end = () => queueMicrotask(() => {
      const versioning = options.path.includes('versioning')
      const body = versioning ? '<VersioningConfiguration/>' : options.method === 'DELETE' ? '' : `<Error><Code>NoSuchKey</Code><Resource>${options.path}</Resource></Error>`
      const res = Readable.from([Buffer.from(body)]); res.statusCode = versioning ? 200 : options.method === 'DELETE' ? 204 : 404; res.headers = {}; res.rawHeaders = []; respond(res)
    }); return req
  }
  const config = { db: s.db, storageCloud: { getUploadMetadata() {}, downloadFile() {} }, clock: () => s.now, environment: { CLOUDBASE_ENV_ID: environmentId } }
  const ownerCloudErasure = { nativeContractAdmission: admission, providerOptions: { environmentId, bucket, region, request, getCredentials: () => ({ SecretId: 'controlled', SecretKey: 'controlled' }) } }
  expect(createRuntimeDependencies(config).ownerCloudErasureWorker).toBeNull()
  expect(await createRuntimeDependencies(config).ownerDeviceTaskRepo.ensure({ jobId: s.jobId, ...s.scope })).toBeNull()
  expect(createRuntimeDependencies({ ...config, environment: {}, ownerCloudErasure }).ownerCloudErasureWorker).toBeNull()
  expect(createRuntimeDependencies({ ...config, environment: { CLOUDBASE_ENV_ID: 'foreign' }, ownerCloudErasure }).ownerCloudErasureWorker).toBeNull()
  expect(nativeCalls).toBe(0)
  const deps = createRuntimeDependencies({ ...config, ownerCloudErasure })
  let receipt
  for (let i = 0; i < 12; i++) { const result = await deps.ownerCloudErasureWorker.runOnce({ jobId: s.jobId }); if (result.receipt) { receipt = result.receipt; break } }
  expect(receipt).toBeDefined(); expect(nativeCalls).toBe(8)
  expect(s.db.rows.owner_device_deletion_tasks[s.jobId].task.cloud_erase_receipt_sha256).toBe(receipt.receipt_sha256)
  expect(await createRuntimeDependencies(config).ownerDeviceTaskRepo.ensure({ jobId: s.jobId, ...s.scope })).toMatchObject({ task: { deletion_job_id: s.jobId } })
})

test('full64 finalization and issuance with forced retries stay document-only within90 without request lookups', async () => {
  let retry = false; let failCallback = true
  const s = await fixture({ full: true, dbOptions: { retryConflicts: 1, beforeCommit({ attempt }) {
    if (retry && attempt.index === 0) { const error = new Error('controlled conflict'); error.code = 'SYNTHETIC_CONFLICT'; throw error }
  } } })
  const tasks = s.tasks(); const native = provider(s)
  const repository = s.repository({ async onFinalizedReceipt(tx, receipt) {
    await tasks.issueVerified(tx, receipt)
    if (failCallback) throw new Error('controlled full64 callback failure')
  } })
  const start = s.db.transactionAttempts.length; retry = true
  await expect(drain(s, repository, native)).rejects.toThrow('controlled full64 callback failure')
  for (const collection of ['owner_cloud_erasure_receipts', 'owner_cloud_erasure_witnesses', 'owner_device_deletion_tasks']) expect(Object.keys(s.db.rows[collection] || {})).toHaveLength(0)
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId].index).toBe(64)
  failCallback = false; s.now += 30001
  const receipt = await drain(s, repository, native)
  expect(receipt.targets).toHaveLength(64); expect(native.calls).toBe(64)
  await drain(s, repository, { step() { throw new Error('replay must not use native') } })
  await s.ensure()
  const attempts = s.db.transactionAttempts.slice(start)
  const finalization = attempts.filter(a => a.calls.some(c => c.collection === 'owner_cloud_erasure_receipts' && c.method === 'set'))
  expect(finalization).toHaveLength(3); expect(finalization.map(a => a.index)).toEqual([0, 0, 1])
  const witnessBytes = Buffer.byteLength(canonicalJson(s.db.rows.owner_cloud_erasure_witnesses[s.jobId]))
  expect(witnessBytes).toBeLessThanOrEqual(2048)
  for (const a of attempts) {
    expect(a.count).toBeLessThanOrEqual(90)
    expect(a.calls.filter(c => c.collection === 'media_storage_objects' && c.method === 'get')).toHaveLength(64)
    expect(a.calls.filter(c => c.collection === 'upload_terminal_receipts')).toHaveLength(4)
    expect(a.calls.filter(c => c.collection === 'managed_upload_requests')).toHaveLength(0)
    expect(a.calls.filter(c => c.method !== 'get').every(c => ['owner_cloud_erasure_executions', 'owner_cloud_erasure_receipts', 'owner_cloud_erasure_witnesses', 'owner_device_deletion_tasks'].includes(c.collection))).toBe(true)
  }
  console.info('OWNER_DEVICE_ISSUANCE_FULL64', { attempts: attempts.length, min: Math.min(...attempts.map(a => a.count)), max: Math.max(...attempts.map(a => a.count)), retries: attempts.filter(a => a.index === 1).length, finalization: finalization.map(a => a.count), witnessBytes, native: native.calls })
}, 120000)

test('late producer retirement preserves issued task and finished replay without current native ports', async () => {
  const s = await fixture({ optional: 1 }); const tasks = s.tasks()
  const receipt = await drain(s, s.repository({ onFinalizedReceipt: tasks.issueVerified }))
  const issued = await s.ensure(); const before = structuredClone(s.db.rows)
  const original = await s.db.runTransaction(tx => s.authority.readWriterClosureInputs(tx, { clipId: s.descriptor.clip_id }))
  expect((await s.retire(0)).contract).toBe('managed_upload_receipt_ack_v1')
  const current = await s.db.runTransaction(tx => s.authority.readWriterClosureInputs(tx, { clipId: s.descriptor.clip_id }))
  for (const key of ['root', 'operations', 'objects', 'birth']) expect(current[key]).toEqual(original[key])
  expect(current.fingerprint_sha256).not.toBe(original.fingerprint_sha256)
  let admissions = 0
  const repository = s.repository({ nativeContractAdmission() { admissions++; throw new Error('finished proof must not request admission') }, onFinalizedReceipt: tasks.issueVerified })
  const start = s.db.transactionAttempts.length
  expect(await s.read({}, repository)).toEqual(receipt); expect(await s.ensure()).toEqual(issued)
  expect(await drain(s, repository, { step() { throw new Error('finished proof must not request native I/O') } })).toEqual(receipt)
  expect(await s.repository({ nativeContractAdmission: null }).finalize({ contract: 'owner_cloud_erasure_replay_v1', job_id: s.jobId, receipt: null })).toEqual(receipt)
  expect(admissions).toBe(0)
  expect(s.db.transactionAttempts.slice(start).flatMap(a => a.calls).every(c => c.method === 'get')).toBe(true)
  for (const key of ['owner_cloud_erasure_receipts', 'owner_cloud_erasure_witnesses', 'owner_device_deletion_tasks']) expect(s.db.rows[key]).toEqual(before[key])
  const witness = s.db.rows.owner_cloud_erasure_witnesses[s.jobId]
  expect(Object.keys(witness).sort()).toEqual(['birth_sha256', 'contract', 'inputs_sha256', 'job_id', 'receipt_page_sha256', 'receipt_sequences', 'receipt_sha256', 'scope_sha256'])
  expect(Buffer.byteLength(canonicalJson(witness))).toBeLessThanOrEqual(2048)
  expect(witness.receipt_sequences).toEqual([2])
})

test('multiple late retirements across pages preserve original membership including originally null page', async () => {
  const s = await fixture({ optional: 20 }); const receipt = await drain(s)
  const original = await s.ensure(); const witness = structuredClone(s.db.rows.owner_cloud_erasure_witnesses[s.jobId])
  expect(witness.receipt_page_sha256[0]).toBe(digest(null))
  await s.retire(0); await s.retire(16, true); await s.retire(19)
  expect(await s.ensure()).toEqual(original); expect(await s.read()).toEqual(receipt)
  expect(s.db.rows.owner_cloud_erasure_witnesses[s.jobId]).toEqual(witness)
})

test.each(['wrapper', 'receipt', 'binding', 'remove'])('original optional no-write %s mutation still denies against valid current catalog', async change => {
  const s = await fixture({ optional: 2, retired: [0] }); await drain(s)
  const page = Object.values(s.db.rows.upload_terminal_receipts).find(p => p.entries[0])
  const entry = page.entries[0]
  if (change === 'wrapper') entry.received_at_utc = '2026-11-01T10:00:01.000Z'
  if (change === 'receipt') { entry.receipt.evidence.local_binding_sha256 = 'f'.repeat(64); entry.receipt_sha256 = digest(entry.receipt) }
  if (change === 'binding') entry.request_binding.descriptor_sha256 = 'f'.repeat(64)
  if (change === 'remove') page.entries[0] = null
  // Independent current catalog validation succeeds; only the original witness
  // distinguishes these changes from legitimate additions.
  await s.db.runTransaction(tx => s.authority.readWriterClosureInputs(tx, { clipId: s.descriptor.clip_id }))
  expect(await s.read()).toBeNull(); expect(await s.ensure()).toBeNull()
})

test.each(['contract', 'job_id', 'receipt_sha256', 'scope_sha256', 'inputs_sha256', 'birth_sha256', 'receipt_page_sha256', 'membership', 'duplicate', 'missing', 'extra', 'oversize'])('present corrupt witness %s denies without strict fallback', async field => {
  const s = await fixture({ optional: 1 }); await drain(s); const witness = s.db.rows.owner_cloud_erasure_witnesses[s.jobId]
  if (field === 'membership') witness.receipt_sequences = [1, 2]
  else if (field === 'duplicate') witness.receipt_sequences = [2, 2]
  else if (field === 'missing') delete witness.birth_sha256
  else if (field === 'extra') witness.extra = true
  else if (field === 'oversize') witness.receipt_sequences = Array(1000).fill(2)
  else witness[field] = field === 'receipt_page_sha256' ? Array(4).fill('f'.repeat(64)) : 'foreign'
  expect(await s.ensure()).toBeNull(); expect(await s.read()).toBeNull()
  expect(await s.repository({ nativeContractAdmission: null }).claim({ jobId: s.jobId })).toBeNull()
})

test.each(['foreign', 'http200', 'duplicate'])('invalid additional catalog %s cannot be filtered out as harmless', async change => {
  const s = await fixture({ optional: 1 }); await drain(s); await s.retire(0)
  const page = Object.values(s.db.rows.upload_terminal_receipts).find(p => p.entries[0]); const entry = page.entries[0]
  if (change === 'foreign') entry.request_binding.table_id = 'foreign'
  if (change === 'http200') { entry.receipt.evidence = structuredClone(page.entries[1].receipt.evidence); entry.receipt.upload_attempt = s.optional[0].attempt; entry.receipt_sha256 = digest(entry.receipt) }
  if (change === 'duplicate') page.entries[2] = structuredClone(entry)
  expect(await s.ensure()).toBeNull(); expect(await s.read()).toBeNull()
})

test('legacy missing witness uses strict fingerprint and never guesses a backfill', async () => {
  const s = await fixture({ optional: 1 }); const receipt = await drain(s); const issued = await s.ensure()
  delete s.db.rows.owner_cloud_erasure_witnesses[s.jobId]
  expect(await s.ensure()).toEqual(issued)
  expect(await s.repository({ nativeContractAdmission: null }).finalize({ contract: 'owner_cloud_erasure_replay_v1', job_id: s.jobId, receipt })).toEqual(receipt)
  expect(s.db.rows.owner_cloud_erasure_witnesses[s.jobId]).toBeUndefined()
  await s.retire(0)
  expect(await s.ensure()).toBeNull(); expect(await s.read()).toBeNull()
  expect(s.db.rows.owner_cloud_erasure_witnesses[s.jobId]).toBeUndefined()
})

test('pre-finalization delta remains strict and native admission receives only scope and assessment', async () => {
  const s = await fixture({ optional: 1 }); const keys = []
  const repository = s.repository({ nativeContractAdmission(input) { keys.push(Object.keys(input).sort()); return admission(input) } })
  const claim = await repository.claim({ jobId: s.jobId })
  await s.retire(0)
  expect(await repository.loadStep(claim)).toBeNull(); expect(await repository.finalize(claim)).toBeNull()
  expect(s.db.rows.owner_cloud_erasure_receipts[s.jobId]).toBeUndefined()
  expect(Object.keys(s.db.rows.owner_cloud_erasure_witnesses || {})).toHaveLength(0)
  s.now += 30001
  const next = await repository.claim({ jobId: s.jobId }); expect(next.inputs_sha256).not.toBe(claim.inputs_sha256)
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId].results).toEqual([expect.objectContaining({ status: 'excluded' })])
  expect(keys.length).toBeGreaterThan(0); expect(keys.every(k => canonicalJson(k) === '["assessment","scope"]')).toBe(true)
})

test('conflicting preexisting witness aborts first receipt and callback issuance', async () => {
  const s = await fixture(); let callbacks = 0
  s.db.rows.owner_cloud_erasure_witnesses ||= {}
  s.db.rows.owner_cloud_erasure_witnesses[s.jobId] = { contract: 'foreign' }
  const repository = s.repository({ async onFinalizedReceipt(tx, receipt) { callbacks++; return s.tasks().issueVerified(tx, receipt) } })
  await expect(drain(s, repository)).rejects.toMatchObject({ code: 'OWNER_CLOUD_ERASURE_CONFLICT' })
  expect(callbacks).toBe(0); expect(s.db.rows.owner_cloud_erasure_receipts[s.jobId]).toBeUndefined()
  expect(Object.keys(s.db.rows.owner_device_deletion_tasks || {})).toHaveLength(0)
  expect(s.db.rows.owner_cloud_erasure_executions[s.jobId].index).toBe(2)
  expect(s.db.rows.owner_cloud_erasure_witnesses[s.jobId]).toEqual({ contract: 'foreign' })
})
