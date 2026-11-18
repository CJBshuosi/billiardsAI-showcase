// Test-only loopback bridge: real routes/auth/repositories; controlled SDK only.
const http = require('http')
const crypto = require('crypto')
const Module = require('module')
const fs = require('fs')
const os = require('os')
const path = require('path')
const assert = require('assert')
const { EventEmitter } = require('events')
const { Readable } = require('stream')
const ownerDeviceDelivery = process.argv.includes('--owner-device-delivery')
const ownerErasure = ownerDeviceDelivery || process.argv.includes('--owner-erasure')
const registeredWriters = ownerErasure || process.argv.includes('--registered-writers')
const temporaryRoot = registeredWriters ? fs.mkdtempSync(path.join(os.tmpdir(), 'signed-writers-')) : null
const objects = new Map()
const sdkCalls = []
const objectPuts = []
const erasureBucket = 'synthetic-media-123456'
const erasureRegion = 'ap-guangzhou'
const fileId = key => ownerErasure ? `cloud://synthetic.${erasureBucket}/${key}` : `cloud://synthetic/object-${objects.size + 1}`
// Controlled native metadata refers to the SAME SDK synthetic bytes, never a
// prevalidated absence. No production/native admission evidence is provided.
const nativeCalls = []
const nativeVersions = new Map()
const nativeDeletedKeys = new Set()
let nativeMode = 'NeverEnabled'
let credentialCalls = 0
let erasureOffset = 0
let erasureInstances = 0
let holdNextNative = false
let heldNative = null
let crashAfterDelete = false
let crashCheckpointPending = false
let loseFinalAck = false
let failNextDeviceWrite = null
const erasureClock = () => Date.now() + erasureOffset
function xmlText(value) { return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;') }
function nativeRequest(options, onResponse) {
  assert(ownerErasure)
  assert.equal(activeTransactions, 0, 'native I/O must be outside document callbacks')
  assert.equal(options.hostname, `${erasureBucket}.cos.${erasureRegion}.myqcloud.com`)
  assert.equal(options.protocol, 'https:')
  assert.equal(options.rejectUnauthorized, true)
  assert.equal(options.agent, false)
  assert.match(options.headers.Authorization, /^q-sign-algorithm=sha1&q-ak=controlled-id&/u)
  const url = new URL(options.path, `https://${options.hostname}`)
  const query = Object.fromEntries(url.searchParams)
  const key = url.pathname === '/' ? query.prefix || null : decodeURIComponent(url.pathname.slice(1))
  if (key !== null) assert(objects.has(key) || nativeDeletedKeys.has(key), 'only a known synthetic exact key')
  const probe = options.headers.Range === 'bytes=0-0'
  const call = { method: options.method, path: options.path, key, query, probe, status: null }
  nativeCalls.push(call)
  const outgoing = new EventEmitter()
  let destroyed = false
  outgoing.destroy = () => { destroyed = true }
  outgoing.end = () => queueMicrotask(() => {
    // The controlled server action happens before any held response delivery.
    assert.equal(activeTransactions, 0)
    let status = 200; let body = ''; const headers = {}
    if (Object.hasOwn(query, 'versioning')) {
      assert.equal(key, null); assert.equal(options.method, 'GET')
      body = nativeMode === 'NeverEnabled' ? '<VersioningConfiguration/>'
        : `<VersioningConfiguration><Status>${nativeMode}</Status></VersioningConfiguration>`
    } else if (Object.hasOwn(query, 'versions')) {
      assert.equal(options.method, 'GET'); assert.equal(query['max-keys'], '16')
      assert(!query['key-marker'] && !query['version-id-marker']) // three-version fixture only
      const entries = nativeVersions.get(key) || []
      assert(entries.length || !objects.get(key)?.bytes, 'empty history requires no retained synthetic media')
      assert(entries.length <= 16)
      body = `<ListVersionsResult><Name>${erasureBucket}</Name><Prefix>${xmlText(key)}</Prefix><KeyMarker></KeyMarker><VersionIdMarker></VersionIdMarker><MaxKeys>16</MaxKeys><IsTruncated>false</IsTruncated>`
        + entries.map((entry, index) => {
          const kind = entry.marker ? 'DeleteMarker' : 'Version'
          return `<${kind}><Key>${xmlText(key)}</Key><VersionId>${entry.id}</VersionId><IsLatest>${index === entries.length - 1}</IsLatest><LastModified>2026-11-01T00:00:00.000Z</LastModified></${kind}>`
        }).join('') + '</ListVersionsResult>'
    } else if (options.method === 'DELETE') {
      assert.equal(probe, false)
      if (nativeMode === 'NeverEnabled') {
        assert.deepEqual(query, {}); objects.delete(key)
      } else {
        assert.deepEqual(Object.keys(query), ['versionId'])
        assert(nativeVersions.has(key) || !objects.get(key)?.bytes, 'retained versioned bytes require inventory')
        const remaining = (nativeVersions.get(key) || []).filter(entry => entry.id !== query.versionId)
        if (remaining.length) nativeVersions.set(key, remaining)
        else nativeVersions.delete(key)
        const current = remaining.at(-1)
        if (current) {
          const object = objects.get(key)
          assert(object)
          object.bytes = current.marker ? null : current.bytes
        } else objects.delete(key)
      }
      nativeDeletedKeys.add(key)
      status = 204
      if (crashAfterDelete) { crashAfterDelete = false; crashCheckpointPending = true }
    } else {
      assert.equal(options.method, 'GET'); assert(probe)
      const current = (nativeVersions.get(key) || []).at(-1)
      const object = objects.get(key)
      if (current?.marker || !object || object.bytes === null) {
        if (current?.marker) headers['x-cos-delete-marker'] = 'true'
        status = 404; body = `<Error><Code>NoSuchKey</Code><Resource>/${xmlText(key)}</Resource></Error>`
      } else { status = 206; body = object.bytes.subarray(0, 1) }
    }
    call.status = status
    const bytes = Buffer.isBuffer(body) ? body : Buffer.from(body)
    call.body_sha256 = crypto.createHash('sha256').update(bytes).digest('hex')
    const deliver = () => {
      if (destroyed) return
      const incoming = Readable.from([bytes.subarray(0, 7), bytes.subarray(7)], { objectMode: false })
      incoming.statusCode = status
      incoming.headers = { ...headers, 'content-length': String(bytes.length) }
      incoming.rawHeaders = Object.entries(incoming.headers).flat()
      onResponse(incoming)
    }
    if (holdNextNative) { holdNextNative = false; heldNative = deliver }
    else deliver()
  })
  return outgoing
}
let holdNextUpload = false
let heldUpload = null
let holdNextGrant = false
let heldGrant = null
const objectHost = 'https://synthetic.objects.invalid'
function objectForUrl(url) {
  const parsed = new URL(url)
  assert.equal(parsed.origin, objectHost)
  assert.equal(parsed.search, '')
  assert.equal(parsed.hash, '')
  const object = [...objects.values()].find(value => value.url === url)
  assert(object)
  return object
}
const controlledSdk = {
  downloadFile: () => forbidden(), // legacy full-buffer download is never allowed
  async getUploadMetadata({ cloudPath }) {
    assert.equal(activeTransactions, 0)
    assert(/^managed-clips-v1\/[a-f0-9]{64}\/staging\//u.test(cloudPath))
    assert(!objects.has(cloudPath), 'each SDK grant must use a fresh reserved target')
    const index = objects.size + 1
    const object = { fileID: fileId(cloudPath), cloudPath,
      url: `${objectHost}/__objects/${index}`, bytes: null }
    objects.set(cloudPath, object)
    sdkCalls.push({ kind: 'grant', cloudPath, fileID: object.fileID })
    if (holdNextGrant) {
      holdNextGrant = false
      await new Promise(resolve => { heldGrant = resolve })
    }
    return { data: { fileId: object.fileID, url: object.url,
      token: 'synthetic-token', authorization: 'synthetic-signature', cosFileId: object.fileID } }
  },
  async getTempFileURL({ fileList }) {
    assert.equal(activeTransactions, 0)
    assert.equal(fileList.length, 1)
    const fileID = typeof fileList[0] === 'string' ? fileList[0] : fileList[0].fileID
    const object = [...objects.values()].find(value => value.fileID === fileID)
    assert(object && object.bytes)
    sdkCalls.push({ kind: 'read', fileID })
    return { fileList: [{ fileID, code: 'SUCCESS', tempFileURL: object.url }] }
  },
  async uploadFile({ cloudPath, fileContent }) {
    assert.equal(activeTransactions, 0)
    assert(/^published\/managed-clips-v1\//u.test(cloudPath))
    assert(!objects.has(cloudPath), 'each SDK copy must use a fresh reserved target')
    assert(fileContent instanceof fs.ReadStream, 'actual adapter must upload its pinned fs.ReadStream')
    assert.equal(fileContent.readableHighWaterMark, 65536)
    for (const directory of fs.readdirSync(temporaryRoot)) {
      assert.equal(fs.statSync(path.join(temporaryRoot, directory)).mode & 0o777, 0o700)
    }
    const pinned = fs.fstatSync(fileContent.fd)
    assert(pinned.isFile())
    assert.equal(pinned.mode & 0o777, 0o600)
    assert.equal(pinned.nlink, 0, 'private pathname removed before SDK invocation')
    const hold = holdNextUpload
    holdNextUpload = false
    const chunks = []
    for await (const chunk of fileContent) chunks.push(chunk)
    assert(fileContent.readableEnded)
    const bytes = Buffer.concat(chunks)
    assert.equal(bytes.length, pinned.size)
    const object = { fileID: fileId(cloudPath), cloudPath,
      url: `${objectHost}/__objects/${objects.size + 1}`, bytes }
    objects.set(cloudPath, object)
    sdkCalls.push({ kind: 'copy', cloudPath, fileID: object.fileID,
      sha256: crypto.createHash('sha256').update(bytes).digest('hex') })
    if (hold) await new Promise(resolve => { heldUpload = resolve })
    return { fileID: object.fileID }
  },
}
function openDownload(url) {
  assert.equal(activeTransactions, 0)
  const object = objectForUrl(url)
  assert(object.bytes)
  // Real Readable with IncomingMessage-like status/headers. The adapter still
  // owns spool permissions, EOF/finish, hash, MP4 validation and pinned upload.
  const body = Readable.from([object.bytes.subarray(0, 7), object.bytes.subarray(7)],
    { objectMode: false, highWaterMark: 65536 })
  body.statusCode = 200
  body.headers = { 'content-length': String(object.bytes.length) }
  return body
}
process.once('SIGTERM', () => {
  if (temporaryRoot) fs.rmdirSync(temporaryRoot)
  process.exit(0)
})
const { ownerDeleteDb } = require('./owner-delete-db')
const { documentId, digest } = require('../../cloudfunctions/shared/shot-ownership')
const { createCloudMediaStorageAuthority } = require('../../cloudfunctions/shared/media-storage-authority')
const db = ownerDeleteDb({
  table_join_codes: { 'table-1': { table_id: 'table-1', status: 'active', current_session_id: null,
    code_digest: crypto.createHash('sha256').update('12345678901234567890123456789012').digest('hex') } },
  ranking_tables: { [documentId('table-1')]: { table_id: 'table-1', venue_id: 'venue-1', status: 'approved' } },
  ranking_venues: { [documentId('venue-1')]: { venue_id: 'venue-1', ranking_scope_id: 'zhoukou-urban-dev', status: 'approved' } },
  sync_devices: { 'device-1': { device_id: 'device-1', table_id: 'table-1', status: 'active', secret: 'synthetic-only-secret' } },
}, { strictTransactions: true, queryIds: ownerDeviceDelivery, failWrite(name) {
  if (ownerDeviceDelivery && failNextDeviceWrite === name) {
    failNextDeviceWrite = null
    throw new Error('controlled document write rollback')
  }
  if (ownerErasure && crashCheckpointPending && name === 'owner_cloud_erasure_executions') {
    crashCheckpointPending = false
    throw new Error('controlled invocation crash before checkpoint commit')
  }
} })
let activeTransactions = 0
const runTransaction = db.runTransaction.bind(db)
db.runTransaction = callback => runTransaction(async tx => {
  activeTransactions++
  try { return await callback(tx) } finally { activeTransactions-- }
})
const forbidden = () => { throw new Error('provider I/O is forbidden in this fixture') }
let phoneOpenid = null
// Explicit synthetic development platform configuration, never inherited credentials.
process.env.RANKING_RUNTIME_ENVIRONMENT = 'development'
const originalLoad = Module._load
Module._load = function (name, ...args) {
  if (name === 'wx-server-sdk') return { init() {}, database: () => db, DYNAMIC_CURRENT_ENV: 'synthetic',
    getWXContext: () => ({ OPENID: phoneOpenid }),
    getTempFileURL: registeredWriters ? controlledSdk.getTempFileURL : forbidden, downloadFile: forbidden, uploadFile: forbidden, callFunction: forbidden }
  if (name === '@cloudbase/node-sdk') return { init: () => ({ getUploadMetadata: forbidden, downloadFile: forbidden }), SYMBOL_CURRENT_ENV: 'synthetic' }
  return originalLoad.call(this, name, ...args)
}
const api = require('../../cloudfunctions/sync-api')
const client = require('../../cloudfunctions/client-api')
Module._load = originalLoad
const { createCloudAuthRepositories } = require('../../cloudfunctions/sync-api/lib/auth')
const { createJoinAdmin } = require('../../cloudfunctions/sync-api/lib/join-admin')
const { createCloudDeviceFieldCommandRepository } = require('../../cloudfunctions/sync-api/lib/field-commands')
const { createCloudFieldCommandRepository, submitFieldCommand } = require('../../cloudfunctions/client-api/lib/field-commands')
const { createCloudSyncRepository, createCloudSavedClipRepository } = require('../../cloudfunctions/sync-api/lib/ingest')
const { createMatchFactsRepository } = require('../../cloudfunctions/shared/match-facts-repository')
const { createMatchSetupService } = require('../../cloudfunctions/shared/match-setup-service')
const birth = '22222222-2222-4222-8222-222222222222'
const session = `managed-${birth}`
let birthIndex = 0
// Trusted constructor configuration in this test process only. HTTP/device
// payloads cannot enable provenance, and api.main remains production-default off.
const admin = createJoinAdmin(db, { managedSessionsEnabled: true, managedMediaAdmissionEnabled: true,
  idFactory: () => birthIndex++ === 0 ? birth : crypto.randomUUID() })
const providerCalls = { authorize: 0, verify: 0 }
const pendingDependencies = {
  auth: { ...createCloudAuthRepositories(db), nowSeconds: () => Math.floor(Date.now() / 1000) },
  joinAdmin: admin, fieldCommandRepo: createCloudDeviceFieldCommandRepository(db),
  matchSetupService: createMatchSetupService({
    repository: createMatchFactsRepository(db, { now: () => new Date().toISOString() }),
    now: () => new Date().toISOString(),
  }),
  repo: createCloudSyncRepository(db), savedClipRepo: createCloudSavedClipRepository(db),
  storage: {
    authorizeUpload() { providerCalls.authorize++; return forbidden() },
    verifyObject() { providerCalls.verify++; return forbidden() },
  },
  idFactory: () => crypto.randomUUID(),
  nowIso: () => new Date().toISOString(),
  runtimeAttestation: { schema_version: '1.0', runtime_environment: 'development',
    cloudbase_environment_id: 'synthetic' },
}
// Separate startup-only composition; no product HTTP request changes capability.
// The default intentional missing-port case remains admission-pending.
function runtimeDependencies() {
  if (ownerErasure) erasureInstances++
  return api.createRuntimeDependencies({
    db, storageCloud: controlledSdk,
    environment: { RANKING_RUNTIME_ENVIRONMENT: 'development', CLOUDBASE_ENV_ID: 'synthetic' },
    managedSessionsEnabled: true, managedMediaAdmissionEnabled: true,
    registeredStorageOptions: { openDownload, temporaryRoot },
    ...(ownerErasure ? { clock: erasureClock, ownerCloudErasure: {
      nativeContractAdmission({ scope, assessment }) {
        return { contract: 'owner_erasure_native_admission_v1', scope_sha256: digest(scope),
          inputs_sha256: assessment.inputs_sha256, targets_sha256: digest(assessment.targets),
          environment_id: 'synthetic', bucket: erasureBucket, region: erasureRegion,
          provider_policy: 'cos_exact_target_absence_v1', evidence: {
            binding_sha256: '1'.repeat(64), writer_drain_sha256: '2'.repeat(64), visibility_sha256: '3'.repeat(64),
            configuration_fence_sha256: '4'.repeat(64), stale_request_idempotence_sha256: '5'.repeat(64),
          } }
      },
      providerOptions: { environmentId: 'synthetic', bucket: erasureBucket, region: erasureRegion,
        getCredentials() {
          assert.equal(activeTransactions, 0); credentialCalls++
          return { SecretId: 'controlled-id', SecretKey: 'controlled-secret' }
        }, request: nativeRequest },
    } } : {}),
  })
}
let dependencies = registeredWriters ? runtimeDependencies() : pendingDependencies
const handler = api.createHandler(dependencies)
let issueIndex = 0
let fault = null
const requests = []
async function control(value) {
  if (ownerDeviceDelivery && value.op === 'device-advance') {
    assert(Number.isSafeInteger(value.advanceMs) && value.advanceMs >= 0 && value.advanceMs <= 60000)
    erasureOffset += value.advanceMs
    return null
  }
  if (ownerDeviceDelivery && value.op === 'device-bad-deliveries') {
    const waiting = db.rows.owner_device_deletion_deliveries?.[value.jobId]
    assert(waiting && waiting.status === 'waiting_cloud')
    // Negative discovery metadata only: never creates proof/task/local authority.
    for (let i = 0; i < 5; i++) {
      const key = `bad-delivery-${i}`
      db.rows.owner_device_deletion_deliveries[key] = { ...waiting, job_id: key,
        status: 'ready', environment_id: 'synthetic', due_at_utc: '1970-01-01T00:00:00.000Z',
        diagnostic: 'synthetic-invalid' }
    }
    return null
  }
  if (ownerDeviceDelivery && value.op === 'device-fail-write') {
    assert(['owner_device_deletion_tasks', 'owner_device_deletion_deliveries', 'owner_device_deletion_receipts'].includes(value.collection))
    failNextDeviceWrite = value.collection
    return null
  }
  if (ownerErasure && value.op.startsWith('erase-')) {
    assert.equal(activeTransactions, 0)
    if (value.op === 'erase-once') {
      assert(!ownerDeviceDelivery, 'device composition advances only through signed poll')
      assert(db.rows.owner_clip_deletion_jobs?.[value.jobId], 'actual owner job required')
      return dependencies.ownerCloudErasureWorker.runOnce({ jobId: value.jobId })
    }
    if (value.op === 'erase-restart') {
      assert(Number.isSafeInteger(value.advanceMs || 0) && (value.advanceMs || 0) >= 0)
      erasureOffset += value.advanceMs || 0
      // Fresh stateless invocation/factory; controlled DB/map remains in process.
      dependencies = runtimeDependencies()
      return null
    }
    if (value.op === 'erase-versioning') {
      assert.equal(nativeCalls.length, 0)
      assert(['NeverEnabled', 'Enabled', 'Suspended'].includes(value.mode))
      assert(db.rows.owner_clip_tombstones?.[value.clipId])
      nativeMode = value.mode
      if (nativeMode !== 'NeverEnabled') {
        const inventory = await db.runTransaction(tx => createCloudMediaStorageAuthority(db).snapshotForDeletion(tx, value.clipId))
        for (const target of inventory.objects) {
          const object = objects.get(target.cloud_path)
          assert(object?.bytes)
          nativeVersions.set(target.cloud_path, [
            { id: 'historical', marker: false, bytes: object.bytes },
            { id: 'current', marker: false, bytes: object.bytes },
            { id: 'marker', marker: true, bytes: null },
          ])
          object.bytes = null // controlled marker current404, media remains in versions
        }
      }
      return null
    }
    if (value.op === 'erase-hold-native') { holdNextNative = true; return null }
    if (value.op === 'erase-release-native') {
      assert(heldNative); const release = heldNative; heldNative = null; release(); return null
    }
    if (value.op === 'erase-crash-after-delete') { crashAfterDelete = true; return null }
    if (value.op === 'erase-lose-final-ack') { loseFinalAck = true; return null }
    throw new Error('unknown owner erasure control')
  }
  if (value.op === 'phone') {
    phoneOpenid = Object.hasOwn(value, 'openid') ? value.openid : 'synthetic-phone-a'
    try {
      return await client.main({ schema_version: '1.0', action: value.action, payload: value.payload })
    } finally { phoneOpenid = null }
  }
  if (value.op === 'issue' && !registeredWriters) {
    const id = documentId(`${session}:alice`)
    db.rows.session_user_bindings ||= {}
    db.rows.session_user_bindings[id] ||= { schema_version: '1.0', session_id: session,
      table_id: 'table-1', user_id: 'alice', role: null, request_id: 'join-1' }
    db.rows.session_join_requests ||= {}
    db.rows.session_join_requests['join-1'] ||= { session_id: session, user_id: 'alice', status: 'pending', role: null }
    return submitFieldCommand({ sessionId: session, actorUserId: 'alice', expectedRevision: value.revision || 0,
      commandType: value.type || 'claim_role', payload: value.payload || { role: 'red' },
      idempotencyKey: `bridge-intent-${String(++issueIndex).padStart(16, '0')}`, now: new Date().toISOString(),
    }, createCloudFieldCommandRepository(db))
  }
  if (value.op === 'hold-upload' && registeredWriters) { holdNextUpload = true; return null }
  if (value.op === 'hold-grant' && registeredWriters) { holdNextGrant = true; return null }
  if (value.op === 'release-grant' && registeredWriters) {
    assert(heldGrant); const release = heldGrant; heldGrant = null; release(); return null
  }
  if (value.op === 'release-upload' && registeredWriters) {
    assert(heldUpload); const release = heldUpload; heldUpload = null; release(); return null
  }
  if (value.op === 'storage-snapshot' && registeredWriters) {
    return db.runTransaction(tx => createCloudMediaStorageAuthority(db).snapshotForDeletion(tx, value.clipId))
  }
  if (value.op === 'compact-saved-job' && registeredWriters) {
    // Controlled compaction stimulus only, after actual authenticated deletion.
    assert.equal(activeTransactions, 0)
    const job = db.rows.saved_clip_sync_jobs[value.jobId]
    assert(job)
    assert.equal(db.rows.clip_storage_authorities[job.clip_id].admission, 'closed')
    assert(db.rows.owner_clip_tombstones[job.clip_id])
    delete db.rows.saved_clip_sync_jobs[value.jobId]
    return null
  }
  if (value.op === 'fault') { fault = value.kind; return null }
  if (value.op === 'remove-command') { delete db.rows.field_session_commands[value.id]; return null }
  if (value.op === 'snapshot') return { rows: db.rows, attempts: db.transactionAttempts, requests, providerCalls, sdkCalls, objectPuts,
    ...(ownerErasure ? { ownerErasure: {
      nativeCalls, credentials: credentialCalls, instances: erasureInstances,
      heldNative: Boolean(heldNative), mode: nativeMode,
      versions: Object.fromEntries([...nativeVersions].map(([key, entries]) => [key, entries.map(entry => ({
        id: entry.id, marker: entry.marker, hex: entry.bytes?.toString('hex') || null,
      }))])),
    } } : {}),
    heldGrant: Boolean(heldGrant),
    heldUpload: Boolean(heldUpload), spoolEntries: temporaryRoot ? fs.readdirSync(temporaryRoot) : [],
    objects: Object.fromEntries([...objects].map(([key, value]) => [key, {
      fileID: value.fileID, hex: value.bytes === null ? null : value.bytes.toString('hex'),
    }])) }
  throw new Error('unknown test control operation')
}
const server = http.createServer(async (request, response) => {
  try {
    if (registeredWriters && request.url.startsWith('/__objects/')) {
      const object = objectForUrl(`${objectHost}${request.url}`)
      if (request.method === 'GET') {
        assert(object.bytes)
        response.writeHead(200, { 'content-type': 'video/mp4' }); response.end(object.bytes); return
      }
      assert.equal(request.method, 'PUT')
      assert.equal(request.headers.signature, 'synthetic-signature')
      assert.equal(request.headers.authorization, 'synthetic-signature')
      assert.equal(request.headers['x-cos-security-token'], 'synthetic-token')
      assert.equal(request.headers['x-cos-meta-fileid'], object.fileID)
      assert.equal(request.headers.key, encodeURIComponent(object.cloudPath))
      assert.equal(request.headers['content-type'], 'video/mp4')
      const chunks = []; for await (const chunk of request) chunks.push(chunk)
      object.bytes = Buffer.concat(chunks)
      objectPuts.push({ cloudPath: object.cloudPath, byteSize: object.bytes.length,
        sha256: crypto.createHash('sha256').update(object.bytes).digest('hex') })
      response.writeHead(200); response.end(); return
    }
    let body = ''
    for await (const chunk of request) { body += chunk; if (body.length > 16384) throw new Error('oversized request') }
    if (request.url === '/__test') {
      const value = JSON.parse(body)
      const result = await control(value)
      if (ownerErasure && value.op === 'erase-once' && (value.loseAck || (loseFinalAck && result.status === 'complete'))) {
        if (result.status === 'complete') loseFinalAck = false
        request.socket.destroy(); return
      }
      response.end(JSON.stringify(result)); return
    }
    const event = { httpMethod: request.method, path: request.url.replace(/^\/default/u, ''), headers: request.headers, body }
    const activeFault = typeof fault === 'string' ? { path: '/complete', kind: fault } : fault
    const matchesFault = activeFault && event.path.endsWith(activeFault.path)
    const recorded = { path: event.path, body: JSON.parse(body || '{}'), bodyRaw: body,
      bodySha256: crypto.createHash('sha256').update(body).digest('hex'), status: null, attempts: [] }
    requests.push(recorded)
    if (matchesFault && activeFault.kind === 'request-loss') {
      fault = null; request.socket.destroy(); return
    }
    const start = db.transactionAttempts.length
    const nativeStart = nativeCalls.length
    const result = await (request.url.startsWith('/default/') ? api.main : handler)(event)
    if (ownerDeviceDelivery && event.path === '/v1/owner-media-deletions/claim') {
      recorded.nativeSteps = nativeCalls.length - nativeStart
      assert(recorded.nativeSteps <= 1, 'one signed poll has at most one provider step')
    }
    recorded.status = result.statusCode
    recorded.attempts = db.transactionAttempts.slice(start).map(attempt => attempt.count)
    recorded.calls = db.transactionAttempts.slice(start).map(attempt => attempt.calls)
    recorded.response = JSON.parse(result.body)
    if (matchesFault && activeFault.kind === 'response-loss') {
      fault = null; request.socket.destroy(); return
    }
    if (event.path.endsWith('/close') && activeFault && activeFault.kind === 'malformed-close') {
      fault = null; const payload = JSON.parse(result.body); payload.data.contiguous_ack_sequence = true
      result.body = JSON.stringify(payload)
    }
    response.writeHead(result.statusCode, result.headers); response.end(result.body)
  } catch (error) { response.writeHead(500); response.end(JSON.stringify({ fixture_error: error.message })) }
})
server.listen(0, '127.0.0.1', () => process.stdout.write(`${server.address().port}\n`))
