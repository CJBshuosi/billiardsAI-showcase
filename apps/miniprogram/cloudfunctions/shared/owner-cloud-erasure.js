const crypto = require('crypto')
const { isPromise } = require('util').types
const { canonicalJson, digest, documentId, validOwnership, validSavedSelection } = require('./shot-ownership')
const { createCloudMediaStorageAuthority, validStorageInventory } = require('./media-storage-authority')
const { validManagedBirth, MANAGED_MEDIA_ADMISSION_CONTRACT } = require('./managed-command-authority')
const { readUploadReceiptCatalog, UPLOAD_RECEIPT_CONTRACT } = require('./upload-terminal-receipts')

class OwnerCloudErasureError extends Error {
  constructor(code) { super(code); this.name = 'OwnerCloudErasureError'; this.code = code }
}
function fail(suffix = 'INCOMPLETE') { throw new OwnerCloudErasureError(`OWNER_CLOUD_ERASURE_${suffix}`) }
function exact(value, fields) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && Reflect.ownKeys(value).length === fields.length && fields.every(key => Object.hasOwn(value, key))
}
const dense = (value, limit) => Array.isArray(value) && value.length <= limit
  && Reflect.ownKeys(value).length === value.length + 1
  && Array.from({ length: value.length }, (_, i) => i).every(i => Object.hasOwn(value, i))
function strictSnapshot(value, depth = 0) {
  if (depth > 32) return false
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (!value || typeof value !== 'object') return false
  if (Array.isArray(value)) return dense(value, 64) && value.every(item => strictSnapshot(item, depth + 1))
  if (![Object.prototype, null].includes(Object.getPrototypeOf(value))) return false
  return Reflect.ownKeys(value).every(key => typeof key === 'string' && key !== '_id'
    && Object.getOwnPropertyDescriptor(value, key).enumerable && strictSnapshot(value[key], depth + 1))
}
// Input is an internal authority snapshot, not an authentication credential.
// Revalidate full rows and reconstruct pages; self-supplied digests prove nothing.
function assessApplicationWriters(inputs) {
  if (!exact(inputs, ['root', 'operations', 'objects', 'receipt_entries', 'receipt_page_sha256', 'birth', 'birth_sha256', 'fingerprint_sha256'])) fail('INVALID')
  try {
    if (!strictSnapshot(inputs)) fail()
    const { root, operations, objects, birth } = inputs
    if (!validStorageInventory(root, operations, objects) || Object.hasOwn(root, '_id')
        || root.admission !== 'closed' || root.identity.publication_kind !== 'saved_private'
        || !root.identity.session_id.startsWith('managed-')
        || !validManagedBirth(birth) || Object.hasOwn(birth, '_id')
        || birth.media_admission_contract !== MANAGED_MEDIA_ADMISSION_CONTRACT
        || !['session_id', 'table_id', 'device_id'].every(key => birth[key] === root.identity[key])
        || digest(birth) !== inputs.birth_sha256
        || !dense(inputs.receipt_entries, 64) || !dense(inputs.receipt_page_sha256, 4) || inputs.receipt_page_sha256.length !== 4) fail()
    const pages = Array(4).fill(null); let last = 0
    for (const entry of inputs.receipt_entries) {
      if (!Number.isSafeInteger(entry?.sequence) || entry.sequence <= last || entry.sequence > root.high_water) fail()
      last = entry.sequence; const index = Math.floor((last - 1) / 16)
      pages[index] ||= { contract: UPLOAD_RECEIPT_CONTRACT, clip_id: root.identity.clip_id, identity_sha256: root.identity_sha256,
        epoch: root.epoch, page_index: index, entries: Array(16).fill(null) }
      pages[index].entries[(last - 1) % 16] = entry
    }
    const catalog = readUploadReceiptCatalog({ root, operations, pages })
    if (canonicalJson(catalog.page_sha256) !== canonicalJson(inputs.receipt_page_sha256)) fail()
    const fingerprint = digest({ root, operations, objects, receipt_page_sha256: catalog.page_sha256, birth_sha256: digest(birth) })
    if (fingerprint !== inputs.fingerprint_sha256) fail()
    const receipts = new Map(catalog.entries.map(entry => [entry.sequence, entry])); const blockers = []
    const targets = operations.map((op, i) => {
      const entry = receipts.get(op.sequence); let writer_state = 'unresolved'; let receipt_sha256 = null
      if (op.outcome === 'no_write' && op.reason === 'not_started' && op.objects.length === 0) writer_state = 'no_write'
      else if (op.outcome === 'server_succeeded') writer_state = 'copy_succeeded'
      else if (op.outcome === 'external_issued' && entry) {
        writer_state = entry.receipt.evidence.kind === 'http200' ? 'upload_succeeded' : 'upload_retired'
        receipt_sha256 = entry.receipt_sha256
      }
      if (writer_state === 'unresolved') blockers.push({ sequence: op.sequence, code: op.outcome === 'external_issued' ? 'upload_receipt_missing' : 'writer_unresolved' })
      return { sequence: op.sequence, operation_id: op.operation_id, object_id: op.target.object_id, kind: op.target.kind,
        cloud_path: op.target.cloud_path, file_id: objects[i].file_id, writer_state, receipt_sha256 }
    })
    return JSON.parse(canonicalJson({ contract: 'owner_application_writers_v1', identity: root.identity, identity_sha256: root.identity_sha256,
      epoch: root.epoch, high_water: root.high_water, authority_revision: root.revision, inputs_sha256: fingerprint,
      status: blockers.length ? 'blocked' : 'closed', targets, blockers, erasable: false }))
  } catch (error) {
    if (error instanceof OwnerCloudErasureError) throw error
    fail()
  }
}


const HASH = /^[a-f0-9]{64}$/u
const ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,128}$/u
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/u
const SCOPE_FIELDS = ['schema_version', 'clip_id', 'owner_user_id', 'actor_role', 'session_id', 'shot_id', 'device_id', 'table_id', 'artifact_id', 'source_job_id', 'content_sha256', 'byte_size', 'ownership_receipt', 'ownership_receipt_sha256', 'save_request_id', 'save_receipt_sha256', 'postgame_manifest_sha256', 'storage_objects', 'storage_authority', 'playback_authorized_until', 'requested_at', 'receipt']
const ADMISSION_FIELDS = ['contract', 'scope_sha256', 'inputs_sha256', 'targets_sha256', 'environment_id', 'bucket', 'region', 'provider_policy', 'evidence']
const NATIVE_EVIDENCE = ['binding_sha256', 'writer_drain_sha256', 'visibility_sha256', 'configuration_fence_sha256', 'stale_request_idempotence_sha256']
const EXECUTION_FIELDS = ['contract', 'job_id', 'scope_sha256', 'inputs_sha256', 'assessment_sha256', 'targets', 'admission', 'token', 'lease_until', 'index', 'checkpoint', 'results', 'revision']
const WITNESS_FIELDS = ['contract', 'job_id', 'receipt_sha256', 'scope_sha256', 'inputs_sha256', 'receipt_sequences', 'receipt_page_sha256', 'birth_sha256']
const CLAIM_FIELDS = ['contract', 'job_id', 'token', 'lease_until', 'scope_sha256', 'inputs_sha256', 'admission_sha256', 'revision']
const CHECKPOINT_FIELDS = ['contract', 'target_sha256', 'mode', 'phase', 'cursor', 'cursor_sha256', 'pages', 'versions', 'transcript_sha256']
const MODES = ['NeverEnabled', 'Enabled', 'Suspended']
const json = value => JSON.parse(canonicalJson(value))
const hash = value => typeof value === 'string' && HASH.test(value)
const id = value => typeof value === 'string' && ID.test(value)
const uuid = value => typeof value === 'string' && UUID.test(value)
function record(value, fields) {
  return Boolean(value && [Object.prototype, null].includes(Object.getPrototypeOf(value)) && exact(value, fields)
    && fields.every(key => { const d = Object.getOwnPropertyDescriptor(value, key); return d.enumerable && Object.hasOwn(d, 'value') }))
}
function safeJson(value, depth = 0) {
  if (depth > 24) return false
  if (value === null || typeof value === 'boolean') return true
  if (typeof value === 'number') return Number.isSafeInteger(value)
  if (typeof value === 'string') return Buffer.byteLength(value) <= 8192 && Buffer.from(value).toString() === value
  if (!value || typeof value !== 'object') return false
  const keys = Reflect.ownKeys(value)
  if (Array.isArray(value)) {
    if (!dense(value, 128)) return false
    return keys.filter(k => k !== 'length').every(k => Object.hasOwn(Object.getOwnPropertyDescriptor(value, k), 'value') && safeJson(value[k], depth + 1))
  }
  return [Object.prototype, null].includes(Object.getPrototypeOf(value)) && keys.length <= 32 && keys.every(k => typeof k === 'string'
    && Object.hasOwn(Object.getOwnPropertyDescriptor(value, k), 'value') && Object.getOwnPropertyDescriptor(value, k).enumerable && safeJson(value[k], depth + 1))
}
const bounded = (value, bytes = 128 * 1024) => safeJson(value) && Buffer.byteLength(canonicalJson(value)) <= bytes
const time = value => typeof value === 'string' && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/u.test(value) && Number.isFinite(Date.parse(value))
async function optional(ref) {
  try { const value = (await ref.get()).data ?? null; return value === null ? null : Object.fromEntries(Object.entries(value).filter(([k]) => k !== '_id')) } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error?.message || '')) return null
    throw error
  }
}
function validScope(scope, job, jobId, inputs, assessment) {
  if (!record(scope, SCOPE_FIELDS) || !bounded(scope) || !record(job, [...SCOPE_FIELDS, 'job_id', 'deletion_job_id', 'job_kind', 'status'])
      || job.job_id !== jobId || job.deletion_job_id !== jobId || job.job_kind !== 'owner_delete_v1' || job.status !== 'pending_cloud'
      || jobId !== `owner-delete-${documentId(scope.clip_id).slice(0, 32)}`
      || canonicalJson(scope) !== canonicalJson(Object.fromEntries(SCOPE_FIELDS.map(k => [k, job[k]])))
      || scope.schema_version !== '1.0' || !validOwnership(scope) || !validSavedSelection(scope)
      || !['clip_id', 'session_id', 'shot_id', 'owner_user_id', 'device_id', 'table_id', 'artifact_id', 'source_job_id'].every(k => id(scope[k]))
      || !hash(scope.content_sha256) || !Number.isSafeInteger(scope.byte_size) || scope.byte_size < 1
      || !time(scope.requested_at) || !(scope.playback_authorized_until === null || time(scope.playback_authorized_until))) return false
  const receipt = scope.receipt; const root = inputs.root; const identity = root.identity; const summary = scope.storage_authority
  if (!record(receipt, ['schema_version', 'clip_id', 'request_id', 'deletion_job_id', 'status', 'requested_at'])
      || receipt.schema_version !== '1.0' || receipt.clip_id !== scope.clip_id || !id(receipt.request_id)
      || receipt.deletion_job_id !== jobId || receipt.status !== 'deletion_requested' || receipt.requested_at !== scope.requested_at
      || !record(summary, ['status', 'contract', 'identity_sha256', 'epoch', 'high_water', 'reasons'])
      || summary.status !== 'tracked_closed' || summary.contract !== root.contract || summary.identity_sha256 !== root.identity_sha256
      || summary.epoch !== root.epoch || summary.high_water !== root.high_water || !dense(summary.reasons, 0)
      || identity.publication_kind !== 'saved_private' || identity.source_kind !== 'usb_live' || identity.content_type !== 'video/mp4'
      || !Object.keys(identity).filter(k => !['publication_kind', 'content_type', 'source_kind'].includes(k)).every(k => identity[k] === scope[k])
      || assessment.status !== 'closed' || !dense(scope.storage_objects, 64) || !scope.storage_objects.length) return false
  const refs = scope.storage_objects
  if (!refs.every(r => record(r, ['kind', 'file_id']) && ['staging', 'published'].includes(r.kind) && typeof r.file_id === 'string')
      || new Set(refs.map(r => r.file_id)).size !== refs.length || !refs.some(r => r.kind === 'published')) return false
  const published = inputs.objects.filter(o => o.references.length).map(o => ({ kind: o.kind, file_id: o.file_id }))
  return refs.length === published.length && refs.every(r => published.some(p => canonicalJson(p) === canonicalJson(r)))
}
// This record asserts reviewed *operational* evidence, not authentication. The
// configuration fence must cover every outstanding exact request, including old
// NeverEnabled DELETEs that could create markers after a versioning change.
// Lease expiry/abort is never evidence that those requests stopped server-side.
function validNativeAdmission(value, scope, assessment) {
  return record(value, ADMISSION_FIELDS) && bounded(value, 4096) && value.contract === 'owner_erasure_native_admission_v1'
    && value.scope_sha256 === digest(scope) && value.inputs_sha256 === assessment.inputs_sha256 && value.targets_sha256 === digest(assessment.targets)
    && typeof value.environment_id === 'string' && /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/u.test(value.environment_id)
    && typeof value.bucket === 'string' && value.bucket.length <= 128 && /^[a-z0-9][a-z0-9-]{1,62}-[0-9]+$/u.test(value.bucket)
    && typeof value.region === 'string' && value.region.length <= 64 && /^[a-z]{2,8}-[a-z]+(?:-[a-z]+)?[0-9]?$/u.test(value.region)
    && value.provider_policy === 'cos_exact_target_absence_v1' && record(value.evidence, NATIVE_EVIDENCE) && NATIVE_EVIDENCE.every(k => hash(value.evidence[k]))
    && assessment.targets.every(t => t.writer_state === 'no_write' ? t.file_id === null : t.file_id === `cloud://${value.environment_id}.${value.bucket}/${t.cloud_path}`)
}
function validCheckpoint(cp, target) {
  if (!record(cp, CHECKPOINT_FIELDS) || !bounded(cp, 48 * 1024) || cp.contract !== 'owner_erasure_provider_v1'
      || cp.target_sha256 !== digest(target) || !hash(cp.transcript_sha256) || !(cp.mode === null || MODES.includes(cp.mode))
      || !['versioning', 'simple_delete', 'simple_probe', 'list', 'delete', 'version_probe', 'recheck'].includes(cp.phase)
      || !Number.isSafeInteger(cp.pages) || cp.pages < 0 || cp.pages > 128 || !dense(cp.cursor_sha256, 128)
      || cp.cursor_sha256.length !== cp.pages || !cp.cursor_sha256.every(hash) || new Set(cp.cursor_sha256).size !== cp.pages
      || !dense(cp.versions, 16) || !cp.versions.every(v => typeof v === 'string' && v.length > 0 && Buffer.byteLength(v) <= 1024 && !/[\u0000-\u001f\u007f]/u.test(v))
      || new Set(cp.versions).size !== cp.versions.length) return false
  if (cp.cursor !== null && (!record(cp.cursor, ['key', 'version']) || typeof cp.cursor.key !== 'string' || !cp.cursor.key.startsWith(target.cloud_path)
      || Buffer.byteLength(cp.cursor.key) > 1024 || typeof cp.cursor.version !== 'string' || Buffer.byteLength(cp.cursor.version) > 1024
      || /[\u0000-\u001f\u007f]/u.test(cp.cursor.key + cp.cursor.version) || cp.cursor_sha256.at(-1) !== digest(cp.cursor))) return false
  if ((cp.cursor === null) !== (cp.pages === 0) || (cp.phase === 'delete') !== (cp.versions.length > 0)) return false
  if (cp.phase === 'versioning') return cp.mode === null && cp.pages === 0
  return cp.mode !== null && (!['simple_delete', 'simple_probe'].includes(cp.phase) || cp.mode === 'NeverEnabled')
    && (!['list', 'delete', 'version_probe'].includes(cp.phase) || cp.mode !== 'NeverEnabled')
    && (!['simple_delete', 'simple_probe', 'recheck'].includes(cp.phase) || cp.pages === 0)
}
function validEvidence(e, target, absent = false) {
  return record(e, ['target_sha256', 'versioning_mode', 'observed_at', 'transcript_sha256', ...(Object.hasOwn(e || {}, 'code') ? ['code'] : [])])
    && e.target_sha256 === digest(target) && (e.versioning_mode === null || MODES.includes(e.versioning_mode))
    && time(e.observed_at) && hash(e.transcript_sha256) && (!Object.hasOwn(e, 'code') || (!absent && /^OWNER_ERASURE_(CREDENTIALS|TRANSPORT|TIMEOUT|BOUNDS|XML|NATIVE|CURSOR|PRESENT|FENCE)$/u.test(e.code)))
    && (!absent || e.versioning_mode !== null)
}
function excluded(target, fingerprint) { return { target_sha256: digest(target), status: 'excluded', evidence: { contract: 'owner_never_started_v1', target_sha256: digest(target), inputs_sha256: fingerprint } } }
function skipNeverStarted(execution) {
  while (execution.targets[execution.index]?.writer_state === 'no_write') {
    execution.results.push(excluded(execution.targets[execution.index], execution.inputs_sha256)); execution.index++
  }
}
function validExecution(e, context, jobId) {
  if (!record(e, EXECUTION_FIELDS) || !bounded(e, 384 * 1024) || e.contract !== 'owner_cloud_erasure_execution_v1' || e.job_id !== jobId
      || e.scope_sha256 !== digest(context.scope) || e.inputs_sha256 !== context.assessment.inputs_sha256 || e.assessment_sha256 !== digest(context.assessment)
      || canonicalJson(e.targets) !== canonicalJson(context.assessment.targets) || canonicalJson(e.admission) !== canonicalJson(context.admission)
      || !(e.token === null || uuid(e.token)) || !Number.isSafeInteger(e.lease_until) || !Number.isSafeInteger(e.revision) || e.revision < 1
      || !Number.isSafeInteger(e.index) || e.index < 0 || e.index > e.targets.length || !dense(e.results, 64) || e.results.length !== e.index
      || !(e.checkpoint === null || (e.index < e.targets.length && validCheckpoint(e.checkpoint, e.targets[e.index])))) return false
  return e.results.every((r, i) => record(r, ['target_sha256', 'status', 'evidence']) && r.target_sha256 === digest(e.targets[i])
    && (e.targets[i].writer_state === 'no_write' ? canonicalJson(r) === canonicalJson(excluded(e.targets[i], e.inputs_sha256)) : r.status === 'absent' && validEvidence(r.evidence, e.targets[i], true)))
}
function makeClaim(e) {
  return { contract: 'owner_cloud_erasure_claim_v1', job_id: e.job_id, token: e.token, lease_until: e.lease_until,
    scope_sha256: e.scope_sha256, inputs_sha256: e.inputs_sha256, admission_sha256: digest(e.admission), revision: e.revision }
}
function matches(claim, e, now, requireRevision = true) {
  return record(claim, CLAIM_FIELDS) && bounded(claim, 2048) && claim.contract === 'owner_cloud_erasure_claim_v1'
    && claim.job_id === e.job_id && uuid(claim.token) && claim.token === e.token && claim.lease_until === e.lease_until && now < e.lease_until
    && claim.scope_sha256 === e.scope_sha256 && claim.inputs_sha256 === e.inputs_sha256 && claim.admission_sha256 === digest(e.admission)
    && (requireRevision ? claim.revision === e.revision : [e.revision, e.revision - 1].includes(claim.revision))
}
// Unsupported Promise results are denied synchronously. Register only a
// rejection sink: never await/accept asynchronous native admission in a document
// transaction. Both the factory wrapper and direct repository use this boundary.
function evaluateSynchronousNativeAdmission(admission, input) {
  try {
    const value = admission(input)
    if (isPromise(value)) {
      Promise.prototype.then.call(value, undefined, () => {})
      return null
    }
    return value
  } catch { return null }
}
function createOwnerCloudErasureRepository(db, { authority = createCloudMediaStorageAuthority(db), clock = Date.now, idFactory = crypto.randomUUID, nativeContractAdmission = null, onFinalizedReceipt = null } = {}) {
  if (onFinalizedReceipt !== null && typeof onFinalizedReceipt !== 'function') fail('INVALID')
  const executionRef = (tx, jobId) => tx.collection('owner_cloud_erasure_executions').doc(jobId)
  const receiptRef = (tx, jobId) => tx.collection('owner_cloud_erasure_receipts').doc(jobId)
  const witnessRef = (tx, jobId) => tx.collection('owner_cloud_erasure_witnesses').doc(jobId)
  function now() { const value = clock(); if (!Number.isSafeInteger(value) || value < 0 || value > 8640000000000000) fail('INVALID'); return value }
  async function storedContext(tx, jobId) {
    if (!id(jobId)) return null
    const job = await optional(tx.collection('owner_clip_deletion_jobs').doc(jobId))
    if (!job || !id(job.clip_id)) return null
    const scope = await optional(tx.collection('owner_clip_tombstones').doc(job.clip_id))
    try {
      const inputs = await authority.readWriterClosureInputs(tx, { clipId: job.clip_id }); const assessment = assessApplicationWriters(inputs)
      if (!validScope(scope, job, jobId, inputs, assessment)) return null
      return { scope, assessment, inputs }
    } catch (error) {
      if (error instanceof OwnerCloudErasureError || error?.name === 'MediaStorageAuthorityError') return null
      throw error
    }
  }
  async function context(tx, jobId) {
    if (typeof nativeContractAdmission !== 'function') return null
    const c = await storedContext(tx, jobId); if (!c) return null
    const admission = evaluateSynchronousNativeAdmission(nativeContractAdmission, json({ scope: c.scope, assessment: c.assessment }))
    return validNativeAdmission(admission, c.scope, c.assessment) ? { ...c, admission: json(admission) } : null
  }
  async function current(tx, claim) {
    if (!record(claim, CLAIM_FIELDS) || !id(claim.job_id)) return null
    const c = await context(tx, claim.job_id); if (!c) return null
    const e = await optional(executionRef(tx, claim.job_id))
    return e && validExecution(e, c, claim.job_id) ? { ...c, e } : null
  }
  function validReceipt(receipt, c, jobId) {
    if (!record(receipt, ['contract', 'deletion_job_id', 'environment_id', 'scope', 'scope_sha256', 'inputs_sha256', 'assessment_sha256', 'authority_epoch', 'targets', 'targets_sha256', 'results', 'native_admission', 'completed_at', 'receipt_sha256']) || !bounded(receipt, 384 * 1024)) return false
    const { receipt_sha256, ...body } = receipt
    return receipt.contract === 'owner_cloud_erase_receipt_v1' && receipt.deletion_job_id === jobId && hash(receipt_sha256) && digest(body) === receipt_sha256
      && canonicalJson(receipt.scope) === canonicalJson(c.scope) && receipt.scope_sha256 === digest(c.scope) && receipt.inputs_sha256 === c.assessment.inputs_sha256
      && receipt.assessment_sha256 === digest(c.assessment) && receipt.authority_epoch === c.assessment.epoch
      && receipt.environment_id === c.admission.environment_id && canonicalJson(receipt.native_admission) === canonicalJson(c.admission)
      && canonicalJson(receipt.targets) === canonicalJson(c.assessment.targets) && receipt.targets_sha256 === digest(receipt.targets) && time(receipt.completed_at)
      && validExecution({ contract: 'owner_cloud_erasure_execution_v1', job_id: jobId, scope_sha256: receipt.scope_sha256, inputs_sha256: receipt.inputs_sha256,
        assessment_sha256: receipt.assessment_sha256, targets: receipt.targets, admission: receipt.native_admission, token: null, lease_until: 0,
        index: receipt.targets.length, checkpoint: null, results: receipt.results, revision: 1 }, c, jobId)
  }
  // Trusted async document-only extension. Throwing aborts the same transaction;
  // completed native evidence remains in the earlier progress transaction.
  function completedExecution(execution, c, receipt, jobId) {
    return execution && validExecution(execution, c, jobId)
      && execution.index === execution.targets.length && execution.checkpoint === null
      && canonicalJson(execution.results) === canonicalJson(receipt.results)
  }
  function validWitness(witness, receipt, c, jobId) {
    return record(witness, WITNESS_FIELDS) && bounded(witness, 2048)
      && witness.contract === 'owner_erasure_closure_witness_v1' && witness.job_id === jobId
      && witness.receipt_sha256 === receipt.receipt_sha256 && witness.scope_sha256 === receipt.scope_sha256
      && witness.inputs_sha256 === receipt.inputs_sha256
      && ['receipt_sha256', 'scope_sha256', 'inputs_sha256', 'birth_sha256'].every(key => hash(witness[key]))
      && witness.birth_sha256 === c.inputs.birth_sha256
      && dense(witness.receipt_page_sha256, 4) && witness.receipt_page_sha256.length === 4 && witness.receipt_page_sha256.every(hash)
      && dense(witness.receipt_sequences, 64) && witness.receipt_sequences.every((sequence, i, values) =>
        Number.isSafeInteger(sequence) && sequence > (i ? values[i - 1] : 0) && sequence <= c.inputs.root.high_water)
  }
  function originalContext(c, receipt, witness, jobId) {
    if (!validWitness(witness, receipt, c, jobId)) return null
    const membership = new Set(witness.receipt_sequences)
    const entries = c.inputs.receipt_entries.filter(entry => membership.has(entry.sequence))
    if (entries.length !== membership.size) return null
    for (const entry of c.inputs.receipt_entries) {
      if (membership.has(entry.sequence)) continue
      const op = c.inputs.operations[entry.sequence - 1]
      if (op.kind !== 'upload_authorization' || op.outcome !== 'no_write' || op.reason !== 'not_started'
          || op.objects.length !== 0 || entry.receipt.evidence.kind !== 'retired_before_start') return null
    }
    const inputs = { ...c.inputs, receipt_entries: entries, receipt_page_sha256: witness.receipt_page_sha256 }
    inputs.fingerprint_sha256 = digest({ root: inputs.root, operations: inputs.operations, objects: inputs.objects,
      receipt_page_sha256: inputs.receipt_page_sha256, birth_sha256: inputs.birth_sha256 })
    if (inputs.fingerprint_sha256 !== receipt.inputs_sha256) return null
    try {
      // This reconstructs and verifies all four ORIGINAL pages, including every
      // original optional entry's full binding and server wrapper metadata.
      const assessment = assessApplicationWriters(inputs)
      return canonicalJson(assessment.targets) === canonicalJson(c.assessment.targets) ? { ...c, inputs, assessment } : null
    } catch (error) {
      if (error instanceof OwnerCloudErasureError) return null
      throw error
    }
  }
  async function retainWitness(tx, receipt, c) {
    const witness = { contract: 'owner_erasure_closure_witness_v1', job_id: receipt.deletion_job_id,
      receipt_sha256: receipt.receipt_sha256, scope_sha256: receipt.scope_sha256, inputs_sha256: receipt.inputs_sha256,
      receipt_sequences: c.inputs.receipt_entries.map(entry => entry.sequence),
      receipt_page_sha256: c.inputs.receipt_page_sha256, birth_sha256: c.inputs.birth_sha256 }
    if (!validWitness(witness, receipt, c, receipt.deletion_job_id)) fail('INVALID')
    const ref = witnessRef(tx, receipt.deletion_job_id); const old = await optional(ref)
    if (old !== null) { if (!validWitness(old, receipt, c, receipt.deletion_job_id) || canonicalJson(old) !== canonicalJson(witness)) fail('CONFLICT') }
    else await ref.set({ data: witness })
  }
  // Finished proof is metadata-only, even when entered through worker replay.
  // Validate the complete current catalog BEFORE selecting original membership.
  async function retained(tx, jobId, storedReceipt) {
    const receipt = storedReceipt === undefined ? await optional(receiptRef(tx, jobId)) : storedReceipt
    if (!receipt || !bounded(receipt, 384 * 1024)) return null
    const current = await storedContext(tx, jobId); if (!current) return null
    const witness = await optional(witnessRef(tx, jobId))
    // Legacy absence retains exact current-fingerprint semantics. A present
    // corrupt witness must never fall back or be reconstructed from today's rows.
    const original = witness === null ? current : originalContext(current, receipt, witness, jobId)
    if (!original || !validNativeAdmission(receipt.native_admission, original.scope, original.assessment)) return null
    const c = { ...original, admission: receipt.native_admission }
    if (!validReceipt(receipt, c, jobId)) return null
    const e = await optional(executionRef(tx, jobId))
    return completedExecution(e, c, receipt, jobId) ? { ...c, e, receipt } : null
  }
  async function finalized(tx, receipt, c, execution) {
    if (onFinalizedReceipt) {
      const retained = execution === undefined ? await optional(executionRef(tx, receipt.deletion_job_id)) : execution
      if (!completedExecution(retained, c, receipt, receipt.deletion_job_id)) return null
      await onFinalizedReceipt(tx, json(receipt))
    }
    return receipt
  }
  return {
    // Read exact server IDs and retained context only. This grants no claim and
    // neither invokes current admission nor performs provider work.
    async readRetainedReceipt(tx, args) {
      if (!record(args, ['jobId', 'tableId', 'deviceId', 'environmentId']) || !Object.values(args).every(id)) return null
      const { jobId, tableId, deviceId, environmentId } = args
      const c = await retained(tx, jobId)
      return c && c.scope.table_id === tableId && c.scope.device_id === deviceId && c.receipt.environment_id === environmentId ? json(c.receipt) : null
    },
    async claim({ jobId, leaseSeconds = 30 } = {}) {
      if (!id(jobId) || !Number.isSafeInteger(leaseSeconds) || leaseSeconds < 15 || leaseSeconds > 120) return null
      return db.runTransaction(async tx => {
        const receipt = await optional(receiptRef(tx, jobId))
        if (receipt !== null) return await retained(tx, jobId, receipt) ? { contract: 'owner_cloud_erasure_replay_v1', job_id: jobId, receipt } : null
        const c = await context(tx, jobId); if (!c) return null
        const old = await optional(executionRef(tx, jobId)); const observed = now()
        if (old !== null && (!bounded(old, 384 * 1024) || old.scope_sha256 !== digest(c.scope) || old.job_id !== jobId || !Number.isSafeInteger(old.lease_until) || old.lease_until > observed)) return null
        let e
        if (old !== null && validExecution(old, c, jobId)) e = old
        else {
          // Reassessment after expiry may change facts, never reserved scope/key
          // or provider configuration. An expired lease does not drain old I/O.
          if (old !== null && (canonicalJson(old.targets?.map(t => [t.sequence, t.operation_id, t.object_id, t.kind, t.cloud_path])) !== canonicalJson(c.assessment.targets.map(t => [t.sequence, t.operation_id, t.object_id, t.kind, t.cloud_path]))
              || !['environment_id', 'bucket', 'region', 'provider_policy'].every(k => old.admission?.[k] === c.admission[k]))) return null
          e = { contract: 'owner_cloud_erasure_execution_v1', job_id: jobId, scope_sha256: digest(c.scope), inputs_sha256: c.assessment.inputs_sha256,
            assessment_sha256: digest(c.assessment), targets: c.assessment.targets, admission: c.admission, token: null, lease_until: 0,
            index: 0, checkpoint: null, results: [], revision: 0 }
        }
        e.token = idFactory(); if (!uuid(e.token)) fail('INVALID')
        e.lease_until = observed + leaseSeconds * 1000; e.revision++; skipNeverStarted(e)
        await executionRef(tx, jobId).set({ data: e }); return makeClaim(e)
      })
    },
    async loadStep(claim) {
      return db.runTransaction(async tx => {
        const c = await current(tx, claim)
        if (!c || !matches(claim, c.e, now()) || c.e.index === c.e.targets.length) return null
        return json({ target: c.e.targets[c.e.index], checkpoint: c.e.checkpoint })
      })
    },
    async recordStep(claim, result) {
      return db.runTransaction(async tx => {
        const c = await current(tx, claim); if (!c || !matches(claim, c.e, now())) return false
        const e = c.e; const target = e.targets[e.index]
        if (!target || !record(result, ['status', 'checkpoint', 'evidence']) || !bounded(result, 52 * 1024)
            || !['pending', 'absent'].includes(result.status) || !validCheckpoint(result.checkpoint, target) || !validEvidence(result.evidence, target, result.status === 'absent')
            || result.evidence.transcript_sha256 !== result.checkpoint.transcript_sha256 || result.evidence.versioning_mode !== result.checkpoint.mode
            || (result.status === 'absent' && result.checkpoint.phase !== 'recheck')) return false
        if (result.status === 'absent') { e.results.push({ target_sha256: digest(target), status: 'absent', evidence: json(result.evidence) }); e.index++; e.checkpoint = null; skipNeverStarted(e) }
        else e.checkpoint = json(result.checkpoint)
        e.revision++; await executionRef(tx, claim.job_id).set({ data: e }); return true
      })
    },
    async finalize(claim) {
      if (record(claim, ['contract', 'job_id', 'receipt']) && claim.contract === 'owner_cloud_erasure_replay_v1') {
        if (!id(claim.job_id)) return null
        return db.runTransaction(async tx => { const c = await retained(tx, claim.job_id); return c ? finalized(tx, c.receipt, c, c.e) : null })
      }
      if (!record(claim, CLAIM_FIELDS) || !id(claim.job_id)) return null
      return db.runTransaction(async tx => {
        const old = await optional(receiptRef(tx, claim.job_id))
        if (old !== null) {
          const c = await retained(tx, claim.job_id, old)
          return c && claim.scope_sha256 === old.scope_sha256 && claim.inputs_sha256 === old.inputs_sha256 ? finalized(tx, old, c, c.e) : null
        }
        const c = await current(tx, claim); if (!c) return null
        if (!matches(claim, c.e, now(), false)) return null
        const e = c.e
        if (e.index !== e.targets.length) { e.token = null; e.lease_until = 0; e.revision++; await executionRef(tx, claim.job_id).set({ data: e }); return null }
        const body = { contract: 'owner_cloud_erase_receipt_v1', deletion_job_id: claim.job_id, environment_id: c.admission.environment_id,
          scope: c.scope, scope_sha256: e.scope_sha256, inputs_sha256: e.inputs_sha256, assessment_sha256: e.assessment_sha256,
          authority_epoch: c.assessment.epoch, targets: e.targets, targets_sha256: digest(e.targets), results: e.results, native_admission: e.admission, completed_at: new Date(now()).toISOString() }
        const receipt = { ...body, receipt_sha256: digest(body) }
        await retainWitness(tx, receipt, c)
        await receiptRef(tx, claim.job_id).set({ data: receipt }); return finalized(tx, receipt, c, e)
      })
    },
  }
}
module.exports = { OwnerCloudErasureError, assessApplicationWriters, createOwnerCloudErasureRepository, evaluateSynchronousNativeAdmission }
