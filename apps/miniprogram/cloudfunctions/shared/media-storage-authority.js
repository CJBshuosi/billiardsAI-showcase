const crypto = require('crypto')
const { canonicalJson, digest, documentId, validOwnership, validSavedSelection } = require('./shot-ownership')
const { STORAGE_AUTHORITY_CONTRACT, ManagedMediaAdmissionError, readManagedMediaBirth } = require('./managed-media-admission')
const { validManagedUploadAttempt } = require('./managed-upload-attempt')
const { validManagedUploadRequestBinding } = require('./managed-upload-request')
const { validUploadReceipt, UploadTerminalReceiptError, receiptPageId, prepareUploadReceipt, readUploadReceiptCatalog } = require('./upload-terminal-receipts')

const PAGE_SIZE = 16
const PAGE_COUNT = 4
const MAX_PAGE_BYTES = 128 * 1024
const PAGE_FIELDS = ['contract', 'clip_id', 'identity_sha256', 'epoch', 'page_index', 'entries']
const LIMIT = 64
const ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,128}$/u
const HASH = /^[a-f0-9]{64}$/u
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/u
const OBJECT_ID = /^key-[a-f0-9]{64}$/u
const IDENTITY_FIELDS = ['publication_kind', 'clip_id', 'session_id', 'shot_id', 'source_job_id', 'artifact_id', 'table_id', 'device_id', 'content_sha256', 'byte_size', 'content_type', 'source_kind', 'owner_user_id', 'ownership_receipt_sha256', 'save_receipt_sha256', 'postgame_manifest_sha256']
const ROOT_FIELDS = ['contract', 'identity', 'identity_sha256', 'epoch', 'high_water', 'object_ids', 'admission', 'revision', 'created_at']
const OP_FIELDS = ['contract', 'clip_id', 'identity_sha256', 'epoch', 'sequence', 'operation_id', 'ingest_job_id', 'kind', 'input_object_id', 'target', 'outcome', 'objects', 'reason', 'revision', 'created_at']
const OBJECT_FIELDS = ['contract', 'record_kind', 'object_id', 'clip_id', 'identity_sha256', 'epoch', 'sequence', 'operation_id', 'kind', 'cloud_path', 'file_id', 'state', 'references', 'revision']
const ALIAS_FIELDS = ['contract', 'record_kind', 'object_id', 'clip_id', 'identity_sha256', 'epoch', 'cloud_path', 'file_id']
const TICKET_FIELDS = ['clip_id', 'operation_id', 'sequence', 'epoch', 'identity_sha256', 'target']
const REASONS = [null, 'not_started', 'provider_error', 'result_lost', 'cancelled', 'unknown']
class MediaStorageAuthorityError extends Error {
  constructor(code) { super(code); this.name = 'MediaStorageAuthorityError'; this.code = code }
}
function fail(suffix = 'INVALID') { throw new MediaStorageAuthorityError(`STORAGE_AUTHORITY_${suffix}`) }
function exact(value, fields, metadata = false) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value)
    && Reflect.ownKeys(value).filter(key => !(metadata && key === '_id')).length === fields.length
    && fields.every(field => Object.hasOwn(value, field))
    && Reflect.ownKeys(value).every(key => fields.includes(key) || (metadata && key === '_id')))
}
const dense = value => Array.isArray(value) && Array.from({ length: value.length }, (_, i) => i).every(i => Object.hasOwn(value, i))
function validString(value) {
  return typeof value === 'string' && value.length > 0 && !/[\u0000-\u001f\u007f]/u.test(value)
    && Buffer.from(value, 'utf8').toString('utf8') === value
}
function validId(value) { return typeof value === 'string' && ID.test(value) }
function validHash(value) { return typeof value === 'string' && HASH.test(value) }
function validUuid(value) { return typeof value === 'string' && UUID.test(value) }
function validObjectId(value) { return typeof value === 'string' && OBJECT_ID.test(value) }
function positive(value) { return Number.isSafeInteger(value) && value > 0 }
function validIdentity(identity) {
  if (!exact(identity, IDENTITY_FIELDS) || !['saved_private', 'ordinary'].includes(identity.publication_kind)
      || !['clip_id', 'session_id', 'shot_id', 'artifact_id', 'table_id', 'device_id', 'source_kind'].every(k => validId(identity[k]))
      || !validHash(identity.content_sha256) || !positive(identity.byte_size) || identity.content_type !== 'video/mp4') return false
  const ownerFields = ['owner_user_id', 'ownership_receipt_sha256', 'save_receipt_sha256', 'postgame_manifest_sha256']
  return identity.publication_kind === 'ordinary'
    ? (identity.source_job_id === null || validId(identity.source_job_id)) && ownerFields.every(k => identity[k] === null)
    : validId(identity.source_job_id) && validId(identity.owner_user_id)
      && ownerFields.slice(1).every(k => validHash(identity[k]))
}
function makeStorageIdentity(clip, { tableId, deviceId, kind } = {}) {
  if (!clip || !['ordinary', 'saved_private'].includes(kind)) fail()
  if ((clip.table_id !== undefined && clip.table_id !== tableId) || (clip._device_id !== undefined && clip._device_id !== deviceId)) fail()
  if (kind === 'saved_private' && (!validOwnership(clip) || !validSavedSelection(clip))) fail()
  const identity = {
    publication_kind: kind, clip_id: clip.clip_id, session_id: clip.session_id, shot_id: clip.shot_id,
    source_job_id: clip.job_id ?? null, artifact_id: clip.artifact_id, table_id: tableId, device_id: deviceId,
    content_sha256: clip.content_sha256, byte_size: clip.byte_size, content_type: clip.content_type, source_kind: clip.source_kind,
    owner_user_id: kind === 'saved_private' ? clip.owner_user_id : null,
    ownership_receipt_sha256: kind === 'saved_private' ? clip.ownership_receipt_sha256 : null,
    save_receipt_sha256: kind === 'saved_private' ? clip.save_receipt_sha256 : null,
    postgame_manifest_sha256: kind === 'saved_private' ? clip.postgame_manifest_sha256 : null,
  }
  if (!validIdentity(identity)) fail()
  return identity
}
async function optional(ref) {
  try { return (await ref.get()).data ?? null } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error?.message || '')) return null
    throw error
  }
}
const pageId = (clipId, index) => `page-${documentId(clipId)}-${index}`
const pageIndex = sequence => Math.floor((sequence - 1) / PAGE_SIZE)
function boundedPage(page) { return Buffer.byteLength(JSON.stringify(page), 'utf8') <= MAX_PAGE_BYTES }
const keyId = path => `key-${documentId(path)}`
const fileId = file => `file-${documentId(file)}`
function targetFor(clipId, operationId, kind) {
  const cloud_path = kind === 'upload_authorization'
    ? `managed-clips-v1/${documentId(clipId)}/staging/${operationId}.mp4`
    : `published/managed-clips-v1/${documentId(clipId)}/${operationId}.mp4`
  return { object_id: keyId(cloud_path), kind: kind === 'upload_authorization' ? 'staging' : 'published', cloud_path }
}
function validTarget(target) { return exact(target, ['object_id', 'kind', 'cloud_path']) && ['staging', 'published'].includes(target.kind) && validString(target.cloud_path) && target.object_id === keyId(target.cloud_path) }
function validObservation(object) {
  return exact(object, ['kind', 'cloud_path', 'file_id']) && ['staging', 'published'].includes(object.kind)
    && validString(object.cloud_path) && validString(object.file_id) && object.file_id.startsWith('cloud://') && object.file_id.length > 8 && object.file_id.length <= 2048
}
function validRoot(root, clipId) {
  return exact(root, ROOT_FIELDS, true) && root.contract === STORAGE_AUTHORITY_CONTRACT
    && validIdentity(root.identity) && root.identity.clip_id === clipId && root.identity_sha256 === digest(root.identity)
    && validUuid(root.epoch) && Number.isSafeInteger(root.high_water) && root.high_water >= 0 && root.high_water <= LIMIT
    && dense(root.object_ids) && root.object_ids.length === root.high_water && new Set(root.object_ids).size === root.object_ids.length
    && root.object_ids.every(validObjectId)
    && ['open', 'closed'].includes(root.admission) && positive(root.revision) && validString(root.created_at) && Number.isFinite(Date.parse(root.created_at))
}
function bound(record, root) {
  return record.contract === STORAGE_AUTHORITY_CONTRACT && record.clip_id === root.identity.clip_id
    && record.identity_sha256 === root.identity_sha256 && record.epoch === root.epoch
}
function validOperation(op, root, sequence) {
  if (!exact(op, OP_FIELDS) || !bound(op, root) || op.sequence !== sequence || !validUuid(op.operation_id)
      || !validId(op.ingest_job_id) || !['upload_authorization', 'server_copy'].includes(op.kind)
      || !validTarget(op.target) || canonicalJson(op.target) !== canonicalJson(targetFor(op.clip_id, op.operation_id, op.kind))
      || root.object_ids[sequence - 1] !== op.target.object_id || !positive(op.revision)
      || !validString(op.created_at) || !Number.isFinite(Date.parse(op.created_at))
      || (op.kind === 'upload_authorization' ? op.input_object_id !== null : !validObjectId(op.input_object_id))
      || !REASONS.includes(op.reason) || !dense(op.objects) || op.objects.length > 1 || !op.objects.every(validObservation)) return false
  if (op.outcome === null) return op.objects.length === 0 && op.reason === null && op.revision === 1
  return ['external_issued', 'server_succeeded', 'no_write', 'unknown'].includes(op.outcome)
    && op.revision >= 2 && op.revision <= 3 && (op.outcome !== 'external_issued' || op.kind === 'upload_authorization')
    && (op.revision !== 3 || (op.outcome === 'unknown' && op.objects.length === 1))
    && (op.outcome !== 'server_succeeded' || op.kind === 'server_copy')
    && (!['external_issued', 'server_succeeded'].includes(op.outcome) || op.objects.length === 1)
    && (op.outcome !== 'no_write' || op.objects.length === 0)
    && op.objects.every(o => o.kind === op.target.kind && o.cloud_path === op.target.cloud_path)
}
function validObject(object, root, op) {
  return exact(object, OBJECT_FIELDS) && bound(object, root) && object.record_kind === 'key'
    && object.object_id === op.target.object_id && object.sequence === op.sequence && object.operation_id === op.operation_id
    && object.kind === op.target.kind && object.cloud_path === op.target.cloud_path
    && (object.file_id === null || validObservation({ kind: object.kind, cloud_path: object.cloud_path, file_id: object.file_id }))
    && ['active', 'delete_reserved', 'deleted'].includes(object.state) && positive(object.revision)
    && dense(object.references) && object.references.length <= LIMIT
    && object.references.every(ref => exact(ref, ['clip_id', 'identity_sha256']) && ref.clip_id === object.clip_id && ref.identity_sha256 === object.identity_sha256)
    && new Set(object.references.map(canonicalJson)).size === object.references.length
    && object.file_id === (op.objects[0]?.file_id ?? null)
    && object.revision === 1 + (object.file_id === null ? 0 : 1) + object.references.length
}
function validAlias(alias, object, root) {
  return exact(alias, ALIAS_FIELDS, true) && bound(alias, root) && alias.record_kind === 'file'
    && ['object_id', 'cloud_path', 'file_id'].every(k => alias[k] === object[k])
}
// Shared validation for the complete original catalog. Database alias existence
// is additionally checked by inventory(); these pure rows bind every file alias.
function validStorageInventory(root, operations, objects) {
  if (!validRoot(root, root?.identity?.clip_id) || !dense(operations) || !dense(objects)
      || operations.length !== root.high_water || objects.length !== root.high_water) return false
  const ids = new Set(); const preceding = new Map()
  for (let i = 0; i < root.high_water; i++) {
    const op = operations[i]; const object = objects[i]
    if (!validOperation(op, root, i + 1) || ids.has(op.operation_id) || !validObject(object, root, op)) return false
    ids.add(op.operation_id)
    if (op.kind === 'server_copy') {
      const input = preceding.get(op.input_object_id)
      if (!input || input.kind !== 'staging' || input.file_id === null) return false
    }
    preceding.set(object.object_id, object)
  }
  const revision = 1 + root.high_water + operations.reduce((sum, op) => sum + op.revision - 1, 0)
    + objects.reduce((sum, object) => sum + object.references.length, 0) + (root.admission === 'closed' ? 1 : 0)
  return root.revision === revision
}
function createCloudMediaStorageAuthority(db, { clock = Date.now, idFactory = crypto.randomUUID } = {}) {
  const rootRef = (tx, clipId) => tx.collection('clip_storage_authorities').doc(clipId)
  const pageRef = (tx, clipId, index) => tx.collection('media_storage_operations').doc(pageId(clipId, index))
  function newPage(root, index) { return { contract: STORAGE_AUTHORITY_CONTRACT, clip_id: root.identity.clip_id, identity_sha256: root.identity_sha256, epoch: root.epoch, page_index: index, entries: [] } }
  async function writePage(tx, root, page) {
    if (!boundedPage(page)) fail('LIMIT')
    await pageRef(tx, root.identity.clip_id, page.page_index).set({ data: page })
  }
  const objectRef = (tx, objectId) => tx.collection('media_storage_objects').doc(objectId)
  const tombstone = (tx, clipId) => optional(tx.collection('owner_clip_tombstones').doc(clipId))
  function now() { const value = clock(); if (!Number.isSafeInteger(value)) fail(); return new Date(value).toISOString() }
  function uuid() { const value = idFactory(); if (!validUuid(value)) fail(); return value }
  async function rootFor(tx, identity) {
    if (!validIdentity(identity)) fail()
    const root = await optional(rootRef(tx, identity.clip_id))
    if (!validRoot(root, identity.clip_id)) fail('INCOMPLETE')
    if (root.identity_sha256 !== digest(identity)) fail('CONFLICT')
    return root
  }
  async function inventory(tx, root) {
    const operations = []; const objects = []; const reasons = []; const operationIds = new Set()
    const precedingObjects = new Map(); const pages = new Map(); const aliases = new Map()
    // Read the entire fixed catalog, including unused slots, to reject tail pages.
    for (let index = 0; index < PAGE_COUNT; index++) {
      const page = await optional(pageRef(tx, root.identity.clip_id, index))
      const expected = Math.max(0, Math.min(PAGE_SIZE, root.high_water - index * PAGE_SIZE))
      if (!expected) { if (page !== null) reasons.push(`unexpected_page:${index}`); continue }
      if (page === null) { reasons.push(`missing_page:${index}`); continue }
      if (!exact(page, PAGE_FIELDS, true) || !bound(page, root) || page.page_index !== index || !dense(page.entries)
          || page.entries.length !== expected || !boundedPage(page)) { reasons.push(`invalid_page:${index}`); continue }
      pages.set(index, page)
      for (let offset = 0; offset < expected; offset++) {
        const sequence = index * PAGE_SIZE + offset + 1; const entry = page.entries[offset]
        if (!exact(entry, ['operation', 'object'])) { reasons.push(`invalid_entry:${sequence}`); continue }
        const { operation: op, object } = entry
        if (!validOperation(op, root, sequence) || operationIds.has(op.operation_id)) { reasons.push(`invalid_operation:${sequence}`); continue }
        operationIds.add(op.operation_id); operations.push(op)
        if (!validObject(object, root, op)) { reasons.push(`invalid_object:${sequence}`); continue }
        objects.push(object)
        let aliasValid = true
        if (object.file_id !== null) {
          const id = fileId(object.file_id)
          if (!aliases.has(id)) aliases.set(id, await optional(objectRef(tx, id)))
          aliasValid = validAlias(aliases.get(id), object, root)
        }
        if (!aliasValid) reasons.push(`invalid_alias:${sequence}`)
        if (op.kind === 'server_copy') {
          // Historical inputs remain valid after reservation/deletion, but must
          // be preceding, fully validated staging members with a file alias.
          const input = precedingObjects.get(op.input_object_id)
          if (!input || input.kind !== 'staging' || input.sequence >= sequence || input.file_id === null) reasons.push(`invalid_input:${sequence}`)
        }
        if (aliasValid) precedingObjects.set(object.object_id, object)
      }
    }
    if (!reasons.length && !validStorageInventory(root, operations, objects)) reasons.push('invalid_authority_revision')
    return { operations, objects, reasons, pages, aliases }
  }
  async function checkedInventory(tx, root) {
    const value = await inventory(tx, root)
    if (value.reasons.length) fail('INCOMPLETE')
    return value
  }
  async function openInventory(tx, identity) {
    const root = await rootFor(tx, identity)
    if (root.admission !== 'open' || await tombstone(tx, identity.clip_id) !== null) fail('CONFLICT')
    return { root, inventory: await checkedInventory(tx, root) }
  }
  async function markedBirth(tx, identity) {
    if (identity.publication_kind !== 'saved_private' || !identity.session_id.startsWith('managed-')) return null
    try {
      return await readManagedMediaBirth(tx, { sessionId: identity.session_id, tableId: identity.table_id, deviceId: identity.device_id })
    } catch (error) {
      // The leaf reader deliberately hides DB failures behind INVALID. Keep
      // unavailable/corrupt provenance retryable without permitting fallback.
      if (error instanceof ManagedMediaAdmissionError) fail(error.code === 'MEDIA_ADMISSION_CONFLICT' ? 'CONFLICT' : 'INCOMPLETE')
      throw error
    }
  }
  async function getAdmission(tx, identity) {
    if (!validIdentity(identity)) fail()
    if (await markedBirth(tx, identity) === null) fail('INCOMPLETE')
    const root = await rootFor(tx, identity)
    if (root.admission !== 'open' || await tombstone(tx, identity.clip_id) !== null) fail('CONFLICT')
    // Binding only: actual operation/reference admission validates the complete
    // inventory once, in the caller's same publication transaction.
    return root
  }
  async function enroll(tx, identity) {
    if (!validIdentity(identity)) fail()
    const untracked = { status: 'legacy_untracked', contract: STORAGE_AUTHORITY_CONTRACT, identity_sha256: digest(identity) }
    if (await markedBirth(tx, identity) === null) return untracked
    if (await tombstone(tx, identity.clip_id) !== null) fail('CONFLICT')
    const old = await optional(rootRef(tx, identity.clip_id))
    if (old !== null) {
      if (!validRoot(old, identity.clip_id)) fail('INCOMPLETE')
      if (old.identity_sha256 !== digest(identity)) fail('CONFLICT')
      if (old.admission !== 'open') fail('CONFLICT')
      await checkedInventory(tx, old); return old
    }
    // The marked birth establishes writer provenance; these permanent per-clip
    // fences prevent adopting an existing publication/job after authority loss.
    // A present pointer blocks fresh admission even if its job is missing or
    // contradictory. Negative job scans cannot establish this provenance.
    const clip = await optional(tx.collection('clips').doc(identity.clip_id))
    const savedJob = await optional(tx.collection('saved_clip_sync_idempotency').doc(documentId(identity.clip_id)))
    if (clip !== null || savedJob !== null) return untracked
    const root = { contract: STORAGE_AUTHORITY_CONTRACT, identity: JSON.parse(canonicalJson(identity)), identity_sha256: digest(identity), epoch: uuid(), high_water: 0, object_ids: [], admission: 'open', revision: 1, created_at: now() }
    await rootRef(tx, identity.clip_id).set({ data: root }); return root
  }
  async function beginOperation(tx, args) {
    if (!exact(args, ['identity', 'ingestJobId', 'kind', ...(Object.hasOwn(args || {}, 'inputObjectId') ? ['inputObjectId'] : [])])
        || !validId(args.ingestJobId) || !['upload_authorization', 'server_copy'].includes(args.kind)) fail()
    const { root, inventory } = await openInventory(tx, args.identity)
    return appendOperation(tx, root, inventory, args)
  }
  async function appendOperation(tx, root, existing, args) {
    if (root.high_water >= LIMIT || root.object_ids.length >= LIMIT) fail('LIMIT')
    const inputObjectId = args.inputObjectId ?? null
    if (args.kind === 'server_copy') {
      const input = existing.objects.find(o => o.object_id === inputObjectId)
      if (!input || input.file_id === null || input.state !== 'active' || input.kind !== 'staging') fail('INCOMPLETE')
    } else if (inputObjectId !== null) fail()
    const operationId = uuid(); if (existing.operations.some(op => op.operation_id === operationId)) fail('CONFLICT')
    const target = targetFor(args.identity.clip_id, operationId, args.kind)
    if (existing.objects.some(object => object.object_id === target.object_id)) fail('CONFLICT')
    const sequence = root.high_water + 1
    const ticket = { clip_id: args.identity.clip_id, operation_id: operationId, sequence, epoch: root.epoch, identity_sha256: root.identity_sha256, target }
    const op = { contract: STORAGE_AUTHORITY_CONTRACT, ...ticket, ingest_job_id: args.ingestJobId, kind: args.kind, input_object_id: inputObjectId, outcome: null, objects: [], reason: null, revision: 1, created_at: now() }
    const object = { contract: STORAGE_AUTHORITY_CONTRACT, record_kind: 'key', object_id: target.object_id, clip_id: ticket.clip_id, identity_sha256: root.identity_sha256, epoch: root.epoch, sequence, operation_id: operationId, kind: target.kind, cloud_path: target.cloud_path, file_id: null, state: 'active', references: [], revision: 1 }
    const index = pageIndex(sequence); const page = existing.pages.get(index) || newPage(root, index)
    page.entries.push({ operation: op, object }); await writePage(tx, root, page)
    await rootRef(tx, ticket.clip_id).update({ data: { high_water: sequence, object_ids: [...root.object_ids, target.object_id], revision: root.revision + 1 } })
    return ticket
  }
  function validateTicket(ticket) {
    if (!exact(ticket, TICKET_FIELDS) || !validId(ticket.clip_id) || !positive(ticket.sequence) || ticket.sequence > LIMIT
        || !validUuid(ticket.operation_id) || !validUuid(ticket.epoch) || !validHash(ticket.identity_sha256) || !validTarget(ticket.target)) fail()
  }
  function operationForTicket(inventory, ticket) {
    const op = inventory.operations.find(value => value.sequence === ticket.sequence)
    if (!op) fail('INCOMPLETE')
    if (!TICKET_FIELDS.every(key => canonicalJson(op[key]) === canonicalJson(ticket[key]))) fail('CONFLICT')
    return op
  }
  async function operationContext(tx, ticket) {
    validateTicket(ticket)
    const root = await optional(rootRef(tx, ticket.clip_id))
    if (!validRoot(root, ticket.clip_id)) fail('INCOMPLETE')
    const inventory = await checkedInventory(tx, root)
    return { root, inventory, op: operationForTicket(inventory, ticket) }
  }
  function stagingForUpload(inventory, op, ingestJobId) {
    if (!op || op.kind !== 'upload_authorization' || op.ingest_job_id !== ingestJobId || op.outcome !== 'external_issued') fail('CONFLICT')
    const input = inventory.objects.find(object => object.object_id === op.target.object_id)
    if (!input || input.state !== 'active' || input.kind !== 'staging' || input.file_id === null) fail('INCOMPLETE')
    return input
  }
  const observedObject = object => ({ kind: object.kind, cloud_path: object.cloud_path, file_id: object.file_id })
  async function beginCopyForUpload(tx, args) {
    if (!exact(args, ['identity', 'ingestJobId', 'uploadAttempt']) || !validId(args.ingestJobId) || !validManagedUploadAttempt(args.uploadAttempt)) fail()
    const { root, inventory } = await openInventory(tx, args.identity)
    const upload = inventory.operations.find(op => op.sequence === args.uploadAttempt.sequence && op.operation_id === args.uploadAttempt.operation_id)
    const input = stagingForUpload(inventory, upload, args.ingestJobId)
    const ticket = await appendOperation(tx, root, inventory, { identity: args.identity, ingestJobId: args.ingestJobId, kind: 'server_copy', inputObjectId: input.object_id })
    return { ticket, input: { object_id: input.object_id, ...observedObject(input) } }
  }
  async function getUploadOperation(tx, args) {
    if (!exact(args, ['identity', 'ingestJobId', 'uploadAttempt']) || !validId(args.ingestJobId) || !validManagedUploadAttempt(args.uploadAttempt)) fail()
    // Recovery is read-only: closed admission and owner tombstones cannot
    // invalidate the identity of an already reserved upload operation.
    const root = await rootFor(tx, args.identity)
    const inventory = await checkedInventory(tx, root)
    const op = inventory.operations.find(value => value.sequence === args.uploadAttempt.sequence && value.operation_id === args.uploadAttempt.operation_id)
    if (!op || op.kind !== 'upload_authorization' || op.ingest_job_id !== args.ingestJobId) fail('CONFLICT')
    return Object.fromEntries(TICKET_FIELDS.map(field => [field, op[field]]))
  }
  async function recordUploadReceipt(tx, { requestBinding, receipt }) {
    if (!validUploadReceipt(receipt)) fail()
    if (!validManagedUploadRequestBinding(requestBinding)) fail('INCOMPLETE')
    const root = await rootFor(tx, requestBinding.identity)
    const inventory = await checkedInventory(tx, root)
    const pages = []
    for (let index = 0; index < PAGE_COUNT; index++) {
      pages.push(await optional(tx.collection('upload_terminal_receipts').doc(receiptPageId(root.identity.clip_id, index))))
    }
    const binding = Object.fromEntries(Object.entries(requestBinding).filter(([key]) => key !== '_id'))
    let prepared
    try {
      prepared = prepareUploadReceipt({ root, operations: inventory.operations, pages, requestBinding: binding, receipt, receivedAt: now() })
    } catch (error) {
      if (error instanceof UploadTerminalReceiptError) fail(error.code)
      throw error
    }
    if (prepared.page) await tx.collection('upload_terminal_receipts').doc(receiptPageId(root.identity.clip_id, prepared.page.page_index)).set({ data: prepared.page })
    return prepared.ack
  }
  async function getOperation(tx, ticket) { return (await operationContext(tx, ticket)).op }
  async function recordOperationResult(args) {
    if (!exact(args, ['ticket', 'outcome', 'objects', ...(Object.hasOwn(args || {}, 'reason') ? ['reason'] : [])])) fail()
    const { ticket, outcome, objects } = args; const reason = args.reason ?? null
    if (!['external_issued', 'server_succeeded', 'no_write', 'unknown'].includes(outcome) || !REASONS.includes(reason)
        || !Array.isArray(objects) || objects.length > 1 || !objects.every(validObservation)) fail()
    return db.runTransaction(async tx => {
      const { op, root, inventory } = await operationContext(tx, ticket)
      if ((outcome === 'external_issued' && op.kind !== 'upload_authorization') || (outcome === 'server_succeeded' && op.kind !== 'server_copy')
          || (['external_issued', 'server_succeeded'].includes(outcome) && objects.length !== 1) || (outcome === 'no_write' && objects.length !== 0)) fail()
      if (objects.some(o => o.kind !== op.target.kind || o.cloud_path !== op.target.cloud_path)) fail('CONFLICT')
      if (op.outcome !== null) {
        if (canonicalJson({ outcome, objects, reason }) === canonicalJson({ outcome: op.outcome, objects: op.objects, reason: op.reason })) return op
        // A lost result can later reveal the reserved target's file identity, but
        // the original unresolved outcome and reason can never be cleared.
        if (op.outcome !== 'unknown' || outcome !== 'unknown' || reason !== op.reason || op.objects.length !== 0 || objects.length !== 1) fail('CONFLICT')
      }
      const object = inventory.objects.find(value => value.object_id === op.target.object_id)
      if (objects.length) {
        const observed = objects[0]; const aliasId = fileId(observed.file_id); const aliasRef = objectRef(tx, aliasId)
        const alias = inventory.aliases.has(aliasId) ? inventory.aliases.get(aliasId) : await optional(aliasRef)
        const updated = { ...object, file_id: observed.file_id, revision: object.revision + 1 }
        if (object.file_id !== null && object.file_id !== observed.file_id) fail('CONFLICT')
        if (alias !== null && !validAlias(alias, updated, root)) fail('CONFLICT')
        await aliasRef.set({ data: { contract: STORAGE_AUTHORITY_CONTRACT, record_kind: 'file', object_id: object.object_id, clip_id: object.clip_id, identity_sha256: object.identity_sha256, epoch: object.epoch, cloud_path: object.cloud_path, file_id: observed.file_id } })
        Object.assign(object, { file_id: observed.file_id, revision: updated.revision })
      }
      const update = { outcome, objects: objects.map(o => ({ ...o })), reason, revision: op.revision + 1 }
      Object.assign(op, update)
      await writePage(tx, root, inventory.pages.get(pageIndex(ticket.sequence)))
      await rootRef(tx, ticket.clip_id).update({ data: { revision: root.revision + 1 } }); return { ...op, ...update }
    })
  }
  async function registerReferences(tx, args) {
    if (!exact(args, ['identity', 'objects'])) fail()
    const { identity, objects } = args
    if (!Array.isArray(objects) || objects.length > LIMIT || !objects.every(validObservation)) fail()
    const { root, inventory } = await openInventory(tx, identity)
    return registerObservedReferences(tx, root, inventory, objects)
  }
  async function registerObservedReferences(tx, root, inventory, objects) {
    const identity = root.identity
    const updates = new Map()
    for (const observed of objects) {
      const object = inventory.objects.find(o => o.cloud_path === observed.cloud_path)
      if (!object || object.kind !== observed.kind || object.file_id !== observed.file_id || object.file_id === null || object.state !== 'active') fail('CONFLICT')
      const reference = { clip_id: identity.clip_id, identity_sha256: root.identity_sha256 }
      if (!object.references.some(ref => canonicalJson(ref) === canonicalJson(reference))) {
        if (object.references.length >= LIMIT) fail('LIMIT')
        updates.set(object.object_id, [object, reference])
      }
    }
    const changedPages = new Set()
    for (const [object, reference] of updates.values()) {
      Object.assign(object, { references: [...object.references, reference], revision: object.revision + 1 }); changedPages.add(pageIndex(object.sequence))
    }
    for (const index of changedPages) await writePage(tx, root, inventory.pages.get(index))
    if (updates.size) await rootRef(tx, identity.clip_id).update({ data: { revision: root.revision + updates.size } })
    return { contract: STORAGE_AUTHORITY_CONTRACT, identity_sha256: root.identity_sha256, object_ids: objects.map(o => keyId(o.cloud_path)) }
  }
  async function registerCopyReferences(tx, args) {
    if (!exact(args, ['identity', 'ingestJobId', 'copyTicket']) || !validId(args.ingestJobId)) fail()
    validateTicket(args.copyTicket)
    const { root, inventory } = await openInventory(tx, args.identity)
    const copy = operationForTicket(inventory, args.copyTicket)
    if (copy.kind !== 'server_copy' || copy.ingest_job_id !== args.ingestJobId || copy.outcome !== 'server_succeeded') fail('CONFLICT')
    const output = inventory.objects.find(object => object.object_id === copy.target.object_id)
    const upload = inventory.operations.find(op => op.target.object_id === copy.input_object_id && op.sequence < copy.sequence)
    const input = stagingForUpload(inventory, upload, args.ingestJobId)
    const objects = [observedObject(input), observedObject(output)]
    await registerObservedReferences(tx, root, inventory, objects)
    return { file_id: output.file_id, staging_file_id: input.file_id, objects }
  }
  async function snapshotForDeletion(tx, clipId) {
    if (!validId(clipId)) fail()
    const root = await optional(rootRef(tx, clipId))
    const base = { contract: STORAGE_AUTHORITY_CONTRACT, clip_id: clipId, complete: false, erasable: false, operations: [], objects: [] }
    if (root === null) return { ...base, status: 'legacy_untracked', reasons: ['missing_authority'] }
    if (!validRoot(root, clipId)) return { ...base, status: 'legacy_authority_incomplete', reasons: ['invalid_authority'] }
    const inventory = await inventoryForSnapshot(tx, root)
    return { ...base, ...inventory, status: inventory.reasons.length ? 'legacy_authority_incomplete' : 'tracked', identity_sha256: root.identity_sha256, epoch: root.epoch, high_water: root.high_water, revision: root.revision, admission: root.admission, complete: inventory.reasons.length === 0 }
  }
  async function readWriterClosureInputs(tx, args) {
    if (!exact(args, ['clipId']) || !validId(args.clipId)) fail()
    const root = await optional(rootRef(tx, args.clipId))
    if (!validRoot(root, args.clipId) || root.admission !== 'closed') fail('INCOMPLETE')
    const birth = await markedBirth(tx, root.identity)
    if (birth === null) fail('INCOMPLETE')
    const value = await checkedInventory(tx, root)
    const pages = []
    for (let index = 0; index < PAGE_COUNT; index++) {
      pages.push(await optional(tx.collection('upload_terminal_receipts').doc(receiptPageId(args.clipId, index))))
    }
    let catalog
    try { catalog = readUploadReceiptCatalog({ root, operations: value.operations, pages }) } catch (error) {
      if (error instanceof UploadTerminalReceiptError) fail('INCOMPLETE')
      throw error
    }
    const clean = row => JSON.parse(canonicalJson(Object.fromEntries(Object.entries(row).filter(([key]) => key !== '_id'))))
    const result = { root: clean(root), operations: value.operations.map(clean), objects: value.objects.map(clean),
      receipt_entries: catalog.entries, receipt_page_sha256: catalog.page_sha256, birth: clean(birth), birth_sha256: digest(clean(birth)) }
    result.fingerprint_sha256 = digest({ root: result.root, operations: result.operations, objects: result.objects,
      receipt_page_sha256: result.receipt_page_sha256, birth_sha256: result.birth_sha256 })
    return result
  }
  async function inventoryForSnapshot(tx, root) {
    const value = await inventory(tx, root)
    for (const op of value.operations) if (![ 'server_succeeded', 'no_write' ].includes(op.outcome)) value.reasons.push(`unresolved_operation:${op.sequence}`)
    return { operations: value.operations, objects: value.objects, reasons: value.reasons }
  }
  async function closeForDeletion(tx, identity) {
    if (!validIdentity(identity)) fail()
    const root = await optional(rootRef(tx, identity.clip_id))
    if (root === null) return { status: 'legacy_untracked', contract: STORAGE_AUTHORITY_CONTRACT, identity_sha256: digest(identity), reasons: ['missing_authority'] }
    if (!validRoot(root, identity.clip_id) || root.identity_sha256 !== digest(identity)) return { status: 'legacy_authority_incomplete', contract: STORAGE_AUTHORITY_CONTRACT, identity_sha256: digest(identity), reasons: ['invalid_authority'] }
    const value = await inventory(tx, root)
    if (root.admission !== 'closed') await rootRef(tx, identity.clip_id).update({ data: { admission: 'closed', revision: root.revision + 1 } })
    return { status: value.reasons.length ? 'legacy_authority_incomplete' : 'tracked_closed', contract: STORAGE_AUTHORITY_CONTRACT, identity_sha256: root.identity_sha256, epoch: root.epoch, high_water: root.high_water, reasons: value.reasons }
  }
  return { enroll, getAdmission, beginOperation, beginCopyForUpload, getUploadOperation, recordUploadReceipt, getOperation, recordOperationResult, registerReferences, registerCopyReferences, closeForDeletion, snapshotForDeletion, readWriterClosureInputs }
}
module.exports = { STORAGE_AUTHORITY_CONTRACT, MediaStorageAuthorityError, makeStorageIdentity, createCloudMediaStorageAuthority, validStorageInventory }
