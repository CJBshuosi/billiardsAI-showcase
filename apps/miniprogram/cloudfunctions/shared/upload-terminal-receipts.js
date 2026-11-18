const { canonicalJson, digest, documentId } = require('./shot-ownership')
const { validManagedUploadRequest, validManagedUploadRequestBinding } = require('./managed-upload-request')
const { validManagedUploadAttempt } = require('./managed-upload-attempt')

const UPLOAD_RECEIPT_CONTRACT = 'managed_upload_receipt_v1'
const UPLOAD_RECEIPT_ACK_CONTRACT = 'managed_upload_receipt_ack_v1'
const PAGE_SIZE = 16
const PAGE_COUNT = 4
const MAX_PAGE_BYTES = 128 * 1024
const HASH = /^[a-f0-9]{64}$/u
const TICKET_FIELDS = ['clip_id', 'operation_id', 'sequence', 'epoch', 'identity_sha256', 'target']
function exact(value, fields, metadata = false) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value)
    && Reflect.ownKeys(value).filter(key => !(metadata && key === '_id')).length === fields.length
    && fields.every(key => Object.hasOwn(value, key))
    && Reflect.ownKeys(value).every(key => fields.includes(key) || (metadata && key === '_id')))
}
function utc(value, precision) {
  if (typeof value !== 'string' || !(precision === 6
    ? /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$/u
    : /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u).test(value)) return false
  const milliseconds = `${value.slice(0, 23)}Z`
  const date = new Date(milliseconds)
  return Number.isFinite(date.getTime()) && date.toISOString() === milliseconds
}
function validUploadReceipt(value) {
  if (!exact(value, ['contract', 'upload_request', 'upload_attempt', 'evidence'])
      || value.contract !== UPLOAD_RECEIPT_CONTRACT || !validManagedUploadRequest(value.upload_request)) return false
  const attempt = value.upload_attempt; const evidence = value.evidence
  if (attempt !== null && (!exact(attempt, ['contract', 'operation_id', 'sequence']) || !validManagedUploadAttempt(attempt))) return false
  if (!evidence || typeof evidence.local_binding_sha256 !== 'string' || !HASH.test(evidence.local_binding_sha256)) return false
  if (evidence.kind === 'retired_before_start') return exact(evidence, ['kind', 'local_binding_sha256'])
  return attempt !== null && exact(evidence, ['kind', 'local_binding_sha256', 'started_at_utc', 'received_at_utc', 'status_code'])
    && evidence.kind === 'http200' && evidence.status_code === 200 && Number.isInteger(evidence.status_code)
    && utc(evidence.started_at_utc, 6) && utc(evidence.received_at_utc, 6)
    // Fixed-width UTC retains all six digits: Date.parse alone loses ordering.
    && evidence.started_at_utc <= evidence.received_at_utc
}
function validUploadReceiptAck(value) {
  return exact(value, ['contract', 'upload_request', 'receipt_sha256', 'received_at_utc', 'erasable'])
    && value.contract === UPLOAD_RECEIPT_ACK_CONTRACT && validManagedUploadRequest(value.upload_request)
    && typeof value.receipt_sha256 === 'string' && HASH.test(value.receipt_sha256)
    && utc(value.received_at_utc, 3) && value.erasable === false
}
class UploadTerminalReceiptError extends Error {
  constructor(code) { super(code); this.code = code }
}
function fail(code = 'INCOMPLETE') { throw new UploadTerminalReceiptError(code) }
const receiptPageId = (clipId, index) => `page-${documentId(clipId)}-${index}`
const bounded = page => Buffer.byteLength(JSON.stringify(page), 'utf8') <= MAX_PAGE_BYTES
function bindingOperation(binding, root, operations) {
  if (!validManagedUploadRequestBinding(binding) || Object.hasOwn(binding, '_id')) fail()
  const op = operations.find(value => value.sequence === binding.ticket.sequence)
  if (canonicalJson(binding.identity) !== canonicalJson(root.identity) || binding.identity_sha256 !== root.identity_sha256
      || binding.table_id !== root.identity.table_id || binding.device_id !== root.identity.device_id
      || !op || op.kind !== 'upload_authorization' || op.ingest_job_id !== binding.job_id
      || !TICKET_FIELDS.every(key => canonicalJson(binding.ticket[key]) === canonicalJson(op[key]))) fail('CONFLICT')
  return op
}
function matchingReceipt(receipt, binding, op) {
  if (!validUploadReceipt(receipt) || receipt.upload_request.request_id !== binding.request_id) fail('CONFLICT')
  if (receipt.upload_attempt !== null && (receipt.upload_attempt.operation_id !== op.operation_id || receipt.upload_attempt.sequence !== op.sequence)) fail('CONFLICT')
  if (receipt.evidence.kind === 'http200' && op.outcome !== 'external_issued') fail('CONFLICT')
}
function ackFor(entry) {
  return { contract: UPLOAD_RECEIPT_ACK_CONTRACT, upload_request: entry.receipt.upload_request,
    receipt_sha256: entry.receipt_sha256, received_at_utc: entry.received_at_utc, erasable: false }
}
// Pure catalog validation/append against one already-checked authority inventory.
// No mutable cache, database lookup, provider call, or negative-write inference.
function readUploadReceiptCatalog({ root, operations, pages }) {
  if (!Array.isArray(pages) || pages.length !== PAGE_COUNT
      || Array.from({ length: PAGE_COUNT }, (_, i) => i).some(i => !Object.hasOwn(pages, i))) fail()
  const requests = new Set(); const operationIds = new Set(); const entries = []; const page_sha256 = []
  for (let index = 0; index < PAGE_COUNT; index++) {
    const page = pages[index]
    if (page === null) { page_sha256.push(digest(null)); continue }
    if (!exact(page, ['contract', 'clip_id', 'identity_sha256', 'epoch', 'page_index', 'entries'], true)
        || page.contract !== UPLOAD_RECEIPT_CONTRACT || page.clip_id !== root.identity.clip_id
        || page.identity_sha256 !== root.identity_sha256 || page.epoch !== root.epoch || page.page_index !== index
        || !Array.isArray(page.entries) || page.entries.length !== PAGE_SIZE
        || Array.from({ length: PAGE_SIZE }, (_, i) => i).some(i => !Object.hasOwn(page.entries, i)) || !bounded(page)
        || index * PAGE_SIZE >= root.high_water || page.entries.every(entry => entry === null)) fail()
    for (let offset = 0; offset < PAGE_SIZE; offset++) {
      const entry = page.entries[offset]; if (entry === null) continue
      const sequence = index * PAGE_SIZE + offset + 1
      if (sequence > root.high_water || !exact(entry, ['sequence', 'request_binding', 'receipt', 'receipt_sha256', 'received_at_utc'])
          || entry.sequence !== sequence || entry.request_binding?.ticket?.sequence !== sequence
          || !validUploadReceipt(entry.receipt) || entry.receipt_sha256 !== digest(entry.receipt) || !utc(entry.received_at_utc, 3)) fail()
      try {
        const original = bindingOperation(entry.request_binding, root, operations)
        matchingReceipt(entry.receipt, entry.request_binding, original)
      } catch (_) { fail() }
      if (requests.has(entry.request_binding.request_id) || operationIds.has(entry.request_binding.ticket.operation_id)) fail()
      requests.add(entry.request_binding.request_id); operationIds.add(entry.request_binding.ticket.operation_id)
      entries.push(entry)
    }
    page_sha256.push(digest(Object.fromEntries(Object.entries(page).filter(([key]) => key !== '_id'))))
  }
  // Detach rows without changing the original receipt request's wire key order.
  return { entries: JSON.parse(JSON.stringify(entries)), page_sha256 }
}
function prepareUploadReceipt({ root, operations, pages, requestBinding, receipt, receivedAt }) {
  const op = bindingOperation(requestBinding, root, operations)
  matchingReceipt(receipt, requestBinding, op)
  const { entries } = readUploadReceiptCatalog({ root, operations, pages })
  const old = entries.find(entry => entry.sequence === op.sequence)
  if (old) {
    if (canonicalJson(old.request_binding) !== canonicalJson(requestBinding) || canonicalJson(old.receipt) !== canonicalJson(receipt)) fail('CONFLICT')
    return { ack: ackFor(old), page: null }
  }
  if (entries.some(entry => entry.request_binding.request_id === requestBinding.request_id
      || entry.request_binding.ticket.operation_id === op.operation_id)) fail('CONFLICT')
  if (!utc(receivedAt, 3)) fail()
  const index = Math.floor((op.sequence - 1) / PAGE_SIZE)
  const page = pages[index] === null ? { contract: UPLOAD_RECEIPT_CONTRACT, clip_id: root.identity.clip_id,
    identity_sha256: root.identity_sha256, epoch: root.epoch, page_index: index, entries: Array(PAGE_SIZE).fill(null) }
    : Object.fromEntries(Object.entries(pages[index]).filter(([key]) => key !== '_id'))
  const entry = { sequence: op.sequence, request_binding: requestBinding, receipt, receipt_sha256: digest(receipt), received_at_utc: receivedAt }
  // Preserve caller snapshots, including arrays, when assembling the one new page.
  page.entries = [...page.entries]; page.entries[(op.sequence - 1) % PAGE_SIZE] = entry
  if (!bounded(page)) fail()
  return { ack: ackFor(entry), page }
}
module.exports = { UPLOAD_RECEIPT_CONTRACT, UPLOAD_RECEIPT_ACK_CONTRACT, validUploadReceipt, validUploadReceiptAck,
  UploadTerminalReceiptError, receiptPageId, readUploadReceiptCatalog, prepareUploadReceipt }
