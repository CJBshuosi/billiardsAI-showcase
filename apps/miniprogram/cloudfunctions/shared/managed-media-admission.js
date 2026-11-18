const { MANAGED_MEDIA_ADMISSION_CONTRACT, validManagedBirth } = require('./managed-command-authority')
const { digest } = require('./shot-ownership')

const STORAGE_AUTHORITY_CONTRACT = 'clip_storage_v2'
const ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,128}$/u
const MANAGED_SESSION = /^managed-[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/iu
const HASH = /^[a-f0-9]{64}$/u
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/u
const IDENTITY_FIELDS = ['publication_kind', 'clip_id', 'session_id', 'shot_id', 'source_job_id',
  'artifact_id', 'table_id', 'device_id', 'content_sha256', 'byte_size', 'content_type', 'source_kind',
  'owner_user_id', 'ownership_receipt_sha256', 'save_receipt_sha256', 'postgame_manifest_sha256']
const MARKER_FIELDS = ['contract', 'storage_contract', 'identity_sha256', 'epoch']
class ManagedMediaAdmissionError extends Error {
  constructor(code) { super(code); this.name = 'ManagedMediaAdmissionError'; this.code = code }
}
function fail(suffix = 'INVALID') { throw new ManagedMediaAdmissionError(`MEDIA_ADMISSION_${suffix}`) }
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value)
const validId = value => typeof value === 'string' && ID.test(value)
const validManagedSession = value => typeof value === 'string' && value.startsWith('managed-') && MANAGED_SESSION.test(value)
const validScopeId = value => typeof value === 'string' && /^[A-Za-z0-9._-]{1,128}$/u.test(value)
const validUuid = value => typeof value === 'string' && UUID.test(value)
const validHash = value => typeof value === 'string' && HASH.test(value)
function exact(value, fields) {
  return record(value) && Object.keys(value).length === fields.length
    && fields.every(field => Object.hasOwn(value, field))
}
function validIdentity(identity) {
  return exact(identity, IDENTITY_FIELDS) && identity.publication_kind === 'saved_private'
    && ['clip_id', 'session_id', 'shot_id', 'source_job_id', 'artifact_id', 'table_id', 'device_id',
      'source_kind', 'owner_user_id'].every(key => validId(identity[key]))
    && validManagedSession(identity.session_id)
    && ['content_sha256', 'ownership_receipt_sha256', 'save_receipt_sha256', 'postgame_manifest_sha256']
      .every(key => validHash(identity[key]))
    && Number.isSafeInteger(identity.byte_size) && identity.byte_size > 0
    && identity.content_type === 'video/mp4'
}
// Positive provenance only: publication, root/inventory completeness and provider
// state remain the caller's separate same-transaction responsibilities.
async function readManagedMediaBirth(transaction, { sessionId, tableId, deviceId }) {
  if (!validManagedSession(sessionId) || !validScopeId(tableId) || !validScopeId(deviceId)) fail()
  let birth
  try {
    birth = (await transaction.collection('managed_session_births').doc(sessionId).get()).data
  } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error?.message || '')) return null
    fail()
  }
  if (birth === null || birth === undefined) return null
  if (!validManagedBirth(birth)) fail()
  if (birth.session_id !== sessionId || birth.table_id !== tableId || birth.device_id !== deviceId) fail('CONFLICT')
  return Object.hasOwn(birth, 'media_admission_contract') ? birth : null
}
function validStorageAdmission(marker, identity, epoch) {
  return exact(marker, MARKER_FIELDS) && validIdentity(identity) && validUuid(epoch)
    && marker.contract === MANAGED_MEDIA_ADMISSION_CONTRACT && marker.storage_contract === STORAGE_AUTHORITY_CONTRACT
    && validHash(marker.identity_sha256) && marker.identity_sha256 === digest(identity)
    && validUuid(marker.epoch) && marker.epoch === epoch
}
function makeStorageAdmission(identity, root) {
  if (!validIdentity(identity) || !record(root) || root.contract !== STORAGE_AUTHORITY_CONTRACT
      || !validUuid(root.epoch) || !validHash(root.identity_sha256) || !validIdentity(root.identity)) fail()
  const identityHash = digest(identity)
  if (root.identity_sha256 !== identityHash || digest(root.identity) !== identityHash) fail('CONFLICT')
  return { contract: MANAGED_MEDIA_ADMISSION_CONTRACT, storage_contract: STORAGE_AUTHORITY_CONTRACT,
    identity_sha256: identityHash, epoch: root.epoch }
}
module.exports = { ManagedMediaAdmissionError, readManagedMediaBirth, makeStorageAdmission,
  validStorageAdmission, STORAGE_AUTHORITY_CONTRACT }
