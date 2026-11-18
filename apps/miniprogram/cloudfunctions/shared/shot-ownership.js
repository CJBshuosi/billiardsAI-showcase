const crypto = require('crypto')
const { isSavedUntilDeletedClip } = require('./retention-policy')

const RECEIPT_FIELDS = [
  'session_id', 'shot_id', 'sequence', 'actor_user_id', 'actor_role', 'match_revision',
  'source_artifact_id', 'source_mainline1_artifact_id', 'source_started_ms',
  'recorded_at_utc', 'evidence_status', 'source_kind', 'revision', 'supersedes_revision',
  'source_start_event_id', 'evidence_reason', 'source_epoch_id',
]
const ID = /^[A-Za-z0-9._-]{1,128}$/u
const HASH = /^[0-9a-f]{64}$/u
function canonicalJson(value) {
  const canonical = item => Array.isArray(item) ? item.map(canonical)
    : item && typeof item === 'object'
      ? Object.fromEntries(Object.keys(item).sort().map(key => [key, canonical(item[key])])) : item
  return JSON.stringify(canonical(value))
}
function digest(value) {
  return crypto.createHash('sha256').update(canonicalJson(value), 'utf8').digest('hex')
}
function documentId(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}
function validOwnership(value) {
  const r = value && value.ownership_receipt
  return Boolean(r && typeof r === 'object' && !Array.isArray(r)
    && Object.keys(r).length === RECEIPT_FIELDS.length
    && Object.keys(r).every(key => RECEIPT_FIELDS.includes(key))
    && ID.test(value.session_id || '') && ID.test(value.shot_id || '')
    && ID.test(value.owner_user_id || '') && ['red', 'blue'].includes(value.actor_role)
    && HASH.test(value.ownership_receipt_sha256 || '')
    && r.session_id === value.session_id && r.shot_id === value.shot_id
    && r.actor_user_id === value.owner_user_id && r.actor_role === value.actor_role
    && Number.isSafeInteger(r.sequence) && r.sequence >= 1
    && Number.isSafeInteger(r.match_revision) && r.match_revision >= 0
    && Number.isSafeInteger(r.source_started_ms) && r.source_started_ms >= 0
    && r.revision === 1 && r.supersedes_revision === null
    && r.evidence_status === 'bound_at_start' && r.source_kind === 'usb_live'
    && typeof r.source_artifact_id === 'string' && r.source_artifact_id.length > 0
    && (r.source_mainline1_artifact_id === null || HASH.test(r.source_mainline1_artifact_id || ''))
    && typeof r.source_epoch_id === 'string' && r.source_epoch_id.length > 0
    && typeof r.source_start_event_id === 'string' && r.source_start_event_id.length > 0
    && (r.evidence_reason === null || (typeof r.evidence_reason === 'string' && r.evidence_reason.length > 0))
    && typeof r.recorded_at_utc === 'string' && r.recorded_at_utc.endsWith('Z')
    && Number.isFinite(Date.parse(r.recorded_at_utc))
    && Object.values(r).filter(item => typeof item === 'string').every(item =>
      item.trim().length > 0 && !/[\u0000-\u001f]/u.test(item)
      && Buffer.from(item, 'utf8').toString('utf8') === item)
    && digest(r) === value.ownership_receipt_sha256)
}
function saveReceipt(descriptor) {
  return {
    schema_version: '2.0', request_id: descriptor.save_request_id,
    session_id: descriptor.session_id, shot_id: descriptor.shot_id,
    postgame_manifest_sha256: descriptor.postgame_manifest_sha256, status: 'SAVED',
    clip_id: descriptor.clip_id, owner_user_id: descriptor.owner_user_id,
    actor_role: descriptor.actor_role, ownership_receipt_sha256: descriptor.ownership_receipt_sha256,
  }
}
function validSavedSelection(descriptor) {
  return ID.test(descriptor.save_request_id || '')
    && HASH.test(descriptor.postgame_manifest_sha256 || '')
    && HASH.test(descriptor.save_receipt_sha256 || '')
    && [saveReceipt(descriptor), { ...saveReceipt(descriptor), status: 'SAVE_PENDING', clip_id: null }]
      .some(receipt => descriptor.save_receipt_sha256 === digest(receipt))
}
function ownershipProjection(descriptor, tableId, deviceId) {
  return {
    session_id: descriptor.session_id, shot_id: descriptor.shot_id,
    owner_user_id: descriptor.owner_user_id, actor_role: descriptor.actor_role,
    ownership_receipt: descriptor.ownership_receipt,
    ownership_receipt_sha256: descriptor.ownership_receipt_sha256,
    table_id: tableId, device_id: deviceId,
  }
}
function matchesProjection(clip, projection) {
  if (!validOwnership(clip) || !projection
      || !ID.test(clip.table_id || '') || !ID.test(clip._device_id || '')) return false
  const expected = ownershipProjection(clip, clip.table_id, clip._device_id)
  if (Object.keys(projection).filter(key => key !== '_id').length !== Object.keys(expected).length) return false
  return Object.entries(expected).every(([key, value]) => canonicalJson(value) === canonicalJson(projection[key]))
}
function isActivePrivateSavedClip(clip) {
  return Boolean(clip && clip.schema_version === '2.0'
    && clip.publication_kind === 'saved_postgame_clip'
    && clip.privacy_policy === 'shot_owner_only_v1' && clip.source_kind === 'usb_live'
    && clip.upload_status === 'published' && clip.consent_status === 'owner'
    && !clip.deleted_at && clip.retention_state === 'active'
    && typeof clip.storage_file_id === 'string' && clip.storage_file_id.startsWith('cloud://')
    && validOwnership(clip) && validSavedSelection(clip) && isSavedUntilDeletedClip(clip))
}
// Deletion authority is immutable; losing viewer eligibility never changes ownership.
function isImmutablePrivateClipOwner(clip, userId, projection) {
  return Boolean(clip && clip.schema_version === '2.0'
    && clip.publication_kind === 'saved_postgame_clip'
    && clip.privacy_policy === 'shot_owner_only_v1' && clip.source_kind === 'usb_live'
    && clip.owner_user_id === userId && validSavedSelection(clip)
    && matchesProjection(clip, projection))
}
function isPrivateClipOwner(clip, userId, projection) {
  return isActivePrivateSavedClip(clip) && isImmutablePrivateClipOwner(clip, userId, projection)
}
module.exports = { isImmutablePrivateClipOwner, isActivePrivateSavedClip, isPrivateClipOwner, canonicalJson, digest, documentId, validOwnership, saveReceipt, validSavedSelection, ownershipProjection, matchesProjection }
