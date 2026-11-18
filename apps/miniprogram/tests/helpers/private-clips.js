const crypto = require('crypto')
const fixture = require('../../../../tests/fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json')
const digest = value => crypto.createHash('sha256').update(value).digest('hex')
const historyId = (revision = 1) => digest(`session-1:red:${revision}`)
function ownershipSeed() {
  return {
    session_role_binding_history: { [historyId()]: structuredClone(fixture.role_binding_history[0]) },
    field_session_commands: { 'claim-red-1': {
      command_id: 'claim-red-1', command_type: 'claim_role', actor_user_id: 'user-a',
      session_id: 'session-1', table_id: 'table-1', claimed_by_device_id: 'device-1',
      validated_payload: { role: 'red' }, status: 'applied', resulting_revision: 1,
    } },
  }
}
function privateClip(change = {}) {
  const value = {
    ...JSON.parse(JSON.stringify(fixture.descriptor)),
    table_id: 'table-1', _device_id: 'device-1',
    upload_status: 'published', privacy_policy: 'shot_owner_only_v1', consent_status: 'owner',
    deleted_at: null, retention_state: 'active', storage_file_id: 'cloud://private/clip.mp4',
    retention_policy: 'saved_until_deleted_v1', ordinary_expires_at: null, retention_expires_at: null,
    ...change,
  }
  value.ownership_receipt.session_id = value.session_id
  value.ownership_receipt.shot_id = value.shot_id
  const { digest, saveReceipt } = require('../../cloudfunctions/shared/shot-ownership')
  value.ownership_receipt_sha256 = digest(value.ownership_receipt)
  value.save_receipt_sha256 = digest(saveReceipt(value))
  return value
}
module.exports = { historyId, ownershipSeed, privateClip }
