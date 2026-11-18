const { digest, documentId } = require('./shot-ownership')
// Pure managed authority validation and deterministic document reads only.
const BIRTH_CONTRACT = 'managed_session_birth_v1'
const MANAGED_MEDIA_ADMISSION_CONTRACT = 'managed_clip_admission_v1'
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu
const isUuid = value => typeof value === 'string' && UUID_V4.test(value)
const isScopeId = value => typeof value === 'string' && /^[A-Za-z0-9._-]{1,128}$/u.test(value)
const isTime = value => typeof value === 'string' && !Number.isNaN(Date.parse(value))
  && new Date(value).toISOString() === value
const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value)
const nonnegative = value => Number.isSafeInteger(value) && value >= 0
const positive = value => Number.isSafeInteger(value) && value > 0
const nullableId = value => value === null || isScopeId(value)
const nullableRole = value => value === null || ['red', 'blue'].includes(value)
function exactKeys(value, keys, document = false) {
  return isRecord(value) && (!document || !Object.hasOwn(value, '_id') || typeof value._id === 'string')
    && Object.keys(value).filter(key => !(document && key === '_id')).length === keys.length
    && keys.every(key => Object.hasOwn(value, key))
}
function isManagedSessionId(sessionId) {
  return typeof sessionId === 'string' && sessionId.startsWith('managed-')
}
function validIdentity(value) {
  return isRecord(value) && isUuid(value.birth_id) && value.session_id === `managed-${value.birth_id}`
    && isScopeId(value.table_id) && isScopeId(value.device_id)
}
function sameManagedIdentity(left, right) {
  return isRecord(left) && isRecord(right)
    && ['session_id', 'birth_id', 'table_id', 'device_id'].every(key => left[key] === right[key])
}
function validManagedBirth(value) {
  const marked = isRecord(value) && Object.hasOwn(value, 'media_admission_contract')
  return exactKeys(value, ['contract', 'session_id', 'birth_id', 'table_id', 'device_id', 'opened_at',
    ...(marked ? ['media_admission_contract'] : [])], true)
    && value.contract === BIRTH_CONTRACT && validIdentity(value) && isTime(value.opened_at)
    && (!marked || value.media_admission_contract === MANAGED_MEDIA_ADMISSION_CONTRACT)
}
function validManagedHead(value) {
  if (!exactKeys(value, ['contract', 'session_id', 'birth_id', 'table_id', 'device_id',
    'contiguous_ack_sequence', 'last_resulting_revision', 'last_applied_server_sequence',
    'role_tails', 'coverage', 'seal', 'revision', 'issued_next', 'closed_target'], true)
      || value.contract !== 'managed_command_authority_v1' || !validIdentity(value)
      || !positive(value.issued_next) || !positive(value.revision)
      || !nonnegative(value.contiguous_ack_sequence) || !nonnegative(value.last_resulting_revision)
      || !nonnegative(value.last_applied_server_sequence)
      || value.contiguous_ack_sequence >= value.issued_next
      || value.last_applied_server_sequence > value.contiguous_ack_sequence
      || !exactKeys(value.role_tails, ['red', 'blue'])
      || !Object.values(value.role_tails).every(tail => tail === null || (typeof tail === 'string' && /^[a-f0-9]{64}$/u.test(tail)))) return false
  if (value.contiguous_ack_sequence === 0 && (value.last_resulting_revision !== 0
      || value.last_applied_server_sequence !== 0 || Object.values(value.role_tails).some(tail => tail !== null))) return false
  if (value.coverage === 'open') return value.closed_target === null && value.seal === null
  if (!['closing', 'sealed'].includes(value.coverage)
      || !nonnegative(value.closed_target) || value.closed_target !== value.issued_next - 1) return false
  if (value.coverage === 'closing') return value.seal === null
  return value.contiguous_ack_sequence === value.closed_target
    && exactKeys(value.seal, ['sealed_at']) && isTime(value.seal.sealed_at)
}
const LIVE_REQUIRED = ['schema_version', 'session_id', 'table_id', 'device_id', 'status',
  'opened_at', 'updated_at', 'environment', 'birth_contract', 'birth_id']
const LIVE_OPTIONAL = {
  control_revision: nonnegative,
  red_user_id: nullableId, blue_user_id: nullableId, active_user_id: nullableId,
  breaker: nullableRole, started_at: value => value === null || isTime(value),
  mode: value => value === null || ['observation', 'assistance'].includes(value),
  format: value => value === null || ['single', 'race'].includes(value),
  race_target: value => value === null || (Number.isSafeInteger(value) && value >= 2 && value <= 99),
  ranking_intent: value => typeof value === 'boolean', settings_revision: nonnegative,
  red_group: value => value === null || ['solid', 'stripe'].includes(value),
  blue_group: value => value === null || ['solid', 'stripe'].includes(value),
}
function validManagedLive(value) {
  if (!isRecord(value)) return false
  const ended = value.status === 'ended'
  const keys = [...LIVE_REQUIRED, ...(ended ? ['ended_at'] : []),
    ...Object.keys(LIVE_OPTIONAL).filter(key => Object.hasOwn(value, key))]
  if (!exactKeys(value, keys, true) || value.schema_version !== '1.0' || !validIdentity(value)
      || value.birth_contract !== BIRTH_CONTRACT || !['open', 'ended'].includes(value.status)
      || typeof value.environment !== 'string' || !value.environment.length
      || !isTime(value.opened_at) || !isTime(value.updated_at) || (ended && !isTime(value.ended_at))
      || !Object.entries(LIVE_OPTIONAL).every(([key, valid]) => !Object.hasOwn(value, key) || valid(value[key]))) return false
  if (value.red_user_id && value.red_user_id === value.blue_user_id) return false
  if (value.active_user_id && ![value.red_user_id, value.blue_user_id].includes(value.active_user_id)) return false
  if (Object.hasOwn(value, 'red_group') || Object.hasOwn(value, 'blue_group')) {
    if (!Object.hasOwn(value, 'red_group') || !Object.hasOwn(value, 'blue_group')) return false
    if (value.red_group === null ? value.blue_group !== null
      : value.blue_group !== (value.red_group === 'solid' ? 'stripe' : 'solid')) return false
  }
  const settingsKeys = ['mode', 'format', 'race_target', 'ranking_intent', 'settings_revision']
  if (settingsKeys.some(key => Object.hasOwn(value, key))) {
    if (!settingsKeys.every(key => Object.hasOwn(value, key))) return false
    if (value.mode === null || value.format === null) {
      if (value.mode !== null || value.format !== null || value.race_target !== null
          || value.ranking_intent !== false || value.settings_revision !== 0) return false
    } else if (value.settings_revision < 1 || (value.mode === 'assistance' && value.ranking_intent)
        || (value.format === 'single' ? value.race_target !== null : value.race_target === null)) return false
  }
  return true
}
async function optional(ref) {
  try { return (await ref.get()).data || null } catch (error) {
    if (/document with _id .+ does not exist/iu.test(String(error && error.message))) return null
    throw error
  }
}
async function readManagedCommandContext(tx, { sessionId, tableId = null, deviceId = null }) {
  if (!isManagedSessionId(sessionId) || !isScopeId(sessionId)) return null
  const birth = await optional(tx.collection('managed_session_births').doc(sessionId))
  const head = await optional(tx.collection('session_command_authority').doc(sessionId))
  const live = await optional(tx.collection('field_live_sessions').doc(sessionId))
  if (!validManagedBirth(birth) || !validManagedHead(head) || !validManagedLive(live)
      || birth.session_id !== sessionId || !sameManagedIdentity(birth, head)
      || !sameManagedIdentity(birth, live) || live.opened_at !== birth.opened_at
      || (tableId !== null && birth.table_id !== tableId)
      || (deviceId !== null && birth.device_id !== deviceId)) return null
  return { birth, head, live }
}
const isHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/u.test(value)
const COMMAND_TYPES = new Set(['claim_role', 'release_role', 'set_breaker', 'assign_groups',
  'continue_turn', 'switch_turn', 'end_session'])
function validManagedCommand(command, birth, head) {
  return isRecord(command) && command.schema_version === '1.0'
    && command.session_id === birth.session_id && command.table_id === birth.table_id
    && isScopeId(command.actor_user_id) && COMMAND_TYPES.has(command.command_type)
    && isHash(command.idempotency_digest) && isHash(command.submitted_intent_sha256)
    && command.command_id === `command-${command.idempotency_digest.slice(0, 48)}`
    && nonnegative(command.expected_revision) && isRecord(command.validated_payload)
    && command.intent_digest === digest({ command_type: command.command_type,
      expected_revision: command.expected_revision, validated_payload: command.validated_payload })
    && positive(command.server_sequence) && command.server_sequence < head.issued_next
    && (head.closed_target === null || command.server_sequence <= head.closed_target)
    && ['pending', 'claimed', 'applied', 'rejected'].includes(command.status)
}
function managedRoleHistoryId(node) {
  return documentId(`${node.session_id}:${node.role}:${node.resulting_revision}`)
}
function validManagedRoleLink(node, id, birth, head, role) {
  return exactKeys(node, ['contract', 'session_id', 'role', 'user_id', 'resulting_revision',
    'command_id', 'table_id', 'device_id', 'birth_id', 'server_sequence', 'previous_id', 'next_id'], true)
    && node.contract === 'managed_role_link_v1' && sameManagedIdentity(node, birth)
    && ['red', 'blue'].includes(node.role) && node.role === role && nullableId(node.user_id)
    && positive(node.resulting_revision) && node.resulting_revision <= head.last_resulting_revision
    && positive(node.server_sequence) && node.server_sequence <= head.contiguous_ack_sequence
    && node.server_sequence <= head.last_applied_server_sequence
    && isScopeId(node.command_id) && managedRoleHistoryId(node) === id
    && [node.previous_id, node.next_id].every(link => link === null || (isHash(link) && link !== id))
    && (node.next_id === null ? head.role_tails[role] === id
      : head.role_tails[role] !== null && head.role_tails[role] !== id)
}
async function verifyManagedShotOwnership(tx, { receipt, job, candidateHistoryId }) {
  if (!isRecord(receipt) || !isRecord(job) || !isManagedSessionId(job.session_id)
      || receipt.session_id !== job.session_id || !['red', 'blue'].includes(receipt.actor_role)
      || !isScopeId(receipt.actor_user_id) || !nonnegative(receipt.match_revision)
      || !isHash(candidateHistoryId)) return false
  // Deliberately do not read mutable live state: sealed permanent authority is the proof.
  const birth = await optional(tx.collection('managed_session_births').doc(job.session_id))
  const head = await optional(tx.collection('session_command_authority').doc(job.session_id))
  if (!validManagedBirth(birth) || !validManagedHead(head) || !sameManagedIdentity(birth, head)
      || birth.session_id !== job.session_id || birth.table_id !== job.table_id
      || birth.device_id !== job._device_id || head.coverage !== 'sealed') return false
  const node = await optional(tx.collection('session_role_binding_history').doc(candidateHistoryId))
  if (!validManagedRoleLink(node, candidateHistoryId, birth, head, receipt.actor_role)
      || node.user_id !== receipt.actor_user_id || node.resulting_revision > receipt.match_revision) return false
  const command = await optional(tx.collection('field_session_commands').doc(node.command_id))
  if (!validManagedCommand(command, birth, head) || command.command_id !== node.command_id
      || command.status !== 'applied' || command.public_result_code !== 'APPLIED'
      || command.command_type !== 'claim_role' || command.actor_user_id !== node.user_id
      || command.claimed_by_device_id !== birth.device_id || command.applied_role !== node.role
      || command.validated_payload.role !== node.role || Object.keys(command.validated_payload).length !== 1
      || !Object.hasOwn(command, 'applied_previous_history_id')
      || command.applied_previous_history_id !== node.previous_id
      || command.server_sequence !== node.server_sequence
      || command.resulting_revision !== node.resulting_revision) return false
  if (node.next_id === null) return head.role_tails[node.role] === candidateHistoryId
  const next = await optional(tx.collection('session_role_binding_history').doc(node.next_id))
  return validManagedRoleLink(next, node.next_id, birth, head, node.role)
    && next.previous_id === candidateHistoryId && next.server_sequence > node.server_sequence
    && next.resulting_revision > receipt.match_revision
}
module.exports = { MANAGED_MEDIA_ADMISSION_CONTRACT, isManagedSessionId, validManagedBirth, validManagedHead, validManagedLive,
  sameManagedIdentity, readManagedCommandContext, validManagedCommand, managedRoleHistoryId,
  validManagedRoleLink, verifyManagedShotOwnership }
