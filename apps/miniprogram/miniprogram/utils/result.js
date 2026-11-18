const STABLE_ERROR_CODES = new Set([
  'AUTH_REQUIRED',
  'ACTION_UNSUPPORTED',
  'COMMAND_FORBIDDEN',
  'COMMAND_IDEMPOTENCY_CONFLICT',
  'COMMAND_INVALID',
  'COMMAND_NOT_FOUND',
  'COMMAND_STALE_REVISION',
  'ENVIRONMENT_MISMATCH',
  'FORBIDDEN',
  'IDENTITY_CONFLICT',
  'GROUP_CORRECTION_REQUIRED',
  'INTERNAL_ERROR',
  'JOIN_RATE_LIMITED',
  'MEDIA_NOT_AVAILABLE',
  'NETWORK_UNAVAILABLE',
  'PROFILE_INVALID',
  'REQUEST_INVALID',
  'ROLE_CONFLICT',
  'SESSION_NOT_JOINABLE',
  'SYNC_CONFLICT',
  'SYNC_PARTIAL',
  'SYNC_PENDING',
  'TABLE_CODE_DISABLED',
  'TABLE_CODE_INVALID',
])

function normalizeResult(value) {
  if (value && value.ok === true && Object.hasOwn(value, 'data')) {
    return { ok: true, data: value.data }
  }
  const candidate = value && value.ok === false && value.error
    ? value.error.code
    : null
  const code = STABLE_ERROR_CODES.has(candidate) ? candidate : 'INTERNAL_ERROR'
  return { ok: false, error: { code } }
}

module.exports = { normalizeResult }
