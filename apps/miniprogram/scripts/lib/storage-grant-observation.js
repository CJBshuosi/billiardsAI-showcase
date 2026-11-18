const { createHash } = require('crypto')
const { types: { isProxy } } = require('util')

const NATIVE_STORAGE_OBSERVATION_CONTRACT = 'native_storage_grant_observation_v1'
const PINNED_STORAGE_SDK_VERSION = '3.17.2'
const INVALID = 'NATIVE_STORAGE_METADATA_INVALID'
const CORE_FIELDS = ['url', 'token', 'authorization', 'fileId', 'cosFileId']
const METADATA_FIELDS = [...CORE_FIELDS, 'download_url']
const CONTEXT_FIELDS = ['requestId', 'cloudPath', 'environmentId', 'observedAt', 'cliVersion']
const REPORT_FIELDS = [
  'contract', 'request_id', 'environment_id_sha256', 'cloud_path_sha256', 'file_id_sha256',
  'observed_at', 'sdk_version', 'cli_version', 'authorization_format', 'endpoint_family',
  'metadata_fields', 'signature_window', 'observation_only', 'closure_evidence', 'erasable',
]
const WINDOW_FIELDS = ['starts_at_seconds', 'ends_at_seconds', 'key_starts_at_seconds', 'key_ends_at_seconds']
const V5_FIELDS = [
  'q-sign-algorithm', 'q-ak', 'q-sign-time', 'q-key-time', 'q-header-list', 'q-url-param-list', 'q-signature',
]
const FORMATS = ['cos_v5_header', 'malformed_cos_v5_header', 'base64_opaque', 'opaque']
const FAMILIES = ['cos', 'cloudbase', 'other_https']

class StorageGrantObservationError extends Error {
  constructor() {
    super(INVALID)
    this.name = 'StorageGrantObservationError'
    this.code = INVALID
  }
}

function invalid() { throw new StorageGrantObservationError() }

// Read only own data descriptors: neither accessors nor coercion hooks are invoked.
function record(value) {
  // Descriptor snapshots cannot constrain later Proxy serialization/get traps.
  if (isProxy(value) || !value || typeof value !== 'object' || Array.isArray(value)) invalid()
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== null && prototype !== Object.prototype) invalid()
  return value
}

function ownValue(value, key, enumerable = false) {
  const descriptor = Object.getOwnPropertyDescriptor(value, key)
  if (!descriptor || !Object.prototype.hasOwnProperty.call(descriptor, 'value') ||
      (enumerable && descriptor.enumerable !== true)) invalid()
  return descriptor.value
}

function exactRecord(value, keys) {
  record(value)
  const actual = Reflect.ownKeys(value)
  if (actual.length !== keys.length || actual.some(key => !keys.includes(key))) invalid()
  const result = Object.create(null)
  for (const key of keys) result[key] = ownValue(value, key, true)
  return result
}

function safeString(value, maximum = 16384) {
  if (typeof value !== 'string' || !value.length || value.length > maximum ||
      /[\u0000-\u001f\u007f-\u009f]/u.test(value)) invalid()
  // Reject unpaired UTF16 code units before UTF8 hashing or byte counting.
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index)
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const low = value.charCodeAt(index + 1)
      if (!(low >= 0xdc00 && low <= 0xdfff)) invalid()
      index += 1
    } else if (unit >= 0xdc00 && unit <= 0xdfff) invalid()
  }
  if (Buffer.byteLength(value, 'utf8') > maximum) invalid()
  return value
}

function contextValues(input) {
  record(input)
  const context = Object.create(null)
  for (const key of CONTEXT_FIELDS) context[key] = safeString(ownValue(input, key))
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u.test(context.requestId) ||
      context.cloudPath !== `billiards-native-probes-v1/${context.requestId}/grant.mp4` ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(context.environmentId)) invalid()
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u.test(context.observedAt) ||
      new Date(context.observedAt).toISOString() !== context.observedAt) invalid()
  if (context.cliVersion.length > 64 || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.test(context.cliVersion) ||
      context.cliVersion.split('.').some(part => !Number.isSafeInteger(Number(part)))) invalid()
  return context
}

function sha256(value) { return createHash('sha256').update(value, 'utf8').digest('hex') }

function endpointFamily(value) {
  // WHATWG URL parsing is local only. Explicitly disallow raw fragment/userinfo delimiters.
  if (!/^https:\/\//u.test(value) || /\s/u.test(value) || value.includes('#') ||
      value.slice(8).split(/[/?]/u, 1)[0].includes('@')) invalid()
  const parsed = new URL(value)
  if (parsed.protocol !== 'https:' || !parsed.hostname || parsed.username || parsed.password || parsed.hash) invalid()
  const host = parsed.hostname
  const label = '[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?'
  if (new RegExp(`^(?:${label}\\.)?cos\\.${label}\\.myqcloud\\.com$`, 'u').test(host)) return 'cos'
  if (host === 'tcb.qcloud.la' || host.endsWith('.tcb.qcloud.la') ||
      host === 'tcb-api.tencentcloudapi.com' || host.endsWith('.tcb-api.tencentcloudapi.com')) return 'cloudbase'
  return 'other_https'
}

function safeSeconds(value) {
  return Number.isSafeInteger(value) && value > 0 && value <= 8640000000000
}

function interval(value) {
  if (!/^[1-9]\d*;[1-9]\d*$/u.test(value)) return null
  const [start, end] = value.split(';').map(Number)
  return safeSeconds(start) && safeSeconds(end) && start < end ? [start, end] : null
}

function canonicalNames(value) {
  if (value === '') return true
  if (!/^[a-z0-9-]+(?:;[a-z0-9-]+)*$/u.test(value)) return false
  const names = value.split(';')
  return names.every((name, index) => index === 0 || names[index - 1] < name)
}

function v5Window(authorization) {
  const pairs = authorization.split('&')
  if (pairs.length !== V5_FIELDS.length) return null
  const fields = Object.create(null)
  for (const pair of pairs) {
    const separator = pair.indexOf('=')
    if (separator < 0 || pair.indexOf('=', separator + 1) >= 0) return null
    const key = pair.slice(0, separator)
    if (!V5_FIELDS.includes(key) || Object.prototype.hasOwnProperty.call(fields, key)) return null
    fields[key] = pair.slice(separator + 1)
  }
  if (fields['q-sign-algorithm'] !== 'sha1' || !/^[A-Za-z0-9_-]+$/u.test(fields['q-ak']) ||
      !/^[a-f0-9]{40}$/u.test(fields['q-signature']) ||
      !canonicalNames(fields['q-header-list']) || !canonicalNames(fields['q-url-param-list'])) return null
  const sign = interval(fields['q-sign-time'])
  const key = interval(fields['q-key-time'])
  if (!sign || !key) return null
  return { starts_at_seconds: sign[0], ends_at_seconds: sign[1], key_starts_at_seconds: key[0], key_ends_at_seconds: key[1] }
}

function canonicalBase64(value) {
  if (value.length % 4 !== 0 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u.test(value)) return false
  // Check unused padding bits directly. Never decode potentially secret plaintext.
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  if (value.endsWith('==')) return alphabet.indexOf(value[value.length - 3]) % 16 === 0
  if (value.endsWith('=')) return alphabet.indexOf(value[value.length - 2]) % 4 === 0
  return true
}

function authorizationObservation(value) {
  if (/(?:^|&)q-sign/u.test(value)) {
    const window = v5Window(value)
    return { format: window ? 'cos_v5_header' : 'malformed_cos_v5_header', window }
  }
  return { format: canonicalBase64(value) ? 'base64_opaque' : 'opaque', window: null }
}

function observeStorageGrant(input) {
  try {
    const context = contextValues(input)
    const metadata = record(ownValue(input, 'metadata'))
    const data = record(ownValue(metadata, 'data'))
    const fields = Object.create(null)
    for (const field of CORE_FIELDS) fields[field] = safeString(ownValue(data, field), field === 'fileId' ? 2048 : 16384)
    if (!fields.fileId.startsWith('cloud://') || fields.fileId.length <= 8) invalid()
    const endpoint = endpointFamily(fields.url)
    const authorization = authorizationObservation(fields.authorization)
    return {
      contract: NATIVE_STORAGE_OBSERVATION_CONTRACT,
      request_id: context.requestId,
      environment_id_sha256: sha256(context.environmentId),
      cloud_path_sha256: sha256(context.cloudPath),
      file_id_sha256: sha256(fields.fileId),
      observed_at: context.observedAt,
      sdk_version: PINNED_STORAGE_SDK_VERSION,
      cli_version: context.cliVersion,
      authorization_format: authorization.format,
      endpoint_family: endpoint,
      metadata_fields: { url: true, token: true, authorization: true, fileId: true, cosFileId: true,
        download_url: !!Object.getOwnPropertyDescriptor(data, 'download_url') },
      signature_window: authorization.window,
      observation_only: true,
      closure_evidence: false,
      erasable: false,
    }
  } catch (_) {
    // Parser, proxy and native exceptions all collapse to one fixed public error.
    throw new StorageGrantObservationError()
  }
}

function validStorageGrantObservation(report, input) {
  try {
    const context = contextValues(input)
    const fields = exactRecord(report, REPORT_FIELDS)
    if (fields.contract !== NATIVE_STORAGE_OBSERVATION_CONTRACT || fields.request_id !== context.requestId ||
        fields.environment_id_sha256 !== sha256(context.environmentId) || fields.cloud_path_sha256 !== sha256(context.cloudPath) ||
        typeof fields.file_id_sha256 !== 'string' || !/^[a-f0-9]{64}$/u.test(fields.file_id_sha256) ||
        fields.observed_at !== context.observedAt || fields.sdk_version !== PINNED_STORAGE_SDK_VERSION ||
        fields.cli_version !== context.cliVersion || !FORMATS.includes(fields.authorization_format) || !FAMILIES.includes(fields.endpoint_family) ||
        fields.observation_only !== true || fields.closure_evidence !== false || fields.erasable !== false) return false
    const metadata = exactRecord(fields.metadata_fields, METADATA_FIELDS)
    if (CORE_FIELDS.some(field => metadata[field] !== true) || typeof metadata.download_url !== 'boolean') return false
    if (fields.authorization_format !== 'cos_v5_header') return fields.signature_window === null
    const window = exactRecord(fields.signature_window, WINDOW_FIELDS)
    return WINDOW_FIELDS.every(field => safeSeconds(window[field])) &&
      window.starts_at_seconds < window.ends_at_seconds && window.key_starts_at_seconds < window.key_ends_at_seconds
  } catch (_) { return false }
}

module.exports = {
  NATIVE_STORAGE_OBSERVATION_CONTRACT,
  PINNED_STORAGE_SDK_VERSION,
  StorageGrantObservationError,
  observeStorageGrant,
  validStorageGrantObservation,
}
