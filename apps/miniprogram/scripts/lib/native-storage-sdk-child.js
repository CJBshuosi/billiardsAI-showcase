const { types: { isProxy } } = require('util')
const { observeStorageGrant, PINNED_STORAGE_SDK_VERSION } = require('./storage-grant-observation')
const INPUT_LIMIT = 64 * 1024
const INPUT_FIELDS = ['requestId', 'cloudPath', 'environmentId', 'observedAt', 'cliVersion', 'credentials']
const CREDENTIAL_FIELDS = ['secretId', 'secretKey', 'token', 'expiredAt', 'isTemporary', 'envId']

class SafeChildError extends Error {
  constructor(reason) { super(reason); this.reason = reason }
}
function reject(reason = 'argument_invalid') { throw new SafeChildError(reason) }
function exact(value, keys) {
  if (isProxy(value) || !value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) reject()
  const actual = Reflect.ownKeys(value)
  if (actual.length !== keys.length || actual.some(key => !keys.includes(key))) reject()
  const copy = Object.create(null)
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key)
    if (!descriptor || !Object.hasOwn(descriptor, 'value') || !descriptor.enumerable) reject()
    copy[key] = descriptor.value
  }
  return copy
}
function string(value, maximum = 16384) {
  return typeof value === 'string' && value.length > 0 && Buffer.byteLength(value, 'utf8') <= maximum &&
    !/[\u0000-\u001f\u007f-\u009f]/u.test(value) && Buffer.from(value, 'utf8').toString('utf8') === value
}
function validatedInput(input) {
  const value = exact(input, INPUT_FIELDS)
  const credentials = exact(value.credentials, CREDENTIAL_FIELDS)
  if (!string(value.requestId, 36) || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u.test(value.requestId) ||
      value.cloudPath !== `billiards-native-probes-v1/${value.requestId}/grant.mp4` ||
      !string(value.environmentId, 128) || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(value.environmentId) ||
      !string(value.observedAt, 24) || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u.test(value.observedAt) ||
      new Date(value.observedAt).toISOString() !== value.observedAt ||
      !string(value.cliVersion, 64) || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.test(value.cliVersion) ||
      value.cliVersion.split('.').some(part => !Number.isSafeInteger(Number(part))) ||
      credentials.isTemporary !== true || credentials.envId !== value.environmentId ||
      ['secretId', 'secretKey', 'token', 'expiredAt'].some(key => !string(credentials[key]))) reject()
  value.credentials = credentials
  if (Buffer.byteLength(JSON.stringify(value), 'utf8') > INPUT_LIMIT) reject()
  return value
}
function installedSdk() {
  return { sdk: require('@cloudbase/node-sdk'), version: require('@cloudbase/node-sdk/package.json').version }
}
async function runSdkObservation(input, { loadSdk = installedSdk } = {}) {
  let value
  try { value = validatedInput(input) } catch (_) { reject() }
  let loaded
  try {
    loaded = await loadSdk()
    if (!loaded || loaded.version !== PINNED_STORAGE_SDK_VERSION || typeof loaded.sdk.init !== 'function') reject('sdk_unavailable')
  } catch (_) { reject('sdk_unavailable') }
  try {
    const { secretId, secretKey, token } = value.credentials
    const app = loaded.sdk.init({ env: value.environmentId, secretId, secretKey, sessionToken: token, timeout: 15000, debug: false })
    const metadata = await app.getUploadMetadata({ cloudPath: value.cloudPath })
    return observeStorageGrant({ metadata, requestId: value.requestId, cloudPath: value.cloudPath,
      environmentId: value.environmentId, observedAt: value.observedAt, cliVersion: value.cliVersion })
  } catch (_) { reject('metadata_unavailable') }
}
async function main() {
  // Keep reading to EOF after exceeding the limit, without retaining excess bytes.
  // There are no SDK imports or provider calls before complete bounded validation.
  let bytes = 0
  const chunks = []
  try {
    for await (const chunk of process.stdin) {
      bytes += chunk.length
      if (bytes <= INPUT_LIMIT) chunks.push(chunk)
    }
    if (bytes > INPUT_LIMIT) reject()
    const report = await runSdkObservation(JSON.parse(Buffer.concat(chunks).toString('utf8')))
    process.stdout.write(`BILLIARDS_NATIVE_STORAGE_REPORT:${JSON.stringify(report)}\n`)
  } catch (error) {
    const reason = error instanceof SafeChildError ? error.reason : 'argument_invalid'
    process.stdout.write(`BILLIARDS_NATIVE_STORAGE_ERROR:${reason}\n`)
    process.exitCode = 2
  }
}
if (require.main === module) main()
module.exports = { runSdkObservation }
