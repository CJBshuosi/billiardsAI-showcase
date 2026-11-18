#!/usr/bin/env node
const fs = require('fs')
const os = require('os')
const path = require('path')
const crypto = require('crypto')
const cp = require('child_process')
const { types: { isProxy } } = require('util')
const { validStorageGrantObservation } = require('./lib/storage-grant-observation')
const CONTRACT = 'native_storage_probe_readiness_v1'
const CAPTURE_LIMIT = 1024 * 1024
const INPUT_LIMIT = 64 * 1024
const TIMEOUT = 20000
const REPORT_MARKER = 'BILLIARDS_NATIVE_STORAGE_REPORT:'
const ERROR_MARKER = 'BILLIARDS_NATIVE_STORAGE_ERROR:'
const CHILD_REASONS = ['argument_invalid', 'sdk_unavailable', 'metadata_unavailable']
const HELP = Object.freeze({ contract: CONTRACT, usage: '--environment development --environment-id VALUE --expected-environment-id VALUE [--execute]', default_action: 'readiness_only', execute_action: 'one_getUploadMetadata', target: 'billiards-native-probes-v1/<fresh-lowercase-UUIDv4>/grant.mp4', observation_only: true, closure_evidence: false, erasable: false })
class SafeFailure extends Error {
  constructor(reason) { super(reason); this.reason = reason }
}
function fail(reason) { throw new SafeFailure(reason) }
function environment(value) { return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(value) }
function uuid(value) { return typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u.test(value) }
function version(value) {
  return typeof value === 'string' && value.length <= 64 && /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.test(value) && value.split('.').every(part => Number.isSafeInteger(Number(part)))
}
function hash(value) { return crypto.createHash('sha256').update(value).digest('hex') }
function parseArguments(argv) {
  if (isProxy(argv) || !Array.isArray(argv)) fail('argument_invalid')
  const argumentsCopy = []
  for (let index = 0; index < argv.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(argv, String(index))
    if (!descriptor || !Object.hasOwn(descriptor, 'value') || typeof descriptor.value !== 'string') fail('argument_invalid')
    argumentsCopy.push(descriptor.value)
  }
  argv = argumentsCopy
  if (argv.length === 1 && argv[0] === '--help') return HELP
  const values = Object.create(null)
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index]
    if (!['--environment', '--environment-id', '--expected-environment-id', '--execute'].includes(flag) || Object.hasOwn(values, flag)) fail('argument_invalid')
    if (flag === '--execute') values[flag] = true
    else {
      const value = argv[++index]
      if (!value || value.startsWith('--')) fail('argument_invalid')
      values[flag] = value
    }
  }
  if (values['--environment'] !== 'development' || !environment(values['--environment-id']) ||
      values['--environment-id'] !== values['--expected-environment-id']) fail('argument_invalid')
  return { environment: values['--environment'], environmentId: values['--environment-id'], expectedEnvironmentId: values['--expected-environment-id'], execute: values['--execute'] === true }
}
function readiness(reason, context = {}) {
  return { contract: CONTRACT, status: reason ? 'not_ready' : 'ready_for_review', reason: reason || null,
    request_id: context.requestId || null, environment_id_sha256: context.environmentId ? hash(context.environmentId) : null,
    cloud_path_sha256: context.cloudPath ? hash(context.cloudPath) : null, cli_version: context.cliVersion || null,
    observation_only: true, closure_evidence: false, erasable: false }
}
// Both pipes are captured, stdin is always ended, and no shell/terminal is involved.
function capturedProcess(file, args, input, cwd) {
  return new Promise(resolve => {
    let child
    let bytes = 0
    let exceeded = false
    try {
      child = cp.execFile(file, args, { cwd, encoding: 'utf8', timeout: TIMEOUT, maxBuffer: CAPTURE_LIMIT, killSignal: 'SIGKILL', shell: false, windowsHide: true }, (error, stdout, stderr) => {
        const exitCode = !error ? 0 : Number.isInteger(error.code) ? error.code : null
        resolve(exceeded ? { exitCode: null, stdout: '', stderr: '' } : { exitCode, stdout, stderr })
      })
      for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
        bytes += Buffer.byteLength(chunk, 'utf8')
        if (bytes > CAPTURE_LIMIT) { exceeded = true; child.kill('SIGKILL') }
      })
      // EPIPE may follow an early child exit; the execFile callback still decides success.
      child.stdin.on('error', () => {})
      child.stdin.end(input)
    } catch (_) { resolve({ exitCode: null, stdout: '', stderr: '' }) }
  })
}
function captured(value) {
  return value && !isProxy(value) && typeof value.stdout === 'string' && typeof value.stderr === 'string' &&
    Buffer.byteLength(value.stdout, 'utf8') + Buffer.byteLength(value.stderr, 'utf8') <= CAPTURE_LIMIT
}
function successful(value) { return captured(value) && value.exitCode === 0 }
function banner(line) {
  // Exact installed/legacy banners only; never skip a diagnostic/error prefix.
  return line === 'CloudBase CLI' || line === 'CloudBase CLI banner' ||
    line === '试试 tcb ai 命令，开启 AI 原生开发' ||
    line === 'Try the tcb ai command to start your AI full-stack development experience' ||
    (line.startsWith('CloudBase CLI ') && version(line.slice('CloudBase CLI '.length)))
}
function lines(stdout) { return stdout.trim().split(/\r?\n/u).map(line => line.trim()) }
function cliVersion(stdout) {
  const candidates = []
  for (const line of lines(stdout)) {
    if (version(line)) candidates.push(line)
    else if (line.startsWith('CloudBase CLI ') && version(line.slice('CloudBase CLI '.length))) candidates.push(line.slice('CloudBase CLI '.length))
    else if (!banner(line)) fail('cli_version_unsupported')
  }
  if (candidates.length !== 1 || Number(candidates[0].split('.')[0]) < 3) fail('cli_version_unsupported')
  return candidates[0]
}
function cliJson(stdout) {
  const values = lines(stdout)
  let index = 0
  while (index < values.length && banner(values[index])) index++
  if (index >= values.length || !/^[{[]/u.test(values[index])) throw Error('invalid')
  return JSON.parse(values.slice(index).join('\n'))
}
function matchingEnvironment(value, expected) {
  let list = value
  if (!Array.isArray(list)) {
    if (!list || typeof list !== 'object' || Object.keys(list).length !== 1) return false
    if (Object.hasOwn(list, 'EnvList')) list = list.EnvList
    else if (Object.hasOwn(list, 'data')) {
      list = list.data
      if (!Array.isArray(list)) {
        if (!list || typeof list !== 'object' || Object.keys(list).length !== 1 || !Object.hasOwn(list, 'EnvList')) return false
        list = list.EnvList
      }
    } else return false
  }
  if (!Array.isArray(list)) return false
  let matches = 0
  for (const row of list) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) return false
    const normalized = Object.hasOwn(row, 'envId')
    const raw = Object.hasOwn(row, 'EnvId')
    if (normalized === raw) return false
    const id = normalized ? row.envId : row.EnvId
    if (!environment(id)) return false
    if (id === expected) matches++
  }
  return matches === 1
}
function credentials(value, expected) {
  // Current official CLI formatter; no inferred envelopes or optional metadata.
  if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 1 && Object.hasOwn(value, 'data')) value = value.data
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('temporary_credentials_unavailable')
  const keys = ['secretId', 'secretKey', 'token', 'expiredAt', 'isTemporary', 'envId']
  if (Object.keys(value).length !== keys.length || Object.keys(value).some(key => !keys.includes(key)) || value.isTemporary !== true || value.envId !== expected ||
      ['secretId', 'secretKey', 'token', 'expiredAt'].some(key => typeof value[key] !== 'string' || !value[key].length || Buffer.byteLength(value[key], 'utf8') > 16384 || /[\u0000-\u001f\u007f-\u009f]/u.test(value[key]) || Buffer.from(value[key], 'utf8').toString('utf8') !== value[key])) fail('temporary_credentials_unavailable')
  return value
}
async function observeInChild(input) {
  const text = JSON.stringify(input)
  if (Buffer.byteLength(text, 'utf8') > INPUT_LIMIT) fail('argument_invalid')
  const result = await capturedProcess(process.execPath, [path.join(__dirname, 'lib/native-storage-sdk-child.js')], text)
  if (!captured(result)) fail('metadata_unavailable')
  const markers = result.stdout.split(/\r?\n/u).filter(line => line.startsWith(REPORT_MARKER) || line.startsWith(ERROR_MARKER))
  if (markers.length !== 1) fail('metadata_unavailable')
  const marker = markers[0]
  if (marker.startsWith(ERROR_MARKER)) {
    const reason = marker.slice(ERROR_MARKER.length)
    if (result.exitCode === 2 && CHILD_REASONS.includes(reason)) fail(reason)
    fail('metadata_unavailable')
  }
  if (result.exitCode !== 0) fail('metadata_unavailable')
  let report
  try { report = JSON.parse(marker.slice(REPORT_MARKER.length)) } catch (_) { fail('metadata_unavailable') }
  if (!validStorageGrantObservation(report, input)) fail('metadata_unavailable')
  return report
}
async function runObservation(argv, { runCli, observeWithSdk = observeInChild, clock = Date.now, idFactory = crypto.randomUUID, temporaryRoot = os.tmpdir() } = {}) {
  let options
  try { options = parseArguments(argv) } catch (_) { return readiness('argument_invalid') }
  if (options === HELP) return HELP
  const context = { environmentId: options.environmentId }
  let directory
  let configFile
  let result
  try {
    const requestId = idFactory()
    if (!uuid(requestId)) fail('argument_invalid')
    context.requestId = requestId
    context.cloudPath = `billiards-native-probes-v1/${context.requestId}/grant.mp4`
    directory = fs.mkdtempSync(path.join(temporaryRoot, 'billiards-native-observation-'))
    fs.chmodSync(directory, 0o700)
    configFile = path.join(directory, 'cloudbaserc.json')
    fs.writeFileSync(configFile, JSON.stringify({ version: '2.0', envId: options.environmentId }), { mode: 0o600, flag: 'wx' })
    fs.chmodSync(configFile, 0o600)
    const invoke = async args => {
      try { return await (runCli || (fixedArgs => capturedProcess('tcb', fixedArgs, undefined, directory)))([...args, '--config-file', configFile]) } catch (_) { return null }
    }
    const installed = await invoke(['--version'])
    if (!successful(installed)) fail('cli_unavailable')
    context.cliVersion = cliVersion(installed.stdout)
    const [major, minor] = context.cliVersion.split('.').map(Number)
    const environmentCommand = major > 3 || (major === 3 && minor >= 7) ? ['env', 'list', '--json'] : ['env:list', '--json']
    const environments = await invoke(environmentCommand)
    if (!successful(environments)) fail('environment_unavailable')
    try { if (!matchingEnvironment(cliJson(environments.stdout), options.environmentId)) fail('environment_unavailable') } catch (_) { fail('environment_unavailable') }
    if (!options.execute) result = readiness(null, context)
    else {
      const temporary = await invoke(['secrets', 'get', '--json', '--env-id', options.environmentId])
      if (!successful(temporary)) fail('temporary_credentials_unavailable')
      let credentialValue
      try { credentialValue = credentials(cliJson(temporary.stdout), options.environmentId) } catch (_) { fail('temporary_credentials_unavailable') }
      // This denotes the sample request time, after CLI readiness and credentials.
      const milliseconds = clock()
      if (!Number.isSafeInteger(milliseconds)) fail('argument_invalid')
      try { context.observedAt = new Date(milliseconds).toISOString() } catch (_) { fail('argument_invalid') }
      if (!/^\d{4}-\d{2}-\d{2}T/u.test(context.observedAt)) fail('argument_invalid')
      const observed = await observeWithSdk({ ...context, credentials: credentialValue })
      if (!validStorageGrantObservation(observed, context)) fail('metadata_unavailable')
      result = JSON.parse(JSON.stringify(observed))
    }
  } catch (error) { result = readiness(error instanceof SafeFailure ? error.reason : 'metadata_unavailable', context) }
  finally {
    try {
      if (configFile) fs.unlinkSync(configFile)
      if (directory) fs.rmdirSync(directory)
    } catch (_) { result = readiness('local_cleanup_unavailable', context) }
  }
  return result
}
if (require.main === module) {
  runObservation(process.argv.slice(2)).then(result => {
    process.stdout.write(`${JSON.stringify(result)}\n`)
    if (result.status === 'not_ready') process.exitCode = 2
  }, () => {
    process.stdout.write(`${JSON.stringify(readiness('argument_invalid'))}\n`)
    process.exitCode = 2
  })
}
module.exports = { parseArguments, runObservation }
