const fs = require('fs')
const os = require('os')
const path = require('path')
const cp = require('child_process')
const { parseArguments, runObservation } = require('../../scripts/observe-native-storage-grant')
const { runSdkObservation } = require('../../scripts/lib/native-storage-sdk-child')
const { observeStorageGrant } = require('../../scripts/lib/storage-grant-observation')
const ID = '12345678-1234-4123-8123-123456789abc'
const ENV = 'fixture-development'
const ARGS = ['--environment', 'development', '--environment-id', ENV, '--expected-environment-id', ENV]
const CREDS = { secretId: 'CANARY_ID_A', secretKey: 'CANARY_KEY_B', token: 'CANARY_SESSION_C', expiredAt: '2030-01-01 12:00:00', isTemporary: true, envId: ENV }
const metadata = { data: { url: 'https://fixture.cos.ap-test.myqcloud.com/upload', token: 'CANARY_GRANT_D', authorization: 'CANARY_AUTH_E', fileId: 'cloud://CANARY_FILE_F', cosFileId: 'CANARY_COS_G' } }
const response = stdout => ({ exitCode: 0, stdout, stderr: 'CANARY_STDERR_H' })
const DEPRECATION_WARNING = '⚠️  此命令将被废弃，请使用新的命令 tcb env list 代替\n'
const canonicalEnvironmentList = args => args[0] === 'env' && args[1] === 'list' && args[2] === '--json'
let root
beforeEach(() => { root = fs.mkdtempSync(path.join(os.tmpdir(), 'native-test-')) })
afterEach(() => { jest.restoreAllMocks(); fs.rmSync(root, { recursive: true, force: true }) })
function setup(overrides = {}) {
  const calls = []
  const runCli = async args => {
    const configFile = args[args.indexOf('--config-file') + 1]
    const config = JSON.parse(fs.readFileSync(configFile, 'utf8'))
    calls.push({ args, configFile, config, fileMode: fs.statSync(configFile).mode & 0o777, dirMode: fs.statSync(path.dirname(configFile)).mode & 0o777 })
    if (args.includes('--version')) return response('CloudBase CLI\n3.7.0\n')
    if (canonicalEnvironmentList(args)) return response('CloudBase CLI banner\n' + JSON.stringify({ data: { EnvList: [{ EnvId: ENV }] } }))
    if (args.includes('secrets')) return response(JSON.stringify(CREDS))
    throw Error('Unexpected cloud operation')
  }
  const observeWithSdk = jest.fn(async input => runSdkObservation(input, { loadSdk: () => ({ version: '3.17.2', sdk: { init: () => ({ getUploadMetadata: async ({ cloudPath }) => { expect(cloudPath).toBe(`billiards-native-probes-v1/${ID}/grant.mp4`); return metadata } }) } }) }))
  return { calls, deps: { runCli, observeWithSdk, clock: () => Date.UTC(2026, 10, 1), idFactory: () => ID, temporaryRoot: root, ...overrides } }
}
function secretFree(value) { const text = JSON.stringify(value); for (const canary of ['CANARY_', ENV, 'https://', 'cloud://']) expect(text).not.toContain(canary) }

test.each([[], ['--help', '--execute'], [...ARGS, '--wat'], [...ARGS, '--execute', '--execute'], [...ARGS, '--environment-id', ENV], ARGS.slice(0, -1), ARGS.map(v => v === 'development' ? 'production' : v), ARGS.map(v => v === ENV ? '../secret' : v), [...ARGS.slice(0, -1), 'foreign']].map(argv => [argv]))('invalid args refuse before effects: %j', async argv => {
  const { deps, calls } = setup()
  expect(() => parseArguments(argv)).toThrow('argument_invalid')
  const result = await runObservation(argv, deps)
  expect(result).toMatchObject({ status: 'not_ready', reason: 'argument_invalid', request_id: null })
  expect(calls).toHaveLength(0); expect(deps.observeWithSdk).not.toHaveBeenCalled(); expect(fs.readdirSync(root)).toEqual([]); secretFree(result)
})
test('standalone help is static with no CLI, SDK, clock, ID or temporary effects', async () => {
  const effects = () => { throw Error('effect') }
  const result = await runObservation(['--help'], { runCli: effects, observeWithSdk: effects, clock: effects, idFactory: effects, temporaryRoot: '/missing' })
  expect(result).toEqual(parseArguments(['--help'])); secretFree(result)
})
test('default checks only version and environment with owned minimal config then cleans it', async () => {
  fs.writeFileSync(path.join(root, 'unrelated'), 'keep')
  const { deps, calls } = setup({ clock: () => { throw Error('default must not sample') } })
  const result = await runObservation(ARGS, deps)
  expect(result).toMatchObject({ contract: 'native_storage_probe_readiness_v1', status: 'ready_for_review', reason: null, request_id: ID, cli_version: '3.7.0', observation_only: true, closure_evidence: false, erasable: false })
  expect(Object.keys(result).sort()).toEqual(['contract','status','reason','request_id','environment_id_sha256','cloud_path_sha256','cli_version','observation_only','closure_evidence','erasable'].sort())
  expect(calls).toHaveLength(2); expect(deps.observeWithSdk).not.toHaveBeenCalled()
  for (const call of calls) { expect(call.config).toEqual({ version: '2.0', envId: ENV }); expect(call.fileMode).toBe(0o600); expect(call.dirMode).toBe(0o700); expect(fs.existsSync(call.configFile)).toBe(false) }
  expect(fs.readdirSync(root)).toEqual(['unrelated']); secretFree(result)
})
test('CLI 3.7 readiness uses canonical JSON instead of legacy deprecation output without secret or SDK calls', async () => {
  const commands = []
  const environmentJson = JSON.stringify({ data: [{ envId: ENV }] })
  const { deps } = setup({ runCli: async args => {
    commands.push(args)
    if (args[0] === '--version') return response('3.7.0')
    if (args[0] === 'env:list') return response(DEPRECATION_WARNING + environmentJson)
    if (canonicalEnvironmentList(args)) return response(environmentJson)
    throw Error('CANARY_FORBIDDEN_SECRET_CALL')
  } })
  const result = await runObservation(ARGS, deps)
  expect(result).toMatchObject({ status: 'ready_for_review', reason: null, cli_version: '3.7.0' })
  const configFile = commands[0][2]
  expect(commands).toEqual([['--version', '--config-file', configFile], ['env', 'list', '--json', '--config-file', configFile]])
  expect(deps.observeWithSdk).not.toHaveBeenCalled(); expect(fs.readdirSync(root)).toEqual([]); secretFree(result)
})
test.each([
  ['3.0.0', ['env:list', '--json']],
  ['3.6.9', ['env:list', '--json']],
  ['3.7.0', ['env', 'list', '--json']],
  ['3.7.1', ['env', 'list', '--json']],
  ['3.8.0', ['env', 'list', '--json']],
  ['4.0.0', ['env', 'list', '--json']]
])('stable CLI %s selects the exact environment command once', async (installedVersion, expectedCommand) => {
  const commands = []
  const { deps } = setup({ runCli: async args => {
    commands.push(args)
    if (args[0] === '--version') return response(installedVersion)
    return response(JSON.stringify({ data: [{ envId: ENV }] }))
  } })
  const result = await runObservation(ARGS, deps)
  expect(result).toMatchObject({ status: 'ready_for_review', reason: null, cli_version: installedVersion })
  const configFile = commands[0][2]
  expect(commands).toEqual([['--version', '--config-file', configFile], [...expectedCommand, '--config-file', configFile]])
  expect(deps.observeWithSdk).not.toHaveBeenCalled(); expect(fs.readdirSync(root)).toEqual([]); secretFree(result)
})
test('controlled execute samples clock after credentials and calls fresh grant exactly once', async () => {
  let grants = 0; let init
  const { deps, calls } = setup({ clock: () => { expect(calls).toHaveLength(3); return Date.UTC(2026, 10, 1) }, observeWithSdk: input => runSdkObservation(input, { loadSdk: () => ({ version: '3.17.2', sdk: { init: options => { init = options; return { getUploadMetadata: async args => { grants++; expect(args).toEqual({ cloudPath: `billiards-native-probes-v1/${ID}/grant.mp4` }); return metadata } } } } }) }) })
  const result = await runObservation([...ARGS, '--execute'], deps)
  expect(result).toMatchObject({ contract: 'native_storage_grant_observation_v1', request_id: ID, observed_at: '2026-11-01T00:00:00.000Z', observation_only: true, closure_evidence: false, erasable: false })
  expect(init).toEqual({ env: ENV, secretId: CREDS.secretId, secretKey: CREDS.secretKey, sessionToken: CREDS.token, timeout: 15000, debug: false }); expect(grants).toBe(1)
  expect(calls[2].args.slice(0, 6)).toEqual(['secrets', 'get', '--json', '--env-id', ENV, '--config-file']); expect(fs.readdirSync(root)).toEqual([]); secretFree(result)
})
test.each([DEPRECATION_WARNING + JSON.stringify({ data: [{ envId: ENV }] }), '\u001b[33mWarning\u001b[0m\n' + JSON.stringify({ data: [{ envId: ENV }] }), '[', '{"error":"CANARY_"}', '{"data":{"EnvList":[{"EnvId":"foreign"}]}}', 'error has snippet {"EnvList":[]}', '{"EnvList":[]} trailing', '{"EnvList":[]}\n{"EnvList":[]}'])('unrecognized or foreign environment fails safely: %s', async stdout => {
  const commands = []
  const { deps } = setup(); const original = deps.runCli
  deps.runCli = args => { commands.push(args.slice(0, -2)); return canonicalEnvironmentList(args) ? response(stdout) : original(args) }
  const result = await runObservation([...ARGS, '--execute'], deps); expect(result.reason).toBe('environment_unavailable'); expect(deps.observeWithSdk).not.toHaveBeenCalled(); secretFree(result)
  expect(commands).toEqual([['--version'], ['env', 'list', '--json']])
})
test('oversize environment output refuses before credentials or SDK', async () => {
  const { deps, calls } = setup(); const original = deps.runCli
  deps.runCli = args => canonicalEnvironmentList(args) ? response('x'.repeat(1048577)) : original(args)
  const result = await runObservation([...ARGS, '--execute'], deps)
  expect(result.reason).toBe('environment_unavailable'); expect(calls).toHaveLength(1); expect(deps.observeWithSdk).not.toHaveBeenCalled(); secretFree(result)
})
test.each([[], { EnvList: [{ EnvId: ENV }] }, { data: [{ EnvId: ENV }] }, { data: { EnvList: [{ EnvId: ENV }] } }])('supports documented list containers, without inventing absent target: %j', async container => {
  if (Array.isArray(container)) container = [{ EnvId: ENV }]
  const { deps } = setup(); const original = deps.runCli; deps.runCli = args => canonicalEnvironmentList(args) ? response(JSON.stringify(container)) : original(args)
  expect((await runObservation(ARGS, deps)).status).toBe('ready_for_review')
})
test.each(['2.9.0', '3.07.0', 'v3.7.0', '3.7.0-beta', '3.7.0+build', '9007199254740992.0.0', '3.7.0\n3.8.0'])('rejects unsupported or unstable version: %s', async version => {
  const { deps } = setup({ runCli: async () => response(version) }); expect((await runObservation(ARGS, deps)).reason).toBe('cli_version_unsupported')
})
test.each([{ exitCode: 1, stdout: '3.7.0', stderr: 'CANARY_FAILURE' }, { exitCode: 0, stdout: 'x'.repeat(1048577), stderr: '' }, null])('CLI failed/oversize result refuses', async value => {
  const { deps } = setup({ runCli: async () => value }); const result = await runObservation(ARGS, deps); expect(result.reason).toBe('cli_unavailable'); secretFree(result)
})
test('CLI timeout/throw never escapes diagnostics', async () => {
  const { deps } = setup({ runCli: async () => { throw Error('CANARY_TIMEOUT') } }); expect((await runObservation(ARGS, deps)).reason).toBe('cli_unavailable')
})
test.each([{ ...CREDS, isTemporary: false }, { ...CREDS, envId: 'foreign' }, { ...CREDS, token: '' }, { ...CREDS, secretKey: '\nCANARY' }, { ...CREDS, unknown: 'CANARY' }, { credentials: CREDS }, { ...CREDS, expiredAt: 12 }, { ...CREDS, expiredAt: 'x'.repeat(16385) }])('invalid/permanent credentials never reach SDK', async credentials => {
  const { deps } = setup(); const original = deps.runCli; deps.runCli = args => args.includes('secrets') ? response(JSON.stringify(credentials)) : original(args)
  const result = await runObservation([...ARGS, '--execute'], deps); expect(result.reason).toBe('temporary_credentials_unavailable'); expect(deps.observeWithSdk).not.toHaveBeenCalled(); secretFree(result)
})
test.each(['environmentId', 'cloudPath', 'requestId', 'observedAt', 'cliVersion'])('binds returned report context: %s', async key => {
  const { deps } = setup({ observeWithSdk: async input => { const changed = { ...input, [key]: key === 'environmentId' ? 'foreign' : input[key] }; const report = observeStorageGrant({ ...changed, metadata }); if (key !== 'environmentId') report[{ cloudPath:'cloud_path_sha256', requestId:'request_id', observedAt:'observed_at', cliVersion:'cli_version' }[key]] = 'wrong'; return report } })
  expect((await runObservation([...ARGS, '--execute'], deps)).reason).toBe('metadata_unavailable')
})
test.each([() => new Proxy({}, { get() { throw Error('CANARY_PROXY') } }), () => ({ secret: 'CANARY_UNKNOWN' }), () => { throw Error('CANARY_CRASH') }])('SDK seam failure fails closed', async fixture => {
  const { deps } = setup({ observeWithSdk: fixture }); const result = await runObservation([...ARGS, '--execute'], deps); expect(result.reason).toBe('metadata_unavailable'); secretFree(result)
})
test('owned cleanup failure returns fixed reason and preserves unrelated assets', async () => {
  const { deps } = setup(); const original = deps.runCli
  deps.runCli = async args => { const result = await original(args); if (canonicalEnvironmentList(args)) fs.writeFileSync(path.join(path.dirname(args[args.indexOf('--config-file') + 1]), 'unowned'), 'keep'); return result }
  const result = await runObservation(ARGS, deps); expect(result.reason).toBe('local_cleanup_unavailable'); const directory = fs.readdirSync(root)[0]; expect(fs.readdirSync(path.join(root, directory))).toEqual(['unowned']); secretFree(result)
})
test('concurrent runs own different config directories', async () => {
  const { deps, calls } = setup(); const results = await Promise.all([runObservation(ARGS, deps), runObservation(ARGS, deps)])
  expect(results.every(r => r.status === 'ready_for_review')).toBe(true); expect(new Set(calls.map(c => c.configFile)).size).toBe(2); expect(fs.readdirSync(root)).toEqual([])
})
// Intercept only the process boundary. The tested adapter still uses real child pipes,
// timeout/maxBuffer handling and its fixed source path; fixtures never load native SDK.
test.each(['success', 'duplicate', 'unknown', 'authority', 'context', 'error', 'crash', 'oversize'])('isolated child capture/framing: %s', async mode => {
  const actual = cp.execFile
  const spy = jest.spyOn(cp, 'execFile').mockImplementation((file, args, options, callback) => {
    expect(file).toBe(process.execPath); expect(args).toEqual([path.resolve(__dirname, '../../scripts/lib/native-storage-sdk-child.js')]); expect(options.timeout).toBe(20000); expect(options.maxBuffer).toBe(1048576); expect(options.shell).toBe(false)
    return actual(file, [path.resolve(__dirname, '../helpers/native-storage-child-fixture.js'), mode], options, callback)
  })
  const { deps } = setup(); delete deps.observeWithSdk
  const result = await runObservation([...ARGS, '--execute'], deps)
  expect(spy).toHaveBeenCalledTimes(1); expect(result.contract).toBe(mode === 'success' ? 'native_storage_grant_observation_v1' : 'native_storage_probe_readiness_v1')
  if (mode !== 'success') expect(result.reason).toBe(mode === 'error' ? 'sdk_unavailable' : 'metadata_unavailable')
  secretFree(result)
})
test('default adapter bounds real CLI capture, closes stdin and never spawns SDK by default', async () => {
  const actual = cp.execFile; let commands = 0
  jest.spyOn(cp, 'execFile').mockImplementation((file, args, options, callback) => {
    expect(file).toBe('tcb'); expect(options.timeout).toBe(20000); expect(options.maxBuffer).toBe(1048576); expect(options.shell).toBe(false); commands++
    const configFile = args[args.indexOf('--config-file') + 1]
    expect(args).toEqual(commands === 1 ? ['--version', '--config-file', configFile] : ['env', 'list', '--json', '--config-file', configFile])
    return actual(process.execPath, ['-e', `process.stdin.resume();process.stdin.on('end',()=>process.stdout.write(${JSON.stringify(commands === 1 ? '3.7.0' : JSON.stringify([{ EnvId: ENV }]))}))`], options, callback)
  })
  const { deps } = setup(); delete deps.runCli
  const result = await runObservation(ARGS, deps); expect(result.status).toBe('ready_for_review'); expect(commands).toBe(2); expect(deps.observeWithSdk).not.toHaveBeenCalled()
})
test.each([[['--help']], [['--execute']]])('real main emits one safe JSON without actual CLI: %j', argv => {
  const result = cp.spawnSync(process.execPath, [path.resolve(__dirname, '../../scripts/observe-native-storage-grant.js'), ...argv], { encoding: 'utf8' })
  expect(result.status).toBe(argv[0] === '--help' ? 0 : 2); expect(result.stderr).toBe(''); expect(result.stdout.trim().split('\n')).toHaveLength(1); secretFree(JSON.parse(result.stdout))
})
test('installed CLI stable version banner and fixed promo are recognized', async () => {
  const { deps } = setup(); const original = deps.runCli
  deps.runCli = args => args.includes('--version') ? response('CloudBase CLI 3.7.0\n试试 tcb ai 命令，开启 AI 原生开发\n') : original(args)
  expect((await runObservation(ARGS, deps)).status).toBe('ready_for_review')
})
test.each([() => 'CANARY_INVALID_UUID', () => ({ toJSON: () => 'CANARY_FACTORY' }), () => new Proxy({}, { get() { throw Error('CANARY_PROXY') } })])('invalid factory output never enters public report or CLI', async idFactory => {
  const { deps, calls } = setup({ idFactory }); const result = await runObservation(ARGS, deps)
  expect(result).toMatchObject({ status: 'not_ready', reason: 'argument_invalid', request_id: null }); expect(calls).toHaveLength(0); secretFree(result)
})
test.each(['malformed', 'error_unknown', 'mixed', 'nonzero', 'aggregate'])('malformed/noisy child cases reject without leaks: %s', async mode => {
  const actual = cp.execFile
  jest.spyOn(cp, 'execFile').mockImplementation((file, args, options, callback) => actual(file, [path.resolve(__dirname, '../helpers/native-storage-child-fixture.js'), mode], options, callback))
  const { deps } = setup(); delete deps.observeWithSdk
  const result = await runObservation([...ARGS, '--execute'], deps); expect(result.reason).toBe('metadata_unavailable'); secretFree(result)
})
test('oversize stdin payload is refused before SDK process spawn', async () => {
  const exec = jest.spyOn(cp, 'execFile')
  const { deps } = setup(); delete deps.observeWithSdk; const original = deps.runCli
  deps.runCli = args => args.includes('secrets') ? response(JSON.stringify({ ...CREDS, secretId: 'i'.repeat(16384), secretKey: 'k'.repeat(16384), token: 't'.repeat(16384), expiredAt: 'e'.repeat(16384) })) : original(args)
  const result = await runObservation([...ARGS, '--execute'], deps); expect(result.reason).toBe('argument_invalid'); expect(exec).not.toHaveBeenCalled(); secretFree(result)
})
test('real child timeout is bounded with one process and no retry/cancellation claim', async () => {
  const actual = cp.execFile; let processes = 0
  jest.spyOn(cp, 'execFile').mockImplementation((file, args, options, callback) => { processes++; return actual(file, [path.resolve(__dirname, '../helpers/native-storage-child-fixture.js'), 'hang'], options, callback) })
  const { deps } = setup(); delete deps.observeWithSdk
  const start = Date.now(); const result = await runObservation([...ARGS, '--execute'], deps)
  expect(result.reason).toBe('metadata_unavailable'); expect(processes).toBe(1); expect(Date.now() - start).toBeLessThan(26000); secretFree(result)
}, 28000)
test('capture absolute timeout uses SIGKILL even if a child ignores SIGTERM', async () => {
  const actual = cp.execFile
  jest.spyOn(cp, 'execFile').mockImplementation((file, args, options, callback) => {
    expect(options.killSignal).toBe('SIGKILL')
    return actual(process.execPath, ['-e', "process.stdin.resume();process.stdin.on('end',()=>process.stdout.write('3.7.0'))"], options, callback)
  })
  const { deps } = setup(); delete deps.runCli
  expect((await runObservation(ARGS, deps)).reason).toBe('environment_unavailable')
})
test('real main check uses fake CLI process only and serializes one readiness report', () => {
  const fake = path.join(root, 'tcb')
  fs.writeFileSync(fake, `#!${process.execPath}\nprocess.stdin.resume();process.stdin.on('end',()=>{if(process.argv.includes('--version'))process.stdout.write('3.7.0');else if(process.argv[2]==='env'&&process.argv[3]==='list'&&process.argv[4]==='--json')process.stdout.write(${JSON.stringify(JSON.stringify({ data: [{ EnvId: ENV }] }))});else{process.stderr.write('CANARY_FORBIDDEN_CLI');process.exitCode=2}})`, { mode: 0o700 })
  const result = cp.spawnSync(process.execPath, [path.resolve(__dirname, '../../scripts/observe-native-storage-grant.js'), ...ARGS], { env: { ...process.env, PATH: root }, encoding: 'utf8', timeout: 5000 })
  expect(result.status).toBe(0); expect(result.stderr).toBe(''); expect(result.stdout.trim().split('\n')).toHaveLength(1); const report = JSON.parse(result.stdout); expect(report.status).toBe('ready_for_review'); secretFree(report)
})
test.each([{ data: [{ envId: ENV, region: 'fixture' }] }, [{ envId: ENV }], { EnvList: [{ envId: ENV }] }, { data: { EnvList: [{ envId: ENV }] } }])('verified normalized CLI environment identifier is accepted: %j', async envelope => {
  const { deps } = setup(); const original = deps.runCli
  deps.runCli = args => canonicalEnvironmentList(args) ? response(JSON.stringify(envelope)) : original(args)
  expect((await runObservation(ARGS, deps)).status).toBe('ready_for_review')
})
test.each([{ data: [{ EnvId: ENV, envId: ENV }] }, { data: [{ envId: ENV }, { EnvId: ENV }] }, { data: [{ envId: ENV }, null] }, { data: [{ envId: 123 }] }, { data: [{ envId: ENV }], error: 'CANARY_ERROR' }, { EnvList: [{ EnvId: ENV }], data: [{ envId: ENV }] }])('ambiguous/malformed environment identity refuses: %j', async envelope => {
  const { deps } = setup(); const original = deps.runCli
  deps.runCli = args => canonicalEnvironmentList(args) ? response(JSON.stringify(envelope)) : original(args)
  expect((await runObservation(ARGS, deps)).reason).toBe('environment_unavailable')
})
test('verified exact data credential wrapper unwraps only six fields for SDK', async () => {
  const { deps } = setup(); const original = deps.runCli
  deps.runCli = args => args.includes('secrets') ? response(JSON.stringify({ data: CREDS })) : original(args)
  expect((await runObservation([...ARGS, '--execute'], deps)).contract).toBe('native_storage_grant_observation_v1')
  expect(deps.observeWithSdk.mock.calls[0][0].credentials).toEqual(CREDS)
})
test.each([{ data: CREDS, meta: {} }, { data: CREDS, unknown: 'CANARY' }, { data: { ...CREDS, secret: 'CANARY' } }, { data: { data: CREDS } }])('credential wrappers never infer unknown envelopes: %j', async envelope => {
  const { deps } = setup(); const original = deps.runCli
  deps.runCli = args => args.includes('secrets') ? response(JSON.stringify(envelope)) : original(args)
  expect((await runObservation([...ARGS, '--execute'], deps)).reason).toBe('temporary_credentials_unavailable'); expect(deps.observeWithSdk).not.toHaveBeenCalled()
})
test('argument array accessors and proxies refuse without invoking traps or leaking', async () => {
  let reads = 0
  const accessor = [...ARGS]
  Object.defineProperty(accessor, '0', { get() { reads++; throw Error('CANARY_ARGUMENT_GETTER') }, enumerable: true })
  const proxy = new Proxy(ARGS, { get() { reads++; throw Error('CANARY_ARGUMENT_PROXY') } })
  for (const argv of [accessor, proxy]) {
    expect(() => parseArguments(argv)).toThrow('argument_invalid')
    const { deps, calls } = setup(); const result = await runObservation(argv, deps)
    expect(result.reason).toBe('argument_invalid'); expect(calls).toHaveLength(0); secretFree(result)
  }
  expect(reads).toBe(0)
})
