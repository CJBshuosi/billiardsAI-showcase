const cp = require('child_process')
const path = require('path')
const { runSdkObservation } = require('../../scripts/lib/native-storage-sdk-child')
const ID = '12345678-1234-4123-8123-123456789abc'
const input = () => ({ requestId: ID, cloudPath: `billiards-native-probes-v1/${ID}/grant.mp4`, environmentId: 'fixture-development', observedAt: '2026-11-01T00:00:00.000Z', cliVersion: '3.7.0', credentials: { secretId: 'CANARY_ID', secretKey: 'CANARY_KEY', token: 'CANARY_SESSION', expiredAt: '2030-01-01 00:00:00', isTemporary: true, envId: 'fixture-development' } })
const child = path.resolve(__dirname, '../../scripts/lib/native-storage-sdk-child.js')
test('exact pinned SDK maps sessionToken and only grants metadata once', async () => {
  const actions = []
  const report = await runSdkObservation(input(), { loadSdk: () => ({ version: '3.17.2', sdk: { init: config => { expect(config).toEqual({ env: 'fixture-development', secretId: 'CANARY_ID', secretKey: 'CANARY_KEY', sessionToken: 'CANARY_SESSION', timeout: 15000, debug: false }); return new Proxy({ getUploadMetadata: async args => { actions.push(['metadata', args]); return { data: { url: 'https://cos.ap-test.myqcloud.com', token: 'CANARY_GRANT', authorization: 'CANARY_AUTH', fileId: 'cloud://CANARY_FILE', cosFileId: 'CANARY_COS' } } } }, { get(target, key) { if (key !== 'getUploadMetadata') throw Error('Other provider method'); return target[key] } }) } } }) })
  expect(actions).toEqual([['metadata', { cloudPath: `billiards-native-probes-v1/${ID}/grant.mp4` }]])
  expect(report).toMatchObject({ contract: 'native_storage_grant_observation_v1', sdk_version: '3.17.2', observation_only: true, closure_evidence: false, erasable: false }); expect(JSON.stringify(report)).not.toContain('CANARY_')
})
test.each([data => ({ ...data, extra: 1 }), data => ({ ...data, requestId: 'bad' }), data => ({ ...data, cloudPath: 'clips/product.mp4' }), data => ({ ...data, observedAt: 'bad' }), data => ({ ...data, cliVersion: '3.7.0-beta' }), data => ({ ...data, credentials: { ...data.credentials, isTemporary: false } }), data => ({ ...data, credentials: { ...data.credentials, envId: 'foreign' } }), data => new Proxy(data, {}), data => ({ ...data, credentials: new Proxy(data.credentials, {}) }), data => Object.defineProperty(data, 'environmentId', { get() { throw Error('CANARY_GETTER') } })])('bad input refuses before SDK loading', async mutate => {
  const loadSdk = jest.fn(() => { throw Error('SDK must not load') }); await expect(runSdkObservation(mutate(input()), { loadSdk })).rejects.toThrow('argument_invalid'); expect(loadSdk).not.toHaveBeenCalled()
})
test.each([() => ({ version: '3.17.1', sdk: { init: () => { throw Error('must not initialize') } } }), () => { throw Error('CANARY_SDK') }])('wrong/missing SDK fixed refusal', async loadSdk => {
  await expect(runSdkObservation(input(), { loadSdk })).rejects.toThrow('sdk_unavailable')
})
test('provider exception is fixed and never retried', async () => {
  let calls = 0
  await expect(runSdkObservation(input(), { loadSdk: () => ({ version: '3.17.2', sdk: { init: () => ({ getUploadMetadata: async () => { calls++; throw Error('CANARY_REMOTE') } }) } }) })).rejects.toThrow('metadata_unavailable'); expect(calls).toBe(1)
})
test.each(['not-json', '{}', 'x'.repeat(65537), JSON.stringify({ ...input(), extra: 'CANARY' })])('child stdin main refuses malformed/oversize input without raw diagnostics', text => {
  const result = cp.spawnSync(process.execPath, [child], { input: text, encoding: 'utf8' }); expect(result.status).toBe(2); expect(result.stderr).toBe(''); expect(result.stdout).toBe('BILLIARDS_NATIVE_STORAGE_ERROR:argument_invalid\n')
})
