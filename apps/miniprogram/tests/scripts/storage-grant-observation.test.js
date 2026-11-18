const {
  NATIVE_STORAGE_OBSERVATION_CONTRACT,
  PINNED_STORAGE_SDK_VERSION,
  StorageGrantObservationError,
  observeStorageGrant,
  validStorageGrantObservation,
} = require('../../scripts/lib/storage-grant-observation')

const context = {
  requestId: '123e4567-e89b-42d3-a456-426614174000',
  cloudPath: 'billiards-native-probes-v1/123e4567-e89b-42d3-a456-426614174000/grant.mp4',
  environmentId: 'development-test-abc',
  observedAt: '2026-11-01T12:00:00.000Z',
  cliVersion: '3.7.0',
}

function metadata(authorization = 'synthetic-opaque-authorization') {
  return { data: {
    url: 'https://synthetic-123.cos.ap-shanghai.myqcloud.com/probe?private=value',
    token: 'synthetic-token',
    authorization,
    fileId: 'cloud://synthetic-observation-fixture/grant.mp4',
    cosFileId: 'synthetic-cos-file-id',
    download_url: 'https://synthetic.tcb.qcloud.la/download?private=value',
  } }
}

function v5(signTime = '1793534400;1793534700', keyTime = signTime) {
  return `q-sign-algorithm=sha1&q-ak=synthetic-access-id&q-sign-time=${signTime}` +
    `&q-key-time=${keyTime}&q-header-list=host&q-url-param-list=&q-signature=${'a'.repeat(40)}`
}

function expectedReport(overrides = {}) {
  return {
    contract: 'native_storage_grant_observation_v1',
    request_id: context.requestId,
    environment_id_sha256: '85b1131bb602223ce98c22a9e98d9cce98d2d586104ecffde092aa844c5cf418',
    cloud_path_sha256: 'e59d655246dc58abf31cf3bda93faadded8c886df0f72aa15b0979bc7e5812ce',
    file_id_sha256: '4adfa0eab02089b780dc3e151de47f13420b6f3733634658d6de0fda7bc2e2de',
    observed_at: context.observedAt,
    sdk_version: '3.17.2',
    cli_version: '3.7.0',
    authorization_format: 'opaque',
    endpoint_family: 'cos',
    metadata_fields: { url: true, token: true, authorization: true, fileId: true, cosFileId: true, download_url: true },
    signature_window: null,
    observation_only: true,
    closure_evidence: false,
    erasable: false,
    ...overrides,
  }
}

test('observes actual-shaped opaque metadata as the exact hashed whitelist without mutation', () => {
  const source = metadata()
  const original = JSON.parse(JSON.stringify(source))
  const report = observeStorageGrant({ ...context, metadata: source })
  expect(report).toEqual(expectedReport())
  expect(source).toEqual(original)
  expect(validStorageGrantObservation(report, context)).toBe(true)
  expect(NATIVE_STORAGE_OBSERVATION_CONTRACT).toBe(report.contract)
  expect(PINNED_STORAGE_SDK_VERSION).toBe(report.sdk_version)
})

test.each([
  ['1793534400;1793534700', 1793534400, 1793534700],
  ['1;2', 1, 2],
  ['1;8640000000000', 1, 8640000000000],
])('recognized v5 interval %s remains observation only even when expired', (interval, start, end) => {
  const report = observeStorageGrant({ ...context, metadata: metadata(v5(interval)) })
  expect(report).toEqual(expectedReport({
    authorization_format: 'cos_v5_header',
    signature_window: { starts_at_seconds: start, ends_at_seconds: end, key_starts_at_seconds: start, key_ends_at_seconds: end },
  }))
  expect(validStorageGrantObservation(report, context)).toBe(true)
})

test('canonical base64 stays opaque without decoding a recognizable expired header', () => {
  const encoded = Buffer.from(v5('1;2')).toString('base64')
  const report = observeStorageGrant({ ...context, metadata: metadata(encoded) })
  expect(report).toEqual(expectedReport({ authorization_format: 'base64_opaque' }))
  expect(JSON.stringify(report)).not.toContain('synthetic-access-id')
})

function expectSafeFailure(input) {
  let failure
  try { observeStorageGrant(input) } catch (error) { failure = error }
  expect(failure).toBeInstanceOf(StorageGrantObservationError)
  expect(failure.name).toBe('StorageGrantObservationError')
  expect(failure.message).toBe('NATIVE_STORAGE_METADATA_INVALID')
  expect(failure.code).toBe('NATIVE_STORAGE_METADATA_INVALID')
  expect(failure.cause).toBeUndefined()
  expect(`${failure}\n${failure.stack}\n${JSON.stringify(failure)}`).not.toContain('CANARY')
}

test.each([
  ['AA==', 'base64_opaque'], ['AAA=', 'base64_opaque'], ['AAAA', 'base64_opaque'],
  ['AB==', 'opaque'], ['AAB=', 'opaque'], ['AAA', 'opaque'], ['AA===', 'opaque'],
  ['AA-_', 'opaque'], ['AA==AA==', 'opaque'], ['a timestamp 1;2', 'opaque'],
])('authorization %s has syntax %s without noncanonical base64 coercion', (authorization, format) => {
  expect(observeStorageGrant({ ...context, metadata: metadata(authorization) }))
    .toEqual(expectedReport({ authorization_format: format }))
})

test.each([
  value => `${value}&q-ak=duplicate`,
  value => `${value}&unknown=CANARY_PRIVATE_KEY`,
  value => value.replace('q-ak=', 'q-%61k='),
  value => value.replace('q-ak=', 'Q-ak='),
  value => value.replace('q-ak=', 'q-ak ='),
  value => value.replace('sha1', 'SHA1'),
  value => value.replace('q-signature=', 'q-signature=%61'),
  value => value.replace('a'.repeat(40), 'A'.repeat(40)),
  value => value.replace('&q-header-list=host', ''),
  value => value.replace('q-ak=synthetic-access-id', 'q-ak='),
  value => value.replace('q-ak=synthetic-access-id', 'q-ak=encoded%2Fid'),
  value => `${value}&`,
  value => value.replace('q-header-list=host', 'q-header-list=host;host'),
  value => value.replace('q-header-list=host', 'q-header-list=z;a'),
  value => value.replace('q-header-list=host', 'q-header-list=Host'),
  value => value.replace('q-header-list=host', 'q-header-list=%68ost'),
  value => value.replace('q-url-param-list=', 'q-url-param-list=z;a'),
])('malformed v5 candidates yield no window or authority', mutate => {
  const report = observeStorageGrant({ ...context, metadata: metadata(mutate(v5())) })
  expect(report).toEqual(expectedReport({ authorization_format: 'malformed_cos_v5_header' }))
  expect(validStorageGrantObservation(report, context)).toBe(true)
})

test('strict v5 accepts reordered known fields and canonical empty or sorted name lists', () => {
  const authorization = v5('1;2').replace('q-header-list=host', 'q-header-list=').replace('q-url-param-list=', 'q-url-param-list=a;b')
    .split('&').reverse().join('&')
  const report = observeStorageGrant({ ...context, metadata: metadata(authorization) })
  expect(report.authorization_format).toBe('cos_v5_header')
  expect(report.signature_window).toEqual({ starts_at_seconds: 1, ends_at_seconds: 2, key_starts_at_seconds: 1, key_ends_at_seconds: 2 })
})

test('signature and key intervals stay separate syntax candidates without inferred containment', () => {
  const report = observeStorageGrant({ ...context, metadata: metadata(v5('3;4', '1;2')) })
  expect(report.signature_window).toEqual({ starts_at_seconds: 3, ends_at_seconds: 4, key_starts_at_seconds: 1, key_ends_at_seconds: 2 })
  expect(validStorageGrantObservation(report, context)).toBe(true)
  expect(report.closure_evidence).toBe(false)
  expect(report.erasable).toBe(false)
})

test.each([
  '0;2', '-1;2', '01;2', '+1;2', '1;1', '2;1', '1.0;2', '1;2;3',
  '1e3;2000', '1%3B2', '1;9007199254740992', '1;8640000000001',
])('unsafe or noncanonical v5 time %s is syntax uncertainty', interval => {
  for (const authorization of [v5(interval), v5('1;2', interval)]) {
    expect(observeStorageGrant({ ...context, metadata: metadata(authorization) }))
      .toEqual(expectedReport({ authorization_format: 'malformed_cos_v5_header' }))
  }
})

test.each([
  ['https://cos.ap-shanghai.myqcloud.com/probe', 'cos'],
  ['https://bucket.cos.ap-shanghai.myqcloud.com/probe', 'cos'],
  ['https://tcb.qcloud.la/probe', 'cloudbase'],
  ['https://synthetic.tcb.qcloud.la/probe', 'cloudbase'],
  ['https://tcb-api.tencentcloudapi.com/probe', 'cloudbase'],
  ['https://other.example/probe', 'other_https'],
  ['https://bucket.cos.ap-shanghai.myqcloud.com.attacker.example/probe', 'other_https'],
  ['https://evilmyqcloud.com/probe', 'other_https'],
  ['https://notcos.ap-shanghai.myqcloud.com/probe', 'other_https'],
  ['https://bucket.cos-internal.ap-shanghai.myqcloud.com/probe', 'other_https'],
  ['https://extra.bucket.cos.ap-shanghai.myqcloud.com/probe', 'other_https'],
  ['https://attacker.myqcloud.com/probe', 'other_https'],
  ['https://cos.attacker.example/probe', 'other_https'],
  ['https://eviltcb.qcloud.la/probe', 'other_https'],
  ['https://tcb-api.tencentcloudapi.com.attacker.example/probe', 'other_https'],
])('endpoint %s classifies only its exact host family as %s', (url, family) => {
  const source = metadata()
  source.data.url = url
  expect(observeStorageGrant({ ...context, metadata: source }).endpoint_family).toBe(family)
})

test.each([
  null, [], {}, { data: null }, { data: [] },
])('malformed metadata container refuses safely', source => {
  expectSafeFailure({ ...context, metadata: source })
})

test.each(['url', 'token', 'authorization', 'fileId', 'cosFileId'])('requires bounded nonempty UTF8 %s', field => {
  for (const invalid of [undefined, null, 1, {}, '', 'x\u0000CANARY', 'x\u007fCANARY', '\ud800CANARY', 'x'.repeat(16385)]) {
    const source = metadata()
    source.data[field] = invalid
    expectSafeFailure({ ...context, metadata: source })
  }
})

test('fileID has its narrower cap and UTF8 byte limits apply', () => {
  const source = metadata()
  source.data.fileId = 'x'.repeat(2049)
  expectSafeFailure({ ...context, metadata: source })
  source.data.fileId = `cloud://${'é'.repeat(1021)}`
  expectSafeFailure({ ...context, metadata: source })
  source.data.fileId = `cloud://${'x'.repeat(2040)}`
  source.data.token = 'é'.repeat(8193)
  expectSafeFailure({ ...context, metadata: source })
  source.data.token = 'é'.repeat(8192)
  expect(validStorageGrantObservation(observeStorageGrant({ ...context, metadata: source }), context)).toBe(true)
})

test.each(['cloud://', 'not-a-cloud-file-id'])('fileID %s needs the native opaque cloud prefix and payload', fileId => {
  const source = metadata()
  source.data.fileId = fileId
  expectSafeFailure({ ...context, metadata: source })
})

test('download URL is optional presence only and no value is consumed', () => {
  const source = metadata()
  delete source.data.download_url
  expect(observeStorageGrant({ ...context, metadata: source })).toEqual(expectedReport({
    metadata_fields: { url: true, token: true, authorization: true, fileId: true, cosFileId: true, download_url: false },
  }))
  const getter = jest.fn(() => { throw new Error('CANARY_DOWNLOAD_GETTER') })
  Object.defineProperty(source.data, 'download_url', { get: getter, enumerable: true })
  expect(observeStorageGrant({ ...context, metadata: source })).toEqual(expectedReport())
  expect(getter).not.toHaveBeenCalled()
})

test.each([
  'http://synthetic.example/CANARY', 'https://user:CANARY@synthetic.example/',
  'https://synthetic.example/#CANARY', 'CANARY_not_a_url',
])('unsafe upload URL refuses with no raw parser error', url => {
  const source = metadata()
  source.data.url = url
  expectSafeFailure({ ...context, metadata: source })
})

test.each([
  ['requestId', '123E4567-e89b-42d3-a456-426614174000'],
  ['requestId', '123e4567-e89b-12d3-a456-426614174000'],
  ['requestId', '123e4567-e89b-42d3-7456-426614174000'],
  ['cloudPath', 'saved-clips/CANARY/grant.mp4'],
  ['cloudPath', `${context.cloudPath}?CANARY`],
  ['environmentId', ''], ['environmentId', 'CANARY/unsafe'],
  ['environmentId', 'x'.repeat(129)], ['environmentId', 'éCANARY'],
  ['observedAt', '2026-11-01T12:00:00Z'],
  ['observedAt', '2026-02-30T12:00:00.000Z'],
  ['observedAt', '2026-11-01T12:00:00.000+00:00'],
  ['observedAt', 1793534400000],
  ['cliVersion', 'v3.7.0'], ['cliVersion', '03.7.0'], ['cliVersion', '3.7'],
  ['cliVersion', '3.7.0-01'], ['cliVersion', '3.7.0-rc.1'], ['cliVersion', '3.7.0+build'],
  ['cliVersion', '9007199254740992.7.0'], ['cliVersion', '3.7.0 CANARY'],
])('invalid trusted context %s refuses before reporting', (field, value) => {
  const invalid = { ...context, [field]: value }
  expectSafeFailure({ ...invalid, metadata: metadata() })
  expect(validStorageGrantObservation(expectedReport(), invalid)).toBe(false)
})

test('safe environment dots and stable zero CLI components bind to matching context', () => {
  const otherContext = { ...context, environmentId: 'synthetic.env_0-1', cliVersion: '0.0.0' }
  const report = observeStorageGrant({ ...otherContext, metadata: metadata() })
  expect(report.cli_version).toBe('0.0.0')
  expect(validStorageGrantObservation(report, otherContext)).toBe(true)
  expect(validStorageGrantObservation(report, context)).toBe(false)
})

test('all sensitive values and unknown fields remain absent from serialized report', () => {
  const source = metadata('CANARY_AUTHORIZATION')
  source.data.url = 'https://CANARY_HOST.example/CANARY_PATH?secret=CANARY_QUERY'
  source.data.token = 'CANARY_TOKEN'
  source.data.fileId = 'cloud://CANARY_FILEID'
  source.data.cosFileId = 'CANARY_COSFILEID'
  source.data.download_url = 'https://private.example/CANARY_DOWNLOAD'
  source.data.CANARY_PRIVATE_FIELD = { nested: 'CANARY_NESTED_VALUE' }
  source.CANARY_ROOT_FIELD = 'CANARY_ROOT_VALUE'
  const original = JSON.stringify(source)
  const report = observeStorageGrant({ ...context, metadata: source })
  expect(JSON.stringify(report)).not.toContain('CANARY')
  expect(Object.keys(report.metadata_fields)).toEqual(['url', 'token', 'authorization', 'fileId', 'cosFileId', 'download_url'])
  expect(JSON.stringify(source)).toBe(original)
})

test('ignores unknown ordinary values without serializing or invoking their hooks', () => {
  const source = metadata()
  source.data.unknown = { toJSON() { throw new Error('CANARY_JSON') }, toString() { throw new Error('CANARY_STRING') } }
  expect(observeStorageGrant({ ...context, metadata: source })).toEqual(expectedReport())
})

test('getters and hostile reflection fail safely without evaluating getters', () => {
  const getter = jest.fn(() => { throw new Error('CANARY_GETTER') })
  for (const location of ['input', 'metadata', 'data']) {
    const input = { ...context, metadata: metadata() }
    const owner = location === 'input' ? input : location === 'metadata' ? input.metadata : input.metadata.data
    const key = location === 'input' ? 'environmentId' : location === 'metadata' ? 'data' : 'token'
    Object.defineProperty(owner, key, { get: getter, enumerable: true })
    expectSafeFailure(input)
  }
  expect(getter).not.toHaveBeenCalled()
  const input = new Proxy({}, { ownKeys() { throw new Error('CANARY_PROXY') } })
  expectSafeFailure(input)
  expect(validStorageGrantObservation(input, context)).toBe(false)
})

test.each(['url', 'token', 'authorization', 'fileId', 'cosFileId'])('never evaluates a consumed %s getter', field => {
  const source = metadata()
  const getter = jest.fn(() => { throw new Error('CANARY_GETTER') })
  Object.defineProperty(source.data, field, { get: getter, enumerable: true })
  expectSafeFailure({ ...context, metadata: source })
  expect(getter).not.toHaveBeenCalled()
})

test('fixed error ignores hostile constructor arguments and carries no raw cause', () => {
  const error = new StorageGrantObservationError('CANARY_MESSAGE', { cause: new Error('CANARY_CAUSE') })
  expect(error.message).toBe('NATIVE_STORAGE_METADATA_INVALID')
  expect(error.code).toBe('NATIVE_STORAGE_METADATA_INVALID')
  expect(error.cause).toBeUndefined()
  expect(`${error}\n${error.stack}\n${JSON.stringify(error)}`).not.toContain('CANARY')
})

test('unknown getters are ignored without access or field-name disclosure', () => {
  const source = metadata()
  const getter = jest.fn(() => { throw new Error('CANARY_UNKNOWN_GETTER') })
  Object.defineProperty(source.data, 'CANARY_PRIVATE_GETTER', { get: getter, enumerable: true })
  expect(observeStorageGrant({ ...context, metadata: source })).toEqual(expectedReport())
  expect(getter).not.toHaveBeenCalled()
})

test.each([
  report => { report.private = 'CANARY_RAW' },
  report => { report.authority_verified = true },
  report => { report.observation_only = false },
  report => { report.closure_evidence = true },
  report => { report.erasable = true },
  report => { report.environment_id_sha256 = '0'.repeat(64) },
  report => { report.cloud_path_sha256 = '0'.repeat(64) },
  report => { report.file_id_sha256 = 'CANARY_HASH' },
  report => { report.request_id = '00000000-0000-4000-8000-000000000000' },
  report => { report.observed_at = '2026-11-01T12:00:01.000Z' },
  report => { report.cli_version = '3.7.1' },
  report => { report.sdk_version = '3.17.3' },
  report => { report.contract = 'different' },
  report => { report.endpoint_family = 'CANARY_HOST' },
  report => { report.authorization_format = 'verified' },
  report => { report.metadata_fields.token = 'true' },
  report => { report.metadata_fields.fileId = false },
  report => { report.metadata_fields.extra = false },
  report => { report.signature_window = { starts_at_seconds: 1 } },
  report => { delete report.file_id_sha256 },
  report => { report[Symbol('CANARY')] = true },
  report => { Object.defineProperty(report, 'closure_evidence', { enumerable: false }) },
  report => { Object.defineProperty(report.metadata_fields, 'token', { enumerable: false }) },
])('contextual validator rejects forged, unknown, private, authority or malformed report fields', mutate => {
  const report = expectedReport()
  mutate(report)
  expect(validStorageGrantObservation(report, context)).toBe(false)
})

test('validator requires exact v5 window shape and rejects format/window mismatches', () => {
  const report = observeStorageGrant({ ...context, metadata: metadata(v5('1;2')) })
  for (const format of ['opaque', 'base64_opaque', 'malformed_cos_v5_header']) {
    expect(validStorageGrantObservation({ ...report, authorization_format: format }, context)).toBe(false)
  }
  expect(validStorageGrantObservation({ ...report, signature_window: null }, context)).toBe(false)
  for (const mutate of [
    window => { window.starts_at_seconds = 0 },
    window => { window.ends_at_seconds = 1 },
    window => { window.key_starts_at_seconds = '1' },
    window => { window.key_ends_at_seconds = 8640000000001 },
    window => { window.extra = 'CANARY' },
    window => { Object.defineProperty(window, 'ends_at_seconds', { enumerable: false }) },
  ]) {
    const candidate = { ...report, signature_window: { ...report.signature_window } }
    mutate(candidate.signature_window)
    expect(validStorageGrantObservation(candidate, context)).toBe(false)
  }
})

test('validator returns false for getters in report, nested shape or context without evaluating them', () => {
  const getter = jest.fn(() => { throw new Error('CANARY_GETTER') })
  for (const location of ['report', 'fields', 'window', 'context']) {
    const report = observeStorageGrant({ ...context, metadata: metadata(v5('1;2')) })
    const boundContext = { ...context }
    const owner = location === 'report' ? report : location === 'fields' ? report.metadata_fields : location === 'window' ? report.signature_window : boundContext
    Object.defineProperty(owner, location === 'context' ? 'environmentId' : 'CANARY', { get: getter, enumerable: true })
    expect(validStorageGrantObservation(report, boundContext)).toBe(false)
  }
  expect(getter).not.toHaveBeenCalled()
  expect(validStorageGrantObservation(null, context)).toBe(false)
  expect(validStorageGrantObservation(expectedReport(), null)).toBe(false)
})

test.each(['report', 'metadata_fields', 'signature_window'])('validator rejects %s Proxy serialization hooks before invoking traps', location => {
  const report = observeStorageGrant({ ...context, metadata: metadata(v5('1;2')) })
  const target = location === 'report' ? report : report[location]
  const get = jest.fn((value, key, receiver) => key === 'toJSON'
    ? () => ({ private_authorization: 'CANARY_PROXY_SECRET' })
    : Reflect.get(value, key, receiver))
  const proxy = new Proxy(target, { get })
  const candidate = location === 'report' ? proxy : { ...report, [location]: proxy }
  expect(validStorageGrantObservation(candidate, context)).toBe(false)
  expect(get).not.toHaveBeenCalled()
})

test.each([
  ['report', 'erasable', true],
  ['report', 'closure_evidence', true],
  ['metadata_fields', 'token', false],
  ['signature_window', 'ends_at_seconds', -1],
])('validator rejects %s Proxy value substitution for %s', (location, field, replacement) => {
  const report = observeStorageGrant({ ...context, metadata: metadata(v5('1;2')) })
  const target = location === 'report' ? report : report[location]
  const get = jest.fn((value, key, receiver) => key === field ? replacement : Reflect.get(value, key, receiver))
  const proxy = new Proxy(target, { get })
  const candidate = location === 'report' ? proxy : { ...report, [location]: proxy }
  expect(validStorageGrantObservation(candidate, context)).toBe(false)
  expect(get).not.toHaveBeenCalled()
})

test.each(['input', 'metadata', 'data'])('observer rejects consumed %s Proxy records before any trap', location => {
  const input = { ...context, metadata: metadata() }
  const target = location === 'input' ? input : location === 'metadata' ? input.metadata : input.metadata.data
  const traps = {
    get: jest.fn(Reflect.get),
    getPrototypeOf: jest.fn(Reflect.getPrototypeOf),
    getOwnPropertyDescriptor: jest.fn(Reflect.getOwnPropertyDescriptor),
    ownKeys: jest.fn(Reflect.ownKeys),
  }
  const proxy = new Proxy(target, traps)
  if (location === 'metadata') input.metadata = proxy
  if (location === 'data') input.metadata.data = proxy
  expectSafeFailure(location === 'input' ? proxy : input)
  for (const trap of Object.values(traps)) expect(trap).not.toHaveBeenCalled()
})

test('validator rejects Proxy context before any trap', () => {
  const traps = {
    get: jest.fn(Reflect.get),
    getPrototypeOf: jest.fn(Reflect.getPrototypeOf),
    getOwnPropertyDescriptor: jest.fn(Reflect.getOwnPropertyDescriptor),
    ownKeys: jest.fn(Reflect.ownKeys),
  }
  expect(validStorageGrantObservation(expectedReport(), new Proxy(context, traps))).toBe(false)
  for (const trap of Object.values(traps)) expect(trap).not.toHaveBeenCalled()
})

test('revoked Proxies fail safely at every consumed record boundary', () => {
  const { proxy, revoke } = Proxy.revocable({}, {})
  revoke()
  expectSafeFailure(proxy)
  expectSafeFailure({ ...context, metadata: proxy })
  expectSafeFailure({ ...context, metadata: { data: proxy } })
  expect(validStorageGrantObservation(proxy, context)).toBe(false)
  expect(validStorageGrantObservation(expectedReport(), proxy)).toBe(false)
  expect(validStorageGrantObservation({ ...expectedReport(), metadata_fields: proxy }, context)).toBe(false)
  expect(validStorageGrantObservation({ ...expectedReport(), authorization_format: 'cos_v5_header', signature_window: proxy }, context)).toBe(false)
})

test('pure module has only hash and Proxy inspection capabilities without I/O or provider ports', () => {
  const fs = require('fs')
  const vm = require('vm')
  const source = fs.readFileSync(require.resolve('../../scripts/lib/storage-grant-observation'), 'utf8')
  const imports = []
  const pureModule = { exports: {} }
  vm.runInNewContext(`${source}\nmodule.report = module.exports.observeStorageGrant(JSON.parse(input))`, {
    module: pureModule,
    exports: pureModule.exports,
    require(name) {
      imports.push(name)
      if (name === 'crypto' || name === 'node:crypto') return { createHash: require('crypto').createHash }
      if (name === 'util' || name === 'node:util') return { types: { isProxy: require('util').types.isProxy } }
      throw new Error('I/O import forbidden')
    },
    URL,
    Buffer: { byteLength: Buffer.byteLength },
    input: JSON.stringify({ ...context, metadata: metadata() }),
  })
  expect(pureModule.report).toEqual(expectedReport())
  expect(imports).toEqual([
    expect.stringMatching(/^(?:node:)?crypto$/),
    expect.stringMatching(/^(?:node:)?util$/),
  ])
})
