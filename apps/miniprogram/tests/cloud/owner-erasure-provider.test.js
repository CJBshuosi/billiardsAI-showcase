const { EventEmitter } = require('events')
const { Readable } = require('stream')
const { digest, documentId } = require('../../cloudfunctions/shared/shot-ownership')
const providerPath = '../../cloudfunctions/sync-api/lib/owner-erasure-provider'
// Network and IAM seams only: signer, parser, transport bounds and state machine are real.
const credentials = { SecretId: 'controlled-id', SecretKey: 'controlled-secret', Token: 'controlled-token' }
const configuration = { environmentId: 'env-test', bucket: 'clips-123456', region: 'ap-shanghai' }
const operationId = '12345678-1234-4123-8123-123456789abc'
const key = `managed-clips-v1/${'a'.repeat(64)}/staging/${operationId}.mp4`
const target = { sequence: 1, operation_id: operationId, object_id: `key-${documentId(key)}`, kind: 'staging', cloud_path: key,
  file_id: `cloud://env-test.clips-123456/${key}`, writer_state: 'upload_succeeded', receipt_sha256: 'b'.repeat(64) }
const mode = value => ({ status: 200, body: value ? `<VersioningConfiguration><Status>${value}</Status></VersioningConfiguration>` : '<VersioningConfiguration/>' })
const deleted = { status: 204, body: '' }
const missing = { status: 404, body: `<Error><Code>NoSuchKey</Code><Resource>/${key}</Resource><RequestId>test</RequestId></Error>` }
const row = (id, resource = key, marker = false) => `<${marker ? 'DeleteMarker' : 'Version'}><Key>${resource}</Key><VersionId>${id}</VersionId><IsLatest>false</IsLatest><LastModified>2026-11-01T00:00:00Z</LastModified>${marker ? '' : '<ETag>test</ETag><Size>1</Size><StorageClass>STANDARD</StorageClass>'}</${marker ? 'DeleteMarker' : 'Version'}>`
function listing(rows = '', { marker = '', version = '', next, nextVersion = '', truncated = false, extra = '' } = {}) {
  return { status: 200, body: `<ListVersionsResult><Name>clips-123456</Name><Prefix>${key}</Prefix><KeyMarker>${marker}</KeyMarker><VersionIdMarker>${version}</VersionIdMarker><MaxKeys>16</MaxKeys><IsTruncated>${truncated}</IsTruncated>${rows}${next === undefined ? '' : `<NextKeyMarker>${next}</NextKeyMarker><NextVersionIdMarker>${nextVersion}</NextVersionIdMarker>`}${extra}</ListVersionsResult>` }
}
function harness(responses, additions = {}) {
  const sent = []; const streams = []; const requests = []
  const request = (options, callback) => {
    sent.push(options)
    const response = responses.shift()
    const req = new EventEmitter(); requests.push(req)
    req.destroy = jest.fn(() => req.emit('error', new Error('network-private-error')))
    req.end = () => queueMicrotask(() => {
      if (!response || response.hang) return
      if (response.error) { req.emit('error', new Error(response.error)); return }
      const stream = response.stream || Readable.from(response.chunks || [Buffer.from(response.body || '')])
      stream.statusCode = response.status; stream.headers = response.headers || {}; stream.rawHeaders = response.rawHeaders || []
      streams.push(stream); callback(stream)
    })
    return req
  }
  const { createOwnerErasureProvider } = require(providerPath)
  return { provider: createOwnerErasureProvider({ ...configuration, getCredentials: async () => credentials, request, clock: () => Date.UTC(2026, 10, 1), ...additions }), sent, streams, requests }
}
async function run(h, count, value = target, checkpoint = null) {
  let result
  for (let i = 0; i < count; i++) {
    const before = h.sent.length
    result = await h.provider.step({ target: value, checkpoint })
    expect(h.sent.length - before).toBeLessThanOrEqual(1)
    if (i < count - 1) expect(result.status).toBe('pending')
    checkpoint = result.checkpoint
  }
  return result
}
function query(options) { return new URL(`https://example.test${options.path}`).searchParams }

describe('bounded exact-target COS erasure provider', () => {
  test('never-enabled needs DELETE, origin NoSuchKey and versioning recheck before absence', async () => {
    const h = harness([mode(), deleted, missing, mode()])
    let result = await run(h, 3)
    expect(result.status).toBe('pending')
    result = await h.provider.step({ target, checkpoint: result.checkpoint })
    expect(result.status).toBe('absent')
    expect(h.sent.map(o => o.method)).toEqual(['GET', 'DELETE', 'GET', 'GET'])
    expect(h.sent[2].headers.Range).toBe('bytes=0-0')
    expect(result.evidence).toMatchObject({ target_sha256: digest(target), versioning_mode: 'NeverEnabled', observed_at: '2026-11-01T00:00:00.000Z' })
    expect(result.evidence.transcript_sha256).toMatch(/^[a-f0-9]{64}$/u)
    expect(JSON.stringify(result)).not.toMatch(/controlled-|https:|Authorization/)
  })
  test.each(['Enabled', 'Suspended'])('%s deletes explicit versions and markers including null, then restarts whole enumeration', async value => {
    const h = harness([mode(value), listing(row('null') + row('delete-marker', key, true) + row('neighbor', `${key}.neighbor`)), deleted, deleted,
      listing(), missing, mode(value)])
    const result = await run(h, 7)
    expect(result.status).toBe('absent')
    expect(h.sent.filter(o => o.method === 'DELETE').map(o => [o.path.split('?')[0], query(o).get('versionId')])).toEqual([[`/${key}`, 'null'], [`/${key}`, 'delete-marker']])
    expect(h.sent[4].path).toBe(h.sent[1].path)
    expect(query(h.sent[1]).get('max-keys')).toBe('16')
  })
  test.each(['Enabled', 'Suspended'])('%s handles official quoted ETags and standard XML escapes on exact and neighbor versions', async value => {
    const encodedId = 'v&amp;&lt;&gt;&quot;&apos;&#65;&#x1F642;'
    const officialRow = (id, resource) => row(id, resource).replace('<ETag>test</ETag>', '<ETag>&quot;d41d8cd98f00b204e9800998ecf8427e&quot;</ETag>')
      .replace('</Version>', '<Owner><ID>123456</ID><DisplayName>A&amp;B&lt;C&gt;&quot;D&quot;&apos;E&#70;&#x47;</DisplayName></Owner></Version>')
    const h = harness([mode(value), listing(officialRow(encodedId, key) + officialRow('neighbor', `${key}.neighbor`)), deleted,
      listing(), missing, mode(value)])
    expect((await run(h, 6)).status).toBe('absent')
    expect(h.sent.filter(options => options.method === 'DELETE').map(options => [options.path.split('?')[0], query(options).get('versionId')]))
      .toEqual([[`/${key}`, 'v&<>"\'A🙂']])
  })
  test.each(['FREQUENT', 'INFREQUENT'])('handles documented intelligent tier %s on exact and neighbor versions', async tier => {
    const tiered = resource => row('version', resource).replace('<StorageClass>STANDARD</StorageClass>', `<StorageClass>INTELLIGENT_TIERING</StorageClass><StorageTier>${tier}</StorageTier>`)
    const h = harness([mode('Enabled'), listing(tiered(key) + tiered(`${key}.neighbor`)), deleted, listing(), missing, mode('Enabled')])
    expect((await run(h, 6)).status).toBe('absent')
    expect(h.sent.filter(options => options.method === 'DELETE').map(options => [options.path.split('?')[0], query(options).get('versionId')]))
      .toEqual([[`/${key}`, 'version']])
  })
  test('StorageTier remains optional for an intelligent-tiering version', async () => {
    const tiered = row('version').replace('<StorageClass>STANDARD</StorageClass>', '<StorageClass>INTELLIGENT_TIERING</StorageClass>')
    const h = harness([mode('Enabled'), listing(tiered), deleted, listing(), missing, mode('Enabled')])
    expect((await run(h, 6)).status).toBe('absent')
  })
  test.each(['&external;', '&nbsp;', '&#0;', '&#xD800;', '&#x110000;', '&#xFFFE;', '&#x;', '&#-1;', '&#xGG;', '&amp', '&AMP;', '&#999999999999999;'])('unsafe/malformed XML reference %s remains pending', async reference => {
    const h = harness([mode('Enabled'), listing(row(reference))]); const result = await run(h, 2)
    expect(result.status).toBe('pending'); expect(result.evidence.code).toBe('OWNER_ERASURE_XML')
    expect(result.checkpoint.phase).toBe('list'); expect(h.sent.some(options => options.method === 'DELETE')).toBe(false)
  })
  test('numeric references retain the decoded scalar bound', async () => {
    const h = harness([mode('Enabled'), listing(row('&#65;'.repeat(1025)))]); const result = await run(h, 2)
    expect(result.evidence.code).toBe('OWNER_ERASURE_XML'); expect(result.checkpoint.phase).toBe('list')
  })
  test.each([
    '<StorageClass>STANDARD</StorageClass><StorageTier>FREQUENT</StorageTier>',
    '<StorageTier>FREQUENT</StorageTier>',
    '<StorageClass>INTELLIGENT_TIERING</StorageClass><StorageTier>ARCHIVE</StorageTier>',
    '<StorageClass>INTELLIGENT_TIERING</StorageClass><StorageTier>frequent</StorageTier>',
    '<StorageClass>INTELLIGENT_TIERING</StorageClass><StorageTier/>',
    '<StorageClass>INTELLIGENT_TIERING</StorageClass><StorageTier>FREQUENT</StorageTier><StorageTier>INFREQUENT</StorageTier>',
    '<StorageClass>INTELLIGENT_TIERING</StorageClass><StorageTier><Value>FREQUENT</Value></StorageTier>',
  ])('invalid StorageTier relationship/shape stays pending for exact and neighbor rows: %s', async metadata => {
    for (const resource of [key, `${key}.neighbor`]) {
      const invalid = row('version', resource).replace('<StorageClass>STANDARD</StorageClass>', metadata)
      const h = harness([mode('Enabled'), listing(invalid)]); const result = await run(h, 2)
      expect(result.status).toBe('pending'); expect(result.evidence.code).toBe('OWNER_ERASURE_XML')
      expect(result.checkpoint.phase).toBe('list'); expect(h.sent.some(options => options.method === 'DELETE')).toBe(false)
    }
  })
  test('persists both page cursors including empty version marker, never deletes prefix neighbors', async () => {
    const neighbor = `${key}.neighbor`
    const h = harness([mode('Enabled'), listing(row('n', neighbor), { truncated: true, next: neighbor, nextVersion: '' }),
      listing('', { marker: neighbor }), missing, mode('Enabled')])
    const result = await run(h, 5)
    expect(result.status).toBe('absent')
    expect(query(h.sent[2]).get('key-marker')).toBe(neighbor)
    expect(query(h.sent[2]).has('version-id-marker')).toBe(true)
    expect(query(h.sent[2]).get('version-id-marker')).toBe('')
    expect(h.sent.some(o => o.method === 'DELETE')).toBe(false)
  })
  test('a changed versioning mode restarts version sweep and crash replay deletes the same immutable version', async () => {
    const h = harness([mode(), deleted, missing, mode('Suspended'), listing(row('v1')), deleted, deleted, listing(), missing, mode('Suspended')])
    let result = await run(h, 5)
    const replay = JSON.parse(JSON.stringify(result.checkpoint))
    await h.provider.step({ target, checkpoint: replay })
    result = await h.provider.step({ target, checkpoint: replay })
    expect(h.sent[5].path).toBe(h.sent[6].path)
    result = await run(h, 3, target, result.checkpoint)
    expect(result.status).toBe('absent')
  })
  test.each([null, 'cloud://foreign.clips-123456/a', `cloud://env-test.foreign/${key}`, `cloud://opaque-id`])('rejects unknown fileID binding %s before IAM or network', async file_id => {
    const iam = jest.fn(); const h = harness([], { getCredentials: iam })
    await expect(h.provider.step({ target: { ...target, file_id }, checkpoint: null })).rejects.toMatchObject({ code: 'OWNER_ERASURE_INVALID_TARGET' })
    expect(iam).not.toHaveBeenCalled(); expect(h.sent).toHaveLength(0)
  })
  test('rejects modified namespace/object/operation/shape and cross-target checkpoints', async () => {
    const h = harness([mode()]); const result = await run(h, 1)
    for (const other of [{ ...target, cloud_path: 'arbitrary.mp4' }, { ...target, object_id: 'key-wrong' }, { ...target, operation_id: 'unknown' }, { ...target, extra: true }]) {
      await expect(h.provider.step({ target: other, checkpoint: null })).rejects.toMatchObject({ code: 'OWNER_ERASURE_INVALID_TARGET' })
    }
    await expect(h.provider.step({ target: { ...target, sequence: 2 }, checkpoint: result.checkpoint })).rejects.toMatchObject({ code: 'OWNER_ERASURE_INVALID_CHECKPOINT' })
    for (const checkpoint of [{ ...result.checkpoint, extra: true }, { ...result.checkpoint, phase: 'complete' }, { ...result.checkpoint, versions: Array(17).fill('v') }]) {
      await expect(h.provider.step({ target, checkpoint })).rejects.toMatchObject({ code: 'OWNER_ERASURE_INVALID_CHECKPOINT' })
    }
    expect(h.sent).toHaveLength(1)
  })
  test.each([
    { status: 404, body: '' }, { status: 404, body: `<Error><Code>NoSuchBucket</Code><Resource>/${key}</Resource></Error>` },
    { status: 404, body: '<Error><Code>NoSuchKey</Code><Resource>/foreign</Resource></Error>' },
    { status: 403, body: '<Error><Code>AccessDenied</Code></Error>' }, { status: 503, body: 'private' }, { status: 302, body: '' },
  ])('ambiguous/errors remain pending at current probe: $status $body', async response => {
    const h = harness([mode(), deleted, response]); const before = await run(h, 2)
    const result = await h.provider.step({ target, checkpoint: before.checkpoint })
    expect(result.status).toBe('pending'); expect(result.evidence.code).toMatch(/^OWNER_ERASURE_/u)
    expect(result.checkpoint.phase).toBe(before.checkpoint.phase)
  })
  test.each(['', '<Unexpected/>', '<VersioningConfiguration><Status>Unknown</Status></VersioningConfiguration>', '<VersioningConfiguration><Status>Enabled</Status><Status>Suspended</Status></VersioningConfiguration>',
    '<!DOCTYPE x [<!ENTITY x "Enabled">]><VersioningConfiguration><Status>&x;</Status></VersioningConfiguration>',
    '<VersioningConfiguration><Status>Enabled</VersioningConfiguration>', '<VersioningConfiguration><Extra/></VersioningConfiguration>'])('invalid versioning XML never normalizes into never-enabled: %s', async body => {
    const h = harness([{ status: 200, body }]); const result = await run(h, 1)
    expect(result.status).toBe('pending'); expect(result.checkpoint.phase).toBe('versioning')
  })
  test.each([
    listing('', { truncated: true }), listing('', { truncated: true, next: '', nextVersion: '' }),
    listing('', { next: key }), listing('', { extra: '<CommonPrefixes><Prefix>foreign</Prefix></CommonPrefixes>' }),
    listing(row('v').repeat(17)), listing(row('v') + row('v')), listing().body.replace('<Name>clips-123456</Name>', '<Name>foreign</Name>'),
  ])('malformed or contradictory listing stays pending', async input => {
    const h = harness([mode('Enabled'), typeof input === 'string' ? { status: 200, body: input } : input])
    const result = await run(h, 2); expect(result.status).toBe('pending'); expect(result.checkpoint.phase).toBe('list')
  })
  test('repeated or decreasing cursors cannot complete', async () => {
    const h = harness([mode('Enabled'), listing('', { truncated: true, next: key, nextVersion: 'v1' }),
      listing('', { marker: key, version: 'v1', truncated: true, next: key, nextVersion: 'v1' })])
    const before = await run(h, 2); const result = await h.provider.step({ target, checkpoint: before.checkpoint })
    expect(result.status).toBe('pending'); expect(result.evidence.code).toBe('OWNER_ERASURE_CURSOR')
    expect(result.checkpoint).toEqual(before.checkpoint)
  })
  test('continues beyond128 pages with bounded recent loop witness and exact current cursor', async () => {
    const pages = Array.from({ length: 130 }, (_, i) => listing('', { marker: i ? `${key}.${String(i - 1).padStart(3, '0')}` : '', truncated: true, next: `${key}.${String(i).padStart(3, '0')}` }))
    const h = harness([mode('Enabled'), ...pages, listing('', { marker: `${key}.129` }), missing, mode('Enabled')])
    const result = await run(h, 131)
    expect(result.status).toBe('pending'); expect(result.evidence.code).toBeUndefined()
    expect(result.checkpoint.cursor).toEqual({ key: `${key}.129`, version: '' })
    expect(result.checkpoint.cursor_sha256).toHaveLength(128)
    expect(Buffer.byteLength(JSON.stringify(result.checkpoint))).toBeLessThan(20000)
    const restored = JSON.parse(JSON.stringify(result.checkpoint))
    expect((await run(h, 3, target, restored)).status).toBe('absent')
    expect(h.sent).toHaveLength(134)
  })
  test.each([200, 206, 416])('range presence status %i immediately aborts media without reading it', async status => {
    const stream = new Readable({ read() { throw new Error('media must never be read') } })
    const h = harness([mode(), deleted, { status, stream }]); const result = await run(h, 3)
    expect(result.status).toBe('pending'); expect(stream.destroyed).toBe(true)
    expect(h.sent[2].headers.Range).toBe('bytes=0-0')
  })
  test.each([
    { status: 200, chunks: [Buffer.alloc(256 * 1024), Buffer.from('x')], code: 'OWNER_ERASURE_BOUNDS' },
    { status: 404, chunks: [Buffer.alloc(8193)], code: 'OWNER_ERASURE_BOUNDS' },
    { status: 200, body: '<VersioningConfiguration/>', headers: { 'content-length': '262145' }, code: 'OWNER_ERASURE_BOUNDS' },
    { status: 200, body: '<VersioningConfiguration/>', rawHeaders: ['x-private', 'x'.repeat(8192)], code: 'OWNER_ERASURE_BOUNDS' },
    { status: 200, body: '<VersioningConfiguration/>', headers: { 'content-encoding': 'gzip' }, code: 'OWNER_ERASURE_BOUNDS' },
    { status: 200, body: `${'<x>'.repeat(40)}${'</x>'.repeat(40)}`, code: 'OWNER_ERASURE_XML' },
  ])('bounds response bytes/headers/parser work without success', async response => {
    const h = harness([response]); const result = await run(h, 1)
    expect(result.status).toBe('pending'); expect(result.evidence.code).toBe(response.code); expect(h.streams[0].destroyed).toBe(true)
  })
  test('absolute 10-second deadline covers headers and streaming body, one request and sanitized error', async () => {
    jest.useFakeTimers()
    try {
      for (const response of [{ hang: true }, { status: 200, stream: new Readable({ read() {} }) }]) {
        const h = harness([response]); const result = h.provider.step({ target, checkpoint: null })
        await jest.advanceTimersByTimeAsync(10001)
        expect((await result).evidence.code).toBe('OWNER_ERASURE_TIMEOUT')
        expect(h.sent).toHaveLength(1); expect(h.requests[0].destroy).toHaveBeenCalled()
      }
    } finally { jest.useRealTimers() }
  })
  test('IAM resolution is lazy and within the same absolute deadline', async () => {
    jest.useFakeTimers()
    try {
      const h = harness([], { getCredentials: () => new Promise(() => {}) })
      const work = h.provider.step({ target, checkpoint: null }); await jest.advanceTimersByTimeAsync(10001)
      expect((await work).evidence.code).toBe('OWNER_ERASURE_TIMEOUT'); expect(h.sent).toHaveLength(0)
    } finally { jest.useRealTimers() }
  })
  test('signs exact host/path/query/security token with official signer, fixed TLS origin and no fallback', async () => {
    const h = harness([mode('Enabled'), listing(row('a+/= b')), deleted]); await run(h, 3)
    const options = h.sent[2]; const COS = require('cos-nodejs-sdk-v5')
    expect(options).toMatchObject({ hostname: 'clips-123456.cos.ap-shanghai.myqcloud.com', protocol: 'https:', port: 443, rejectUnauthorized: true, minVersion: 'TLSv1.2', agent: false, maxHeaderSize: 8192 })
    expect(query(options).get('versionId')).toBe('a+/= b')
    expect(options.headers.Authorization).toBe(COS.getAuthorization({ ...credentials, Method: 'DELETE', Pathname: `/${key}`, Query: { versionId: 'a+/= b' }, Headers: { Host: options.hostname, 'x-cos-security-token': credentials.Token }, KeyTime: '1793491200;1793491260' }))
    expect(options.headers.Authorization).toContain('q-url-param-list=versionid')
    expect(options.headers.Authorization).toContain('host;x-cos-security-token')
  })
  test('construction/import does no IAM/network and invalid fixed configuration is rejected', () => {
    const iam = jest.fn(); const h = harness([], { getCredentials: iam }); expect(iam).not.toHaveBeenCalled(); expect(h.sent).toHaveLength(0)
    const { createOwnerErasureProvider } = require(providerPath)
    expect(() => createOwnerErasureProvider()).toThrow()
    for (const change of [{ bucket: 'evil.test/x' }, { region: 'ap-shanghai.evil' }, { environmentId: 'env/foreign' }]) {
      expect(() => createOwnerErasureProvider({ ...configuration, ...change })).toThrow(expect.objectContaining({ code: 'OWNER_ERASURE_INVALID_CONFIG' }))
    }
  })
  test('default credentials fail closed without invoking default HTTPS or native role context', async () => {
    const https = require('https'); const network = jest.spyOn(https, 'request').mockImplementation(() => { throw new Error('unexpected network') })
    try {
      const { createOwnerErasureProvider } = require(providerPath)
      const provider = createOwnerErasureProvider(configuration)
      expect((await provider.step({ target })).evidence.code).toBe('OWNER_ERASURE_CREDENTIALS')
      expect(network).not.toHaveBeenCalled()
    } finally { network.mockRestore() }
  })
  test('strict checkpoint validation rejects forged state shapes with current target digest', async () => {
    const h = harness([mode('Enabled')]); const start = await run(h, 1)
    for (const checkpoint of [
      { ...start.checkpoint, transcript_sha256: Symbol('private') },
      { ...start.checkpoint, phase: 'delete', versions: [] },
      { ...start.checkpoint, mode: 'NeverEnabled', phase: 'list' },
      { ...start.checkpoint, cursor: { key: 'foreign', version: '' }, pages: 1, cursor_sha256: [digest({ key: 'foreign', version: '' })] },
      { ...start.checkpoint, cursor: { key, version: 'v' }, pages: 1, cursor_sha256: ['a'.repeat(64)] },
      { ...start.checkpoint, phase: 'recheck', cursor: { key, version: '' }, pages: 1, cursor_sha256: [digest({ key, version: '' })] },
    ]) {
      await expect(h.provider.step({ target, checkpoint })).rejects.toMatchObject({ code: 'OWNER_ERASURE_INVALID_CHECKPOINT' })
    }
    expect(h.sent).toHaveLength(1)
  })
  test('checkpoint arrays reject accessors before reading their contents', async () => {
    const h = harness([mode('Enabled')]); const start = await run(h, 1)
    const getter = jest.fn(() => { throw new Error('private-checkpoint-error') }); const versions = []
    Object.defineProperty(versions, '0', { get: getter, enumerable: true })
    await expect(h.provider.step({ target, checkpoint: { ...start.checkpoint, phase: 'delete', versions } }))
      .rejects.toMatchObject({ code: 'OWNER_ERASURE_INVALID_CHECKPOINT' })
    expect(getter).not.toHaveBeenCalled()
  })
  test('contradictory page rows before the requested cursor remain pending', async () => {
    const cursorKey = `${key}.neighbor`
    const h = harness([mode('Enabled'), listing('', { truncated: true, next: cursorKey }),
      listing(row('before', key), { marker: cursorKey })])
    const before = await run(h, 2); const result = await h.provider.step({ target, checkpoint: before.checkpoint })
    expect(result.evidence.code).toBe('OWNER_ERASURE_CURSOR')
    expect(result.checkpoint).toEqual(before.checkpoint)
  })
  test('next cursor cannot precede returned rows', async () => {
    const h = harness([mode('Enabled'), listing(row('neighbor', `${key}.z`), { truncated: true, next: `${key}.a` })])
    const result = await run(h, 2)
    expect(result.evidence.code).toBe('OWNER_ERASURE_CURSOR'); expect(result.checkpoint.pages).toBe(0)
  })
  test.each([deleted, { status: 200, body: '<Error><Code>AccessDenied</Code></Error>' }, { status: 404, body: `<Error><Code>NoSuchBucket</Code><Resource>/${key}</Resource></Error>` }])('delete result alone never establishes absence', async response => {
    const h = harness([mode('Enabled'), listing(row('v1')), response]); const result = await run(h, 3)
    expect(result.status).toBe('pending')
    if (response !== deleted) expect(result.checkpoint.phase).toBe('delete')
  })
  test('IAM late resolution after timeout cannot start a late network operation', async () => {
    jest.useFakeTimers()
    try {
      let resolve; const h = harness([], { getCredentials: () => new Promise(done => { resolve = done }) })
      const work = h.provider.step({ target }); await jest.advanceTimersByTimeAsync(10001)
      expect((await work).evidence.code).toBe('OWNER_ERASURE_TIMEOUT')
      resolve(credentials); await Promise.resolve(); await Promise.resolve()
      expect(h.sent).toHaveLength(0)
    } finally { jest.useRealTimers() }
  })
  test('namespaced official XML is accepted; output is detached from input checkpoints', async () => {
    const h = harness([{ status: 200, body: '<?xml version="1.0" encoding="UTF-8"?><VersioningConfiguration xmlns="http://doc.s3.amazonaws.com/2006-03-01"><Status>Enabled</Status></VersioningConfiguration>' }, listing()])
    const first = await run(h, 1); const persisted = JSON.parse(JSON.stringify(first.checkpoint))
    const second = await h.provider.step({ target, checkpoint: persisted })
    second.checkpoint.cursor_sha256.push('bad'); expect(persisted).toEqual(first.checkpoint)
    expect(second.checkpoint.phase).toBe('version_probe')
  })
  test('a versioning-mode change after version absence restarts the full sweep', async () => {
    const h = harness([mode('Enabled'), listing(), missing, mode('Suspended'), listing(row('new-version')), deleted, listing(), missing, mode('Suspended')])
    const transition = await run(h, 4)
    expect(transition.status).toBe('pending'); expect(transition.checkpoint.phase).toBe('list')
    expect((await run(h, 5, target, transition.checkpoint)).status).toBe('absent')
  })
  test('version marker current404 cannot bypass existing versions; bucket becoming never-enabled remains pending', async () => {
    const h = harness([mode('Enabled'), listing(row('older-version')), deleted, listing(), missing, mode()])
    const result = await run(h, 6)
    expect(result.status).toBe('pending'); expect(result.evidence.code).toBe('OWNER_ERASURE_NATIVE')
    expect(result.checkpoint.phase).toBe('recheck'); expect(result.checkpoint.mode).toBe('Enabled')
  })
  test('one-byte chunks cannot evade body bounds, and truncated transport cannot be parsed as absence', async () => {
    const raw = '<VersioningConfiguration/>'
    const h = harness([{ status: 200, chunks: Array.from(Buffer.from(raw), byte => Buffer.from([byte])), headers: { 'content-length': String(Buffer.byteLength(raw)) } }])
    expect((await run(h, 1)).checkpoint.phase).toBe('simple_delete')
    const short = harness([{ status: 200, body: raw, headers: { 'content-length': '100' } }])
    expect((await run(short, 1)).evidence.code).toBe('OWNER_ERASURE_TRANSPORT')
  })
  test('metadata exactly256KiB is bounded and a valid document, one more byte fails', async () => {
    const start = '<VersioningConfiguration>'; const end = '</VersioningConfiguration>'
    const body = start + ' '.repeat(256 * 1024 - start.length - end.length) + end
    const h = harness([{ status: 200, body }]); const result = await run(h, 1)
    expect(result.checkpoint.phase).toBe('simple_delete')
    expect(Buffer.byteLength(body)).toBe(262144)
  })
})

describe('per-step durable fence', () => {
  test.each([() => false, async () => false, async () => { throw new Error('private authority detail') }])('a denied/rejected fence prevents native issue', async beforeRequest => {
    const h = harness([mode()]); const result = await h.provider.step({ target, beforeRequest })
    expect(result).toMatchObject({ status: 'pending', evidence: { code: 'OWNER_ERASURE_FENCE' } })
    expect(h.sent).toHaveLength(0); expect(JSON.stringify(result)).not.toContain('private authority detail')
  })
  test('fence is once per invocation after IAM and before request, never serialized', async () => {
    const events = []; const h = harness([mode()], { getCredentials: async () => { events.push('iam'); return credentials } })
    const result = await h.provider.step({ target, beforeRequest: async () => { events.push(`fence:${h.sent.length}`); return true } })
    expect(events).toEqual(['iam', 'fence:0']); expect(h.sent).toHaveLength(1)
    expect(JSON.stringify(result)).not.toContain('beforeRequest')
  })
  test('a fence resolving after the absolute deadline cannot issue a late request', async () => {
    jest.useFakeTimers()
    try {
      let release; const gate = new Promise(resolve => { release = resolve }); const h = harness([mode()])
      const pending = h.provider.step({ target, beforeRequest: () => gate })
      await jest.advanceTimersByTimeAsync(10001)
      expect((await pending).evidence.code).toBe('OWNER_ERASURE_TIMEOUT')
      release(true); await jest.advanceTimersByTimeAsync(0); expect(h.sent).toHaveLength(0)
    } finally { jest.useRealTimers() }
  })
})
