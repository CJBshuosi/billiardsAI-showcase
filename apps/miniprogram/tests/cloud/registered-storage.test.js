const crypto = require('crypto')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { Readable } = require('stream')
const { createRegisteredStorageProvider } = require('../../cloudfunctions/sync-api/lib/registered-storage')
const { createCloudBaseStorageProvider, createStoragePort } = require('../../cloudfunctions/sync-api/lib/storage')
const staging = `managed-clips-v1/${'a'.repeat(64)}/staging/11111111-1111-4111-8111-111111111111.mp4`
const published = `published/managed-clips-v1/${'a'.repeat(64)}/22222222-2222-4222-8222-222222222222.mp4`
const fileID = 'cloud://opaque/staging'
function mp4(payload = 'video') {
  return Buffer.concat([Buffer.from([0, 0, 0, 24]), Buffer.from('ftypisom'), Buffer.alloc(4), Buffer.from('isommp42'), Buffer.from(payload)])
}
function constraints(bytes) {
  return { staging_file_id: fileID, publish_object_key: published, byte_size: bytes.length,
    content_sha256: crypto.createHash('sha256').update(bytes).digest('hex'), media_type: 'video/mp4' }
}
function response(bytes, extra = {}) {
  return Object.assign(Readable.from((function* () { for (let i = 0; i < bytes.length; i += 7) yield bytes.subarray(i, i + 7) })()), { statusCode: 200, headers: {}, ...extra })
}
let root
beforeEach(() => { root = fs.mkdtempSync(path.join(os.tmpdir(), 'registered-test-')) })
afterEach(() => { fs.rmSync(root, { recursive: true, force: true }); jest.restoreAllMocks() })
function sdk(bytes, upload) {
  return {
    getUploadMetadata: jest.fn(async () => ({ data: { fileId: fileID, url: 'https://storage.invalid/put', token: 'secret-token', authorization: 'secret-signature', cosFileId: 'cos-id' } })),
    getTempFileURL: jest.fn(async () => ({ fileList: [{ fileID, code: 'SUCCESS', tempFileURL: 'https://storage.invalid/get' }] })),
    downloadFile: jest.fn(() => { throw Error('registered read must not use buffered download') }),
    uploadFile: jest.fn(upload || (async ({ fileContent }) => {
      expect(fileContent).toBeInstanceOf(fs.ReadStream)
      const chunks = []; for await (const chunk of fileContent) chunks.push(chunk)
      expect(Buffer.concat(chunks)).toEqual(bytes)
      return { fileID: 'cloud://opaque/published' }
    })),
  }
}

test('registered grant retains observed fileID with malformed credentials and sanitizes invocation failures', async () => {
  const cloud = sdk(mp4())
  const provider = createRegisteredStorageProvider(cloud, { temporaryRoot: root })
  const input = { object_key: staging, ...constraints(mp4()) }
  expect(await provider.authorizeRegisteredUpload(input)).toEqual({ outcome: 'external_issued', file_id: fileID, authorization: {
    method: 'PUT', url: 'https://storage.invalid/put', headers: { Signature: 'secret-signature', authorization: 'secret-signature', 'x-cos-security-token': 'secret-token', 'x-cos-meta-fileid': 'cos-id', key: encodeURIComponent(staging) }, expires_in_seconds: 300,
  } })
  cloud.getUploadMetadata.mockResolvedValueOnce({ data: { fileId: fileID, url: 'http://unsafe', token: 'secret' } })
  expect(await provider.authorizeRegisteredUpload(input)).toEqual({ outcome: 'unknown', reason: 'provider_error', file_id: fileID })
  cloud.getUploadMetadata.mockRejectedValueOnce(Error('secret URL and credentials'))
  expect(await provider.authorizeRegisteredUpload(input)).toEqual({ outcome: 'unknown', reason: 'provider_error' })
})

test('registered copy uses registered ID and exact owned readonly stream after staging changes', async () => {
  const bytes = mp4()
  let staged = bytes
  const cloud = sdk(bytes, async ({ cloudPath, fileContent }) => {
    expect(cloudPath).toBe(published)
    expect(fileContent).toBeInstanceOf(fs.ReadStream)
    expect(fs.fstatSync(fileContent.fd).mode & 0o777).toBe(0o600)
    staged = mp4('replaced')
    const chunks = []; for await (const part of fileContent) chunks.push(part)
    expect(Buffer.concat(chunks)).toEqual(bytes)
    return { fileID: 'cloud://opaque/published' }
  })
  const provider = createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(staged) })
  expect(await provider.copyRegisteredObject(constraints(bytes))).toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
  expect(cloud.getTempFileURL).toHaveBeenCalledWith({ fileList: [fileID] })
  expect(cloud.getUploadMetadata).not.toHaveBeenCalled()
  expect(cloud.downloadFile).not.toHaveBeenCalled()
  expect(fs.readdirSync(root)).toEqual([])
})

test('keeps the original spool inode allocated until the readonly descriptor is pinned', async () => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const realOpen = fs.promises.open.bind(fs.promises)
  let original; let pinnedWhileAllocated = false
  jest.spyOn(fs.promises, 'open').mockImplementation(async (...args) => {
    if (args[1] & fs.constants.O_WRONLY) {
      original = await realOpen(...args)
      return original
    }
    // A live descriptor prevents Linux from recycling the original inode after
    // unlink/replacement, so dev+ino identity remains meaningful at handoff.
    pinnedWhileAllocated = original.fd !== -1
    return realOpen(...args)
  })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
  expect(pinnedWhileAllocated).toBe(true)
  expect(original.fd).toBe(-1)
  expect(fs.readdirSync(root)).toEqual([])
})

test.each([staging, published])('legacy providers and ports refuse reserved key %s before SDK activity', async object_key => {
  const cloud = sdk(mp4())
  const provider = createCloudBaseStorageProvider(cloud)
  const input = { object_key, ...constraints(mp4()) }
  await expect(provider.createPresignedUpload(input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  await expect(provider.verifyPrivateObject(input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  const legacy = { createPresignedUpload: jest.fn(), verifyPrivateObject: jest.fn() }
  const independentPort = createStoragePort(legacy)
  await expect(independentPort.authorizeUpload(input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  await expect(independentPort.verifyObject(input)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  await expect(createStoragePort(provider).verifyObject({ ...input, object_key: 'legacy.mp4', publish_object_key: object_key })).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(cloud.getUploadMetadata).not.toHaveBeenCalled()
  expect(legacy.createPresignedUpload).not.toHaveBeenCalled()
  expect(legacy.verifyPrivateObject).not.toHaveBeenCalled()
})

test.each([
  ['short EOF', b => response(b.subarray(0, -1)), {}, true],
  ['overlong', b => response(Buffer.concat([b, Buffer.from('x')])), {}, true],
  ['wrong hash', b => response(b), { content_sha256: '0'.repeat(64) }, true],
  ['content length mismatch', b => response(b, { headers: { 'content-length': '999' } }), {}, false],
  ['redirect', b => response(b, { statusCode: 302 }), {}, false],
  ['error', b => Object.assign(Readable.from((async function* () { yield b.subarray(0, 4); throw Error('secret failure') })()), { statusCode: 200, headers: {} }), {}, false],
  ['aborted', b => response(b, { aborted: true }), {}, false],
])('refuses %s before any write and cleans its owned spool', async (_label, stream, override, permanent) => {
  const bytes = mp4()
  const cloud = sdk(bytes)
  const provider = createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => stream(bytes) })
  expect(await provider.copyRegisteredObject({ ...constraints(bytes), ...override })).toEqual({ outcome: 'no_write', reason: 'not_started', permanent })
  expect(cloud.uploadFile).not.toHaveBeenCalled()
  expect(fs.readdirSync(root)).toEqual([])
})

test.each(['http://storage.invalid/a', 'https://user:secret@storage.invalid/a', 'https://storage.invalid/a#fragment', 'https://storage.invalid/a#', 'https://@storage.invalid/a'])('rejects unsafe temporary URL %s', async tempFileURL => {
  const bytes = mp4(); const cloud = sdk(bytes); const openDownload = jest.fn()
  cloud.getTempFileURL.mockResolvedValueOnce({ fileList: [{ fileID, code: 'SUCCESS', tempFileURL }] })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(openDownload).not.toHaveBeenCalled()
})

test.each([
  [{ fileID: 'cloud://other', code: 'SUCCESS', tempFileURL: 'https://storage.invalid/a' }],
  [{ fileID, code: 'FAILED', tempFileURL: 'https://storage.invalid/a' }],
  Array(2).fill({ fileID, code: 'SUCCESS', tempFileURL: 'https://storage.invalid/a' }),
])('requires one exact successful registered URL row', async fileList => {
  const bytes = mp4(); const cloud = sdk(bytes); const openDownload = jest.fn()
  cloud.getTempFileURL.mockResolvedValueOnce({ fileList })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(openDownload).not.toHaveBeenCalled()
})

test('waits for download EOF even after all expected bytes arrive', async () => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const body = Object.assign(new Readable({ read() {} }), { statusCode: 200, headers: {} })
  const pending = createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => body }).copyRegisteredObject(constraints(bytes))
  body.push(bytes)
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(cloud.uploadFile).not.toHaveBeenCalled()
  body.push(null)
  expect(await pending).toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
})

test.each(['bad brand', 'misalignment', 'incomplete box', 'zero size', 'huge extended'])('rejects MP4 %s with exact hash', async kind => {
  const bytes = mp4()
  if (kind === 'bad brand') { bytes.write('zzzz', 8); bytes.write('zzzzzzzz', 16) }
  if (kind === 'misalignment') bytes.writeUInt32BE(23)
  if (kind === 'incomplete box') bytes.writeUInt32BE(100)
  if (kind === 'zero size') bytes.writeUInt32BE(0)
  if (kind === 'huge extended') { bytes.writeUInt32BE(1); bytes.writeBigUInt64BE(2n ** 63n, 8) }
  const cloud = sdk(bytes)
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: true })
  expect(cloud.uploadFile).not.toHaveBeenCalled()
})

test.each([false, true])('accepts a large complete ftyp with a late compatible brand (extended=%s)', async extended => {
  const header = extended ? 16 : 8
  const bytes = Buffer.alloc(128 * 1024 + header + 8)
  bytes.writeUInt32BE(extended ? 1 : bytes.length); bytes.write('ftyp', 4)
  if (extended) bytes.writeBigUInt64BE(BigInt(bytes.length), 8)
  bytes.write('zzzz', header); bytes.write('isom', bytes.length - 4)
  const cloud = sdk(bytes)
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
})

test.each([null, { fileID: 'https://secret.invalid' }, new Error('secret credentials')])('SDK invalid success or rejection becomes safe unknown after invocation', async result => {
  const bytes = mp4(); const cloud = sdk(bytes, async ({ fileContent }) => {
    for await (const _chunk of fileContent) { /* real SDK consumption */ }
    if (result instanceof Error) throw result
    return result
  })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'unknown', reason: 'provider_error' })
  expect(fs.readdirSync(root)).toEqual([])
})

test('known SDK fileID survives upload stream failure', async () => {
  const bytes = mp4(); const cloud = sdk(bytes, async ({ fileContent }) => {
    fileContent.destroy(Error('secret read failure'))
    return { fileID: 'cloud://opaque/published' }
  })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'unknown', reason: 'provider_error', file_id: 'cloud://opaque/published' })
})

test('concurrent copies remove only their own spool and preserve user files', async () => {
  const bytes = mp4(); const marker = path.join(root, 'user-asset'); fs.writeFileSync(marker, 'preserve')
  let release; const blocked = new Promise(resolve => { release = resolve }); let entered
  const ready = new Promise(resolve => { entered = resolve })
  const cloud = sdk(bytes, async ({ fileContent }) => {
    for await (const _chunk of fileContent) { /* consume */ }
    entered(); await blocked; return { fileID: 'cloud://opaque/published' }
  })
  const provider = createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) })
  const first = provider.copyRegisteredObject(constraints(bytes)); await ready
  const failing = sdk(bytes, async () => { throw Error('secret') })
  expect(await createRegisteredStorageProvider(failing, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'unknown', reason: 'provider_error' })
  expect(fs.readdirSync(root)).toHaveLength(2)
  release(); await first
  expect(fs.readdirSync(root)).toEqual(['user-asset'])
  expect(fs.readFileSync(marker, 'utf8')).toBe('preserve')
})

test('cleanup failure emits fixed safe warning without masking SDK result or logger failure', async () => {
  const bytes = mp4(); const cloud = sdk(bytes); const logger = { warn: jest.fn(() => { throw Error('logger failed') }) }
  const real = fs.promises.rmdir.bind(fs.promises)
  jest.spyOn(fs.promises, 'rmdir').mockImplementation(async dir => { if (dir.startsWith(root)) throw Error('secret path'); return real(dir) })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, logger, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
  expect(logger.warn).toHaveBeenCalledWith('REGISTERED_STORAGE_CLEANUP_FAILED')
})

test('waits for actual file finish before SDK write', async () => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const realOpen = fs.promises.open.bind(fs.promises)
  let finish; let reached
  const ready = new Promise(resolve => { reached = resolve })
  jest.spyOn(fs.promises, 'open').mockImplementation(async (...args) => {
    const handle = await realOpen(...args)
    if (args[1] & fs.constants.O_WRONLY) {
      const create = handle.createWriteStream.bind(handle)
      handle.createWriteStream = options => {
        const stream = create(options)
        stream._final = callback => { finish = callback; reached() }
        return stream
      }
    }
    return handle
  })
  const pending = createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes))
  await ready
  expect(cloud.uploadFile).not.toHaveBeenCalled()
  finish()
  expect(await pending).toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
})

test('SDK success waits for upload stream EOF', async () => {
  const bytes = mp4(); let stream; let received
  const ready = new Promise(resolve => { received = resolve })
  const cloud = sdk(bytes, async ({ fileContent }) => { stream = fileContent; received(); return { fileID: 'cloud://opaque/published' } })
  let settled = false
  const pending = createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)).then(result => { settled = true; return result })
  await ready; await Promise.resolve()
  expect(settled).toBe(false)
  for await (const _part of stream) { /* actual stream EOF */ }
  expect(await pending).toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
})

test('transport errors cannot forge permanent mismatch classification', async () => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const body = Object.assign(Readable.from((async function* () { throw Object.assign(Error('remote failure'), { byteMismatch: true }) })()), { statusCode: 200, headers: {} })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => body }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
})

test('default HTTPS owns the complete body deadline and destroys both body and request', async () => {
  const https = require('https'); const { EventEmitter } = require('events')
  const bytes = mp4(); const cloud = sdk(bytes)
  const body = Object.assign(new Readable({ read() {} }), { statusCode: 200, headers: {} })
  const request = new EventEmitter()
  request.destroy = jest.fn(error => { request.emit('error', error); return request })
  let expire; let started
  const read = new Promise(resolve => { started = resolve })
  let sent = false
  body._read = function () { if (!sent) { sent = true; this.push(bytes.subarray(0, 4)); started() } }
  jest.spyOn(global, 'setTimeout').mockImplementation((callback, delay) => {
    expect(delay).toBe(20000); expire = callback; return { unref() {} }
  })
  jest.spyOn(global, 'clearTimeout').mockImplementation(() => {})
  jest.spyOn(https, 'get').mockImplementation((url, options, callback) => {
    expect(url).toBe('https://storage.invalid/get'); expect(options.highWaterMark).toBe(65536)
    process.nextTick(() => callback(body)); return request
  })
  const pending = createRegisteredStorageProvider(cloud, { temporaryRoot: root }).copyRegisteredObject(constraints(bytes))
  await read
  expire()
  expect(await pending).toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(body.destroyed).toBe(true); expect(request.destroy).toHaveBeenCalled()
  expect(cloud.uploadFile).not.toHaveBeenCalled()
})

test('default HTTPS does not follow redirects and safely destroys invalid response', async () => {
  const https = require('https'); const { EventEmitter } = require('events')
  const bytes = mp4(); const cloud = sdk(bytes)
  const body = Object.assign(new Readable({ read() {} }), { statusCode: 302, headers: { location: 'https://other.invalid' } })
  const request = new EventEmitter()
  request.destroy = jest.fn(error => { process.nextTick(() => request.emit('error', error)); return request })
  jest.spyOn(https, 'get').mockImplementation((_url, _options, callback) => { process.nextTick(() => callback(body)); return request })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(https.get).toHaveBeenCalledTimes(1)
  expect(body.destroyed).toBe(true)
})

// One reusable 64KiB zero block and incremental hashes keep this max-contract
// exercise independent of whole-video Buffers on both sides of the adapter.
test.each([0, 1])('chunked 256MiB boundary with %s extra byte(s) uses bounded exact stream', async extra => {
  const total = 256 * 1024 * 1024
  const header = mp4(''); const zero = Buffer.alloc(65536)
  function* chunks() {
    yield header
    for (let remaining = total - header.length; remaining > 0; remaining -= zero.length) yield zero.subarray(0, Math.min(remaining, zero.length))
  }
  const hash = crypto.createHash('sha256'); for (const chunk of chunks()) hash.update(chunk)
  const digest = hash.digest('hex'); let uploaded = 0
  const cloud = sdk(null, async ({ fileContent }) => {
    expect(fileContent.readableHighWaterMark).toBe(65536)
    const outputHash = crypto.createHash('sha256')
    for await (const chunk of fileContent) { expect(chunk.length).toBeLessThanOrEqual(65536); uploaded += chunk.length; outputHash.update(chunk) }
    expect(outputHash.digest('hex')).toBe(digest)
    return { fileID: 'cloud://opaque/published' }
  })
  const openDownload = async () => Object.assign(Readable.from((function* () { yield* chunks(); if (extra) yield Buffer.from('x') })(), { objectMode: false, highWaterMark: 65536 }), { statusCode: 200, headers: {} })
  const result = await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload }).copyRegisteredObject({ ...constraints(header), byte_size: total, content_sha256: digest })
  expect(result).toEqual(extra ? { outcome: 'no_write', reason: 'not_started', permanent: true } : { outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
  expect(uploaded).toBe(extra ? 0 : total)
  expect(fs.readdirSync(root)).toEqual([])
}, 15000)

test('short positional reads do not change valid MP4 acceptance', async () => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const realOpen = fs.promises.open.bind(fs.promises)
  jest.spyOn(fs.promises, 'open').mockImplementation(async (...args) => {
    const handle = await realOpen(...args)
    if (!(args[1] & fs.constants.O_WRONLY)) {
      const read = handle.read.bind(handle)
      handle.read = (buffer, offset, length, position) => read(buffer, offset, Math.min(2, length), position)
    }
    return handle
  })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'server_succeeded', file_id: 'cloud://opaque/published' })
})

test.each(['symlink', 'replacement'])('rejects spool %s without reading or deleting user assets', async replacement => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const userFile = path.join(root, 'user-video'); fs.writeFileSync(userFile, bytes)
  const realOpen = fs.promises.open.bind(fs.promises)
  jest.spyOn(fs.promises, 'open').mockImplementation(async (...args) => {
    if (!(args[1] & fs.constants.O_WRONLY)) {
      await fs.promises.unlink(args[0])
      if (replacement === 'symlink') await fs.promises.symlink(userFile, args[0])
      else await fs.promises.writeFile(args[0], bytes)
    }
    return realOpen(...args)
  })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(cloud.uploadFile).not.toHaveBeenCalled()
  expect(fs.readFileSync(userFile)).toEqual(bytes)
  expect(fs.readdirSync(root)).toEqual(['user-video'])
})

test('file finish error prevents SDK write and closes owned file', async () => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const realOpen = fs.promises.open.bind(fs.promises)
  jest.spyOn(fs.promises, 'open').mockImplementation(async (...args) => {
    const handle = await realOpen(...args)
    if (args[1] & fs.constants.O_WRONLY) {
      const create = handle.createWriteStream.bind(handle)
      handle.createWriteStream = options => { const stream = create(options); stream._final = cb => cb(Error('disk secret')); return stream }
    }
    return handle
  })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => response(bytes) }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(cloud.uploadFile).not.toHaveBeenCalled()
  expect(fs.readdirSync(root)).toEqual([])
})

test.each([
  { publish_object_key: staging }, { publish_object_key: published.replace('22222222-2222-4222', '22222222-2222-1222') },
  { byte_size: 256 * 1024 * 1024 + 1 }, { byte_size: 0 }, { byte_size: '28' },
  { media_type: 'application/octet-stream' }, { staging_file_id: 'cloud://' }, { content_sha256: 'A'.repeat(64) },
])('malformed registered input is no-write before provider access (%j)', async change => {
  const bytes = mp4(); const cloud = sdk(bytes)
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root }).copyRegisteredObject({ ...constraints(bytes), ...change }))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
  expect(cloud.uploadFile).not.toHaveBeenCalled()
})

test.each([
  { object_key: published }, { object_key: staging.replace('/staging/', '/other/') },
  { byte_size: 256 * 1024 * 1024 + 1 }, { content_sha256: '0' },
])('malformed grant is no-write before provider access (%j)', async change => {
  const cloud = sdk(mp4())
  expect(await createRegisteredStorageProvider(cloud).authorizeRegisteredUpload({ ...constraints(mp4()), object_key: staging, ...change }))
    .toEqual({ outcome: 'no_write', reason: 'not_started' })
  expect(cloud.getUploadMetadata).not.toHaveBeenCalled()
})

test('early body error while private spool opens is handled as transient no-write', async () => {
  const bytes = mp4(); const cloud = sdk(bytes)
  const body = Object.assign(new Readable({ read() {} }), { statusCode: 200, headers: {} })
  const real = fs.promises.mkdtemp.bind(fs.promises)
  jest.spyOn(fs.promises, 'mkdtemp').mockImplementation(async (...args) => {
    body.destroy(Error('early secret failure'))
    return real(...args)
  })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => body }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(cloud.uploadFile).not.toHaveBeenCalled()
  expect(fs.readdirSync(root)).toEqual([])
})

test.each(['close', 'aborted'])('midstream %s prevents SDK write even with all claimed bytes', async event => {
  const bytes = mp4(); const cloud = sdk(bytes)
  let sent = false
  const body = Object.assign(new Readable({ read() {
    if (!sent) { sent = true; this.push(bytes); process.nextTick(() => { if (event === 'aborted') this.emit('aborted'); else this.destroy() }) }
  } }), { statusCode: 200, headers: {} })
  expect(await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => body }).copyRegisteredObject(constraints(bytes)))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(cloud.uploadFile).not.toHaveBeenCalled()
  expect(fs.readdirSync(root)).toEqual([])
})

// Run without process-level error handlers: an SDK-wrapper error must crash the
// isolated probe and fail this regression, rather than being swallowed by Jest.
test.each(['read failure', 'early SDK rejection', 'success'])('pinned SDK-shaped wrapping contains %s with real fs.ReadStream', scenario => {
  const { spawnSync } = require('child_process')
  const probe = spawnSync(process.execPath, ['-e', `
    const fs = require('fs')
    const crypto = require('crypto')
    const { Readable, Writable } = require('stream')
    const { finished } = require('stream/promises')
    const { createRegisteredStorageProvider } = require(process.argv[1])
    const root = process.argv[2]
    const scenario = process.argv[3]
    const bytes = Buffer.from(process.argv[4], 'base64')
    let wrapped
    let received = 0
    const hash = crypto.createHash('sha256')
    const cloud = {
      getTempFileURL: async () => ({ fileList: [{ fileID: 'cloud://opaque/staging', code: 'SUCCESS', tempFileURL: 'https://storage.invalid/get' }] }),
      uploadFile: async ({ fileContent }) => {
        if (!(fileContent instanceof fs.ReadStream)) throw Error('not a real fs.ReadStream')
        // Exact pinned SDK pattern: separate Readable.from wrapper, pipe into
        // request, and no wrapper error handler. Only request errors reject SDK.
        wrapped = Readable.from(fileContent)
        const request = new Writable({ write(chunk, _encoding, callback) {
          received += chunk.length; hash.update(chunk); callback()
        } })
        const response = new Promise((resolve, reject) => {
          request.once('finish', resolve)
          request.on('error', reject)
        })
        wrapped.pipe(request)
        if (scenario === 'read failure') process.nextTick(() => fileContent.destroy(Error('synthetic private read error')))
        if (scenario === 'early SDK rejection') throw Error('synthetic request rejection')
        await response
        return { fileID: 'cloud://opaque/published' }
      },
    }
    const body = () => Object.assign(Readable.from([bytes]), { statusCode: 200, headers: {} })
    ;(async () => {
      const result = await createRegisteredStorageProvider(cloud, { temporaryRoot: root, openDownload: async () => body() }).copyRegisteredObject({
        staging_file_id: 'cloud://opaque/staging',
        publish_object_key: 'published/managed-clips-v1/' + 'a'.repeat(64) + '/22222222-2222-4222-8222-222222222222.mp4',
        byte_size: bytes.length, content_sha256: crypto.createHash('sha256').update(bytes).digest('hex'), media_type: 'video/mp4',
      })
      // Await the separate wrapper after provider cleanup without adding an
      // error listener before it can expose the original SDK error boundary.
      await finished(wrapped)
      console.log(JSON.stringify({ result, remaining: fs.readdirSync(root), received, sha256: hash.digest('hex') }))
    })().catch(error => { console.error(error); process.exitCode = 1 })
  `, require.resolve('../../cloudfunctions/sync-api/lib/registered-storage'), root, scenario, mp4().toString('base64')], { encoding: 'utf8', timeout: 5000 })
  expect({ status: probe.status, stderr: probe.stderr, error: probe.error }).toEqual({ status: 0, stderr: '', error: undefined })
  const output = JSON.parse(probe.stdout)
  expect(output.result).toEqual(scenario === 'success'
    ? { outcome: 'server_succeeded', file_id: 'cloud://opaque/published' }
    : { outcome: 'unknown', reason: 'provider_error', ...(scenario === 'read failure' ? { file_id: 'cloud://opaque/published' } : {}) })
  expect(output.remaining).toEqual([])
  if (scenario === 'success') {
    expect(output.received).toBe(mp4().length)
    expect(output.sha256).toBe(crypto.createHash('sha256').update(mp4()).digest('hex'))
  }
})
