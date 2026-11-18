const crypto = require('crypto')
const fs = require('fs')
const https = require('https')
const os = require('os')
const path = require('path')
const { Readable } = require('stream')
const { pipeline, finished } = require('stream/promises')

const BLOCK = 64 * 1024
const MAX_BYTES = 256 * 1024 * 1024
const UUID = '[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}'
const STAGING = new RegExp(`^managed-clips-v1/[a-f0-9]{64}/staging/${UUID}\\.mp4$`, 'u')
const PUBLISHED = new RegExp(`^published/managed-clips-v1/[a-f0-9]{64}/${UUID}\\.mp4$`, 'u')
const BRANDS = new Set(['avc1', 'dash', 'iso2', 'iso3', 'iso4', 'iso5', 'iso6', 'iso7', 'iso8', 'iso9', 'isom', 'M4V ', 'mp41', 'mp42', 'MSNV'])
const noWrite = permanent => ({ outcome: 'no_write', reason: 'not_started', permanent })
const unknown = file => ({ outcome: 'unknown', reason: 'provider_error', ...(validFileId(file) ? { file_id: file } : {}) })
function validText(value) {
  return typeof value === 'string' && value.length > 0 && !/[\u0000-\u001f\u007f]/u.test(value)
    && Buffer.from(value, 'utf8').toString('utf8') === value
}
function validFileId(value) { return validText(value) && value.startsWith('cloud://') && value.length > 8 && value.length <= 2048 }
function validUrl(value) {
  if (!validText(value) || value.trim() !== value) return false
  try {
    const authority = /^https:\/\/([^/?#]*)/iu.exec(value)
    if (!authority || authority[1].includes('@') || value.includes('#')) return false
    const url = new URL(value)
    return url.protocol === 'https:' && Boolean(url.hostname) && !url.username && !url.password && !url.hash
  } catch (_) { return false }
}
function validMedia(input) {
  return input && Number.isSafeInteger(input.byte_size) && input.byte_size > 0 && input.byte_size <= MAX_BYTES
    && typeof input.content_sha256 === 'string' && /^[a-f0-9]{64}$/u.test(input.content_sha256) && input.media_type === 'video/mp4'
}
function isReservedManagedKey(value) {
  return typeof value === 'string' && (value === 'managed-clips-v1' || value.startsWith('managed-clips-v1/')
    || value === 'published/managed-clips-v1' || value.startsWith('published/managed-clips-v1/'))
}
class MediaMismatchError extends Error {}
function byteMismatch() { return new MediaMismatchError('invalid media') }

// Only the GET body is returned. The deadline covers headers and the complete body;
// no redirect handler exists, and neither the URL nor native errors escape this module.
function openHttpsDownload(url) {
  return new Promise((resolve, reject) => {
    let body
    let timer
    const fail = () => {
      clearTimeout(timer)
      const error = new Error('download unavailable')
      if (body) body.destroy(error)
      request.destroy(error)
      reject(error)
    }
    const request = https.get(url, { highWaterMark: BLOCK }, incoming => {
      body = incoming
      // A response can fail before the awaiting caller attaches its pipeline.
      incoming.on('error', () => {})
      incoming.once('end', () => clearTimeout(timer))
      incoming.once('close', () => clearTimeout(timer))
      incoming.once('aborted', fail)
      if (incoming.statusCode !== 200) { fail(); return }
      resolve(incoming)
    })
    request.on('error', () => {
      clearTimeout(timer)
      if (body) body.destroy(new Error('download unavailable'))
      reject(new Error('download unavailable'))
    })
    timer = setTimeout(fail, 20000)
    timer.unref?.()
  })
}

// Scan the complete first box using one fixed-size read buffer. A large ftyp box
// gets the same brand/alignment semantics as the legacy validator, without retaining it.
async function validMp4(handle, byteSize) {
  const buffer = Buffer.alloc(BLOCK)
  const headerLength = Math.min(24, byteSize)
  let headerBytes = 0
  while (headerBytes < headerLength) {
    const read = await handle.read(buffer, headerBytes, headerLength - headerBytes, headerBytes)
    if (!read.bytesRead) return false
    headerBytes += read.bytesRead
  }
  if (headerBytes < 16 || buffer.toString('ascii', 4, 8) !== 'ftyp') return false
  let boxSize = buffer.readUInt32BE(0)
  let brandOffset = 8
  if (boxSize === 1) {
    if (headerBytes < 24) return false
    const extended = buffer.readBigUInt64BE(8)
    if (extended > BigInt(Number.MAX_SAFE_INTEGER)) return false
    boxSize = Number(extended); brandOffset = 16
  }
  if (boxSize < brandOffset + 8 || boxSize > byteSize || (boxSize - brandOffset - 8) % 4 !== 0) return false
  let recognized = BRANDS.has(buffer.toString('ascii', brandOffset, brandOffset + 4))
  for (let position = brandOffset + 8; position < boxSize;) {
    const length = Math.min(BLOCK, boxSize - position)
    // FileHandle.read may legally return fewer bytes than requested.
    let count = 0
    while (count < length) {
      const read = await handle.read(buffer, count, length - count, position + count)
      if (!read.bytesRead) return false
      count += read.bytesRead
    }
    for (let offset = 0; offset < count; offset += 4) {
      if (BRANDS.has(buffer.toString('ascii', offset, offset + 4))) recognized = true
    }
    position += count
  }
  return recognized
}

function createRegisteredStorageProvider(cloud, { openDownload = openHttpsDownload, temporaryRoot = os.tmpdir(), logger = console } = {}) {
  return {
    async authorizeRegisteredUpload(input) {
      if (!validMedia(input) || typeof input.object_key !== 'string' || !STAGING.test(input.object_key)
          || !cloud || typeof cloud.getUploadMetadata !== 'function') return { outcome: 'no_write', reason: 'not_started' }
      let fileId
      try {
        const metadata = await cloud.getUploadMetadata({ cloudPath: input.object_key })
        const data = metadata && metadata.data
        fileId = data && data.fileId
        if (!validFileId(fileId) || !validUrl(data.url)
            || ![data.token, data.authorization, data.cosFileId].every(validText)) return unknown(fileId)
        return { outcome: 'external_issued', file_id: fileId, authorization: {
          method: 'PUT', url: data.url,
          headers: { Signature: data.authorization, authorization: data.authorization,
            'x-cos-security-token': data.token, 'x-cos-meta-fileid': data.cosFileId, key: encodeURIComponent(input.object_key) },
          expires_in_seconds: 300,
        } }
      } catch (_) { return unknown(fileId) }
    },
    async copyRegisteredObject(input) {
      if (!validMedia(input) || !validFileId(input.staging_file_id) || typeof input.publish_object_key !== 'string'
          || !PUBLISHED.test(input.publish_object_key) || !cloud || typeof cloud.getTempFileURL !== 'function'
          || typeof cloud.uploadFile !== 'function') return noWrite(false)
      let directory; let filePath; let writeHandle; let readHandle; let body; let writer; let uploadStream
      let invoked = false; let observedFileId
      let aborted
      try {
        const result = await cloud.getTempFileURL({ fileList: [input.staging_file_id] })
        const rows = result && result.fileList
        if (!Array.isArray(rows) || rows.length !== 1 || rows[0]?.fileID !== input.staging_file_id
            || rows[0].code !== 'SUCCESS' || !validUrl(rows[0].tempFileURL)) return noWrite(false)
        body = await openDownload(rows[0].tempFileURL)
        if (!(body instanceof Readable)) return noWrite(false)
        // Own early errors while asynchronous private-file setup is still pending.
        body.on('error', () => {})
        aborted = () => body.destroy(new Error('download unavailable'))
        body.once('aborted', aborted)
        if (body.statusCode !== 200 || body.aborted || body.destroyed) return noWrite(false)
        const length = body.headers && body.headers['content-length']
        if (length !== undefined && (typeof length !== 'string' || !/^(0|[1-9][0-9]*)$/u.test(length)
            || Number(length) !== input.byte_size)) return noWrite(false)
        directory = await fs.promises.mkdtemp(path.join(temporaryRoot, 'registered-'))
        await fs.promises.chmod(directory, 0o700)
        filePath = path.join(directory, 'media.mp4')
        writeHandle = await fs.promises.open(filePath, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW, 0o600)
        const created = await writeHandle.stat()
        if (!created.isFile()) throw new Error('spool unavailable')
        const hash = crypto.createHash('sha256')
        let total = 0
        writer = writeHandle.createWriteStream({ autoClose: false, highWaterMark: BLOCK })
        await pipeline(body, async function* (source) {
          for await (const chunk of source) {
            if (!Buffer.isBuffer(chunk)) throw new Error('download unavailable')
            // Count before forwarding any chunk to disk; subarrays do not copy media.
            total += chunk.length
            if (total > input.byte_size || total > MAX_BYTES) throw byteMismatch()
            for (let offset = 0; offset < chunk.length; offset += BLOCK) {
              const part = chunk.subarray(offset, offset + BLOCK)
              hash.update(part)
              yield part
            }
          }
        }, writer)
        if (!body.readableEnded || body.aborted) throw new Error('download unavailable')
        if (total !== input.byte_size || hash.digest('hex') !== input.content_sha256) throw byteMismatch()
        // Keep the writer descriptor alive across reopening: after unlink, Linux
        // may immediately recycle an inode if its final descriptor was closed.
        readHandle = await fs.promises.open(filePath, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW)
        const pinned = await readHandle.stat()
        if (!pinned.isFile() || pinned.dev !== created.dev || pinned.ino !== created.ino
            || pinned.size !== total) throw new Error('spool unavailable')
        writer.destroy()
        await writeHandle.close(); writeHandle = null
        // Remove the pathname before validation/upload: future path replacement cannot
        // replace bytes held by the private readonly descriptor.
        await fs.promises.unlink(filePath); filePath = null
        if (!await validMp4(readHandle, total)) throw byteMismatch()
        uploadStream = readHandle.createReadStream({ autoClose: true, start: 0, highWaterMark: BLOCK })
        const ended = finished(uploadStream, { cleanup: true }).then(() => true, () => false)
        // The pinned SDK pipes Readable.from(fileContent) without a wrapper error
        // listener. Contain only this instance's async iteration so original read
        // errors/cleanup close cannot escape through that separate SDK stream.
        // `ended` still observes the original failure and prevents success below.
        const iterateUpload = uploadStream[Symbol.asyncIterator].bind(uploadStream)
        uploadStream[Symbol.asyncIterator] = async function* () {
          try { yield* iterateUpload() } catch (_) { /* original stream outcome remains authoritative */ }
        }
        invoked = true
        const published = await cloud.uploadFile({ cloudPath: input.publish_object_key, fileContent: uploadStream })
        observedFileId = published && published.fileID
        if (!validFileId(observedFileId)) { uploadStream.destroy(); await ended; return unknown() }
        if (!await ended || !uploadStream.readableEnded) return unknown(observedFileId)
        return { outcome: 'server_succeeded', file_id: observedFileId }
      } catch (error) {
        return invoked ? unknown(observedFileId) : noWrite(error instanceof MediaMismatchError)
      } finally {
        let failed = false
        const clean = async action => { try { await action() } catch (_) { failed = true } }
        if (body instanceof Readable) {
          if (aborted) body.removeListener('aborted', aborted)
          body.destroy()
        }
        if (writer) { writer.destroy(); await clean(() => finished(writer, { cleanup: true }).catch(() => {})) }
        if (uploadStream) { uploadStream.destroy(); await clean(() => finished(uploadStream, { cleanup: true }).catch(() => {})) }
        if (writeHandle) await clean(() => writeHandle.close())
        if (readHandle) await clean(() => readHandle.close())
        if (filePath) await clean(() => fs.promises.unlink(filePath))
        if (directory) await clean(() => fs.promises.rmdir(directory))
        if (failed) { try { logger.warn('REGISTERED_STORAGE_CLEANUP_FAILED') } catch (_) { /* preserve the provider observation */ } }
      }
    },
  }
}
module.exports = { createRegisteredStorageProvider, isReservedManagedKey }
