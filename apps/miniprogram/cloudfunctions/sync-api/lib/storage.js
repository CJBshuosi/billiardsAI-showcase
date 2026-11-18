const crypto = require('crypto')

const { SyncApiError } = require('./auth')
const { createRegisteredStorageProvider, isReservedManagedKey } = require('./registered-storage')

function rejectManaged(constraints) {
  if (isReservedManagedKey(constraints?.object_key) || isReservedManagedKey(constraints?.publish_object_key)) {
    throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
  }
}

const MP4_BRANDS = new Set([
  'avc1', 'dash', 'iso2', 'iso3', 'iso4', 'iso5', 'iso6', 'iso7', 'iso8', 'iso9',
  'isom', 'M4V ', 'mp41', 'mp42', 'MSNV',
])

function isMp4FileTypeContainer(content) {
  if (!Buffer.isBuffer(content) || content.byteLength < 16
      || content.toString('ascii', 4, 8) !== 'ftyp') return false
  let boxSize = content.readUInt32BE(0)
  let brandOffset = 8
  if (boxSize === 1) {
    if (content.byteLength < 24) return false
    const extendedSize = content.readBigUInt64BE(8)
    if (extendedSize > BigInt(Number.MAX_SAFE_INTEGER)) return false
    boxSize = Number(extendedSize)
    brandOffset = 16
  }
  if (boxSize < brandOffset + 8 || boxSize > content.byteLength
      || (boxSize - brandOffset - 8) % 4 !== 0) return false
  const brands = [content.toString('ascii', brandOffset, brandOffset + 4)]
  for (let offset = brandOffset + 8; offset + 4 <= boxSize; offset += 4) {
    brands.push(content.toString('ascii', offset, offset + 4))
  }
  return brands.some(brand => MP4_BRANDS.has(brand))
}


function createCloudBaseStorageProvider(cloud, registeredOptions) {
  if (!cloud || typeof cloud.getUploadMetadata !== 'function'
      || typeof cloud.downloadFile !== 'function') {
    throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
  }
  return {
    ...createRegisteredStorageProvider(cloud, registeredOptions),
    async createPresignedUpload({ object_key: objectKey, expires_in_seconds: expiresIn }) {
      rejectManaged({ object_key: objectKey })
      let metadata
      try {
        metadata = await cloud.getUploadMetadata({ cloudPath: objectKey })
      } catch (_error) {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      const data = metadata && metadata.data
      if (!data || !data.url || !data.token || !data.authorization || !data.cosFileId) {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      return {
        method: 'PUT',
        url: data.url,
        headers: {
          Signature: data.authorization,
          authorization: data.authorization,
          'x-cos-security-token': data.token,
          'x-cos-meta-fileid': data.cosFileId,
          key: encodeURIComponent(objectKey),
        },
        expires_in_seconds: expiresIn,
      }
    },
    async verifyPrivateObject({
      object_key: objectKey,
      publish_object_key: publishObjectKey,
      byte_size: expectedBytes,
      content_sha256: expectedHash,
    }) {
      rejectManaged({ object_key: objectKey, publish_object_key: publishObjectKey })
      let metadata
      let downloaded
      try {
        metadata = await cloud.getUploadMetadata({ cloudPath: objectKey })
        const resolvedFileId = metadata && metadata.data && metadata.data.fileId
        if (!resolvedFileId) return { verified: false }
        downloaded = await cloud.downloadFile({ fileID: resolvedFileId })
      } catch (_error) {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      const fileId = metadata.data.fileId
      if (!downloaded || downloaded.fileContent === undefined) return { verified: false }
      // Own the exact downloaded bytes before validation; never re-read staging
      // for a later copy while an outstanding edge credential can replace it.
      const content = Buffer.from(downloaded.fileContent)
      const actualHash = crypto.createHash('sha256').update(content).digest('hex')
      if (content.byteLength !== expectedBytes || actualHash !== expectedHash) {
        return { verified: false, permanent: true }
      }
      if (!isMp4FileTypeContainer(content)) return { verified: false, permanent: true }
      if (publishObjectKey) {
        if (!publishObjectKey.startsWith('published/') || publishObjectKey === objectKey
            || typeof cloud.uploadFile !== 'function') throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
        try {
          const published = await cloud.uploadFile({ cloudPath: publishObjectKey, fileContent: content })
          if (!published || !published.fileID) throw new Error('missing published file ID')
          return { verified: true, file_id: published.fileID, staging_file_id: fileId }
        } catch (_error) { throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE') }
      }
      return { verified: true, file_id: fileId }
    },
    async deletePrivateObjects(fileIds) {
      if (typeof cloud.deleteFile !== 'function' || !Array.isArray(fileIds) || !fileIds.length
          || fileIds.some(fileId => typeof fileId !== 'string' || !fileId.startsWith('cloud://'))) {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      let result
      try { result = await cloud.deleteFile({ fileList: fileIds }) } catch (_error) {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      const rows = result && result.fileList
      if (!Array.isArray(rows) || rows.length !== fileIds.length) {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      const byId = new Map(rows.map(row => [row && (row.fileID || row.fileId), row]))
      const success = code => code === 'SUCCESS' || code === 'STORAGE_FILE_NONEXIST'
        || code === 0 || code === '0'
      if (fileIds.some(fileId => !byId.has(fileId) || !success(byId.get(fileId).code))) {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      for (const fileId of fileIds) {
        try {
          await cloud.downloadFile({ fileID: fileId })
          throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
        } catch (error) {
          if (error instanceof SyncApiError || !error || error.code !== 'STORAGE_FILE_NONEXIST') {
            throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
          }
        }
      }
      return { deleted: true, count: fileIds.length }
    },
  }
}


function createStoragePort(provider) {
  if (!provider || typeof provider.createPresignedUpload !== 'function'
      || typeof provider.verifyPrivateObject !== 'function') {
    throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
  }
  return {
    async authorizeUpload(constraints) {
      rejectManaged(constraints)
      return provider.createPresignedUpload({ ...constraints, expires_in_seconds: 300 })
    },
    async verifyObject(constraints) {
      rejectManaged(constraints)
      return provider.verifyPrivateObject(constraints)
    },
    async authorizeRegisteredUpload(constraints) {
      if (typeof provider.authorizeRegisteredUpload !== 'function') return { outcome: 'no_write', reason: 'not_started' }
      return provider.authorizeRegisteredUpload(constraints)
    },
    async copyRegisteredObject(constraints) {
      if (typeof provider.copyRegisteredObject !== 'function') return { outcome: 'no_write', reason: 'not_started', permanent: false }
      return provider.copyRegisteredObject(constraints)
    },
    async deleteObjects(fileIds) {
      if (typeof provider.deletePrivateObjects !== 'function') {
        throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
      }
      return provider.deletePrivateObjects(fileIds)
    },
  }
}


module.exports = { createCloudBaseStorageProvider, createStoragePort }
