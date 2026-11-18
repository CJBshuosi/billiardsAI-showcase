const crypto = require('crypto')

const {
  createCloudBaseStorageProvider,
  createStoragePort,
} = require('../../cloudfunctions/sync-api/lib/storage')

function mp4Bytes(payload = '') {
  return Buffer.concat([
    Buffer.from([0, 0, 0, 24]),
    Buffer.from('ftypisom'),
    Buffer.from([0, 0, 0, 0]),
    Buffer.from('isommp42'),
    Buffer.from(payload),
  ])
}


test('CloudBase private upload authorization exposes only short-lived PUT credentials', async () => {
  const cloud = {
    getUploadMetadata: jest.fn().mockResolvedValue({
      data: {
        url: 'https://cos.invalid/upload',
        token: 'temporary-token',
        authorization: 'temporary-signature',
        fileId: 'cloud://development/private/clip-1.mp4',
        cosFileId: 'cos-file-id',
      },
    }),
    downloadFile: jest.fn(),
  }
  const storage = createStoragePort(createCloudBaseStorageProvider(cloud))

  const result = await storage.authorizeUpload({
    object_key: 'sessions/session-1/clip-1.mp4',
    byte_size: 13,
    content_sha256: 'a'.repeat(64),
    media_type: 'video/mp4',
  })

  expect(cloud.getUploadMetadata).toHaveBeenCalledWith({
    cloudPath: 'sessions/session-1/clip-1.mp4',
  })
  expect(result).toEqual({
    method: 'PUT',
    url: 'https://cos.invalid/upload',
    headers: {
      Signature: 'temporary-signature',
      authorization: 'temporary-signature',
      'x-cos-security-token': 'temporary-token',
      'x-cos-meta-fileid': 'cos-file-id',
      key: 'sessions%2Fsession-1%2Fclip-1.mp4',
    },
    expires_in_seconds: 300,
  })
  expect(result).not.toHaveProperty('file_id')
})


test('CloudBase private object verification checks exact MP4 bytes and SHA-256', async () => {
  const content = mp4Bytes('private-video')
  const cloud = {
    getUploadMetadata: jest.fn().mockResolvedValue({
      data: { fileId: 'cloud://development/private/clip-1.mp4' },
    }),
    downloadFile: jest.fn().mockResolvedValue({ fileContent: content }),
  }
  const storage = createStoragePort(createCloudBaseStorageProvider(cloud))

  await expect(storage.verifyObject({
    object_key: 'sessions/session-1/clip-1.mp4',
    byte_size: content.byteLength,
    content_sha256: crypto.createHash('sha256').update(content).digest('hex'),
    media_type: 'video/mp4',
  })).resolves.toEqual({
    verified: true,
    file_id: 'cloud://development/private/clip-1.mp4',
  })
  expect(cloud.downloadFile).toHaveBeenCalledWith({
    fileID: 'cloud://development/private/clip-1.mp4',
  })

  await expect(storage.verifyObject({
    object_key: 'sessions/session-1/clip-1.mp4',
    byte_size: content.byteLength,
    content_sha256: '0'.repeat(64),
    media_type: 'video/mp4',
  })).resolves.toEqual({ verified: false, permanent: true })
})

test('CloudBase private object verification rejects digest-matching non-MP4 bytes', async () => {
  const content = Buffer.from('not-an-iso-bmff-container')
  const cloud = {
    getUploadMetadata: jest.fn().mockResolvedValue({
      data: { fileId: 'cloud://development/private/not-video.mp4' },
    }),
    downloadFile: jest.fn().mockResolvedValue({ fileContent: content }),
  }
  const storage = createStoragePort(createCloudBaseStorageProvider(cloud))

  await expect(storage.verifyObject({
    object_key: 'sessions/session-1/not-video.mp4',
    byte_size: content.byteLength,
    content_sha256: crypto.createHash('sha256').update(content).digest('hex'),
    media_type: 'video/mp4',
  })).resolves.toEqual({ verified: false, permanent: true })
})

test('publication copies only verified bytes away from still-valid upload credentials', async () => {
  const original = mp4Bytes('private-video')
  let staged = Buffer.from(original)
  const objects = new Map()
  const cloud = {
    getUploadMetadata: jest.fn().mockResolvedValue({ data: { fileId: 'cloud://staging' } }),
    downloadFile: jest.fn(async () => ({ fileContent: staged })),
    uploadFile: jest.fn(async ({ cloudPath, fileContent }) => {
      objects.set(cloudPath, Buffer.from(fileContent))
      return { fileID: `cloud://${cloudPath}` }
    }),
  }
  const storage = createStoragePort(createCloudBaseStorageProvider(cloud))
  const constraints = {
    object_key: 'sessions/session-1/job-1/clip-1.mp4',
    publish_object_key: 'published/session-1/job-1/clip-1.mp4',
    byte_size: original.length,
    content_sha256: crypto.createHash('sha256').update(original).digest('hex'),
    media_type: 'video/mp4',
  }
  await expect(storage.verifyObject(constraints)).resolves.toEqual({
    verified: true,
    file_id: 'cloud://published/session-1/job-1/clip-1.mp4',
    staging_file_id: 'cloud://staging',
  })
  staged = Buffer.from('tampered-data') // replay an outstanding PUT credential
  expect(objects.get(constraints.publish_object_key)).toEqual(original)
  await expect(storage.verifyObject(constraints))
    .resolves.toEqual({ verified: false, permanent: true })
  expect(cloud.uploadFile).toHaveBeenCalledTimes(1)
  cloud.uploadFile.mockRejectedValueOnce(new Error('storage unavailable'))
  staged = original
  await expect(storage.verifyObject(constraints)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  await expect(storage.verifyObject(constraints)).resolves.toHaveProperty('verified', true)
})

test('CloudBase deletion validates every requested object result', async () => {
  const cloud = {
    getUploadMetadata: jest.fn(),
    downloadFile: jest.fn().mockRejectedValue({ code: 'STORAGE_FILE_NONEXIST' }),
    deleteFile: jest.fn().mockResolvedValue({
      fileList: [
        { fileID: 'cloud://published/clip.mp4', code: 'SUCCESS' },
        { fileID: 'cloud://staging/clip.mp4', code: 'SUCCESS' },
      ],
    }),
  }
  const storage = createStoragePort(createCloudBaseStorageProvider(cloud))
  await expect(storage.deleteObjects([
    'cloud://published/clip.mp4', 'cloud://staging/clip.mp4',
  ])).resolves.toEqual({ deleted: true, count: 2 })
  expect(cloud.downloadFile).toHaveBeenCalledTimes(2)

  cloud.deleteFile.mockResolvedValueOnce({
    fileList: [{ fileID: 'cloud://published/clip.mp4', code: 'SUCCESS' }],
  })
  await expect(storage.deleteObjects([
    'cloud://published/clip.mp4', 'cloud://staging/clip.mp4',
  ])).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })

  cloud.deleteFile.mockResolvedValueOnce({
    fileList: [
      { fileID: 'cloud://published/clip.mp4', code: 'SUCCESS' },
      { fileID: 'cloud://staging/clip.mp4', code: 'SUCCESS' },
    ],
  })
  cloud.downloadFile.mockResolvedValueOnce({ fileContent: Buffer.from('still there') })
  await expect(storage.deleteObjects([
    'cloud://published/clip.mp4', 'cloud://staging/clip.mp4',
  ])).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })

  cloud.deleteFile.mockResolvedValueOnce({
    fileList: [
      { fileID: 'cloud://published/clip.mp4', code: 'STORAGE_FILE_NONEXIST' },
      { fileID: 'cloud://staging/clip.mp4', code: 'STORAGE_FILE_NONEXIST' },
    ],
  })
  cloud.downloadFile.mockRejectedValue({ code: 'STORAGE_FILE_NONEXIST' })
  await expect(storage.deleteObjects([
    'cloud://published/clip.mp4', 'cloud://staging/clip.mp4',
  ])).resolves.toEqual({ deleted: true, count: 2 })
})

test('legacy SDK construction remains supported and registered capability is checked only when invoked', async () => {
  const cloud = { getUploadMetadata: jest.fn(), downloadFile: jest.fn() }
  const port = createStoragePort(createCloudBaseStorageProvider(cloud))
  expect(await port.copyRegisteredObject({ staging_file_id: 'cloud://opaque', publish_object_key: `published/managed-clips-v1/${'a'.repeat(64)}/11111111-1111-4111-8111-111111111111.mp4`, byte_size: 24, content_sha256: 'a'.repeat(64), media_type: 'video/mp4' }))
    .toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
  expect(cloud.getUploadMetadata).not.toHaveBeenCalled()
})

test('composed storage exposes actual registered grant and copy ports without changing legacy results', async () => {
  const key = `managed-clips-v1/${'a'.repeat(64)}/staging/11111111-1111-4111-8111-111111111111.mp4`
  const cloud = { getUploadMetadata: jest.fn(async () => ({ data: { fileId: 'cloud://opaque', url: 'https://storage.invalid/put', token: 'token', authorization: 'signature', cosFileId: 'cos' } })), downloadFile: jest.fn() }
  const port = createStoragePort(createCloudBaseStorageProvider(cloud))
  expect(await port.authorizeRegisteredUpload({ object_key: key, byte_size: 256 * 1024 * 1024, content_sha256: 'a'.repeat(64), media_type: 'video/mp4' }))
    .toMatchObject({ outcome: 'external_issued', file_id: 'cloud://opaque', authorization: { expires_in_seconds: 300 } })
  const legacy = createStoragePort({ createPresignedUpload() {}, verifyPrivateObject() {} })
  expect(await legacy.authorizeRegisteredUpload({})).toEqual({ outcome: 'no_write', reason: 'not_started' })
  expect(await legacy.copyRegisteredObject({})).toEqual({ outcome: 'no_write', reason: 'not_started', permanent: false })
})
