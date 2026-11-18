const crypto = require('crypto')

const {
  authenticate,
  canonicalSignatureInput,
  createCloudAuthRepositories,
} = require('../../cloudfunctions/sync-api/lib/auth')


const SECRET = 'development-device-secret-32-bytes'


function createRequest(overrides = {}) {
  const body = JSON.stringify({ schema_version: '1.0' })
  const timestamp = '1787752800'
  const nonce = 'nonce-0000000000000001'
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const input = canonicalSignatureInput({
    method: 'POST', path: '/v1/sync/jobs', timestamp, nonce, bodyHash,
  })
  const signature = crypto.createHmac('sha256', SECRET).update(input).digest('hex')
  return {
    method: 'POST',
    path: '/v1/sync/jobs',
    body,
    headers: {
      'x-device-id': 'device-raytheon-1',
      'x-timestamp': timestamp,
      'x-nonce': nonce,
      'x-body-sha256': bodyHash,
      'x-signature': signature,
    },
    ...overrides,
  }
}


function createDependencies() {
  const consumed = new Set()
  return {
    nowSeconds: () => 1787752800,
    deviceRepo: {
      async getActiveDevice(deviceId) {
        return deviceId === 'device-raytheon-1'
          ? { device_id: deviceId, table_id: 'table-1', secret: SECRET, status: 'active' }
          : null
      },
    },
    nonceRepo: {
      async consume(deviceId, nonce) {
        const key = `${deviceId}:${nonce}`
        if (consumed.has(key)) return false
        consumed.add(key)
        return true
      },
    },
  }
}


test('authenticates a correctly signed device request and returns table scope', async () => {
  await expect(authenticate(createRequest(), createDependencies())).resolves.toEqual({
    device_id: 'device-raytheon-1',
    table_id: 'table-1',
  })
})


test('rejects nonce replay even when signature is valid', async () => {
  const dependencies = createDependencies()
  const request = createRequest()
  await authenticate(request, dependencies)
  await expect(authenticate(request, dependencies))
    .rejects.toMatchObject({ code: 'SYNC_REPLAYED' })
})


test('rejects body tampering and stale timestamps before nonce consumption', async () => {
  const dependencies = createDependencies()
  dependencies.nonceRepo.consume = jest.fn()
  await expect(authenticate(createRequest({ body: '{"tampered":true}' }), dependencies))
    .rejects.toMatchObject({ code: 'SYNC_UNAUTHORIZED' })
  await expect(authenticate(createRequest({
    headers: { ...createRequest().headers, 'x-timestamp': '1787752000' },
  }), dependencies)).rejects.toMatchObject({ code: 'SYNC_UNAUTHORIZED' })
  expect(dependencies.nonceRepo.consume).not.toHaveBeenCalled()
})


test('first use of a nonce succeeds when CloudBase reports a missing document', async () => {
  const writes = new Map()
  const transaction = {
    collection(name) {
      return {
        doc(id) {
          return {
            async get() {
              throw new Error(`document.get:fail document with _id ${id} does not exist`)
            },
            async set({ data }) {
              writes.set(`${name}/${id}`, data)
            },
          }
        },
      }
    },
  }
  const repositories = createCloudAuthRepositories({
    runTransaction: async (callback) => callback(transaction),
  })

  await expect(repositories.nonceRepo.consume(
    'device-raytheon-1', 'nonce-0000000000000001', 1787752800,
  )).resolves.toBe(true)
  expect(writes.size).toBe(1)
})
