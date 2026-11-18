const crypto = require('crypto')


class SyncApiError extends Error {
  constructor(code) {
    super(code)
    this.name = 'SyncApiError'
    this.code = code
  }
}


function canonicalSignatureInput({ method, path, timestamp, nonce, bodyHash }) {
  return `${method.toUpperCase()}\n${path}\n${timestamp}\n${nonce}\n${bodyHash}`
}


function getHeader(headers, name) {
  const entry = Object.entries(headers || {})
    .find(([key]) => key.toLowerCase() === name.toLowerCase())
  return entry ? String(entry[1]) : ''
}


function isHex(value, length) {
  return typeof value === 'string' && value.length === length && /^[0-9a-f]+$/u.test(value)
}


function secureEqualHex(left, right) {
  if (!isHex(left, 64) || !isHex(right, 64)) return false
  return crypto.timingSafeEqual(Buffer.from(left, 'hex'), Buffer.from(right, 'hex'))
}


async function authenticate(request, dependencies) {
  const deviceId = getHeader(request.headers, 'x-device-id')
  const timestamp = getHeader(request.headers, 'x-timestamp')
  const nonce = getHeader(request.headers, 'x-nonce')
  const claimedBodyHash = getHeader(request.headers, 'x-body-sha256')
  const signature = getHeader(request.headers, 'x-signature')
  if (!/^[A-Za-z0-9._-]{3,128}$/u.test(deviceId)
      || !/^\d{10}$/u.test(timestamp)
      || !/^[A-Za-z0-9._-]{16,128}$/u.test(nonce)
      || !isHex(claimedBodyHash, 64)
      || !isHex(signature, 64)) {
    throw new SyncApiError('SYNC_UNAUTHORIZED')
  }
  const timestampNumber = Number(timestamp)
  if (Math.abs(dependencies.nowSeconds() - timestampNumber) > 300) {
    throw new SyncApiError('SYNC_UNAUTHORIZED')
  }
  const actualBodyHash = crypto.createHash('sha256')
    .update(typeof request.body === 'string' ? request.body : '')
    .digest('hex')
  if (!secureEqualHex(claimedBodyHash, actualBodyHash)) {
    throw new SyncApiError('SYNC_UNAUTHORIZED')
  }

  const device = await dependencies.deviceRepo.getActiveDevice(deviceId)
  if (!device || device.status !== 'active' || typeof device.secret !== 'string') {
    throw new SyncApiError('SYNC_UNAUTHORIZED')
  }
  const expected = crypto.createHmac('sha256', device.secret)
    .update(canonicalSignatureInput({
      method: request.method,
      path: request.path,
      timestamp,
      nonce,
      bodyHash: claimedBodyHash,
    }))
    .digest('hex')
  if (!secureEqualHex(signature, expected)) {
    throw new SyncApiError('SYNC_UNAUTHORIZED')
  }
  if (!await dependencies.nonceRepo.consume(deviceId, nonce, timestampNumber)) {
    throw new SyncApiError('SYNC_REPLAYED')
  }
  return { device_id: device.device_id, table_id: device.table_id }
}


function createCloudAuthRepositories(db) {
  return {
    deviceRepo: {
      async getActiveDevice(deviceId) {
        const result = await db.collection('sync_devices').doc(deviceId).get()
        return result.data || null
      },
    },
    nonceRepo: {
      async consume(deviceId, nonce, timestamp) {
        const nonceId = crypto.createHash('sha256')
          .update(`${deviceId}:${nonce}`, 'utf8')
          .digest('hex')
        return db.runTransaction(async (transaction) => {
          const ref = transaction.collection('sync_nonces').doc(nonceId)
          let existing
          try {
            existing = await ref.get()
          } catch (error) {
            const message = error && error.message ? error.message : String(error)
            if (!/document with _id .+ does not exist/i.test(message)) throw error
            existing = { data: null }
          }
          if (existing.data) return false
          await ref.set({
            data: {
              schema_version: '1.0',
              device_id: deviceId,
              nonce_digest: nonceId,
              request_timestamp: timestamp,
              expires_at: new Date((timestamp + 600) * 1000).toISOString(),
            },
          })
          return true
        })
      },
    },
  }
}


module.exports = {
  SyncApiError,
  authenticate,
  canonicalSignatureInput,
  createCloudAuthRepositories,
  getHeader,
}
