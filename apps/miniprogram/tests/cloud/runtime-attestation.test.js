const crypto = require('crypto')

const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
const {
  createHandler,
  runtimeAttestationFromEnvironment,
} = require('../../cloudfunctions/sync-api/index')

const SECRET = 'development-device-secret-32-bytes'
const NOW_SECONDS = 1788782400

function signedEvent() {
  const method = 'POST'
  const path = '/v1/runtime-attestation'
  const body = '{}'
  const timestamp = String(NOW_SECONDS)
  const nonce = `nonce-${crypto.randomUUID()}`
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const signature = crypto.createHmac('sha256', SECRET).update(canonicalSignatureInput({
    method, path, timestamp, nonce, bodyHash,
  })).digest('hex')
  return {
    httpMethod: method, path, body,
    headers: {
      'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce,
      'x-body-sha256': bodyHash, 'x-signature': signature,
    },
  }
}

test('authenticated runtime attestation returns immutable deployment identity', async () => {
  const attestation = runtimeAttestationFromEnvironment({
    RANKING_RUNTIME_ENVIRONMENT: 'development',
    CLOUDBASE_ENV_ID: 'cloud-development-1',
  })
  expect(Object.isFrozen(attestation)).toBe(true)
  const handler = createHandler({
    auth: {
      nowSeconds: () => NOW_SECONDS,
      deviceRepo: { getActiveDevice: async () => ({
        device_id: 'device-1', table_id: 'table-1', status: 'active', secret: SECRET,
      }) },
      nonceRepo: { consume: async () => true },
    },
    runtimeAttestation: attestation,
  })

  const response = await handler(signedEvent())

  expect(response.statusCode).toBe(200)
  expect(JSON.parse(response.body)).toEqual({ ok: true, data: {
    schema_version: '1.0',
    runtime_environment: 'development',
    cloudbase_environment_id: 'cloud-development-1',
  } })
})

test('missing deployment identity is attested as unknown, never development by default', () => {
  expect(runtimeAttestationFromEnvironment({})).toEqual({
    schema_version: '1.0',
    runtime_environment: 'unknown',
    cloudbase_environment_id: 'unknown',
  })
})
