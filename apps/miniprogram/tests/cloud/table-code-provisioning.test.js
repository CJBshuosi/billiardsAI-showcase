const crypto = require('crypto')

const {
  createProvisioningHandler,
  createRegistrationHandler,
} = require('../../provisioning/table-code')

const tableCode = 'AbCdEfGhIjKlMnOpQrStUvWxYz012345'
const authorization = 'provisioning-secret-token'
const expectedAuthDigest = crypto.createHash('sha256').update(authorization).digest('hex')
const fixedNow = '2026-08-27T12:00:00.000Z'
const image = Buffer.from('\x89PNG\r\n\x1a\npermanent-table-code', 'binary')

function cloudFixture({ qrError = null, missingDocument = false } = {}) {
  const saved = []
  const previous = {
    schema_version: '1.0',
    table_id: 'raytheon-01',
    code_digest: 'old-digest',
    status: 'active',
    current_session_id: 'session-existing',
  }
  let transactions = 0
  const document = {
    get: jest.fn(async () => {
      if (missingDocument) throw new Error('document.get:fail document does not exist')
      return { data: previous }
    }),
    set: jest.fn(async ({ data }) => saved.push(data)),
  }
  const db = {
    runTransaction: jest.fn(async (callback) => {
      transactions += 1
      return callback({ collection: () => ({ doc: () => document }) })
    }),
  }
  const getUnlimited = jest.fn(async () => {
    if (qrError) throw qrError
    return image
  })
  return {
    cloud: {
      database: () => db,
      openapi: { wxacode: { getUnlimited } },
    },
    getUnlimited,
    saved,
    transactions: () => transactions,
  }
}

test('provisioning persists only the digest after QR generation succeeds', async () => {
  const fixture = cloudFixture()
  const handler = createProvisioningHandler({
    cloud: fixture.cloud,
    expectedAuthDigest,
    now: () => fixedNow,
  })

  const result = await handler({
    authorization,
    table_code: tableCode,
    table_id: 'raytheon-01',
  })

  expect(fixture.getUnlimited).toHaveBeenCalledWith({
    page: 'pages/join/index',
    scene: tableCode,
    envVersion: 'develop',
    checkPath: true,
  })
  expect(fixture.saved).toEqual([{
    schema_version: '1.0',
    table_id: 'raytheon-01',
    code_digest: crypto.createHash('sha256').update(tableCode).digest('hex'),
    status: 'active',
    current_session_id: 'session-existing',
    updated_at: fixedNow,
  }])
  expect(JSON.stringify(fixture.saved)).not.toContain(tableCode)
  expect(result).toEqual({
    schema_version: '1.0',
    status: 'PROVISIONED',
    image_base64: image.toString('base64'),
    image_sha256: crypto.createHash('sha256').update(image).digest('hex'),
  })
})

test('development preview registration persists the digest without calling wxacode', async () => {
  const fixture = cloudFixture()
  const handler = createRegistrationHandler({
    cloud: fixture.cloud,
    expectedAuthDigest,
    now: () => fixedNow,
  })

  const result = await handler({
    authorization,
    table_code: tableCode,
    table_id: 'raytheon-01',
  })

  expect(fixture.getUnlimited).not.toHaveBeenCalled()
  expect(fixture.saved).toEqual([{
    schema_version: '1.0',
    table_id: 'raytheon-01',
    code_digest: crypto.createHash('sha256').update(tableCode).digest('hex'),
    status: 'active',
    current_session_id: 'session-existing',
    updated_at: fixedNow,
  }])
  expect(result).toEqual({ schema_version: '1.0', status: 'REGISTERED' })
})

test('development preview registration creates the first table document when it does not exist', async () => {
  const fixture = cloudFixture({ missingDocument: true })
  const handler = createRegistrationHandler({
    cloud: fixture.cloud,
    expectedAuthDigest,
    now: () => fixedNow,
  })

  await expect(handler({
    authorization,
    table_code: tableCode,
    table_id: 'raytheon-01',
  })).resolves.toEqual({ schema_version: '1.0', status: 'REGISTERED' })
  expect(fixture.saved[0].current_session_id).toBeNull()
})

test('wrong authorization is rejected before QR or database access', async () => {
  const fixture = cloudFixture()
  const handler = createProvisioningHandler({
    cloud: fixture.cloud,
    expectedAuthDigest,
    now: () => fixedNow,
  })

  await expect(handler({
    authorization: 'wrong-token',
    table_code: tableCode,
    table_id: 'raytheon-01',
  })).rejects.toThrow('PROVISION_AUTH_INVALID')
  expect(fixture.getUnlimited).not.toHaveBeenCalled()
  expect(fixture.transactions()).toBe(0)
})

test('QR failure leaves the existing table-code document untouched', async () => {
  const fixture = cloudFixture({ qrError: new Error('WXACODE_FAILED') })
  const handler = createProvisioningHandler({
    cloud: fixture.cloud,
    expectedAuthDigest,
    now: () => fixedNow,
  })

  await expect(handler({
    authorization,
    table_code: tableCode,
    table_id: 'raytheon-01',
  })).rejects.toThrow('WXACODE_FAILED')
  expect(fixture.transactions()).toBe(0)
  expect(fixture.saved).toEqual([])
})
