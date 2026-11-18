const crypto = require('crypto')

const TABLE_CODE_PATTERN = /^[A-Za-z0-9_-]{32}$/u
const TABLE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex')
}

function authorized(value, expectedDigest) {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/u.test(expectedDigest)) return false
  const actual = Buffer.from(sha256(value), 'hex')
  const expected = Buffer.from(expectedDigest, 'hex')
  return actual.length === expected.length && crypto.timingSafeEqual(actual, expected)
}

function imageBuffer(result) {
  if (Buffer.isBuffer(result)) return result
  if (result && Buffer.isBuffer(result.buffer)) return result.buffer
  if (result && typeof result.buffer === 'string') return Buffer.from(result.buffer, 'base64')
  throw new Error('WXACODE_RESPONSE_INVALID')
}

async function previousDocument(document) {
  try {
    const existing = await document.get()
    return existing && existing.data ? existing.data : {}
  } catch (error) {
    if (/document.*does not exist|DOCUMENT_NOT_EXIST/i.test(String(error && error.message || error))) {
      return {}
    }
    throw error
  }
}

function createProvisioningHandler({ cloud, expectedAuthDigest, now }) {
  if (!cloud || typeof now !== 'function') throw new Error('PROVISION_CONFIG_INVALID')
  return async (event) => {
    if (!event || !authorized(event.authorization, expectedAuthDigest)) {
      throw new Error('PROVISION_AUTH_INVALID')
    }
    const tableCode = event.table_code
    const tableId = event.table_id
    if (!TABLE_CODE_PATTERN.test(tableCode || '') || !TABLE_ID_PATTERN.test(tableId || '')) {
      throw new Error('PROVISION_REQUEST_INVALID')
    }

    const image = imageBuffer(await cloud.openapi.wxacode.getUnlimited({
      page: 'pages/join/index',
      scene: tableCode,
      envVersion: 'develop',
      checkPath: true,
    }))
    const codeDigest = sha256(tableCode)
    const updatedAt = now()
    await cloud.database().runTransaction(async (transaction) => {
      const document = transaction.collection('table_join_codes').doc(tableId)
      const previous = await previousDocument(document)
      await document.set({
        data: {
          schema_version: '1.0',
          table_id: tableId,
          code_digest: codeDigest,
          status: 'active',
          current_session_id: previous.current_session_id || null,
          updated_at: updatedAt,
        },
      })
    })
    return {
      schema_version: '1.0',
      status: 'PROVISIONED',
      image_base64: image.toString('base64'),
      image_sha256: sha256(image),
    }
  }
}

function createRegistrationHandler({ cloud, expectedAuthDigest, now }) {
  if (!cloud || typeof now !== 'function') throw new Error('PROVISION_CONFIG_INVALID')
  return async (event) => {
    if (!event || !authorized(event.authorization, expectedAuthDigest)) {
      throw new Error('PROVISION_AUTH_INVALID')
    }
    const tableCode = event.table_code
    const tableId = event.table_id
    if (!TABLE_CODE_PATTERN.test(tableCode || '') || !TABLE_ID_PATTERN.test(tableId || '')) {
      throw new Error('PROVISION_REQUEST_INVALID')
    }

    const codeDigest = sha256(tableCode)
    const updatedAt = now()
    await cloud.database().runTransaction(async (transaction) => {
      const document = transaction.collection('table_join_codes').doc(tableId)
      const previous = await previousDocument(document)
      await document.set({
        data: {
          schema_version: '1.0',
          table_id: tableId,
          code_digest: codeDigest,
          status: 'active',
          current_session_id: previous.current_session_id || null,
          updated_at: updatedAt,
        },
      })
    })
    return { schema_version: '1.0', status: 'REGISTERED' }
  }
}

module.exports = { createProvisioningHandler, createRegistrationHandler }
