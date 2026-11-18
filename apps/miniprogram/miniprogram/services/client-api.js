const { normalizeResult } = require('../utils/result')

async function callClientApi(action, payload = {}) {
  try {
    const response = await wx.cloud.callFunction({
      name: 'client-api',
      data: { action, payload, schema_version: '1.0' },
    })
    return normalizeResult(response.result)
  } catch (_error) {
    return { ok: false, error: { code: 'NETWORK_UNAVAILABLE' } }
  }
}

module.exports = { callClientApi }
