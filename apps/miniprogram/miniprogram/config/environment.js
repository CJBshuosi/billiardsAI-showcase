function loadEnvironment() {
  let config
  try {
    config = require('./env.local')
  } catch (_error) {
    return { ok: false, error: { code: 'ENVIRONMENT_MISMATCH' } }
  }
  const value = config && config.cloudbaseEnvId
  if (!value || value === 'REPLACE_WITH_CLOUDBASE_ENV_ID') {
    return { ok: false, error: { code: 'ENVIRONMENT_MISMATCH' } }
  }
  let preview = null
  try {
    preview = require('./preview-entry.local')
  } catch (_error) {
    preview = null
  }
  const previewTableCode = preview && /^[A-Za-z0-9_-]{32}$/u.test(preview.tableCode || '')
    ? preview.tableCode
    : null
  return { ok: true, data: { cloudbaseEnvId: value, previewTableCode } }
}

module.exports = { loadEnvironment }
