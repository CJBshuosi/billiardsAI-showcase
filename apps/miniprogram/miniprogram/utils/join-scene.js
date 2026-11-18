const TABLE_CODE_PATTERN = /^[A-Za-z0-9_-]{32}$/u

function decode(value) {
  try {
    return decodeURIComponent(value)
  } catch (_error) {
    return ''
  }
}

function tableCodeFromLaunchOptions(options) {
  if (!options || typeof options !== 'object') return ''
  const scene = typeof options.scene === 'string' ? decode(options.scene) : ''
  const direct = typeof options.table_code === 'string' ? options.table_code : ''
  if (scene && direct && scene !== direct) return ''
  const tableCode = scene || direct
  return TABLE_CODE_PATTERN.test(tableCode) ? tableCode : ''
}

module.exports = { tableCodeFromLaunchOptions }
