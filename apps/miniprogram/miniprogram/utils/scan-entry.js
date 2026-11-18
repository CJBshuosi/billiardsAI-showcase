const { tableCodeFromLaunchOptions } = require('./join-scene')
function tableCodeFromScanResult(result) {
  if (!result || typeof result !== 'object') return ''
  if (typeof result.path === 'string' && result.path) {
    const parts = result.path.replace(/^\//u, '').split('?')
    if (parts.length !== 2 || parts[0] !== 'pages/join/index' || parts[1].includes('#')) return ''
    const values = {}
    try {
      for (const field of parts[1].split('&')) {
        const at = field.indexOf('=')
        if (at < 0) return ''
        const key = decodeURIComponent(field.slice(0, at))
        if (key !== 'scene' && key !== 'table_code') continue
        if (Object.prototype.hasOwnProperty.call(values, key)) return ''
        values[key] = decodeURIComponent(field.slice(at + 1))
      }
    } catch (_error) { return '' }
    return tableCodeFromLaunchOptions(values)
  }
  return tableCodeFromLaunchOptions({ table_code: result.result })
}
module.exports = { tableCodeFromScanResult }
