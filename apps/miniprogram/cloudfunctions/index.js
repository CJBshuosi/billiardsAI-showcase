const ENTRY_MODULES = Object.freeze({
  'client-api': './client-api/index',
  'sync-api': './sync-api/index',
  'ranking-worker': './ranking-worker/index',
  'media-retention': './media-retention/index',
  'shot-projection-worker': './shot-projection-worker/index',
})


function entryModulePath(entryName) {
  const modulePath = ENTRY_MODULES[entryName]
  if (!modulePath) throw new Error('FUNCTION_ENTRY_INVALID')
  return modulePath
}


exports.main = (...args) => require(entryModulePath(
  process.env.BILLIARDS_FUNCTION_ENTRY,
)).main(...args)
exports.entryModulePath = entryModulePath
