const path = require('path')


const routerPath = path.resolve(__dirname, '../../cloudfunctions/index.js')


test('shared cloud-function entry routes only declared functions', () => {
  const { entryModulePath } = require(routerPath)

  expect(entryModulePath('client-api')).toBe('./client-api/index')
  expect(entryModulePath('sync-api')).toBe('./sync-api/index')
  expect(entryModulePath('ranking-worker')).toBe('./ranking-worker/index')
  expect(entryModulePath('media-retention')).toBe('./media-retention/index')
  expect(entryModulePath('shot-projection-worker')).toBe('./shot-projection-worker/index')
  expect(() => entryModulePath('unknown')).toThrow('FUNCTION_ENTRY_INVALID')
  expect(() => entryModulePath('')).toThrow('FUNCTION_ENTRY_INVALID')
})
