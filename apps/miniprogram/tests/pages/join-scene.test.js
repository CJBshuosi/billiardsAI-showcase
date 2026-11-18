const { tableCodeFromLaunchOptions } = require('../../miniprogram/utils/join-scene')


const TABLE_CODE = 'DevelopmentTableCode_00000000001'


test('production scene and development table_code resolve one permanent code', () => {
  expect(tableCodeFromLaunchOptions({ scene: encodeURIComponent(TABLE_CODE) }))
    .toBe(TABLE_CODE)
  expect(tableCodeFromLaunchOptions({ table_code: TABLE_CODE })).toBe(TABLE_CODE)
})


test('malformed, wrong-length, and conflicting launch values fail closed', () => {
  expect(tableCodeFromLaunchOptions({ scene: 'short' })).toBe('')
  expect(tableCodeFromLaunchOptions({ scene: '%E0%A4%A' })).toBe('')
  expect(tableCodeFromLaunchOptions({
    scene: TABLE_CODE,
    table_code: 'B'.repeat(32),
  })).toBe('')
})
