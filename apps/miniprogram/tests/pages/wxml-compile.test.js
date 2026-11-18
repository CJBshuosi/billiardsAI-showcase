const path = require('path')
const compiler = require('miniprogram-compiler')


test('all mini program WXML templates compile with the official compiler', () => {
  const root = path.join(__dirname, '../../miniprogram')
  expect(() => compiler.wxmlToJs(root)).not.toThrow()
})
