const fs = require('fs')
const path = require('path')


const readPage = (name) => fs.readFileSync(
  path.join(__dirname, `../../miniprogram/pages/${name}/index.wxml`),
  'utf8',
)


test('current session page distinguishes field and command states', () => {
  const view = readPage('current-session')
  expect(view).toContain("status === 'open'")
  expect(view).toContain("status === 'ended'")
  expect(view).toContain("commandState === 'pending'")
  expect(view).toContain("commandState === 'applied'")
  expect(view).toContain("commandState === 'rejected'")
  expect(view).toContain('TABLE_CODE_DISABLED')
  expect(view).toContain('SESSION_NOT_JOINABLE')
  expect(view).toContain('NETWORK_UNAVAILABLE')
  expect(view).toContain('bindtap="refresh"')
})


test('scan entry stays narrow while the current field owns participant controls', () => {
  const join = readPage('join')
  const current = readPage('current-session')
  expect(join).not.toMatch(/切换用户|继续击球|认领红色|认领蓝色/u)
  expect(current).toMatch(/切换用户|继续击球|认领红色|认领蓝色/u)
  expect(`${join}\n${current}`).not.toContain('join.confirmRole')
})

test('join page submits a permanent table code without token wording', () => {
  const view = readPage('join')
  expect(view).toContain('永久球桌码')
  expect(view).not.toMatch(/临时码|token/u)
})
