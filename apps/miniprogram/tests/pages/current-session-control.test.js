const fs = require('fs')
const path = require('path')

const {
  createCommandIntent,
  loadPendingCommand,
  nextCommandState,
  rememberPendingCommand,
  describeFieldCommand,
} = require('../../miniprogram/utils/field-command')


test('command intents bind revision and a fresh opaque idempotency key', async () => {
  const intent = await createCommandIntent(
    'claim_role', { role: 'red' }, 4, () => Buffer.alloc(16, 7),
  )
  expect(intent).toEqual({
    command_type: 'claim_role', payload: { role: 'red' }, expected_revision: 4,
    idempotency_key: '07070707070707070707070707070707',
  })
})


test('wechat random values create a different key for every field command', async () => {
  const values = [
    Uint8Array.from({ length: 16 }, (_, index) => index + 1),
    Uint8Array.from({ length: 16 }, (_, index) => index + 17),
  ]
  global.wx = {
    getRandomValues: jest.fn(({ length, success }) => {
      expect(length).toBe(16)
      success({ randomValues: values.shift().buffer })
    }),
  }

  const first = await createCommandIntent('claim_role', { role: 'red' }, 0)
  const second = await createCommandIntent('set_breaker', { role: 'red' }, 2)

  expect(first.idempotency_key).toBe('0102030405060708090a0b0c0d0e0f10')
  expect(second.idempotency_key).toBe('1112131415161718191a1b1c1d1e1f20')
  expect(second.idempotency_key).not.toBe(first.idempotency_key)
  expect(wx.getRandomValues).toHaveBeenCalledTimes(2)
  delete global.wx
})


test('pending command survives page reload until a terminal result clears it', () => {
  const values = new Map()
  const storage = {
    getStorageSync: (key) => values.get(key),
    setStorageSync: (key, value) => values.set(key, value),
    removeStorageSync: (key) => values.delete(key),
  }
  rememberPendingCommand(storage, 'session-1', { commandId: 'command-1', state: 'pending' })
  expect(loadPendingCommand(storage, 'session-1')).toEqual({
    commandId: 'command-1', state: 'pending', resultCode: null,
  })
  rememberPendingCommand(storage, 'session-1', { commandId: 'command-1', state: 'applied' })
  expect(loadPendingCommand(storage, 'session-1')).toBeNull()
})


test('submitted commands remain pending until edge terminal status arrives', () => {
  const pending = nextCommandState(null, {
    command_id: 'command-1', status: 'pending', public_result_code: null,
  })
  expect(pending).toEqual({ commandId: 'command-1', state: 'pending', resultCode: null })
  expect(nextCommandState(pending, {
    command_id: 'command-1', status: 'applied', public_result_code: 'APPLIED',
  })).toEqual({ commandId: 'command-1', state: 'applied', resultCode: 'APPLIED' })
})


test('phone command feedback explains the applied match change in Chinese', () => {
  expect(describeFieldCommand('set_breaker', { role: 'red' })).toBe('已设置红方开球')
  expect(describeFieldCommand('assign_groups', { red_group: 'stripe' }))
    .toBe('已设置红方花色、蓝方全色')
  expect(describeFieldCommand('continue_turn', {}, 'red')).toBe('红方继续击球')
  expect(describeFieldCommand('switch_turn', {}, 'red')).toBe('已请求切换至蓝方')
})


test('current field UI lets either joined player confirm ending the session', () => {
  const root = path.join(__dirname, '../../miniprogram/pages/current-session')
  const source = fs.readFileSync(path.join(root, 'index.js'), 'utf8')
  const view = fs.readFileSync(path.join(root, 'index.wxml'), 'utf8')
  for (const command of [
    'claim_role', 'release_role', 'set_breaker', 'assign_groups',
    'continue_turn', 'switch_turn', 'end_session',
  ]) expect(`${source}\n${view}`).toContain(command)
  expect(`${source}\n${view}`).not.toMatch(/data-command="(?:start|pause|reset)"/u)
  expect(source).toContain('wx.showModal')
  expect(view).toContain('提交给双方确认')
  expect(view).toContain('任一修改都会让旧确认失效')
  expect(view).toContain('通知球桌开始')
  expect(view).not.toContain('请由当前击球方结束本场')
  expect(view).toContain('bindtap="endSession" disabled="{{submitting')
  expect(source).not.toContain('this.data.activeRole !== this.data.actorRole')
  expect(view).toContain('等待现场确认')
  expect(view).toContain("{{commandMessage || '现场状态已更新'}}")
  expect(view).not.toMatch(/\((?:release_role|set_breaker|assign_groups|continue_turn|switch_turn|end_session)\)/u)
})
