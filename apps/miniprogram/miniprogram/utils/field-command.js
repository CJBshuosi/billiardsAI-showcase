function defaultRandomBytes() {
  return new Promise((resolve, reject) => {
    wx.getRandomValues({
      length: 16,
      success: ({ randomValues }) => resolve(new Uint8Array(randomValues)),
      fail: reject,
    })
  })
}


function roleLabel(role) { return role === 'blue' ? '蓝方' : '红方' }


function groupLabel(group) { return group === 'solid' ? '全色' : '花色' }


function describeFieldCommand(commandType, payload = {}, activeRole = null) {
  if (commandType === 'claim_role') return `已认领${roleLabel(payload.role)}身份`
  if (commandType === 'release_role') return '已释放本场身份'
  if (commandType === 'set_breaker') return `已设置${roleLabel(payload.role)}开球`
  if (commandType === 'assign_groups') {
    const blueGroup = payload.red_group === 'stripe' ? 'solid' : 'stripe'
    return `已设置红方${groupLabel(payload.red_group)}、蓝方${groupLabel(blueGroup)}`
  }
  if (commandType === 'continue_turn') return `${roleLabel(activeRole)}继续击球`
  if (commandType === 'switch_turn') {
    return `已请求切换至${roleLabel(activeRole === 'red' ? 'blue' : 'red')}`
  }
  if (commandType === 'end_session') return '本场已结束'
  return '现场状态已更新'
}


async function createCommandIntent(commandType, payload, revision, randomBytes = defaultRandomBytes) {
  if (typeof commandType !== 'string' || !Number.isInteger(revision) || revision < 0) {
    throw new Error('field command intent is invalid')
  }
  const bytes = Array.from(await randomBytes())
  if (bytes.length !== 16 || bytes.some((value) => !Number.isInteger(value) || value < 0 || value > 255)) {
    throw new Error('field command idempotency source is invalid')
  }
  return {
    command_type: commandType,
    payload: { ...(payload || {}) },
    expected_revision: revision,
    idempotency_key: bytes.map((value) => value.toString(16).padStart(2, '0')).join(''),
  }
}


function nextCommandState(previous, command) {
  if (!command || typeof command.command_id !== 'string') return previous
  if (previous && previous.commandId !== command.command_id) return previous
  return {
    commandId: command.command_id,
    state: command.status,
    resultCode: command.public_result_code || null,
  }
}


function storageKey(sessionId) { return `field-command:${sessionId}` }


function loadPendingCommand(storage, sessionId) {
  const value = storage.getStorageSync(storageKey(sessionId))
  if (!value || typeof value.commandId !== 'string'
      || !['pending', 'claimed'].includes(value.state)) return null
  return { commandId: value.commandId, state: value.state, resultCode: null }
}


function rememberPendingCommand(storage, sessionId, command) {
  if (command && ['pending', 'claimed'].includes(command.state)) {
    storage.setStorageSync(storageKey(sessionId), {
      commandId: command.commandId, state: command.state, resultCode: null,
    })
    return
  }
  storage.removeStorageSync(storageKey(sessionId))
}


module.exports = {
  createCommandIntent,
  describeFieldCommand,
  loadPendingCommand,
  nextCommandState,
  rememberPendingCommand,
}
