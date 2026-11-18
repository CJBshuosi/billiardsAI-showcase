const IDENTITY_ESTABLISHED_KEY = 'billiardsAI.identity-established.v1'


function hasEstablishedIdentity(storage) {
  return storage.getStorageSync(IDENTITY_ESTABLISHED_KEY) === true
}


function rememberEstablishedIdentity(storage) {
  storage.setStorageSync(IDENTITY_ESTABLISHED_KEY, true)
}


function resolveEntryState({ configurationError, identityEstablished }) {
  if (configurationError) {
    return {
      message: `配置未完成：${configurationError}`,
      ready: false,
      resume: false,
    }
  }
  return identityEstablished
    ? { message: '正在识别长期身份…', ready: true, resume: true }
    : { message: '开发环境已连接', ready: true, resume: false }
}

function resolveEntryRoute({ configurationError, identityEstablished, pendingTableCode }) {
  if (configurationError) return null
  if (/^[A-Za-z0-9_-]{32}$/u.test(pendingTableCode || '')) {
    return `/pages/join/index?table_code=${encodeURIComponent(pendingTableCode)}`
  }
  return identityEstablished ? '/pages/home/index' : null
}


module.exports = {
  hasEstablishedIdentity,
  rememberEstablishedIdentity,
  resolveEntryRoute,
  resolveEntryState,
}
