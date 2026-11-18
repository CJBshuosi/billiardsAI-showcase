const {
  hasEstablishedIdentity,
  rememberEstablishedIdentity,
  resolveEntryRoute,
  resolveEntryState,
} = require('../../miniprogram/utils/identity-session')


function createStorage() {
  const values = new Map()
  return {
    getStorageSync(key) {
      return values.get(key)
    },
    setStorageSync(key, value) {
      values.set(key, value)
    },
  }
}


test('identity is remembered only after the first successful login', () => {
  const storage = createStorage()

  expect(hasEstablishedIdentity(storage)).toBe(false)
  rememberEstablishedIdentity(storage)
  expect(hasEstablishedIdentity(storage)).toBe(true)
})

test('a packaged preview table code takes precedence over identity resume', () => {
  expect(resolveEntryRoute({
    configurationError: null,
    identityEstablished: true,
    pendingTableCode: 'AbCdEfGhIjKlMnOpQrStUvWxYz012345',
  })).toBe('/pages/join/index?table_code=AbCdEfGhIjKlMnOpQrStUvWxYz012345')
  expect(resolveEntryRoute({
    configurationError: 'ENVIRONMENT_MISMATCH',
    identityEstablished: true,
    pendingTableCode: 'AbCdEfGhIjKlMnOpQrStUvWxYz012345',
  })).toBeNull()
})


test('entry requires an explicit first login and resumes returning users', () => {
  expect(resolveEntryState({ configurationError: null, identityEstablished: false }))
    .toEqual({ message: '开发环境已连接', ready: true, resume: false })
  expect(resolveEntryState({ configurationError: null, identityEstablished: true }))
    .toEqual({ message: '正在识别长期身份…', ready: true, resume: true })
  expect(resolveEntryState({ configurationError: 'CLOUD_ENV_MISSING', identityEstablished: true }))
    .toEqual({ message: '配置未完成：CLOUD_ENV_MISSING', ready: false, resume: false })
})
