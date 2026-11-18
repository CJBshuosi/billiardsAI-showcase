const {
  assertRankingScopePolicy,
  resolveRuntimeRankingScope,
} = require('../../cloudfunctions/shared/ranking-scope-policy')


test('development runtime resolves only to the isolated test scope', () => {
  expect(resolveRuntimeRankingScope({ environment: 'development' })).toEqual({
    ranking_scope_id: 'zhoukou-urban-dev',
    mode: 'development_acceptance',
    is_test: true,
  })
})


test('production runtime resolves only to the formal scope', () => {
  expect(resolveRuntimeRankingScope({ environment: 'production' })).toEqual({
    ranking_scope_id: 'zhoukou-urban',
    mode: 'formal',
    is_test: false,
  })
})


test.each([
  ['development', { ranking_scope_id: 'zhoukou-urban', mode: 'formal', is_test: false }],
  ['production', { ranking_scope_id: 'zhoukou-urban-dev', mode: 'development_acceptance', is_test: true }],
])('%s runtime rejects the other environment scope', (environment, scope) => {
  expect(() => assertRankingScopePolicy({ environment, scope }))
    .toThrow('RANKING_ENVIRONMENT_SCOPE_MISMATCH')
})


test('development scope must remain marked as test data', () => {
  try {
    assertRankingScopePolicy({
      environment: 'development',
      scope: {
        ranking_scope_id: 'zhoukou-urban-dev', mode: 'development_acceptance', is_test: false,
      },
    })
    throw new Error('expected ranking scope policy to reject the document')
  } catch (error) {
    expect(error).toMatchObject({
      message: 'RANKING_TEST_SCOPE_REQUIRED', code: 'RANKING_TEST_SCOPE_REQUIRED',
    })
  }
})


test('formal scope must never be marked as test data', () => {
  expect(() => assertRankingScopePolicy({
    environment: 'production',
    scope: { ranking_scope_id: 'zhoukou-urban', mode: 'formal', is_test: true },
  })).toThrow('RANKING_FORMAL_SCOPE_REQUIRED')
})


test.each([
  [undefined, { ranking_scope_id: 'zhoukou-urban-dev', mode: 'development_acceptance', is_test: true }],
  ['staging', { ranking_scope_id: 'zhoukou-urban-dev', mode: 'development_acceptance', is_test: true }],
  ['development', { ranking_scope_id: 'zhoukou-urban-dev', mode: 'formal', is_test: true }],
])('unsupported or malformed environment/scope policy fails closed', (environment, scope) => {
  expect(() => assertRankingScopePolicy({ environment, scope }))
    .toThrow('RANKING_TEST_CONFIGURATION_INVALID')
})
