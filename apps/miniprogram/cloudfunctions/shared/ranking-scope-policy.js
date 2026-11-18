const RUNTIME_SCOPES = Object.freeze({
  development: Object.freeze({
    ranking_scope_id: 'zhoukou-urban-dev',
    mode: 'development_acceptance',
    is_test: true,
  }),
  production: Object.freeze({
    ranking_scope_id: 'zhoukou-urban',
    mode: 'formal',
    is_test: false,
  }),
})


function fail(code) {
  const error = new Error(code)
  error.code = code
  throw error
}


function normalizeRuntimeEnvironment(environment) {
  if (environment !== 'development' && environment !== 'production') {
    fail('RANKING_TEST_CONFIGURATION_INVALID')
  }
  return environment
}


function resolveRuntimeRankingScope({ environment }) {
  const normalized = normalizeRuntimeEnvironment(environment)
  return { ...RUNTIME_SCOPES[normalized] }
}


function assertRankingScopePolicy({ environment, scope }) {
  const expected = resolveRuntimeRankingScope({ environment })
  if (!scope || typeof scope !== 'object') fail('RANKING_TEST_CONFIGURATION_INVALID')
  if (scope.ranking_scope_id !== expected.ranking_scope_id) {
    fail('RANKING_ENVIRONMENT_SCOPE_MISMATCH')
  }
  if (scope.mode !== expected.mode) fail('RANKING_TEST_CONFIGURATION_INVALID')
  if (environment === 'development' && scope.is_test !== true) {
    fail('RANKING_TEST_SCOPE_REQUIRED')
  }
  if (environment === 'production' && scope.is_test !== false) {
    fail('RANKING_FORMAL_SCOPE_REQUIRED')
  }
  return { ...expected }
}


module.exports = {
  FORMAL_RANKING_SCOPE: RUNTIME_SCOPES.production,
  DEVELOPMENT_RANKING_SCOPE: RUNTIME_SCOPES.development,
  normalizeRuntimeEnvironment,
  resolveRuntimeRankingScope,
  assertRankingScopePolicy,
}
