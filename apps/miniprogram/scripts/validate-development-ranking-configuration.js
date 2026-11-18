const fs = require('fs')

const { validateRankingConfiguration } = require('./validate-ranking-configuration')


const FORBIDDEN_DATA_KEYS = ['users', 'matches', 'ratings', 'events', 'snapshots']
const ENVIRONMENT_ID_PATTERN = /^[a-z0-9][a-z0-9-]{2,127}$/iu


function fail(code) {
  const error = new Error(code)
  error.code = code
  throw error
}


function validateDevelopmentRankingConfiguration(configuration) {
  if (!configuration || configuration.environment !== 'development'
      || FORBIDDEN_DATA_KEYS.some(key => Object.hasOwn(configuration, key))) {
    if (configuration && FORBIDDEN_DATA_KEYS.some(key => Object.hasOwn(configuration, key))) {
      fail('RANKING_TEST_CONFIGURATION_CONFLICT')
    }
    fail('RANKING_TEST_CONFIGURATION_INVALID')
  }
  const scope = configuration.scope
  if (!scope || scope.ranking_scope_id !== 'zhoukou-urban-dev'
      || scope.mode !== 'development_acceptance' || scope.is_test !== true) {
    fail('RANKING_TEST_CONFIGURATION_INVALID')
  }
  let summary
  try {
    summary = validateRankingConfiguration(configuration)
  } catch (_error) {
    fail('RANKING_TEST_CONFIGURATION_INVALID')
  }
  if (!summary.opening_ready) return summary
  if (!ENVIRONMENT_ID_PATTERN.test(configuration.environment_id || '')
      || !configuration.seasons.every(item => item.is_test === true)
      || !configuration.venues.every(item => item.is_test === true)
      || !configuration.tables.every(item => item.is_test === true)) {
    fail('RANKING_TEST_CONFIGURATION_INVALID')
  }
  return summary
}


if (require.main === module) {
  try {
    const file = process.argv[2]
    if (!file) fail('RANKING_CONFIGURATION_FILE_REQUIRED')
    const configuration = JSON.parse(fs.readFileSync(file, 'utf8'))
    process.stdout.write(`${JSON.stringify(validateDevelopmentRankingConfiguration(configuration))}\n`)
  } catch (error) {
    process.stderr.write(`${error.code || error.message}\n`)
    process.exitCode = 2
  }
}


module.exports = { validateDevelopmentRankingConfiguration }
