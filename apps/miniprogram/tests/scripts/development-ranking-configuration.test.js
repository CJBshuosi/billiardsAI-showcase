const fs = require('fs')
const path = require('path')

const {
  validateDevelopmentRankingConfiguration,
} = require('../../scripts/validate-development-ranking-configuration')
const {
  buildBootstrapPlan,
  commandFor,
} = require('../../scripts/bootstrap-development-ranking')


const root = path.resolve(__dirname, '../..')


function validConfiguration() {
  return {
    schema_version: '1.0',
    environment: 'development',
    environment_id: 'billiards-dev-1a2b3c',
    scope: {
      ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
      mode: 'development_acceptance', is_test: true,
      status: 'approved', publication_status: 'open', configuration_version: 1,
    },
    seasons: [{
      season_id: 'zhoukou-dev-s1', ranking_scope_id: 'zhoukou-urban-dev',
      is_test: true, status: 'active',
      starts_at: '2026-09-01T00:00:00.000Z', ends_at: '2026-10-31T00:00:00.000Z',
    }],
    venues: [{
      venue_id: 'venue-field-1', ranking_scope_id: 'zhoukou-urban-dev',
      is_test: true, display_name: '真实测试球房', status: 'approved',
    }],
    tables: [{
      table_id: 'table-field-1', venue_id: 'venue-field-1', is_test: true,
      status: 'approved', device_deployment_status: 'approved', device_id: 'device-field-1',
    }],
  }
}


test('checked-in development example is isolated but cannot open without real infrastructure', () => {
  const configuration = JSON.parse(fs.readFileSync(
    path.join(root, 'database/ranking-development.example.json'), 'utf8',
  ))

  expect(validateDevelopmentRankingConfiguration(configuration)).toEqual({
    ranking_scope_id: 'zhoukou-urban-dev', opening_ready: false,
    season_count: 0, venue_count: 0, table_count: 0,
  })
})


test('valid development configuration requires one real 60-day season and bound table', () => {
  expect(validateDevelopmentRankingConfiguration(validConfiguration())).toEqual({
    ranking_scope_id: 'zhoukou-urban-dev', opening_ready: true,
    season_count: 1, venue_count: 1, table_count: 1,
  })
})


test.each([
  configuration => { configuration.scope.ranking_scope_id = 'zhoukou-urban' },
  configuration => { configuration.scope.mode = 'formal' },
  configuration => { configuration.scope.is_test = false },
  configuration => { configuration.seasons[0].is_test = false },
  configuration => { configuration.venues[0].is_test = false },
  configuration => { configuration.tables[0].is_test = false },
])('development configuration rejects formal or unmarked records', mutate => {
  const configuration = validConfiguration()
  mutate(configuration)
  expect(() => validateDevelopmentRankingConfiguration(configuration))
    .toThrow('RANKING_TEST_CONFIGURATION_INVALID')
})


test('development configuration rejects data payloads that could fabricate ranking results', () => {
  for (const key of ['users', 'matches', 'ratings', 'events', 'snapshots']) {
    const configuration = validConfiguration()
    configuration[key] = []
    expect(() => validateDevelopmentRankingConfiguration(configuration))
      .toThrow('RANKING_TEST_CONFIGURATION_CONFLICT')
  }
})


test('bootstrap plan writes only configuration plus an empty test generation', () => {
  const plan = buildBootstrapPlan(validConfiguration())
  const collections = plan.map(operation => operation.collection)

  expect(collections).toEqual([
    'ranking_scopes', 'ranking_seasons', 'ranking_venues', 'ranking_tables',
    'rating_generations', 'rating_scope_heads',
  ])
  expect(plan.every(operation => operation.data.is_test === true)).toBe(true)
  expect(JSON.stringify(plan)).not.toMatch(/user_id|match_id|rating_event|snapshot_id/u)
})


test('bootstrap upsert query includes immutable scope identity so conflicts fail as duplicate keys', () => {
  const scopeOperation = buildBootstrapPlan(validConfiguration())[0]
  const command = JSON.parse(commandFor(scopeOperation).Command)

  expect(command.updates[0].q).toEqual({
    _id: scopeOperation.document_id,
    ranking_scope_id: 'zhoukou-urban-dev',
    mode: 'development_acceptance',
    is_test: true,
  })
})
