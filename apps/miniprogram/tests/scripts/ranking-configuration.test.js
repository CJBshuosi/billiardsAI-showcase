const fs = require('fs')
const path = require('path')

const {
  validateRankingConfiguration,
} = require('../../scripts/validate-ranking-configuration')

const root = path.resolve(__dirname, '../..')


test('checked-in example stays closed and contains no invented venue, table or season', () => {
  const configuration = JSON.parse(fs.readFileSync(
    path.join(root, 'database/ranking-configuration.example.json'), 'utf8',
  ))

  expect(validateRankingConfiguration(configuration)).toEqual({
    ranking_scope_id: 'zhoukou-urban', opening_ready: false,
    season_count: 0, venue_count: 0, table_count: 0,
  })
})


test('opening rejects missing operational evidence instead of publishing an empty board', () => {
  expect(() => validateRankingConfiguration({
    schema_version: '1.0',
    scope: {
      ranking_scope_id: 'zhoukou-urban', display_name: '周口市区',
      status: 'approved', publication_status: 'open', configuration_version: 1,
    },
    seasons: [], venues: [], tables: [],
  })).toThrow('RANKING_CONFIGURATION_OPENING_INCOMPLETE')
})


test('an opening configuration requires one exact 60-day active season and device-bound tables', () => {
  const configuration = {
    schema_version: '1.0',
    scope: {
      ranking_scope_id: 'zhoukou-urban', display_name: '周口市区',
      status: 'approved', publication_status: 'open', configuration_version: 3,
    },
    seasons: [{
      season_id: 'zhoukou-s1', ranking_scope_id: 'zhoukou-urban', status: 'active',
      starts_at: '2026-09-01T16:00:00.000Z', ends_at: '2026-10-31T16:00:00.000Z',
    }],
    venues: [{
      venue_id: 'venue-approved', ranking_scope_id: 'zhoukou-urban',
      display_name: '已现场批准球房', status: 'approved',
    }],
    tables: [{
      table_id: 'table-approved', venue_id: 'venue-approved', status: 'approved',
      device_deployment_status: 'approved', device_id: 'device-field-1',
    }],
  }

  expect(validateRankingConfiguration(configuration)).toMatchObject({
    opening_ready: true, season_count: 1, venue_count: 1, table_count: 1,
  })
  configuration.seasons[0].ends_at = '2026-10-30T16:00:00.000Z'
  expect(() => validateRankingConfiguration(configuration))
    .toThrow('RANKING_CONFIGURATION_SEASON_INVALID')
})
