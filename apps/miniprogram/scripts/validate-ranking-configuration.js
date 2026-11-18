const fs = require('fs')


const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u
const SIXTY_DAYS_MS = 60 * 24 * 60 * 60 * 1000


function fail(code) {
  throw new Error(code)
}


function validId(value) {
  return typeof value === 'string' && ID_PATTERN.test(value)
}


function validateRankingConfiguration(configuration) {
  if (!configuration || configuration.schema_version !== '1.0'
      || !configuration.scope || !Array.isArray(configuration.seasons)
      || !Array.isArray(configuration.venues) || !Array.isArray(configuration.tables)) {
    fail('RANKING_CONFIGURATION_INVALID')
  }
  const scope = configuration.scope
  if (!validId(scope.ranking_scope_id) || typeof scope.display_name !== 'string'
      || !['disabled', 'approved'].includes(scope.status)
      || !['closed', 'open'].includes(scope.publication_status)
      || !Number.isInteger(scope.configuration_version) || scope.configuration_version < 1) {
    fail('RANKING_CONFIGURATION_INVALID')
  }
  const summary = {
    ranking_scope_id: scope.ranking_scope_id,
    opening_ready: scope.status === 'approved' && scope.publication_status === 'open',
    season_count: configuration.seasons.length,
    venue_count: configuration.venues.length,
    table_count: configuration.tables.length,
  }
  if (!summary.opening_ready) return summary

  const activeSeasons = configuration.seasons.filter(season => season.status === 'active')
  const approvedVenues = configuration.venues.filter(venue => venue.status === 'approved')
  const approvedTables = configuration.tables.filter(table => table.status === 'approved')
  if (activeSeasons.length !== 1 || approvedVenues.length === 0 || approvedTables.length === 0) {
    fail('RANKING_CONFIGURATION_OPENING_INCOMPLETE')
  }
  const season = activeSeasons[0]
  const startsAt = Date.parse(season.starts_at)
  const endsAt = Date.parse(season.ends_at)
  if (!validId(season.season_id) || season.ranking_scope_id !== scope.ranking_scope_id
      || !Number.isFinite(startsAt) || !Number.isFinite(endsAt)
      || endsAt - startsAt !== SIXTY_DAYS_MS) {
    fail('RANKING_CONFIGURATION_SEASON_INVALID')
  }
  const venueIds = new Set()
  for (const venue of approvedVenues) {
    if (!validId(venue.venue_id) || venue.ranking_scope_id !== scope.ranking_scope_id
        || typeof venue.display_name !== 'string' || venue.display_name.trim().length === 0) {
      fail('RANKING_CONFIGURATION_VENUE_INVALID')
    }
    venueIds.add(venue.venue_id)
  }
  for (const table of approvedTables) {
    if (!validId(table.table_id) || !venueIds.has(table.venue_id)
        || table.device_deployment_status !== 'approved' || !validId(table.device_id)) {
      fail('RANKING_CONFIGURATION_TABLE_INVALID')
    }
  }
  return summary
}


if (require.main === module) {
  const file = process.argv[2]
  if (!file) fail('RANKING_CONFIGURATION_FILE_REQUIRED')
  const result = validateRankingConfiguration(JSON.parse(fs.readFileSync(file, 'utf8')))
  process.stdout.write(`${JSON.stringify(result)}\n`)
}


module.exports = { validateRankingConfiguration }
