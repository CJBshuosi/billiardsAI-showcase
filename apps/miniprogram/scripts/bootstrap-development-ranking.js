const crypto = require('crypto')
const fs = require('fs')
const { spawnSync } = require('child_process')

const { ALGORITHM_VERSION } = require('../cloudfunctions/shared/rating-engine')
const {
  validateDevelopmentRankingConfiguration,
} = require('./validate-development-ranking-configuration')


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


function buildBootstrapPlan(configuration) {
  const summary = validateDevelopmentRankingConfiguration(configuration)
  if (!summary.opening_ready) throw new Error('RANKING_TEST_CONFIGURATION_INVALID')
  const scope = structuredClone(configuration.scope)
  const season = structuredClone(configuration.seasons[0])
  const generationId = digest(`${scope.ranking_scope_id}:${ALGORITHM_VERSION}:generation-1`)
  const operation = (collection, logicalId, data, writeMode = 'upsert') => ({
    collection, document_id: digest(logicalId), write_mode: writeMode,
    data: { schema_version: '1.0', ...structuredClone(data), is_test: true },
  })
  return [
    operation('ranking_scopes', scope.ranking_scope_id, scope),
    operation('ranking_seasons', season.season_id, season),
    ...configuration.venues.map(venue => operation('ranking_venues', venue.venue_id, venue)),
    ...configuration.tables.map(table => operation('ranking_tables', table.table_id, table)),
    operation('rating_generations', generationId, {
      generation_id: generationId, ranking_scope_id: scope.ranking_scope_id,
      algorithm_version: ALGORITHM_VERSION, status: 'active', created_at: season.starts_at,
    }, 'insert_if_absent'),
    operation('rating_scope_heads', scope.ranking_scope_id, {
      ranking_scope_id: scope.ranking_scope_id, active_generation_id: generationId,
      next_sequence: 1, updated_at: season.starts_at,
    }, 'insert_if_absent'),
  ]
}


function commandFor(operation) {
  const data = { ...operation.data }
  const identity = { _id: operation.document_id }
  for (const field of ['ranking_scope_id', 'mode', 'is_test', 'season_id', 'venue_id', 'table_id']) {
    if (data[field] != null) identity[field] = data[field]
  }
  const update = operation.write_mode === 'insert_if_absent'
    ? { $setOnInsert: { _id: operation.document_id, ...data } }
    : { $set: data, $setOnInsert: { _id: operation.document_id } }
  return {
    TableName: operation.collection,
    CommandType: 'UPDATE',
    Command: JSON.stringify({
      update: operation.collection,
      updates: [{
        q: identity,
        u: update,
        upsert: true,
      }],
    }),
  }
}


function executeBootstrapPlan({ environmentId, plan, run = spawnSync }) {
  const result = run('tcb', [
    '-e', environmentId, 'db', 'nosql', 'execute',
    '--command', JSON.stringify(plan.map(commandFor)), '--json',
  ], { encoding: 'utf8' })
  if (!result || result.status !== 0) {
    const detail = [result && result.stderr, result && result.stdout].filter(Boolean).join('\n')
    if (/duplicate|E11000|conflict/iu.test(detail)) {
      throw new Error('RANKING_TEST_CONFIGURATION_CONFLICT')
    }
    throw new Error(`RANKING_TEST_BOOTSTRAP_FAILED: ${detail.trim() || 'unknown tcb error'}`)
  }
  return { operation_count: plan.length }
}


function main(argv) {
  const file = argv[0]
  if (!file) throw new Error('RANKING_CONFIGURATION_FILE_REQUIRED')
  const configuration = JSON.parse(fs.readFileSync(file, 'utf8'))
  const plan = buildBootstrapPlan(configuration)
  const result = executeBootstrapPlan({ environmentId: configuration.environment_id, plan })
  process.stdout.write(`PASS: development ranking ready; operations=${result.operation_count}\n`)
}


if (require.main === module) {
  try {
    main(process.argv.slice(2))
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  }
}


module.exports = { buildBootstrapPlan, commandFor, executeBootstrapPlan, main }
