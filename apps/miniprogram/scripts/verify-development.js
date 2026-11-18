const fs = require('fs')
const path = require('path')


function readJson(root, name) {
  return JSON.parse(fs.readFileSync(path.join(root, name), 'utf8'))
}


function main(argv) {
  const environmentIndex = argv.indexOf('--environment')
  if (environmentIndex < 0 || argv[environmentIndex + 1] !== 'development') {
    throw new Error('verification requires explicit development environment')
  }
  const root = path.resolve(__dirname, '..')
  const config = readJson(root, 'cloudbaserc.json')
  const collectionManifest = readJson(root, 'database/collections.json')
  const indexes = readJson(root, 'database/indexes.json')
  const rules = readJson(root, 'database/security-rules.json')
  const storageRules = readJson(root, 'storage/security-rules.json')
  const names = new Set(config.functions.map((item) => item.name))
  if (!names.has('client-api') || !names.has('sync-api')
      || !names.has('ranking-worker') || !names.has('media-retention')
      || !names.has('shot-projection-worker')) {
    throw new Error('required cloud functions are missing')
  }
  const expectedHandlers = new Map([
    ['client-api', 'index.main'],
    ['sync-api', 'index.main'],
    ['ranking-worker', 'index.main'],
    ['media-retention', 'index.main'],
    ['shot-projection-worker', 'index.main'],
  ])
  for (const item of config.functions) {
    if (item.dir !== './cloudfunctions' || item.handler !== expectedHandlers.get(item.name)) {
      throw new Error(`cloud function deployment layout is invalid: ${item.name}`)
    }
  }
  const projectionWorker = config.functions.find(item => item.name === 'shot-projection-worker')
  if (projectionWorker.envVariables?.BILLIARDS_FUNCTION_ENTRY !== 'shot-projection-worker'
      || projectionWorker.envVariables?.SHOT_PROJECTION_WORKER_TOKEN !== '{{env.SHOT_PROJECTION_WORKER_TOKEN}}'
      || projectionWorker.installDependency !== true) {
    throw new Error('shot projection worker environment or dependency packaging is invalid')
  }
  const workerConfig = readJson(root, 'cloudfunctions/shot-projection-worker/config.json')
  if (!Array.isArray(workerConfig.triggers) || workerConfig.triggers.length !== 0) {
    throw new Error('shot projection worker scheduling activation is not authorized')
  }
  if (!fs.existsSync(path.join(root, 'cloudfunctions/package-lock.json'))) {
    throw new Error('shared cloud function dependency lock is missing')
  }
  if (!fs.existsSync(path.join(root, 'cloudfunctions/shared/managed-command-authority.js'))) {
    throw new Error('managed command authority helper is missing')
  }
  if (rules.read !== false || rules.write !== false) {
    throw new Error('database client access must default to denied')
  }
  if (storageRules.acl !== 'ADMINONLY') {
    throw new Error('storage client access must default to denied')
  }
  const requiredCollections = [
    'clip_evidence_holds', 'clip_favorite_slots', 'clip_favorites', 'clips',
    'field_command_sequences', 'field_live_sessions',
    'field_session_commands', 'session_join_requests',
    'join_rate_limits', 'media_deletion_jobs', 'session_participants', 'session_role_bindings',
    'session_user_bindings', 'sessions', 'shots', 'shot_projection_jobs', 'sync_devices',
    'sync_idempotency', 'sync_jobs', 'sync_nonces', 'table_join_codes', 'users',
    'wechat_identities',
    'managed_session_births', 'managed_session_open_requests',
    'session_command_authority', 'session_role_binding_history',
  ]
  const declaredCollections = new Set(collectionManifest.collections)
  for (const required of requiredCollections) {
    if (!declaredCollections.has(required)) throw new Error(`missing database collection: ${required}`)
  }
  for (const [collection, fields] of [
    ['managed_session_births', [['session_id', 'asc']]],
    ['managed_session_births', [['birth_id', 'asc']]],
    ['managed_session_open_requests', [['device_id', 'asc'], ['open_request_id', 'asc']]],
    ['session_command_authority', [['session_id', 'asc']]],
    ['session_role_binding_history', [['session_id', 'asc'], ['role', 'asc'], ['resulting_revision', 'desc']]],
  ]) {
    if (!indexes.indexes.some(item => item.collection === collection && item.unique === true
        && JSON.stringify(item.fields) === JSON.stringify(fields))) {
      throw new Error(`missing managed authority index: ${collection}`)
    }
  }
  if (!indexes.indexes.some(item => item.collection === 'shot_projection_jobs'
      && item.unique === false
      && JSON.stringify(item.fields) === JSON.stringify([
        ['status', 'asc'], ['updated_at', 'asc'], ['job_id', 'asc'],
      ]))) {
    throw new Error('missing pending shot projection discovery index')
  }
  const collections = new Set(indexes.indexes.map((item) => item.collection))
  for (const required of [
    'session_participants', 'session_role_bindings', 'clip_favorites', 'clip_evidence_holds',
    'clips', 'media_deletion_jobs', 'sync_idempotency', 'table_join_codes',
  ]) {
    if (!collections.has(required)) throw new Error(`missing index coverage: ${required}`)
  }
  process.stdout.write('PASS: development deployment manifest verified\n')
}


if (require.main === module) {
  try {
    main(process.argv.slice(2))
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  }
}


module.exports = { main }
