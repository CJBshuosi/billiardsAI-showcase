const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const API_VERSION = '2018-06-08'
const MISSING_COLLECTION = /(?:TableNotExist|not\s+exists?|ResourceNotFound|不存在|-502005)/i

function loadCollectionManifest(root) {
  return JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
}

function apiArgs(action, { tag, environmentId }, collection) {
  const body = { TableName: collection }
  if (environmentId) body.EnvId = environmentId
  else body.Tag = tag
  if (action === 'CreateTable' && environmentId) {
    body.PermissionInfo = { AclTag: 'ADMINONLY', EnvId: environmentId }
  }
  return [
    'api',
    'tcb',
    action,
    '--api-version',
    API_VERSION,
    '--body',
    JSON.stringify(body),
    '--json',
  ]
}

function defaultRun(command, args) {
  return spawnSync(command, args, { encoding: 'utf8' })
}

function commandError(result) {
  return [result.error && result.error.message, result.stderr, result.stdout]
    .filter(Boolean)
    .join('\n')
    .trim()
}

function ensureCollections({ tag, environmentId, collections, run = defaultRun }) {
  if (!tag && !environmentId) throw new Error('database scope is required')
  const existing = []
  const created = []
  for (const collection of collections) {
    const inspected = run('tcb', apiArgs('DescribeTable', { tag, environmentId }, collection))
    if (inspected.status === 0) {
      existing.push(collection)
      continue
    }
    const inspectError = commandError(inspected)
    if (!MISSING_COLLECTION.test(inspectError)) {
      throw new Error(`cannot inspect collection ${collection}: ${inspectError || 'unknown tcb error'}`)
    }
    const creation = run('tcb', apiArgs('CreateTable', { tag, environmentId }, collection))
    if (creation.status !== 0) {
      throw new Error(`cannot create collection ${collection}: ${commandError(creation) || 'unknown tcb error'}`)
    }
    created.push(collection)
  }
  return { existing, created }
}

function argumentValue(argv, name) {
  const index = argv.indexOf(name)
  return index >= 0 ? argv[index + 1] : undefined
}

function main(argv) {
  if (argumentValue(argv, '--environment') !== 'development') {
    throw new Error('database initialization requires --environment development')
  }
  const tag = argumentValue(argv, '--database-tag')
  const environmentId = argumentValue(argv, '--environment-id')
  if (tag && !/^tnt-[a-z0-9]+$/i.test(tag)) {
    throw new Error('database initialization requires a valid --database-tag tnt-...')
  }
  if (environmentId && !/^[a-z0-9][a-z0-9-]{2,127}$/i.test(environmentId)) {
    throw new Error('database initialization requires a valid --environment-id')
  }
  if (!tag && !environmentId) {
    throw new Error('database initialization requires --environment-id or --database-tag')
  }
  const root = path.resolve(__dirname, '..')
  const manifest = loadCollectionManifest(root)
  const result = ensureCollections({ tag, environmentId, collections: manifest.collections })
  process.stdout.write(`PASS: database collections ready; created=${result.created.length}; existing=${result.existing.length}\n`)
}

if (require.main === module) {
  try {
    main(process.argv.slice(2))
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  }
}

module.exports = { ensureCollections, loadCollectionManifest, main }
