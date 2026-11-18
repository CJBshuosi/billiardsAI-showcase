const crypto = require('crypto')
const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')


const API_VERSION = '2018-06-08'


function direction(value) {
  if (value === 'asc' || value === '1' || value === 1) return '1'
  if (value === 'desc' || value === '-1' || value === -1) return '-1'
  throw new Error(`unsupported index direction: ${value}`)
}


function signature(fields, unique) {
  return JSON.stringify({
    fields: fields.map(([name, order]) => [name, direction(order)]),
    unique: Boolean(unique),
  })
}


function createIndexDefinition(index) {
  const digest = crypto.createHash('sha256')
    .update(signature(index.fields, index.unique), 'utf8').digest('hex').slice(0, 16)
  return {
    IndexName: `idx_${digest}`,
    MgoKeySchema: {
      MgoIndexKeys: index.fields.map(([name, order]) => ({
        Name: name, Direction: direction(order),
      })),
      MgoIsUnique: Boolean(index.unique),
      MgoIsSparse: false,
    },
  }
}


function existingSignature(index) {
  return signature(
    (index.Keys || []).map(key => [key.Name, key.Direction]),
    index.Unique,
  )
}


function missingIndexDefinitions(desired, existing) {
  const signatures = new Set(existing.map(existingSignature))
  return desired
    .filter(index => !signatures.has(signature(index.fields, index.unique)))
    .map(createIndexDefinition)
}


function apiArgs(action, body) {
  return [
    'api', 'tcb', action, '--api-version', API_VERSION,
    '--body', JSON.stringify(body), '--json',
  ]
}


function parsePayload(result, operation) {
  if (!result || result.status !== 0) {
    const detail = [result && result.stderr, result && result.stdout]
      .filter(Boolean).join('\n').trim()
    throw new Error(`${operation}: ${detail || 'unknown tcb error'}`)
  }
  const raw = result.stdout || ''
  const starts = [raw.indexOf('{'), raw.indexOf('[')].filter(value => value >= 0)
  if (starts.length === 0) throw new Error(`${operation}: invalid JSON response`)
  return JSON.parse(raw.slice(Math.min(...starts)))
}


function defaultRun(command, args) {
  return spawnSync(command, args, { encoding: 'utf8' })
}


function defaultWait(milliseconds) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds)
}


function transientUpdateFailure(result) {
  const detail = [result && result.stderr, result && result.stdout].filter(Boolean).join('\n')
  return /FailedOperation\.Timeout|ResourceInUse\.IndexCreating|temporarily busy/i.test(detail)
}


function ensureIndexes({ environmentId, indexes, run = defaultRun, wait = defaultWait }) {
  const groups = new Map()
  for (const index of indexes) {
    if (!groups.has(index.collection)) groups.set(index.collection, [])
    groups.get(index.collection).push(index)
  }
  let added = 0
  let existing = 0
  for (const [collection, desired] of groups) {
    let transientAttempts = 0
    while (true) {
      const described = parsePayload(run('tcb', apiArgs('DescribeTable', {
        EnvId: environmentId, TableName: collection,
      })), `cannot inspect indexes for ${collection}`)
      const cloudIndexes = described.data && described.data.Indexes
        || described.Response && described.Response.Indexes || []
      const missing = missingIndexDefinitions(desired, cloudIndexes)
      if (missing.length === 0) {
        existing += desired.length
        break
      }
      const update = run('tcb', apiArgs('UpdateTable', {
        EnvId: environmentId, TableName: collection, CreateIndexes: missing,
      }))
      if (update && update.status === 0) {
        parsePayload(update, `cannot create indexes for ${collection}`)
        existing += desired.length - missing.length
        added += missing.length
        break
      }
      if (!transientUpdateFailure(update) || transientAttempts >= 3) {
        parsePayload(update, `cannot create indexes for ${collection}`)
      }
      transientAttempts += 1
      wait(3000)
    }
  }
  return { added, existing }
}


function argumentValue(argv, name) {
  const index = argv.indexOf(name)
  return index >= 0 ? argv[index + 1] : undefined
}


function main(argv) {
  if (argumentValue(argv, '--environment') !== 'development') {
    throw new Error('index initialization requires --environment development')
  }
  const environmentId = argumentValue(argv, '--environment-id')
  if (!environmentId || !/^[a-z0-9][a-z0-9-]{2,127}$/i.test(environmentId)) {
    throw new Error('index initialization requires a valid --environment-id')
  }
  const root = path.resolve(__dirname, '..')
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))
  const result = ensureIndexes({ environmentId, indexes: manifest.indexes })
  process.stdout.write(`PASS: database indexes ready; added=${result.added}; existing=${result.existing}\n`)
}


if (require.main === module) {
  try {
    main(process.argv.slice(2))
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  }
}


module.exports = { createIndexDefinition, ensureIndexes, main, missingIndexDefinitions }
