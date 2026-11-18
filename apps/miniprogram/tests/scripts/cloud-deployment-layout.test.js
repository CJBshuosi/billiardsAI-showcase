const fs = require('fs')
const path = require('path')


const root = path.resolve(__dirname, '../..')


test('CloudBase deploys all handlers from the shared cloud-function source unit', () => {
  const config = JSON.parse(fs.readFileSync(path.join(root, 'cloudbaserc.json'), 'utf8'))
  const functions = new Map(config.functions.map(item => [item.name, item]))

  expect([...functions.keys()]).toEqual([
    'client-api', 'sync-api', 'ranking-worker', 'media-retention', 'shot-projection-worker',
  ])
  expect(functions.get('client-api')).toMatchObject({
    dir: './cloudfunctions',
    handler: 'index.main',
    envVariables: {
      BILLIARDS_FUNCTION_ENTRY: 'client-api',
      RANKING_WORKER_TOKEN: '{{env.RANKING_WORKER_TOKEN}}',
    },
  })
  expect(functions.get('sync-api')).toMatchObject({
    dir: './cloudfunctions',
    handler: 'index.main',
    envVariables: {
      BILLIARDS_FUNCTION_ENTRY: 'sync-api',
      RANKING_RUNTIME_ENVIRONMENT: 'development',
    },
  })
  expect(functions.get('ranking-worker')).toMatchObject({
    dir: './cloudfunctions',
    handler: 'index.main',
    envVariables: {
      BILLIARDS_FUNCTION_ENTRY: 'ranking-worker',
      RANKING_WORKER_TOKEN: '{{env.RANKING_WORKER_TOKEN}}',
    },
  })
  expect(functions.get('media-retention')).toMatchObject({
    dir: './cloudfunctions',
    handler: 'index.main',
    envVariables: {
      BILLIARDS_FUNCTION_ENTRY: 'media-retention',
    },
  })
  expect(functions.get('shot-projection-worker')).toMatchObject({
    dir: './cloudfunctions', handler: 'index.main', installDependency: true,
    envVariables: { BILLIARDS_FUNCTION_ENTRY: 'shot-projection-worker',
      SHOT_PROJECTION_WORKER_TOKEN: '{{env.SHOT_PROJECTION_WORKER_TOKEN}}' },
  })
  expect(fs.existsSync(path.join(root, 'cloudfunctions/index.js'))).toBe(true)
  expect(fs.existsSync(path.join(root, 'cloudfunctions/package-lock.json'))).toBe(true)
  expect(fs.existsSync(path.join(root, 'cloudfunctions/shared/managed-command-authority.js'))).toBe(true)
  for (const name of functions.keys()) {
    expect(fs.existsSync(path.join(root, 'cloudfunctions', name, 'package.json'))).toBe(false)
  }
})

test('projection worker has no active scheduling trigger', () => {
  const worker = JSON.parse(fs.readFileSync(
    path.join(root, 'cloudfunctions/shot-projection-worker/config.json'), 'utf8',
  ))
  expect(worker).toEqual({ permissions: { openapi: [] }, triggers: [] })
})

test.each(['missing-function', 'entry', 'token', 'trigger', 'collection', 'index'])(
  'source verifier fails closed for invalid projection deployment %s', kind => {
    const read = fs.readFileSync.bind(fs)
    jest.spyOn(fs, 'readFileSync').mockImplementation((file, ...args) => {
      const content = read(file, ...args)
      if (String(file) === path.join(root, 'cloudbaserc.json')) {
        const config = JSON.parse(content)
        if (kind === 'missing-function') config.functions = config.functions.filter(item => item.name !== 'shot-projection-worker')
        const worker = config.functions.find(item => item.name === 'shot-projection-worker')
        if (kind === 'entry' && worker) worker.envVariables.BILLIARDS_FUNCTION_ENTRY = 'sync-api'
        if (kind === 'token' && worker) worker.envVariables.SHOT_PROJECTION_WORKER_TOKEN = 'literal-credential'
        return JSON.stringify(config)
      }
      if (kind === 'trigger' && String(file).endsWith('/shot-projection-worker/config.json')) {
        return JSON.stringify({ permissions: { openapi: [] }, triggers: [{ name: 'active' }] })
      }
      if (kind === 'collection' && String(file).endsWith('/database/collections.json')) {
        const manifest = JSON.parse(content)
        manifest.collections = manifest.collections.filter(name => name !== 'shot_projection_jobs')
        return JSON.stringify(manifest)
      }
      if (kind === 'index' && String(file).endsWith('/database/indexes.json')) {
        const manifest = JSON.parse(content)
        manifest.indexes = manifest.indexes.filter(index => index.collection !== 'shot_projection_jobs')
        return JSON.stringify(manifest)
      }
      return content
    })
    const output = jest.spyOn(process.stdout, 'write').mockImplementation(() => true)
    try {
      expect(() => require('../../scripts/verify-development').main(['--environment', 'development'])).toThrow()
      expect(output).not.toHaveBeenCalled()
    } finally { jest.restoreAllMocks() }
  },
)

test('source verifier accepts the dormant projection worker package without cloud calls', () => {
  const output = jest.spyOn(process.stdout, 'write').mockImplementation(() => true)
  try {
    expect(() => require('../../scripts/verify-development').main(['--environment', 'development'])).not.toThrow()
    expect(output).toHaveBeenCalledWith('PASS: development deployment manifest verified\n')
  } finally { output.mockRestore() }
})


test('retention worker stays dormant unless explicitly invoked in synthetic-only mode', () => {
  const worker = JSON.parse(fs.readFileSync(
    path.join(root, 'cloudfunctions/media-retention/config.json'), 'utf8',
  ))
  expect(worker.triggers).toEqual([])
  expect(process.env.MEDIA_RETENTION_EXECUTION_ENABLED).not.toBe('synthetic-only')
})


test('retention queries have declared database index coverage', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))
  const signatures = new Set(manifest.indexes.map(item => JSON.stringify([
    item.collection, item.fields,
  ])))
  for (const signature of [
    ['clips', [['source_kind', 'asc'], ['upload_status', 'asc'], ['retention_state', 'asc']]],
    ['clip_favorites', [['clip_id', 'asc']]],
    ['clip_evidence_holds', [['clip_id', 'asc']]],
    ['media_deletion_jobs', [
      ['device_id', 'asc'], ['table_id', 'asc'], ['status', 'asc'], ['created_at', 'asc'],
    ]],
  ]) expect(signatures).toContain(JSON.stringify(signature))
})

const managedIndexes = [
  ['managed_session_births', [['session_id', 'asc']]],
  ['managed_session_births', [['birth_id', 'asc']]],
  ['managed_session_open_requests', [['device_id', 'asc'], ['open_request_id', 'asc']]],
  ['session_command_authority', [['session_id', 'asc']]],
  ['session_role_binding_history', [['session_id', 'asc'], ['role', 'asc'], ['resulting_revision', 'desc']]],
]
test.each([...new Set(managedIndexes.map(([collection]) => collection))])(
  'source verifier refuses missing managed collection %s', collection => {
    const read = fs.readFileSync.bind(fs)
    jest.spyOn(fs, 'readFileSync').mockImplementation((file, ...args) => {
      const content = read(file, ...args)
      if (!String(file).endsWith('/database/collections.json')) return content
      const manifest = JSON.parse(content)
      manifest.collections = manifest.collections.filter(name => name !== collection)
      return JSON.stringify(manifest)
    })
    const output = jest.spyOn(process.stdout, 'write').mockImplementation(() => true)
    try {
      expect(() => require('../../scripts/verify-development').main(['--environment', 'development']))
        .toThrow(`missing database collection: ${collection}`)
      expect(output).not.toHaveBeenCalled()
    } finally { jest.restoreAllMocks() }
  },
)
test.each(managedIndexes.flatMap(([collection, fields]) => ['missing', 'nonunique'].map(kind => [collection, fields, kind])))(
  'source verifier refuses managed index %s/%j (%s)', (collection, fields, kind) => {
    const read = fs.readFileSync.bind(fs)
    jest.spyOn(fs, 'readFileSync').mockImplementation((file, ...args) => {
      const content = read(file, ...args)
      if (!String(file).endsWith('/database/indexes.json')) return content
      const manifest = JSON.parse(content)
      manifest.indexes = manifest.indexes.flatMap(index => index.collection === collection
        && JSON.stringify(index.fields) === JSON.stringify(fields)
        ? (kind === 'missing' ? [] : [{ ...index, unique: false }]) : [index])
      return JSON.stringify(manifest)
    })
    const output = jest.spyOn(process.stdout, 'write').mockImplementation(() => true)
    try {
      expect(() => require('../../scripts/verify-development').main(['--environment', 'development']))
        .toThrow(`missing managed authority index: ${collection}`)
      expect(output).not.toHaveBeenCalled()
    } finally { jest.restoreAllMocks() }
  },
)
test('source verifier refuses missing packaged managed helper', () => {
  const exists = fs.existsSync.bind(fs)
  jest.spyOn(fs, 'existsSync').mockImplementation(file =>
    String(file).endsWith('/shared/managed-command-authority.js') ? false : exists(file))
  const output = jest.spyOn(process.stdout, 'write').mockImplementation(() => true)
  try {
    expect(() => require('../../scripts/verify-development').main(['--environment', 'development']))
      .toThrow('managed command authority helper is missing')
    expect(output).not.toHaveBeenCalled()
  } finally { jest.restoreAllMocks() }
})
