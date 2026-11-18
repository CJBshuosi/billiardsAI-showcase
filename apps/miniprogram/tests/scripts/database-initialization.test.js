const fs = require('fs')
const path = require('path')

const {
  ensureCollections,
  loadCollectionManifest,
} = require('../../scripts/initialize-development-database')

const root = path.resolve(__dirname, '../..')

function sourceCollectionNames() {
  const functionRoot = path.join(root, 'cloudfunctions')
  const files = []
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      // SDK dependencies/examples do not define this application's collections.
      if (entry.name === 'node_modules') continue
      const fullPath = path.join(directory, entry.name)
      if (entry.isDirectory()) visit(fullPath)
      else if (entry.name.endsWith('.js')) files.push(fullPath)
    }
  }
  visit(functionRoot)
  const names = new Set()
  for (const file of files) {
    const source = fs.readFileSync(file, 'utf8')
    for (const match of source.matchAll(/\.collection\s*\(\s*['"]([^'"]+)['"]/g)) names.add(match[1])
    for (const match of source.matchAll(/\bcollection\s*\(\s*transaction\s*,\s*['"]([^'"]+)['"]/g)) names.add(match[1])
  }
  return [...names].sort()
}

test('database collection manifest covers every cloud function collection', () => {
  const manifest = loadCollectionManifest(root)
  expect(manifest.collections).toEqual(expect.arrayContaining(sourceCollectionNames()))
})

test('database manifests reserve append-only match facts with unique identities and ordering', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))

  expect(collections.collections).toEqual(expect.arrayContaining(['matches', 'match_events']))
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    { collection: 'matches', fields: [['session_id', 'asc']], unique: true },
    { collection: 'match_events', fields: [['event_id', 'asc']], unique: true },
    {
      collection: 'match_events',
      fields: [['match_id', 'asc'], ['sequence', 'asc']],
      unique: true,
    },
  ]))
})

test('database manifests reserve immutable rating ledger generations and player states', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))

  expect(collections.collections).toEqual(expect.arrayContaining([
    'rating_scope_heads', 'rating_generations', 'rating_events',
    'rating_match_consumptions', 'player_skill_states', 'ranking_settlement_jobs',
    'ranking_test_resets',
  ]))
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    {
      collection: 'rating_events',
      fields: [['generation_id', 'asc'], ['scope_sequence', 'asc']],
      unique: true,
    },
    {
      collection: 'rating_events',
      fields: [['generation_id', 'asc'], ['match_id', 'asc']],
      unique: true,
    },
    {
      collection: 'rating_match_consumptions',
      fields: [['ranking_scope_id', 'asc'], ['match_id', 'asc']],
      unique: true,
    },
    {
      collection: 'player_skill_states',
      fields: [['ranking_scope_id', 'asc'], ['user_id', 'asc']],
      unique: true,
    },
    {
      collection: 'ranking_settlement_jobs',
      fields: [['ranking_scope_id', 'asc'], ['status', 'asc'], ['created_at', 'asc']],
      unique: false,
    },
  ]))
})

test('database manifests reserve risk review, reports and replay requests', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))

  expect(collections.collections).toEqual(expect.arrayContaining([
    'match_risk_assessments', 'match_review_events', 'rating_replay_requests', 'user_match_reports',
  ]))
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    {
      collection: 'match_review_events',
      fields: [['match_id', 'asc'], ['sequence', 'asc']],
      unique: true,
    },
    {
      collection: 'rating_replay_requests',
      fields: [['ranking_scope_id', 'asc'], ['status', 'asc'], ['from_scope_sequence', 'asc']],
      unique: false,
    },
    {
      collection: 'user_match_reports',
      fields: [['match_id', 'asc'], ['reporter_user_id', 'asc']],
      unique: true,
    },
  ]))
})

test('database manifests reserve public profiles, unique nicknames, seasons and snapshot pages', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))
  expect(collections.collections).toEqual(expect.arrayContaining([
    'public_ranking_profiles', 'public_nickname_claims', 'ranking_seasons',
    'ranking_snapshots', 'ranking_snapshot_entries', 'ranking_snapshot_heads',
  ]))
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    { collection: 'public_nickname_claims', fields: [['nickname_key', 'asc']], unique: true },
    { collection: 'public_ranking_profiles', fields: [['user_id', 'asc']], unique: true },
    {
      collection: 'ranking_snapshot_entries',
      fields: [['snapshot_id', 'asc'], ['position', 'asc']],
      unique: true,
    },
    {
      collection: 'ranking_snapshots',
      fields: [['ranking_scope_id', 'asc'], ['kind', 'asc'], ['status', 'asc'], ['as_of', 'desc']],
      unique: false,
    },
    {
      collection: 'ranking_snapshot_entries',
      fields: [['snapshot_id', 'asc'], ['user_id', 'asc']],
      unique: true,
    },
  ]))
})

test('database manifests reserve challenges, pair slots and idempotent requests', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))
  expect(collections.collections).toEqual(expect.arrayContaining([
    'challenges', 'challenge_pair_slots', 'challenge_request_keys',
  ]))
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    { collection: 'challenges', fields: [['challenge_id', 'asc']], unique: true },
    {
      collection: 'challenges',
      fields: [['target_user_id', 'asc'], ['status', 'asc'], ['expires_at', 'asc']],
      unique: false,
    },
    { collection: 'challenge_pair_slots', fields: [['pair_key', 'asc']], unique: true },
    {
      collection: 'challenge_request_keys',
      fields: [['challenger_user_id', 'asc'], ['idempotency_key_digest', 'asc']],
      unique: true,
    },
  ]))
})

test('database manifests reserve ranking authority and immutable pipeline decisions', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))
  expect(collections.collections).toEqual(expect.arrayContaining([
    'ranking_scopes', 'ranking_venues', 'ranking_tables', 'match_qualification_decisions',
  ]))
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    { collection: 'ranking_scopes', fields: [['ranking_scope_id', 'asc']], unique: true },
    { collection: 'ranking_venues', fields: [['venue_id', 'asc']], unique: true },
    { collection: 'ranking_tables', fields: [['table_id', 'asc']], unique: true },
    {
      collection: 'match_qualification_decisions',
      fields: [['match_id', 'asc'], ['result_revision', 'asc']],
      unique: true,
    },
    {
      collection: 'match_risk_assessments',
      fields: [['match_id', 'asc'], ['result_revision', 'asc']],
      unique: true,
    },
    { collection: 'matches', fields: [['red_user_id', 'asc'], ['status', 'asc']], unique: false },
    { collection: 'matches', fields: [['blue_user_id', 'asc'], ['status', 'asc']], unique: false },
    { collection: 'matches', fields: [['red_user_id', 'asc'], ['started_at', 'desc']], unique: false },
    { collection: 'matches', fields: [['blue_user_id', 'asc'], ['started_at', 'desc']], unique: false },
    { collection: 'matches', fields: [['table_id', 'asc'], ['started_at', 'desc']], unique: false },
  ]))
})

test('database initialization creates only missing collections', () => {
  const calls = []
  const run = (_command, args) => {
    calls.push(args)
    const action = args[2]
    const body = JSON.parse(args[args.indexOf('--body') + 1])
    if (action === 'DescribeTable' && body.TableName === 'users') {
      return { status: 0, stdout: '{}', stderr: '' }
    }
    if (action === 'DescribeTable') {
      return { status: 1, stdout: '', stderr: 'TableNotExist' }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  const result = ensureCollections({
    environmentId: 'development-env',
    collections: ['users', 'wechat_identities'],
    run,
  })

  expect(result).toEqual({ existing: ['users'], created: ['wechat_identities'] })
  expect(calls.filter((args) => args[2] === 'CreateTable')).toHaveLength(1)
  expect(calls.every((args) => args[1] === 'tcb')).toBe(true)
  expect(calls.every((args) => args.includes('2018-06-08'))).toBe(true)
  const createArgs = calls.find(args => args[2] === 'CreateTable')
  expect(JSON.parse(createArgs[createArgs.indexOf('--body') + 1])).toEqual({
    EnvId: 'development-env',
    TableName: 'wechat_identities',
    PermissionInfo: { AclTag: 'ADMINONLY', EnvId: 'development-env' },
  })
})

test('database initialization stops on an unrelated describe failure', () => {
  expect(() => ensureCollections({
    environmentId: 'development-env',
    collections: ['users'],
    run: () => ({ status: 1, stdout: '', stderr: 'network timeout' }),
  })).toThrow('cannot inspect collection users: network timeout')
})

test('database manifests reserve immutable start ownership and owner-filtered clip queries', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))
  expect(collections.collections).toEqual(expect.arrayContaining(['session_role_binding_history', 'field_shot_ownership']))
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    { collection: 'session_role_binding_history', fields: [['session_id', 'asc'], ['role', 'asc'], ['resulting_revision', 'desc']], unique: true },
    { collection: 'field_shot_ownership', fields: [['session_id', 'asc'], ['shot_id', 'asc']], unique: true },
    { collection: 'clips', fields: [['owner_user_id', 'asc'], ['published_at', 'desc'], ['clip_id', 'desc']], unique: false },
    { collection: 'clips', fields: [['owner_user_id', 'asc'], ['session_id', 'asc'], ['published_at', 'desc'], ['clip_id', 'desc']], unique: false },
    { collection: 'clips', fields: [['owner_user_id', 'asc'], ['sequence', 'asc']], unique: false },
    { collection: 'clips', fields: [['owner_user_id', 'asc'], ['session_id', 'asc'], ['sequence', 'asc']], unique: false },
  ]))
})

test('database manifests reserve private media authority, contiguous operations and opaque aliases', () => {
  const collections = loadCollectionManifest(root)
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8')).indexes
  expect(collections.collections).toEqual(expect.arrayContaining(['clip_storage_authorities', 'media_storage_operations', 'media_storage_objects']))
  expect(indexes).toEqual(expect.arrayContaining([
    { collection: 'clip_storage_authorities', fields: [['identity.clip_id', 'asc']], unique: true },
    { collection: 'media_storage_operations', fields: [['clip_id', 'asc'], ['page_index', 'asc']], unique: true },
    { collection: 'media_storage_objects', fields: [['record_kind', 'asc'], ['cloud_path', 'asc']], unique: true },
    { collection: 'saved_clip_sync_jobs', fields: [['clip_id', 'asc']], unique: false },
  ]))
  expect(JSON.parse(fs.readFileSync(path.join(root, 'database/security-rules.json'), 'utf8'))).toEqual({ read: false, write: false })
})

test('database manifests reserve permanent private managed birth, request and initial command authority', () => {
  const collections = loadCollectionManifest(root)
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8')).indexes
  expect(collections.collections).toEqual(expect.arrayContaining([
    'managed_session_births', 'managed_session_open_requests', 'session_command_authority',
  ]))
  expect(indexes).toEqual(expect.arrayContaining([
    { collection: 'managed_session_births', fields: [['session_id', 'asc']], unique: true },
    { collection: 'managed_session_births', fields: [['birth_id', 'asc']], unique: true },
    { collection: 'managed_session_open_requests', fields: [['device_id', 'asc'], ['open_request_id', 'asc']], unique: true },
    { collection: 'session_command_authority', fields: [['session_id', 'asc']], unique: true },
  ]))
  expect(JSON.parse(fs.readFileSync(path.join(root, 'database/security-rules.json'), 'utf8'))).toEqual({ read: false, write: false })
})

test('shot projection jobs remain private with unique identity and deterministic pending discovery', () => {
  const collections = JSON.parse(fs.readFileSync(path.join(root, 'database/collections.json'), 'utf8'))
  const indexes = JSON.parse(fs.readFileSync(path.join(root, 'database/indexes.json'), 'utf8'))
  expect(collections.collections).toContain('shot_projection_jobs')
  expect(indexes.indexes).toEqual(expect.arrayContaining([
    { collection: 'shot_projection_jobs', fields: [['session_id', 'asc']], unique: true },
    { collection: 'shot_projection_jobs', fields: [['job_id', 'asc']], unique: true },
    { collection: 'shot_projection_jobs', fields: [['status', 'asc'], ['updated_at', 'asc'], ['job_id', 'asc']], unique: false },
  ]))
  const rules = JSON.parse(fs.readFileSync(path.join(root, 'database/security-rules.json'), 'utf8'))
  expect(rules).toEqual({ read: false, write: false })
})
