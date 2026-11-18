const crypto = require('crypto')
const { createJoinAdmin } = require('../../cloudfunctions/sync-api/lib/join-admin')
const { createCloudJoinRepository } = require('../../cloudfunctions/client-api/lib/join')
const { createCloudFieldCommandRepository, submitFieldCommand, publicCommand } = require('../../cloudfunctions/client-api/lib/field-commands')
const { createCloudDeviceFieldCommandRepository } = require('../../cloudfunctions/sync-api/lib/field-commands')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')

const birthId = '12345678-1234-4234-8234-123456789abc'
const sessionId = `managed-${birthId}`
const now = '2026-11-01T10:00:00.000Z'
const hash = value => crypto.createHash('sha256').update(value).digest('hex')
const base = { sessionId, actorUserId: 'user-red', expectedRevision: 0,
  commandType: 'claim_role', payload: { role: 'red' }, idempotencyKey: 'intent-000000000001', now }
async function fixture(dbOptions = {}) {
  const db = ownerDeleteDb({ table_join_codes: { 'table-1': {
    table_id: 'table-1', status: 'active', current_session_id: null,
  } } }, { strictTransactions: true, ...dbOptions })
  const admin = createJoinAdmin(db, { managedSessionsEnabled: true, nowIso: () => now, idFactory: () => birthId })
  await admin.createManagedSession({ tableId: 'table-1', deviceId: 'device-1', openRequestId: birthId })
  db.rows.session_user_bindings = { [hash(`${sessionId}:user-red`)]: {
    schema_version: '1.0', session_id: sessionId, table_id: 'table-1', user_id: 'user-red', role: null, request_id: 'join-1',
  } }
  const repo = createCloudFieldCommandRepository(db)
  return { db, repo, admin }
}
function budgets(db) {
  for (const attempt of db.transactionAttempts.slice(1)) {
    expect(attempt.count).toBeLessThanOrEqual(16)
    expect(attempt.calls.every(call => ['get', 'set', 'update'].includes(call.method))).toBe(true)
  }
}
function roster(db) {
  db.rows.session_role_bindings = Object.fromEntries(['red', 'blue'].map(role => [hash(`${sessionId}:${role}`), {
    schema_version: '1.0', session_id: sessionId, role, user_id: `user-${role}`, request_id: `join-${role}`,
  }]))
  Object.assign(db.rows.field_live_sessions[sessionId], { red_user_id: 'user-red', blue_user_id: 'user-blue' })
  db.rows.session_user_bindings[hash(`${sessionId}:user-red`)].role = 'red'
}

test('managed issue atomically uses positive head without a legacy sequence fallback', async () => {
  const { db, repo } = await fixture()
  const command = await submitFieldCommand(base, repo)
  expect(command.server_sequence).toBe(1)
  expect(db.rows.session_command_authority[sessionId]).toMatchObject({ issued_next: 2, revision: 2 })
  expect(db.rows.field_command_sequences).toBeUndefined()
  expect(db.rows.field_session_commands[command.command_id].submitted_intent_sha256).toMatch(/^[a-f0-9]{64}$/u)
  budgets(db)
})

test('exact raw replay survives closing/ended, changed roster, stale revision and changed enrichment', async () => {
  const { db, repo } = await fixture()
  const command = await submitFieldCommand(base, repo)
  Object.assign(db.rows.session_command_authority[sessionId], { coverage: 'closing', closed_target: 1, revision: 3 })
  Object.assign(db.rows.field_live_sessions[sessionId], { status: 'ended', ended_at: now, control_revision: 5 })
  db.rows.session_user_bindings = {}
  expect(await submitFieldCommand(base, repo)).toEqual(command)
  expect(db.transactionAttempts.at(-1).count).toBeLessThanOrEqual(12)
  await expect(submitFieldCommand({ ...base, payload: { role: 'blue' } }, repo))
    .rejects.toMatchObject({ code: 'COMMAND_IDEMPOTENCY_CONFLICT' })
  budgets(db)
})

test.each(['missing-head', 'missing-birth', 'missing-live', 'missing-digest', 'outside-prefix'])(
  'managed replay fails closed for %s', async mutation => {
    const { db, repo } = await fixture()
    const first = await submitFieldCommand(base, repo)
    if (mutation === 'missing-head') delete db.rows.session_command_authority[sessionId]
    if (mutation === 'missing-birth') delete db.rows.managed_session_births[sessionId]
    if (mutation === 'missing-live') delete db.rows.field_live_sessions[sessionId]
    if (mutation === 'missing-digest') delete db.rows.field_session_commands[first.command_id].submitted_intent_sha256
    if (mutation === 'outside-prefix') db.rows.field_session_commands[first.command_id].server_sequence = 99
    await expect(submitFieldCommand(base, repo)).rejects.toBeDefined()
    expect(Object.keys(db.rows.field_session_commands)).toHaveLength(1)
    budgets(db)
  },
)

test('managed submit requires positive replay capability', async () => {
  const { repo } = await fixture()
  delete repo.getManagedCommandReplay
  await expect(submitFieldCommand(base, repo)).rejects.toMatchObject({ code: 'COMMAND_FORBIDDEN' })
})

test.each(['table', 'actor-binding', 'role-binding', 'revision', 'closing'])(
  'managed issuance rechecks %s inside transaction after discovery', async mutation => {
    const { db, repo } = await fixture()
    const original = repo.createCommand
    repo.createCommand = async (...args) => {
      if (mutation === 'table') db.rows.table_join_codes['table-1'].current_session_id = 'other'
      if (mutation === 'actor-binding') db.rows.session_user_bindings[hash(`${sessionId}:user-red`)].user_id = 'other'
      if (mutation === 'role-binding') {
        db.rows.session_role_bindings = { [hash(`${sessionId}:red`)]: { session_id: 'other', role: 'red', user_id: 'other' } }
      }
      if (mutation === 'revision') db.rows.field_live_sessions[sessionId].control_revision = 1
      if (mutation === 'closing') Object.assign(db.rows.session_command_authority[sessionId], { coverage: 'closing', closed_target: 0, revision: 2 })
      return original(...args)
    }
    await expect(submitFieldCommand(base, repo)).rejects.toBeDefined()
    expect(Object.values(db.rows.field_session_commands || {})).toHaveLength(0)
    expect(db.rows.session_command_authority[sessionId].issued_next).toBe(1)
    budgets(db)
  },
)

test('second transaction replay check returns stored command before changed authorization', async () => {
  const { db, repo } = await fixture()
  const create = repo.createCommand
  let first
  repo.createCommand = async (...args) => {
    first = await create(...args)
    db.rows.table_join_codes['table-1'].status = 'disabled'
    db.rows.session_user_bindings = {}
    return create(...args)
  }
  expect(await submitFieldCommand(base, repo)).toEqual(publicCommand(first))
  expect(Object.keys(db.rows.field_session_commands)).toHaveLength(1)
  budgets(db)
})

test('managed release freezes actual cloud role only in device payload', async () => {
  const { db, repo } = await fixture()
  roster(db)
  const response = await submitFieldCommand({ ...base, commandType: 'release_role', payload: {} }, repo)
  expect(db.rows.field_session_commands[response.command_id].validated_payload).toEqual({ role: 'red' })
  expect(response).not.toHaveProperty('validated_payload')
  budgets(db)
})

test.each(['issued_next', 'revision'])('managed %s overflow never partially issues', async field => {
  const { db, repo } = await fixture()
  db.rows.session_command_authority[sessionId][field] = Number.MAX_SAFE_INTEGER
  await expect(submitFieldCommand(base, repo)).rejects.toBeDefined()
  expect(Object.values(db.rows.field_session_commands || {})).toHaveLength(0)
  expect(db.rows.session_command_authority[sessionId][field]).toBe(Number.MAX_SAFE_INTEGER)
  budgets(db)
})

test('command write failure rolls back head; callback conflict retry issues once', async () => {
  const { db, repo } = await fixture()
  const original = db.runTransaction.bind(db)
  db.runTransaction = callback => original(async tx => {
    const result = await callback(tx)
    if (result?.command_id) throw new Error('synthetic rollback')
    return result
  })
  await expect(submitFieldCommand(base, repo)).rejects.toThrow('synthetic rollback')
  expect(db.rows.session_command_authority[sessionId].issued_next).toBe(1)
  expect(Object.values(db.rows.field_session_commands || {})).toHaveLength(0)
  db.runTransaction = original
  await expect(submitFieldCommand(base, repo)).resolves.toMatchObject({ server_sequence: 1 })
  budgets(db)
})

test.each(['pending', 'confirmed'])('both legacy confirm writers refuse managed %s before retry', async status => {
  const { db, admin } = await fixture()
  db.rows.session_join_requests = { 'join-1': { session_id: sessionId, user_id: 'user-red', status, role: status === 'confirmed' ? 'red' : null } }
  const snapshot = JSON.stringify(db.rows)
  await expect(admin.confirm({ sessionId, tableId: 'table-1', requestId: 'join-1', role: 'red', operatorId: 'device-1' }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(createCloudJoinRepository(db).confirmRole({ requestId: 'join-1', role: 'red', operatorId: 'device-1' }))
    .rejects.toMatchObject({ code: 'COMMAND_FORBIDDEN' })
  expect(JSON.stringify(db.rows)).toBe(snapshot)
})

test('managed claim and complete require positive authority rather than legacy ACK projection', async () => {
  const { db, repo } = await fixture()
  const response = await submitFieldCommand(base, repo)
  const device = createCloudDeviceFieldCommandRepository(db)
  delete db.rows.session_command_authority[sessionId]
  const snapshot = JSON.stringify(db.rows)
  await expect(device.claimNext({ tableId: 'table-1', deviceId: 'device-1', now }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(JSON.stringify(db.rows)).toBe(snapshot)
  const command = db.rows.field_session_commands[response.command_id]
  command.status = 'claimed'; command.claimed_by_device_id = 'device-1'
  const claimed = JSON.stringify(db.rows)
  await expect(device.complete({ commandId: response.command_id, tableId: 'table-1', deviceId: 'device-1',
    status: 'applied', publicResultCode: 'APPLIED', resultingRevision: 1, now }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(JSON.stringify(db.rows)).toBe(claimed)
  budgets(db)
})

const { matchId } = require('../../cloudfunctions/shared/match-setup-service')
const { matchDocumentId } = require('../../cloudfunctions/shared/match-facts-repository')
function ranking(db) {
  roster(db)
  const id = matchId(sessionId)
  const projection = { match_id: id, session_id: sessionId, table_id: 'table-1', status: 'ready',
    participants: { red: 'user-red', blue: 'user-blue' }, settings_confirmations: ['red', 'blue'],
    settings: { settings_revision: 2, mode: 'observation', format: 'single', race_target: null, first_breaker: 'red', ranking_intent: true } }
  db.rows.matches = { [matchDocumentId(id)]: { schema_version: '1.0', match_id: id, session_id: sessionId,
    table_id: 'table-1', projection, event_ids: Array.from({ length: 128 }, (_, i) => `event-${i}`) } }
  return db.rows.matches[matchDocumentId(id)]
}
const breaker = { ...base, commandType: 'set_breaker', payload: { role: 'red' } }

test('managed breaker uses one projection document with 128 event IDs and raw replay ignores changed settings', async () => {
  const { db, repo } = await fixture()
  ranking(db)
  const response = await submitFieldCommand(breaker, repo)
  expect(db.rows.field_session_commands[response.command_id].validated_payload).toEqual({
    role: 'red', settings_revision: 2, mode: 'observation', format: 'single', race_target: null, ranking_intent: true,
  })
  const issue = db.transactionAttempts.at(-1)
  expect(issue.calls.filter(call => call.collection === 'matches' && call.method === 'get')).toHaveLength(1)
  expect(issue.calls.some(call => call.collection === 'match_events')).toBe(false)
  db.rows.matches = {}
  db.rows.session_role_bindings = {}
  expect(await submitFieldCommand(breaker, repo)).toEqual(response)
  budgets(db)
})

test.each(['identity', 'settings', 'status', 'participants', 'confirmations'])(
  'managed breaker rechecks ranking %s in issue transaction', async mutation => {
    const { db, repo } = await fixture()
    const match = ranking(db)
    const create = repo.createCommand
    repo.createCommand = (...args) => {
      if (mutation === 'identity') match.table_id = 'other'
      if (mutation === 'settings') match.projection.settings.settings_revision++
      if (mutation === 'status') match.projection.status = 'awaiting_settings_confirmations'
      if (mutation === 'participants') match.projection.participants.red = 'other'
      if (mutation === 'confirmations') match.projection.settings_confirmations = ['red']
      return create(...args)
    }
    await expect(submitFieldCommand(breaker, repo)).rejects.toBeDefined()
    expect(db.rows.session_command_authority[sessionId].issued_next).toBe(1)
    expect(Object.values(db.rows.field_session_commands || {})).toHaveLength(0)
    budgets(db)
  },
)

async function proposalFixture() {
  const setup = await fixture()
  roster(setup.db)
  Object.assign(setup.db.rows.field_live_sessions[sessionId], { red_group: 'stripe', blue_group: 'solid' })
  const proposal = await submitFieldCommand({ ...base, commandType: 'assign_groups', payload: { red_group: 'solid' } }, setup.repo)
  setup.db.rows.session_user_bindings[hash(`${sessionId}:user-blue`)] = {
    schema_version: '1.0', session_id: sessionId, table_id: 'table-1', user_id: 'user-blue', role: 'blue', request_id: 'join-blue',
  }
  return { ...setup, proposal }
}
const confirmation = { ...base, actorUserId: 'user-blue', commandType: 'assign_groups', payload: { red_group: 'solid' } }

test('managed correction confirmation rereads actual proposal doc in constant issue budget', async () => {
  const { db, repo, proposal } = await proposalFixture()
  const response = await submitFieldCommand(confirmation, repo)
  expect(db.rows.field_session_commands[response.command_id].validated_payload).toEqual({
    red_group: 'solid', correction_phase: 'confirmation', proposal_command_id: proposal.command_id,
  })
  expect(db.transactionAttempts.at(-1).calls.filter(call => call.id === proposal.command_id && call.method === 'get')).toHaveLength(1)
  budgets(db)
})

test.each(['missing', 'identity', 'payload', 'status', 'actor', 'sequence'])(
  'managed confirmation rechecks proposal %s after discovery', async mutation => {
    const { db, repo, proposal } = await proposalFixture()
    const create = repo.createCommand
    repo.createCommand = (...args) => {
      const stored = db.rows.field_session_commands[proposal.command_id]
      if (mutation === 'missing') delete db.rows.field_session_commands[proposal.command_id]
      if (mutation === 'identity') stored.session_id = 'other'
      if (mutation === 'payload') stored.validated_payload.red_group = 'stripe'
      if (mutation === 'status') stored.status = 'rejected'
      if (mutation === 'actor') stored.actor_user_id = 'user-blue'
      if (mutation === 'sequence') stored.server_sequence = 4
      return create(...args)
    }
    await expect(submitFieldCommand(confirmation, repo)).rejects.toMatchObject({ code: 'GROUP_CORRECTION_REQUIRED' })
    expect(db.rows.session_command_authority[sessionId].issued_next).toBe(2)
    budgets(db)
  },
)

test('managed issuance callback retry commits exactly one prefix increment', async () => {
  let retry = true
  const { db, repo } = await fixture({ retryConflicts: 1, beforeCommit: ({ attempt }) => {
    if (retry && attempt.calls.some(call => call.collection === 'field_session_commands' && call.method === 'set')) {
      retry = false
      throw Object.assign(new Error('retry issue'), { code: 'SYNTHETIC_CONFLICT' })
    }
  } })
  const response = await submitFieldCommand(base, repo)
  expect(response.server_sequence).toBe(1)
  expect(Object.keys(db.rows.field_session_commands)).toHaveLength(1)
  expect(db.rows.session_command_authority[sessionId]).toMatchObject({ issued_next: 2, revision: 2 })
  expect(db.transactionAttempts.at(-1).index).toBe(1)
  budgets(db)
})

const authority = require('../../cloudfunctions/shared/managed-command-authority')

test.each(['managed-bad', 'managed-', sessionId])('all reserved prefixes %s stay managed and never fall back', async id => {
  expect(authority.isManagedSessionId(id)).toBe(true)
  const { db, repo } = await fixture()
  if (id !== sessionId) {
    await expect(submitFieldCommand({ ...base, sessionId: id }, repo)).rejects.toMatchObject({ code: 'SESSION_NOT_JOINABLE' })
    expect(db.rows.field_command_sequences).toBeUndefined()
  }
})

test('shared context validates positive identity, pins and evolved head/live using only three docs', async () => {
  const { db } = await fixture()
  expect(authority.validManagedBirth(db.rows.managed_session_births[sessionId])).toBe(true)
  expect(authority.validManagedHead(db.rows.session_command_authority[sessionId])).toBe(true)
  expect(authority.validManagedLive(db.rows.field_live_sessions[sessionId])).toBe(true)
  expect(authority.sameManagedIdentity(db.rows.managed_session_births[sessionId], db.rows.field_live_sessions[sessionId])).toBe(true)
  for (const pins of [{}, { tableId: 'table-1', deviceId: 'device-1' }]) {
    const result = await db.runTransaction(tx => authority.readManagedCommandContext(tx, { sessionId, ...pins }))
    expect(result.birth.session_id).toBe(sessionId)
    expect(db.transactionAttempts.at(-1).count).toBe(3)
  }
  for (const pins of [{ tableId: 'table-2' }, { deviceId: 'device-2' }]) {
    expect(await db.runTransaction(tx => authority.readManagedCommandContext(tx, { sessionId, ...pins }))).toBeNull()
  }
})

test.each([
  { issued_next: undefined }, { issued_next: 0 }, { issued_next: 1.1 }, { issued_next: Number.MAX_SAFE_INTEGER + 1 },
  { closed_target: undefined }, { revision: 0 }, { revision: Number.MAX_SAFE_INTEGER + 1 },
  { contiguous_ack_sequence: -1 }, { contiguous_ack_sequence: 1 },
  { last_applied_server_sequence: 1 }, { last_resulting_revision: -1 },
  { last_resulting_revision: 1 }, { role_tails: { red: null, blue: null, unknown: null } },
  { role_tails: { red: null, blue: null, _id: 'nested' } }, { role_tails: { red: 'bad', blue: null } },
  { coverage: 'closing', closed_target: 1 }, { coverage: 'sealed', closed_target: 0, seal: null },
  { coverage: 'sealed', closed_target: 0, seal: { sealed_at: now, _id: 'nested' } },
  { coverage: 'sealed', closed_target: 0, seal: { sealed_at: 'yesterday' } },
  { seal: { sealed_at: now } }, { unknown: 1 }, { _id: [] },
])('shared strict head rejects malformed range/state %p', async mutation => {
  const { db } = await fixture()
  const head = { ...db.rows.session_command_authority[sessionId], ...mutation }
  expect(authority.validManagedHead(head)).toBe(false)
})

test.each([
  { status: 'ended' }, { control_revision: Number.MAX_SAFE_INTEGER + 1 }, { started_at: 'yesterday' },
  { red_user_id: ['user-red'] }, { red_user_id: 'same', blue_user_id: 'same' }, { active_user_id: 'unknown' },
  { red_group: 'solid' }, { red_group: 'solid', blue_group: 'solid' },
  { mode: 'observation' }, { breaker: 'unknown' }, { _id: 1 }, { unknown: true },
  { mode: 'assistance', format: 'single', race_target: null, settings_revision: 1, ranking_intent: true },
  { mode: 'observation', format: 'race', race_target: null, settings_revision: 1, ranking_intent: false },
  { mode: 'observation', format: 'single', race_target: 2, settings_revision: 1, ranking_intent: false },
])('shared strict live rejects malformed projection %p', async mutation => {
  const { db } = await fixture()
  expect(authority.validManagedLive({ ...db.rows.field_live_sessions[sessionId], ...mutation })).toBe(false)
})

test.each(['open', 'closing', 'sealed'])('shared head accepts evolved %s with safe positive ranges', async coverage => {
  const { db } = await fixture()
  Object.assign(db.rows.session_command_authority[sessionId], {
    issued_next: 129, contiguous_ack_sequence: 128, last_applied_server_sequence: 127,
    last_resulting_revision: 65, revision: 258, coverage,
    role_tails: { red: 'a'.repeat(64), blue: 'b'.repeat(64) },
    closed_target: coverage === 'open' ? null : 128,
    seal: coverage === 'sealed' ? { sealed_at: now } : null,
  })
  expect(authority.validManagedHead(db.rows.session_command_authority[sessionId])).toBe(true)
})

test('context distinguishes SDK outages from missing/corrupt authority', async () => {
  const { db } = await fixture()
  Object.assign(db.rows.field_live_sessions[sessionId], { opened_at: '2026-11-01T09:00:00.000Z' })
  expect(await db.runTransaction(tx => authority.readManagedCommandContext(tx, { sessionId }))).toBeNull()
  const fault = { collection: () => ({ doc: () => ({ get: async () => { throw new Error('SDK unavailable') } }) }) }
  await expect(authority.readManagedCommandContext(fault, { sessionId })).rejects.toThrow('SDK unavailable')
  const missing = { collection: () => ({ doc: id => ({ get: async () => {
    throw new Error(`document.get:fail document with _id ${id} does not exist`)
  } }) }) }
  expect(await authority.readManagedCommandContext(missing, { sessionId })).toBeNull()
})

test.each([65, 128])('issue/replay remains constant after %s synthetic issued commands', async length => {
  const { db, repo } = await fixture()
  const commands = {}
  for (let i = 1; i <= length; i++) commands[`old-${i}`] = { session_id: sessionId, server_sequence: i, status: 'applied' }
  db.rows.field_session_commands = commands
  Object.assign(db.rows.session_command_authority[sessionId], {
    issued_next: length + 1, contiguous_ack_sequence: length, last_applied_server_sequence: length,
    last_resulting_revision: length, revision: length * 2 + 1,
  })
  const response = await submitFieldCommand(base, repo)
  expect(response.server_sequence).toBe(length + 1)
  expect(db.transactionAttempts.at(-1).count).toBe(10)
  expect(await submitFieldCommand(base, repo)).toEqual(response)
  expect(db.transactionAttempts.at(-1).count).toBe(4)
  budgets(db)
})

test.each(['schema', 'idempotency', 'intent'])('managed confirmation rejects corrupt proposal %s', async mutation => {
  const { db, repo, proposal } = await proposalFixture()
  const create = repo.createCommand
  repo.createCommand = (...args) => {
    const stored = db.rows.field_session_commands[proposal.command_id]
    if (mutation === 'schema') stored.schema_version = '2.0'
    if (mutation === 'idempotency') stored.idempotency_digest = 'b'.repeat(64)
    if (mutation === 'intent') delete stored.intent_digest
    return create(...args)
  }
  await expect(submitFieldCommand(confirmation, repo)).rejects.toMatchObject({ code: 'GROUP_CORRECTION_REQUIRED' })
  expect(db.rows.session_command_authority[sessionId].issued_next).toBe(2)
  budgets(db)
})

test.each([{ expectedRevision: 1 }, { commandType: 'release_role', payload: {} }])(
  'raw replay rejects changed submitted intent %p before mutable authorization', async mutation => {
    const { db, repo } = await fixture()
    await submitFieldCommand(base, repo)
    Object.assign(db.rows.field_live_sessions[sessionId], { status: 'ended', ended_at: now })
    await expect(submitFieldCommand({ ...base, ...mutation }, repo)).rejects.toMatchObject({ code: 'COMMAND_IDEMPOTENCY_CONFLICT' })
  },
)

test('managed release freezes role from the issue transaction rather than discovery', async () => {
  const { db, repo } = await fixture()
  roster(db)
  const create = repo.createCommand
  repo.createCommand = (...args) => {
    const live = db.rows.field_live_sessions[sessionId]
    live.red_user_id = 'user-blue'; live.blue_user_id = 'user-red'
    db.rows.session_role_bindings[hash(`${sessionId}:red`)].user_id = 'user-blue'
    db.rows.session_role_bindings[hash(`${sessionId}:blue`)].user_id = 'user-red'
    db.rows.session_user_bindings[hash(`${sessionId}:user-red`)].role = 'blue'
    return create(...args)
  }
  const response = await submitFieldCommand({ ...base, commandType: 'release_role', payload: {} }, repo)
  expect(db.rows.field_session_commands[response.command_id].validated_payload).toEqual({ role: 'blue' })
  expect(Object.keys(response)).toHaveLength(7)
  budgets(db)
})

test('head write failure rolls back the already written command', async () => {
  let armed = false
  const { db, repo } = await fixture({ failWrite: collection => {
    if (armed && collection === 'session_command_authority') throw new Error('synthetic head failure')
  } })
  armed = true
  await expect(submitFieldCommand(base, repo)).rejects.toThrow('synthetic head failure')
  expect(db.rows.session_command_authority[sessionId]).toMatchObject({ issued_next: 1, revision: 1 })
  expect(Object.values(db.rows.field_session_commands || {})).toHaveLength(0)
  budgets(db)
})

test('managed replay rejects a stored validated payload that disagrees with its immutable intent digest', async () => {
  const { db, repo } = await fixture()
  const response = await submitFieldCommand(base, repo)
  db.rows.field_session_commands[response.command_id].validated_payload.role = 'blue'
  await expect(submitFieldCommand(base, repo)).rejects.toMatchObject({ code: 'COMMAND_IDEMPOTENCY_CONFLICT' })
  expect(db.rows.session_command_authority[sessionId].issued_next).toBe(2)
})

test('shared live validator accepts actual populated started/ended projections and top-level SDK identity', async () => {
  const { db } = await fixture()
  const live = { ...db.rows.field_live_sessions[sessionId], _id: sessionId, status: 'ended', ended_at: now,
    control_revision: 65, red_user_id: 'user-red', blue_user_id: 'user-blue', active_user_id: 'user-blue',
    breaker: 'red', started_at: now, mode: 'observation', format: 'race', race_target: 5,
    settings_revision: 2, ranking_intent: true, red_group: 'solid', blue_group: 'stripe' }
  expect(authority.validManagedLive(live)).toBe(true)
})
