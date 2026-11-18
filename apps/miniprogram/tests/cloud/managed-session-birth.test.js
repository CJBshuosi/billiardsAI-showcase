const crypto = require('crypto')
const { createJoinAdmin } = require('../../cloudfunctions/sync-api/lib/join-admin')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')

const birthId = '12345678-1234-4234-8234-123456789abc'
const requestId = '98765432-1234-4234-9234-123456789abc'
const sessionId = `managed-${birthId}`
const now = '2026-10-29T10:00:00.000Z'
const input = { openRequestId: requestId, tableId: 'table-1', deviceId: 'device-1' }
const requestKey = crypto.createHash('sha256').update(`device-1:${requestId}`).digest('hex')
function setup(options = {}, seed = {}) {
  const db = ownerDeleteDb({ table_join_codes: {
    'table-1': { table_id: 'table-1', status: 'active', current_session_id: null },
    'table-2': { table_id: 'table-2', status: 'active', current_session_id: null },
  }, ...seed }, { strictTransactions: true, ...options.dbOptions })
  const idFactory = options.idFactory || jest.fn(() => birthId)
  const admin = createJoinAdmin(db, {
    managedSessionsEnabled: true, nowIso: () => now, idFactory, ...options,
  })
  return { db, admin, idFactory }
}
const expected = {
  schema_version: '1.0', session_id: sessionId, birth_id: birthId,
  birth_contract: 'managed_session_birth_v1', table_id: 'table-1', device_id: 'device-1', status: 'open',
}
function budget(db) {
  for (const attempt of db.transactionAttempts) {
    expect(attempt.count).toBeLessThanOrEqual(25)
    expect(attempt.calls.every(call => ['get', 'set', 'update'].includes(call.method))).toBe(true)
  }
}

test('managed allocation is default disabled, even when environment opts in', async () => {
  const { db } = setup()
  await expect(createJoinAdmin(db).createManagedSession(input))
    .rejects.toMatchObject({ code: 'SYNC_CONFIGURATION_REQUIRED' })
  expect(db.transactionAttempts).toHaveLength(0)
})

test('server allocates exact permanent records and public identity without command/media effects', async () => {
  const { db, admin, idFactory } = setup()
  expect(await admin.createManagedSession(input)).toEqual(expected)
  expect(idFactory).toHaveBeenCalledTimes(1)
  expect(db.rows.managed_session_births[sessionId]).toEqual({
    contract: 'managed_session_birth_v1', session_id: sessionId, birth_id: birthId,
    table_id: 'table-1', device_id: 'device-1', opened_at: now,
  })
  expect(db.rows.managed_session_open_requests[requestKey]).toEqual({
    contract: 'managed_session_open_request_v1', open_request_id: requestId,
    session_id: sessionId, birth_id: birthId, table_id: 'table-1', device_id: 'device-1',
  })
  expect(db.rows.session_command_authority[sessionId]).toEqual({
    contract: 'managed_command_authority_v1', session_id: sessionId, birth_id: birthId,
    table_id: 'table-1', device_id: 'device-1', contiguous_ack_sequence: 0,
    last_resulting_revision: 0, last_applied_server_sequence: 0,
    role_tails: { red: null, blue: null }, coverage: 'open', seal: null, revision: 1,
    issued_next: 1, closed_target: null,
  })
  expect(db.rows.field_live_sessions[sessionId]).toEqual({
    schema_version: '1.0', session_id: sessionId, table_id: 'table-1', device_id: 'device-1',
    status: 'open', opened_at: now, updated_at: now, environment: 'development',
    birth_contract: 'managed_session_birth_v1', birth_id: birthId,
  })
  expect(db.rows.table_join_codes['table-1'].current_session_id).toBe(sessionId)
  expect(db.transactionAttempts[0].calls.map(call => call.collection)).not.toEqual(expect.arrayContaining([
    'field_session_commands', 'clip_storage_authorities', 'media_storage_operations', 'saved_clip_sync_jobs',
  ]))
  budget(db)
})

test('exact replay reads existing identity before randomness and does not rewrite records', async () => {
  const { db, admin, idFactory } = setup()
  await admin.createManagedSession(input)
  const snapshot = JSON.stringify(db.rows)
  expect(await admin.createManagedSession(input)).toEqual(expected)
  expect(idFactory).toHaveBeenCalledTimes(1)
  expect(JSON.stringify(db.rows)).toBe(snapshot)
  expect(db.transactionAttempts[1].calls.every(call => call.method === 'get')).toBe(true)
  budget(db)
})

test('ended replay stays ended even with a new current session', async () => {
  const { db, admin } = setup()
  await admin.createManagedSession(input)
  Object.assign(db.rows.field_live_sessions[sessionId], { status: 'ended', ended_at: now })
  Object.assign(db.rows.session_command_authority[sessionId], {
    coverage: 'sealed', closed_target: 0, seal: { sealed_at: now }, revision: 2,
  })
  db.rows.table_join_codes['table-1'].current_session_id = 'new-live'
  expect(await admin.createManagedSession(input)).toEqual({ ...expected, status: 'ended' })
  budget(db)
})

test('same device/request cannot change table and another device cannot replay its identity', async () => {
  const { db, admin, idFactory } = setup()
  await admin.createManagedSession(input)
  await expect(admin.createManagedSession({ ...input, tableId: 'table-2' })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await expect(admin.createManagedSession({ ...input, deviceId: 'device-2' })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(idFactory).toHaveBeenCalledTimes(1)
  budget(db)
})

test.each(['managed_session_births', 'managed_session_open_requests', 'session_command_authority', 'field_live_sessions'])(
  'missing %s while the table points to the open session denies allocation or replay', async collection => {
    // Missing request authority is blocked here by the open table pointer; permanent mapping loss after end is unsupported.
    const { db, admin, idFactory } = setup()
    await admin.createManagedSession(input)
    delete db.rows[collection][collection === 'managed_session_open_requests' ? requestKey : sessionId]
    await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(idFactory).toHaveBeenCalledTimes(1)
    budget(db)
  },
)

test.each([
  ['managed_session_births', 'opened_at', 'not-a-time'],
  ['managed_session_births', 'table_id', ['table-1']],
  ['managed_session_births', 'unexpected', true],
  ['managed_session_open_requests', 'birth_id', 'bad'],
  ['managed_session_open_requests', 'device_id', 'device-2'],
  ['managed_session_open_requests', 'unexpected', true],
  ['session_command_authority', 'role_tails', []],
  ['session_command_authority', 'unexpected', true],
  ['session_command_authority', 'table_id', 'table-2'],
  ['field_live_sessions', 'birth_id', 'bad'],
  ['field_live_sessions', 'device_id', 'device-2'],
  ['field_live_sessions', 'status', 'unknown'],
  ['field_live_sessions', 'updated_at', '2026-02-30T10:00:00.000Z'],
  ['field_live_sessions', 'unexpected', true],
])('corrupt %s.%s fails closed', async (collection, key, value) => {
  const { db, admin } = setup()
  await admin.createManagedSession(input)
  db.rows[collection][collection === 'managed_session_open_requests' ? requestKey : sessionId][key] = value
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  budget(db)
})

test('SDK top-level _id is allowed on every permanent and live record', async () => {
  const { db, admin } = setup()
  await admin.createManagedSession(input)
  for (const collection of ['managed_session_births', 'managed_session_open_requests', 'session_command_authority', 'field_live_sessions']) {
    const key = collection === 'managed_session_open_requests' ? requestKey : sessionId
    db.rows[collection][key]._id = key
  }
  expect(await admin.createManagedSession(input)).toEqual(expected)
})

test.each([null, 'other-session'])('open replay requires unchanged table pointer (%s)', async pointer => {
  const { db, admin } = setup()
  await admin.createManagedSession(input)
  db.rows.table_join_codes['table-1'].current_session_id = pointer
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test.each(['managed_session_births', 'field_live_sessions', 'sessions', 'session_command_authority'])(
  'generated identity collision in %s rolls back without overwrite', async collection => {
    const { db, admin } = setup({}, { [collection]: { [sessionId]: { occupied: true } } })
    const snapshot = JSON.stringify(db.rows)
    await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(JSON.stringify(db.rows)).toBe(snapshot)
    budget(db)
  },
)

test.each(['open', 'missing', 'wrong-table', 'ended'])(
  'current table session guard handles %s', async status => {
    const { db, admin } = setup()
    db.rows.table_join_codes['table-1'].current_session_id = 'old-session'
    db.rows.field_live_sessions = status === 'missing' ? {} : { 'old-session': {
      session_id: 'old-session', table_id: status === 'wrong-table' ? 'table-2' : 'table-1',
      status: status === 'ended' || status === 'wrong-table' ? 'ended' : 'open',
    } }
    if (status === 'ended') expect(await admin.createManagedSession(input)).toEqual(expected)
    else await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    budget(db)
  },
)

test.each(['managed_session_births', 'managed_session_open_requests', 'session_command_authority', 'field_live_sessions', 'table_join_codes'])(
  'failure writing %s rolls back the entire birth', async target => {
    const { db, admin } = setup({ dbOptions: { failWrite: collection => { if (collection === target) throw new Error('synthetic write failure') } } })
    const snapshot = JSON.stringify(db.rows)
    await expect(admin.createManagedSession(input)).rejects.toThrow('synthetic write failure')
    expect(JSON.stringify(db.rows)).toBe(snapshot)
    budget(db)
  },
)

test('concurrent same request converges and competing request for table conflicts', async () => {
  const { db, admin, idFactory } = setup()
  const results = await Promise.all([admin.createManagedSession(input), admin.createManagedSession(input)])
  expect(results).toEqual([expected, expected])
  await expect(admin.createManagedSession({ ...input, openRequestId: birthId })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(idFactory).toHaveBeenCalledTimes(1)
  budget(db)
})

test('callback conflict retries atomically with a fresh random identity', async () => {
  const secondId = 'aaaaaaaa-1234-4234-9234-123456789abc'
  const idFactory = jest.fn().mockReturnValueOnce(birthId).mockReturnValueOnce(secondId)
  const { db, admin } = setup({ idFactory, dbOptions: { retryConflicts: 1, beforeCommit: ({ attempt }) => {
    if (!attempt.index) throw Object.assign(new Error('retry'), { code: 'SYNTHETIC_CONFLICT' })
  } } })
  expect(await admin.createManagedSession(input)).toEqual({ ...expected, session_id: `managed-${secondId}`, birth_id: secondId })
  expect(db.rows.managed_session_births[sessionId]).toBeUndefined()
  expect(db.rows.managed_session_open_requests[requestKey].birth_id).toBe(secondId)
  budget(db)
})

test.each(['open', 'ended', 'missing'])(
  'legacy lifecycle rejects reserved managed namespace with %s metadata even disabled', async status => {
    const { db } = setup({}, { field_live_sessions: status === 'missing' ? {} : {
      [sessionId]: { session_id: sessionId, table_id: 'table-1', status },
    } })
    const admin = createJoinAdmin(db)
    const snapshot = JSON.stringify(db.rows)
    for (const method of ['openSession', 'closeSession']) {
      await expect(admin[method]({ sessionId, tableId: 'table-1' })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
      await expect(admin[method]({ sessionId: 'managed-unknown', tableId: 'table-1' })).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    }
    expect(JSON.stringify(db.rows)).toBe(snapshot)
  },
)

test.each([null, [], {}, '', '12345678-1234-1234-8234-123456789abc'])(
  'unsafe open request %p is invalid before database access', async openRequestId => {
    const { db, admin } = setup()
    await expect(admin.createManagedSession({ ...input, openRequestId })).rejects.toMatchObject({ code: 'SYNC_INVALID' })
    expect(db.transactionAttempts).toHaveLength(0)
  },
)

test.each([null, [], 'not-a-uuid'])('invalid server ID %p fails before writes', async id => {
  const { db, admin } = setup({ idFactory: () => id })
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.transactionAttempts[0].calls.every(call => call.method === 'get')).toBe(true)
})

const { createHandler } = require('../../cloudfunctions/sync-api')
const { createCloudAuthRepositories, canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
function signedEvent(payload, overrides = {}) {
  const body = JSON.stringify(payload)
  const path = overrides.path || '/v2/field-sessions/open'
  const timestamp = '1793268000'
  const nonce = crypto.randomBytes(16).toString('hex')
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const signature = crypto.createHmac('sha256', 'test-secret').update(canonicalSignatureInput({ method: 'POST', path, timestamp, nonce, bodyHash })).digest('hex')
  return { httpMethod: 'POST', path, body, headers: {
    'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce,
    'x-body-sha256': bodyHash, 'x-signature': signature,
  }, ...overrides }
}
function handlerSetup(options) {
  const fixture = setup(options, { sync_devices: { 'device-1': {
    device_id: 'device-1', table_id: 'table-1', status: 'active', secret: 'test-secret',
  } } })
  const storage = { createUploadAuthorization: jest.fn(() => { throw new Error('unexpected provider call') }) }
  return { ...fixture, storage, handler: createHandler({
    auth: { ...createCloudAuthRepositories(fixture.db), nowSeconds: () => 1793268000 },
    joinAdmin: fixture.admin, storage,
  }) }
}

test('signed route allocates using authenticated scope and stays within composed doc budget', async () => {
  const { db, handler, storage } = handlerSetup()
  const response = await handler(signedEvent({ schema_version: '1.0', open_request_id: requestId }))
  expect(response.statusCode).toBe(200)
  expect(JSON.parse(response.body)).toEqual({ ok: true, data: expected })
  expect(storage.createUploadAuthorization).not.toHaveBeenCalled()
  // Authentication device lookup (1), nonce transaction and allocation transaction.
  expect(1 + db.transactionAttempts.reduce((sum, attempt) => sum + attempt.count, 0)).toBeLessThanOrEqual(25)
  const replay = await handler(signedEvent({ schema_version: '1.0', open_request_id: requestId }))
  expect(JSON.parse(replay.body)).toEqual({ ok: true, data: expected })
  delete db.rows.field_live_sessions[sessionId]
  expect((await handler(signedEvent({ schema_version: '1.0', open_request_id: requestId }))).statusCode).toBe(409)
})

test.each([
  {}, { schema_version: '2.0', open_request_id: requestId },
  { schema_version: '1.0', open_request_id: [requestId] },
  ...['device_id', 'table_id', 'session_id', 'birth_id', 'unknown'].map(field => ({ schema_version: '1.0', open_request_id: requestId, [field]: 'injected' })),
])('signed route rejects malformed or extra body fields %p without birth writes', async payload => {
  const { db, handler, idFactory } = handlerSetup()
  const response = await handler(signedEvent(payload))
  expect(response.statusCode).toBe(400)
  expect(JSON.parse(response.body)).toEqual({ ok: false, error: { code: 'SYNC_INVALID' } })
  expect(idFactory).not.toHaveBeenCalled()
  expect(db.rows.managed_session_births).toBeUndefined()
})

test('authentication precedes allocation and default-disabled route returns configuration error', async () => {
  const { db, handler, idFactory } = handlerSetup()
  const bad = signedEvent({ schema_version: '1.0', open_request_id: requestId })
  bad.headers['x-signature'] = '0'.repeat(64)
  expect((await handler(bad)).statusCode).toBe(401)
  expect(db.transactionAttempts).toHaveLength(0)
  expect(idFactory).not.toHaveBeenCalled()
  const disabled = handlerSetup({ managedSessionsEnabled: false })
  const response = await disabled.handler(signedEvent({ schema_version: '1.0', open_request_id: requestId }))
  expect(JSON.parse(response.body)).toEqual({ ok: false, error: { code: 'SYNC_CONFIGURATION_REQUIRED' } })
})

test('production composition stays disabled even when an environment switch is present', async () => {
  const { db } = handlerSetup()
  const previous = process.env.MANAGED_SESSIONS_ENABLED
  process.env.MANAGED_SESSIONS_ENABLED = 'true'
  const dateSpy = jest.spyOn(Date, 'now').mockReturnValue(1793268000000)
  jest.doMock('wx-server-sdk', () => ({ init() {}, database: () => db, DYNAMIC_CURRENT_ENV: 'test' }))
  jest.doMock('@cloudbase/node-sdk', () => ({ init: () => ({
    getUploadMetadata() { throw new Error('unexpected provider call') },
    downloadFile() { throw new Error('unexpected provider call') },
  }), SYMBOL_CURRENT_ENV: 'test' }))
  try {
    let productionMain
    jest.isolateModules(() => { productionMain = require('../../cloudfunctions/sync-api').main })
    const response = await productionMain(signedEvent({ schema_version: '1.0', open_request_id: requestId }))
    expect(JSON.parse(response.body)).toEqual({ ok: false, error: { code: 'SYNC_CONFIGURATION_REQUIRED' } })
    expect(db.rows.managed_session_births).toBeUndefined()
    expect(db.transactionAttempts).toHaveLength(1)
    expect(db.transactionAttempts[0].calls.map(call => call.collection)).toEqual(['sync_nonces', 'sync_nonces'])
  } finally {
    dateSpy.mockRestore()
    jest.dontMock('wx-server-sdk')
    jest.dontMock('@cloudbase/node-sdk')
    if (previous === undefined) delete process.env.MANAGED_SESSIONS_ENABLED
    else process.env.MANAGED_SESSIONS_ENABLED = previous
  }
})

test('two concurrent new requests for a table commit exactly one birth', async () => {
  const { db, admin } = setup()
  const results = await Promise.allSettled([
    admin.createManagedSession(input),
    admin.createManagedSession({ ...input, openRequestId: birthId }),
  ])
  expect(results[0]).toEqual({ status: 'fulfilled', value: expected })
  expect(results[1]).toMatchObject({ status: 'rejected', reason: { code: 'SYNC_CONFLICT' } })
  expect(Object.keys(db.rows.managed_session_births)).toEqual([sessionId])
  expect(Object.keys(db.rows.managed_session_open_requests)).toEqual([requestKey])
  budget(db)
})

test.each(['inactive', 'wrong-table', 'missing'])('table %s denies managed allocation before randomness', async state => {
  const { db, admin, idFactory } = setup()
  if (state === 'missing') delete db.rows.table_join_codes['table-1']
  else if (state === 'inactive') db.rows.table_join_codes['table-1'].status = 'disabled'
  else db.rows.table_join_codes['table-1'].table_id = 'table-2'
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFIGURATION_REQUIRED' })
  expect(idFactory).not.toHaveBeenCalled()
  budget(db)
})

test('another authenticated device with the same request ID cannot claim an existing birth on another table', async () => {
  const { db, admin } = setup()
  await admin.createManagedSession(input)
  await expect(admin.createManagedSession({ ...input, deviceId: 'device-2', tableId: 'table-2' }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.rows.managed_session_open_requests[requestKey].device_id).toBe('device-1')
  expect(Object.keys(db.rows.managed_session_open_requests)).toHaveLength(1)
  expect(db.rows.table_join_codes['table-2'].current_session_id).toBeNull()
  budget(db)
})

test.each(['open', 'close'])('signed legacy %s cannot mutate managed identities', async operation => {
  const { db, handler } = handlerSetup()
  const result = await handler(signedEvent({}, { path: `/v1/field-sessions/managed-unknown/${operation}` }))
  expect(result.statusCode).toBe(409)
  expect(JSON.parse(result.body)).toEqual({ ok: false, error: { code: 'SYNC_CONFLICT' } })
  expect(db.rows.field_live_sessions).toBeUndefined()
  expect(db.rows.table_join_codes['table-1'].current_session_id).toBeNull()
})

test.each(['managed_session_births', 'managed_session_open_requests', 'session_command_authority', 'field_live_sessions'])(
  'malformed SDK _id in %s is rejected', async collection => {
    const { db, admin } = setup()
    await admin.createManagedSession(input)
    const key = collection === 'managed_session_open_requests' ? requestKey : sessionId
    db.rows[collection][key]._id = [key]
    await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  },
)

test('CloudBase missing active table error becomes configuration required', async () => {
  const { db, admin } = setup()
  const runTransaction = db.runTransaction.bind(db)
  db.runTransaction = callback => runTransaction(transaction => callback({ collection(name) {
    const collection = transaction.collection(name)
    if (name !== 'table_join_codes') return collection
    return { doc(id) { return { async get() {
      await collection.doc(id).get()
      throw new Error(`document.get:fail document with _id ${id} does not exist`)
    } } } }
  } }))
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFIGURATION_REQUIRED' })
})


test('allocation replay accepts evolved positive head and real live projection', async () => {
  const { db, admin } = setup()
  await admin.createManagedSession(input)
  Object.assign(db.rows.session_command_authority[sessionId], {
    issued_next: 4, contiguous_ack_sequence: 2, last_applied_server_sequence: 1,
    last_resulting_revision: 1, revision: 6, role_tails: { red: 'a'.repeat(64), blue: null },
  })
  Object.assign(db.rows.field_live_sessions[sessionId], {
    control_revision: 1, red_user_id: 'user-red', blue_user_id: null,
    active_user_id: null, breaker: null, started_at: null,
    mode: null, format: null, race_target: null, ranking_intent: false, settings_revision: 0,
    red_group: null, blue_group: null,
  })
  expect(await admin.createManagedSession(input)).toEqual(expected)
  expect(db.transactionAttempts.at(-1).count).toBeLessThanOrEqual(12)
})

test.each(['createManagedSession', 'openSession'])('%s requires a sealed managed predecessor', async method => {
  const { db, admin } = setup({ idFactory: () => 'aaaaaaaa-1234-4234-9234-123456789abc' })
  const first = createJoinAdmin(db, { managedSessionsEnabled: true, nowIso: () => now, idFactory: () => birthId })
  await first.createManagedSession(input)
  Object.assign(db.rows.field_live_sessions[sessionId], { status: 'ended', ended_at: now })
  const next = { ...input, openRequestId: birthId, sessionId: 'legacy-next' }
  await expect(admin[method](next)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  Object.assign(db.rows.session_command_authority[sessionId], {
    coverage: 'sealed', issued_next: 1, closed_target: 0, seal: { sealed_at: now }, revision: 2,
  })
  await expect(admin[method](next)).resolves.toMatchObject({ status: 'open' })
  budget(db)
})

test.each(['open', 'closing'])('allocation replay cannot bypass %s predecessor by changing table pointer', async coverage => {
  const { db, admin } = setup()
  await admin.createManagedSession(input)
  Object.assign(db.rows.field_live_sessions[sessionId], { status: 'ended', ended_at: now })
  if (coverage === 'closing') Object.assign(db.rows.session_command_authority[sessionId], { coverage, closed_target: 0, revision: 2 })
  db.rows.table_join_codes['table-1'].current_session_id = 'some-other-session'
  await expect(admin.createManagedSession(input)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  budget(db)
})

test('signed managed close has exact scope-bound response and independent authentication budget', async () => {
  const { db, admin, handler } = handlerSetup()
  await admin.createManagedSession(input)
  const offset = db.transactionAttempts.length
  const response = await handler(signedEvent({ schema_version: '1.0' }, { path: `/v2/field-sessions/${sessionId}/close` }))
  expect(response.statusCode).toBe(200)
  expect(JSON.parse(response.body)).toEqual({ ok: true, data: {
    schema_version: '1.0', session_id: sessionId, birth_id: birthId, birth_contract: 'managed_session_birth_v1',
    table_id: 'table-1', device_id: 'device-1', coverage: 'sealed', closed_target: 0, contiguous_ack_sequence: 0,
  } })
  expect(db.transactionAttempts.slice(offset).map(attempt => attempt.count)).toEqual([2, 6])
  expect(db.rows.table_join_codes['table-1'].current_session_id).toBe(sessionId)
})

test.each([{}, { schema_version: '2.0' }, ...['closed_target', 'local_revision', 'request_id', 'device_id', 'table_id', '_id']
  .map(key => ({ schema_version: '1.0', [key]: 'injected' }))])('managed close rejects nonexact request %p', async payload => {
  const { db, admin, handler } = handlerSetup()
  await admin.createManagedSession(input)
  const response = await handler(signedEvent(payload, { path: `/v2/field-sessions/${sessionId}/close` }))
  expect(JSON.parse(response.body)).toEqual({ ok: false, error: { code: 'SYNC_INVALID' } })
  expect(db.rows.session_command_authority[sessionId].coverage).toBe('open')
})

test('close remains disabled by default and malformed managed suffix never falls back', async () => {
  const disabled = handlerSetup({ managedSessionsEnabled: false })
  const response = await disabled.handler(signedEvent({ schema_version: '1.0' }, { path: `/v2/field-sessions/${sessionId}/close` }))
  expect(JSON.parse(response.body)).toEqual({ ok: false, error: { code: 'SYNC_CONFIGURATION_REQUIRED' } })
  const { handler } = handlerSetup()
  const malformed = await handler(signedEvent({ schema_version: '1.0' }, { path: '/v2/field-sessions/managed-not-uuid/close' }))
  expect(JSON.parse(malformed.body)).toEqual({ ok: false, error: { code: 'SYNC_CONFLICT' } })
})
