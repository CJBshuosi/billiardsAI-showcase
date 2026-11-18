const {
  createJoinAdmin,
  publicPendingRequest,
} = require('../../cloudfunctions/sync-api/lib/join-admin')


function createDatabase(seed = {}, { cloudDocumentIds = false, throwOnMissing = false } = {}) {
  const collections = new Map()
  for (const [name, values] of Object.entries(seed)) {
    collections.set(name, new Map(Object.entries(values)))
  }
  function collection(name) {
    if (!collections.has(name)) collections.set(name, new Map())
    const records = collections.get(name)
    return {
      where(query) {
        let selected = [...records.values()].filter((record) => (
          Object.entries(query).every(([key, value]) => record[key] === value)
        ))
        const chain = {
          orderBy(key, direction) {
            selected.sort((left, right) => String(left[key]).localeCompare(String(right[key])))
            if (direction === 'desc') selected.reverse()
            return chain
          },
          limit(count) { selected = selected.slice(0, count); return chain },
          async get() { return { data: selected } },
        }
        return chain
      },
      doc(id) {
        return {
          async get() {
            if (throwOnMissing && !records.has(id)) {
              throw new Error(`document.get:fail document with _id ${id} does not exist`)
            }
            const data = records.get(id)
            return { data: data && cloudDocumentIds ? { ...data, _id: id } : data }
          },
          async set({ data }) {
            if (cloudDocumentIds && Object.hasOwn(data, '_id')) {
              throw new Error('database set failed: _id is reserved')
            }
            records.set(id, data)
          },
          async update({ data }) { records.set(id, { ...records.get(id), ...data }) },
        }
      },
    }
  }
  return {
    collections,
    collection,
    async runTransaction(callback) { return callback({ collection }) },
  }
}


function configuredDatabase(options) {
  return createDatabase({
    table_join_codes: {
      'table-1': {
        schema_version: '1.0', table_id: 'table-1', status: 'active',
        current_session_id: null,
      },
    },
  }, options)
}


test('pending request exposes public identity without internal user id', () => {
  expect(publicPendingRequest({
    request_id: 'request-1', user_id: 'user-secret', display_name: '球友 FC9529',
    status: 'pending', role: null,
  })).toEqual({
    schema_version: '1.0', request_id: 'request-1', display_name: '球友 FC9529',
    status: 'pending', role: null,
  })
})


test('field display query keeps confirmed player names and roles', async () => {
  const db = createDatabase({
    field_live_sessions: {
      'session-1': { session_id: 'session-1', table_id: 'table-1', status: 'open' },
    },
    session_join_requests: {
      'request-red': {
        request_id: 'request-red', session_id: 'session-1', user_id: 'secret-red',
        display_name: '球友 87BC6C', status: 'confirmed', role: 'red', created_at: '1',
      },
      'request-blue': {
        request_id: 'request-blue', session_id: 'session-1', user_id: 'secret-blue',
        display_name: '球友 42AF10', status: 'confirmed', role: 'blue', created_at: '2',
      },
      ended: {
        request_id: 'ended', session_id: 'session-1', user_id: 'secret-old',
        display_name: '旧用户', status: 'ended', role: null, created_at: '0',
      },
    },
  })
  const admin = createJoinAdmin(db)

  await expect(admin.listRequests({ sessionId: 'session-1', tableId: 'table-1' }))
    .resolves.toEqual({
      schema_version: '1.0',
      requests: [
        {
          schema_version: '1.0', request_id: 'request-red', display_name: '球友 87BC6C',
          status: 'confirmed', role: 'red',
        },
        {
          schema_version: '1.0', request_id: 'request-blue', display_name: '球友 42AF10',
          status: 'confirmed', role: 'blue',
        },
      ],
    })
})


test('opening one table session is idempotent and another active session conflicts', async () => {
  const db = configuredDatabase()
  const admin = createJoinAdmin(db, { nowIso: () => '2026-08-27T10:00:00.000Z' })

  await expect(admin.openSession({ sessionId: 'session-1', tableId: 'table-1' }))
    .resolves.toMatchObject({ session_id: 'session-1', table_id: 'table-1', status: 'open' })
  await expect(admin.openSession({ sessionId: 'session-1', tableId: 'table-1' }))
    .resolves.toMatchObject({ session_id: 'session-1', status: 'open' })
  await expect(admin.openSession({ sessionId: 'session-2', tableId: 'table-1' }))
    .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })

  expect(db.collections.get('table_join_codes').get('table-1').current_session_id)
    .toBe('session-1')
  expect(db.collections.get('field_live_sessions').get('session-1'))
    .not.toHaveProperty('source_kind', 'live_capture')
})


test('opening a new session handles CloudBase missing-document reads', async () => {
  const db = configuredDatabase({ throwOnMissing: true })
  const admin = createJoinAdmin(db, { nowIso: () => '2026-08-27T10:00:00.000Z' })

  await expect(admin.openSession({ sessionId: 'session-new', tableId: 'table-1' }))
    .resolves.toMatchObject({ session_id: 'session-new', table_id: 'table-1', status: 'open' })
})


test('reopening and closing a session never writes CloudBase document metadata', async () => {
  const db = configuredDatabase({ cloudDocumentIds: true, throwOnMissing: true })
  const admin = createJoinAdmin(db, { nowIso: () => '2026-08-30T15:20:00.000Z' })

  await admin.openSession({ sessionId: 'session-1', tableId: 'table-1' })
  await expect(admin.openSession({ sessionId: 'session-1', tableId: 'table-1' }))
    .resolves.toMatchObject({ session_id: 'session-1', status: 'open' })
  await expect(admin.closeSession({ sessionId: 'session-1', tableId: 'table-1' }))
    .resolves.toMatchObject({ session_id: 'session-1', status: 'ended' })
})


test('close clears only the matching current session and repeated close is idempotent', async () => {
  const db = configuredDatabase()
  const admin = createJoinAdmin(db, { nowIso: () => '2026-08-27T10:00:00.000Z' })
  await admin.openSession({ sessionId: 'session-1', tableId: 'table-1' })

  const first = await admin.closeSession({ sessionId: 'session-1', tableId: 'table-1' })
  const second = await admin.closeSession({ sessionId: 'session-1', tableId: 'table-1' })

  expect(first).toMatchObject({ session_id: 'session-1', status: 'ended' })
  expect(second).toEqual(first)
  expect(db.collections.get('table_join_codes').get('table-1').current_session_id)
    .toBeNull()
})


test('close repairs a stale table binding when the session is already ended', async () => {
  const db = createDatabase({
    table_join_codes: {
      'table-1': {
        schema_version: '1.0', table_id: 'table-1', status: 'active',
        current_session_id: 'session-ended',
      },
    },
    field_live_sessions: {
      'session-ended': {
        schema_version: '1.0', session_id: 'session-ended', table_id: 'table-1',
        status: 'ended', ended_at: '2026-08-30T09:15:18.710Z',
      },
    },
  })
  const admin = createJoinAdmin(db, { nowIso: () => '2026-08-30T15:20:00.000Z' })

  await expect(admin.closeSession({ sessionId: 'session-ended', tableId: 'table-1' }))
    .resolves.toMatchObject({ session_id: 'session-ended', status: 'ended' })
  expect(db.collections.get('table_join_codes').get('table-1').current_session_id)
    .toBeNull()
})


test('opening a new session replaces a stale binding to an ended session', async () => {
  const db = createDatabase({
    table_join_codes: {
      'table-1': {
        schema_version: '1.0', table_id: 'table-1', status: 'active',
        current_session_id: 'session-ended',
      },
    },
    field_live_sessions: {
      'session-ended': {
        schema_version: '1.0', session_id: 'session-ended', table_id: 'table-1',
        status: 'ended', ended_at: '2026-08-30T09:15:18.710Z',
      },
    },
  })
  const admin = createJoinAdmin(db, { nowIso: () => '2026-08-30T15:20:00.000Z' })

  await expect(admin.openSession({ sessionId: 'session-new', tableId: 'table-1' }))
    .resolves.toMatchObject({ session_id: 'session-new', table_id: 'table-1', status: 'open' })
  expect(db.collections.get('table_join_codes').get('table-1').current_session_id)
    .toBe('session-new')
})


test('missing table configuration and wrong table scope fail closed', async () => {
  const db = configuredDatabase()
  const admin = createJoinAdmin(db)
  await admin.openSession({ sessionId: 'session-1', tableId: 'table-1' })

  await expect(admin.openSession({ sessionId: 'session-2', tableId: 'table-2' }))
    .rejects.toMatchObject({ code: 'SYNC_CONFIGURATION_REQUIRED' })
  await expect(admin.closeSession({ sessionId: 'session-1', tableId: 'table-2' }))
    .rejects.toMatchObject({ code: 'SYNC_UNAUTHORIZED' })
})
