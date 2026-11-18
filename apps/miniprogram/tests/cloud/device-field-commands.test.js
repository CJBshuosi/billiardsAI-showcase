const {
  claimNextFieldCommand,
  completeFieldCommand,
  createCloudDeviceFieldCommandRepository,
  pickClaimCandidate,
  projectSessionAfterApplied,
} = require('../../cloudfunctions/sync-api/lib/field-commands')


test('restart resumes this device claimed command before later pending work', () => {
  expect(pickClaimCandidate(
    [{ command_id: 'pending-8', server_sequence: 8, status: 'pending' }],
    [{ command_id: 'claimed-7', server_sequence: 7, status: 'claimed', claimed_by_device_id: 'edge-1' }],
    'edge-1',
  )).toMatchObject({ command_id: 'claimed-7' })
})


test('device claims the lowest pending sequence for its own table only', async () => {
  const repo = {
    async claimNext({ tableId, deviceId, now }) {
      expect({ tableId, deviceId, now }).toEqual({
        tableId: 'table-1', deviceId: 'edge-1', now: '2026-08-27T10:00:00.000Z',
      })
      return {
        schema_version: '1.0', command_id: 'command-7', session_id: 'session-1',
        command_type: 'switch_turn', validated_payload: {}, expected_revision: 6,
        server_sequence: 7, status: 'claimed',
      }
    },
  }
  await expect(claimNextFieldCommand({
    tableId: 'table-1', deviceId: 'edge-1', now: '2026-08-27T10:00:00.000Z', repo,
  })).resolves.toMatchObject({ server_sequence: 7, status: 'claimed' })
})


test('applied commands project a public cloud mirror while proposals remain no-ops', () => {
  const session = {
    status: 'open', red_user_id: 'user-red', blue_user_id: 'user-blue',
    active_user_id: 'user-red', red_group: 'stripe', blue_group: 'solid',
  }
  expect(projectSessionAfterApplied(session, {
    command_type: 'claim_role', actor_user_id: 'user-red',
    validated_payload: { role: 'red' },
  }, 'now')).toMatchObject({ red_user_id: 'user-red' })
  expect(projectSessionAfterApplied(session, {
    command_type: 'switch_turn', actor_user_id: 'user-red', validated_payload: {},
  }, 'now')).toMatchObject({ active_user_id: 'user-blue' })
  expect(projectSessionAfterApplied(session, {
    command_type: 'set_breaker', actor_user_id: 'user-red',
    validated_payload: {
      role: 'red', mode: 'observation', format: 'single', race_target: null,
      ranking_intent: true, settings_revision: 2,
    },
  }, 'now')).toMatchObject({
    breaker: 'red', mode: 'observation', format: 'single', race_target: null,
    ranking_intent: true, settings_revision: 2,
  })
  expect(projectSessionAfterApplied(session, {
    command_type: 'assign_groups', actor_user_id: 'user-red',
    validated_payload: { red_group: 'solid', correction_phase: 'proposal' },
  }, 'now')).toEqual(session)
})


test('empty claim is explicit and does not invent a command', async () => {
  const repo = { claimNext: jest.fn().mockResolvedValue(null) }
  await expect(claimNextFieldCommand({
    tableId: 'table-1', deviceId: 'edge-1', now: 'now', repo,
  })).resolves.toBeNull()
})


test('completion is idempotent and accepts only stable terminal fields', async () => {
  const repo = {
    complete: jest.fn().mockResolvedValue({
      schema_version: '1.0', command_id: 'command-7', status: 'applied',
      public_result_code: 'APPLIED', resulting_revision: 7,
    }),
  }
  const input = {
    commandId: 'command-7', tableId: 'table-1', deviceId: 'edge-1',
    status: 'applied', publicResultCode: 'APPLIED', resultingRevision: 7,
    now: '2026-08-27T10:00:01.000Z', repo,
  }
  const first = await completeFieldCommand(input)
  const second = await completeFieldCommand(input)
  expect(second).toEqual(first)
  await expect(completeFieldCommand({ ...input, status: 'pending' }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
  await expect(completeFieldCommand({ ...input, resultingRevision: -1 }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
})


test('applied role claim confirms the matching join request and binding atomically', async () => {
  const crypto = require('crypto')
  const roleId = crypto.createHash('sha256').update('session-1:red').digest('hex')
  const bindingId = crypto.createHash('sha256').update('session-1:user-red').digest('hex')
  const collections = new Map([
    ['field_session_commands', new Map([['command-1', {
      _id: 'command-1', schema_version: '1.0', command_id: 'command-1', session_id: 'session-1',
      table_id: 'table-1', actor_user_id: 'user-red', command_type: 'claim_role',
      validated_payload: { role: 'red' }, expected_revision: 0, server_sequence: 1,
      status: 'claimed', claimed_by_device_id: 'edge-1',
    }]])],
    ['field_live_sessions', new Map([['session-1', {
      _id: 'session-1', schema_version: '1.0', session_id: 'session-1', table_id: 'table-1',
      status: 'open', control_revision: 0, red_user_id: null, blue_user_id: null,
    }]])],
    ['session_user_bindings', new Map([[bindingId, {
      _id: bindingId, schema_version: '1.0', session_id: 'session-1', table_id: 'table-1',
      user_id: 'user-red', request_id: 'join-1', role: null,
    }]])],
    ['session_join_requests', new Map([['join-1', {
      _id: 'join-1', schema_version: '1.0', request_id: 'join-1', session_id: 'session-1',
      table_id: 'table-1', user_id: 'user-red', status: 'pending', role: null,
    }]])],
    ['session_role_bindings', new Map()],
    ['session_role_binding_history', new Map()],
  ])
  const collection = (name) => ({
    doc(id) {
      return {
        async get() {
          const data = collections.get(name).get(id)
          if (!data) throw new Error(`document.get:fail document with _id ${id} does not exist`)
          return { data }
        },
        async set({ data }) {
          if (Object.hasOwn(data, '_id')) throw new Error('database set failed: _id is reserved')
          collections.get(name).set(id, data)
        },
        async update({ data }) {
          collections.get(name).set(id, { ...collections.get(name).get(id), ...data })
        },
      }
    },
  })
  const db = {
    async runTransaction(callback) { return callback({ collection }) },
  }
  const repo = createCloudDeviceFieldCommandRepository(db)

  await expect(completeFieldCommand({
    commandId: 'command-1', tableId: 'table-1', deviceId: 'edge-1',
    status: 'applied', publicResultCode: 'APPLIED', resultingRevision: 1,
    now: '2026-08-30T15:00:00.000Z', repo,
  })).resolves.toMatchObject({ status: 'applied', resulting_revision: 1 })
  expect(collections.get('field_live_sessions').get('session-1'))
    .toMatchObject({ red_user_id: 'user-red', control_revision: 1 })
  expect(collections.get('session_role_bindings').get(roleId))
    .toMatchObject({ user_id: 'user-red', request_id: 'join-1', role: 'red' })
  expect(collections.get('session_user_bindings').get(bindingId))
    .toMatchObject({ role: 'red' })
  expect(collections.get('session_join_requests').get('join-1'))
    .toMatchObject({ status: 'confirmed', role: 'red' })
})
