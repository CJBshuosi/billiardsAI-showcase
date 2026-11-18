const {
  createCloudFieldCommandRepository,
  getFieldCommandStatus,
  getCurrentField,
  submitFieldCommand,
} = require('../../cloudfunctions/client-api/lib/field-commands')


const base = {
  actorUserId: 'user-red',
  sessionId: 'session-1',
  commandType: 'claim_role',
  payload: { role: 'red' },
  expectedRevision: 3,
  idempotencyKey: 'intent-000000000001',
  now: '2026-08-27T10:00:00.000Z',
}


function repository(context = {}) {
  const commands = new Map()
  return {
    commands,
    async getAuthorizationContext() {
      return {
        session_id: 'session-1', table_id: 'table-1', status: 'open', revision: 3,
        started: false, joined: true, red_user_id: null, blue_user_id: null,
        red_group: null, blue_group: null, ...context,
      }
    },
    async getPendingGroupCorrection() {
      return [...commands.values()].find((item) => (
        item.command_type === 'assign_groups'
        && item.validated_payload.correction_phase === 'proposal'
        && !['rejected', 'expired'].includes(item.status)
      )) || null
    },
    async getRankingMatchContext() {
      return {
        status: 'ready',
        settings: {
          settings_revision: 2, mode: 'observation', format: 'single', race_target: null,
          first_breaker: 'red', ranking_intent: true,
        },
      }
    },
    async createCommand(command) {
      const key = `${command.session_id}:${command.actor_user_id}:${command.idempotency_digest}`
      const previous = commands.get(key)
      if (previous) {
        if (previous.intent_digest !== command.intent_digest) {
          const error = Object.assign(new Error('conflict'), {
            code: 'COMMAND_IDEMPOTENCY_CONFLICT',
          })
          throw error
        }
        return previous
      }
      const stored = { ...command, command_id: 'command-1', server_sequence: 7 }
      commands.set(key, stored)
      return stored
    },
    async getCommand({ commandId, actorUserId }) {
      const found = [...commands.values()].find((item) => item.command_id === commandId)
      return found && found.actor_user_id === actorUserId ? found : null
    },
  }
}


test('same idempotency key reuses one command and a changed payload conflicts', async () => {
  const repo = repository()
  const first = await submitFieldCommand(base, repo)
  const second = await submitFieldCommand(base, repo)
  expect(second.command_id).toBe(first.command_id)
  await expect(submitFieldCommand({ ...base, payload: { role: 'blue' } }, repo))
    .rejects.toMatchObject({ code: 'COMMAND_IDEMPOTENCY_CONFLICT' })
})


test('closed schema rejects extra fields, stale revisions, third users, and ended sessions', async () => {
  await expect(submitFieldCommand({ ...base, payload: { role: 'red', user_id: 'forged' } }, repository()))
    .rejects.toMatchObject({ code: 'COMMAND_INVALID' })
  await expect(submitFieldCommand({ ...base, expectedRevision: 2 }, repository()))
    .rejects.toMatchObject({ code: 'COMMAND_STALE_REVISION' })
  await expect(submitFieldCommand({ ...base, commandType: 'continue_turn', payload: {} }, repository()))
    .rejects.toMatchObject({ code: 'COMMAND_FORBIDDEN' })
  await expect(submitFieldCommand(base, repository({ joined: false })))
    .rejects.toMatchObject({ code: 'COMMAND_FORBIDDEN' })
  await expect(submitFieldCommand(base, repository({ status: 'ended' })))
    .rejects.toMatchObject({ code: 'SESSION_NOT_JOINABLE' })
})


test('participant permissions freeze setup and bind turn commands to the active actor', async () => {
  const started = repository({
    started: true, red_user_id: 'user-red', blue_user_id: 'user-blue',
    active_user_id: 'user-red', red_group: 'stripe', blue_group: 'solid',
  })
  await expect(submitFieldCommand({ ...base, commandType: 'release_role', payload: {} }, started))
    .rejects.toMatchObject({ code: 'COMMAND_FORBIDDEN' })
  await expect(submitFieldCommand({ ...base, commandType: 'continue_turn', payload: {} }, started))
    .resolves.toMatchObject({ command_type: 'continue_turn', status: 'pending' })
  await expect(submitFieldCommand({
    ...base, actorUserId: 'user-blue', commandType: 'continue_turn', payload: {},
  }, started)).rejects.toMatchObject({ code: 'COMMAND_FORBIDDEN' })
})

test('breaker command is accepted only after both ranking settings confirmations', async () => {
  const ready = repository({ red_user_id: 'user-red', blue_user_id: 'user-blue' })
  await expect(submitFieldCommand({
    ...base, commandType: 'set_breaker', payload: { role: 'red' },
  }, ready)).resolves.toMatchObject({ command_type: 'set_breaker' })
  expect([...ready.commands.values()][0].validated_payload).toEqual({
    role: 'red', settings_revision: 2, mode: 'observation', format: 'single',
    race_target: null, ranking_intent: true,
  })

  ready.getRankingMatchContext = async () => ({
    status: 'awaiting_settings_confirmations',
    settings: { first_breaker: 'red' },
  })
  await expect(submitFieldCommand({
    ...base, commandType: 'set_breaker', payload: { role: 'red' },
  }, ready)).rejects.toMatchObject({ code: 'MATCH_SETTINGS_NOT_READY' })
})


test('public command status excludes table, actor, idempotency, and validated internals', async () => {
  const repo = repository()
  const submitted = await submitFieldCommand(base, repo)
  const status = await getFieldCommandStatus({
    commandId: submitted.command_id, actorUserId: 'user-red', repo,
  })
  expect(status).toEqual({
    schema_version: '1.0', command_id: 'command-1', command_type: 'claim_role',
    status: 'pending', server_sequence: 7, public_result_code: null,
    resulting_revision: null,
  })
  expect(JSON.stringify(status)).not.toMatch(/table-1|user-red|idempotency|validated_payload/u)
})


test('current field exposes role occupancy and revision without internal user ids', async () => {
  const value = await getCurrentField({
    sessionId: 'session-1', actorUserId: 'user-red',
    repo: repository({
      red_user_id: 'user-red', blue_user_id: 'user-blue',
      active_user_id: 'user-red', breaker: 'red',
    }),
  })
  expect(value).toMatchObject({
    actor_role: 'red', active_role: 'red', red_occupied: true,
    blue_occupied: true, revision: 3,
  })
  expect(JSON.stringify(value)).not.toMatch(/user-red|user-blue|table-1/u)
})


test('current field tolerates missing red and blue role documents before roles are claimed', async () => {
  const documents = new Map([
    ['field_live_sessions:session-1', {
      session_id: 'session-1', table_id: 'table-1', status: 'open', control_revision: 0,
    }],
    ['session_user_bindings:binding-1', {
      session_id: 'session-1', table_id: 'table-1', user_id: 'user-red',
    }],
  ])
  const crypto = require('crypto')
  const bindingId = crypto.createHash('sha256')
    .update('session-1:user-red', 'utf8').digest('hex')
  documents.set(`session_user_bindings:${bindingId}`, documents.get('session_user_bindings:binding-1'))
  const db = {
    collection(name) {
      return {
        doc(id) {
          return {
            async get() {
              const data = documents.get(`${name}:${id}`)
              if (!data) throw new Error(`document.get:fail document with _id ${id} does not exist`)
              return { data }
            },
          }
        },
      }
    },
  }
  const repo = createCloudFieldCommandRepository(db)

  await expect(getCurrentField({
    sessionId: 'session-1', actorUserId: 'user-red', repo,
  })).resolves.toMatchObject({
    actor_role: null, red_occupied: false, blue_occupied: false, revision: 0,
  })
})


test('changing frozen groups requires one participant proposal and the other confirmation', async () => {
  const repo = repository({
    red_user_id: 'user-red', blue_user_id: 'user-blue',
    red_group: 'stripe', blue_group: 'solid',
  })
  const proposal = await submitFieldCommand({
    ...base, commandType: 'assign_groups', payload: { red_group: 'solid' },
  }, repo)
  expect(proposal).toMatchObject({ command_type: 'assign_groups', status: 'pending' })
  await expect(submitFieldCommand({
    ...base, idempotencyKey: 'intent-000000000002', commandType: 'assign_groups',
    payload: { red_group: 'solid' },
  }, repo)).rejects.toMatchObject({ code: 'COMMAND_FORBIDDEN' })
  await expect(submitFieldCommand({
    ...base, actorUserId: 'user-blue', idempotencyKey: 'intent-000000000003',
    commandType: 'assign_groups', payload: { red_group: 'solid' },
  }, repo)).resolves.toMatchObject({ command_type: 'assign_groups', status: 'pending' })
})
