const {
  getFieldCommandStatus,
  submitFieldCommand,
} = require('../../cloudfunctions/client-api/lib/field-commands')
const {
  claimNextFieldCommand,
  completeFieldCommand,
  projectSessionAfterApplied,
} = require('../../cloudfunctions/sync-api/lib/field-commands')


function loopRepository() {
  const commands = new Map()
  let sequence = 1
  let session = {
    session_id: 'session-loop', table_id: 'table-loop', status: 'open', revision: 0,
    started: false, joined: true, red_user_id: null, blue_user_id: null,
    red_group: null, blue_group: null, breaker: null, active_user_id: null,
  }
  return {
    async getAuthorizationContext() { return { ...session } },
    async getPendingGroupCorrection() { return null },
    async createCommand(command) {
      const key = `${command.session_id}:${command.actor_user_id}:${command.idempotency_digest}`
      const existing = commands.get(key)
      if (existing) return existing
      const stored = {
        ...command, command_id: `command-${sequence}`, server_sequence: sequence++,
      }
      commands.set(key, stored)
      return stored
    },
    async getCommand({ commandId, actorUserId }) {
      return [...commands.values()].find((item) => (
        item.command_id === commandId && item.actor_user_id === actorUserId
      )) || null
    },
    async claimNext({ tableId, deviceId, now }) {
      const candidate = [...commands.values()]
        .filter((item) => item.table_id === tableId && (item.status === 'pending'
          || (item.status === 'claimed' && item.claimed_by_device_id === deviceId)))
        .sort((left, right) => left.server_sequence - right.server_sequence)[0]
      if (!candidate) return null
      Object.assign(candidate, {
        status: 'claimed', claimed_by_device_id: deviceId, claimed_at: now,
      })
      return candidate
    },
    async complete({
      commandId, deviceId, status, publicResultCode, resultingRevision, now,
    }) {
      const command = [...commands.values()]
        .find((item) => item.command_id === commandId)
      if (command.status === 'applied' || command.status === 'rejected') return command
      Object.assign(command, {
        status, public_result_code: publicResultCode,
        resulting_revision: resultingRevision, completed_at: now,
      })
      if (status === 'applied') {
        session = {
          ...projectSessionAfterApplied(session, command, now),
          revision: resultingRevision,
        }
      }
      expect(command.claimed_by_device_id).toBe(deviceId)
      return command
    },
  }
}


async function submit(repo, {
  actorUserId, type, payload, revision, key,
}) {
  return submitFieldCommand({
    actorUserId, sessionId: 'session-loop', commandType: type, payload,
    expectedRevision: revision, idempotencyKey: key,
    now: '2026-08-27T10:00:00.000Z',
  }, repo)
}


test('two phones converge through ordered claim, completion, and idempotent retry', async () => {
  const repo = loopRepository()
  const red = await submit(repo, {
    actorUserId: 'user-red', type: 'claim_role', payload: { role: 'red' },
    revision: 0, key: 'intent-red-00000001',
  })
  expect((await submit(repo, {
    actorUserId: 'user-red', type: 'claim_role', payload: { role: 'red' },
    revision: 0, key: 'intent-red-00000001',
  })).command_id).toBe(red.command_id)
  const claimedRed = await claimNextFieldCommand({
    tableId: 'table-loop', deviceId: 'raytheon-01',
    now: '2026-08-27T10:00:01.000Z', repo,
  })
  await completeFieldCommand({
    commandId: claimedRed.command_id, tableId: 'table-loop', deviceId: 'raytheon-01',
    status: 'applied', publicResultCode: 'APPLIED', resultingRevision: 1,
    now: '2026-08-27T10:00:02.000Z', repo,
  })
  const blue = await submit(repo, {
    actorUserId: 'user-blue', type: 'claim_role', payload: { role: 'blue' },
    revision: 1, key: 'intent-blue-0000001',
  })
  const claimedBlue = await claimNextFieldCommand({
    tableId: 'table-loop', deviceId: 'raytheon-01',
    now: '2026-08-27T10:00:03.000Z', repo,
  })
  await completeFieldCommand({
    commandId: claimedBlue.command_id, tableId: 'table-loop', deviceId: 'raytheon-01',
    status: 'applied', publicResultCode: 'APPLIED', resultingRevision: 2,
    now: '2026-08-27T10:00:04.000Z', repo,
  })

  await expect(getFieldCommandStatus({
    commandId: red.command_id, actorUserId: 'user-red', repo,
  })).resolves.toMatchObject({ status: 'applied', resulting_revision: 1 })
  await expect(getFieldCommandStatus({
    commandId: blue.command_id, actorUserId: 'user-blue', repo,
  })).resolves.toMatchObject({ status: 'applied', resulting_revision: 2 })
  await expect(claimNextFieldCommand({
    tableId: 'table-loop', deviceId: 'raytheon-01',
    now: '2026-08-27T10:00:05.000Z', repo,
  })).resolves.toBeNull()
})


test('concurrent same-role intents end with one applied and one stable rejection', async () => {
  const repo = loopRepository()
  const first = await submit(repo, {
    actorUserId: 'user-first', type: 'claim_role', payload: { role: 'red' },
    revision: 0, key: 'intent-first-000001',
  })
  const second = await submit(repo, {
    actorUserId: 'user-second', type: 'claim_role', payload: { role: 'red' },
    revision: 0, key: 'intent-second-00001',
  })
  const claimedFirst = await claimNextFieldCommand({
    tableId: 'table-loop', deviceId: 'raytheon-01', now: 'now-1', repo,
  })
  await completeFieldCommand({
    commandId: claimedFirst.command_id, tableId: 'table-loop', deviceId: 'raytheon-01',
    status: 'applied', publicResultCode: 'APPLIED', resultingRevision: 1,
    now: 'now-2', repo,
  })
  const claimedSecond = await claimNextFieldCommand({
    tableId: 'table-loop', deviceId: 'raytheon-01', now: 'now-3', repo,
  })
  await completeFieldCommand({
    commandId: claimedSecond.command_id, tableId: 'table-loop', deviceId: 'raytheon-01',
    status: 'rejected', publicResultCode: 'STALE_REVISION', resultingRevision: 1,
    now: 'now-4', repo,
  })

  await expect(getFieldCommandStatus({
    commandId: first.command_id, actorUserId: 'user-first', repo,
  })).resolves.toMatchObject({ status: 'applied' })
  await expect(getFieldCommandStatus({
    commandId: second.command_id, actorUserId: 'user-second', repo,
  })).resolves.toMatchObject({ status: 'rejected', public_result_code: 'STALE_REVISION' })
})
