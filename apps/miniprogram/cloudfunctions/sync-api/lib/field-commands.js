const crypto = require('crypto')

const { SyncApiError } = require('./auth')
const { digest, documentId } = require('../../shared/shot-ownership')
const { isManagedSessionId, readManagedCommandContext, validManagedCommand,
  validManagedLive, validManagedHead, validManagedRoleLink, managedRoleHistoryId } = require('../../shared/managed-command-authority')


const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u
const RESULT_PATTERN = /^[A-Z][A-Z0-9_]{1,63}$/u
const TERMINAL = new Set(['applied', 'rejected'])


function withoutDocumentId(document) {
  if (!document || typeof document !== 'object') return document
  const { _id: _documentId, ...data } = document
  return data
}


async function getOptionalDocument(reference) {
  try {
    return await reference.get()
  } catch (error) {
    const message = error && error.message ? error.message : String(error)
    if (/document with _id .+ does not exist/i.test(message)) return { data: null }
    throw error
  }
}


function pickClaimCandidate(pending, claimed, deviceId) {
  return [...(pending || []), ...(claimed || [])]
    .filter((item) => item && (item.status === 'pending'
      || (item.status === 'claimed' && item.claimed_by_device_id === deviceId)))
    .sort((left, right) => left.server_sequence - right.server_sequence)[0] || null
}


function projectSessionAfterApplied(session, command, now) {
  const payload = command.validated_payload || {}
  if (command.command_type === 'assign_groups' && payload.correction_phase === 'proposal') {
    return session
  }
  const updated = { ...session, updated_at: now }
  if (command.command_type === 'claim_role') {
    updated[`${payload.role}_user_id`] = command.actor_user_id
  } else if (command.command_type === 'release_role') {
    if (updated.red_user_id === command.actor_user_id) updated.red_user_id = null
    if (updated.blue_user_id === command.actor_user_id) updated.blue_user_id = null
  } else if (command.command_type === 'set_breaker') {
    updated.breaker = payload.role
    updated.active_user_id = updated[`${payload.role}_user_id`] || null
    updated.started_at = updated.started_at || now
    updated.mode = payload.mode || null
    updated.format = payload.format || null
    updated.race_target = Number.isInteger(payload.race_target) ? payload.race_target : null
    updated.ranking_intent = payload.ranking_intent === true
    updated.settings_revision = Number.isInteger(payload.settings_revision)
      ? payload.settings_revision : 0
  } else if (command.command_type === 'assign_groups') {
    updated.red_group = payload.red_group
    updated.blue_group = payload.red_group === 'solid' ? 'stripe' : 'solid'
  } else if (command.command_type === 'continue_turn') {
    updated.started_at = updated.started_at || now
  } else if (command.command_type === 'switch_turn') {
    updated.started_at = updated.started_at || now
    updated.active_user_id = updated.active_user_id === updated.red_user_id
      ? updated.blue_user_id : updated.red_user_id
  } else if (command.command_type === 'end_session') {
    updated.status = 'ended'
    updated.ended_at = now
  }
  return updated
}


async function claimNextFieldCommand({ tableId, deviceId, now, repo }) {
  if (!ID_PATTERN.test(tableId || '') || !ID_PATTERN.test(deviceId || '')
      || typeof now !== 'string') {
    throw new SyncApiError('SYNC_INVALID')
  }
  return repo.claimNext({ tableId, deviceId, now })
}


async function completeFieldCommand({
  commandId, tableId, deviceId, status, publicResultCode, resultingRevision, now, repo,
}) {
  if (!ID_PATTERN.test(commandId || '') || !ID_PATTERN.test(tableId || '')
      || !ID_PATTERN.test(deviceId || '') || !TERMINAL.has(status)
      || !RESULT_PATTERN.test(publicResultCode || '')
      || !Number.isSafeInteger(resultingRevision) || resultingRevision < 0
      || typeof now !== 'string') {
    throw new SyncApiError('SYNC_INVALID')
  }
  return repo.complete({
    commandId, tableId, deviceId, status, publicResultCode, resultingRevision, now,
  })
}


const DURABLE_REJECTIONS = new Set(['SESSION_ENDED', 'STALE_REVISION',
  'TURN_CONFIRMATION_MISMATCH', 'COMMAND_FORBIDDEN', 'COMMAND_INVALID', 'SHOT_UNSETTLED'])
function conflictUnless(condition) {
  if (!condition) throw new SyncApiError('SYNC_CONFLICT')
}
async function managedContext(tx, command, tableId, deviceId) {
  const context = await readManagedCommandContext(tx, { sessionId: command.session_id, tableId, deviceId })
  conflictUnless(context && validManagedCommand(command, context.birth, context.head))
  conflictUnless(context.head.coverage === 'open' || context.live.status === 'ended')
  return context
}
function requireNext(command, head) {
  conflictUnless(head.coverage !== 'sealed' && command.server_sequence === head.contiguous_ack_sequence + 1)
}
function validResult(command, status, code) {
  return status === 'rejected' ? DURABLE_REJECTIONS.has(code)
    : status === 'applied' && (code === 'APPLIED' || (code === 'GROUP_CORRECTION_PROPOSED'
      && command.command_type === 'assign_groups' && command.validated_payload.correction_phase === 'proposal'))
}
async function readLink(tx, id, context, role) {
  const node = (await getOptionalDocument(tx.collection('session_role_binding_history').doc(id))).data
  conflictUnless(validManagedRoleLink(node, id, context.birth, context.head, role))
  return node
}
async function replayManagedRole(tx, command, context) {
  const role = command.validated_payload.role
  conflictUnless(['red', 'blue'].includes(role) && Object.keys(command.validated_payload).length === 1
    && command.applied_role === role && command.public_result_code === 'APPLIED')
  const id = managedRoleHistoryId({ ...command, role })
  const node = await readLink(tx, id, context, role)
  conflictUnless(Object.hasOwn(command, 'applied_previous_history_id')
    && node.previous_id === command.applied_previous_history_id
    && node.command_id === command.command_id && node.server_sequence === command.server_sequence
    && node.user_id === (command.command_type === 'claim_role' ? command.actor_user_id : null))
  if (node.previous_id !== null) {
    const previous = await readLink(tx, node.previous_id, context, role)
    conflictUnless(previous.next_id === id && previous.server_sequence < node.server_sequence
      && previous.resulting_revision < node.resulting_revision)
  }
  if (node.next_id !== null) {
    const next = await readLink(tx, node.next_id, context, role)
    conflictUnless(next.previous_id === id && next.server_sequence > node.server_sequence
      && next.resulting_revision > node.resulting_revision)
  } else conflictUnless(context.head.role_tails[role] === id)
}
async function appendManagedRole(tx, command, completed, context, now) {
  const { birth, head, live } = context
  const role = command.validated_payload.role
  const claim = command.command_type === 'claim_role'
  conflictUnless(['red', 'blue'].includes(role) && Object.keys(command.validated_payload).length === 1)
  conflictUnless(!claim || ![live.red_user_id, live.blue_user_id].includes(command.actor_user_id))
  conflictUnless(claim ? !live[`${role}_user_id`] : live[`${role}_user_id`] === command.actor_user_id)
  const id = managedRoleHistoryId({ ...completed, role })
  const historyRef = tx.collection('session_role_binding_history').doc(id)
  conflictUnless(!(await getOptionalDocument(historyRef)).data)
  const previousId = head.role_tails[role]
  let previous = null
  if (previousId !== null) {
    previous = await readLink(tx, previousId, context, role)
    conflictUnless(previous.next_id === null && previous.server_sequence < command.server_sequence
      && previous.resulting_revision < completed.resulting_revision
      && previous.user_id === (live[`${role}_user_id`] || null))
  } else conflictUnless(!live[`${role}_user_id`])
  const bindingRef = tx.collection('session_user_bindings').doc(documentId(`${birth.session_id}:${command.actor_user_id}`))
  const binding = (await getOptionalDocument(bindingRef)).data
  conflictUnless(binding && binding.session_id === birth.session_id && binding.table_id === birth.table_id
    && binding.user_id === command.actor_user_id && (claim ? binding.role === null : binding.role === role))
  const roleRef = tx.collection('session_role_bindings').doc(documentId(`${birth.session_id}:${role}`))
  const roleBinding = (await getOptionalDocument(roleRef)).data
  conflictUnless(claim ? !roleBinding : roleBinding && roleBinding.session_id === birth.session_id
    && roleBinding.role === role && roleBinding.user_id === command.actor_user_id)
  const requestId = binding.request_id
  let requestRef
  if (requestId != null) {
    conflictUnless(ID_PATTERN.test(requestId))
    requestRef = tx.collection('session_join_requests').doc(requestId)
    const request = (await getOptionalDocument(requestRef)).data
    conflictUnless(request && request.session_id === birth.session_id && request.user_id === command.actor_user_id)
  }
  await historyRef.set({ data: { contract: 'managed_role_link_v1', ...birthIdentity(birth), role,
    user_id: claim ? command.actor_user_id : null, resulting_revision: completed.resulting_revision,
    command_id: command.command_id, server_sequence: command.server_sequence, previous_id: previousId, next_id: null } })
  if (previous) await tx.collection('session_role_binding_history').doc(previousId).update({ data: { next_id: id } })
  if (claim) await roleRef.set({ data: { schema_version: '1.0', session_id: birth.session_id,
    user_id: command.actor_user_id, role, request_id: requestId || null, operator_id: birth.device_id, confirmed_at: now } })
  else await roleRef.remove()
  await bindingRef.set({ data: { ...withoutDocumentId(binding), role: claim ? role : null, updated_at: now } })
  if (requestRef) await requestRef.update({ data: { status: claim ? 'confirmed' : 'pending', role: claim ? role : null, updated_at: now } })
  completed.applied_role = role
  completed.applied_previous_history_id = previousId
  return { ...head.role_tails, [role]: id }
}
function birthIdentity(birth) {
  return { session_id: birth.session_id, birth_id: birth.birth_id, table_id: birth.table_id, device_id: birth.device_id }
}
async function completeManagedCommand(tx, command, input) {
  const { commandId, tableId, deviceId, status, publicResultCode, resultingRevision, now } = input
  const context = await managedContext(tx, command, tableId, deviceId)
  const { head, live } = context
  conflictUnless(command.command_id === commandId && validResult(command, status, publicResultCode)
    && Number.isSafeInteger(resultingRevision) && resultingRevision >= 0
    && typeof now === 'string' && !Number.isNaN(Date.parse(now)) && new Date(now).toISOString() === now)
  const roleChange = status === 'applied' && ['claim_role', 'release_role'].includes(command.command_type)
  if (TERMINAL.has(command.status)) {
    conflictUnless(command.status === status && command.public_result_code === publicResultCode
      && command.resulting_revision === resultingRevision && command.server_sequence <= head.contiguous_ack_sequence
      && resultingRevision <= head.last_resulting_revision
      && (status !== 'applied' || command.server_sequence <= head.last_applied_server_sequence))
    if (roleChange) await replayManagedRole(tx, command, context)
    return command
  }
  requireNext(command, head)
  conflictUnless(command.status === 'claimed' && head.revision < Number.MAX_SAFE_INTEGER
    && resultingRevision >= head.last_resulting_revision
    && (!roleChange || resultingRevision > head.last_resulting_revision))
  const completed = { ...withoutDocumentId(command), status, public_result_code: publicResultCode,
    resulting_revision: resultingRevision, completed_at: now }
  const roleTails = roleChange ? await appendManagedRole(tx, command, completed, context, now) : head.role_tails
  const projected = status === 'applied' ? projectSessionAfterApplied(withoutDocumentId(live), command, now) : withoutDocumentId(live)
  // Close fixes the lifecycle end time; draining a prior end command cannot change it.
  if (live.status === 'ended') { projected.status = 'ended'; projected.ended_at = live.ended_at }
  if (roleChange && command.command_type === 'release_role' && projected.active_user_id === command.actor_user_id) projected.active_user_id = null
  const nextLive = { ...projected, control_revision: resultingRevision, updated_at: now }
  const sealed = head.coverage === 'closing' && command.server_sequence === head.closed_target
  const nextHead = { ...withoutDocumentId(head), contiguous_ack_sequence: command.server_sequence,
    last_resulting_revision: resultingRevision,
    last_applied_server_sequence: status === 'applied' ? command.server_sequence : head.last_applied_server_sequence,
    role_tails: roleTails, revision: head.revision + 1,
    coverage: sealed ? 'sealed' : head.coverage, seal: sealed ? { sealed_at: now } : head.seal }
  conflictUnless(validManagedHead(nextHead) && validManagedLive(nextLive))
  await tx.collection('field_session_commands').doc(commandId).set({ data: completed })
  await tx.collection('field_live_sessions').doc(command.session_id).set({ data: nextLive })
  await tx.collection('session_command_authority').doc(command.session_id).set({ data: nextHead })
  return completed
}

function createCloudDeviceFieldCommandRepository(db) {
  return {
    async claimNext({ tableId, deviceId, now }) {
      const [pendingQuery, claimedQuery] = await Promise.all([
        db.collection('field_session_commands')
          .where({ table_id: tableId, status: 'pending' })
          .orderBy('server_sequence', 'asc').limit(1).get(),
        db.collection('field_session_commands')
          .where({
            table_id: tableId, status: 'claimed', claimed_by_device_id: deviceId,
          })
          .orderBy('server_sequence', 'asc').limit(1).get(),
      ])
      const candidate = pickClaimCandidate(pendingQuery.data, claimedQuery.data, deviceId)
      if (!candidate) return null
      return db.runTransaction(async (transaction) => {
        const ref = transaction.collection('field_session_commands').doc(candidate.command_id)
        const current = await ref.get()
        if (!current.data || current.data.table_id !== tableId) {
          throw new SyncApiError('SYNC_CONFLICT')
        }
        if (isManagedSessionId(candidate.session_id) || isManagedSessionId(current.data.session_id)) {
          const command = current.data
          const context = await managedContext(transaction, command, tableId, deviceId)
          requireNext(command, context.head)
          conflictUnless(command.command_id === candidate.command_id && command.session_id === candidate.session_id)
        }
        if (current.data.status === 'claimed'
            && current.data.claimed_by_device_id === deviceId) return current.data
        if (current.data.status !== 'pending') throw new SyncApiError('SYNC_CONFLICT')
        const claimed = {
          ...withoutDocumentId(current.data),
          status: 'claimed', claimed_by_device_id: deviceId, claimed_at: now,
        }
        await ref.set({ data: claimed })
        return claimed
      })
    },
    async complete({
      commandId, tableId, deviceId, status, publicResultCode, resultingRevision, now,
    }) {
      return db.runTransaction(async (transaction) => {
        const ref = transaction.collection('field_session_commands').doc(commandId)
        const result = await ref.get()
        const command = result.data
        if (!command || command.table_id !== tableId
            || command.claimed_by_device_id !== deviceId) {
          throw new SyncApiError('SYNC_UNAUTHORIZED')
        }
        if (isManagedSessionId(command.session_id)) {
          return completeManagedCommand(transaction, command, {
            commandId, tableId, deviceId, status, publicResultCode, resultingRevision, now,
          })
        }
        if (TERMINAL.has(command.status)) {
          if (command.status !== status || command.public_result_code !== publicResultCode
              || command.resulting_revision !== resultingRevision) {
            throw new SyncApiError('SYNC_CONFLICT')
          }
          if (status === 'applied' && ['claim_role', 'release_role'].includes(command.command_type)) {
            const role = command.command_type === 'claim_role'
              ? command.validated_payload?.role : command.applied_role
            if (['red', 'blue'].includes(role)) {
              const history = await getOptionalDocument(transaction.collection('session_role_binding_history')
                .doc(documentId(`${command.session_id}:${role}:${resultingRevision}`)))
              const expected = {
                session_id: command.session_id, role,
                user_id: command.command_type === 'claim_role' ? command.actor_user_id : null,
                resulting_revision: resultingRevision, command_id: commandId,
                table_id: tableId, device_id: deviceId,
              }
              if (history.data && digest(withoutDocumentId(history.data)) !== digest(expected)) {
                throw new SyncApiError('SYNC_CONFLICT')
              }
            }
          }
          // Historical terminal commands are never migrated from today's roster.
          return command
        }
        if (command.status !== 'claimed') throw new SyncApiError('SYNC_CONFLICT')
        const completed = {
          ...withoutDocumentId(command), status, public_result_code: publicResultCode,
          resulting_revision: resultingRevision, completed_at: now,
        }
        const sessionRef = transaction.collection('field_live_sessions').doc(command.session_id)
        const session = await sessionRef.get()
        if (status === 'applied' && ['claim_role', 'release_role'].includes(command.command_type)) {
          if (!session.data || session.data.session_id !== command.session_id
              || session.data.table_id !== tableId) throw new SyncApiError('SYNC_CONFLICT')
          const roles = command.command_type === 'claim_role'
            ? [command.validated_payload && command.validated_payload.role]
            : ['red', 'blue'].filter(role => session.data[`${role}_user_id`] === command.actor_user_id)
          if (roles.length !== 1 || !['red', 'blue'].includes(roles[0])
              || !ID_PATTERN.test(command.actor_user_id || '')) throw new SyncApiError('SYNC_CONFLICT')
          const history = {
            session_id: command.session_id, role: roles[0],
            user_id: command.command_type === 'claim_role' ? command.actor_user_id : null,
            resulting_revision: resultingRevision, command_id: commandId,
            table_id: tableId, device_id: deviceId,
          }
          const historyRef = transaction.collection('session_role_binding_history')
            .doc(documentId(`${command.session_id}:${roles[0]}:${resultingRevision}`))
          const prior = await getOptionalDocument(historyRef)
          if (prior.data && digest(withoutDocumentId(prior.data)) !== digest(history)) {
            throw new SyncApiError('SYNC_CONFLICT')
          }
          if (!prior.data) await historyRef.set({ data: history })
          // Persist release's pre-apply role on the immutable command so a later
          // history check never needs the final roster to verify the release.
          completed.applied_role = roles[0]
        }
        await ref.set({ data: completed })
        if (session.data && session.data.table_id === tableId) {
          const storedSession = withoutDocumentId(session.data)
          const projected = status === 'applied'
            ? projectSessionAfterApplied(storedSession, command, now)
            : storedSession
          await sessionRef.set({ data: {
            ...projected, control_revision: resultingRevision, updated_at: now,
          } })
          if (status === 'applied' && command.command_type === 'claim_role') {
            const role = command.validated_payload.role
            const bindingId = crypto.createHash('sha256')
              .update(`${command.session_id}:${command.actor_user_id}`, 'utf8').digest('hex')
            const bindingRef = transaction.collection('session_user_bindings').doc(bindingId)
            const bindingResult = await getOptionalDocument(bindingRef)
            const requestId = bindingResult.data && bindingResult.data.request_id
            const roleId = crypto.createHash('sha256')
              .update(`${command.session_id}:${role}`, 'utf8').digest('hex')
            await transaction.collection('session_role_bindings').doc(roleId).set({ data: {
              schema_version: '1.0', session_id: command.session_id,
              user_id: command.actor_user_id, role, request_id: requestId || null,
              operator_id: deviceId, confirmed_at: now,
            } })
            if (bindingResult.data) {
              await bindingRef.set({ data: {
                ...withoutDocumentId(bindingResult.data), role, updated_at: now,
              } })
            }
            if (requestId) {
              const requestRef = transaction.collection('session_join_requests').doc(requestId)
              const requestResult = await getOptionalDocument(requestRef)
              const request = requestResult.data
              if (!request || request.session_id !== command.session_id
                  || request.user_id !== command.actor_user_id) {
                throw new SyncApiError('SYNC_CONFLICT')
              }
              await requestRef.update({ data: { status: 'confirmed', role, updated_at: now } })
            }
          }
          if (status === 'applied' && command.command_type === 'release_role') {
            const role = completed.applied_role
            const roleId = crypto.createHash('sha256')
              .update(`${command.session_id}:${role}`, 'utf8').digest('hex')
            await transaction.collection('session_role_bindings').doc(roleId).remove()
          }
        }
        return completed
      })
    },
  }
}


module.exports = {
  claimNextFieldCommand,
  completeFieldCommand,
  createCloudDeviceFieldCommandRepository,
  pickClaimCandidate,
  projectSessionAfterApplied,
}
