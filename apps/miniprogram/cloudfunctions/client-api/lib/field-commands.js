const crypto = require('crypto')

const { ClientApiError } = require('./errors')
const { isManagedSessionId, readManagedCommandContext } = require('../../shared/managed-command-authority')
const { matchId } = require('../../shared/match-setup-service')
const { matchDocumentId } = require('../../shared/match-facts-repository')


const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u
const IDEMPOTENCY_PATTERN = /^[A-Za-z0-9._-]{16,128}$/u
const ROLE_VALUES = new Set(['red', 'blue'])
const GROUP_VALUES = new Set(['solid', 'stripe'])
const COMMAND_TYPES = new Set([
  'claim_role', 'release_role', 'set_breaker', 'assign_groups',
  'continue_turn', 'switch_turn', 'end_session',
])


async function getOptionalDocument(reference) {
  try {
    return await reference.get()
  } catch (error) {
    const message = error && error.message ? error.message : String(error)
    if (/document with _id .+ does not exist/i.test(message)) return { data: null }
    throw error
  }
}


function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}


function exactPayload(payload, keys) {
  if (!payload || Array.isArray(payload) || typeof payload !== 'object') {
    throw new ClientApiError('COMMAND_INVALID')
  }
  const actual = Object.keys(payload).sort()
  const expected = keys.slice().sort()
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new ClientApiError('COMMAND_INVALID')
  }
  return { ...payload }
}


function validatePayload(commandType, payload) {
  if (!COMMAND_TYPES.has(commandType)) throw new ClientApiError('COMMAND_INVALID')
  if (commandType === 'claim_role' || commandType === 'set_breaker') {
    const value = exactPayload(payload, ['role'])
    if (!ROLE_VALUES.has(value.role)) throw new ClientApiError('COMMAND_INVALID')
    return value
  }
  if (commandType === 'assign_groups') {
    const value = exactPayload(payload, ['red_group'])
    if (!GROUP_VALUES.has(value.red_group)) throw new ClientApiError('COMMAND_INVALID')
    return value
  }
  return exactPayload(payload, [])
}


function actorRole(context, actorUserId) {
  if (context.red_user_id === actorUserId) return 'red'
  if (context.blue_user_id === actorUserId) return 'blue'
  return null
}


function authorize(commandType, payload, context, actorUserId) {
  if (!context || context.status !== 'open') throw new ClientApiError('SESSION_NOT_JOINABLE')
  if (context.joined !== true) throw new ClientApiError('COMMAND_FORBIDDEN')
  const role = actorRole(context, actorUserId)
  if (commandType === 'claim_role') {
    if (context.started || role || context[`${payload.role}_user_id`]) {
      throw new ClientApiError('COMMAND_FORBIDDEN')
    }
    return
  }
  if (!role) throw new ClientApiError('COMMAND_FORBIDDEN')
  if (commandType === 'release_role') {
    if (context.started) throw new ClientApiError('COMMAND_FORBIDDEN')
    return
  }
  if (commandType === 'set_breaker') {
    if (context.started || !context.red_user_id || !context.blue_user_id) {
      throw new ClientApiError('COMMAND_FORBIDDEN')
    }
    return
  }
  if (commandType === 'assign_groups') {
    if (!context.red_user_id || !context.blue_user_id) {
      throw new ClientApiError('COMMAND_FORBIDDEN')
    }
    return
  }
  if ((commandType === 'continue_turn' || commandType === 'switch_turn')
      && context.active_user_id !== actorUserId) {
    throw new ClientApiError('COMMAND_FORBIDDEN')
  }
}


function publicCommand(command) {
  return {
    schema_version: '1.0', command_id: command.command_id,
    command_type: command.command_type, status: command.status,
    server_sequence: command.server_sequence,
    public_result_code: command.public_result_code || null,
    resulting_revision: Number.isInteger(command.resulting_revision)
      ? command.resulting_revision : null,
  }
}


async function getCurrentField({ sessionId, actorUserId, repo }) {
  if (!ID_PATTERN.test(sessionId || '') || !ID_PATTERN.test(actorUserId || '')) {
    throw new ClientApiError('COMMAND_INVALID')
  }
  const context = await repo.getAuthorizationContext({ sessionId, actorUserId })
  if (!context || context.joined !== true) throw new ClientApiError('COMMAND_FORBIDDEN')
  const role = actorRole(context, actorUserId)
  const activeRole = context.active_user_id === context.red_user_id
    ? 'red' : context.active_user_id === context.blue_user_id ? 'blue' : null
  return {
    schema_version: '1.0', session_id: context.session_id, status: context.status,
    revision: context.revision, started: context.started, actor_role: role,
    red_occupied: Boolean(context.red_user_id), blue_occupied: Boolean(context.blue_user_id),
    breaker: context.breaker || null, active_role: activeRole,
    mode: context.mode || null, format: context.format || null,
    race_target: Number.isInteger(context.race_target) ? context.race_target : null,
    ranking_intent: context.ranking_intent === true,
    settings_revision: Number.isInteger(context.settings_revision)
      ? context.settings_revision : 0,
    red_group: context.red_group || null, blue_group: context.blue_group || null,
  }
}


async function submitFieldCommand(input, repo) {
  if (!input || !ID_PATTERN.test(input.actorUserId || '')
      || !ID_PATTERN.test(input.sessionId || '')
      || !Number.isSafeInteger(input.expectedRevision) || input.expectedRevision < 0
      || !IDEMPOTENCY_PATTERN.test(input.idempotencyKey || '')
      || typeof input.now !== 'string') {
    throw new ClientApiError('COMMAND_INVALID')
  }
  let validatedPayload = validatePayload(input.commandType, input.payload)
  const idempotencyDigest = crypto.createHash('sha256')
    .update(`${input.sessionId}:${input.actorUserId}:${input.idempotencyKey}`, 'utf8')
    .digest('hex')
  const managed = isManagedSessionId(input.sessionId)
  const submittedIntentSha256 = managed ? sha256(canonical({
    command_type: input.commandType, expected_revision: input.expectedRevision, payload: validatedPayload,
  })) : null
  if (managed) {
    if (typeof repo.getManagedCommandReplay !== 'function') throw new ClientApiError('COMMAND_FORBIDDEN')
    const replay = await repo.getManagedCommandReplay({
      sessionId: input.sessionId, actorUserId: input.actorUserId,
      idempotencyDigest, submittedIntentSha256,
    })
    if (replay) return publicCommand(replay)
  }
  const context = await repo.getAuthorizationContext({
    sessionId: input.sessionId, actorUserId: input.actorUserId,
  })
  if (!context || context.session_id !== input.sessionId) {
    throw new ClientApiError('SESSION_NOT_JOINABLE')
  }
  if (context.revision !== input.expectedRevision) {
    throw new ClientApiError('COMMAND_STALE_REVISION')
  }
  if (input.commandType === 'set_breaker') {
    if (typeof repo.getRankingMatchContext !== 'function') {
      throw new ClientApiError('MATCH_SETTINGS_NOT_READY')
    }
    const rankingMatch = await repo.getRankingMatchContext({ sessionId: input.sessionId })
    const settings = rankingMatch && rankingMatch.settings
    if (!rankingMatch || rankingMatch.status !== 'ready' || !settings
        || settings.first_breaker !== validatedPayload.role) {
      throw new ClientApiError('MATCH_SETTINGS_NOT_READY')
    }
    validatedPayload = {
      role: settings.first_breaker,
      settings_revision: settings.settings_revision,
      mode: settings.mode,
      format: settings.format,
      race_target: settings.race_target,
      ranking_intent: settings.ranking_intent,
    }
  }
  authorize(input.commandType, validatedPayload, context, input.actorUserId)
  if (input.commandType === 'assign_groups' && context.red_group
      && context.red_group !== validatedPayload.red_group) {
    const proposal = await repo.getPendingGroupCorrection({ sessionId: input.sessionId })
    if (!proposal) {
      validatedPayload = { ...validatedPayload, correction_phase: 'proposal' }
    } else if (proposal.actor_user_id === input.actorUserId) {
      if (proposal.idempotency_digest !== idempotencyDigest) {
        throw new ClientApiError('COMMAND_FORBIDDEN')
      }
      validatedPayload = { ...validatedPayload, correction_phase: 'proposal' }
    } else if (proposal.validated_payload.red_group !== validatedPayload.red_group) {
      throw new ClientApiError('GROUP_CORRECTION_REQUIRED')
    } else {
      validatedPayload = {
        ...validatedPayload,
        correction_phase: 'confirmation',
        proposal_command_id: proposal.command_id,
      }
    }
  }
  const intentDigest = crypto.createHash('sha256').update(canonical({
    command_type: input.commandType, expected_revision: input.expectedRevision,
    validated_payload: validatedPayload,
  }), 'utf8').digest('hex')
  const command = await repo.createCommand({
    schema_version: '1.0', session_id: input.sessionId, table_id: context.table_id,
    actor_user_id: input.actorUserId, command_type: input.commandType,
    validated_payload: validatedPayload, expected_revision: input.expectedRevision,
    idempotency_digest: idempotencyDigest, intent_digest: intentDigest,
    status: 'pending', public_result_code: null, resulting_revision: null,
    created_at: input.now,
    ...(managed ? { submitted_intent_sha256: submittedIntentSha256 } : {}),
  }, managed ? { submittedPayload: input.payload } : undefined)
  return publicCommand(command)
}


async function getFieldCommandStatus({ commandId, actorUserId, repo }) {
  if (!ID_PATTERN.test(commandId || '') || !ID_PATTERN.test(actorUserId || '')) {
    throw new ClientApiError('COMMAND_INVALID')
  }
  const command = await repo.getCommand({ commandId, actorUserId })
  if (!command) throw new ClientApiError('COMMAND_NOT_FOUND')
  return publicCommand(command)
}


const sha256 = value => crypto.createHash('sha256').update(value, 'utf8').digest('hex')
const isHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/u.test(value)
const commandDocumentId = digest => `command-${digest.slice(0, 48)}`
const bindingDocumentId = (sessionId, subject) => sha256(`${sessionId}:${subject}`)
function validStoredIntent(command) {
  return Number.isSafeInteger(command.expected_revision) && command.expected_revision >= 0
    && COMMAND_TYPES.has(command.command_type) && isHash(command.intent_digest)
    && command.intent_digest === sha256(canonical({ command_type: command.command_type,
      expected_revision: command.expected_revision, validated_payload: command.validated_payload }))
}
function managedReplay(command, context, { sessionId, actorUserId, idempotencyDigest, submittedIntentSha256 }) {
  if (!command) return null
  if (!isHash(command.submitted_intent_sha256) || command.submitted_intent_sha256 !== submittedIntentSha256
      || command.session_id !== sessionId || command.actor_user_id !== actorUserId
      || command.idempotency_digest !== idempotencyDigest
      || command.command_id !== commandDocumentId(idempotencyDigest)
      || command.schema_version !== '1.0' || !validStoredIntent(command)
      || command.table_id !== context.birth.table_id
      || !['pending', 'claimed', 'applied', 'rejected'].includes(command.status)
      || !Number.isSafeInteger(command.server_sequence) || command.server_sequence < 1
      || command.server_sequence >= context.head.issued_next
      || (context.head.closed_target !== null && command.server_sequence > context.head.closed_target)) {
    throw new ClientApiError('COMMAND_IDEMPOTENCY_CONFLICT')
  }
  return command
}
async function managedAuthorizationContext(transaction, authority, actorUserId) {
  const { birth, live } = authority
  const read = async (collection, subject) => (await getOptionalDocument(transaction.collection(collection)
    .doc(bindingDocumentId(birth.session_id, subject)))).data
  const [actor, red, blue, tableResult] = await Promise.all([
    read('session_user_bindings', actorUserId), read('session_role_bindings', 'red'),
    read('session_role_bindings', 'blue'),
    getOptionalDocument(transaction.collection('table_join_codes').doc(birth.table_id)),
  ])
  const table = tableResult.data
  if (!table || table.status !== 'active' || table.table_id !== birth.table_id
      || table.current_session_id !== birth.session_id) throw new ClientApiError('SESSION_NOT_JOINABLE')
  if (!actor || actor.schema_version !== '1.0' || actor.session_id !== birth.session_id
      || actor.table_id !== birth.table_id || actor.user_id !== actorUserId
      || ![null, 'red', 'blue'].includes(actor.role)) throw new ClientApiError('COMMAND_FORBIDDEN')
  for (const [role, binding] of [['red', red], ['blue', blue]]) {
    if (binding && (binding.schema_version !== '1.0' || binding.session_id !== birth.session_id
        || binding.role !== role || !ID_PATTERN.test(binding.user_id || '')
        || (binding.table_id !== undefined && binding.table_id !== birth.table_id))) throw new ClientApiError('COMMAND_FORBIDDEN')
    if ((binding && binding.user_id || null) !== (live[`${role}_user_id`] || null)) throw new ClientApiError('COMMAND_FORBIDDEN')
  }
  return { ...live, revision: live.control_revision ?? 0, joined: true, started: Boolean(live.started_at),
    red_user_id: red && red.user_id, blue_user_id: blue && blue.user_id }
}
function breakerPayload(settings) {
  if (!settings || !Number.isSafeInteger(settings.settings_revision) || settings.settings_revision < 1
      || !['observation', 'assistance'].includes(settings.mode) || !['single', 'race'].includes(settings.format)
      || !ROLE_VALUES.has(settings.first_breaker) || typeof settings.ranking_intent !== 'boolean'
      || (settings.mode === 'assistance' && settings.ranking_intent)
      || (settings.format === 'single' ? settings.race_target !== null
        : !Number.isSafeInteger(settings.race_target) || settings.race_target < 2 || settings.race_target > 99)) {
    throw new ClientApiError('MATCH_SETTINGS_NOT_READY')
  }
  return { role: settings.first_breaker, settings_revision: settings.settings_revision,
    mode: settings.mode, format: settings.format, race_target: settings.race_target, ranking_intent: settings.ranking_intent }
}
async function managedPayload(transaction, command, context, authority, raw) {
  const payload = validatePayload(command.command_type, raw)
  if (command.command_type === 'release_role') return { role: actorRole(context, command.actor_user_id) }
  if (command.command_type === 'set_breaker') {
    // Read just the private projection document, never its unbounded event stream.
    const id = matchId(command.session_id)
    const match = (await getOptionalDocument(transaction.collection('matches').doc(matchDocumentId(id)))).data
    const projection = match && match.projection
    if (!match || match.schema_version !== '1.0' || match.match_id !== id
        || match.session_id !== command.session_id || match.table_id !== authority.birth.table_id
        || !projection || projection.match_id !== id || projection.session_id !== command.session_id
        || projection.table_id !== authority.birth.table_id || projection.status !== 'ready'
        || projection.participants?.red !== context.red_user_id || projection.participants?.blue !== context.blue_user_id
        || !Array.isArray(projection.settings_confirmations)
        || projection.settings_confirmations.length !== 2
        || !['red', 'blue'].every(role => projection.settings_confirmations.includes(role))) throw new ClientApiError('MATCH_SETTINGS_NOT_READY')
    const enriched = breakerPayload(projection.settings)
    if (enriched.role !== payload.role) throw new ClientApiError('MATCH_SETTINGS_NOT_READY')
    return enriched
  }
  if (command.command_type === 'assign_groups' && context.red_group && context.red_group !== payload.red_group) {
    if (command.validated_payload.correction_phase === 'confirmation') {
      const proposalId = command.validated_payload.proposal_command_id
      if (!ID_PATTERN.test(proposalId || '')) throw new ClientApiError('GROUP_CORRECTION_REQUIRED')
      const proposal = (await getOptionalDocument(transaction.collection('field_session_commands').doc(proposalId))).data
      if (!proposal || proposal.schema_version !== '1.0' || proposal.command_id !== proposalId
          || !isHash(proposal.idempotency_digest) || commandDocumentId(proposal.idempotency_digest) !== proposalId
          || !validStoredIntent(proposal) || proposal.session_id !== command.session_id
          || proposal.table_id !== authority.birth.table_id || proposal.command_type !== 'assign_groups'
          || !['pending', 'claimed', 'applied'].includes(proposal.status)
          || ![context.red_user_id, context.blue_user_id].includes(proposal.actor_user_id)
          || proposal.actor_user_id === command.actor_user_id
          || !Number.isSafeInteger(proposal.server_sequence) || proposal.server_sequence < 1
          || proposal.server_sequence >= authority.head.issued_next
          || !isHash(proposal.submitted_intent_sha256)
          || canonical(proposal.validated_payload) !== canonical({ red_group: payload.red_group, correction_phase: 'proposal' })) {
        throw new ClientApiError('GROUP_CORRECTION_REQUIRED')
      }
      return { ...payload, correction_phase: 'confirmation', proposal_command_id: proposalId }
    }
    return { ...payload, correction_phase: 'proposal' }
  }
  return payload
}
async function createManagedCommand(db, command, { submittedPayload } = {}) {
  if (!isHash(command.idempotency_digest) || !isHash(command.submitted_intent_sha256)
      || !ID_PATTERN.test(command.actor_user_id || '')
      || !Number.isSafeInteger(command.expected_revision) || command.expected_revision < 0
      || sha256(canonical({ command_type: command.command_type, expected_revision: command.expected_revision,
        payload: validatePayload(command.command_type, submittedPayload) })) !== command.submitted_intent_sha256) {
    throw new ClientApiError('COMMAND_INVALID')
  }
  return db.runTransaction(async transaction => {
    const commandId = commandDocumentId(command.idempotency_digest)
    const ref = transaction.collection('field_session_commands').doc(commandId)
    const existing = (await getOptionalDocument(ref)).data
    const authority = await readManagedCommandContext(transaction, { sessionId: command.session_id })
    if (!authority) throw new ClientApiError('SESSION_NOT_JOINABLE')
    const replay = managedReplay(existing, authority, { sessionId: command.session_id,
      actorUserId: command.actor_user_id, idempotencyDigest: command.idempotency_digest,
      submittedIntentSha256: command.submitted_intent_sha256 })
    if (replay) return replay
    const { head, birth } = authority
    if (head.coverage !== 'open') throw new ClientApiError('SESSION_NOT_JOINABLE')
    if (head.issued_next === Number.MAX_SAFE_INTEGER || head.revision === Number.MAX_SAFE_INTEGER) {
      throw new ClientApiError('COMMAND_FORBIDDEN')
    }
    const context = await managedAuthorizationContext(transaction, authority, command.actor_user_id)
    if (context.revision !== command.expected_revision) throw new ClientApiError('COMMAND_STALE_REVISION')
    authorize(command.command_type, submittedPayload, context, command.actor_user_id)
    const payload = await managedPayload(transaction, command, context, authority, submittedPayload)
    if (command.table_id !== birth.table_id || canonical(payload) !== canonical(
      command.command_type === 'release_role' ? { role: actorRole(context, command.actor_user_id) } : command.validated_payload
    )) throw new ClientApiError('COMMAND_FORBIDDEN')
    const stored = { ...command, validated_payload: payload,
      intent_digest: sha256(canonical({ command_type: command.command_type, expected_revision: command.expected_revision,
        validated_payload: payload })), command_id: commandId, server_sequence: head.issued_next }
    await ref.set({ data: stored })
    const { _id, ...headData } = head
    await transaction.collection('session_command_authority').doc(command.session_id).set({ data: {
      ...headData, issued_next: head.issued_next + 1, revision: head.revision + 1,
    } })
    return stored
  })
}


function createCloudFieldCommandRepository(db) {
  const roleId = (sessionId, role) => crypto.createHash('sha256')
    .update(`${sessionId}:${role}`, 'utf8').digest('hex')
  return {
    async getManagedCommandReplay({ sessionId, actorUserId, idempotencyDigest, submittedIntentSha256 }) {
      if (!isManagedSessionId(sessionId) || !isHash(idempotencyDigest) || !isHash(submittedIntentSha256)) {
        throw new ClientApiError('COMMAND_FORBIDDEN')
      }
      return db.runTransaction(async transaction => {
        const commandId = commandDocumentId(idempotencyDigest)
        const existing = (await getOptionalDocument(transaction.collection('field_session_commands').doc(commandId))).data
        const context = await readManagedCommandContext(transaction, { sessionId })
        if (!context) throw new ClientApiError('SESSION_NOT_JOINABLE')
        return managedReplay(existing, context, { sessionId, actorUserId, idempotencyDigest, submittedIntentSha256 })
      })
    },
    async getRankingMatchContext({ sessionId }) {
      if (isManagedSessionId(sessionId)) {
        const match = (await getOptionalDocument(db.collection('matches').doc(matchDocumentId(matchId(sessionId))))).data
        return match && match.projection ? { status: match.projection.status, settings: match.projection.settings } : null
      }
      const result = await db.collection('matches').where({ session_id: sessionId }).limit(2).get()
      const matches = result.data || []
      if (matches.length !== 1 || !matches[0].projection || !matches[0].projection.settings) return null
      return {
        status: matches[0].projection.status,
        settings: { ...matches[0].projection.settings },
      }
    },
    async getAuthorizationContext({ sessionId, actorUserId }) {
      const bindingId = crypto.createHash('sha256')
        .update(`${sessionId}:${actorUserId}`, 'utf8').digest('hex')
      const [sessionResult, redResult, blueResult, bindingResult] = await Promise.all([
        getOptionalDocument(db.collection('field_live_sessions').doc(sessionId)),
        getOptionalDocument(db.collection('session_role_bindings').doc(roleId(sessionId, 'red'))),
        getOptionalDocument(db.collection('session_role_bindings').doc(roleId(sessionId, 'blue'))),
        getOptionalDocument(db.collection('session_user_bindings').doc(bindingId)),
      ])
      const session = sessionResult.data
      if (!session) return null
      return {
        session_id: sessionId, table_id: session.table_id, status: session.status,
        revision: Number.isInteger(session.control_revision) ? session.control_revision : 0,
        joined: Boolean(bindingResult.data),
        started: Boolean(session.started_at),
        red_user_id: redResult.data && redResult.data.user_id,
        blue_user_id: blueResult.data && blueResult.data.user_id,
        active_user_id: session.active_user_id || null,
        breaker: session.breaker || null,
        mode: session.mode || null, format: session.format || null,
        race_target: session.race_target ?? null,
        ranking_intent: session.ranking_intent === true,
        settings_revision: Number.isInteger(session.settings_revision)
          ? session.settings_revision : 0,
        red_group: session.red_group || null, blue_group: session.blue_group || null,
      }
    },
    async createCommand(command, options = {}) {
      if (isManagedSessionId(command.session_id)) return createManagedCommand(db, command, options)
      const commandId = commandDocumentId(command.idempotency_digest)
      return db.runTransaction(async (transaction) => {
        const commandRef = transaction.collection('field_session_commands').doc(commandId)
        const existing = await getOptionalDocument(commandRef)
        if (existing.data) {
          if (existing.data.intent_digest !== command.intent_digest) {
            throw new ClientApiError('COMMAND_IDEMPOTENCY_CONFLICT')
          }
          return existing.data
        }
        const sequenceRef = transaction.collection('field_command_sequences')
          .doc(command.session_id)
        const sequenceResult = await getOptionalDocument(sequenceRef)
        const serverSequence = Number.isInteger(sequenceResult.data && sequenceResult.data.next)
          ? sequenceResult.data.next : 1
        const stored = { ...command, command_id: commandId, server_sequence: serverSequence }
        await commandRef.set({ data: stored })
        await sequenceRef.set({ data: {
          schema_version: '1.0', session_id: command.session_id,
          next: serverSequence + 1, updated_at: command.created_at,
        } })
        return stored
      })
    },
    async getPendingGroupCorrection({ sessionId }) {
      if (isManagedSessionId(sessionId)) {
        // Discover only current candidates; a missing result is never prefix evidence.
        const results = await Promise.all(['pending', 'claimed', 'applied'].map(status => db.collection('field_session_commands')
          .where({ session_id: sessionId, command_type: 'assign_groups', status,
            'validated_payload.correction_phase': 'proposal' })
          .orderBy('server_sequence', 'desc').limit(1).get()))
        return results.flatMap(result => result.data || [])
          .sort((left, right) => right.server_sequence - left.server_sequence)[0] || null
      }
      const result = await db.collection('field_session_commands')
        .where({ session_id: sessionId, command_type: 'assign_groups' })
        .orderBy('server_sequence', 'desc').limit(20).get()
      return (result.data || []).find((item) => (
        item.validated_payload
        && item.validated_payload.correction_phase === 'proposal'
        && !['rejected', 'expired'].includes(item.status)
      )) || null
    },
    async getCommand({ commandId, actorUserId }) {
      const result = await getOptionalDocument(
        db.collection('field_session_commands').doc(commandId),
      )
      return result.data && result.data.actor_user_id === actorUserId ? result.data : null
    },
  }
}


module.exports = {
  createCloudFieldCommandRepository,
  getCurrentField,
  getFieldCommandStatus,
  publicCommand,
  submitFieldCommand,
  validatePayload,
}
