const { createJoinAdmin } = require('../../cloudfunctions/sync-api/lib/join-admin')
const { createCloudFieldCommandRepository, submitFieldCommand } = require('../../cloudfunctions/client-api/lib/field-commands')
const { createCloudDeviceFieldCommandRepository } = require('../../cloudfunctions/sync-api/lib/field-commands')
const { verifyManagedShotOwnership } = require('../../cloudfunctions/shared/managed-command-authority')
const { documentId } = require('../../cloudfunctions/shared/shot-ownership')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const birthId = '12345678-1234-4234-8234-123456789abc'
const sessionId = `managed-${birthId}`
const now = '2026-11-01T10:00:00.000Z'
const scope = { tableId: 'table-1', deviceId: 'device-1', now }
const closeInput = { ...scope, sessionId }
const historyId = revision => documentId(`${sessionId}:red:${revision}`)
async function setup(options = {}) {
  const db = ownerDeleteDb({ table_join_codes: { 'table-1': {
    table_id: 'table-1', status: 'active', current_session_id: null,
  } } }, { strictTransactions: true, ...options })
  const admin = createJoinAdmin(db, { managedSessionsEnabled: true, managedMediaAdmissionEnabled: true, nowIso: () => now, idFactory: () => birthId })
  await admin.createManagedSession({ ...scope, openRequestId: birthId })
  db.rows.session_user_bindings = { [documentId(`${sessionId}:user-red`)]: {
    schema_version: '1.0', session_id: sessionId, table_id: 'table-1', user_id: 'user-red', role: null, request_id: 'join-1',
  } }
  db.rows.session_join_requests = { 'join-1': { session_id: sessionId, user_id: 'user-red', status: 'pending', role: null } }
  const issuer = createCloudFieldCommandRepository(db)
  const device = createCloudDeviceFieldCommandRepository(db)
  let index = 0
  const issue = (type = 'claim_role', payload = { role: 'red' }) => submitFieldCommand({
    sessionId, actorUserId: 'user-red', expectedRevision: db.rows.field_live_sessions[sessionId].control_revision || 0,
    commandType: type, payload, idempotencyKey: `intent-${String(++index).padStart(16, '0')}`, now,
  }, issuer)
  const complete = (command, status = 'applied', code = 'APPLIED', revision = command.server_sequence) => device.complete({
    ...scope, commandId: command.command_id, status, publicResultCode: code, resultingRevision: revision,
  })
  const apply = async (type, payload) => {
    const command = await issue(type, payload)
    await device.claimNext(scope)
    return complete(command)
  }
  return { db, admin, device, issue, complete, apply, issuer }
}
function budget(db, maximum) { expect(db.transactionAttempts.at(-1).count).toBeLessThanOrEqual(maximum) }

test('zero close seals exact frozen prefix without clearing table pointer; old replay survives new allocation', async () => {
  const { db, admin } = await setup()
  expect(typeof admin.closeManagedSession).toBe('function')
  const closed = await admin.closeManagedSession(closeInput)
  expect(closed).toEqual({ schema_version: '1.0', session_id: sessionId, birth_id: birthId,
    birth_contract: 'managed_session_birth_v1', table_id: 'table-1', device_id: 'device-1',
    coverage: 'sealed', closed_target: 0, contiguous_ack_sequence: 0 })
  budget(db, 10)
  expect(db.rows.table_join_codes['table-1'].current_session_id).toBe(sessionId)
  const other = createJoinAdmin(db, { managedSessionsEnabled: true, nowIso: () => now,
    idFactory: () => '22345678-1234-4234-8234-123456789abc' })
  const next = await other.createManagedSession({ ...scope, openRequestId: '32345678-1234-4234-8234-123456789abc' })
  const snapshot = JSON.stringify(db.rows)
  expect(await admin.closeManagedSession(closeInput)).toEqual(closed)
  expect(JSON.stringify(db.rows)).toBe(snapshot)
  expect(db.rows.table_join_codes['table-1'].current_session_id).toBe(next.session_id)
})

test('close freezes issued target, drains ordered results and seals only at actual final ACK', async () => {
  const { db, admin, device, issue, complete } = await setup()
  const first = await issue(); const second = await issue()
  expect(typeof admin.closeManagedSession).toBe('function')
  expect(await admin.closeManagedSession(closeInput)).toMatchObject({ coverage: 'closing', closed_target: 2, contiguous_ack_sequence: 0 })
  budget(db, 10)
  await expect(issue()).rejects.toBeDefined()
  await expect(complete(second, 'rejected', 'SESSION_ENDED', 0)).rejects.toBeDefined()
  expect((await device.claimNext(scope)).command_id).toBe(first.command_id); budget(db, 6)
  await complete(first, 'rejected', 'SESSION_ENDED', 0); budget(db, 10)
  expect(await admin.closeManagedSession(closeInput)).toMatchObject({ coverage: 'closing', closed_target: 2, contiguous_ack_sequence: 1 })
  await device.claimNext(scope)
  await complete(second, 'rejected', 'SESSION_ENDED', 0); budget(db, 10)
  expect(db.rows.session_command_authority[sessionId]).toMatchObject({ coverage: 'sealed', closed_target: 2, contiguous_ack_sequence: 2 })
})

test('managed claim/complete validates authority and stores immutable reciprocal claim-release-reclaim links', async () => {
  const { db, device, issue, complete, apply } = await setup()
  const first = await issue()
  expect(await device.claimNext(scope)).toMatchObject({ command_id: first.command_id, status: 'claimed' }); budget(db, 6)
  await complete(first); budget(db, 22)
  await apply('release_role', {}); budget(db, 22)
  expect(db.rows.session_user_bindings[documentId(`${sessionId}:user-red`)].role).toBeNull()
  await apply(); budget(db, 22)
  const nodes = db.rows.session_role_binding_history
  expect(nodes[historyId(1)]).toEqual({ contract: 'managed_role_link_v1', session_id: sessionId,
    birth_id: birthId, role: 'red', user_id: 'user-red', resulting_revision: 1,
    command_id: first.command_id, table_id: 'table-1', device_id: 'device-1',
    server_sequence: 1, previous_id: null, next_id: historyId(2) })
  expect(nodes[historyId(2)]).toMatchObject({ user_id: null, previous_id: historyId(1), next_id: historyId(3) })
  expect(db.rows.session_command_authority[sessionId].role_tails.red).toBe(historyId(3))
  const snapshot = JSON.stringify(db.rows)
  await complete(first); budget(db, 12)
  expect(JSON.stringify(db.rows)).toBe(snapshot)
  delete nodes[historyId(1)]
  await expect(complete(first)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(nodes[historyId(1)]).toBeUndefined()
})

test.each([
  ['rejected', 'COMMAND_OUT_OF_ORDER'], ['rejected', 'COMMAND_IDEMPOTENCY_CONFLICT'],
  ['applied', 'STALE_SESSION_CLOSED'], ['applied', 'GROUP_CORRECTION_PROPOSED'], ['rejected', 'UNKNOWN_CODE'],
])('managed rejects nondurable result %s/%s without prefix advance', async (status, code) => {
  const { db, issue, device, complete } = await setup()
  const command = await issue(); await device.claimNext(scope)
  const before = JSON.stringify(db.rows)
  await expect(complete(command, status, code)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(JSON.stringify(db.rows)).toBe(before)
})

test.each(['SESSION_ENDED', 'STALE_REVISION', 'TURN_CONFIRMATION_MISMATCH', 'COMMAND_FORBIDDEN', 'COMMAND_INVALID', 'SHOT_UNSETTLED'])(
  'managed durable rejection %s advances contiguous ACK without role link', async code => {
    const { db, issue, device, complete } = await setup()
    const command = await issue(); await device.claimNext(scope)
    await complete(command, 'rejected', code, 0); budget(db, 10)
    expect(db.rows.session_command_authority[sessionId]).toMatchObject({ contiguous_ack_sequence: 1, last_resulting_revision: 0, role_tails: { red: null, blue: null } })
  },
)

test('role revisions strictly increase while exact replay verifies reciprocal link and terminal result', async () => {
  const { db, device, issue, complete, apply } = await setup()
  const first = await apply()
  const release = await issue('release_role', {}); await device.claimNext(scope)
  await expect(complete(release, 'applied', 'APPLIED', 1)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await complete(release)
  db.rows.session_role_binding_history[historyId(2)].previous_id = null
  await expect(complete(first)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test('bounded ownership proof uses sealed birth/head and adjacent link, never mutable live or final roster', async () => {
  const { db, admin, apply } = await setup()
  expect(typeof verifyManagedShotOwnership).toBe('function')
  await apply(); await apply('release_role', {}); await apply()
  await admin.closeManagedSession(closeInput)
  delete db.rows.field_live_sessions[sessionId]
  const proof = (revision, id = historyId(1), actor = 'user-red') => db.runTransaction(tx => verifyManagedShotOwnership(tx, {
    receipt: { session_id: sessionId, actor_role: 'red', actor_user_id: actor, match_revision: revision },
    job: { session_id: sessionId, table_id: 'table-1', _device_id: 'device-1' }, candidateHistoryId: id,
  }))
  expect(await proof(1)).toBe(true); expect(db.transactionAttempts.at(-1).count).toBe(5)
  expect(await proof(2)).toBe(false); budget(db, 5)
  expect(await proof(3)).toBe(false); budget(db, 5)
  expect(await proof(3, historyId(3))).toBe(true); expect(db.transactionAttempts.at(-1).count).toBe(4)
  expect(await proof(1, historyId(1), 'other')).toBe(false)
  expect(db.transactionAttempts.slice(-5).flatMap(attempt => attempt.calls).some(call => call.collection === 'field_live_sessions')).toBe(false)
})

test.each([65, 128])('%i actual role transitions retain constant doc-only completion and proof budgets', async count => {
  const { db, admin, apply } = await setup()
  expect(typeof verifyManagedShotOwnership).toBe('function')
  const counts = []
  for (let i = 0; i < count; i++) {
    await apply(i % 2 ? 'release_role' : 'claim_role', i % 2 ? {} : { role: 'red' })
    budget(db, 22); counts.push(db.transactionAttempts.at(-1).count)
  }
  expect(Object.keys(db.rows.session_role_binding_history)).toHaveLength(count)
  expect(new Set(counts.slice(2).filter((_, i) => i % 2 === 0)).size).toBe(1)
  await admin.closeManagedSession(closeInput); budget(db, 10)
  const valid = await db.runTransaction(tx => verifyManagedShotOwnership(tx, {
    receipt: { session_id: sessionId, actor_role: 'red', actor_user_id: 'user-red', match_revision: 1 },
    job: { session_id: sessionId, table_id: 'table-1', _device_id: 'device-1' }, candidateHistoryId: historyId(1),
  }))
  expect(valid).toBe(true); expect(db.transactionAttempts.at(-1).count).toBe(5)
})

test.each([65, 128])('%i history with 64 storage records retains bounded proof and inventory across retry attempts', async count => {
  let retry = false
  const { db, admin, apply } = await setup({ retryConflicts: 1, beforeCommit({ attempt }) {
    if (retry && attempt.index === 0) {
      const error = new Error('synthetic callback retry'); error.code = 'SYNTHETIC_CONFLICT'; throw error
    }
  } })
  for (let i = 0; i < count; i++) {
    await apply(i % 2 ? 'release_role' : 'claim_role', i % 2 ? {} : { role: 'red' })
    budget(db, 22)
  }
  await admin.closeManagedSession(closeInput)
  const { makeStorageIdentity, createCloudMediaStorageAuthority } = require('../../cloudfunctions/shared/media-storage-authority')
  const { privateClip } = require('../helpers/private-clips')
  const clip = privateClip({ session_id: sessionId, owner_user_id: 'user-red',
    ownership_receipt: { ...privateClip().ownership_receipt, actor_user_id: 'user-red' } })
  const identity = makeStorageIdentity(clip, { tableId: 'table-1', deviceId: 'device-1', kind: 'saved_private' })
  let operation = 0
  const storage = createCloudMediaStorageAuthority(db, { clock: () => Date.parse(now),
    idFactory: () => `00000000-0000-4000-8000-${String(++operation).padStart(12, '0')}` })
  // Positive admission uses the actual trusted marked server allocation above;
  // the separate managed ownership proof below retains its full history gate.
  await db.runTransaction(tx => storage.enroll(tx, identity))
  expect(db.transactionAttempts.at(-1).count).toBe(6)
  for (let i = 0; i < 64; i++) {
    const ticket = await db.runTransaction(tx => storage.beginOperation(tx, {
      identity, ingestJobId: 'synthetic-ingest', kind: 'upload_authorization',
    }))
    await storage.recordOperationResult({ ticket, outcome: 'external_issued', objects: [{
      kind: ticket.target.kind, cloud_path: ticket.target.cloud_path, file_id: `cloud://synthetic/${i}`,
    }] })
  }
  const start = db.transactionAttempts.length
  retry = true
  const result = await db.runTransaction(async tx => {
    // Exactly one complete storage inventory; this is not a writer admission path.
    const inventory = await storage.snapshotForDeletion(tx, identity.clip_id)
    const owned = await verifyManagedShotOwnership(tx, {
      receipt: clip.ownership_receipt,
      job: { session_id: sessionId, table_id: 'table-1', _device_id: 'device-1' }, candidateHistoryId: historyId(1),
    })
    return { inventory, owned }
  })
  expect(result.owned).toBe(true)
  expect(result.inventory.objects).toHaveLength(64)
  expect(result.inventory.operations).toHaveLength(64)
  expect(Object.keys(db.rows.session_role_binding_history)).toHaveLength(count)
  expect(db.transactionAttempts.slice(start).map(attempt => attempt.count)).toEqual([74, 74])
  for (const attempt of db.transactionAttempts.slice(start)) {
    expect(attempt.calls.every(call => call.method === 'get')).toBe(true)
    const ids = attempt.calls.map(call => `${call.collection}:${call.id}`)
    expect(new Set(ids).size).toBe(ids.length)
  }
})

const { digest, saveReceipt } = require('../../cloudfunctions/shared/shot-ownership')
const { createCloudSavedClipRepository, createSavedClipJob, authorizeSavedClip, completeSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
async function publicationSetup() {
  const context = await setup()
  const { db, admin, apply } = context
  await apply(); await apply('release_role', {}); await apply()
  await admin.closeManagedSession(closeInput)
  const descriptor = structuredClone(require('../../../../tests/fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json').descriptor)
  Object.assign(descriptor, { session_id: sessionId, owner_user_id: 'user-red' })
  Object.assign(descriptor.ownership_receipt, { session_id: sessionId, actor_user_id: 'user-red', match_revision: 1 })
  descriptor.clip_id = `clip-${documentId(JSON.stringify([sessionId, descriptor.shot_id, descriptor.job_id, descriptor.artifact_id]))}`
  descriptor.ownership_receipt_sha256 = digest(descriptor.ownership_receipt)
  descriptor.save_receipt_sha256 = digest(saveReceipt(descriptor))
  db.rows.sessions = { [sessionId]: {
    session_id: sessionId, status: 'ended', sync_status: 'published', evidence_source_kind: 'usb_live',
    artifact_id: descriptor.ownership_receipt.source_artifact_id, postgame_manifest_sha256: descriptor.postgame_manifest_sha256,
    participants: [{ user_id: 'other-final-player', role: 'red' }],
    shots: [{ shot_id: descriptor.shot_id, sequence: descriptor.ownership_receipt.sequence,
      started_ms: descriptor.ownership_receipt.source_started_ms, settled_ms: descriptor.end_ms,
      live_job_id: descriptor.job_id, actor_role: 'red' }],
    _sync_publication: { table_id: scope.tableId, device_id: scope.deviceId },
  } }
  const repo = createCloudSavedClipRepository(db)
  const createInput = { ...scope, descriptor, idempotencyKey: descriptor.clip_id, repo, idFactory: () => 'saved-job-1' }
  const storage = {
    authorizeUpload: jest.fn(async () => ({ method: 'PUT', url: 'https://upload.invalid/private', headers: {} })),
    verifyObject: jest.fn(async () => ({ verified: true, file_id: 'cloud://private/saved.mp4' })),
  }
  const jobInput = { ...scope, jobId: 'saved-job-1', repo, storage }
  return { ...context, repo, descriptor, createInput, storage, jobInput }
}

test('managed saved publication and pending writer guards retain strict doc-only ownership proof', async () => {
  const { db, createInput, jobInput, storage } = await publicationSetup()
  const offset = db.transactionAttempts.length
  expect(await createSavedClipJob(createInput)).toMatchObject({ status: 'pending' })
  expect(await createSavedClipJob(createInput)).toMatchObject({ status: 'pending' })
  await expect(authorizeSavedClip(jobInput)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  await expect(completeSavedClipJob(jobInput)).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(await createSavedClipJob(createInput)).toMatchObject({ status: 'pending' })
  expect(storage.authorizeUpload).not.toHaveBeenCalled(); expect(storage.verifyObject).not.toHaveBeenCalled()
  expect(db.rows.clips[createInput.descriptor.clip_id]).toBeUndefined()
  expect(Object.values(db.rows.field_shot_ownership)[0].owner_user_id).toBe('user-red')
  for (const [index, attempt] of db.transactionAttempts.slice(offset).entries()) {
    // Fresh birth classification + publication + enrollment, then lightweight
    // binding + publication. Count repeated reads; never double-enumerate inventory.
    expect(attempt.calls.filter(call => call.collection === 'managed_session_births')).toHaveLength(index === 0 ? 3 : 2)
    expect(attempt.calls.filter(call => call.collection === 'session_command_authority')).toHaveLength(1)
    expect(attempt.calls.some(call => call.collection === 'field_live_sessions')).toBe(false)
  }
})

test.each(['create', 'validate', 'upload', 'complete', 'replay'])(
  'managed publication %s fails closed when successor invalidates the private candidate', async phase => {
    const { db, repo, createInput, jobInput } = await publicationSetup()
    if (phase !== 'create') await createSavedClipJob(createInput)
    db.rows.session_role_binding_history[historyId(2)].previous_id = null
    const action = phase === 'create' || phase === 'replay' ? () => createSavedClipJob(createInput)
      : phase === 'validate' ? async () => repo.validatePublication(await repo.get('saved-job-1'))
        : phase === 'upload' ? () => repo.markUploading('saved-job-1')
          : () => repo.markCompleted('saved-job-1', now, { file_id: 'cloud://private/saved.mp4' })
    await expect(action()).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  },
)

test.each(['managed_session_births', 'session_command_authority', 'field_live_sessions'].flatMap(collection =>
  ['claim', 'complete', 'close'].map(phase => [collection, phase])))(
  'positive lifecycle authority requires %s at %s', async (collection, phase) => {
    const { db, admin, device, issue, complete } = await setup()
    const command = await issue()
    if (phase === 'complete') await device.claimNext(scope)
    delete db.rows[collection][sessionId]
    const before = JSON.stringify(db.rows)
    const action = phase === 'claim' ? () => device.claimNext(scope)
      : phase === 'close' ? () => admin.closeManagedSession(closeInput) : () => complete(command)
    await expect(action()).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(JSON.stringify(db.rows)).toBe(before)
  },
)

test.each(['gap', 'outside-issued', 'wrong-device', 'corrupt-intent', 'missing-digest', 'malformed-prefix'])(
  'claim cannot use candidate discovery as authority: %s', async mutation => {
    const { db, device, issue } = await setup()
    const first = await issue(); const second = await issue()
    if (mutation === 'gap') delete db.rows.field_session_commands[first.command_id]
    const command = db.rows.field_session_commands[mutation === 'gap' ? second.command_id : first.command_id]
    if (mutation === 'outside-issued') command.server_sequence = 5
    if (mutation === 'corrupt-intent') command.validated_payload.role = 'blue'
    if (mutation === 'missing-digest') delete command.submitted_intent_sha256
    if (mutation === 'malformed-prefix') command.session_id = 'managed-invalid'
    const before = JSON.stringify(db.rows)
    await expect(device.claimNext(mutation === 'wrong-device' ? { ...scope, deviceId: 'other' } : scope))
      .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(JSON.stringify(db.rows)).toBe(before)
  },
)

test('out-of-order claimed command cannot ACK or advance head', async () => {
  const { db, issue, complete } = await setup()
  await issue(); const second = await issue()
  Object.assign(db.rows.field_session_commands[second.command_id], { status: 'claimed', claimed_by_device_id: scope.deviceId })
  const before = JSON.stringify(db.rows)
  await expect(complete(second, 'rejected', 'COMMAND_INVALID', 0)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(JSON.stringify(db.rows)).toBe(before)
})

test.each(['missing-node', 'missing-prev', 'missing-next', 'bad-prev', 'bad-next', 'not-tail', 'bad-result', 'bad-role'])(
  'exact ACK replay refuses %s without history repair', async mutation => {
    const { db, apply, complete } = await setup()
    await apply(); const middle = await apply('release_role', {}); await apply()
    const nodes = db.rows.session_role_binding_history
    if (mutation === 'missing-node') delete nodes[historyId(2)]
    if (mutation === 'missing-prev') delete nodes[historyId(1)]
    if (mutation === 'missing-next') delete nodes[historyId(3)]
    if (mutation === 'bad-prev') nodes[historyId(1)].next_id = null
    if (mutation === 'bad-next') nodes[historyId(3)].previous_id = null
    if (mutation === 'not-tail') nodes[historyId(2)].next_id = null
    if (mutation === 'bad-role') db.rows.field_session_commands[middle.command_id].applied_role = 'blue'
    const before = JSON.stringify(db.rows)
    await expect(complete(middle, 'applied', mutation === 'bad-result' ? 'GROUP_CORRECTION_PROPOSED' : 'APPLIED'))
      .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(JSON.stringify(db.rows)).toBe(before); budget(db, 12)
  },
)

test.each(['existing-node', 'tail-has-next', 'missing-tail', 'mismatched-tail-user', 'role-drift', 'overflow'])(
  'role append rejects %s atomically', async mutation => {
    const { db, device, issue, complete, apply } = await setup()
    await apply()
    const release = await issue('release_role', {}); await device.claimNext(scope)
    if (mutation === 'existing-node') db.rows.session_role_binding_history[historyId(2)] = { injected: true }
    if (mutation === 'tail-has-next') db.rows.session_role_binding_history[historyId(1)].next_id = historyId(2)
    if (mutation === 'missing-tail') delete db.rows.session_role_binding_history[historyId(1)]
    if (mutation === 'mismatched-tail-user') db.rows.session_role_binding_history[historyId(1)].user_id = 'other'
    if (mutation === 'role-drift') db.rows.field_live_sessions[sessionId].red_user_id = 'other'
    if (mutation === 'overflow') db.rows.session_command_authority[sessionId].revision = Number.MAX_SAFE_INTEGER
    const before = JSON.stringify(db.rows)
    await expect(complete(release)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(JSON.stringify(db.rows)).toBe(before); budget(db, 22)
  },
)

test('legal no-op group proposal ACK may equal prior revision; lower revision is refused', async () => {
  const { db, device, issue, apply, complete } = await setup()
  await apply()
  db.rows.session_user_bindings[documentId(`${sessionId}:user-blue`)] = {
    schema_version: '1.0', session_id: sessionId, table_id: 'table-1', user_id: 'user-blue', role: null, request_id: null,
  }
  const blue = await submitFieldCommand({ sessionId, actorUserId: 'user-blue', expectedRevision: 1,
    commandType: 'claim_role', payload: { role: 'blue' }, idempotencyKey: 'blue-claim-00000001', now,
  }, createCloudFieldCommandRepository(db))
  await device.claimNext(scope); await complete(blue)
  await apply('assign_groups', { red_group: 'solid' })
  const proposal = await issue('assign_groups', { red_group: 'stripe' }); await device.claimNext(scope)
  await expect(complete(proposal, 'applied', 'GROUP_CORRECTION_PROPOSED', 2)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  await complete(proposal, 'applied', 'GROUP_CORRECTION_PROPOSED', 3); budget(db, 10)
  expect(db.rows.field_live_sessions[sessionId]).toMatchObject({ red_group: 'solid', blue_group: 'stripe', control_revision: 3 })
  expect(db.rows.session_command_authority[sessionId]).toMatchObject({ contiguous_ack_sequence: 4, last_resulting_revision: 3 })
  await complete(proposal, 'applied', 'GROUP_CORRECTION_PROPOSED', 3); budget(db, 12)
})

test.each(['issue-first', 'close-first'])('concurrent %s fixes one immutable issued target', async order => {
  const { db, admin, issue, issuer } = await setup()
  const create = issuer.createCommand.bind(issuer)
  let closing
  issuer.createCommand = (...args) => {
    if (order === 'close-first') {
      closing = admin.closeManagedSession(closeInput)
      return create(...args)
    }
    const issued = create(...args)
    closing = admin.closeManagedSession(closeInput)
    return issued
  }
  if (order === 'close-first') await expect(issue()).rejects.toBeDefined()
  else expect(await issue()).toMatchObject({ server_sequence: 1 })
  await closing
  const target = order === 'issue-first' ? 1 : 0
  expect(db.rows.session_command_authority[sessionId]).toMatchObject({
    coverage: target ? 'closing' : 'sealed', closed_target: target, issued_next: target + 1,
  })
})

test.each(['pointer', 'device', 'overflow', 'head-shape'])(
  'close rejects %s while keeping issued range unchanged', async mutation => {
    const { db, admin, issue } = await setup()
    await issue()
    if (mutation === 'pointer') db.rows.table_join_codes['table-1'].current_session_id = 'new-session'
    if (mutation === 'overflow') db.rows.session_command_authority[sessionId].revision = Number.MAX_SAFE_INTEGER
    if (mutation === 'head-shape') db.rows.session_command_authority[sessionId].issued_next = 1.5
    const before = JSON.stringify(db.rows)
    await expect(admin.closeManagedSession(mutation === 'device' ? { ...closeInput, deviceId: 'other' } : closeInput))
      .rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(JSON.stringify(db.rows)).toBe(before); budget(db, 10)
  },
)

test('missing query candidate does not infer a drained prefix or seal', async () => {
  const { db, admin, device, issue } = await setup()
  const first = await issue(); await admin.closeManagedSession(closeInput)
  delete db.rows.field_session_commands[first.command_id]
  expect(await device.claimNext(scope)).toBeNull()
  expect(await admin.closeManagedSession(closeInput)).toMatchObject({ coverage: 'closing', closed_target: 1, contiguous_ack_sequence: 0 })
})

test.each(['birth-missing', 'birth-device', 'head-missing', 'head-open', 'head-tail',
  'candidate-missing', 'candidate-user', 'candidate-identity', 'candidate-extra', 'candidate-revision',
  'command-missing', 'command-code', 'command-device', 'command-sequence', 'command-payload',
  'successor-missing', 'successor-identity', 'successor-previous', 'successor-sequence', 'successor-extra'])(
  'bounded proof refuses corrupt %s', async mutation => {
    const { db, admin, apply } = await setup()
    const first = await apply(); await apply('release_role', {}); await apply()
    await admin.closeManagedSession(closeInput)
    const node = db.rows.session_role_binding_history[historyId(1)]
    const next = db.rows.session_role_binding_history[historyId(2)]
    const command = db.rows.field_session_commands[first.command_id]
    if (mutation === 'birth-missing') delete db.rows.managed_session_births[sessionId]
    if (mutation === 'birth-device') db.rows.managed_session_births[sessionId].device_id = 'other'
    if (mutation === 'head-missing') delete db.rows.session_command_authority[sessionId]
    if (mutation === 'head-open') Object.assign(db.rows.session_command_authority[sessionId], { coverage: 'open', closed_target: null, seal: null })
    if (mutation === 'head-tail') node.next_id = null
    if (mutation === 'candidate-missing') delete db.rows.session_role_binding_history[historyId(1)]
    if (mutation === 'candidate-user') node.user_id = 'other'
    if (mutation === 'candidate-identity') node.birth_id = '22345678-1234-4234-8234-123456789abc'
    if (mutation === 'candidate-extra') node.unexpected = true
    if (mutation === 'candidate-revision') node.resulting_revision = 0
    if (mutation === 'command-missing') delete db.rows.field_session_commands[first.command_id]
    if (mutation === 'command-code') command.public_result_code = 'STALE_SESSION_CLOSED'
    if (mutation === 'command-device') command.claimed_by_device_id = 'other'
    if (mutation === 'command-sequence') command.server_sequence = 3
    if (mutation === 'command-payload') command.validated_payload.role = 'blue'
    if (mutation === 'successor-missing') delete db.rows.session_role_binding_history[historyId(2)]
    if (mutation === 'successor-identity') next.session_id = 'managed-invalid'
    if (mutation === 'successor-previous') next.previous_id = null
    if (mutation === 'successor-sequence') next.server_sequence = 1
    if (mutation === 'successor-extra') next.unexpected = true
    expect(await db.runTransaction(tx => verifyManagedShotOwnership(tx, {
      receipt: { session_id: sessionId, actor_role: 'red', actor_user_id: 'user-red', match_revision: 1 },
      job: { session_id: sessionId, table_id: 'table-1', _device_id: 'device-1' }, candidateHistoryId: historyId(1),
    }))).toBe(false)
    budget(db, 5)
  },
)

test('ACK write failure rolls back all link and projection changes; transaction conflict retry appends once', async () => {
  let fail = false; let conflict = false
  const { db, device, issue, complete, apply } = await setup({
    retryConflicts: 1,
    failWrite: name => { if (fail && name === 'session_command_authority') throw new Error('synthetic write failure') },
    beforeCommit: () => { if (conflict) { conflict = false; throw Object.assign(new Error('retry'), { code: 'SYNTHETIC_CONFLICT' }) } },
  })
  await apply(); const release = await issue('release_role', {}); await device.claimNext(scope)
  const before = JSON.stringify(db.rows)
  fail = true
  await expect(complete(release)).rejects.toThrow('synthetic write failure')
  expect(JSON.stringify(db.rows)).toBe(before)
  fail = false; conflict = true
  const offset = db.transactionAttempts.length
  await complete(release)
  expect(db.transactionAttempts.slice(offset)).toHaveLength(2)
  expect(db.transactionAttempts.slice(offset).map(attempt => attempt.count)).toEqual([17, 17])
  expect(Object.keys(db.rows.session_role_binding_history)).toHaveLength(2)
  expect(db.rows.session_role_binding_history[historyId(1)].next_id).toBe(historyId(2))
})

test('managed create idempotency race revalidates hinted ownership in the transaction replay branch', async () => {
  const { db, repo, createInput } = await publicationSetup()
  await createSavedClipJob(createInput)
  const stored = await repo.get('saved-job-1')
  expect(await repo.create(stored)).toMatchObject({ status: 'pending' })
  db.rows.session_role_binding_history[historyId(2)].previous_id = null
  await expect(repo.create(stored)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test('stale or missing discovered candidate only rejects; discovery cannot provide proof authority', async () => {
  const { db, createInput } = await publicationSetup()
  const collection = db.collection.bind(db)
  for (const candidates of [[], [{ resulting_revision: 3 }], [{ resulting_revision: 2 }]]) {
    db.collection = name => {
      const value = collection(name)
      if (name === 'session_role_binding_history') value.where = () => ({
        orderBy() { return this }, limit() { return this }, async get() { return { data: candidates } },
      })
      return value
    }
    await expect(createSavedClipJob(createInput)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(Object.keys(db.rows.clips || {})).toHaveLength(0)
  }
})

test('proof propagates real SDK read failures instead of treating them as absent history', async () => {
  const { db, admin, apply } = await setup()
  await apply(); await admin.closeManagedSession(closeInput)
  const tx = { collection: () => ({ doc: () => ({ async get() { throw new Error('SDK unavailable') } }) }) }
  await expect(verifyManagedShotOwnership(tx, {
    receipt: { session_id: sessionId, actor_role: 'red', actor_user_id: 'user-red', match_revision: 1 },
    job: { session_id: sessionId, table_id: 'table-1', _device_id: 'device-1' }, candidateHistoryId: historyId(1),
  })).rejects.toThrow('SDK unavailable')
  expect(db.rows.session_command_authority[sessionId].coverage).toBe('sealed')
})

test('observed document budgets are per callback, including replay and saved-publication compositions', async () => {
  const { db, admin, device, issue, complete } = await setup()
  const first = await issue()
  await device.claimNext(scope); expect(db.transactionAttempts.at(-1).count).toBe(5)
  await device.claimNext(scope); expect(db.transactionAttempts.at(-1).count).toBe(4)
  await complete(first); expect(db.transactionAttempts.at(-1).count).toBe(15)
  await complete(first); expect(db.transactionAttempts.at(-1).count).toBe(5)
  const release = await issue('release_role', {})
  await device.claimNext(scope); await complete(release); expect(db.transactionAttempts.at(-1).count).toBe(17)
  await complete(first); expect(db.transactionAttempts.at(-1).count).toBe(6)
  await admin.closeManagedSession(closeInput); expect(db.transactionAttempts.at(-1).count).toBe(6)
  await admin.closeManagedSession(closeInput); expect(db.transactionAttempts.at(-1).count).toBe(4)
  const publication = await publicationSetup()
  const { repo } = publication
  await createSavedClipJob(publication.createInput); expect(publication.db.transactionAttempts.at(-1).count).toBe(20)
  const job = await repo.get('saved-job-1')
  await repo.validatePublication(job); expect(publication.db.transactionAttempts.at(-1).count).toBe(13)
  await expect(repo.markUploading('saved-job-1')).rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(publication.db.transactionAttempts.at(-1).count).toBe(13)
  await expect(repo.markCompleted('saved-job-1', now, { file_id: 'cloud://private/saved.mp4' }))
    .rejects.toMatchObject({ code: 'SYNC_STORAGE_UNAVAILABLE' })
  expect(publication.db.transactionAttempts.at(-1).count).toBe(13)
})

test('managed release cannot apply without the frozen role even if legacy-shaped intent hash is coherent', async () => {
  const { db, device, issue, complete, apply } = await setup()
  await apply()
  const release = await issue('release_role', {})
  const command = db.rows.field_session_commands[release.command_id]
  command.validated_payload = {}
  command.intent_digest = digest({ command_type: command.command_type,
    expected_revision: command.expected_revision, validated_payload: {} })
  await device.claimNext(scope)
  const before = JSON.stringify(db.rows)
  await expect(complete(release)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(JSON.stringify(db.rows)).toBe(before)
})

test.each(['missing-witness', 'null-interior-previous', 'mismatched-witness'])(
  'role ACK replay rejects %s using the original command relation witness', async mutation => {
    const { db, complete, apply } = await setup()
    const first = await apply(); const middle = await apply('release_role', {}); await apply()
    const command = db.rows.field_session_commands[mutation === 'missing-witness' ? first.command_id : middle.command_id]
    if (mutation === 'missing-witness') delete command.applied_previous_history_id
    if (mutation === 'null-interior-previous') db.rows.session_role_binding_history[historyId(2)].previous_id = null
    if (mutation === 'mismatched-witness') command.applied_previous_history_id = historyId(3)
    const before = JSON.stringify(db.rows)
    await expect(complete(command)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
    expect(JSON.stringify(db.rows)).toBe(before); budget(db, 12)
  },
)

test('ownership candidate must match its permanent command relation witness', async () => {
  const { db, admin, apply } = await setup()
  await apply(); await apply('release_role', {}); const last = await apply()
  await admin.closeManagedSession(closeInput)
  db.rows.session_role_binding_history[historyId(3)].previous_id = null
  expect(await db.runTransaction(tx => verifyManagedShotOwnership(tx, {
    receipt: { session_id: sessionId, actor_role: 'red', actor_user_id: 'user-red', match_revision: 3 },
    job: { session_id: sessionId, table_id: 'table-1', _device_id: 'device-1' }, candidateHistoryId: historyId(3),
  }))).toBe(false)
  expect(db.rows.field_session_commands[last.command_id].applied_previous_history_id).toBe(historyId(2))
  budget(db, 5)
})

test.each([null, historyId(1)])('proof and role replay reject a head tail inconsistent with an interior node: %p', async tail => {
  const { db, admin, apply, complete } = await setup()
  const first = await apply(); await apply('release_role', {}); await apply()
  await admin.closeManagedSession(closeInput)
  db.rows.session_command_authority[sessionId].role_tails.red = tail
  expect(await db.runTransaction(tx => verifyManagedShotOwnership(tx, {
    receipt: { session_id: sessionId, actor_role: 'red', actor_user_id: 'user-red', match_revision: 1 },
    job: { session_id: sessionId, table_id: 'table-1', _device_id: 'device-1' }, candidateHistoryId: historyId(1),
  }))).toBe(false)
  await expect(complete(first)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})
