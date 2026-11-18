const crypto = require('crypto')
const { managedMediaBirthFixture, sessionId, now, input: birthInput } = require('../helpers/managed-media-admission-fixture')
const { createCloudJoinRepository, requestJoin } = require('../../cloudfunctions/client-api/lib/join')
const { createCloudFieldCommandRepository, submitFieldCommand } = require('../../cloudfunctions/client-api/lib/field-commands')
const { createCloudDeviceFieldCommandRepository } = require('../../cloudfunctions/sync-api/lib/field-commands')
const { digest, documentId, saveReceipt } = require('../../cloudfunctions/shared/shot-ownership')
const { makeStorageIdentity, createCloudMediaStorageAuthority } = require('../../cloudfunctions/shared/media-storage-authority')
const { createSavedClipJob, createCloudSavedClipRepository,
  createSyncJob, createCloudSyncRepository, completeSyncJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const source = require('../../../../tests/fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json')

// Real allocation, join, command ACK/seal and facts publication. The shot and
// large-screen receipt are synthetic input, not seeded cloud ownership/admission.
async function setup({ marked = true, dbOptions = {} } = {}) {
  const { db, admin } = managedMediaBirthFixture({ managedMediaAdmissionEnabled: marked }, dbOptions)
  const code = '12345678901234567890123456789012'
  db.rows.table_join_codes['table-1'].code_digest = documentId(code)
  await admin.createManagedSession(birthInput)
  await requestJoin({ tableCode: code, userId: 'user-a', now, repo: createCloudJoinRepository(db) })
  const command = await submitFieldCommand({ sessionId, actorUserId: 'user-a', expectedRevision: 0,
    commandType: 'claim_role', payload: { role: 'red' }, idempotencyKey: 'managed-admission-claim-0001', now,
  }, createCloudFieldCommandRepository(db))
  const device = createCloudDeviceFieldCommandRepository(db)
  await device.claimNext({ tableId: 'table-1', deviceId: 'device-1', now })
  await device.complete({ commandId: command.command_id, tableId: 'table-1', deviceId: 'device-1',
    status: 'applied', publicResultCode: 'APPLIED', resultingRevision: 1, now })
  await admin.closeManagedSession({ ...birthInput, sessionId })
  const descriptor = structuredClone(source.descriptor)
  descriptor.session_id = sessionId
  descriptor.clip_id = `clip-${crypto.createHash('sha256').update(JSON.stringify([
    sessionId, descriptor.shot_id, descriptor.job_id, descriptor.artifact_id,
  ])).digest('hex')}`
  descriptor.ownership_receipt.session_id = sessionId
  descriptor.ownership_receipt_sha256 = digest(descriptor.ownership_receipt)
  descriptor.save_receipt_sha256 = digest(saveReceipt(descriptor))
  const session = { ...structuredClone(source.published_session), session_id: sessionId, limitations: [] }
  session.participants = session.participants.slice(0, 1)
  session.shots = session.shots.slice(0, 1)
  delete session._sync_publication; delete session.sync_status
  const facts = { schema_version: '1.0', session, media: [] }
  const factsRepo = createCloudSyncRepository(db)
  const scope = { tableId: 'table-1', deviceId: 'device-1' }
  const factJob = await createSyncJob({ ...scope, envelope: facts, idempotencyKey: 'managed-facts',
    manifestHash: 'f'.repeat(64), repo: factsRepo, idFactory: () => 'facts-job', now })
  await completeSyncJob({ ...scope, jobId: factJob.job_id, repo: factsRepo, storage: {}, now })
  const repo = createCloudSavedClipRepository(db)
  const input = { ...scope, descriptor, idempotencyKey: descriptor.clip_id, repo, idFactory: () => 'saved-job', now }
  const identity = makeStorageIdentity(descriptor, { ...scope, kind: 'saved_private' })
  const authority = createCloudMediaStorageAuthority(db)
  return { db, repo, input, descriptor, identity, authority, scope, facts }
}
module.exports = { setup }
