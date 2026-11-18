const crypto = require('crypto')
const { setup } = require('./managed-saved-flow-fixture')
const { createSavedClipJob, authorizeSavedClip, completeSavedClipJob } = require('../../cloudfunctions/sync-api/lib/ingest')
const { createCloudOwnerDeleteRepository, requestOwnerClipDeletion } = require('../../cloudfunctions/shared/owner-delete')
const { digest } = require('../../cloudfunctions/shared/shot-ownership')
const { createOwnerCloudErasureRepository } = require('../../cloudfunctions/shared/owner-cloud-erasure')
const { createOwnerCloudErasureWorker } = require('../../cloudfunctions/sync-api/lib/owner-cloud-erasure-worker')
const device = require('../../cloudfunctions/shared/owner-device-deletion')
const NOW = Date.parse('2026-11-01T10:00:00.000Z')
const environmentId = 'env-test'; const bucket = 'media-123456'; const region = 'ap-guangzhou'
const file = path => `cloud://${environmentId}.${bucket}/${path}`
function admission({ scope, assessment }) {
  return { contract: 'owner_erasure_native_admission_v1', scope_sha256: digest(scope), inputs_sha256: assessment.inputs_sha256,
    targets_sha256: digest(assessment.targets), environment_id: environmentId, bucket, region, provider_policy: 'cos_exact_target_absence_v1',
    evidence: { binding_sha256: '1'.repeat(64), writer_drain_sha256: '2'.repeat(64), visibility_sha256: '3'.repeat(64), configuration_fence_sha256: '4'.repeat(64), stale_request_idempotence_sha256: '5'.repeat(64) } }
}
// Real managed birth, ownership, save, storage catalog and owner deletion. Only
// external storage/native I/O is synthetic; no deployed acceptance is asserted.
async function fixture({ full = false, optional = 0, retired = [], dbOptions = {} } = {}) {
  const s = await setup({ dbOptions }); s.now = NOW
  s.job = await createSavedClipJob(s.input)
  s.optional = []
  for (let i = 0; i < optional; i++) {
    const uploadRequest = { contract: 'managed_upload_request_v1', request_id: crypto.randomUUID() }
    const { ticket } = await s.repo.beginTrackedUploadRequest({ ...s.scope, jobId: s.job.job_id, uploadRequest })
    await s.repo.recordTrackedResult({ ticket, outcome: 'no_write', objects: [], reason: 'not_started' })
    s.optional.push({ request: uploadRequest, attempt: { contract: 'managed_upload_attempt_v1', operation_id: ticket.operation_id, sequence: ticket.sequence } })
  }
  s.retire = (index, nonnull = false) => s.repo.recordTrackedUploadReceipt({ ...s.scope, jobId: s.job.job_id, receipt: {
    contract: 'managed_upload_receipt_v1', upload_request: s.optional[index].request, upload_attempt: nonnull ? s.optional[index].attempt : null,
    evidence: { kind: 'retired_before_start', local_binding_sha256: 'a'.repeat(64) },
  } })
  for (const index of retired) await s.retire(index)
  const storage = {
    authorizeRegisteredUpload: async x => ({ outcome: 'external_issued', file_id: file(x.object_key), authorization: { method: 'PUT', url: 'https://upload.invalid', headers: {}, expires_in_seconds: 300 } }),
    copyRegisteredObject: async x => ({ outcome: 'server_succeeded', file_id: file(x.publish_object_key) }),
  }
  for (let i = 0; i < (full ? 63 : 1); i++) {
    const request = { contract: 'managed_upload_request_v1', request_id: crypto.randomUUID() }
    const grant = await authorizeSavedClip({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage, uploadRequest: request })
    s.first ||= grant
    await s.repo.recordTrackedUploadReceipt({ ...s.scope, jobId: s.job.job_id, receipt: {
      contract: 'managed_upload_receipt_v1', upload_request: request, upload_attempt: grant.upload_attempt,
      evidence: { kind: 'http200', local_binding_sha256: 'a'.repeat(64), started_at_utc: '2024-02-29T23:59:59.123456Z', received_at_utc: '2024-02-29T23:59:59.123457Z', status_code: 200 },
    } })
  }
  await completeSavedClipJob({ ...s.scope, jobId: s.job.job_id, repo: s.repo, storage, uploadAttempt: s.first.upload_attempt, now: new Date(NOW).toISOString() })
  const receipt = await requestOwnerClipDeletion({ clipId: s.descriptor.clip_id, requestId: 'owner-request', userId: 'user-a', repo: createCloudOwnerDeleteRepository(s.db), now: new Date(NOW).toISOString() })
  s.jobId = receipt.deletion_job_id
  s.repository = options => createOwnerCloudErasureRepository(s.db, { authority: s.authority, clock: () => s.now, nativeContractAdmission: admission, ...options })
  s.tasks = () => device.createCloudOwnerDeviceTaskRepository(s.db, { environmentId, authority: s.authority, clock: () => s.now })
  s.read = (options = {}, repository = s.repository({ nativeContractAdmission: null })) => s.db.runTransaction(tx => repository.readRetainedReceipt(tx, { jobId: s.jobId, ...s.scope, environmentId, ...options }))
  s.ensure = (options = {}) => s.tasks().ensure({ jobId: s.jobId, ...s.scope, ...options })
  return s
}
// Native absence evidence is a controlled external boundary. All claim, progress,
// finalization, issuance and transaction accounting are the production code.
function provider(s) {
  let calls = 0
  return { get calls() { return calls }, async step({ target, beforeRequest }) {
    if (!await beforeRequest()) throw new Error('stale request')
    calls++
    return { status: 'absent', checkpoint: { contract: 'owner_erasure_provider_v1', target_sha256: digest(target), mode: 'NeverEnabled', phase: 'recheck', cursor: null, cursor_sha256: [], pages: 0, versions: [], transcript_sha256: 'e'.repeat(64) },
      evidence: { target_sha256: digest(target), versioning_mode: 'NeverEnabled', observed_at: new Date(s.now).toISOString(), transcript_sha256: 'e'.repeat(64) } }
  } }
}
async function drain(s, repository = s.repository(), native = provider(s)) {
  const worker = createOwnerCloudErasureWorker({ repository, provider: native })
  for (let i = 0; i < 70; i++) { const result = await worker.runOnce({ jobId: s.jobId }); if (result.receipt) return result.receipt }
  throw new Error('controlled fixture did not finalize')
}
module.exports = { fixture, provider, drain, admission, NOW, environmentId, bucket, region, file }
