const { canonicalJson, digest, documentId } = require('./shot-ownership')
const { createOwnerCloudErasureRepository } = require('./owner-cloud-erasure')

const TASK_FIELDS = ['contract', 'deletion_job_id', 'environment_id', 'device_id', 'table_id', 'session_id', 'shot_id', 'source_job_id', 'artifact_id', 'clip_id', 'owner_user_id', 'actor_role', 'content_sha256', 'byte_size', 'ownership_receipt_sha256', 'save_request_id', 'save_receipt_sha256', 'postgame_manifest_sha256', 'cloud_erase_receipt_sha256']
const ISSUANCE_FIELDS = ['contract', 'deletion_job_id', 'environment_id', 'device_id', 'table_id', 'clip_id', 'task', 'task_sha256', 'cloud_erase_receipt_sha256', 'issued_at_utc']
const HASH = /^[a-f0-9]{64}$/u
const ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,128}$/u
const SESSION = /^managed-[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/u
const hash = value => typeof value === 'string' && HASH.test(value)
const id = value => typeof value === 'string' && ID.test(value)
const copy = value => JSON.parse(canonicalJson(value))
function record(value, fields) {
  return Boolean(value && [Object.prototype, null].includes(Object.getPrototypeOf(value))
    && Reflect.ownKeys(value).length === fields.length && fields.every(key => {
      const d = Object.getOwnPropertyDescriptor(value, key)
      return d && d.enumerable && Object.hasOwn(d, 'value')
    }))
}
class OwnerDeviceDeletionError extends Error {
  constructor(suffix) { super(`OWNER_DEVICE_DELETION_${suffix}`); this.name = 'OwnerDeviceDeletionError'; this.code = this.message }
}
function fail(suffix = 'INVALID') { throw new OwnerDeviceDeletionError(suffix) }
// Original cloud IDs are ASCII. With this stricter subset of Python task
// identities, sorted JS UTF-8 JSON equals Python's sorted ensure_ascii bytes.
function validOwnerDeleteDeviceTask(value) {
  return record(value, TASK_FIELDS) && value.contract === 'owner_delete_device_v1'
    && TASK_FIELDS.filter(k => !['byte_size', 'save_request_id'].includes(k)).every(k => id(value[k]))
    && typeof value.save_request_id === 'string' && /^[A-Za-z0-9._-]{1,128}$/u.test(value.save_request_id)
    && SESSION.test(value.session_id) && ['red', 'blue'].includes(value.actor_role)
    && ['artifact_id', 'content_sha256', 'ownership_receipt_sha256', 'save_receipt_sha256', 'postgame_manifest_sha256', 'cloud_erase_receipt_sha256'].every(k => hash(value[k]))
    && Number.isSafeInteger(value.byte_size) && value.byte_size > 0 && value.byte_size <= 256 * 1024 * 1024
    && value.clip_id === `clip-${digest([value.session_id, value.shot_id, value.source_job_id, value.artifact_id])}`
    && value.deletion_job_id === `owner-delete-${documentId(value.clip_id).slice(0, 32)}`
}
// Pure codec, not proof validation. Callers must use the retained-proof seam or
// the trusted finalization callback before persisting an issuance.
function makeOwnerDeleteDeviceTask(receipt) {
  if (!receipt || receipt.contract !== 'owner_cloud_erase_receipt_v1' || !receipt.scope) fail()
  const task = Object.fromEntries(TASK_FIELDS.map(key => [key, receipt.scope[key]]))
  Object.assign(task, { contract: 'owner_delete_device_v1', deletion_job_id: receipt.deletion_job_id,
    environment_id: receipt.environment_id, cloud_erase_receipt_sha256: receipt.receipt_sha256 })
  if (!validOwnerDeleteDeviceTask(task)) fail()
  return copy(task)
}
async function optional(ref) {
  try {
    const row = (await ref.get()).data ?? null
    return row === null ? null : Object.fromEntries(Object.entries(row).filter(([key]) => key !== '_id'))
  } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error?.message || '')) return null
    throw error
  }
}
function validIssuance(value) {
  return record(value, ISSUANCE_FIELDS) && value.contract === 'owner_device_deletion_issuance_v1'
    && validOwnerDeleteDeviceTask(value.task) && value.task_sha256 === digest(value.task)
    && ['deletion_job_id', 'environment_id', 'device_id', 'table_id', 'clip_id', 'cloud_erase_receipt_sha256'].every(k => value[k] === value.task[k])
    && typeof value.issued_at_utc === 'string' && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/u.test(value.issued_at_utc)
    && Number.isFinite(Date.parse(value.issued_at_utc))
}
function createCloudOwnerDeviceTaskRepository(db, { environmentId, authority, clock = Date.now } = {}) {
  const proof = createOwnerCloudErasureRepository(db, { authority })
  // Internal composition capability only: receipt was validated in this same
  // finalization transaction. Never expose it as a route or accept request proof.
  async function issueVerified(tx, receipt) {
    if (!id(environmentId) || receipt?.environment_id !== environmentId) fail()
    const task = makeOwnerDeleteDeviceTask(receipt); const task_sha256 = digest(task)
    const ref = tx.collection('owner_device_deletion_tasks').doc(task.deletion_job_id)
    const old = await optional(ref)
    if (old !== null) {
      if (!validIssuance(old) || canonicalJson(old.task) !== canonicalJson(task) || old.task_sha256 !== task_sha256) fail('CONFLICT')
      return copy({ task: old.task, task_sha256: old.task_sha256 })
    }
    const now = clock()
    if (!Number.isSafeInteger(now) || now < 0 || now > 8640000000000000) fail()
    await ref.set({ data: { contract: 'owner_device_deletion_issuance_v1', deletion_job_id: task.deletion_job_id,
      environment_id: task.environment_id, device_id: task.device_id, table_id: task.table_id, clip_id: task.clip_id,
      task, task_sha256, cloud_erase_receipt_sha256: task.cloud_erase_receipt_sha256, issued_at_utc: new Date(now).toISOString() } })
    return copy({ task, task_sha256 })
  }
  function validArgs(args) {
    return id(environmentId) && record(args, ['jobId', 'tableId', 'deviceId']) && Object.values(args).every(id)
  }
  async function ensureInTransaction(tx, args) {
    if (!validArgs(args)) return null
    const receipt = await proof.readRetainedReceipt(tx, { ...args, environmentId })
    return receipt ? issueVerified(tx, receipt) : null
  }
  return {
    issueVerified,
    ensureInTransaction,
    async readVerified(tx, args) {
      if (!validArgs(args)) return null
      const receipt = await proof.readRetainedReceipt(tx, { ...args, environmentId })
      if (!receipt) return null
      const old = await optional(tx.collection('owner_device_deletion_tasks').doc(args.jobId))
      if (old === null) return null
      const task = makeOwnerDeleteDeviceTask(receipt)
      if (!validIssuance(old) || canonicalJson(old.task) !== canonicalJson(task)) fail('CONFLICT')
      return copy({ task: old.task, task_sha256: old.task_sha256 })
    },
    async ensure(args) {
      if (!validArgs(args)) return null
      return db.runTransaction(tx => ensureInTransaction(tx, args))
    },
  }
}
module.exports = { makeOwnerDeleteDeviceTask, validOwnerDeleteDeviceTask, createCloudOwnerDeviceTaskRepository }
