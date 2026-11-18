const { digest, documentId } = require('./shot-ownership')
const { validOwnerDeleteDeviceTask } = require('./owner-device-deletion')
const ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,128}$/u
const HASH = /^[a-f0-9]{64}$/u
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/u
const FIELDS = ['contract','job_id','device_id','table_id','clip_id','environment_id','status','task_sha256','due_at_utc','lease_until_utc','delivery_token','revision','diagnostic']
function id(x) { return typeof x === 'string' && ID.test(x) }
function time(x) { return typeof x === 'string' && Number.isFinite(Date.parse(x)) && new Date(Date.parse(x)).toISOString() === x }
async function optional(ref) {
  try {
    const row = (await ref.get()).data ?? null
    return row === null ? null : Object.fromEntries(Object.entries(row).filter(([k]) => k !== '_id'))
  } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error?.message || '')) return null
    throw error
  }
}
function validPendingJob(job, jobId = job?.job_id) {
  return job && job.schema_version === '1.0' && job.job_kind === 'owner_delete_v1' && job.status === 'pending_cloud'
    && ['job_id','device_id','table_id','clip_id'].every(k => id(job[k])) && job.job_id === jobId
    && job.deletion_job_id === jobId && jobId === `owner-delete-${documentId(job.clip_id).slice(0,32)}` && time(job.requested_at)
}
function validDelivery(row, jobId) {
  return row && Object.keys(row).length === FIELDS.length && FIELDS.every(k => Object.hasOwn(row,k))
    && row.contract === 'owner_device_deletion_dispatch_v1' && row.job_id === jobId
    && ['job_id','device_id','table_id','clip_id'].every(k=>id(row[k]))
    && ['waiting_cloud','ready','leased','completed','blocked'].includes(row.status)
    && Number.isSafeInteger(row.revision) && row.revision >= 1 && row.revision < Number.MAX_SAFE_INTEGER
    && time(row.due_at_utc) && (row.environment_id === null || id(row.environment_id))
    && (row.task_sha256 === null || (typeof row.task_sha256==='string' && HASH.test(row.task_sha256)))
    && (row.lease_until_utc === null || time(row.lease_until_utc))
    && (row.delivery_token === null || (typeof row.delivery_token==='string' && UUID.test(row.delivery_token)))
    && [null,'invalid_metadata','invalid_job','retained_proof_unavailable','issuance_conflict'].includes(row.diagnostic)
    && (row.status !== 'waiting_cloud' || (row.task_sha256===null && row.environment_id===null && row.delivery_token===null && row.lease_until_utc===null))
    && (!['ready','leased','completed'].includes(row.status) || (id(row.environment_id) && typeof row.task_sha256==='string' && HASH.test(row.task_sha256)))
    && (row.status !== 'leased' || (time(row.lease_until_utc) && typeof row.delivery_token==='string' && UUID.test(row.delivery_token) && row.due_at_utc===row.lease_until_utc))
}
function conflict() { const error=new Error('OWNER_DEVICE_DELETION_CONFLICT'); error.code=error.message; throw error }
function sameScope(row, value) { return ['device_id','table_id','clip_id'].every(k=>row[k]===value[k]) }
async function enqueuePendingOwnerDeviceDeletion(tx, {job}) {
  if (!validPendingJob(job)) return false
  const ref=tx.collection('owner_device_deletion_deliveries').doc(job.job_id); const old=await optional(ref)
  if (old!==null) return false // Owner replays cannot reset scheduling or permanent completion.
  if (await optional(tx.collection('owner_device_deletion_receipts').doc(job.job_id)) !== null) return false
  // No environment/task authority is available until retained proof validates.
  await ref.set({data:{contract:'owner_device_deletion_dispatch_v1',job_id:job.job_id,device_id:job.device_id,table_id:job.table_id,
    clip_id:job.clip_id,environment_id:null,status:'waiting_cloud',task_sha256:null,due_at_utc:job.requested_at,
    lease_until_utc:null,delivery_token:null,revision:1,diagnostic:null}})
  return true
}
async function enqueueReadyOwnerDeviceDeletion(tx, {task,task_sha256}) {
  if (!validOwnerDeleteDeviceTask(task) || task_sha256 !== digest(task)) conflict()
  const ref=tx.collection('owner_device_deletion_deliveries').doc(task.deletion_job_id); const old=await optional(ref)
  if (old!==null && (!validDelivery(old,task.deletion_job_id) || !sameScope(old,task)
      || (old.environment_id!==null && old.environment_id!==task.environment_id)
      || (old.task_sha256!==null && old.task_sha256!==task_sha256))) conflict()
  if (old && old.status!=='waiting_cloud') return false
  if (old===null && await optional(tx.collection('owner_device_deletion_receipts').doc(task.deletion_job_id)) !== null) return false
  await ref.set({data:{contract:'owner_device_deletion_dispatch_v1',job_id:task.deletion_job_id,device_id:task.device_id,table_id:task.table_id,
    clip_id:task.clip_id,environment_id:task.environment_id,status:'ready',task_sha256,due_at_utc:'1970-01-01T00:00:00.000Z',
    lease_until_utc:null,delivery_token:null,revision:(old?.revision || 0)+1,diagnostic:null}})
  return true
}
module.exports={enqueuePendingOwnerDeviceDeletion,enqueueReadyOwnerDeviceDeletion,optional,validPendingJob,validDelivery,id,time,UUID,sameScope}
