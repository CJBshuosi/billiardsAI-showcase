const crypto = require('crypto')
const { SyncApiError } = require('./auth')
const { canonicalJson, digest, documentId } = require('../../shared/shot-ownership')
const { enqueuePendingOwnerDeviceDeletion, enqueueReadyOwnerDeviceDeletion, optional, validPendingJob, validDelivery, id, time, UUID, sameScope } = require('../../shared/owner-device-deletion-dispatch')
const LIMIT = 16 * 1024
const LEASE_MS = 30000
const WAIT_MS = 5000
const HASH = /^[a-f0-9]{64}$/u
function exact(value, fields) {
  return Boolean(value && [Object.prototype,null].includes(Object.getPrototypeOf(value)) && Reflect.ownKeys(value).length===fields.length
    && fields.every(k=>{const d=Object.getOwnPropertyDescriptor(value,k);return d&&d.enumerable&&Object.hasOwn(d,'value')}))
}
function fail(code='SYNC_CONFLICT') { throw new SyncApiError(code) }
function bounded(value) { return Buffer.byteLength(JSON.stringify(value),'utf8')<=LIMIT }
function validCompletion(body) {
  const r=body?.local_receipt
  return exact(body,['contract','task_sha256','local_receipt','local_receipt_sha256']) && body.contract==='owner_delete_device_completion_v1'
    && exact(r,['contract','deletion_job_id','clip_id','task_sha256','cloud_erase_receipt_sha256','journal_sha256','reclaimed_bytes'])
    && r.contract==='local_owner_erase_receipt_v1' && id(r.deletion_job_id) && id(r.clip_id)
    && ['task_sha256','cloud_erase_receipt_sha256','journal_sha256'].every(k=>typeof r[k]==='string'&&HASH.test(r[k]))
    && typeof body.local_receipt_sha256==='string' && HASH.test(body.local_receipt_sha256)
    && body.task_sha256===r.task_sha256 && body.local_receipt_sha256===digest(r)
    && Number.isSafeInteger(r.reclaimed_bytes) && r.reclaimed_bytes>=0 && bounded(body)
}
function sameScopeArgs(row,{tableId,deviceId}) { return row?.device_id===deviceId && row?.table_id===tableId }
function knownConflict(error) { return error?.code==='OWNER_DEVICE_DELETION_CONFLICT' || error?.code==='OWNER_DEVICE_DELETION_INVALID' }
function createCloudOwnerDeviceDeliveryRepository(db,{environmentId,taskRepository,cloudErasureWorker=null,clock=Date.now,idFactory=()=>crypto.randomUUID()}={}) {
  const tasks=taskRepository
  function now() { const n=clock(); if(!Number.isSafeInteger(n)||n<0||n>8640000000000000-LEASE_MS) fail('SYNC_INVALID'); return n }
  function scopeValid(scope) { return id(environmentId) && id(scope.tableId) && id(scope.deviceId) }
  async function quarantine(tx,ref,row,diagnostic) {
    await ref.update({data:{status:'blocked',diagnostic,revision:Number.isSafeInteger(row.revision)&&row.revision<Number.MAX_SAFE_INTEGER?row.revision+1:1}})
  }
  async function originalJob(tx,key,scope) {
    const job=await optional(tx.collection('owner_clip_deletion_jobs').doc(key))
    if(!validPendingJob(job,key)||!sameScopeArgs(job,scope)) return null
    const tombstone=await optional(tx.collection('owner_clip_tombstones').doc(job.clip_id))
    const {job_id,deletion_job_id,job_kind,status,...frozen}=job
    return tombstone && canonicalJson(frozen)===canonicalJson(tombstone) ? job : null
  }
  async function repair(scope,stamp) {
    const key=documentId(canonicalJson([environmentId,scope.deviceId,scope.tableId]))
    const ref=db.collection('owner_device_deletion_repair_cursors').doc(key)
    const previous=await optional(ref)
    const valid=previous && previous.contract==='owner_device_deletion_repair_v1'
      && previous.environment_id===environmentId && sameScopeArgs(previous,scope)
      && typeof previous.cursor==='string' && (previous.upper===null||typeof previous.upper==='string')
      && time(previous.upper_time_utc) && Number.isSafeInteger(previous.revision) && previous.revision>=1 && previous.revision<Number.MAX_SAFE_INTEGER
    let upper=valid?previous.upper:null; const cursor=valid?previous.cursor:''
    const upperTime=valid && upper!==null?previous.upper_time_utc:stamp
    const filter={device_id:scope.deviceId,table_id:scope.tableId,requested_at:db.command.lte(upperTime)}
    if(upper===null) {
      const result=await db.collection('owner_clip_deletion_jobs').where(filter).orderBy('_id','desc').limit(1).get()
      upper=result.data?.[0]?._id ?? null
    }
    let candidate=null
    if(upper!==null) {
      const result=await db.collection('owner_clip_deletion_jobs').where(db.command.and([filter,{_id:db.command.gt(cursor)},{_id:db.command.lte(upper)}])).orderBy('_id','asc').limit(1).get()
      candidate=result.data?.[0] || null
    }
    // Persist authoritative key advancement before validating the candidate.
    // Fixed upper prevents new lower/higher arrivals from restarting this sweep.
    await db.runTransaction(async tx=>{
      const cursorRef=tx.collection('owner_device_deletion_repair_cursors').doc(key)
      const current=await optional(cursorRef)
      if(canonicalJson(current)!==canonicalJson(previous)) return
      await cursorRef.set({data:{contract:'owner_device_deletion_repair_v1',environment_id:environmentId,device_id:scope.deviceId,table_id:scope.tableId,
        cursor:candidate?candidate._id:'',upper:candidate?upper:null,upper_time_utc:upperTime,revision:(valid?previous.revision:0)+1}})
    })
    if(!candidate) return
    // Independent doc-only callback: validation failure cannot roll back progress.
    // A stale concurrent sweep is harmless; immutable IDs and issuance revalidate.
    await db.runTransaction(async tx=>{
      if(!id(candidate._id)) return
      const job=await originalJob(tx,candidate._id,scope)
      if(!job) return
      await enqueuePendingOwnerDeviceDeletion(tx,{job})
      let issued
      try { issued=await tasks.ensureInTransaction(tx,{jobId:job.job_id,...scope}) }
      catch(error) { if(knownConflict(error)) return; throw error }
      if(issued && issued.task.environment_id===environmentId && sameScopeArgs(issued.task,scope)) {
        try { await enqueueReadyOwnerDeviceDeletion(tx,issued) } catch(error) { if(!knownConflict(error)) throw error }
      }
    })
  }
  async function advanceWaiting(scope,stamp) {
    if(!cloudErasureWorker) return
    const result=await db.collection('owner_device_deletion_deliveries').where({device_id:scope.deviceId,table_id:scope.tableId,status:'waiting_cloud',due_at_utc:db.command.lte(stamp)}).orderBy('due_at_utc','asc').orderBy('_id','asc').limit(1).get()
    const candidate=result.data?.[0]; if(!candidate) return
    const jobId=await db.runTransaction(async tx=>{
      const ref=tx.collection('owner_device_deletion_deliveries').doc(candidate._id); const row=await optional(ref)
      if(!row || !sameScopeArgs(row,scope) || row.status!=='waiting_cloud' || row.due_at_utc>stamp) return null
      if(!validDelivery(row,candidate._id)) { await quarantine(tx,ref,row,'invalid_metadata'); return null }
      const job=await originalJob(tx,candidate._id,scope)
      if(!job || !sameScope(row,job)) { await quarantine(tx,ref,row,'invalid_job'); return null }
      await ref.update({data:{due_at_utc:new Date(Date.parse(stamp)+WAIT_MS).toISOString(),revision:row.revision+1}})
      return job.job_id
    })
    if(jobId) await cloudErasureWorker.runOnce({jobId}) // At most one provider step, outside every transaction.
  }
  return {
    async poll(scope) {
      if(!scopeValid(scope)) fail('SYNC_INVALID')
      const stamp=new Date(now()).toISOString()
      await repair(scope,stamp)
      await advanceWaiting(scope,stamp)
      const result=await db.collection('owner_device_deletion_deliveries').where(db.command.and([
        {device_id:scope.deviceId,table_id:scope.tableId,environment_id:environmentId,due_at_utc:db.command.lte(stamp)},
        db.command.or([{status:'ready'},{status:'leased'}]),
      ])).orderBy('due_at_utc','asc').orderBy('_id','asc').limit(4).get()
      for(const candidate of result.data || []) {
        const envelope=await db.runTransaction(async tx=>{
          const ref=tx.collection('owner_device_deletion_deliveries').doc(candidate._id); const row=await optional(ref)
          if(!row || !sameScopeArgs(row,scope) || row.environment_id!==environmentId || !['ready','leased'].includes(row.status) || row.due_at_utc>stamp) return null
          if(!validDelivery(row,candidate._id)) { await quarantine(tx,ref,row,'invalid_metadata'); return null }
          let issued
          try { issued=await tasks.readVerified(tx,{jobId:candidate._id,...scope}) }
          catch(error) { if(!knownConflict(error)) throw error; await quarantine(tx,ref,row,'issuance_conflict'); return null }
          if(!issued || issued.task.environment_id!==environmentId || issued.task_sha256!==row.task_sha256 || !sameScope(row,issued.task)) { await quarantine(tx,ref,row,'retained_proof_unavailable'); return null }
          const token=idFactory(); if(typeof token!=='string'||!UUID.test(token)) fail('SYNC_INVALID')
          const lease=new Date(Date.parse(stamp)+LEASE_MS).toISOString()
          const value={contract:'owner_delete_device_delivery_v1',...issued,delivery_token:token,lease_until_utc:lease}
          if(!bounded(value)) fail('SYNC_CONFLICT')
          await ref.update({data:{status:'leased',delivery_token:token,lease_until_utc:lease,due_at_utc:lease,revision:row.revision+1,diagnostic:null}})
          return value
        })
        if(envelope) return envelope
      }
      return null
    },
    async complete({jobId,tableId,deviceId,body}) {
      if(!id(jobId)||!scopeValid({tableId,deviceId})||!validCompletion(body)) fail('SYNC_INVALID')
      try { return await db.runTransaction(async tx=>{
        const issued=await tasks.readVerified(tx,{jobId,tableId,deviceId})
        if(!issued || issued.task.environment_id!==environmentId || !sameScopeArgs(issued.task,{tableId,deviceId})) fail()
        const task=issued.task; const receipt=body.local_receipt
        if(body.task_sha256!==issued.task_sha256 || receipt.deletion_job_id!==jobId || receipt.clip_id!==task.clip_id || receipt.cloud_erase_receipt_sha256!==task.cloud_erase_receipt_sha256) fail()
        const ref=tx.collection('owner_device_deletion_receipts').doc(jobId); const old=await optional(ref)
        if(old!==null) {
          const ack=old.ack
          if(!exact(old,['contract','deletion_job_id','environment_id','device_id','table_id','local_receipt','local_receipt_sha256','ack'])
            || old.contract!=='owner_device_deletion_receipt_v1' || old.deletion_job_id!==jobId || old.environment_id!==environmentId || !sameScopeArgs(old,{tableId,deviceId})
            || canonicalJson(old.local_receipt)!==canonicalJson(receipt) || old.local_receipt_sha256!==body.local_receipt_sha256
            || !exact(ack,['contract','deletion_job_id','clip_id','task_sha256','cloud_erase_receipt_sha256','local_receipt_sha256','received_at_utc'])
            || ack.contract!=='owner_delete_device_ack_v1' || !time(ack.received_at_utc)
            || ['deletion_job_id','clip_id','task_sha256','cloud_erase_receipt_sha256'].some(k=>ack[k]!==receipt[k]) || ack.local_receipt_sha256!==body.local_receipt_sha256) fail()
          return ack
        }
        const ack={contract:'owner_delete_device_ack_v1',deletion_job_id:jobId,clip_id:task.clip_id,task_sha256:issued.task_sha256,
          cloud_erase_receipt_sha256:task.cloud_erase_receipt_sha256,local_receipt_sha256:body.local_receipt_sha256,received_at_utc:new Date(now()).toISOString()}
        await ref.set({data:{contract:'owner_device_deletion_receipt_v1',deletion_job_id:jobId,environment_id:environmentId,device_id:deviceId,table_id:tableId,
          local_receipt:receipt,local_receipt_sha256:body.local_receipt_sha256,ack}})
        const deliveryRef=tx.collection('owner_device_deletion_deliveries').doc(jobId); const delivery=await optional(deliveryRef)
        // Completion relies on issuance + retained proof, independent of a lost,
        // rotated, expired, missing or blocked delivery lease.
        await deliveryRef.set({data:{contract:'owner_device_deletion_dispatch_v1',job_id:jobId,device_id:deviceId,table_id:tableId,clip_id:task.clip_id,
          environment_id:environmentId,status:'completed',task_sha256:issued.task_sha256,due_at_utc:ack.received_at_utc,lease_until_utc:null,delivery_token:null,
          revision:Number.isSafeInteger(delivery?.revision)&&delivery.revision<Number.MAX_SAFE_INTEGER?delivery.revision+1:1,diagnostic:null}})
        return ack
      }) } catch(error) { if(knownConflict(error)) fail(); throw error }
    },
  }
}
module.exports={createCloudOwnerDeviceDeliveryRepository,validCompletion,exact,LIMIT}
