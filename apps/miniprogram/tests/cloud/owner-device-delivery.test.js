const crypto = require('crypto')
const { createCloudOwnerDeleteRepository, requestOwnerClipDeletion } = require('../../cloudfunctions/shared/owner-delete')
const { digest } = require('../../cloudfunctions/shared/shot-ownership')
const { createOwnerCloudErasureWorker } = require('../../cloudfunctions/sync-api/lib/owner-cloud-erasure-worker')
const { fixture, provider, drain, admission, NOW, environmentId, bucket, region } = require('../helpers/owner-erasure-fixture')
const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
function dispatch() { return require('../../cloudfunctions/shared/owner-device-deletion-dispatch') }
function repository(s, options = {}) {
  return require('../../cloudfunctions/sync-api/lib/owner-device-delivery').createCloudOwnerDeviceDeliveryRepository(s.db, {
    environmentId, taskRepository: s.tasks(), clock: () => s.now, ...options,
  })
}
async function ready(s, fail = false) {
  const tasks = s.tasks()
  return drain(s, s.repository({ async onFinalizedReceipt(tx, receipt) {
    await dispatch().enqueueReadyOwnerDeviceDeletion(tx, await tasks.issueVerified(tx, receipt))
    if (fail) throw new Error('controlled callback rollback')
  } }))
}
function completion(issued, patch = {}) {
  const local_receipt = { contract: 'local_owner_erase_receipt_v1', deletion_job_id: issued.task.deletion_job_id,
    clip_id: issued.task.clip_id, task_sha256: issued.task_sha256, cloud_erase_receipt_sha256: issued.task.cloud_erase_receipt_sha256,
    journal_sha256: 'a'.repeat(64), reclaimed_bytes: 48, ...patch }
  return { contract: 'owner_delete_device_completion_v1', task_sha256: issued.task_sha256, local_receipt, local_receipt_sha256: digest(local_receipt) }
}
function runtime(s, options = {}) {
  jest.doMock('wx-server-sdk', () => ({ init() {}, DYNAMIC_CURRENT_ENV: 'synthetic', database: () => s.db }))
  jest.doMock('@cloudbase/node-sdk', () => ({ init: () => ({ getUploadMetadata() {}, downloadFile() {} }), SYMBOL_CURRENT_ENV: 'synthetic' }))
  let api
  jest.isolateModules(() => { api = require('../../cloudfunctions/sync-api') })
  const deps = api.createRuntimeDependencies({ db: s.db, storageCloud: { getUploadMetadata() {}, downloadFile() {} }, environment: { CLOUDBASE_ENV_ID: environmentId }, clock: () => s.now, ...options })
  return { deps, main: api.createHandler(deps) }
}
let nonce = 0
function signed(s, path, value, deviceId = 'device-1', { raw, base64 = false } = {}) {
  const body = raw === undefined ? JSON.stringify(value) : raw
  const timestamp = String(Math.floor(s.now / 1000)); const requestNonce = `owner-delivery-nonce-${++nonce}`
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const signature = crypto.createHmac('sha256', 'controlled-secret').update(canonicalSignatureInput({ method: 'POST', path, timestamp, nonce: requestNonce, bodyHash })).digest('hex')
  return { httpMethod: 'POST', path, body: base64 ? Buffer.from(body).toString('base64') : body, isBase64Encoded: base64,
    headers: { 'x-device-id': deviceId, 'x-timestamp': timestamp, 'x-nonce': requestNonce, 'x-body-sha256': bodyHash, 'x-signature': signature } }
}
function devices(s) {
  s.db.rows.sync_devices = Object.fromEntries([['device-1','table-1'], ['device-2','table-1'], ['device-3','table-2']].map(([device_id,table_id]) => [device_id,{ device_id, table_id, secret: 'controlled-secret', status: 'active' }]))
}
afterEach(() => { jest.dontMock('wx-server-sdk'); jest.dontMock('@cloudbase/node-sdk') })
const claimPath = '/v1/owner-media-deletions/claim'
const pollBody = { contract: 'owner_delete_device_poll_v1' }

test('actual HMAC nonce route accepts exact poll with default native worker absent', async () => {
  const s = { db: ownerDeleteDb({}, { strictTransactions: true, queryIds: true }), now: NOW }; devices(s)
  const h = runtime(s); const event = signed(s, claimPath, pollBody); const result = await h.main(event)
  expect(result.statusCode).toBe(200); expect(JSON.parse(result.body)).toEqual({ ok: true, data: null })
  expect(h.deps.ownerCloudErasureWorker).toBeNull(); expect((await h.main(event)).statusCode).toBe(401)
})

test('owner request replay restores missing waiting metadata without changing original job or ready state', async () => {
  const s = await fixture({ dbOptions: { queryIds: true } })
  expect(s.db.rows.owner_device_deletion_deliveries?.[s.jobId]).toMatchObject({ status: 'waiting_cloud', device_id: 'device-1', table_id: 'table-1' })
  const original = structuredClone(s.db.rows.owner_clip_deletion_jobs)
  delete s.db.rows.owner_device_deletion_deliveries[s.jobId]
  const request = requestId => requestOwnerClipDeletion({ clipId: s.descriptor.clip_id, requestId, userId: 'user-a', repo: createCloudOwnerDeleteRepository(s.db), now: new Date(s.now).toISOString() })
  await request('owner-request'); expect(s.db.rows.owner_device_deletion_deliveries[s.jobId].status).toBe('waiting_cloud')
  expect(s.db.rows.owner_clip_deletion_jobs).toEqual(original)
  await ready(s); const row = structuredClone(s.db.rows.owner_device_deletion_deliveries[s.jobId])
  await request('replay-2'); expect(s.db.rows.owner_device_deletion_deliveries[s.jobId]).toEqual(row)
})

test('exact envelope and offline expired rotated lease completion produce immutable stable ACK', async () => {
  const s = await fixture({ dbOptions: { queryIds: true } }); await ready(s); devices(s)
  const h = runtime(s); const poll = async () => JSON.parse((await h.main(signed(s, claimPath, pollBody))).body).data
  const first = await poll()
  expect(Object.keys(first).sort()).toEqual(['contract','task','task_sha256','delivery_token','lease_until_utc'].sort())
  expect(first.contract).toBe('owner_delete_device_delivery_v1'); expect(first.delivery_token).toMatch(/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/)
  expect(first.task_sha256).toBe(digest(first.task)); expect(Object.keys(first.task)).toHaveLength(19)
  expect(await poll()).toBeNull(); s.now = Date.parse(first.lease_until_utc) + 1
  const second = await poll(); expect(second.delivery_token).not.toBe(first.delivery_token)
  s.now = Date.parse(second.lease_until_utc) + 100000
  const body = completion(first); const path = `/v1/owner-media-deletions/${s.jobId}/complete`
  const result = await h.main(signed(s, path, body)); expect(result.statusCode).toBe(200)
  const ack = JSON.parse(result.body).data
  expect(Object.keys(ack).sort()).toEqual(['contract','deletion_job_id','clip_id','task_sha256','cloud_erase_receipt_sha256','local_receipt_sha256','received_at_utc'].sort())
  expect(ack).toMatchObject({ contract:'owner_delete_device_ack_v1',received_at_utc:new Date(s.now).toISOString(),local_receipt_sha256:body.local_receipt_sha256 })
  const before = structuredClone(s.db.rows); const start = s.db.transactionAttempts.length; s.now++
  expect(JSON.parse((await h.main(signed(s, path, body))).body).data).toEqual(ack)
  delete before.sync_nonces; const after = structuredClone(s.db.rows); delete after.sync_nonces; expect(after).toEqual(before)
  expect(s.db.transactionAttempts.slice(start).flatMap(a=>a.calls).filter(c=>c.method!=='get').every(c=>c.collection==='sync_nonces')).toBe(true)
  expect((await h.main(signed(s,path,completion(first,{reclaimed_bytes:49})))).statusCode).toBe(409)
})

test.each(['device-2','device-3'])('authenticated %s cannot receive or complete original task', async foreign => {
  const s = await fixture({ dbOptions:{queryIds:true} }); await ready(s); devices(s); const h=runtime(s); const issued=await s.ensure()
  expect(JSON.parse((await h.main(signed(s,claimPath,pollBody,foreign))).body).data).toBeNull()
  expect((await h.main(signed(s,`/v1/owner-media-deletions/${s.jobId}/complete`,completion(issued),foreign))).statusCode).toBe(409)
})

test('strict wire fields types hashes and raw16KiB bound reject before new protocol parsing', async () => {
  const s=await fixture({dbOptions:{queryIds:true}}); await ready(s); devices(s); const h=runtime(s); const issued=await s.ensure(); const path=`/v1/owner-media-deletions/${s.jobId}/complete`
  for(const body of [{},{...pollBody,receipt:{}},{contract:['owner_delete_device_poll_v1']}]) expect((await h.main(signed(s,claimPath,body))).statusCode).toBe(400)
  for(const body of [completion(issued,{reclaimed_bytes:true}),completion(issued,{reclaimed_bytes:-1}),completion(issued,{reclaimed_bytes:9007199254740992}),completion(issued,{journal_sha256:'A'.repeat(64)}),completion(issued,{journal:[]}),{...completion(issued),task_sha256:'0'.repeat(64)},{...completion(issued),local_receipt_sha256:'0'.repeat(64)},{...completion(issued),receipt:{}}]) expect((await h.main(signed(s,path,body))).statusCode).toBe(400)
  for(const base64 of [false,true]) expect((await h.main(signed(s,claimPath,null,'device-1',{raw:' '.repeat(16385)+'{bad',base64}))).statusCode).toBe(400)
  expect(Object.keys(s.db.rows.owner_device_deletion_receipts||{})).toHaveLength(0)
})

test.each(['missing-proof','missing-task','unknown-job','foreign-env','corrupt-proof'])('completion rejects %s with retained proof',async kind=>{
  const s=await fixture({dbOptions:{queryIds:true}}); await ready(s); const issued=await s.ensure(); let repo=repository(s)
  if(kind==='missing-proof') delete s.db.rows.owner_cloud_erasure_receipts[s.jobId]
  if(kind==='missing-task') delete s.db.rows.owner_device_deletion_tasks[s.jobId]
  if(kind==='foreign-env') repo=repository(s,{environmentId:'foreign'})
  if(kind==='corrupt-proof') s.db.rows.owner_cloud_erasure_witnesses[s.jobId].birth_sha256='f'.repeat(64)
  await expect(repo.complete({...s.scope,jobId:kind==='unknown-job'?'unknown-job':s.jobId,body:completion(issued)})).rejects.toMatchObject({code:'SYNC_CONFLICT'})
})

test('callback failure rolls back receipt witness task and ready metadata together',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); const waiting=structuredClone(s.db.rows.owner_device_deletion_deliveries)
  await expect(ready(s,true)).rejects.toThrow('controlled callback rollback'); expect(s.db.rows.owner_device_deletion_deliveries).toEqual(waiting)
  for(const c of ['owner_cloud_erasure_receipts','owner_cloud_erasure_witnesses','owner_device_deletion_tasks']) expect(Object.keys(s.db.rows[c]||{})).toHaveLength(0)
  s.now+=30001; await ready(s); expect(s.db.rows.owner_device_deletion_deliveries[s.jobId].status).toBe('ready')
})

test('fixed upper sweep advances past20 bad historical rows with continuous new arrivals',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); await drain(s); delete s.db.rows.owner_device_deletion_deliveries[s.jobId]
  for(let i=0;i<20;i++) s.db.rows.owner_clip_deletion_jobs[`bad-${String(i).padStart(2,'0')}`]={device_id:'device-1',table_id:'table-1',job_id:'wrong',requested_at:new Date(NOW).toISOString()}
  const repo=repository(s); let found=await repo.poll(s.scope)
  for(let i=0;i<28&&!found;i++) { s.now++; s.db.rows.owner_clip_deletion_jobs[`aaa-new-${i}`]={device_id:'device-1',table_id:'table-1',job_id:'wrong',requested_at:new Date(s.now).toISOString()}; s.db.rows.owner_clip_deletion_jobs[`zzz-new-${i}`]={device_id:'device-1',table_id:'table-1',job_id:'wrong',requested_at:new Date(s.now).toISOString()}; found=await repo.poll(s.scope) }
  expect(found.task.deletion_job_id).toBe(s.jobId)
  const cursor=Object.values(s.db.rows.owner_device_deletion_repair_cursors)[0]; expect(cursor.cursor).toBe(s.jobId); expect(cursor.upper).toBe(s.jobId)
  expect(s.db.transactionAttempts.every(a=>a.count<=90)).toBe(true)
})

test('ready inspections quarantine20 invalid heads and concurrent retry polls deliver once',async()=>{
  let retry=false
  const s=await fixture({dbOptions:{queryIds:true,retryConflicts:1,beforeCommit({attempt}) { if(retry&&attempt.index===0) { const e=new Error('retry'); e.code='SYNTHETIC_CONFLICT'; throw e } }}}); await ready(s)
  const good=structuredClone(s.db.rows.owner_device_deletion_deliveries[s.jobId])
  for(let i=0;i<20;i++) s.db.rows.owner_device_deletion_deliveries[`bad-${i}`]={...good,job_id:'wrong',due_at_utc:'1960-01-01T00:00:00.000Z'}
  const repo=repository(s); retry=true; let result=null
  for(let i=0;i<8&&!result;i++) result=await repo.poll(s.scope)
  expect(result.task.deletion_job_id).toBe(s.jobId); expect(Object.values(s.db.rows.owner_device_deletion_deliveries).filter(x=>x.status==='blocked')).toHaveLength(20)
  s.now=Date.parse(result.lease_until_utc)+1
  const results=await Promise.all([repo.poll(s.scope),repo.poll(s.scope)]); expect(results.filter(Boolean)).toHaveLength(1)
})

test('optional trusted worker advances original same-device waiting job at most one provider step per poll',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); const tasks=s.tasks(); const native=provider(s)
  const worker=createOwnerCloudErasureWorker({repository:s.repository({async onFinalizedReceipt(tx,r) {await dispatch().enqueueReadyOwnerDeviceDeletion(tx,await tasks.issueVerified(tx,r))}}),provider:native})
  const repo=repository(s,{cloudErasureWorker:worker})
  expect(await repo.poll({tableId:'table-1',deviceId:'device-2'})).toBeNull(); expect(native.calls).toBe(0)
  await repo.poll(s.scope); expect(native.calls).toBe(1); await repo.poll(s.scope); expect(native.calls).toBe(1)
  s.now+=5001; await repo.poll(s.scope); expect(native.calls).toBe(2)
})

test('completed permanent receipt prevents missing delivery repair or callback replay from redelivering',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); await ready(s); const issued=await s.ensure(); const repo=repository(s)
  await repo.complete({...s.scope,jobId:s.jobId,body:completion(issued)})
  delete s.db.rows.owner_device_deletion_deliveries[s.jobId]
  await requestOwnerClipDeletion({clipId:s.descriptor.clip_id,requestId:'owner-request',userId:'user-a',repo:createCloudOwnerDeleteRepository(s.db),now:new Date(s.now).toISOString()})
  await ready(s)
  expect(await repo.poll(s.scope)).toBeNull()
  expect(s.db.rows.owner_device_deletion_deliveries[s.jobId]?.status).not.toBe('waiting_cloud')
})

test('raw byte exact limit accepts signed whitespace padding; oversized response is fixed conflict',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); devices(s); const h=runtime(s)
  const json=JSON.stringify(pollBody)
  const result=await h.main(signed(s,claimPath,null,'device-1',{raw:' '.repeat(16384-Buffer.byteLength(json))+json,base64:true}))
  expect(result.statusCode).toBe(200)
  h.deps.ownerDeviceDeliveryRepo.poll=async()=>({payload:'x'.repeat(16384)})
  const tooLarge=await h.main(signed(s,claimPath,pollBody)); expect(tooLarge.statusCode).toBe(409)
  expect(JSON.parse(tooLarge.body)).toEqual({ok:false,error:{code:'SYNC_CONFLICT'}})
})

test('device moved table and missing runtime environment deny original delivery authority',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); await ready(s); devices(s); const h=runtime(s); const issued=await s.ensure()
  s.db.rows.sync_devices['device-1'].table_id='replacement-table'
  expect(JSON.parse((await h.main(signed(s,claimPath,pollBody))).body).data).toBeNull()
  expect((await h.main(signed(s,`/v1/owner-media-deletions/${s.jobId}/complete`,completion(issued)))).statusCode).toBe(409)
  await expect(repository(s,{environmentId:null}).poll(s.scope)).rejects.toMatchObject({code:'SYNC_INVALID'})
})

test('historical sweep wraps after fixed upper and observes deferred later arrival',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); await drain(s); delete s.db.rows.owner_device_deletion_deliveries[s.jobId]
  const job=structuredClone(s.db.rows.owner_clip_deletion_jobs[s.jobId]); delete s.db.rows.owner_clip_deletion_jobs[s.jobId]
  s.db.rows.owner_clip_deletion_jobs.bad={device_id:'device-1',table_id:'table-1',requested_at:new Date(NOW).toISOString()}
  const repo=repository(s); expect(await repo.poll(s.scope)).toBeNull()
  s.now++
  // Preserve original authority: legitimate newly visible original row retains
  // original timestamp. The cursor must still defer keys above its fixed upper.
  s.db.rows.owner_clip_deletion_jobs[s.jobId]=job
  expect(await repo.poll(s.scope)).toBeNull()
  const cursor=Object.values(s.db.rows.owner_device_deletion_repair_cursors)[0]; expect(cursor.upper).toBeNull()
  expect(await repo.poll(s.scope)).toBeNull() // New sweep starts at bad key again.
  expect((await repo.poll(s.scope)).task.deletion_job_id).toBe(s.jobId)
})

test('queries remain scoped bounded outside document-only retries and do not grant pending task',async()=>{
  const s=await fixture({dbOptions:{queryIds:true}}); const repo=repository(s); const calls=[]; const original=s.db.collection
  s.db.collection=name=>{
    const q=original(name); const record={name,orders:[]}; calls.push(record)
    const where=q.where,orderBy=q.orderBy,limit=q.limit
    q.where=function(value){record.filter=value;return where.call(this,value)}
    q.orderBy=function(key,direction){record.orders.push([key,direction]);return orderBy.call(this,key,direction)}
    q.limit=function(value){record.limit=value;return limit.call(this,value)}
    return q
  }
  expect(await repo.poll(s.scope)).toBeNull()
  expect(Object.keys(s.db.rows.owner_device_deletion_tasks||{})).toHaveLength(0)
  for(const query of calls.filter(c=>c.filter)) {
    const text=JSON.stringify(query.filter); expect(text).toContain('device-1'); expect(text).toContain('table-1')
    expect([1,4]).toContain(query.limit)
  }
  expect(calls.filter(c=>c.filter&&c.name==='owner_clip_deletion_jobs')).toHaveLength(2)
  expect(s.db.transactionAttempts.every(a=>a.count<=90)).toBe(true)
})

test('real runtime HMAC poll invokes one explicitly bound provider step and same-device metadata only',async()=>{
  const {EventEmitter}=require('events'); const {Readable}=require('stream')
  const s=await fixture({dbOptions:{queryIds:true}}); devices(s); const requests=[]
  const request=(options,onResponse)=>{
    requests.push(options); const req=new EventEmitter(); req.destroy=()=>{}; req.end=()=>queueMicrotask(()=>{
      const res=Readable.from([Buffer.from('<VersioningConfiguration/>')]); res.statusCode=200; res.headers={}; res.rawHeaders=[]; onResponse(res)
    }); return req
  }
  const h=runtime(s,{ownerCloudErasure:{nativeContractAdmission:admission,providerOptions:{environmentId,bucket,region,
    getCredentials:()=>({SecretId:'controlled-id',SecretKey:'controlled-secret'}),request}}})
  const foreign=await h.main(signed(s,claimPath,pollBody,'device-2')); expect(foreign.statusCode).toBe(200); expect(requests).toHaveLength(0)
  const result=await h.main(signed(s,claimPath,pollBody)); expect(result.statusCode).toBe(200); expect(JSON.parse(result.body).data).toBeNull()
  expect(requests).toHaveLength(1); expect(s.db.rows.owner_cloud_erasure_executions[s.jobId].checkpoint).toMatchObject({mode:'NeverEnabled',phase:'simple_delete'})
})

test('full64 callback issuance witness ready enqueue and delivery completion remain <=90 under forced retry',async()=>{
  let retry=false; let failCallback=true
  const s=await fixture({full:true,dbOptions:{queryIds:true,retryConflicts:1,beforeCommit({attempt}) {
    if(retry&&attempt.index===0) { const error=new Error('controlled conflict'); error.code='SYNTHETIC_CONFLICT'; throw error }
  }}})
  const tasks=s.tasks(); const native=provider(s); const waiting=structuredClone(s.db.rows.owner_device_deletion_deliveries)
  const cloud=s.repository({async onFinalizedReceipt(tx,r) {
    await dispatch().enqueueReadyOwnerDeviceDeletion(tx,await tasks.issueVerified(tx,r))
    if(failCallback) throw new Error('full64 callback rollback')
  }})
  const start=s.db.transactionAttempts.length; retry=true
  await expect(drain(s,cloud,native)).rejects.toThrow('full64 callback rollback')
  expect(s.db.rows.owner_device_deletion_deliveries).toEqual(waiting)
  for(const c of ['owner_cloud_erasure_receipts','owner_cloud_erasure_witnesses','owner_device_deletion_tasks']) expect(Object.keys(s.db.rows[c]||{})).toHaveLength(0)
  failCallback=false; s.now+=30001
  const receipt=await drain(s,cloud,native); expect(receipt.targets).toHaveLength(64); expect(native.calls).toBe(64)
  await drain(s,cloud,{step(){throw new Error('replay cannot invoke native')}})
  const workerEnd=s.db.transactionAttempts.length; const repo=repository(s); const delivered=await repo.poll(s.scope)
  const body=completion(delivered); const ack=await repo.complete({...s.scope,jobId:s.jobId,body})
  expect(await repo.complete({...s.scope,jobId:s.jobId,body})).toEqual(ack)
  const attempts=s.db.transactionAttempts.slice(start); const finalizations=attempts.filter(a=>a.calls.some(c=>c.collection==='owner_cloud_erasure_receipts'&&c.method==='set'))
  expect(finalizations).toHaveLength(3)
  for(const a of attempts) {
    expect(a.count).toBeLessThanOrEqual(90)
    expect(a.calls.filter(c=>c.collection==='managed_upload_requests')).toHaveLength(0)
    expect(a.calls.filter(c=>c.method!=='get').every(c=>['owner_cloud_erasure_executions','owner_cloud_erasure_receipts','owner_cloud_erasure_witnesses','owner_device_deletion_tasks','owner_device_deletion_deliveries','owner_device_deletion_repair_cursors','owner_device_deletion_receipts'].includes(c.collection))).toBe(true)
  }
  for(const a of s.db.transactionAttempts.slice(start,workerEnd)) {
    expect(a.calls.filter(c=>c.collection==='media_storage_objects'&&c.method==='get')).toHaveLength(64)
    expect(a.calls.filter(c=>c.collection==='upload_terminal_receipts')).toHaveLength(4)
  }
  console.info('OWNER_DEVICE_DELIVERY_FULL64',{attempts:attempts.length,min:Math.min(...attempts.map(a=>a.count)),max:Math.max(...attempts.map(a=>a.count)),retries:attempts.filter(a=>a.index===1).length,finalizations:finalizations.map(a=>a.count),native:native.calls})
},120000)

test('malformed retained ACK cannot be overwritten and a failed completion write rolls back all settlement',async()=>{
  let fail=false
  const s=await fixture({dbOptions:{queryIds:true,failWrite(name){if(fail&&name==='owner_device_deletion_deliveries') throw new Error('controlled settlement failure')}}}); await ready(s)
  const issued=await s.ensure(); const body=completion(issued); const repo=repository(s); const original=structuredClone(s.db.rows.owner_device_deletion_deliveries)
  fail=true; await expect(repo.complete({...s.scope,jobId:s.jobId,body})).rejects.toThrow('controlled settlement failure')
  expect(Object.keys(s.db.rows.owner_device_deletion_receipts||{})).toHaveLength(0); expect(s.db.rows.owner_device_deletion_deliveries).toEqual(original)
  fail=false; await repo.complete({...s.scope,jobId:s.jobId,body})
  const permanent=s.db.rows.owner_device_deletion_receipts[s.jobId]; permanent.ack.received_at_utc='invalid'
  const before=structuredClone(permanent)
  await expect(repo.complete({...s.scope,jobId:s.jobId,body})).rejects.toMatchObject({code:'SYNC_CONFLICT'})
  expect(s.db.rows.owner_device_deletion_receipts[s.jobId]).toEqual(before)
})

// Inject only controlled database boundary failures; auth, nonce repositories,
// route dispatch and settlement still execute their production implementations.
function failDatabaseReads(db, collectionName, enabled) {
  const collection = db.collection
  db.collection = name => {
    const ref = collection(name)
    if (name !== collectionName) return ref
    const get = ref.get
    ref.get = async function () {
      if (enabled()) throw new Error('controlled private database diagnostic')
      return get.call(this)
    }
    const doc = ref.doc
    ref.doc = function (key) {
      const document = doc.call(this, key); const read = document.get
      document.get = async function () {
        if (enabled()) throw new Error('controlled private database diagnostic')
        return read.call(this)
      }
      return document
    }
    return ref
  }
}
function expectStorageUnavailable(result) {
  expect(result.statusCode).toBe(503)
  expect(JSON.parse(result.body)).toEqual({ ok: false, error: { code: 'SYNC_STORAGE_UNAVAILABLE' } })
  expect(result.body).not.toContain('controlled private')
}

test.each(['poll', 'complete'])('actual signed owner %s transaction failure returns retryable fixed503 and recovers', async operation => {
  let fail = false
  const s = await fixture({ dbOptions: { queryIds: true, failWrite(name) {
    if (fail && name === 'owner_device_deletion_deliveries') throw new Error('controlled private settlement diagnostic')
  } } })
  await ready(s); devices(s)
  const h = runtime(s); h.deps.logger = { error: jest.fn() }
  const issued = await s.ensure()
  const path = operation === 'poll' ? claimPath : `/v1/owner-media-deletions/${s.jobId}/complete`
  const body = operation === 'poll' ? pollBody : completion(issued)
  const original = structuredClone(s.db.rows.owner_device_deletion_deliveries)
  fail = true
  expectStorageUnavailable(await h.main(signed(s, path, body)))
  expect(s.db.rows.owner_device_deletion_deliveries).toEqual(original)
  expect(Object.keys(s.db.rows.owner_device_deletion_receipts || {})).toHaveLength(0)
  fail = false
  const retry = await h.main(signed(s, path, body))
  expect(retry.statusCode).toBe(200)
  expect(JSON.parse(retry.body).data.contract).toBe(operation === 'poll' ? 'owner_delete_device_delivery_v1' : 'owner_delete_device_ack_v1')
})

test.each([
  ['poll', 'device-read'], ['poll', 'nonce-write'],
  ['complete', 'device-read'], ['complete', 'nonce-write'],
])('actual signed owner %s %s fault is retryable before repository dispatch', async (operation, fault) => {
  let fail = false
  const s = await fixture({ dbOptions: { queryIds: true, failWrite(name) {
    if (fail && fault === 'nonce-write' && name === 'sync_nonces') throw new Error('controlled private nonce diagnostic')
  } } })
  await ready(s); devices(s)
  if (fault === 'device-read') failDatabaseReads(s.db, 'sync_devices', () => fail)
  const h = runtime(s); h.deps.logger = { error: jest.fn() }
  const issued = await s.ensure()
  const path = operation === 'poll' ? claimPath : `/v1/owner-media-deletions/${s.jobId}/complete`
  const body = operation === 'poll' ? pollBody : completion(issued)
  const original = structuredClone(s.db.rows.owner_device_deletion_deliveries)
  const event = signed(s, path, body)
  fail = true
  expectStorageUnavailable(await h.main(event))
  expect(s.db.rows.owner_device_deletion_deliveries).toEqual(original)
  expect(Object.keys(s.db.rows.sync_nonces || {})).toHaveLength(0)
  expect(Object.keys(s.db.rows.owner_device_deletion_receipts || {})).toHaveLength(0)
  fail = false
  expect((await h.main(event)).statusCode).toBe(200) // Failed auth storage did not consume the nonce.
})

test('owner operational boundary preserves exact validation authentication replay and conflict errors', async () => {
  const s = await fixture({ dbOptions: { queryIds: true } }); await ready(s); devices(s)
  const h = runtime(s); const issued = await s.ensure()
  const invalid = await h.main(signed(s, claimPath, { ...pollBody, receipt: {} }))
  expect(invalid.statusCode).toBe(400); expect(JSON.parse(invalid.body)).toEqual({ ok: false, error: { code: 'SYNC_INVALID' } })
  const unsigned = signed(s, claimPath, pollBody); unsigned.headers['x-signature'] = '0'.repeat(64)
  const unauthorized = await h.main(unsigned)
  expect(unauthorized.statusCode).toBe(401); expect(JSON.parse(unauthorized.body)).toEqual({ ok: false, error: { code: 'SYNC_UNAUTHORIZED' } })
  const event = signed(s, claimPath, pollBody); expect((await h.main(event)).statusCode).toBe(200)
  const replay = await h.main(event)
  expect(replay.statusCode).toBe(401); expect(JSON.parse(replay.body)).toEqual({ ok: false, error: { code: 'SYNC_REPLAYED' } })
  const conflict = await h.main(signed(s, `/v1/owner-media-deletions/${s.jobId}/complete`, completion(issued), 'device-2'))
  expect(conflict.statusCode).toBe(409); expect(JSON.parse(conflict.body)).toEqual({ ok: false, error: { code: 'SYNC_CONFLICT' } })
})

test.each(['sync_devices', 'media_deletion_jobs'])('legacy route %s operational failure retains unknown400 response', async collection => {
  const s = { db: ownerDeleteDb({}, { strictTransactions: true, queryIds: true }), now: NOW }; devices(s)
  failDatabaseReads(s.db, collection, () => true)
  const h = runtime(s); h.deps.logger = { error: jest.fn() }
  const result = await h.main(signed(s, '/v1/media-deletions/claim', {}))
  expect(result.statusCode).toBe(400)
  expect(JSON.parse(result.body)).toEqual({ ok: false, error: { code: 'INTERNAL_ERROR' } })
  expect(result.body).not.toContain('controlled private')
})
