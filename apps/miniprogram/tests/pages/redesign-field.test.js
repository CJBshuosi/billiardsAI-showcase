function setup(call) {
  let definition
  const values=new Map()
  global.Page=value=>{definition=value}
  global.wx={
    cloud:{callFunction:call},
    getRandomValues:({success})=>success({randomValues:new Uint8Array(16).fill(3).buffer}),
    getStorageSync:key=>values.get(key),setStorageSync:(key,value)=>values.set(key,value),removeStorageSync:key=>values.delete(key),
    showModal:jest.fn(),
  }
  jest.isolateModules(()=>require('../../miniprogram/pages/current-session/index'))
  const page={...definition,data:{...definition.data},setData(value){Object.assign(this.data,value)}}
  page.onLoad({request_id:'join-a'})
  Object.assign(page.data,{sessionId:'s-a',status:'open',loading:false,revision:2})
  return {page,values}
}
afterEach(()=>{jest.useRealTimers();delete global.Page;delete global.wx})
const ok=data=>({result:{ok:true,data}})

test('rapid role taps submit only one command before the cloud receipt arrives', async()=>{
  let resolve;const requests=[]
  const {page}=setup(({data})=>{requests.push(data);return new Promise(r=>{resolve=r})})
  const first=page.claimRole({currentTarget:{dataset:{role:'red'}}})
  const second=page.claimRole({currentTarget:{dataset:{role:'blue'}}})
  await new Promise(setImmediate)
  expect(requests).toHaveLength(1)
  resolve(ok({command_id:'c-a',status:'pending'}));await first;await second
  expect(page.data.commandState).toBe('pending')
  expect(page.data.actorRole).toBeNull()
})

test('choosing a breaker only edits the draft until both players confirm settings', async()=>{
  const requests=[];const {page}=setup(async({data})=>{requests.push(data);return ok({command_id:'c-a',status:'pending'})})
  Object.assign(page.data,{actorRole:'red',redOccupied:true,blueOccupied:true})
  page.setBreaker({currentTarget:{dataset:{role:'blue'}}})
  expect(requests).toHaveLength(0)
  expect(page.data.selectedBreaker).toBe('blue')
})

test('an applied breaker command keeps both choices locked until field state catches up', async()=>{
  const requests=[];const {page}=setup(async({data})=>{requests.push(data);return ok({command_id:'c-a',status:'applied'})})
  Object.assign(page.data,{actorRole:'red',redOccupied:true,blueOccupied:true,
    matchState:{status:'ready',settings:{first_breaker:'red'}}})
  await page.startMatch()
  expect(page.data.commandState).toBe('applied')
  await page.startMatch()
  expect(requests).toHaveLength(1)
  expect(page.data.breakerLocked).toBe(true)
})

test('ended sessions cannot submit old controls even when invoked directly',async()=>{
  const requests=[];const {page}=setup(async({data})=>{requests.push(data);return ok({command_id:'c-a',status:'pending'})})
  Object.assign(page.data,{status:'ended',actorRole:'red'})
  await page.continueTurn();await page.claimRole({currentTarget:{dataset:{role:'red'}}})
  expect(requests).toEqual([])
})

test('either joined role can open the end-session confirmation',()=>{
  const {page}=setup(async()=>ok({command_id:'c-a',status:'pending'}))
  Object.assign(page.data,{actorRole:'blue',activeRole:'red',started:true})
  page.endSession()
  expect(wx.showModal).toHaveBeenCalledTimes(1)
  page.data.activeRole='blue';page.endSession()
  expect(wx.showModal).toHaveBeenCalledTimes(2)
})

test('a lost submission response retries exactly the same intent rather than generating a second command',async()=>{
  const requests=[];const {page}=setup(async({data})=>{
    requests.push(data)
    if(requests.length===1) throw Error('lost response')
    return ok({command_id:'c-a',status:'pending'})
  })
  await page.claimRole({currentTarget:{dataset:{role:'red'}}})
  expect(page.data.commandState).toBe('unknown')
  await page.retrySubmission()
  expect(requests).toHaveLength(2)
  expect(requests[1].payload).toEqual(requests[0].payload)
  expect(page.data.commandState).toBe('pending')
})

test('an unknown intent remains unresolved if a retry sees a changed revision',async()=>{
  let count=0
  const {page}=setup(async()=>{count++;if(count===1)throw Error('lost');return {result:{ok:false,error:{code:'COMMAND_STALE_REVISION'}}}})
  await page.claimRole({currentTarget:{dataset:{role:'red'}}});await page.retrySubmission()
  expect(page.data.commandState).toBe('unknown')
  expect(page.data.actorRole).toBeNull()
})

test('hiding stops polling and prevents an old read from replacing visible state',async()=>{
  jest.useFakeTimers()
  let resolve
  const {page}=setup(()=>new Promise(r=>{resolve=r}))
  page.onShow();page.onHide()
  expect(jest.getTimerCount()).toBe(0)
  resolve(ok({session_id:'s-old',display_name:'旧用户'}))
  await Promise.resolve();await Promise.resolve()
  expect(page.data.sessionId).toBe('s-a')
})

test('restored pending commands remain locked until their authoritative status is terminal',async()=>{
  const {page,values}=setup(async({data})=>{
    if(data.action==='join.status')return ok({session_id:'s-a',display_name:'球友甲'})
    if(data.action==='field.current')return ok({session_id:'s-a',status:'open',revision:2,actor_role:null,red_occupied:false,blue_occupied:false})
    return ok({command_id:'c-a',status:'claimed'})
  })
  values.set('field-command:s-a',{commandId:'c-a',state:'pending'})
  await page.refresh()
  expect(page.data.commandId).toBe('c-a');expect(page.data.commandState).toBe('claimed')
})
