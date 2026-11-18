const path = require('path')
const vm = require('vm')
const compiler = require('miniprogram-compiler')
const tick = () => new Promise(setImmediate)
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r }); return { promise, resolve } }
const ok = data => ({ ok: true, data })
const receipt = (clip_id = 'clip-1', request_id = 'original-request') => ({ schema_version: '1.0', clip_id, request_id,
  deletion_job_id: 'owner-delete-job', status: 'deletion_requested', requested_at: '2026-10-30T10:00:00.000Z' })
let render
beforeAll(() => {
  const source = compiler.wxmlToJs(path.resolve(__dirname, '../../miniprogram'))
  render = vm.runInNewContext(`(function(){${source}})()`, { window: {}, console })('pages/clip-detail/index.wxml')
})
afterEach(() => { delete global.Page; delete global.wx; delete global.getCurrentPages; jest.dontMock('wx-server-sdk') })
function instantiate(name) {
  let definition; global.Page = value => { definition = value }
  jest.isolateModules(() => require(`../../miniprogram/pages/${name}/index`))
  return { ...definition, data: { ...definition.data }, setData: jest.fn(function(value, callback) {
    Object.assign(this.data, value); if (callback) callback()
  }) }
}
function setup({ responder, storage = new Map(), confirm = true } = {}) {
  const stop = jest.fn()
  global.wx = {
    cloud: { callFunction: jest.fn(async ({ data }) => ({ result: responder ? await responder(data) : data.action === 'clips.delete'
      ? ok(receipt(data.payload.clip_id)) : ok({ url: 'https://private.invalid/clip.mp4' }) })) },
    getRandomValues: jest.fn(({ length, success }) => { expect(length).toBe(16); success({ randomValues: new Uint8Array(16).fill(7).buffer }) }),
    getStorageSync: jest.fn(key => storage.get(key)), setStorageSync: jest.fn((key, value) => storage.set(key, value)),
    removeStorageSync: jest.fn(key => storage.delete(key)), showModal: jest.fn(({ success }) => success({ confirm })),
    createVideoContext: jest.fn(() => ({ stop })), pageScrollTo: jest.fn(), navigateBack: jest.fn(), navigateTo: jest.fn(),
  }
  const detail = instantiate('clip-detail'); detail.route = 'pages/clip-detail/index'
  detail.onLoad({ clip_id: 'clip-1' })
  return { detail, storage, stop, deletes: () => wx.cloud.callFunction.mock.calls.filter(([r]) => r.data.action === 'clips.delete') }
}
function nodes(tree) { return [tree, ...(tree.children || []).flatMap(nodes)] }
function tapDelete(detail) {
  const button = nodes(render(detail.data)).find(node => node.tag === 'wx-button' && node.attr.bindtap === 'requestDeletion')
  expect(button).toBeDefined()
  return detail[button.attr.bindtap]()
}

test('rendered owner-delete control confirms; cancellation sends no delete and keeps video', async () => {
  const { detail, deletes, stop } = setup({ confirm: false }); await tick()
  await tapDelete(detail)
  expect(deletes()).toHaveLength(0); expect(stop).not.toHaveBeenCalled()
  expect(detail.data.playbackUrl).toBe('https://private.invalid/clip.mp4')
  expect(wx.showModal.mock.calls[0][0].content).toContain('存储清理')
  expect(JSON.stringify(render(detail.data))).toContain('本人删除')
})

test('duplicate taps lock confirmation/API; stop and remove player before stable intent is submitted', async () => {
  const response = deferred(); const modal = deferred(); let detail; let storage
  const context = setup({ responder: async ({ action, payload }) => {
    if (action !== 'clips.delete') return ok({ url: 'https://private.invalid/clip.mp4' })
    expect(detail.data.playbackUrl).toBe(''); expect(detail.data.authorized).toBe(false)
    expect(context.stop).toHaveBeenCalled(); expect(storage.size).toBe(1)
    expect([...storage.values()][0]).toEqual(payload)
    expect(nodes(render(detail.data)).some(n => n.tag === 'wx-video')).toBe(false)
    return response.promise
  } }); ({ detail, storage } = context); await tick()
  wx.showModal.mockImplementation(({ success }) => modal.promise.then(success))
  const first = detail.requestDeletion(); const second = detail.requestDeletion()
  expect(wx.showModal).toHaveBeenCalledTimes(1)
  modal.resolve({ confirm: true }); await tick()
  await detail.requestDeletion(); expect(context.deletes()).toHaveLength(1)
  response.resolve(ok(receipt())); await Promise.all([first, second])
  expect(detail.data.deletionState).toBe('accepted'); expect(storage.size).toBe(0)
  expect(detail.data.deletionMessage).toContain('存储清理处理中')
  expect(JSON.stringify(render(detail.data))).not.toContain('永久擦除')
})

test.each(['rng', 'short-rng', 'read', 'write'])('%s failure prevents request with unstable intent', async failure => {
  const { detail, deletes } = setup(); await tick()
  if (failure === 'rng') wx.getRandomValues.mockImplementation(({ fail }) => fail(new Error('private entropy error')))
  if (failure === 'short-rng') wx.getRandomValues.mockImplementation(({ success }) => success({ randomValues: new Uint8Array(2).buffer }))
  if (failure === 'read') wx.getStorageSync.mockImplementation(() => { throw new Error('private read error') })
  if (failure === 'write') wx.setStorageSync.mockImplementation(() => { throw new Error('private write error') })
  await detail.requestDeletion()
  expect(deletes()).toHaveLength(0); expect(detail.data.deletionState).toBe('intent_error')
  expect(detail.data.deletionMessage).not.toContain('private')
})

test('lost acknowledgement/reopen reuses persisted request, blocks playback and accepts original receipt with different request ID', async () => {
  let fail = true
  const { detail, storage, deletes } = setup({ responder: ({ action, payload }) => {
    if (action === 'clips.delete') { if (fail) throw new Error('lost ACK'); return ok(receipt(payload.clip_id)) }
    return ok({ url: 'https://private.invalid/clip.mp4' })
  } }); await tick(); await detail.requestDeletion()
  const request = deletes()[0][0].data.payload
  expect(request).toEqual({ clip_id: 'clip-1', request_id: '07070707070707070707070707070707' })
  expect(detail.data.deletionState).toBe('unknown'); expect(storage.size).toBe(1)
  const reopened = instantiate('clip-detail'); reopened.onLoad({ clip_id: 'clip-1' }); await tick()
  expect(reopened.data.deletionState).toBe('unknown'); expect(reopened.data.playbackUrl).toBe('')
  await reopened.loadPlayback(); expect(wx.cloud.callFunction).toHaveBeenCalledTimes(2)
  fail = false; await reopened.requestDeletion()
  expect(deletes()[1][0].data.payload).toEqual(request)
  expect(reopened.data.deletionState).toBe('accepted'); expect(storage.size).toBe(0)
})

test.each([null, {}, { ...receipt(), clip_id: 'other' }, { ...receipt(), status: 'physically_completed' },
  { ...receipt(), deletion_job_id: '' }, { ...receipt(), requested_at: 'invalid' }])('malformed acceptance %p leaves retry intent and lists intact', async data => {
  const { detail, storage } = setup({ responder: ({ action }) => action === 'clips.delete' ? ok(data) : ok({ url: 'https://private.invalid/clip.mp4' }) })
  const list = { route: 'pages/clips/index', onClipDeletionAccepted: jest.fn() }
  global.getCurrentPages = () => [list, detail]
  await tick(); await detail.requestDeletion()
  expect(detail.data.deletionState).toBe('unknown'); expect(storage.size).toBe(1)
  expect(list.onClipDeletionAccepted).not.toHaveBeenCalled()
})

test.each(['MEDIA_NOT_AVAILABLE', 'AUTH_REQUIRED', 'REQUEST_CONFLICT', 'INTERNAL_ERROR'])('%s failure never removes list items or exposes raw messages', async code => {
  const { detail, storage } = setup({ responder: ({ action }) => action === 'clips.delete'
    ? { ok: false, error: { code, message: 'private SDK error' } } : { ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } } })
  const list = { route: 'pages/clips/index', onClipDeletionAccepted: jest.fn() }
  global.getCurrentPages = () => [list, detail]
  await tick(); await tapDelete(detail)
  expect(list.onClipDeletionAccepted).not.toHaveBeenCalled(); expect(storage.size).toBe(1)
  expect(detail.data.deletionMessage).not.toContain('private'); expect(detail.data.playbackUrl).toBe('')
})

test('late playback success/error cannot revive a deletion attempt; cleanup failure cannot undo accepted state', async () => {
  const playback = deferred(); const { detail } = setup({ responder: ({ action }) => action === 'clips.delete' ? ok(receipt()) : playback.promise })
  wx.removeStorageSync.mockImplementation(() => { throw new Error('private cleanup error') })
  await detail.requestDeletion(); const before = { ...detail.data }
  playback.resolve(ok({ url: 'https://stale.invalid/clip' })); await tick(); detail.onPlaybackError(); await detail.loadPlayback()
  expect(detail.data).toEqual(before); expect(detail.data.deletionState).toBe('accepted')
})

test('accepted response after unload invalidates captured open lists without detail setData or playback errors', async () => {
  const response = deferred(); const { detail } = setup({ responder: ({ action }) => action === 'clips.delete' ? response.promise : ok({ url: 'https://private.invalid/clip' }) })
  const list = { route: 'pages/clips/index', onClipDeletionAccepted: jest.fn() }
  global.getCurrentPages = () => [list, detail]; await tick()
  const pending = detail.requestDeletion(); await tick(); detail.onUnload()
  detail.setData.mockClear(); global.getCurrentPages = () => [list]
  response.resolve(ok(receipt())); await pending; detail.onPlaybackError()
  expect(list.onClipDeletionAccepted).toHaveBeenCalledWith('clip-1'); expect(detail.setData).not.toHaveBeenCalled()
})

test('unload fences in-flight playback and a still-open confirmation', async () => {
  const playback = deferred(); const modal = deferred()
  const { detail, deletes } = setup({ responder: () => playback.promise })
  wx.showModal.mockImplementation(({ success }) => modal.promise.then(success))
  const pending = detail.requestDeletion(); detail.onUnload(); detail.setData.mockClear()
  modal.resolve({ confirm: true }); playback.resolve(ok({ url: 'https://stale.invalid' })); await pending; await tick()
  expect(detail.setData).not.toHaveBeenCalled(); expect(deletes()).toHaveLength(0)
})

test.each([['', 'native'], ['', 'custom'], ['session-long', 'native'], ['session-long', 'custom']])(
  'accepted deletion preserves 125-item %s collection pages/cursor/scroll after %s back and filters late insertion', async (sessionId, back) => {
    const items = Array.from({ length: 151 }, (_, i) => ({ clip_id: `clip-${i}`, session_id: 'session-long', sequence: i + 1 }))
    const late = deferred()
    setup({ responder: ({ action, payload }) => {
      if (action === 'clips.delete') return ok(receipt(payload.clip_id))
      if (action === 'clips.getPlayback') return ok({ url: 'https://private.invalid/clip' })
      expect(payload.session_id).toBe(sessionId || undefined)
      const start = Number(payload.cursor || 0)
      return start === 125 ? late.promise : ok({ items: items.slice(start, start + 25), next_cursor: String(start + 25) })
    } })
    const list = instantiate('clips'); list.route = 'pages/clips/index'; list.onLoad(sessionId ? { session_id: sessionId } : {}); await tick()
    for (let i = 0; i < 4; i++) await list.loadMore()
    expect(list.data.clips).toHaveLength(125); list.onPageScroll({ scrollTop: 5000 })
    const pendingPage = list.loadMore()
    const detail = instantiate('clip-detail'); detail.onLoad({ clip_id: 'clip-124' }); await tick()
    const stack = [list, detail]; global.getCurrentPages = () => stack
    wx.navigateBack.mockImplementation(() => { stack.pop(); list.onShow() })
    await detail.requestDeletion()
    if (back === 'custom') detail.backToClips(); else { stack.pop(); detail.onUnload(); list.onShow() }
    expect(list.data.clips.map(c => c.clip_id)).toEqual(items.slice(0, 124).map(c => c.clip_id))
    expect(list.data.nextCursor).toBe('125'); expect(list._scrollTop).toBe(5000)
    expect(wx.pageScrollTo).toHaveBeenLastCalledWith({ scrollTop: 5000, duration: 0 })
    late.resolve(ok({ items: [items[124], ...items.slice(125, 150)], next_cursor: '150' })); await pendingPage
    expect(list.data.clips).toHaveLength(149); expect(list.data.clips.some(c => c.clip_id === 'clip-124')).toBe(false)
    expect(list.data.nextCursor).toBe('150'); expect(wx.navigateTo).not.toHaveBeenCalled()
  },
)

test('removing final visible item keeps continuation and accepts empty result without resetting scroll', async () => {
  setup(); const list = instantiate('clips'); list.data.clips = [{ clip_id: 'clip-1' }]; list.data.nextCursor = 'next'; list._scrollTop = 400
  list.onClipDeletionAccepted('clip-1')
  expect(list.data.clips).toEqual([]); expect(list.data.nextCursor).toBe('next'); expect(list._scrollTop).toBe(400)
  expect(wx.pageScrollTo).not.toHaveBeenCalled()
})

test.each([
  { clip_id: 'other', request_id: '07070707070707070707070707070707' },
  { clip_id: 'clip-1', request_id: 'weak' },
  { clip_id: 'clip-1', request_id: 10000000000000000000000000000000 },
])('corrupt cached intent %p cannot submit or replace pending identity', async pending => {
  const storage = new Map([['clip-deletion:clip-1', pending]])
  const { detail, deletes } = setup({ storage }); await tick(); await detail.requestDeletion()
  expect(deletes()).toHaveLength(0); expect(storage.get('clip-deletion:clip-1')).toEqual(pending)
})

test('a response expanding the receipt beyond the six-field contract is not accepted', async () => {
  const { detail, storage } = setup({ responder: ({ action }) => action === 'clips.delete'
    ? ok({ ...receipt(), physically_completed: true }) : ok({ url: 'https://private.invalid' }) })
  await tick(); await detail.requestDeletion()
  expect(detail.data.deletionState).toBe('unknown'); expect(storage.size).toBe(1)
})

test.each([true, false])('initial unreadable intent storage still uses fresh server playback authorization; allowed=%s', async allowed => {
  const storage = new Map()
  storage.get = () => { throw new Error('storage temporarily unavailable') }
  const { detail, deletes } = setup({ storage, responder: ({ action }) => {
    expect(action).toBe('clips.getPlayback')
    return allowed ? ok({ url: 'https://fresh-authorized.invalid/clip' }) : { ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } }
  } }); await tick()
  expect(detail.data.playbackUrl).toBe(allowed ? 'https://fresh-authorized.invalid/clip' : '')
  expect(wx.cloud.callFunction).toHaveBeenCalledTimes(1)
  await detail.requestDeletion()
  expect(deletes()).toHaveLength(0); expect(detail.data.playbackUrl).toBe('')
  expect(detail.data.deletionState).toBe('intent_error')
})

test('late playback failure and error event preserve unknown-ACK stopped state', async () => {
  const playback = deferred()
  const { detail } = setup({ responder: ({ action }) => action === 'clips.delete'
    ? { ok: false, error: { code: 'NETWORK_UNAVAILABLE' } } : playback.promise })
  await detail.requestDeletion(); const before = { ...detail.data }
  playback.resolve({ ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } }); await tick()
  detail.onPlaybackError(); expect(detail.data).toEqual(before)
})

test('unload during entropy acquisition persists the same retry intent but sends no API or late detail update', async () => {
  const { detail, storage, deletes } = setup(); await tick()
  let randomSuccess
  wx.getRandomValues.mockImplementation(({ success }) => { randomSuccess = success })
  const pending = detail.requestDeletion(); await tick(); expect(randomSuccess).toBeDefined()
  detail.onUnload(); detail.setData.mockClear()
  randomSuccess({ randomValues: new Uint8Array(16).fill(8).buffer }); await pending
  expect(deletes()).toHaveLength(0); expect(detail.setData).not.toHaveBeenCalled()
  expect([...storage.values()]).toEqual([{ clip_id: 'clip-1', request_id: '08080808080808080808080808080808' }])
})

test('stopped accepted UI offers no further delete submission, save, favorite or share controls', async () => {
  const { detail, deletes } = setup(); await tick()
  const activeActions = nodes(render(detail.data)).filter(n => n.attr && n.attr.bindtap).map(n => n.attr.bindtap)
  expect(activeActions).toEqual(['backToClips', 'requestDeletion'])
  await detail.requestDeletion(); await detail.requestDeletion()
  expect(deletes()).toHaveLength(1)
  const accepted = nodes(render(detail.data))
  expect(accepted.some(n => n.tag === 'wx-video')).toBe(false)
  expect(accepted.filter(n => n.attr && n.attr.bindtap).map(n => n.attr.bindtap)).toEqual(['backToClips'])
})

test('canonical earlier request ID may use server-valid dots and still accepts the captured clip receipt', async () => {
  const { detail } = setup({ responder: ({ action }) => action === 'clips.delete'
    ? ok(receipt('clip-1', 'earlier.request-1')) : ok({ url: 'https://private.invalid/clip' }) })
  await tick(); await detail.requestDeletion()
  expect(detail.data.deletionState).toBe('accepted')
})
