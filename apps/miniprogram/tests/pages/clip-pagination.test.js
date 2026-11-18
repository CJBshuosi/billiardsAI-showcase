const path = require('path')
const vm = require('vm')
const compiler = require('miniprogram-compiler')
let template
beforeAll(() => {
  const source = compiler.wxmlToJs(path.resolve(__dirname, '../../miniprogram'))
  template = vm.runInNewContext(`(function(){${source}})()`, { window: {}, console })('pages/clips/index.wxml')
})
afterEach(() => { delete global.Page; delete global.wx; delete global.getCurrentPages })
const tick = () => new Promise(setImmediate)
function page(responder) {
  global.wx = { getStorageSync: () => '', cloud: { callFunction: jest.fn(async ({ data }) => ({ result: await responder(data) })) } }
  let definition; global.Page = value => { definition = value }
  jest.isolateModules(() => require('../../miniprogram/pages/clips/index'))
  return { ...definition, data: { ...definition.data }, setData(value, callback) { Object.assign(this.data, value); if (callback) callback() } }
}
function success(items, next_cursor = null) { return { ok: true, data: { schema_version: '1.0', items, next_cursor } } }

test.each([undefined, 'former-session'])('own/session collection traverses 101 rows with explicit continuation and refresh resets %p scope', async sessionId => {
  const all = Array.from({ length: 101 }, (_, i) => ({ clip_id: `clip-${i}`, sequence: 1, actor_role: 'red' }))
  const p = page(({ action, payload }) => {
    expect(action).toBe('clips.listPage')
    expect(payload.session_id).toBe(sessionId)
    const start = payload.cursor ? Number(payload.cursor) : 0
    return success(all.slice(start, start + 25), start + 25 < all.length ? String(start + 25) : null)
  })
  p.onLoad(sessionId ? { session_id: sessionId } : {}); await tick()
  expect(p.data.clips).toHaveLength(25)
  expect(JSON.stringify(template(p.data))).toContain('加载更多')
  while (p.data.nextCursor) await p.onReachBottom()
  expect(p.data.clips.map(clip => clip.clip_id)).toEqual(all.map(clip => clip.clip_id))
  await p.refresh()
  expect(p.data.clips).toHaveLength(25)
  expect(wx.cloud.callFunction.mock.calls.at(-1)[0].data.payload).not.toHaveProperty('cursor')
})

test('an empty filtered page still offers continuation; appends deduplicate without reordering server pages', async () => {
  const p = page(({ payload }) => payload.cursor === 'next' ? success([{ clip_id: 'same', sequence: 8 }, { clip_id: 'last', sequence: 1 }], 'final')
    : payload.cursor === 'final' ? success([{ clip_id: 'same', sequence: 8 }]) : success([], 'next'))
  p.onLoad({}); await tick()
  expect(p.data.clips).toEqual([])
  expect(JSON.stringify(template(p.data))).toContain('加载更多')
  expect(JSON.stringify(template(p.data))).not.toContain('暂无已保存片段')
  await p.loadMore(); await p.onReachBottom()
  expect(p.data.clips.map(clip => clip.clip_id)).toEqual(['same', 'last'])
  expect(p.data.nextCursor).toBeNull()
})

test('load-more errors retain existing items and allow retrying the same cursor', async () => {
  let failed = false
  const p = page(({ payload }) => !payload.cursor ? success([{ clip_id: 'first' }], 'next')
    : !failed ? (failed = true, { ok: false, error: { code: 'NETWORK_UNAVAILABLE' } }) : success([{ clip_id: 'last' }]))
  p.onLoad({}); await tick(); await p.loadMore()
  expect(p.data.clips.map(clip => clip.clip_id)).toEqual(['first'])
  expect(p.data.nextCursor).toBe('next')
  expect(JSON.stringify(template(p.data))).toContain('加载更多')
  await p.loadMore()
  expect(p.data.clips.map(clip => clip.clip_id)).toEqual(['first', 'last'])
})

test.each([undefined, 'long-session'])('detail opens a selected later-page clip directly without bounded session metadata lookup %p', async sessionId => {
  let definition
  global.Page = value => { definition = value }
  global.wx = { getStorageSync: () => '', cloud: { callFunction: jest.fn(async ({ data }) => {
    expect(data.action).toBe('clips.getPlayback')
    expect(data.payload).toEqual({ clip_id: 'clip-101' })
    return { result: { ok: true, data: { url: 'https://private.invalid/101.mp4', clip: {
      clip_id: 'clip-101', session_id: 'long-session', sequence: 101, actor_role: 'red',
    } } } }
  }) } }
  jest.isolateModules(() => require('../../miniprogram/pages/clip-detail/index'))
  const p = { ...definition, data: { ...definition.data }, setData(value) { Object.assign(this.data, value) } }
  p.onLoad({ clip_id: 'clip-101', ...(sessionId ? { session_id: sessionId } : {}) }); await tick()
  expect(p.data.playbackUrl).toBe('https://private.invalid/101.mp4')
  expect(p.data.clip.title).toBe('红方 · 第 101 杆')
  expect(wx.cloud.callFunction).toHaveBeenCalledTimes(1)
})

test.each([
  ['', 'custom'], ['', 'native'], ['former-session', 'custom'], ['former-session', 'native'],
])('repeated %s collection detail/%s back preserves loaded pages, cursor and scroll without stacking collections', async (sessionId, returnMode) => {
  const all = Array.from({ length: 151 }, (_, index) => ({
    clip_id: `clip-${index}`, session_id: 'former-session', sequence: index + 1, actor_role: 'red',
  }))
  const collection = page(({ action, payload }) => {
    if (action === 'clips.getPlayback') return { ok: true, data: { url: 'https://private.invalid/clip.mp4', clip: all.find(clip => clip.clip_id === payload.clip_id) } }
    expect(action).toBe('clips.listPage')
    expect(payload.session_id).toBe(sessionId || undefined)
    const start = payload.cursor ? Number(payload.cursor) : 0
    return success(all.slice(start, start + 25), start + 25 < all.length ? String(start + 25) : null)
  })
  collection.route = 'pages/clips/index'
  collection.onLoad(sessionId ? { session_id: sessionId } : {})
  collection.onShow(); await tick()
  await collection.loadMore(); await collection.loadMore(); await collection.loadMore(); await collection.loadMore()
  const loaded = [...collection.data.clips]
  expect(loaded).toHaveLength(125)
  expect(collection.data.nextCursor).toBe('125')
  collection.onPageScroll({ scrollTop: 5000 })
  wx.pageScrollTo = jest.fn()
  const stack = [{ route: 'pages/home/index' }, collection]
  global.getCurrentPages = () => stack
  wx.navigateBack = jest.fn(({ delta }) => { stack.splice(-delta); stack.at(-1).onShow() })
  wx.navigateTo = jest.fn(() => { stack.push({ route: 'pages/clips/index' }) })
  const listRequests = () => wx.cloud.callFunction.mock.calls.filter(([request]) => request.data.action === 'clips.listPage').length
  const initialRequests = listRequests()
  function navigators(node) {
    return [...(node.tag === 'wx-navigator' ? [node] : []), ...(node.children || []).flatMap(navigators)]
  }
  for (let cycle = 0; cycle < 3; cycle += 1) {
    const link = navigators(template(collection.data)).find(node => node.attr.url.includes('clip_id=clip-124'))
    const options = Object.fromEntries(new URL(link.attr.url, 'https://mini.invalid').searchParams)
    // This is the real item session even though the global parent filter is ''.
    expect(options.session_id).toBe('former-session')
    let definition
    global.Page = value => { definition = value }
    jest.isolateModules(() => require('../../miniprogram/pages/clip-detail/index'))
    const detail = { ...definition, route: 'pages/clip-detail/index', data: { ...definition.data }, setData(value) { Object.assign(this.data, value) } }
    stack.push(detail)
    detail.onLoad(options); await tick()
    expect(detail.data.clip.clip_id).toBe('clip-124')
    if (returnMode === 'custom') detail.backToClips()
    else { stack.pop(); collection.onShow() }
    await tick()
    expect(stack.map(value => value.route)).toEqual(['pages/home/index', 'pages/clips/index'])
    expect(stack.at(-1)).toBe(collection)
    expect(collection.data.clips).toHaveLength(125)
    expect(collection.data.clips).toEqual(loaded)
    expect(collection.data.nextCursor).toBe('125')
    expect(collection._scrollTop).toBe(5000)
    expect(wx.pageScrollTo).toHaveBeenLastCalledWith({ scrollTop: 5000, duration: 0 })
    expect(listRequests()).toBe(initialRequests)
    expect(wx.navigateTo).not.toHaveBeenCalled()
  }
  await collection.loadMore()
  expect(collection.data.clips).toHaveLength(150)
  expect(wx.cloud.callFunction.mock.calls.at(-1)[0].data.payload.cursor).toBe('125')
  await collection.refresh()
  expect(collection.data.clips).toHaveLength(25)
  expect(collection.data.nextCursor).toBe('25')
  expect(collection._scrollTop).toBe(0)
  expect(wx.pageScrollTo).toHaveBeenLastCalledWith({ scrollTop: 0, duration: 0 })
})

test('detail opened without a parent collection retains its global collection fallback', () => {
  let definition
  global.Page = value => { definition = value }
  global.wx = { navigateBack: jest.fn(), navigateTo: jest.fn() }
  jest.isolateModules(() => require('../../miniprogram/pages/clip-detail/index'))
  const detail = { ...definition, data: { ...definition.data, sessionId: 'direct-session' } }
  global.getCurrentPages = () => [{ route: 'pages/home/index' }, detail]
  detail.backToClips()
  expect(wx.navigateTo).toHaveBeenCalledWith({ url: '/pages/clips/index' })
  expect(wx.navigateBack).not.toHaveBeenCalled()
})
