const path = require('path')
const vm = require('vm')
const compiler = require('miniprogram-compiler')

let template
beforeAll(() => {
  const source = compiler.wxmlToJs(path.resolve(__dirname, '../../miniprogram'))
  template = vm.runInNewContext(`(function(){${source}})()`, { window: {}, console })
})
afterEach(() => { delete global.Page; delete global.wx; delete global.getCurrentPages })
const tick = () => new Promise(setImmediate)
function page(name, responder) {
  global.wx = { getStorageSync: () => '', cloud: { callFunction: jest.fn(async ({ data }) => {
    const result = await responder(data)
    if (name === 'clips') {
      expect(data.action).toBe('clips.listPage')
      if (result.ok && Array.isArray(result.data)) return { result: { ok: true, data: { schema_version: '1.0', items: result.data, next_cursor: null } } }
    }
    return { result }
  }) } }
  let definition
  global.Page = value => { definition = value }
  jest.isolateModules(() => require(`../../miniprogram/pages/${name}/index`))
  return { ...definition, data: JSON.parse(JSON.stringify(definition.data)), setData(value, callback) { Object.assign(this.data, value); if (callback) callback() } }
}
const success = data => ({ ok: true, data })
const failure = code => ({ ok: false, error: { code } })
const render = (name, data) => JSON.stringify(template(`pages/${name}/index.wxml`)(data))

test('saved-until-deleted clip copy shows retention without an invented deadline', async () => {
  const p = page('clips', () => success([{
    clip_id: 'saved', sequence: 1, upload_status: 'published',
    retention_policy: 'saved_until_deleted_v1', available_until: null,
  }]))
  p.onLoad({ session_id: 's' }); await tick()
  expect(render('clips', p.data)).toContain('已保存，保留至删除')
  expect(render('clips', p.data)).not.toContain('可查看至')
})

test('report leaves unknown shot count unknown and maps private limitation codes to safe text', async () => {
  const p = page('session-detail', () => success({ session_id: 's', status: 'ended', limitations: ['internal_revision_secret'] }))
  p.onLoad({ session_id: 's' }); await tick()
  expect(p.data.shotCount).toBe('—')
  expect(render('session-detail', p.data)).not.toContain('internal_revision_secret')
  expect(render('session-detail', p.data)).toContain('尚未取得')
})

test('report distinguishes an actual empty shot array from missing data and names replay restrictions', async () => {
  const p = page('session-detail', () => success({ session_id: 's', shots: [], limitations: ['development_replay_media'] }))
  p.onLoad({ session_id: 's' }); await tick()
  expect(p.data.shotCount).toBe(0)
  expect(render('session-detail', p.data)).toContain('开发回放测试')
  expect(render('session-detail', p.data)).toContain('不计入正式个人成绩')
})

test('failed refresh clears previously authorized report and never exposes raw errors', async () => {
  let denied = false
  const p = page('session-detail', () => denied ? failure('FORBIDDEN') : success({ session_id: 'private-session', shots: [] }))
  p.onLoad({ session_id: 's' }); await tick(); denied = true; await p.refresh()
  expect(p.data.session).toBeNull()
  expect(render('session-detail', p.data)).not.toContain('private-session')
  expect(render('session-detail', p.data)).not.toContain('FORBIDDEN')
})

test('clip list preserves server page order and unknown sequence without a favorite filter', async () => {
  const p = page('clips', () => success([
    { clip_id: 'unknown', upload_status: 'published', available_until: '2027-09-03T12:00:00.000Z' },
    { clip_id: 'third', sequence: 3 },
    { clip_id: 'first', sequence: 1 },
  ]))
  p.onLoad({ session_id: 's' }); await tick()
  expect(p.data.clips.map(c => c.clip_id)).toEqual(['unknown', 'third', 'first'])
  expect(p.data.clips[0].title).toBe('击球片段 · 序号待确认')
  expect(render('clips', p.data)).toContain('击球片段 · 序号待确认')
  expect(render('clips', p.data)).toContain('大屏保存')
  expect(p.setFilter).toBeUndefined()
})

test('playback error discards expired URL and explicit retry reauthorizes without autoplay', async () => {
  let attempts = 0
  const p = page('clip-detail', () => ++attempts === 1 ? success({ url: 'https://media.example/old' }) : failure('FORBIDDEN'))
  p.onLoad({ clip_id: 'c' }); await tick()
  p.onPlaybackError()
  expect(p.data.playbackUrl).toBe('')
  await p.loadPlayback()
  expect(attempts).toBe(2)
  expect(p.data.playbackUrl).toBe('')
  expect(p.data.authorized).toBe(false)
  expect(render('clip-detail', p.data)).not.toContain('FORBIDDEN')
})

test('private detail has no share hook or share control', () => {
  const p = page('clip-detail', () => success({}))
  expect(p.onShareAppMessage).toBeUndefined()
  const view = render('clip-detail', { ...p.data, authorized: true })
  expect(view).not.toContain('open-type')
  expect(view).not.toContain('分享')
})

test('private clip title includes the historical role', async () => {
  const p = page('clips', () => success([{ clip_id: 'c', sequence: 1, actor_role: 'red' }]))
  p.onLoad({}); await tick()
  expect(p.data.clips[0].title).toBe('红方 · 第 1 杆')
  expect(wx.cloud.callFunction.mock.calls[0][0].data.payload).toEqual({})
})

test('denied playback never fetches session metadata supplied by a shared URL', async () => {
  const p = page('clip-detail', () => failure('FORBIDDEN'))
  p.onLoad({ clip_id: 'private', session_id: 'private-session' }); await tick()
  expect(wx.cloud.callFunction).toHaveBeenCalledTimes(1)
  expect(p.data.clip).toBeNull()
})

test('direct authorized metadata supplies historical role without trusting URL titles or listing a session', async () => {
  const p = page('clip-detail', ({ action }) => {
    expect(action).toBe('clips.getPlayback')
    return success({ url: 'https://media.example/temp', clip: {
      clip_id: 'c', session_id: 's', sequence: 7, actor_role: 'red', source_kind: 'usb_live', upload_status: 'published',
    } })
  })
  p.onLoad({ clip_id: 'c', session_id: 's', sequence: '99' }); await tick()
  expect(p.data.clip.title).toBe('红方 · 第 7 杆')
  expect(wx.cloud.callFunction).toHaveBeenCalledTimes(1)
})

test('failed reauthorization clears direct clip metadata and previously authorized playback', async () => {
  let revoked = false
  const p = page('clip-detail', () => revoked ? failure('MEDIA_NOT_AVAILABLE') : success({
    url: 'https://media.example/temp', clip: { clip_id: 'c', session_id: 's', sequence: 7, actor_role: 'red' },
  }))
  p.onLoad({ clip_id: 'c', session_id: 's' }); await tick()
  expect(p.data.clip.title).toBe('红方 · 第 7 杆')
  revoked = true; await p.loadPlayback()
  expect(p.data.playbackUrl).toBe('')
  expect(p.data.authorized).toBe(false)
  expect(p.data.clip).toBeNull()
})

test('return to the same clip list uses the existing page preserving its scroll', () => {
  const p = page('clip-detail', () => success({}))
  p.setData({ sessionId: 's' })
  global.getCurrentPages = () => [{ route: 'pages/clips/index', data: { sessionId: 's' } }, p]
  wx.navigateBack = jest.fn()
  wx.navigateTo = jest.fn()
  p.backToClips()
  expect(wx.navigateBack).toHaveBeenCalledWith({ delta: 1 })
  expect(wx.navigateTo).not.toHaveBeenCalled()
})

test('ordinary collection return keeps its content and scroll without an implicit refresh', async () => {
  const p = page('clips', () => success([{ clip_id: 'c', sequence: 2 }]))
  p.onLoad({ session_id: 's' }); p.onShow(); await tick()
  p.onPageScroll({ scrollTop: 320 })
  wx.pageScrollTo = jest.fn()
  const requests = wx.cloud.callFunction.mock.calls.length
  p.onShow(); await tick()
  expect(wx.cloud.callFunction).toHaveBeenCalledTimes(requests)
  expect(wx.pageScrollTo).toHaveBeenCalledWith({ scrollTop: 320, duration: 0 })
})

test.each([[0, 0], [99, 99], [100, '100+']])('report labels a %i item response without claiming a capped exact total', async (count, label) => {
  const p = page('session-detail', ({ action }) => success(action === 'sessions.get'
    ? { session_id: 's', shots: [] }
    : Array.from({ length: count }, (_, i) => ({ clip_id: `c${i}` }))))
  p.onLoad({ session_id: 's' }); await tick()
  expect(p.data.clipCount).toBe(label)
  expect(render('session-detail', p.data)).toContain(String(label))
})
