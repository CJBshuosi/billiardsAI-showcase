const path = require('path')
const vm = require('vm')
const compiler = require('miniprogram-compiler')

let renderClips

beforeAll(() => {
  const source = compiler.wxmlToJs(path.resolve(__dirname, '../../miniprogram'))
  const context = vm.createContext({ window: {}, console })
  const getTemplate = vm.runInContext(`(function(){${source}})()`, context)
  renderClips = getTemplate('pages/clips/index.wxml')
})

afterEach(() => {
  delete global.Page
  delete global.wx
})

function openListedClip() {
  global.wx.getStorageSync = () => ''
  const tree = renderClips({
    loading: false,
    clips: [{ clip_id: 'clip-test', sequence: 1, upload_status: 'published' }],
  })
  function findNavigator(node) {
    if (node.tag === 'wx-navigator') return node
    return (node.children || []).map(findNavigator).find(Boolean)
  }
  const url = new URL(findNavigator(tree).attr.url, 'https://mini.example')
  let definition
  global.Page = value => { definition = value }
  jest.isolateModules(() => require('../../miniprogram/pages/clip-detail/index'))
  const page = {
    ...definition,
    data: { ...definition.data },
    setData(update) { Object.assign(this.data, update) },
  }
  page.onLoad(Object.fromEntries(url.searchParams))
  return page
}

test('opening a published list item loads its playback URL through the client API', async () => {
  global.wx = { cloud: { callFunction: jest.fn(async ({ data }) => {
    expect(data).toEqual({
      action: 'clips.getPlayback', payload: { clip_id: 'clip-test' }, schema_version: '1.0',
    })
    return { result: { ok: true, data: { url: 'https://media.example/shot.mp4' } } }
  }) } }
  const page = openListedClip()
  expect(page.data).not.toHaveProperty('favorite')
  await new Promise(setImmediate)
  expect(page.data).toMatchObject({
    clipId: 'clip-test', state: 'published', playbackUrl: 'https://media.example/shot.mp4',
    loading: false, errorCode: null,
  })
  expect(wx.cloud.callFunction).toHaveBeenCalledTimes(1)
})

test('opening a listed clip never trusts an upload state from the URL', async () => {
  global.wx = { cloud: { callFunction: jest.fn(async () => ({
    result: { ok: false, error: { code: 'MEDIA_NOT_AVAILABLE' } },
  })) } }
  const page = openListedClip()
  await new Promise(setImmediate)
  expect(page.data.state).toBe('unavailable')
  expect(page.data.playbackUrl).toBe('')
  expect(wx.cloud.callFunction).toHaveBeenCalledTimes(1)
})
