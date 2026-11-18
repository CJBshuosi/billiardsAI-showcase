const path = require('path')
const vm = require('vm')
const compiler = require('miniprogram-compiler')

let getTemplate
beforeAll(() => {
  const source = compiler.wxmlToJs(path.resolve(__dirname, '../../miniprogram'))
  getTemplate = vm.runInNewContext(`(function(){${source}})()`, { window: {}, console })
})
afterEach(() => { delete global.Page; delete global.wx })

function navigators(node) {
  return [...(node.tag === 'wx-navigator' ? [node] : []), ...(node.children || []).flatMap(navigators)]
}

function open(url) {
  const parsed = new URL(url, 'https://mini.example')
  let definition
  global.Page = value => { definition = value }
  jest.isolateModules(() => require(`../../miniprogram${parsed.pathname}`))
  const page = { ...definition, data: { ...definition.data }, setData(update) { Object.assign(this.data, update) } }
  if (page.onLoad) page.onLoad(Object.fromEntries(parsed.searchParams))
  if (page.onShow) page.onShow()
  return { page, path: parsed.pathname.slice(1) + '.wxml' }
}

test('home opens own collection without final-session membership lookup', async () => {
  const requests = []
  global.wx = { cloud: { async callFunction({ data }) {
    requests.push(data)
    return { result: { ok: true, data: { schema_version: '1.0', items: [{ clip_id: 'mine', actor_role: 'red', sequence: 1 }], next_cursor: null } } }
  } } }
  const home = getTemplate('pages/home/index.wxml')({ loading: false, user: { identity: {} } })
  const clips = open(navigators(home)[1].attr.url)
  await new Promise(setImmediate)
  expect(clips.page.data.clips[0].clip_id).toBe('mine')
  expect(requests.map(request => [request.action, request.payload])).toEqual([['clips.listPage', {}]])
})

test('report session selection still opens the report detail', () => {
  const tree = getTemplate('pages/sessions/index.wxml')({
    loading: false, clipsMode: false, sessions: [{ session_id: 'session-first' }],
  })
  expect(navigators(tree)[0].attr.url).toBe('/pages/session-detail/index?session_id=session-first')
})
