const path = require('path')
const vm = require('vm')
const compiler = require('miniprogram-compiler')

let template
beforeAll(() => {
  const source = compiler.wxmlToJs(path.resolve(__dirname, '../../miniprogram'))
  template = vm.runInNewContext(`(function(){${source}})()`, { window: {}, console })
})
beforeEach(() => { jest.useFakeTimers() })
afterEach(() => {
  jest.useRealTimers()
  delete global.Page
  delete global.wx
})

function page(status, rankingScope = null) {
  const requests = []
  global.wx = { cloud: { callFunction: jest.fn(async ({ data }) => {
    requests.push(data)
    if (data.action === 'sessions.analysis.get') return { result: { ok: true, data: {
      schema_version: '1.0', session_id: 'session-1', revision: 2, status,
      live_shots_recorded: 3,
      total_shots: 3, available_shots: status === 'AVAILABLE' ? 3 : status === 'PARTIAL' ? 1 : 0,
      retrying_shots: status === 'FAILED_RETRYING' || status === 'PARTIAL' ? 1 : 0,
      report_available: status === 'AVAILABLE' || status === 'PARTIAL',
      deadline_at: '2026-09-07T00:02:00.000Z', updated_at: '2026-09-07T00:01:00.000Z',
    } } }
    if (data.action === 'sessions.get') return { result: { ok: true, data: {
      session_id: 'session-1', status: 'ended', shots: [{ shot_id: 'shot-1' }], limitations: [],
    } } }
    if (data.action === 'clips.list') return { result: { ok: true, data: [] } }
    if (data.action === 'ranking.match.get' && rankingScope) return { result: { ok: true, data: {
      match_id: 'match-1', status: 'fact_package_ready', ranking_scope: rankingScope,
      result: { winner_role: 'red', red_score: 1, blue_score: 0 },
      ranking_outcome: {
        status: 'settled', placement_status: 'placed', display_rating: 1210,
        rating_delta: 10,
      },
    } } }
    return { result: { ok: true, data: null } }
  }) } }
  let definition
  global.Page = value => { definition = value }
  jest.isolateModules(() => require('../../miniprogram/pages/session-detail/index'))
  const instance = {
    ...definition,
    data: JSON.parse(JSON.stringify(definition.data)),
    setData(value) { Object.assign(this.data, value) },
  }
  return { instance, requests }
}

const render = data => JSON.stringify(template('pages/session-detail/index.wxml')(data))

test.each([
  ['PROCESSING', '正在整理本场回放与逐杆分析', false],
  ['AVAILABLE', '本场回放与逐杆分析已可查看', true],
  ['PARTIAL', '部分逐杆分析已可查看', true],
  ['FAILED_RETRYING', '系统正在重试', false],
])('%s renders honestly and gates content reads', async (status, copy, readsContent) => {
  const { instance, requests } = page(status)
  await instance.onLoad({ session_id: 'session-1' })

  expect(render(instance.data)).toContain(copy)
  expect(requests[0]).toMatchObject({
    action: 'sessions.analysis.get', payload: { session_id: 'session-1' },
  })
  expect(requests.some(item => item.action === 'sessions.get')).toBe(readsContent)
  expect(requests.some(item => item.action === 'clips.list')).toBe(readsContent)
  expect(requests.every(item => !item.payload?.session_id
    || item.payload.session_id === 'session-1')).toBe(true)
  instance.onUnload()
})

test('processing view shows live recorded shots without reading clips', async () => {
  const { instance, requests } = page('PROCESSING')
  await instance.onLoad({ session_id: 'session-1' })

  expect(render(instance.data)).toContain('本场已实时记录 3 杆')
  expect(requests.some(item => item.action === 'clips.list')).toBe(false)
  instance.onUnload()
})

test('polling stops offscreen and resumes with bounded backoff', async () => {
  const { instance, requests } = page('PROCESSING')
  await instance.onLoad({ session_id: 'session-1' })
  expect(jest.getTimerCount()).toBe(1)

  instance.onHide()
  expect(jest.getTimerCount()).toBe(0)
  jest.advanceTimersByTime(20000)
  expect(requests).toHaveLength(1)

  await instance.onShow()
  expect(requests).toHaveLength(2)
  expect(jest.getTimerCount()).toBe(1)
  instance.onUnload()
})

test('an in-flight analysis response cannot update a hidden page', async () => {
  const { instance } = page('PROCESSING')
  let resolveRequest
  global.wx.cloud.callFunction = jest.fn(() => new Promise(resolve => {
    resolveRequest = resolve
  }))

  const loading = instance.onLoad({ session_id: 'session-1' })
  instance.onHide()
  resolveRequest({ result: { ok: true, data: {
    schema_version: '1.0', session_id: 'session-1', revision: 2,
    status: 'AVAILABLE', total_shots: 1, available_shots: 1, retrying_shots: 0,
    report_available: true, deadline_at: '2026-09-07T00:02:00.000Z',
    updated_at: '2026-09-07T00:01:00.000Z',
  } } })
  await loading

  expect(instance.data.analysis).toBeNull()
  expect(jest.getTimerCount()).toBe(0)
  instance.onUnload()
})

test('development outcome is never labelled as a formal result or score', async () => {
  const { instance } = page('AVAILABLE', {
    ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
    mode: 'development_acceptance', is_test: true,
  })
  await instance.onLoad({ session_id: 'session-1' })
  const output = render(instance.data)
  expect(output).toContain('开发测试赛果')
  expect(output).toContain('开发测试积分')
  expect(output).not.toContain('正式赛果')
  expect(output).not.toContain('正式积分已更新')
  instance.onUnload()
})
