const path = require('path')
const vm = require('vm')
const compiler = require('miniprogram-compiler')

test('replay reports distinguish test observations from official personal performance', () => {
  const source = compiler.wxmlToJs(path.resolve(__dirname, '../../miniprogram'))
  const templates = vm.runInNewContext(`(function(){${source}})()`, { window: {}, console })
  const report = JSON.stringify(templates('pages/session-detail/index.wxml')({
    loading: false, session: { session_id: 's', shots: [], limitations: ['development_replay_media'] },
  }))
  expect(report).toContain('开发回放测试')
  expect(report).toContain('不计入正式个人成绩')
  const clips = JSON.stringify(templates('pages/clips/index.wxml')({ loading: false, clips: [] }))
  expect(clips).toContain('大屏保存')
  expect(clips).toContain('不代表系统精选')
})
