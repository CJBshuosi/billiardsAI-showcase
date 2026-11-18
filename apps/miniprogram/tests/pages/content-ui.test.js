const fs = require('fs')
const path = require('path')


const pageRoot = path.join(__dirname, '../../miniprogram/pages')
const pageSource = (page, extension) => fs.readFileSync(
  path.join(pageRoot, page, `index.${extension}`),
  'utf8',
)


test('media UI names only large-screen-saved clips and one unavailable state', () => {
  const source = [
    pageSource('clips', 'wxml'),
    pageSource('clip-detail', 'wxml'),
  ].join('\n')
  expect(source).toContain('击球片段')
  expect(source).toContain('大屏')
  expect(source).toContain('暂不可用')
  expect(source).toContain('autoplay="{{false}}"')
  expect(source).not.toContain('正在上传')
  expect(source).not.toContain('上传失败')
  expect(source).not.toContain('精彩片段')
})


test('private clips expose no sharing hook, control or enabled page option', () => {
  expect(pageSource('clip-detail', 'js')).not.toContain('onShareAppMessage')
  expect(pageSource('clip-detail', 'wxml')).not.toContain('open-type="share"')
  expect(JSON.parse(pageSource('clip-detail', 'json')).enableShareAppMessage).toBe(false)
})

test('clip pages keep favorite actions unsupported alongside owner deletion', () => {
  const source = [
    pageSource('clips', 'js'), pageSource('clips', 'wxml'),
    pageSource('clip-detail', 'js'), pageSource('clip-detail', 'wxml'),
  ].join('\n')
  expect(source).not.toMatch(/favorites\.(set|list)/)
  expect(source).not.toContain('/pages/favorites')
  expect(source).not.toContain('收藏')
  expect(source).toContain('大屏')
})

test('saved-only clip view has no dead upload or deletion status copy', () => {
  const source = fs.readFileSync(
    path.join(__dirname, '../../miniprogram/utils/content-view.js'),
    'utf8',
  )
  expect(source).not.toContain('正在上传')
  expect(source).not.toContain('上传失败')
  expect(source).not.toContain('已删除')
})
