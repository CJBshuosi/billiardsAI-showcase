const TABLE_CODE = 'AbCdEfGhIjKlMnOpQrStUvWxYz012345'


test('an established user can return home after the join entry has been presented', () => {
  let pageDefinition
  const app = {
    globalData: {
      currentUser: { identity: { display_name: '球友 A' } },
      pendingTableCode: TABLE_CODE,
    },
  }
  global.Page = (definition) => { pageDefinition = definition }
  global.getApp = () => app
  global.wx = { redirectTo: jest.fn() }

  jest.isolateModules(() => require('../../miniprogram/pages/join/index'))
  const page = {
    data: { ...pageDefinition.data },
    setData(values) { this.data = { ...this.data, ...values } },
  }
  pageDefinition.onLoad.call(page, { table_code: TABLE_CODE })

  expect(page.data.tableCode).toBe(TABLE_CODE)
  expect(app.globalData.pendingTableCode).toBeNull()
  expect(wx.redirectTo).not.toHaveBeenCalled()

  delete global.Page
  delete global.getApp
  delete global.wx
})

test('rescanning the same preview after a completed session reuses its packaged table code', () => {
  let definition
  const app = { globalData: { currentUser: { identity: { display_name: '球友 A' } }, pendingTableCode: null, previewTableCode: TABLE_CODE } }
  global.Page = value => { definition = value }
  global.getApp = () => app
  global.wx = { redirectTo: jest.fn() }
  jest.isolateModules(() => require('../../miniprogram/pages/join/index'))
  const page = { data: { ...definition.data }, setData(values) { Object.assign(this.data, values) } }
  definition.onLoad.call(page, {})
  expect(page.data.tableCode).toBe(TABLE_CODE)
  expect(page.data.errorMessage).toBe('')
  expect(wx.redirectTo).not.toHaveBeenCalled()
  delete global.Page;delete global.getApp;delete global.wx
})
