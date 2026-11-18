function pageAt(name) {
  let definition
  global.Page = value => { definition = value }
  jest.isolateModules(() => require(`../../miniprogram/pages/${name}/index`))
  return { ...definition, data: { ...definition.data }, setData(update) { Object.assign(this.data, update) } }
}

afterEach(() => { delete global.Page; delete global.wx; delete global.getApp })

test('first-time visitors can browse without silently creating an account', async () => {
  const app = { globalData: {} }
  global.getApp = () => app
  global.wx = { getStorageSync: () => false, cloud: { callFunction: jest.fn() } }
  const page = pageAt('home'); await page.onShow()
  expect(wx.cloud.callFunction).not.toHaveBeenCalled()
  expect(page.data.guest).toBe(true)
  expect(page.data.loading).toBe(false)
  expect(app.globalData.currentUser).toBeUndefined()
})

test('profile saves preserve established identity when update endpoint returns profile only', async () => {
  const app = { globalData: { currentUser: { user_id: 'user-a', identity: { display_name: '球友甲' }, profile: { nickname: '旧名' } } } }
  global.getApp = () => app
  global.wx = { cloud: { callFunction: async () => ({ result: { ok: true, data: { user_id: 'user-a', profile: { nickname: '新名' }, status: 'active' } } }) } }
  const page = pageAt('profile')
  page.onLoad()
  page.updateNickname({ detail: { value: '新名' } })
  await page.save()
  expect(app.globalData.currentUser.identity.display_name).toBe('球友甲')
  expect(page.data.saved).toBe(true)
})

test('profile save failure keeps the editing value and does not claim success', async () => {
  global.getApp = () => ({ globalData: { currentUser: { profile: {} } } })
  global.wx = { cloud: { callFunction: async () => ({ result: { ok: false, error: { code: 'PROFILE_INVALID' } } }) } }
  const page=pageAt('profile'); page.onLoad()
  page.updateNickname({ detail: { value: '待修改' } }); await page.save()
  expect(page.data.nickname).toBe('待修改')
  expect(page.data.saved).toBe(false)
  expect(page.data.errorMessage).toBeTruthy()
  expect(page.data.errorMessage).not.toContain('PROFILE_INVALID')
})

test('unconfigured home can show honest unavailable features without querying a cloud environment', async () => {
  const requests=[]
  global.getApp=()=>({globalData:{configurationError:'ENVIRONMENT_MISMATCH'}})
  global.wx={cloud:{callFunction:async request=>{requests.push(request); return {result:{ok:false,error:{code:'NETWORK_UNAVAILABLE'}}}}}}
  const page=pageAt('home'); await page.onShow()
  expect(requests).toHaveLength(0)
  expect(page.data.errorMessage).toBeTruthy()
  expect(page.data.errorMessage).not.toContain('ENVIRONMENT_MISMATCH')
  expect(page.data.loading).toBe(false)
})

test('only table-code scan results route into the existing join flow', () => {
  const { tableCodeFromScanResult }=require('../../miniprogram/utils/scan-entry')
  const code='A'.repeat(32)
  expect(tableCodeFromScanResult({path:`pages/join/index?scene=${code}`})).toBe(code)
  expect(tableCodeFromScanResult({result:code})).toBe(code)
  expect(tableCodeFromScanResult({path:`pages/clip-detail/index?table_code=${code}`})).toBe('')
  expect(tableCodeFromScanResult({result:`https://evil.example/?table_code=${code}`})).toBe('')
  expect(tableCodeFromScanResult({path:`pages/join/index?scene=${code}&scene=${'B'.repeat(32)}`})).toBe('')
  expect(tableCodeFromScanResult({path:'pages/join/index?scene=%E0%A4%A'})).toBe('')
})

test('scan cancellation causes no navigation or cloud join and restores the scan button', () => {
  let callbacks; const paths=[]
  global.wx={scanCode:options=>{callbacks=options},navigateTo:options=>paths.push(options.url)}
  const page=pageAt('scan'); page.scan()
  expect(page.data.scanning).toBe(true)
  callbacks.fail({errMsg:'scanCode:fail cancel'})
  expect(page.data.scanning).toBe(false)
  expect(page.data.errorMessage).toBe('')
  expect(paths).toEqual([])
})

test('a packaged development preview enters its bound test table without rescanning a preview ticket', () => {
  const code='A'.repeat(32);const paths=[]
  global.getApp=()=>({globalData:{previewTableCode:code}})
  global.wx={scanCode:jest.fn(),navigateTo:options=>paths.push(options.url)}
  const page=pageAt('scan');page.onLoad();page.enterPreviewTable()
  expect(page.data.previewMode).toBe(true)
  expect(wx.scanCode).not.toHaveBeenCalled()
  expect(paths).toEqual([`/pages/join/index?table_code=${code}`])
})

test('static future feature pages never make requests or produce personal scores', async () => {
  const requests=[]
  global.getApp=()=>({globalData:{}})
  global.wx={cloud:{callFunction:async value=>{requests.push(value)}}}
  for(const name of ['ranking','calibration','insight']) {
    const page=pageAt(name)
    if(page.onLoad) await page.onLoad({kind:'strength'})
    if(page.onShow) await page.onShow()
    expect(page.data.available).toBe(false)
    expect(page.data.score).toBeUndefined()
  }
  expect(requests).toEqual([])
})
