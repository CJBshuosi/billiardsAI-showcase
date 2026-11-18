const { callClientApi } = require('../../services/client-api')
const { tableCodeFromLaunchOptions } = require('../../utils/join-scene')
const { errorMessage } = require('../../utils/ui')


Page({
  data: {
    displayName: '',
    errorCode: null,
    joining: false,
    tableCode: '',
    errorMessage: '',
  },

  onLoad(options) {
    const app = getApp()
    const tableCode = tableCodeFromLaunchOptions(options) || app.globalData.previewTableCode || ''
    const user = app.globalData.currentUser
    app.globalData.pendingTableCode = user ? null : tableCode || null
    this.setData({
      displayName: user ? (user.profile || {}).nickname || (user.identity || {}).display_name || '球友' : '',
      tableCode,
      errorMessage: tableCode ? '' : '没有识别到有效球桌，请重新扫描大屏上的二维码。',
    })
    if (!user) {
      wx.redirectTo({ url: '/pages/login/index' })
    }
  },

  async join() {
    if (this.data.joining || !this.data.tableCode) return
    this.setData({ joining: true, errorCode: null, errorMessage: '' })
    const result = await callClientApi('join.request', { table_code: this.data.tableCode })
    if (!result.ok) {
      this.setData({ joining: false, errorCode: result.error.code, errorMessage: result.error.code === 'NETWORK_UNAVAILABLE' ? '暂未确认加入结果。连接恢复后可重试，同一账号会核对本场加入状态。' : errorMessage(result.error.code) })
      return
    }
    getApp().globalData.pendingTableCode = null
    wx.redirectTo({
      url: `/pages/current-session/index?request_id=${encodeURIComponent(result.data.request_id)}`,
    })
  },
})
