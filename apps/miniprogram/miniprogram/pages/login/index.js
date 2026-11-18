const { callClientApi } = require('../../services/client-api')
const { rememberEstablishedIdentity } = require('../../utils/identity-session')
const { errorMessage } = require('../../utils/ui')

Page({
  data: {
    errorCode: null,
    loading: false,
    errorMessage: '',
  },

  async authorize() {
    if (this.data.loading) return
    const configurationError = getApp().globalData.configurationError
    if (configurationError) {
      this.setData({ errorMessage: errorMessage(configurationError) })
      return
    }
    this.setData({ loading: true, errorCode: null, errorMessage: '' })
    const result = await callClientApi('auth.me')
    if (!result.ok) {
      this.setData({ loading: false, errorCode: result.error.code, errorMessage: errorMessage(result.error.code) })
      return
    }
    getApp().globalData.currentUser = result.data
    rememberEstablishedIdentity(wx)
    const tableCode = getApp().globalData.pendingTableCode
    wx.reLaunch({
      url: tableCode
        ? `/pages/join/index?table_code=${encodeURIComponent(tableCode)}`
        : '/pages/home/index',
    })
  },
})
