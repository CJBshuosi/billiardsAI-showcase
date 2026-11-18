const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')
const { hasEstablishedIdentity } = require('../../utils/identity-session')


Page({
  data: { errorCode: null, errorMessage: '', loading: true, guest: true, user: null, displayName: '球友' },

  async onShow() {
    const configurationError = getApp().globalData.configurationError
    if (configurationError) {
      this.setData({ errorCode: configurationError, errorMessage: errorMessage(configurationError), loading: false, user: null, displayName: '球友' })
      return
    }
    if (!getApp().globalData.currentUser && !hasEstablishedIdentity(wx)) {
      this.setData({ guest: true, loading: false, user: null, displayName: '球友', errorCode: null, errorMessage: '' })
      return
    }
    this.setData({ loading: true, errorCode: null, errorMessage: '' })
    const result = await callClientApi('auth.me')
    if (!result.ok) {
      this.setData({ errorCode: result.error.code, errorMessage: errorMessage(result.error.code), loading: false, user: null, displayName: '球友' })
      return
    }
    getApp().globalData.currentUser = result.data
    this.setData({ errorCode: null, loading: false, guest: false, user: result.data,
      displayName: (result.data.profile || {}).nickname || (result.data.identity || {}).display_name || '球友' })
  },
  refresh() { return this.onShow() },
})
