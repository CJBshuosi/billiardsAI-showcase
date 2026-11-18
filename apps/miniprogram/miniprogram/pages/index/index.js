const {
  hasEstablishedIdentity,
  resolveEntryRoute,
  resolveEntryState,
} = require('../../utils/identity-session')
const { errorMessage } = require('../../utils/ui')


Page({
  data: {
    title: 'billiardsAI',
    message: '正在检查开发环境…',
    ready: false,
    resuming: false,
  },

  onLoad() {
    const app = getApp()
    const code = app.globalData.configurationError
    const state = resolveEntryState({
      configurationError: code,
      identityEstablished: hasEstablishedIdentity(wx),
    })
    this.setData({ message: code ? errorMessage(code) : (state.resume ? '正在恢复你的账号…' : '登录后，记录每一场球。'), ready: state.ready, resuming: state.resume })
    const route = resolveEntryRoute({
      configurationError: code,
      identityEstablished: hasEstablishedIdentity(wx),
      pendingTableCode: app.globalData.pendingTableCode,
    })
    if (route) wx.reLaunch({ url: route })
  },
})
