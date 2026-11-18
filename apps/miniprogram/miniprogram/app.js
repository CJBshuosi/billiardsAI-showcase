const { loadEnvironment } = require('./config/environment')

App({
  globalData: {
    configurationError: null,
    currentUser: null,
    pendingTableCode: null,
    previewTableCode: null,
  },

  onLaunch() {
    const environment = loadEnvironment()
    if (!environment.ok) {
      this.globalData.configurationError = environment.error.code
      return
    }
    this.globalData.previewTableCode = environment.data.previewTableCode
    this.globalData.pendingTableCode = environment.data.previewTableCode
    wx.cloud.init({ env: environment.data.cloudbaseEnvId, traceUser: true })
  },
})
