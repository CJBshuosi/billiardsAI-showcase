const { tableCodeFromScanResult } = require('../../utils/scan-entry')
Page({
  data: { scanning: false, errorMessage: '', previewMode: false, previewTableCode: '' },
  onLoad() {
    const code = (getApp().globalData || {}).previewTableCode || ''
    const previewTableCode = /^[A-Za-z0-9_-]{32}$/u.test(code) ? code : ''
    this.setData({ previewMode: Boolean(previewTableCode), previewTableCode })
  },
  enterPreviewTable() {
    if (!this.data.previewTableCode) return
    wx.navigateTo({
      url: `/pages/join/index?table_code=${encodeURIComponent(this.data.previewTableCode)}`,
      fail: () => this.setData({ errorMessage: '暂时无法打开加入页面，请重试。' }),
    })
  },
  scan() {
    if (this.data.scanning) return
    this.setData({ scanning: true, errorMessage: '' })
    wx.scanCode({
      onlyFromCamera: true,
      success: result => {
        this.setData({ scanning: false })
        const code = tableCodeFromScanResult(result)
        if (!code) { this.setData({ errorMessage: '没有识别到本产品的球桌码，请扫描球桌大屏上的二维码。' }); return }
        wx.navigateTo({ url: `/pages/join/index?table_code=${encodeURIComponent(code)}`,
          fail: () => this.setData({ errorMessage: '暂时无法打开加入页面，请重试。' }) })
      },
      fail: error => this.setData({ scanning: false, errorMessage: /cancel/i.test(error.errMsg || '') ? '' : '暂时无法扫码，请检查相机权限后重试。' }),
    })
  },
})
