const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')

Page({
  data: {
    avatarFileId: '',
    errorCode: null,
    nickname: '',
    saving: false,
    saved: false,
    errorMessage: '',
  },

  onLoad() {
    const user = getApp().globalData.currentUser
    if (!user) {
      wx.reLaunch({ url: '/pages/login/index' })
      return
    }
    this.setData({
      avatarFileId: (user.profile || {}).avatar_file_id || '',
      nickname: (user.profile || {}).nickname || '',
    })
  },

  updateNickname(event) {
    this.setData({ nickname: event.detail.value, saved: false })
  },

  async save() {
    if (this.data.saving) return
    this.setData({ saving: true, saved: false, errorCode: null, errorMessage: '' })
    const result = await callClientApi('profile.update', {
      nickname: this.data.nickname,
      avatar_file_id: this.data.avatarFileId || null,
    })
    if (!result.ok) {
      this.setData({ saving: false, errorCode: result.error.code, errorMessage: errorMessage(result.error.code) })
      return
    }
    getApp().globalData.currentUser = { ...getApp().globalData.currentUser, ...result.data }
    this.setData({ saving: false, saved: true })
  },
})
