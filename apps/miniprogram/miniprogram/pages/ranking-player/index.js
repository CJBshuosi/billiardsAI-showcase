const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')

Page({
  data: {
    loading: true, player: null, errorMessage: '', isSelf: false,
    rankingTitle: '周口球技榜', isTest: false, scopeNotice: '',
  },
  async onLoad(options) {
    const userId = options && options.user_id
    if (typeof userId !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/u.test(userId)) {
      this.setData({ loading: false, errorMessage: '这个球友主页暂不可查看。' }); return
    }
    this._userId = userId
    const snapshotId = options && typeof options.snapshot_id === 'string'
      && /^[A-Za-z0-9._-]{1,128}$/u.test(options.snapshot_id) ? options.snapshot_id : null
    const current = getApp().globalData.currentUser
    this.setData({ isSelf: Boolean(current && current.user_id === userId) })
    const result = await callClientApi('ranking.player.get', {
      user_id: userId, ...(snapshotId ? { snapshot_id: snapshotId } : {}),
    })
    if (!result.ok) {
      this.setData({ loading: false, errorMessage: errorMessage(result.error.code) }); return
    }
    const scope = result.data.ranking_scope || {}
    const isTest = scope.is_test === true
    this.setData({
      loading: false, player: result.data, errorMessage: '',
      rankingTitle: scope.display_name || '周口球技榜',
      isTest,
      scopeNotice: isTest ? '仅用于开发验收，不代表周口正式排名' : '',
    })
  },
  challenge() {
    if (!this.data.player || !this.data.player.accept_challenges || this.data.isSelf) return
    wx.navigateTo({ url: `/pages/challenge/index?user_id=${encodeURIComponent(this.data.player.user_id)}` })
  },
})
