const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')

Page({
  data: {
    available: false, guest: true, loading: false, kind: 'long_term',
    entries: [], myPosition: null, myStatus: null, reasonCode: '',
    snapshotId: '', snapshotVersion: null, asOf: '', errorMessage: '',
    page: 0, hasMore: false, loadingMore: false,
    rankingTitle: '周口球技榜', isTest: false, scopeNotice: '',
  },
  async onShow() {
    const app = getApp()
    const guest = !app.globalData.currentUser
    this.setData({ guest })
    if (guest || app.globalData.configurationError) {
      this.setData({ available: false, loading: false, entries: [], reasonCode: 'RANKING_NOT_PUBLISHED' })
      return
    }
    await this.loadRanking(false)
  },
  async loadRanking(append = false) {
    if (this.data.loading || this.data.loadingMore) return
    const nextPage = append ? this.data.page + 1 : 1
    this.setData(append ? { loadingMore: true, errorMessage: '' } : { loading: true, errorMessage: '' })
    const result = await callClientApi('ranking.list', {
      kind: this.data.kind, page: nextPage, page_size: 50,
    })
    if (!result.ok) {
      this.setData({
        loading: false, loadingMore: false,
        available: append ? this.data.available : false,
        entries: append ? this.data.entries : [],
        errorMessage: errorMessage(result.error.code),
      })
      return
    }
    const data = result.data
    if (append && data.snapshot_id !== this.data.snapshotId) {
      this.setData({ loadingMore: false, hasMore: false, errorMessage: '排名刚刚更新，请重新加载。' })
      return
    }
    const incoming = Array.isArray(data.entries) ? data.entries : []
    const scope = data.ranking_scope || {}
    const isTest = scope.is_test === true
    this.setData({
      loading: false, loadingMore: false,
      available: data.availability === 'published',
      entries: append ? [...this.data.entries, ...incoming] : incoming,
      myPosition: data.my_position || null,
      myStatus: data.my_status || null,
      reasonCode: data.reason_code || '',
      snapshotId: data.snapshot_id || '',
      snapshotVersion: data.snapshot_version || null,
      asOf: data.as_of || '',
      page: data.page || 0, hasMore: data.has_more === true,
      rankingTitle: scope.display_name || '周口球技榜',
      isTest,
      scopeNotice: isTest ? '仅用于开发验收，不代表周口正式排名' : '',
    })
  },
  async loadMore() {
    if (!this.data.available || !this.data.hasMore) return
    await this.loadRanking(true)
  },
  async switchKind(event) {
    const kind = event.currentTarget.dataset.kind
    if (!['long_term', 'season'].includes(kind) || kind === this.data.kind || this.data.loading) return
    this.setData({ kind, available: false, entries: [], myPosition: null })
    this.setData({ page: 0, hasMore: false })
    await this.loadRanking(false)
  },
  openPlayer(event) {
    const userId = event.currentTarget.dataset.userId
    if (typeof userId !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/u.test(userId)) return
    const snapshot = typeof this.data.snapshotId === 'string'
      && /^[A-Za-z0-9._-]{1,128}$/u.test(this.data.snapshotId)
      ? `&snapshot_id=${encodeURIComponent(this.data.snapshotId)}` : ''
    wx.navigateTo({ url: `/pages/ranking-player/index?user_id=${encodeURIComponent(userId)}${snapshot}` })
  },
})
