const { callClientApi } = require('../../services/client-api')
const { contentError, clipView } = require('../../utils/content-view')
Page({
  data: { clips: [], errorCode: null, error: '', loading: true, loadingMore: false, nextCursor: null, sessionId: '' },
  onLoad(options) {
    this._unloaded = false
    this.setData({ sessionId: options && options.session_id ? options.session_id : '' })
    this.refresh()
  },
  onShow() {
    if (this._scrollTop && typeof wx.pageScrollTo === 'function') wx.pageScrollTo({ scrollTop: this._scrollTop, duration: 0 })
  },
  onPageScroll(event) {
    if (!this._refreshing) this._scrollTop = event.scrollTop
  },
  onUnload() { this._unloaded = true },
  onClipDeletionAccepted(clipId) {
    if (this._unloaded) return
    if (!this._deletedClipIds) this._deletedClipIds = new Set()
    this._deletedClipIds.add(clipId)
    this.setData({ clips: this.data.clips.filter(clip => clip.clip_id !== clipId) })
  },
  onReachBottom() { return this.loadMore() },
  async refresh() {
    if (this._refreshing || this.data.loadingMore) return
    this._refreshing = true
    this._scrollTop = 0
    this.setData({ loading: true, clips: [], nextCursor: null, errorCode: null, error: '' })
    await this.loadPage(true)
    this._refreshing = false
  },
  async loadMore() {
    if (this.data.loading || this.data.loadingMore || !this.data.nextCursor) return
    this.setData({ loadingMore: true, errorCode: null, error: '' })
    await this.loadPage(false)
  },
  async loadPage(reset) {
    const payload = this.data.sessionId ? { session_id: this.data.sessionId } : {}
    if (!reset) payload.cursor = this.data.nextCursor
    const result = await callClientApi('clips.listPage', payload)
    if (result.ok) {
      const clips = reset ? [] : [...this.data.clips]
      const known = new Set(clips.map(clip => clip.clip_id))
      for (const item of result.data.items) {
        if (!known.has(item.clip_id) && !(this._deletedClipIds && this._deletedClipIds.has(item.clip_id))) {
          clips.push(clipView(item))
          known.add(item.clip_id)
        }
      }
      this.setData({ clips, nextCursor: result.data.next_cursor, loading: false, loadingMore: false }, () => {
        if (reset && typeof wx.pageScrollTo === 'function') wx.pageScrollTo({ scrollTop: 0, duration: 0 })
      })
    } else this.setData({ errorCode: result.error.code, error: contentError(result.error.code), loading: false, loadingMore: false })
  },
})
