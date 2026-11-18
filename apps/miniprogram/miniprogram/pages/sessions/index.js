const { callClientApi } = require('../../services/client-api')
const { contentError, sessionView } = require('../../utils/content-view')
Page({
  data: { clipsMode: false, errorCode: null, error: '', loading: true, sessions: [] },
  onLoad(options) { this.setData({ clipsMode: Boolean(options && options.mode === 'clips') }) },
  onShow() { this.refresh() },
  async refresh() {
    if (this._refreshing) return
    this._refreshing = true
    this.setData({ errorCode: null, error: '', loading: true })
    const result = await callClientApi('sessions.list')
    this.setData(result.ok
      ? { loading: false, sessions: result.data.map(sessionView) }
      : { errorCode: result.error.code, error: contentError(result.error.code), sessions: [], loading: false })
    this._refreshing = false
  },
})
