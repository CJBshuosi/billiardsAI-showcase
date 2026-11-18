const { callClientApi } = require('../../services/client-api')
const { contentError, clipView } = require('../../utils/content-view')
const { loadDeletionIntent, ensureDeletionIntent, clearDeletionIntent, isDeletionReceipt } = require('../../utils/clip-deletion')
const UNKNOWN_MESSAGE = '删除结果暂未确认，请重试确认同一删除请求。存储清理状态仍待确认。'
const INTENT_ERROR_MESSAGE = '暂时无法保存删除请求，请稍后重试。'
Page({
  data: {
    clipId: '', sessionId: '', clip: null, errorCode: null, error: '',
    loading: false, playbackUrl: '', state: 'unavailable', authorized: false,
    deletionState: 'idle', deletionMessage: '', deletionBusy: false, playbackBlocked: false,
  },
  onLoad(options) {
    this._unloaded = false
    this._playbackGeneration = 0
    this.setData({
      clipId: options && options.clip_id ? options.clip_id : '',
      sessionId: options && options.session_id ? options.session_id : '',
      state: 'unavailable',
    })
    if (!this.data.clipId) return
    try {
      if (loadDeletionIntent(this.data.clipId)) {
        this._playbackBlocked = true
        this.setData({ deletionState: 'unknown', deletionMessage: UNKNOWN_MESSAGE, playbackBlocked: true })
        return
      }
    } catch (_) {
      // An unreadable UX cache grants no rights; fresh playback still asks the server.
      this.setData({ deletionState: 'intent_error', deletionMessage: INTENT_ERROR_MESSAGE })
    }
    this.loadPlayback()
  },
  async loadPlayback() {
    if (this._unloaded || this._playbackBlocked || this.data.loading) return
    const generation = ++this._playbackGeneration
    this.setData({ errorCode: null, error: '', loading: true, playbackUrl: '', clip: null, authorized: false })
    const result = await callClientApi('clips.getPlayback', { clip_id: this.data.clipId })
    if (this._unloaded || this._playbackBlocked || generation !== this._playbackGeneration) return
    if (result.ok) {
      this.setData({
        playbackUrl: result.data.url, state: 'published', authorized: true,
        clip: result.data.clip ? clipView(result.data.clip) : null,
        loading: false,
      })
    } else this.setData({ errorCode: result.error.code, error: contentError(result.error.code), loading: false, state: 'unavailable' })
  },
  onPlaybackError() {
    if (this._unloaded || this._playbackBlocked) return
    this.setData({ playbackUrl: '', authorized: false, error: '片段还没打开，请重新加载播放内容。', state: 'unavailable' })
  },
  stopPlayer() {
    try { wx.createVideoContext('clip-player', this).stop() } catch (_) { /* Removing src also stops playback. */ }
  },
  async requestDeletion() {
    if (this._unloaded || this._deletionBusy || !this.data.clipId || this.data.deletionState === 'accepted') return
    this._deletionBusy = true
    this.setData({ deletionBusy: true })
    const clipId = this.data.clipId
    try {
      const confirmed = await new Promise(resolve => {
        wx.showModal({ title: '删除本人片段',
          content: '仅片段本人可申请删除。请求受理后从列表移除，存储清理随后进行。是否继续？',
          confirmText: '确认删除', confirmColor: '#a52c25',
          success: result => resolve(result.confirm === true), fail: () => resolve(false) })
      })
      if (!confirmed || this._unloaded) return
      this._playbackBlocked = true
      this._playbackGeneration += 1
      this.stopPlayer()
      await new Promise(resolve => this.setData({ playbackUrl: '', authorized: false, loading: false,
        playbackBlocked: true, error: '', errorCode: null, state: 'unavailable',
        deletionState: 'pending', deletionMessage: '正在提交删除请求…' }, resolve))
      if (this._unloaded) return
      let intent
      try { intent = await ensureDeletionIntent(clipId) } catch (_) {
        if (!this._unloaded) this.setData({ deletionState: 'intent_error', deletionMessage: INTENT_ERROR_MESSAGE })
        return
      }
      if (this._unloaded) return
      // Keep page references across navigation, including a response after detail unload.
      const lists = typeof getCurrentPages === 'function'
        ? getCurrentPages().filter(page => page.route === 'pages/clips/index') : []
      const result = await callClientApi('clips.delete', intent)
      if (result.ok && isDeletionReceipt(result.data, clipId)) {
        clearDeletionIntent(clipId)
        for (const list of lists) {
          if (typeof list.onClipDeletionAccepted === 'function') list.onClipDeletionAccepted(clipId)
        }
        if (!this._unloaded) this.setData({ deletionState: 'accepted',
          deletionMessage: '删除请求已受理，存储清理处理中。' })
      } else if (!this._unloaded) this.setData({ deletionState: 'unknown',
        deletionMessage: result.ok ? UNKNOWN_MESSAGE : `${contentError(result.error.code)} ${UNKNOWN_MESSAGE}` })
    } finally {
      this._deletionBusy = false
      if (!this._unloaded) this.setData({ deletionBusy: false })
    }
  },
  onUnload() {
    this._unloaded = true
    this._playbackBlocked = true
    this._playbackGeneration += 1
    this.stopPlayer()
  },
  backToClips() {
    const pages = typeof getCurrentPages === 'function' ? getCurrentPages() : []
    const previous = pages[pages.length - 2]
    if (previous && previous.route === 'pages/clips/index') wx.navigateBack({ delta: 1 })
    else wx.navigateTo({ url: '/pages/clips/index' })
  },
})
