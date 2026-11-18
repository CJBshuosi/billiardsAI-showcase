const { callClientApi } = require('../../services/client-api')
const { analysisView, contentError, sessionView } = require('../../utils/content-view')
const { errorMessage } = require('../../utils/ui')

const REPORT_REASONS = ['suspected_substitution', 'identity_mismatch', 'result_incorrect', 'match_not_completed', 'other']
const REPORT_LABELS = ['疑似他人代打', '参赛身份不一致', '赛果不正确', '比赛没有完成', '其他问题']
const APPEAL_REASONS = ['result_is_valid', 'identity_evidence_available', 'other']
const APPEAL_LABELS = ['赛果有效，请复核', '我可以补充身份依据', '其他说明']
const ANALYSIS_STATES = new Set(['PROCESSING', 'AVAILABLE', 'PARTIAL', 'FAILED_RETRYING'])
const POLL_DELAYS = [1500, 3000, 5000, 10000]

function outcomeView(outcome, isTest = false) {
  if (!outcome) return null
  const labels = {
    awaiting_confirmation: '等待双方确认赛果', disputed: '赛果正在核对',
    confirmation_expired: '确认已超时，本场不自动计分', processing: '资格与积分处理中',
    review_pending: '本场暂缓审核', not_ranked: '本场暂不计排名',
    corrected: '赛果已更正，排名重算中', revoked: '本场已撤销',
    settled: isTest ? '开发测试积分已更新' : '正式积分已更新',
  }
  const view = { ...outcome, label: labels[outcome.status] || '排名状态待核对' }
  if (outcome.status === 'settled' && outcome.placement_status === 'placed') {
    view.delta_label = `${outcome.rating_delta > 0 ? '+' : ''}${outcome.rating_delta}`
  }
  return view
}

function choose(labels) {
  return new Promise(resolve => {
    wx.showActionSheet({ itemList: labels, success: result => resolve(result.tapIndex), fail: () => resolve(-1) })
  })
}

Page({
  data: {
    errorCode: null, error: '', loading: true, session: null, sessionId: '',
    analysis: null, analysisView: null, contentReady: false,
    shotCount: '—', clipCount: '—', rankingMatch: null, rankingScope: null, rankingOutcome: null,
    rankingResult: null, rankingOutcomeView: null,
    rankingActionBusy: false, rankingActionError: '', reportSubmitted: false,
  },
  onLoad(options) {
    this.setData({ sessionId: options && options.session_id ? options.session_id : '' })
    this._analysisVisible = true
    this._analysisLoaded = true
    this._analysisGeneration = 0
    this._pollAttempt = 0
    return this.refresh()
  },
  onShow() {
    if (!this._analysisLoaded || this._analysisVisible) return
    this._analysisVisible = true
    if (this._refreshing) {
      this.scheduleAnalysisPolling()
      return
    }
    return this.refresh()
  },
  onHide() {
    this._analysisVisible = false
    this._analysisGeneration += 1
    this.stopAnalysisPolling()
  },
  onUnload() {
    this._analysisVisible = false
    this._analysisGeneration += 1
    this.stopAnalysisPolling()
  },
  stopAnalysisPolling() {
    if (this._analysisTimer) clearTimeout(this._analysisTimer)
    this._analysisTimer = null
  },
  scheduleAnalysisPolling() {
    this.stopAnalysisPolling()
    if (!this._analysisVisible) return
    const delay = POLL_DELAYS[Math.min(this._pollAttempt, POLL_DELAYS.length - 1)]
    this._pollAttempt += 1
    this._analysisTimer = setTimeout(() => {
      this._analysisTimer = null
      if (this._analysisVisible) this.refresh()
    }, delay)
  },
  async refresh() {
    if (this._refreshing) return
    this._refreshing = true
    const sessionId = this.data.sessionId
    const generation = ++this._analysisGeneration
    if (!this.data.analysis) this.setData({ loading: true, errorCode: null, error: '' })
    try {
      const statusResult = await callClientApi('sessions.analysis.get', { session_id: sessionId })
      if (!this._analysisVisible || generation !== this._analysisGeneration
          || sessionId !== this.data.sessionId) return
      const status = statusResult.ok && statusResult.data && statusResult.data.status
      if (ANALYSIS_STATES.has(status)) {
        this.setData({
          loading: false, errorCode: null, error: '', analysis: statusResult.data,
          analysisView: analysisView(statusResult.data),
        })
        if (status === 'AVAILABLE' || status === 'PARTIAL') {
          await this.loadFinalContent(generation, sessionId)
        } else {
          this.setData({
            session: null, contentReady: false, shotCount: '—', clipCount: '—',
            rankingMatch: null, rankingScope: null, rankingOutcome: null, rankingOutcomeView: null,
            rankingResult: null,
          })
        }
        if (status !== 'AVAILABLE') this.scheduleAnalysisPolling()
        else this.stopAnalysisPolling()
        return
      }
      await this.loadFinalContent(generation, sessionId)
    } catch (_error) {
      if (this._analysisVisible && generation === this._analysisGeneration
          && sessionId === this.data.sessionId) {
        this.setData({ loading: false, errorCode: 'NETWORK_UNAVAILABLE', error: contentError('NETWORK_UNAVAILABLE') })
      }
    } finally {
      this._refreshing = false
    }
  },
  async loadFinalContent(generation, sessionId) {
    const result = await callClientApi('sessions.get', { session_id: sessionId })
    if (!this._analysisVisible || generation !== this._analysisGeneration
        || sessionId !== this.data.sessionId) return
    if (!result.ok) {
      this.setData({
        errorCode: result.error.code, error: contentError(result.error.code), loading: false,
        session: null, contentReady: false,
      })
      return
    }
    this.setData({
      loading: false, session: sessionView(result.data), contentReady: true,
      shotCount: Array.isArray(result.data.shots) ? result.data.shots.length : '—',
      rankingMatch: null, rankingScope: null, rankingOutcome: null, rankingOutcomeView: null,
      rankingActionError: '', rankingResult: null,
    })
    const clips = await callClientApi('clips.list', { session_id: sessionId })
    if (!this._analysisVisible || generation !== this._analysisGeneration
        || sessionId !== this.data.sessionId) return
    if (clips.ok && Array.isArray(clips.data)) {
      this.setData({ clipCount: clips.data.length >= 100 ? '100+' : clips.data.length })
    }
    if (!clips.ok && clips.error.code === 'FORBIDDEN') {
      this.setData({
        session: null, contentReady: false, errorCode: clips.error.code,
        error: contentError(clips.error.code), shotCount: '—', clipCount: '—',
      })
    }
    if (!this.data.session) return
    const ranking = await callClientApi('ranking.match.get', { session_id: sessionId })
    if (!this._analysisVisible || generation !== this._analysisGeneration
        || sessionId !== this.data.sessionId) return
    if (ranking.ok && ranking.data && ranking.data.match_id) {
      const outcome = ranking.data.ranking_outcome || null
      const rankingScope = ranking.data.ranking_scope || null
      this.setData({
        rankingMatch: ranking.data, rankingScope, rankingOutcome: outcome,
        rankingResult: outcome && outcome.corrected_result
          ? outcome.corrected_result : ranking.data.result,
        rankingOutcomeView: outcomeView(outcome, Boolean(rankingScope && rankingScope.is_test)),
      })
    }
  },
  async reportRankingMatch() {
    if (!this.data.rankingMatch || this.data.rankingActionBusy || this.data.reportSubmitted) return
    const index = await choose(REPORT_LABELS)
    const reasonCode = REPORT_REASONS[index]
    if (!reasonCode) return
    this.setData({ rankingActionBusy: true, rankingActionError: '' })
    const result = await callClientApi('ranking.match.report', {
      match_id: this.data.rankingMatch.match_id, reason_code: reasonCode,
    })
    if (!result.ok) {
      this.setData({ rankingActionBusy: false, rankingActionError: errorMessage(result.error.code) }); return
    }
    this.setData({ rankingActionBusy: false, reportSubmitted: true })
  },
  async appealRankingReview() {
    const outcome = this.data.rankingOutcome
    if (!this.data.rankingMatch || !outcome || outcome.can_appeal !== true
        || outcome.actor_appealed || this.data.rankingActionBusy) return
    const index = await choose(APPEAL_LABELS)
    const reasonCode = APPEAL_REASONS[index]
    if (!reasonCode) return
    this.setData({ rankingActionBusy: true, rankingActionError: '' })
    const result = await callClientApi('ranking.review.appeal', {
      match_id: this.data.rankingMatch.match_id, reason_code: reasonCode,
    })
    if (!result.ok) {
      this.setData({ rankingActionBusy: false, rankingActionError: errorMessage(result.error.code) }); return
    }
    const updated = { ...outcome, actor_appealed: true }
    this.setData({
      rankingActionBusy: false, rankingOutcome: updated,
      rankingOutcomeView: outcomeView(
        updated,
        Boolean(this.data.rankingScope && this.data.rankingScope.is_test),
      ),
    })
  },
})
