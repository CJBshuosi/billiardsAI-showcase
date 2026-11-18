const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')


const DISPUTE_REASONS = ['winner_mismatch', 'score_mismatch', 'not_completed', 'identity_issue', 'other']
const DISPUTE_LABELS = ['胜方不对', '比分不对', '比赛没有完成', '参赛身份有问题', '其他问题']

Page({
  data: { loading: true, matches: [], actingId: '', errorMessage: '' },
  async onShow() { await this.load() },
  async load() {
    this.setData({ loading: true, errorMessage: '' })
    const result = await callClientApi('ranking.results.pending')
    if (!result.ok) {
      this.setData({ loading: false, errorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({ loading: false, matches: Array.isArray(result.data) ? result.data : [] })
  },
  async confirm(event) {
    const matchId = event.currentTarget.dataset.id
    const match = this.data.matches.find(item => item.match_id === matchId)
    if (!match || match.actor_result_confirmed || this.data.actingId) return
    await this.apply('ranking.result.confirm', matchId, {})
  },
  dispute(event) {
    const matchId = event.currentTarget.dataset.id
    const match = this.data.matches.find(item => item.match_id === matchId)
    if (!match || match.actor_result_confirmed || this.data.actingId) return
    wx.showActionSheet({
      itemList: DISPUTE_LABELS,
      success: result => {
        const reason = DISPUTE_REASONS[result.tapIndex]
        if (reason) this.apply('ranking.result.dispute', matchId, { reason })
      },
    })
  },
  async apply(action, matchId, extra) {
    this.setData({ actingId: matchId, errorMessage: '' })
    const result = await callClientApi(action, { match_id: matchId, ...extra })
    if (!result.ok) {
      this.setData({ actingId: '', errorMessage: errorMessage(result.error.code) }); return
    }
    const updated = result.data
    this.setData({
      actingId: '',
      matches: ['fact_package_ready', 'disputed'].includes(updated.status)
        ? this.data.matches.filter(item => item.match_id !== matchId)
        : this.data.matches.map(item => item.match_id === matchId ? updated : item),
    })
  },
})
