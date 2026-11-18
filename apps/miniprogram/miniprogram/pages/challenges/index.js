const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')


Page({
  data: { loading: true, challenges: [], actingId: '', errorMessage: '' },
  async onShow() { await this.load() },
  async load() {
    this.setData({ loading: true, errorMessage: '' })
    const result = await callClientApi('challenge.list')
    if (!result.ok) {
      this.setData({ loading: false, errorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({ loading: false, challenges: Array.isArray(result.data) ? result.data : [] })
  },
  async act(event) {
    const challengeId = event.currentTarget.dataset.id
    const action = event.currentTarget.dataset.action
    const challenge = this.data.challenges.find(item => item.challenge_id === challengeId)
    const allowed = challenge && challenge.status === 'pending'
      && ((challenge.direction === 'incoming' && ['accept', 'reject'].includes(action))
        || (challenge.direction === 'outgoing' && action === 'cancel'))
    if (!allowed || this.data.actingId) return
    this.setData({ actingId: challengeId, errorMessage: '' })
    const result = await callClientApi(`challenge.${action}`, { challenge_id: challengeId })
    if (!result.ok) {
      this.setData({ actingId: '', errorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({
      actingId: '',
      challenges: this.data.challenges.map(item => item.challenge_id === challengeId
        ? { ...item, status: result.data.status } : item),
    })
  },
  openScanner() {
    wx.navigateTo({ url: '/pages/scan/index' })
  },
})
