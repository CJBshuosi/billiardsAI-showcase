const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')


function invitationKey() {
  return `challenge-${Date.now()}-${Math.random().toString(36).slice(2, 14)}`
}


Page({
  data: {
    loading: true, submitting: false, submitted: false, player: null,
    venues: [], venueIndex: -1, proposedDate: '', proposedTime: '',
    format: 'single', raceTarget: 3, errorMessage: '', challenge: null,
  },
  async onLoad(options) {
    const userId = options && options.user_id
    if (typeof userId !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/u.test(userId)) {
      this.setData({ loading: false, errorMessage: '邀请对象无效。' }); return
    }
    this._targetUserId = userId
    const [playerResult, venuesResult] = await Promise.all([
      callClientApi('ranking.player.get', { user_id: userId }),
      callClientApi('ranking.venues.list'),
    ])
    if (!playerResult.ok || !venuesResult.ok) {
      const code = !playerResult.ok ? playerResult.error.code : venuesResult.error.code
      this.setData({ loading: false, errorMessage: errorMessage(code) }); return
    }
    if (!playerResult.data.accept_challenges) {
      this.setData({ loading: false, player: playerResult.data, errorMessage: errorMessage('CHALLENGE_TARGET_UNAVAILABLE') }); return
    }
    this.setData({
      loading: false, player: playerResult.data,
      venues: Array.isArray(venuesResult.data) ? venuesResult.data : [],
    })
  },
  resetIntent() { this._idempotencyKey = '' },
  chooseVenue(event) { this.resetIntent(); this.setData({ venueIndex: Number(event.detail.value), submitted: false }) },
  chooseDate(event) { this.resetIntent(); this.setData({ proposedDate: event.detail.value, submitted: false }) },
  chooseTime(event) { this.resetIntent(); this.setData({ proposedTime: event.detail.value, submitted: false }) },
  chooseFormat(event) {
    const format = event.currentTarget.dataset.format
    if (!['single', 'race'].includes(format)) return
    this.resetIntent(); this.setData({ format, submitted: false })
  },
  updateRaceTarget(event) {
    this.resetIntent(); this.setData({ raceTarget: Number(event.detail.value), submitted: false })
  },
  async submit() {
    if (this.data.submitting || this.data.submitted) return
    const venue = this.data.venues[this.data.venueIndex]
    if (!venue || !this.data.proposedDate || !this.data.proposedTime) {
      this.setData({ errorMessage: errorMessage('CHALLENGE_INVALID') }); return
    }
    const proposedAt = new Date(`${this.data.proposedDate}T${this.data.proposedTime}:00+08:00`).toISOString()
    this._idempotencyKey = this._idempotencyKey || invitationKey()
    this.setData({ submitting: true, errorMessage: '' })
    const result = await callClientApi('challenge.create', {
      idempotency_key: this._idempotencyKey,
      target_user_id: this._targetUserId,
      venue_id: venue.venue_id,
      proposed_at: proposedAt,
      format: this.data.format,
      race_target: this.data.format === 'race' ? this.data.raceTarget : null,
    })
    if (!result.ok) {
      this.setData({ submitting: false, errorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({ submitting: false, submitted: true, challenge: result.data })
  },
})
