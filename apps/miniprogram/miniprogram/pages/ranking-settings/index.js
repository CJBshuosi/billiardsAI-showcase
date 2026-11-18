const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')

Page({
  data: {
    loading: true, saving: false, nickname: '', isPublic: false,
    acceptChallenges: false, saved: false, errorMessage: '',
  },
  async onLoad() {
    const result = await callClientApi('ranking.profile.get')
    if (!result.ok) {
      this.setData({ loading: false, errorMessage: errorMessage(result.error.code) }); return
    }
    const profile = result.data
    this.setData({
      loading: false,
      nickname: profile && profile.nickname || '',
      isPublic: Boolean(profile && profile.is_public),
      acceptChallenges: Boolean(profile && profile.accept_challenges),
    })
  },
  updateNickname(event) { this.setData({ nickname: event.detail.value, saved: false }) },
  togglePublic(event) { this.setData({ isPublic: event.detail.value === true, saved: false }) },
  toggleChallenges(event) { this.setData({ acceptChallenges: event.detail.value === true, saved: false }) },
  async save() {
    if (this.data.saving) return
    this.setData({ saving: true, saved: false, errorMessage: '' })
    const result = await callClientApi('ranking.profile.save', {
      nickname: this.data.nickname,
      is_public: this.data.isPublic,
      accept_challenges: this.data.acceptChallenges,
    })
    if (!result.ok) {
      this.setData({ saving: false, errorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({
      saving: false, saved: true, nickname: result.data.nickname,
      isPublic: result.data.is_public, acceptChallenges: result.data.accept_challenges,
    })
  },
})
