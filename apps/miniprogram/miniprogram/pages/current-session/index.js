const { callClientApi } = require('../../services/client-api')
const { errorMessage } = require('../../utils/ui')
const { createCommandIntent, describeFieldCommand, loadPendingCommand, nextCommandState, rememberPendingCommand } = require('../../utils/field-command')

const BUSY = ['pending', 'claimed', 'unknown']
const intentKey = sessionId => `field-ui-intent:${sessionId}`

Page({
  data: {
    actorRole: null, activeRole: null, blueGroup: null, blueOccupied: false,
    breaker: null, breakerLocked: false, commandId: '', commandMessage: '', commandResultCode: null,
    commandState: null, commandErrorMessage: '', displayName: '', errorCode: null,
    errorMessage: '', loading: true, refreshing: false, redGroup: null,
    redOccupied: false, requestId: '', revision: 0, sessionId: '', started: false,
    status: 'pending', submitting: false, showDebug: false,
    matchState: null, matchBusy: false, matchErrorMessage: '',
    actorSettingsConfirmed: false,
    matchMode: 'observation', matchFormat: 'single', raceTarget: 3,
    rankingIntent: true, selectedBreaker: null,
  },
  onLoad(options) {
    this._unloaded = false
    this.setData({ requestId: options && options.request_id ? options.request_id : '' })
  },
  onShow() {
    this._hidden = false
    clearInterval(this.poller)
    this.refresh()
    this.poller = setInterval(() => this.refresh(), 1500)
  },
  onHide() {
    this._hidden = true
    this._refreshToken = (this._refreshToken || 0) + 1
    this._refreshing = false
    clearInterval(this.poller); this.poller = null
  },
  onUnload() { this.onHide(); this._unloaded = true },
  async refresh() {
    if (this._refreshing || this._unloaded || this._hidden) return
    if (!this.data.requestId) {
      this.setData({ loading: false, errorCode: 'REQUEST_INVALID', errorMessage: errorMessage('REQUEST_INVALID') }); return
    }
    this._refreshing = true
    const token = this._refreshToken = (this._refreshToken || 0) + 1
    const requestId = this.data.requestId
    const current = () => token === this._refreshToken && requestId === this.data.requestId && !this._unloaded && !this._hidden
    this.setData({ refreshing: true })
    try {
      const joined = await callClientApi('join.status', { request_id: requestId })
      if (!current()) return
      if (!joined.ok) { this.setData({ loading: false, errorCode: joined.error.code, errorMessage: errorMessage(joined.error.code) }); return }
      const field = await callClientApi('field.current', { session_id: joined.data.session_id })
      if (!current()) return
      if (!field.ok) { this.setData({ loading: false, errorCode: field.error.code, errorMessage: errorMessage(field.error.code) }); return }
      if (this.data.sessionId && this.data.sessionId !== joined.data.session_id) {
        this._intent = null
        this.setData({ breakerLocked: false, commandId: '', commandState: null, commandMessage: '', commandErrorMessage: '' })
      }
      this.setData({
        actorRole: field.data.actor_role || null, activeRole: field.data.active_role || null,
        blueGroup: field.data.blue_group || null, blueOccupied: Boolean(field.data.blue_occupied),
        breaker: field.data.breaker || null,
        breakerLocked: Boolean(field.data.started || field.data.breaker || this.data.breakerLocked),
        displayName: joined.data.display_name,
        errorCode: null, errorMessage: '', loading: false,
        redGroup: field.data.red_group || null, redOccupied: Boolean(field.data.red_occupied),
        revision: field.data.revision, sessionId: joined.data.session_id,
        started: Boolean(field.data.started), status: field.data.status,
      })
      if (field.data.actor_role && field.data.red_occupied && field.data.blue_occupied) {
        await this.refreshRankingMatch()
        if (!current()) return
      }
      if (!this.data.commandId) {
        const pending = loadPendingCommand(wx, this.data.sessionId)
        if (pending) this.setData({ commandId: pending.commandId, commandState: pending.state })
        else {
          const saved = wx.getStorageSync(intentKey(this.data.sessionId))
          if (saved && saved.intent && typeof saved.intent.idempotency_key === 'string') {
            this._intent = saved.intent
            this.setData({ commandState: 'unknown', commandMessage: saved.message || '' })
          }
        }
      }
      if (this.data.commandId && ['pending', 'claimed'].includes(this.data.commandState)) await this.refreshCommand()
    } finally {
      if (current()) { this._refreshing = false; this.setData({ refreshing: false }) }
    }
  },
  async refreshRankingMatch() {
    if (!this.data.sessionId) return
    const result = await callClientApi('ranking.match.get', { session_id: this.data.sessionId })
    if (!result.ok) {
      this.setData({ matchErrorMessage: errorMessage(result.error.code) }); return
    }
    const state = result.data
    const settings = state && state.settings
    this.setData({
      matchState: state,
      actorSettingsConfirmed: Boolean(state && state.settings_confirmations
        && state.settings_confirmations.includes(state.actor_role)),
      matchErrorMessage: '',
      matchMode: settings ? settings.mode : this.data.matchMode,
      matchFormat: settings ? settings.format : this.data.matchFormat,
      raceTarget: settings && settings.race_target || this.data.raceTarget,
      rankingIntent: settings ? settings.ranking_intent : this.data.rankingIntent,
      selectedBreaker: settings ? settings.first_breaker : this.data.selectedBreaker,
    })
  },
  async refreshCommand() {
    const commandId = this.data.commandId
    const sessionId = this.data.sessionId
    const result = await callClientApi('field.command.status', { command_id: commandId })
    if (this._unloaded || this._hidden || commandId !== this.data.commandId || sessionId !== this.data.sessionId) return
    if (!result.ok) { this.setData({ commandErrorMessage: errorMessage(result.error.code) }); return }
    const command = nextCommandState({ commandId, state: this.data.commandState, resultCode: this.data.commandResultCode }, result.data)
    this.setData({ commandState: command.state, commandResultCode: command.resultCode,
      breakerLocked: ['rejected','expired'].includes(command.state) ? false : this.data.breakerLocked,
      commandErrorMessage: ['rejected','expired'].includes(command.state) ? errorMessage(command.resultCode) : '' })
    rememberPendingCommand(wx, sessionId, command)
  },
  async submitCommand(commandType, payload = {}) {
    if (!this.data.sessionId || this.data.status !== 'open' || this.data.errorCode || this.data.loading || this._submitting || BUSY.includes(this.data.commandState)) return
    this._submitting = true
    const sessionId = this.data.sessionId
    this.setData({ submitting: true, commandErrorMessage: '' })
    try {
      const intent = await createCommandIntent(commandType, payload, this.data.revision)
      if (this._unloaded || sessionId !== this.data.sessionId || this.data.status !== 'open') return
      const message = describeFieldCommand(commandType, payload, this.data.activeRole)
      // Persist before sending. If storage fails, no command is submitted.
      wx.setStorageSync(intentKey(sessionId), { intent, message })
      this._intent = intent
      this.setData({ commandId: '', commandMessage: message })
      await this.sendIntent(sessionId, intent, false)
    } catch (_error) {
      if (!this._unloaded) this.setData({
        breakerLocked: commandType === 'set_breaker' && !this._intent ? false : this.data.breakerLocked,
        commandState: this._intent ? 'unknown' : 'rejected',
        commandErrorMessage: '暂未发出或确认操作，请检查设备后重试。',
      })
    } finally {
      this._submitting = false
      if (!this._unloaded) this.setData({ submitting: false })
    }
  },
  async sendIntent(sessionId, intent, retry) {
    const result = await callClientApi('field.command.submit', { session_id: sessionId, ...intent })
    if (!result.ok) {
      const uncertain = retry || ['NETWORK_UNAVAILABLE', 'INTERNAL_ERROR'].includes(result.error.code)
      if (!uncertain) { wx.removeStorageSync(intentKey(sessionId)); this._intent = null }
      if (!this._unloaded && sessionId === this.data.sessionId) this.setData({
        breakerLocked: intent.command_type === 'set_breaker' && !uncertain ? false : this.data.breakerLocked,
        commandState: uncertain ? 'unknown' : 'rejected', commandResultCode: result.error.code,
        commandErrorMessage: uncertain ? '尚未确认操作是否完成。请核对大屏，可用原请求重新查询；不要另发同一操作。' : errorMessage(result.error.code),
      })
      return
    }
    const command = nextCommandState(null, result.data)
    if (!command) {
      if (!this._unloaded) this.setData({ commandState: 'unknown', commandErrorMessage: '返回结果不完整，请联系工作人员核对。' })
      return
    }
    rememberPendingCommand(wx, sessionId, command)
    wx.removeStorageSync(intentKey(sessionId))
    this._intent = null
    if (!this._unloaded && sessionId === this.data.sessionId) this.setData({
      breakerLocked: intent.command_type === 'set_breaker' && ['rejected','expired'].includes(command.state)
        ? false : this.data.breakerLocked,
      commandId: command.commandId, commandState: command.state, commandResultCode: command.resultCode,
      commandErrorMessage: ['rejected','expired'].includes(command.state) ? errorMessage(command.resultCode) : '',
    })
  },
  async retrySubmission() {
    if (!this._intent || this._submitting || this.data.commandState !== 'unknown' || this.data.status !== 'open') return
    this._submitting = true; this.setData({ submitting: true })
    try { await this.sendIntent(this.data.sessionId, this._intent, true) }
    finally { this._submitting = false; if (!this._unloaded) this.setData({ submitting: false }) }
  },
  claimRole(event) {
    const role = event.currentTarget.dataset.role
    if (!['red','blue'].includes(role) || this.data.actorRole || this.data.started || this.data[`${role}Occupied`]) return
    return this.submitCommand('claim_role', { role })
  },
  releaseRole() {
    if (!this.data.actorRole || this.data.started) return
    wx.showModal({ title: '释放本场身份？', content: '释放后原身份可能被其他球友认领，重新加入时需再次选择。', confirmText: '确认释放',
      success: result => { if (result.confirm) this.submitCommand('release_role') } })
  },
  setBreaker(event) {
    const role = event.currentTarget.dataset.role
    if (!['red','blue'].includes(role) || !this.data.actorRole || this.data.started || this.data.breaker
        || this.data.breakerLocked || this._submitting || this.data.matchBusy || BUSY.includes(this.data.commandState)
        || !this.data.redOccupied || !this.data.blueOccupied) return
    this.setData({ selectedBreaker: role, matchErrorMessage: '' })
  },
  chooseMatchMode(event) {
    if (this.data.matchBusy || this.data.started) return
    const mode = event.currentTarget.dataset.mode
    if (!['observation', 'assistance'].includes(mode)) return
    this.setData({ matchMode: mode, rankingIntent: mode === 'observation' })
  },
  chooseMatchFormat(event) {
    if (this.data.matchBusy || this.data.started) return
    const format = event.currentTarget.dataset.format
    if (!['single', 'race'].includes(format)) return
    this.setData({ matchFormat: format })
  },
  updateRaceTarget(event) { this.setData({ raceTarget: Number(event.detail.value) }) },
  toggleRankingIntent(event) {
    if (this.data.matchMode !== 'observation') return
    this.setData({ rankingIntent: event.detail.value === true })
  },
  async proposeMatchSettings() {
    if (this.data.matchBusy || this.data.started || !this.data.selectedBreaker) return
    this.setData({ matchBusy: true, matchErrorMessage: '' })
    const result = await callClientApi('ranking.match.settings.propose', {
      session_id: this.data.sessionId,
      mode: this.data.matchMode,
      format: this.data.matchFormat,
      race_target: this.data.matchFormat === 'race' ? this.data.raceTarget : null,
      first_breaker: this.data.selectedBreaker,
      ranking_intent: this.data.matchMode === 'observation' && this.data.rankingIntent,
    })
    if (!result.ok) {
      this.setData({ matchBusy: false, matchErrorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({
      matchBusy: false, matchState: result.data,
      actorSettingsConfirmed: result.data.settings_confirmations.includes(result.data.actor_role),
    })
  },
  async confirmMatchSettings() {
    if (this.data.matchBusy || !this.data.matchState
        || this.data.matchState.status !== 'awaiting_settings_confirmations') return
    this.setData({ matchBusy: true, matchErrorMessage: '' })
    const result = await callClientApi('ranking.match.settings.confirm', { session_id: this.data.sessionId })
    if (!result.ok) {
      this.setData({ matchBusy: false, matchErrorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({
      matchBusy: false, matchState: result.data,
      actorSettingsConfirmed: result.data.settings_confirmations.includes(result.data.actor_role),
    })
  },
  async confirmCurrentResult() {
    const state = this.data.matchState
    if (!state || state.status !== 'awaiting_result_confirmations'
        || state.actor_result_confirmed || this.data.matchBusy) return
    this.setData({ matchBusy: true, matchErrorMessage: '' })
    const result = await callClientApi('ranking.result.confirm', { match_id: state.match_id })
    if (!result.ok) {
      this.setData({ matchBusy: false, matchErrorMessage: errorMessage(result.error.code) }); return
    }
    this.setData({ matchBusy: false, matchState: result.data })
  },
  disputeCurrentResult() {
    const state = this.data.matchState
    if (!state || state.status !== 'awaiting_result_confirmations'
        || state.actor_result_confirmed || this.data.matchBusy) return
    const reasons = ['winner_mismatch', 'score_mismatch', 'not_completed', 'identity_issue', 'other']
    wx.showActionSheet({
      itemList: ['胜方不对', '比分不对', '比赛没有完成', '参赛身份有问题', '其他问题'],
      success: async selection => {
        const reason = reasons[selection.tapIndex]
        if (!reason) return
        this.setData({ matchBusy: true, matchErrorMessage: '' })
        const result = await callClientApi('ranking.result.dispute', {
          match_id: state.match_id, reason,
        })
        if (!result.ok) {
          this.setData({ matchBusy: false, matchErrorMessage: errorMessage(result.error.code) }); return
        }
        this.setData({ matchBusy: false, matchState: result.data })
      },
    })
  },
  startMatch() {
    const state = this.data.matchState
    if (!state || state.status !== 'ready' || !state.settings || this.data.breakerLocked) return
    this.setData({ breakerLocked: true })
    return this.submitCommand('set_breaker', { role: state.settings.first_breaker })
  },
  assignGroups(event) { return this.submitCommand('assign_groups', { red_group: event.currentTarget.dataset.group }) },
  continueTurn() { return this.submitCommand('continue_turn') },
  switchTurn() { return this.submitCommand('switch_turn') },
  toggleDebug() { this.setData({ showDebug: !this.data.showDebug }) },
  endSession() {
    if (this.data.status !== 'open' || !this.data.actorRole) return
    wx.showModal({ title: '确认结束本场？', content: '结束后本场不能继续控制。报告与片段需等待同步，请勿重复结束。', confirmText: '结束本场',
      success: result => { if (result.confirm) this.submitCommand('end_session') } })
  },
})
