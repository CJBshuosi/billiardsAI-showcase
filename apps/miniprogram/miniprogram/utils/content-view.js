const { errorMessage } = require('./ui')

function contentError(code) {
  if (code === 'FORBIDDEN') return '当前账号无法查看这份内容。'
  if (code === 'MEDIA_NOT_AVAILABLE') return '这段视频暂不可用，请稍后重新查看。'
  if (code === 'SESSION_NOT_JOINABLE') return '暂未取得这场记录，请稍后重新查看。'
  if (code === 'NETWORK_UNAVAILABLE') return '连接暂时出了问题，请检查网络后重试。'
  return errorMessage(code)
}
function dateLabel(value) {
  if (!value) return '场次时间未记录'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '场次时间未记录'
  const pad = number => String(number).padStart(2, '0')
  return `${date.getFullYear()}.${pad(date.getMonth() + 1)}.${pad(date.getDate())} · ${pad(date.getHours())}:${pad(date.getMinutes())}`
}
function analysisView(analysis) {
  if (!analysis) return null
  const labels = {
    PROCESSING: '正在整理本场回放与逐杆分析',
    AVAILABLE: '本场回放与逐杆分析已可查看',
    PARTIAL: '部分逐杆分析已可查看，其余仍在重试',
    FAILED_RETRYING: '本场分析暂未完成，系统正在重试',
  }
  return {
    ...analysis,
    label: labels[analysis.status] || '本场分析状态待确认',
    hasLiveCount: Number.isInteger(analysis.live_shots_recorded)
      && analysis.live_shots_recorded >= 0,
    hasCounts: Number.isInteger(analysis.total_shots)
      && Number.isInteger(analysis.available_shots),
  }
}
function sessionView(session) {
  return {
    ...session,
    dateLabel: dateLabel(session.created_at_utc || session.ended_at_utc),
    statusLabel: ({ ended: '已结束', active: '进行中', playing: '进行中', waiting: '等待开局' })[session.status] || '状态待确认',
    participantsLabel: Array.isArray(session.participants) ? session.participants.map(p => ({ red: '红方参与者', blue: '蓝方参与者' })[p.role] || '本场参与者').join(' · ') : '参与者信息暂未取得',
    analysisView: analysisView(session.analysis),
  }
}
function clipView(clip) {
  const sequence = Number.isInteger(clip.sequence) && clip.sequence > 0 ? clip.sequence : null
  const seconds = Number.isFinite(clip.duration_ms) && clip.duration_ms >= 0 ? Math.round(clip.duration_ms / 1000) : null
  const role = ({ red: '红方', blue: '蓝方' })[clip.actor_role]
  return {
    ...clip,
    title: sequence === null ? '击球片段 · 序号待确认' : `${role ? role + ' · ' : ''}第 ${sequence} 杆`,
    durationLabel: seconds === null ? '' : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`,
    stateLabel: clip.upload_status === 'published' ? '可查看' : '暂不可用',
    isReplay: clip.source_kind === 'development_replay',
    retentionLabel: clip.retention_policy === 'saved_until_deleted_v1'
      && clip.available_until === null ? '已保存，保留至删除'
      : clip.available_until ? `可查看至 ${dateLabel(clip.available_until)}` : '',
  }
}
function orderedClips(clips) {
  return clips.map(clipView).sort((a, b) => {
    const order = value => Number.isInteger(value) && value > 0 ? value : Infinity
    return order(a.sequence) - order(b.sequence)
  })
}
module.exports = { analysisView, contentError, dateLabel, sessionView, clipView, orderedClips }
