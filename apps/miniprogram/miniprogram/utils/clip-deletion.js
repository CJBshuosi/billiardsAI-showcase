const intentKey = clipId => `clip-deletion:${clipId}`
const requestIdPattern = /^[a-f0-9]{32}$/

// Local intent provides retry continuity only. The server authenticates the owner.
function loadDeletionIntent(clipId) {
  const value = wx.getStorageSync(intentKey(clipId))
  if (value === undefined || value === null || value === '') return null
  if (!value || value.clip_id !== clipId || typeof value.request_id !== 'string' || !requestIdPattern.test(value.request_id)) {
    throw new Error('Invalid deletion intent')
  }
  return { clip_id: clipId, request_id: value.request_id }
}
async function ensureDeletionIntent(clipId) {
  const pending = loadDeletionIntent(clipId)
  if (pending) return pending
  const bytes = await new Promise((resolve, reject) => {
    wx.getRandomValues({ length: 16,
      success: ({ randomValues }) => resolve(new Uint8Array(randomValues)), fail: reject })
  })
  if (bytes.length !== 16) throw new Error('Invalid deletion intent entropy')
  const intent = { clip_id: clipId,
    request_id: Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('') }
  wx.setStorageSync(intentKey(clipId), intent)
  return intent
}
function clearDeletionIntent(clipId) {
  try { wx.removeStorageSync(intentKey(clipId)) } catch (_) { /* Acceptance remains authoritative. */ }
}
function isDeletionReceipt(value, clipId) {
  return !!value && Object.keys(value).length === 6
    && value.schema_version === '1.0' && value.clip_id === clipId
    && value.status === 'deletion_requested'
    && typeof value.request_id === 'string' && /^[A-Za-z0-9._-]{1,128}$/.test(value.request_id)
    && typeof value.deletion_job_id === 'string' && value.deletion_job_id.length > 0
    && typeof value.requested_at === 'string' && Number.isFinite(Date.parse(value.requested_at))
}
module.exports = { loadDeletionIntent, ensureDeletionIntent, clearDeletionIntent, isDeletionReceipt }
