const { timestamp, isSavedUntilDeletedClip } = require('../../shared/retention-policy')


function buildRetentionInventory({ clips, favorites, evidenceHolds, pendingJobs, now }) {
  const nowMs = timestamp(now)
  const favoriteByClip = new Map()
  for (const hold of favorites || []) {
    if (!favoriteByClip.has(hold.clip_id)) favoriteByClip.set(hold.clip_id, [])
    favoriteByClip.get(hold.clip_id).push(hold)
  }
  const evidenceByClip = new Map()
  for (const hold of evidenceHolds || []) {
    if (!evidenceByClip.has(hold.clip_id)) evidenceByClip.set(hold.clip_id, [])
    evidenceByClip.get(hold.clip_id).push(hold)
  }
  const pending = new Set((pendingJobs || [])
    .filter(job => !['completed', 'failed', 'cancelled'].includes(job.status))
    .flatMap(job => Array.isArray(job.clip_ids) ? job.clip_ids : []))

  const dueClips = []
  const invalidIds = []
  let heldCount = 0
  let heldBytes = 0
  for (const clip of clips || []) {
    try {
      if (!clip || typeof clip.clip_id !== 'string' || !clip.clip_id
          || !Number.isInteger(clip.byte_size) || clip.byte_size < 0
          || clip.deleted_at || clip.upload_status !== 'published') {
        throw new Error('invalid clip')
      }
      if (Object.hasOwn(clip, 'retention_policy')) {
        if (!isSavedUntilDeletedClip(clip)) throw new Error('invalid retention policy')
        heldCount += 1
        heldBytes += clip.byte_size
        continue
      }
      let effectiveMs = timestamp(clip.ordinary_expires_at)
      let effectiveExpiresAt = clip.ordinary_expires_at
      for (const hold of favoriteByClip.get(clip.clip_id) || []) {
        const expiry = hold.active === true ? hold.expires_at
          : hold.active === false ? hold.grace_expires_at : null
        if (!expiry) continue
        const expiryMs = timestamp(expiry)
        if (expiryMs > effectiveMs) {
          effectiveMs = expiryMs
          effectiveExpiresAt = expiry
        }
      }
      for (const hold of evidenceByClip.get(clip.clip_id) || []) {
        if (hold.active !== true) continue
        const expiryMs = timestamp(hold.expires_at)
        if (expiryMs > effectiveMs) {
          effectiveMs = expiryMs
          effectiveExpiresAt = hold.expires_at
        }
      }
      if (pending.has(clip.clip_id) || effectiveMs > nowMs) {
        heldCount += 1
        heldBytes += clip.byte_size
      } else {
        dueClips.push({
          clip_id: clip.clip_id,
          byte_size: clip.byte_size,
          effective_expires_at: effectiveExpiresAt,
        })
      }
    } catch (_error) {
      if (clip && typeof clip.clip_id === 'string') invalidIds.push(clip.clip_id)
    }
  }
  dueClips.sort((a, b) => a.clip_id.localeCompare(b.clip_id))
  invalidIds.sort()
  return {
    due: {
      count: dueClips.length,
      byte_size: dueClips.reduce((total, clip) => total + clip.byte_size, 0),
      clips: dueClips,
    },
    held: { count: heldCount, byte_size: heldBytes },
    invalid: { count: invalidIds.length, clip_ids: invalidIds },
  }
}


module.exports = { buildRetentionInventory }
