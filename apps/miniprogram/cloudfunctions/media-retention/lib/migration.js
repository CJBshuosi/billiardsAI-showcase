const crypto = require('crypto')

const { ordinaryExpiry, timestamp } = require('../../shared/retention-policy')


function opaqueOwnerId(userId) {
  return crypto.createHash('sha256').update(String(userId), 'utf8').digest('hex').slice(0, 16)
}


function storageObjectCount(clip) {
  if (Array.isArray(clip.storage_objects) && clip.storage_objects.length) {
    const valid = clip.storage_objects.filter(item => item && typeof item.file_id === 'string' && item.file_id)
    return valid.length === clip.storage_objects.length ? valid.length : 0
  }
  return typeof clip.storage_file_id === 'string' && clip.storage_file_id ? 1 : 0
}


function buildLegacyMigrationPreview({ sessions = [], clips = [], favorites = [] }) {
  const authority = new Map()
  const conflictingSessions = new Set()
  for (const session of sessions) {
    if (!session || typeof session.session_id !== 'string') continue
    const current = authority.get(session.session_id)
    const candidate = session.status === 'ended' ? session.ended_at_utc : null
    if (current !== undefined && current !== candidate) conflictingSessions.add(session.session_id)
    else authority.set(session.session_id, candidate)
  }

  const blocked = []
  const clipPatches = []
  for (const clip of clips) {
    if (!clip || typeof clip.clip_id !== 'string') continue
    const count = storageObjectCount(clip)
    if (count === 0) {
      blocked.push({ kind: 'clip', record_id: clip.clip_id, reason: 'MISSING_STORAGE_LINEAGE' })
      continue
    }
    let ordinary = clip.ordinary_expires_at
    if (!ordinary) {
      const endedAt = authority.get(clip.session_id)
      if (!endedAt || conflictingSessions.has(clip.session_id)) {
        blocked.push({ kind: 'clip', record_id: clip.clip_id, reason: 'MISSING_AUTHORITATIVE_END' })
        continue
      }
      try { ordinary = ordinaryExpiry(endedAt) } catch (_error) {
        blocked.push({ kind: 'clip', record_id: clip.clip_id, reason: 'MISSING_AUTHORITATIVE_END' })
        continue
      }
    }
    try { timestamp(ordinary) } catch (_error) {
      blocked.push({ kind: 'clip', record_id: clip.clip_id, reason: 'INVALID_RETENTION_DEADLINE' })
      continue
    }
    clipPatches.push({
      clip_id: clip.clip_id,
      ordinary_expires_at: ordinary,
      retention_expires_at: clip.retention_expires_at || ordinary,
      retention_state: clip.retention_state || 'active',
      storage_object_count: count,
    })
  }

  const byOwner = new Map()
  for (const favorite of favorites) {
    if (!favorite || favorite.active !== true || typeof favorite.user_id !== 'string'
        || typeof favorite.favorite_id !== 'string' || Number.isInteger(favorite.slot)) continue
    if (!byOwner.has(favorite.user_id)) byOwner.set(favorite.user_id, [])
    byOwner.get(favorite.user_id).push(favorite)
  }
  const favoriteSlotPatches = []
  for (const [userId, rows] of byOwner.entries()) {
    rows.sort((a, b) => a.favorite_id.localeCompare(b.favorite_id))
    if (rows.length > 20) {
      blocked.push({
        kind: 'favorite-owner', record_id: opaqueOwnerId(userId), reason: 'FAVORITE_LIMIT_EXCEEDED',
      })
      continue
    }
    rows.forEach((row, slot) => favoriteSlotPatches.push({ favorite_id: row.favorite_id, slot }))
  }
  for (const sessionId of [...conflictingSessions].sort()) {
    blocked.push({ kind: 'session', record_id: sessionId, reason: 'CONFLICTING_SESSION_AUTHORITY' })
  }
  clipPatches.sort((a, b) => a.clip_id.localeCompare(b.clip_id))
  favoriteSlotPatches.sort((a, b) => a.favorite_id.localeCompare(b.favorite_id))
  blocked.sort((a, b) => `${a.kind}:${a.record_id}`.localeCompare(`${b.kind}:${b.record_id}`))
  return {
    summary: {
      clips_scanned: clips.length,
      clip_patches: clipPatches.length,
      favorites_scanned: favorites.length,
      favorite_slot_patches: favoriteSlotPatches.length,
      blocked: blocked.length,
    },
    clip_patches: clipPatches,
    favorite_slot_patches: favoriteSlotPatches,
    blocked,
  }
}


module.exports = { buildLegacyMigrationPreview }
