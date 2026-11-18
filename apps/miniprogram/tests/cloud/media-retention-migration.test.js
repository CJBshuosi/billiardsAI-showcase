const { buildLegacyMigrationPreview } = require('../../cloudfunctions/media-retention/lib/migration')


test('legacy migration preview assigns deterministic favorite slots and clip expiry patches', () => {
  const preview = buildLegacyMigrationPreview({
    sessions: [{ session_id: 'session-1', status: 'ended', ended_at_utc: '2026-09-01T00:00:00.000Z' }],
    clips: [{
      clip_id: 'clip-1', session_id: 'session-1', upload_status: 'published',
      storage_file_id: 'cloud://private/clip-1.mp4', byte_size: 12,
    }],
    favorites: [
      { favorite_id: 'fav-b', user_id: 'user-1', clip_id: 'clip-1', active: true },
      { favorite_id: 'fav-a', user_id: 'user-1', clip_id: 'clip-2', active: true },
    ],
  })

  expect(preview.summary).toEqual({
    clips_scanned: 1, clip_patches: 1, favorites_scanned: 2,
    favorite_slot_patches: 2, blocked: 0,
  })
  expect(preview.clip_patches).toEqual([{
    clip_id: 'clip-1', ordinary_expires_at: '2026-09-03T00:00:00.000Z',
    retention_expires_at: '2026-09-03T00:00:00.000Z', retention_state: 'active',
    storage_object_count: 1,
  }])
  expect(preview.favorite_slot_patches).toEqual([
    { favorite_id: 'fav-a', slot: 0 }, { favorite_id: 'fav-b', slot: 1 },
  ])
  expect(JSON.stringify(preview)).not.toContain('user-1')
  expect(JSON.stringify(preview)).not.toContain('cloud://')
})

test('legacy migration preview fails closed on missing authority and conflicts', () => {
  const favorites = Array.from({ length: 21 }, (_, index) => ({
    favorite_id: `fav-${String(index).padStart(2, '0')}`,
    user_id: 'over-limit-user',
    clip_id: `clip-${index}`,
    active: true,
  }))
  const preview = buildLegacyMigrationPreview({
    sessions: [
      { session_id: 'duplicate', status: 'ended', ended_at_utc: '2026-09-01T00:00:00.000Z' },
      { session_id: 'duplicate', status: 'ended', ended_at_utc: '2026-09-02T00:00:00.000Z' },
    ],
    clips: [
      { clip_id: 'missing-session', session_id: 'unknown', upload_status: 'published', storage_file_id: 'cloud://one', byte_size: 1 },
      { clip_id: 'missing-storage', session_id: 'duplicate', upload_status: 'published', byte_size: 1 },
    ],
    favorites,
  })

  expect(preview.summary.blocked).toBe(4)
  expect(preview.blocked).toEqual([
    { kind: 'clip', record_id: 'missing-session', reason: 'MISSING_AUTHORITATIVE_END' },
    { kind: 'clip', record_id: 'missing-storage', reason: 'MISSING_STORAGE_LINEAGE' },
    { kind: 'favorite-owner', record_id: expect.stringMatching(/^[a-f0-9]{16}$/), reason: 'FAVORITE_LIMIT_EXCEEDED' },
    { kind: 'session', record_id: 'duplicate', reason: 'CONFLICTING_SESSION_AUTHORITY' },
  ])
  expect(preview.favorite_slot_patches).toEqual([])
  expect(JSON.stringify(preview)).not.toContain('over-limit-user')
})
