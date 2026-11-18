const { buildRetentionInventory } = require('../../cloudfunctions/media-retention/lib/inventory')


const NOW = '2026-09-03T00:00:00.000Z'


function clip(id, expiresAt, byteSize = 100) {
  return {
    clip_id: id,
    ordinary_expires_at: expiresAt,
    byte_size: byteSize,
    storage_file_id: `cloud://private/${id}.mp4`,
    upload_status: 'published',
    deleted_at: null,
  }
}


test('inventory lists only clips whose complete retention scope is due', () => {
  const result = buildRetentionInventory({
    clips: [
      clip('ordinary-due', '2026-09-02T00:00:00.000Z', 100),
      clip('favorite-held', '2026-09-02T00:00:00.000Z', 200),
      clip('shared-held', '2026-09-02T00:00:00.000Z', 300),
      clip('evidence-held', '2026-09-02T00:00:00.000Z', 400),
      clip('pending-held', '2026-09-02T00:00:00.000Z', 500),
    ],
    favorites: [
      { clip_id: 'favorite-held', active: true, expires_at: '2026-10-01T00:00:00.000Z' },
      { clip_id: 'shared-held', active: false, grace_expires_at: '2026-09-04T00:00:00.000Z' },
      { clip_id: 'shared-held', active: true, expires_at: '2026-10-02T00:00:00.000Z' },
    ],
    evidenceHolds: [
      { clip_id: 'evidence-held', active: true, expires_at: '2026-11-01T00:00:00.000Z' },
    ],
    pendingJobs: [{ clip_ids: ['pending-held'], status: 'uploading' }],
    now: NOW,
  })
  expect(result.due).toEqual({ count: 1, byte_size: 100, clips: [{
    clip_id: 'ordinary-due', byte_size: 100, effective_expires_at: '2026-09-02T00:00:00.000Z',
  }] })
  expect(result.held).toEqual({ count: 4, byte_size: 1400 })
  expect(JSON.stringify(result)).not.toContain('cloud://')
})

test('invalid or incomplete authority is excluded rather than guessed due', () => {
  const result = buildRetentionInventory({
    clips: [clip('bad-date', 'yesterday'), { ...clip('bad-size', '2026-09-02T00:00:00.000Z'), byte_size: -1 }],
    favorites: [], evidenceHolds: [], pendingJobs: [], now: NOW,
  })
  expect(result.due.count).toBe(0)
  expect(result.invalid).toEqual({ count: 2, clip_ids: ['bad-date', 'bad-size'] })
})

test('active saved-until-deleted clips are held indefinitely without an expiry sentinel', () => {
  const result = buildRetentionInventory({ clips: [{
    ...clip('saved', null, 234), publication_kind: 'saved_postgame_clip',
    retention_policy: 'saved_until_deleted_v1', retention_expires_at: null,
    retention_state: 'active',
  }], now: '2046-09-03T00:00:00.000Z' })
  expect(result).toEqual({
    due: { count: 0, byte_size: 0, clips: [] },
    held: { count: 1, byte_size: 234 }, invalid: { count: 0, clip_ids: [] },
  })
})

test.each([
  { retention_policy: 'unknown' }, { retention_policy: null },
  { ordinary_expires_at: '2026-09-02T00:00:00.000Z' },
  { retention_expires_at: '2026-09-02T00:00:00.000Z' },
  { publication_kind: 'ordinary' }, { retention_state: 'released' },
])('malformed indefinite policy is invalid and never becomes due: %p', change => {
  const result = buildRetentionInventory({ clips: [{
    ...clip('saved', null), publication_kind: 'saved_postgame_clip',
    retention_policy: 'saved_until_deleted_v1', retention_expires_at: null,
    retention_state: 'active', ...change,
  }], now: NOW })
  expect(result.due.count).toBe(0)
  expect(result.invalid).toEqual({ count: 1, clip_ids: ['saved'] })
})

test.each(['unknown', null, 'saved_until_deleted_v1'])(
  'explicit policy %p cannot use a finite ordinary deadline as deletion authority', retentionPolicy => {
    const result = buildRetentionInventory({ clips: [{
      ...clip('mixed', '2026-09-02T00:00:00.000Z'), retention_policy: retentionPolicy,
      retention_expires_at: '2026-09-02T00:00:00.000Z',
    }], now: NOW })
    expect(result.due.count).toBe(0)
    expect(result.invalid).toEqual({ count: 1, clip_ids: ['mixed'] })
  },
)
