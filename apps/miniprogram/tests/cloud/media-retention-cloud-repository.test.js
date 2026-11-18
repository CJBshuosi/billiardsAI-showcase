const { createCloudRetentionRepository } = require('../../cloudfunctions/media-retention/lib/cloud-repository')


function fakeDatabase() {
  const maps = {
    clips: new Map([['clip-1', {
      clip_id: 'clip-1', session_id: 'session-1', table_id: 'table-1', _device_id: 'device-1',
      source_kind: 'synthetic_retention_test', upload_status: 'published', retention_state: 'active',
      deleted_at: null, byte_size: 4, content_sha256: 'a'.repeat(64),
      ordinary_expires_at: '2026-09-01T00:00:00.000Z',
      storage_objects: [{ kind: 'published', file_id: 'cloud://published/clip-1.mp4' }],
    }]]),
    clip_favorites: new Map(), clip_evidence_holds: new Map(),
    media_deletion_jobs: new Map(),
  }
  const collection = name => ({
    doc(id) { return {
      async get() { return { data: maps[name].get(id) || null } },
      async set({ data }) { maps[name].set(id, data) },
      async update({ data }) { maps[name].set(id, { ...maps[name].get(id), ...data }) },
    } },
    where(query) { return {
      orderBy() { return this }, limit() { return this },
      async get() { return { data: [...maps[name].values()].filter(row =>
        Object.entries(query).every(([key, value]) => row[key] === value)) } },
    } },
  })
  return {
    maps, collection,
    async runTransaction(callback) { return callback({ collection }) },
  }
}


test('cloud retention repository atomically blocks playback and persists cloud evidence', async () => {
  const db = fakeDatabase()
  const repo = createCloudRetentionRepository(db)
  const [authority] = await repo.listCandidateAuthorities({
    now: '2026-09-03T00:00:00.000Z', limit: 25,
  })
  expect(authority.clip.clip_id).toBe('clip-1')
  const { prepareDeletionJob } = require('../../cloudfunctions/media-retention/lib/deletion')
  const pending = prepareDeletionJob({ ...authority, now: '2026-09-03T00:00:00.000Z' })
  const created = await repo.createPending(pending, authority)
  expect(db.maps.clips.get('clip-1').retention_state).toBe('deletion_pending')
  const claimed = await repo.claim(created, {
    workerId: 'worker-1', now: '2026-09-03T00:00:00.000Z', leaseSeconds: 300,
  })
  const awaiting = { ...claimed, status: 'awaiting_edge', cloud_deleted: true }
  await repo.saveCloudDeletion(awaiting)
  expect(db.maps.media_deletion_jobs.get(created.job_id)).toMatchObject({
    status: 'awaiting_edge', cloud_deleted: true,
  })
})


test('cloud retention repository rechecks a late favorite before creating the job', async () => {
  const db = fakeDatabase()
  const repo = createCloudRetentionRepository(db)
  const [authority] = await repo.listCandidateAuthorities({ now: '2026-09-03T00:00:00.000Z', limit: 25 })
  db.maps.clip_favorites.set('late', {
    clip_id: 'clip-1', active: true, expires_at: '2026-10-01T00:00:00.000Z',
  })
  const { prepareDeletionJob } = require('../../cloudfunctions/media-retention/lib/deletion')
  const pending = prepareDeletionJob({ ...authority, now: '2026-09-03T00:00:00.000Z' })
  await expect(repo.createPending(pending, authority)).rejects.toThrow('retention authority changed')
  expect(db.maps.clips.get('clip-1').retention_state).toBe('active')
  await expect(repo.markFailed({ job_id: 'never-created' })).resolves.toBeUndefined()
})
