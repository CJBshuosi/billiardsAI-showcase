const {
  claimNextEdgeDeletion,
  completeEdgeDeletion,
  createCloudMediaDeletionRepository,
} = require('../../cloudfunctions/sync-api/lib/media-retention')


test('device claim exposes deletion identity but never cloud locators', async () => {
  const repo = { claimNext: jest.fn().mockResolvedValue({
    job_id: 'delete-1', clip_id: 'clip-1', device_id: 'device-1', table_id: 'table-1',
    content_sha256: 'a'.repeat(64), claim_token: 'b'.repeat(64), status: 'awaiting_edge',
    cloud_objects: [{ file_id: 'cloud://secret', kind: 'published' }],
  }) }
  const result = await claimNextEdgeDeletion({
    deviceId: 'device-1', tableId: 'table-1', now: 'now', repo,
  })
  expect(result).toEqual({
    schema_version: '1.0', job_id: 'delete-1', clip_id: 'clip-1',
    content_sha256: 'a'.repeat(64), claim_token: 'b'.repeat(64),
  })
  expect(JSON.stringify(result)).not.toContain('cloud://')
})


test('edge completion accepts only matching terminal evidence', async () => {
  const repo = { complete: jest.fn().mockResolvedValue({ status: 'completed' }) }
  const input = {
    jobId: 'delete-1', deviceId: 'device-1', tableId: 'table-1',
    claimToken: 'b'.repeat(64), clipId: 'clip-1', outcome: 'already_absent', now: 'now', repo,
  }
  await expect(completeEdgeDeletion(input)).resolves.toEqual({ status: 'completed' })
  const { repo: _repo, ...expected } = input
  expect(repo.complete).toHaveBeenCalledWith(expected)
  await expect(completeEdgeDeletion({ ...input, outcome: 'failed' }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
  await expect(completeEdgeDeletion({ ...input, claimToken: 'short' }))
    .rejects.toMatchObject({ code: 'SYNC_INVALID' })
})


test('cloud repository tombstones locators only after matching edge receipt', async () => {
  const jobs = new Map([['delete-1', {
    job_id: 'delete-1', clip_id: 'clip-1', device_id: 'device-1', table_id: 'table-1',
    content_sha256: 'a'.repeat(64), claim_token: 'b'.repeat(64),
    status: 'awaiting_edge', cloud_deleted: true, created_at: 'before',
  }]])
  const clips = new Map([['clip-1', {
    clip_id: 'clip-1', _device_id: 'device-1', table_id: 'table-1',
    storage_file_id: 'cloud://private', storage_objects: [{ file_id: 'cloud://private' }],
  }]])
  const maps = { media_deletion_jobs: jobs, clips }
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
  const db = { collection, async runTransaction(callback) { return callback({ collection }) } }
  const repo = createCloudMediaDeletionRepository(db)
  await expect(repo.claimNext({ deviceId: 'device-1', tableId: 'table-1' }))
    .resolves.toMatchObject({ job_id: 'delete-1' })
  await expect(repo.complete({
    jobId: 'delete-1', deviceId: 'device-1', tableId: 'table-1',
    claimToken: 'b'.repeat(64), clipId: 'clip-1', outcome: 'deleted', now: 'now',
  })).resolves.toMatchObject({ status: 'completed' })
  expect(clips.get('clip-1')).toMatchObject({
    deleted_at: 'now', storage_file_id: null, storage_objects: [], retention_state: 'released',
  })
})
