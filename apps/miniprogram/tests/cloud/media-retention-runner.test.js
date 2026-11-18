const { runSyntheticRetentionBatch } = require('../../cloudfunctions/media-retention/lib/runner')


function authority(id, { held = false } = {}) {
  return {
    clip: {
      clip_id: id, session_id: 'session-1', table_id: 'table-1', _device_id: 'device-1',
      source_kind: 'synthetic_retention_test', upload_status: 'published', deleted_at: null,
      byte_size: 4, content_sha256: id === 'clip-due' ? 'a'.repeat(64) : 'b'.repeat(64),
      ordinary_expires_at: '2026-09-01T00:00:00.000Z',
      storage_objects: [{ kind: 'published', file_id: `cloud://published/${id}.mp4` }],
    },
    favorites: held ? [{ clip_id: id, active: true, expires_at: '2026-10-01T00:00:00.000Z' }] : [],
    evidenceHolds: [], pendingJobs: [],
  }
}


test('synthetic batch is default-off and carries one due job to awaiting edge', async () => {
  const jobs = new Map()
  const repo = {
    listCandidateAuthorities: jest.fn().mockResolvedValue([
      authority('clip-due'), authority('clip-held', { held: true }),
    ]),
    async createPending(job) { jobs.set(job.job_id, job); return job },
    async claim(job) {
      const { claimDeletionJob } = require('../../cloudfunctions/media-retention/lib/deletion')
      const claimed = claimDeletionJob(job, {
        workerId: 'worker-1', now: '2026-09-03T00:00:00.000Z', leaseSeconds: 300,
      })
      jobs.set(job.job_id, claimed)
      return claimed
    },
    async saveCloudDeletion(job) { jobs.set(job.job_id, job); return job },
    markFailed: jest.fn(),
  }
  const storage = { deleteObjects: jest.fn().mockResolvedValue({ deleted: true, count: 1 }) }
  await expect(runSyntheticRetentionBatch({ repo, storage, env: {}, workerId: 'worker-1',
    now: '2026-09-03T00:00:00.000Z' })).resolves.toEqual({
    enabled: false, scanned: 0, awaiting_edge: 0, held: 0, failed: 0,
  })
  expect(repo.listCandidateAuthorities).not.toHaveBeenCalled()

  await expect(runSyntheticRetentionBatch({ repo, storage,
    env: { MEDIA_RETENTION_EXECUTION_ENABLED: 'synthetic-only' }, workerId: 'worker-1',
    now: '2026-09-03T00:00:00.000Z' })).resolves.toEqual({
    enabled: true, scanned: 2, awaiting_edge: 1, held: 1, failed: 0,
  })
  expect([...jobs.values()][0]).toMatchObject({ status: 'awaiting_edge', cloud_deleted: true })
  expect(storage.deleteObjects).toHaveBeenCalledWith(['cloud://published/clip-due.mp4'])
})
