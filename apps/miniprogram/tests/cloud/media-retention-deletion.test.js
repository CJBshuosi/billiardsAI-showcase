const {
  applyCloudDeletion,
  applyEdgeReceipt,
  buildDeletionJob,
  claimDeletionJob,
  executeClaimedCloudDeletion,
  executionAllowed,
  prepareDeletionJob,
} = require('../../cloudfunctions/media-retention/lib/deletion')


const CLIP = {
  clip_id: 'clip-synthetic-1',
  session_id: 'session-1',
  _device_id: 'device-1',
  source_kind: 'synthetic_retention_test',
  upload_status: 'published',
  byte_size: 12,
  content_sha256: 'c'.repeat(64),
  ordinary_expires_at: '2026-09-02T00:00:00.000Z',
  storage_objects: [
    { kind: 'published', file_id: 'cloud://published/clip-1.mp4' },
    { kind: 'staging', file_id: 'cloud://staging/clip-1.mp4' },
  ],
}

test('ordinary deletion cannot prepare, build or execute a saved-until-deleted clip', async () => {
  const clip = { ...CLIP, publication_kind: 'saved_postgame_clip',
    retention_policy: 'saved_until_deleted_v1', ordinary_expires_at: null,
    retention_expires_at: null, retention_state: 'active',
  }
  const now = '2046-09-03T00:00:00.000Z'
  expect(prepareDeletionJob({ clip, now })).toBeNull()
  expect(() => buildDeletionJob({ clip, now })).toThrow('clip retention authority is invalid')
  const job = claimDeletionJob(buildDeletionJob({ clip: CLIP, now }), {
    workerId: 'worker-1', now, leaseSeconds: 300,
  })
  const storage = { deleteObjects: jest.fn() }
  await expect(executeClaimedCloudDeletion({ job, clip, storage, now,
    env: { MEDIA_RETENTION_EXECUTION_ENABLED: 'synthetic-only' },
  })).rejects.toThrow('physical deletion execution is disabled')
  expect(storage.deleteObjects).not.toHaveBeenCalled()
})

test.each(['unknown', null, 'saved_until_deleted_v1'])(
  'ordinary deletion rejects explicit policy %p even with a cached due date', retentionPolicy => {
    const clip = { ...CLIP, retention_policy: retentionPolicy }
    const now = '2026-09-03T00:00:00.000Z'
    expect(() => prepareDeletionJob({ clip, now })).toThrow('clip retention authority is invalid')
    expect(() => buildDeletionJob({ clip, now })).toThrow('clip retention authority is invalid')
    expect(executionAllowed({ clip, env: { MEDIA_RETENTION_EXECUTION_ENABLED: 'synthetic-only' } }))
      .toBe(false)
  },
)


test('physical deletion is default-off and only permits explicitly synthetic clips', () => {
  expect(executionAllowed({ env: {}, clip: CLIP })).toBe(false)
  expect(executionAllowed({ env: { MEDIA_RETENTION_EXECUTION_ENABLED: 'true' }, clip: CLIP })).toBe(false)
  expect(executionAllowed({
    env: { MEDIA_RETENTION_EXECUTION_ENABLED: 'synthetic-only' },
    clip: { ...CLIP, source_kind: 'recorded_replay' },
  })).toBe(false)
  expect(executionAllowed({
    env: { MEDIA_RETENTION_EXECUTION_ENABLED: 'synthetic-only' }, clip: CLIP,
  })).toBe(true)
})


test('deletion job is deterministic, lease-safe and completes only after both layers confirm', () => {
  const job = buildDeletionJob({ clip: CLIP, now: '2026-09-03T00:00:00.000Z' })
  expect(job).toMatchObject({
    job_id: expect.stringMatching(/^delete-[a-f0-9]{32}$/),
    clip_id: 'clip-synthetic-1', device_id: 'device-1', status: 'pending', revision: 0,
    playback_blocked: true,
    cloud_objects: [
      { kind: 'published', file_id: 'cloud://published/clip-1.mp4' },
      { kind: 'staging', file_id: 'cloud://staging/clip-1.mp4' },
    ],
  })
  expect(buildDeletionJob({ clip: CLIP, now: 'later' }).job_id).toBe(job.job_id)

  const claim = claimDeletionJob(job, {
    workerId: 'worker-1', now: '2026-09-03T00:00:01.000Z', leaseSeconds: 300,
  })
  expect(claim.status).toBe('claimed')
  expect(claim.claim_token).toMatch(/^[a-f0-9]{64}$/)
  expect(() => claimDeletionJob(claim, {
    workerId: 'worker-2', now: '2026-09-03T00:01:00.000Z', leaseSeconds: 300,
  })).toThrow('deletion job is not claimable')

  const cloudDone = applyCloudDeletion(claim, {
    claimToken: claim.claim_token, now: '2026-09-03T00:01:30.000Z', deletedCount: 2,
  })
  expect(cloudDone).toMatchObject({ status: 'awaiting_edge', cloud_deleted: true })
  expect(() => applyEdgeReceipt(cloudDone, {
    claimToken: claim.claim_token,
    receipt: { clip_id: 'other', device_id: 'device-1', outcome: 'deleted' },
    now: '2026-09-03T00:02:00.000Z',
  })).toThrow('edge receipt does not match deletion job')

  const complete = applyEdgeReceipt(cloudDone, {
    claimToken: claim.claim_token,
    receipt: { clip_id: CLIP.clip_id, device_id: 'device-1', outcome: 'already_absent' },
    now: '2026-09-03T00:02:00.000Z',
  })
  expect(complete).toMatchObject({ status: 'completed', edge_deleted: true,
    completed_at: '2026-09-03T00:02:00.000Z' })
  expect(applyEdgeReceipt(complete, {
    claimToken: claim.claim_token,
    receipt: { clip_id: CLIP.clip_id, device_id: 'device-1', outcome: 'already_absent' },
    now: 'later',
  })).toEqual(complete)
  expect(() => applyEdgeReceipt({ ...cloudDone, cloud_deleted: false }, {
    claimToken: claim.claim_token,
    receipt: { clip_id: CLIP.clip_id, device_id: 'device-1', outcome: 'deleted' },
    now: 'later',
  })).toThrow('edge receipt does not match deletion job')
})


test('invalid storage lineage fails closed before a deletion job exists', () => {
  expect(() => buildDeletionJob({
    clip: { ...CLIP, storage_objects: [{ kind: 'published', file_id: '/tmp/video.mp4' }] },
    now: '2026-09-03T00:00:00.000Z',
  })).toThrow('clip storage lineage is invalid')
  expect(() => buildDeletionJob({
    clip: { ...CLIP, storage_objects: [CLIP.storage_objects[0], CLIP.storage_objects[0]] },
    now: '2026-09-03T00:00:00.000Z',
  })).toThrow('clip storage lineage is invalid')
})

test('cloud executor refuses production and advances only verified synthetic deletion', async () => {
  const pending = buildDeletionJob({ clip: CLIP, now: '2026-09-03T00:00:00.000Z' })
  const job = claimDeletionJob(pending, {
    workerId: 'worker-1', now: '2026-09-03T00:00:01.000Z', leaseSeconds: 300,
  })
  const storage = { deleteObjects: jest.fn().mockResolvedValue({ deleted: true, count: 2 }) }
  await expect(executeClaimedCloudDeletion({ job, clip: CLIP, env: {}, storage,
    now: '2026-09-03T00:01:00.000Z' })).rejects.toThrow('physical deletion execution is disabled')
  expect(storage.deleteObjects).not.toHaveBeenCalled()

  await expect(executeClaimedCloudDeletion({
    job, clip: CLIP, env: { MEDIA_RETENTION_EXECUTION_ENABLED: 'synthetic-only' }, storage,
    now: '2026-09-03T00:01:00.000Z',
  })).resolves.toMatchObject({ status: 'awaiting_edge', cloud_deleted: true })
  expect(storage.deleteObjects).toHaveBeenCalledWith([
    'cloud://published/clip-1.mp4', 'cloud://staging/clip-1.mp4',
  ])
})

test('preparation recomputes all holds and never trusts the cached deadline alone', () => {
  const authority = {
    clip: CLIP,
    favorites: [{ clip_id: CLIP.clip_id, active: true, expires_at: '2026-10-01T00:00:00.000Z' }],
    evidenceHolds: [], pendingJobs: [],
  }
  expect(prepareDeletionJob({ ...authority, now: '2026-09-03T00:00:00.000Z' })).toBeNull()
  expect(prepareDeletionJob({ ...authority, favorites: [],
    now: '2026-09-03T00:00:00.000Z' })).toMatchObject({ clip_id: CLIP.clip_id })
  expect(prepareDeletionJob({ ...authority, favorites: [],
    pendingJobs: [{ status: 'uploading', clip_ids: [CLIP.clip_id] }],
    now: '2026-09-03T00:00:00.000Z' })).toBeNull()
})
