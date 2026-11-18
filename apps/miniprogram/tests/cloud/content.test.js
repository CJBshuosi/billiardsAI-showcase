const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { createCloudContentRepository } = require('../../cloudfunctions/client-api/lib/content')
const { privateClip } = require('../helpers/private-clips')
const { ownershipProjection } = require('../../cloudfunctions/shared/shot-ownership')
const {
  getClipPlayback,
  getSessionAnalysis,
  getSession,
  listClips,
  listSessions,
} = require('../../cloudfunctions/client-api/lib/content')


function createRepository() {
  const participants = new Set(['session-1:user-a', 'session-1:user-b'])
  const sessions = new Map([['session-1', {
    schema_version: '1.0',
    session_id: 'session-1',
    status: 'ended',
    sync_status: 'published',
    limitations: ['no_global_fixed_ball_id'],
    shots: [{ shot_id: 'shot-1', sequence: 1 }],
  }]])
  const clips = new Map([['clip-1', privateClip({ clip_id: 'clip-1', shot_id: 'shot-1' })]])
  const projection = ownershipProjection(clips.get('clip-1'), 'table-1', 'device-1')
  const analysis = new Map([['session-1', {
    schema_version: '1.0', session_id: 'session-1', revision: 1,
    status: 'PROCESSING', total_shots: 1, available_shots: 0, retrying_shots: 0,
    report_available: false, deadline_at: '2026-08-26T14:07:00.000Z',
    updated_at: '2026-08-26T14:05:00.000Z', _device_id: 'private-device',
  }]])
  const confirmedBindings = new Set()
  const liveProgress = new Map([['session-1', {
    schema_version: '1.0', session_id: 'session-1', shot_count: 3,
    last_sequence: 3, last_shot_id: 'shot-3', updated_at: '2026-08-26T14:05:30.000Z',
  }]])
  return {
    async hasParticipant(sessionId, userId) {
      return participants.has(`${sessionId}:${userId}`)
    },
    async getSession(sessionId) {
      return sessions.get(sessionId) || null
    },
    async getSessionAnalysis(sessionId) {
      return analysis.get(sessionId) || null
    },
    async getLiveShotProgress(sessionId) {
      return liveProgress.get(sessionId) || null
    },
    async hasConfirmedSessionBinding(sessionId, userId) {
      return confirmedBindings.has(`${sessionId}:${userId}`)
    },
    async getEndedFieldSession(sessionId) {
      return sessionId === 'session-1' ? {
        session_id: 'session-1', status: 'ended',
        ended_at: '2026-08-26T14:05:00.000Z', updated_at: '2026-08-26T14:05:01.000Z',
      } : null
    },
    async listSessions(userId) {
      return [...sessions.values()].filter((session) => participants.has(`${session.session_id}:${userId}`))
    },
    async getOwnerClipTombstone() { return null },
    async authorizePlaybackForOwner(args) {
      const clip = clips.get(args.clipId)
      const db = ownerDeleteDb({ clips: clip ? { [args.clipId]: clip } : {}, field_shot_ownership: {
        [require('../../cloudfunctions/shared/shot-ownership').documentId('session-1:shot-1')]: projection,
      } })
      return createCloudContentRepository(db, {}).authorizePlaybackForOwner(args)
    },
    async getShotOwnership() { return projection },
    async getClip(clipId) {
      return clips.get(clipId) || null
    },
    async listClips(sessionId) {
      return [...clips.values()]
        .filter((clip) => !sessionId || clip.session_id === sessionId)
        .map((clip) => ({ ...clip }))
    },
    async createPlaybackUrl(fileId) {
      return { url: `https://temporary.invalid/${encodeURIComponent(fileId)}`, expires_in_seconds: 300 }
    },
    seedAnalysis(value) { analysis.set(value.session_id, value) },
    confirmBinding(sessionId, userId) { confirmedBindings.add(`${sessionId}:${userId}`) },
  }
}

async function permanentRepository(change = {}) {
  const repo = createRepository()
  Object.assign(await repo.getClip('clip-1'), {
    retention_policy: 'saved_until_deleted_v1', ordinary_expires_at: null,
    retention_expires_at: null, ...change,
  })
  return repo
}

test.each(['user-a'])('saved-until-deleted media remains listed and playable for %s years later', async userId => {
  const repo = await permanentRepository()
  const now = '2046-09-01T00:00:00.000Z'
  const listed = await listClips({ sessionId: 'session-1', userId, repo, now })
  expect(listed).toEqual([expect.objectContaining({
    clip_id: 'clip-1', retention_policy: 'saved_until_deleted_v1', available_until: null,
  })])
  expect(listed[0]).not.toHaveProperty('storage_file_id')
  await expect(getClipPlayback({ clipId: 'clip-1', userId, repo, now }))
    .resolves.toMatchObject({ expires_in_seconds: 300 })
})

test.each([
  { retention_policy: 'unknown' },
  { retention_policy: null },
  { ordinary_expires_at: '2027-09-03T00:00:00.000Z' },
  { retention_expires_at: '2027-09-03T00:00:00.000Z' },
  { ordinary_expires_at: undefined },
  { retention_expires_at: undefined },
  { consent_status: 'revoked' },
  { deleted_at: '2026-09-01T00:00:00.000Z' },
  { retention_state: 'released' },
  { retention_state: 'deletion_pending' },
])('indefinite media fails closed on contradictory or private state %p', async change => {
  const repo = await permanentRepository(change)
  repo.createPlaybackUrl = jest.fn()
  const now = '2026-09-01T00:00:00.000Z'
  await expect(listClips({ sessionId: 'session-1', userId: 'user-a', repo, now })).resolves.toEqual([])
  await expect(getClipPlayback({ clipId: 'clip-1', userId: 'user-a', repo, now }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(repo.createPlaybackUrl).not.toHaveBeenCalled()
})

test('indefinite policy never grants an outsider list or playback access', async () => {
  const repo = await permanentRepository()
  repo.createPlaybackUrl = jest.fn()
  await expect(listClips({ sessionId: 'session-1', userId: 'outsider', repo })).resolves.toEqual([])
  await expect(getClipPlayback({ clipId: 'clip-1', userId: 'outsider', repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(repo.createPlaybackUrl).not.toHaveBeenCalled()
})

test.each([null, 'unknown', 'saved_until_deleted_v1'])(
  'a finite row with explicit policy %p cannot be interpreted as legacy', async retentionPolicy => {
    const repo = createRepository()
    Object.assign(await repo.getClip('clip-1'), { retention_policy: retentionPolicy,
      ordinary_expires_at: '2027-09-03T00:00:00.000Z', retention_expires_at: '2027-09-03T00:00:00.000Z' })
    const now = '2026-09-01T00:00:00.000Z'
    await expect(listClips({ sessionId: 'session-1', userId: 'user-a', repo, now })).resolves.toEqual([])
    await expect(getClipPlayback({ clipId: 'clip-1', userId: 'user-a', repo, now }))
      .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  },
)


test('non-participant cannot read a private session', async () => {
  await expect(getSession({ sessionId: 'session-1', userId: 'outsider', repo: createRepository() }))
    .rejects.toMatchObject({ code: 'FORBIDDEN' })
})


test('participant can list and read only their own sessions and clips', async () => {
  const repo = createRepository()
  await expect(listSessions({ userId: 'user-a', repo })).resolves.toHaveLength(1)
  await expect(getSession({ sessionId: 'session-1', userId: 'user-a', repo }))
    .resolves.toMatchObject({ session_id: 'session-1', analysis: { status: 'PROCESSING' } })
  const clips = await listClips({ sessionId: 'session-1', userId: 'user-a', repo })
  expect(clips).toEqual([expect.objectContaining({ clip_id: 'clip-1' })])
  expect(clips[0]).not.toHaveProperty('storage_file_id')
  expect(clips[0]).toMatchObject({ available_until: null })
  expect(clips[0]).not.toHaveProperty('ordinary_expires_at')
  expect(clips[0]).not.toHaveProperty('retention_expires_at')
})


test.each(['PROCESSING', 'AVAILABLE', 'PARTIAL', 'FAILED_RETRYING'])(
  'sessions list and get merge the %s analysis projection', async status => {
    const repo = createRepository()
    repo.seedAnalysis({
      schema_version: '1.0', session_id: 'session-1', revision: 3, status,
      total_shots: 3, available_shots: status === 'AVAILABLE' ? 3 : status === 'PARTIAL' ? 1 : 0,
      retrying_shots: status === 'FAILED_RETRYING' || status === 'PARTIAL' ? 1 : 0,
      report_available: status === 'AVAILABLE' || status === 'PARTIAL',
      deadline_at: '2026-08-26T14:07:00.000Z', updated_at: '2026-08-26T14:06:00.000Z',
      _device_id: 'must-not-leak', frozen_shots_digest: 'a'.repeat(64),
    })

    await expect(listSessions({ userId: 'user-a', repo }))
      .resolves.toEqual([expect.objectContaining({ analysis: expect.objectContaining({ status }) })])
    const detail = await getSession({ sessionId: 'session-1', userId: 'user-a', repo })
    expect(detail.analysis.status).toBe(status)
    expect(detail.analysis).not.toHaveProperty('_device_id')
    expect(detail.analysis).not.toHaveProperty('frozen_shots_digest')
  },
)


test('confirmed pre-publication binding grants status only, not report content', async () => {
  const repo = createRepository()
  repo.confirmBinding('session-1', 'bound-user')

  await expect(getSessionAnalysis({ sessionId: 'session-1', userId: 'bound-user', repo }))
    .resolves.toMatchObject({
      session_id: 'session-1', status: 'PROCESSING', live_shots_recorded: 3,
    })
  await expect(getSession({ sessionId: 'session-1', userId: 'bound-user', repo }))
    .rejects.toMatchObject({ code: 'FORBIDDEN' })
})

test('live shot progress is published only after participant authorization', async () => {
  const repo = createRepository()

  await expect(getSessionAnalysis({ sessionId: 'session-1', userId: 'user-a', repo }))
    .resolves.toMatchObject({ live_shots_recorded: 3 })
  await expect(getSessionAnalysis({ sessionId: 'session-1', userId: 'outsider', repo }))
    .rejects.toMatchObject({ code: 'FORBIDDEN' })
})


test('missing projection for an ended bound session returns unknown processing counts', async () => {
  const repo = createRepository()
  repo.seedAnalysis({ session_id: 'other-session' })
  repo.getSessionAnalysis = async () => null
  repo.confirmBinding('session-1', 'bound-user')

  await expect(getSessionAnalysis({ sessionId: 'session-1', userId: 'bound-user', repo }))
    .resolves.toEqual({
      schema_version: '1.0', session_id: 'session-1', revision: 1,
      status: 'PROCESSING', total_shots: null, available_shots: null,
      retrying_shots: null, report_available: false,
      deadline_at: '2026-08-26T14:07:00.000Z', updated_at: '2026-08-26T14:05:01.000Z',
    })
})


test('cross-session analysis projection fails closed', async () => {
  const repo = createRepository()
  repo.getSessionAnalysis = async () => ({
    schema_version: '1.0', session_id: 'session-2', revision: 1, status: 'PROCESSING',
  })
  await expect(getSessionAnalysis({ sessionId: 'session-1', userId: 'user-a', repo }))
    .rejects.toMatchObject({ code: 'SESSION_NOT_JOINABLE' })
})


test('authorized legacy published session remains readable without analysis authority', async () => {
  const repo = createRepository()
  repo.getSessionAnalysis = async () => null
  repo.getEndedFieldSession = async () => null

  await expect(listSessions({ userId: 'user-a', repo }))
    .resolves.toEqual([expect.objectContaining({ session_id: 'session-1', analysis: null })])
  await expect(getSession({ sessionId: 'session-1', userId: 'user-a', repo }))
    .resolves.toMatchObject({ session_id: 'session-1', analysis: null })
})


test('legacy or unknown clip source fails closed without publication', async () => {
  for (const source_kind of ['development_presentation', 'file_replay']) {
    const repo = createRepository()
    Object.assign(await repo.getClip('clip-1'), { source_kind, schema_version: '1.0' })
    await expect(listClips({ sessionId: 'session-1', userId: 'user-a', repo })).resolves.toEqual([])
  }
})

test('non-participant cannot mint a private clip playback URL', async () => {
  const repo = createRepository()
  repo.createPlaybackUrl = jest.fn()

  await expect(getClipPlayback({ clipId: 'clip-1', userId: 'outsider', repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(repo.createPlaybackUrl).not.toHaveBeenCalled()
})


test('playback URL is short-lived and never included in the stored clip', async () => {
  const repo = createRepository()
  const result = await getClipPlayback({
    clipId: 'clip-1', userId: 'user-a', repo, now: '2026-09-01T00:00:00.000Z',
  })
  expect(result).toMatchObject({ expires_in_seconds: 300 })
  expect(result).not.toHaveProperty('favorite')
  expect(result.url).toMatch(/^https:\/\/temporary\.invalid\//)
  expect(await repo.getClip('clip-1')).not.toHaveProperty('playback_url')
})


test.each([
  ['2026-09-03T00:00:00.000Z', '2026-09-03T00:00:00.000Z'],
  ['not-a-date', '2026-09-01T00:00:00.000Z'],
])('expired or invalid retention deadline %s is never signed', async (deadline, now) => {
  const repo = createRepository()
  const clip = await repo.getClip('clip-1')
  clip.retention_expires_at = deadline
  repo.createPlaybackUrl = jest.fn()
  await expect(getClipPlayback({ clipId: 'clip-1', userId: 'user-a', repo, now }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(repo.createPlaybackUrl).not.toHaveBeenCalled()
})

test('deletion-pending clips cannot issue playback', async () => {
  const repo = createRepository()
  const clip = await repo.getClip('clip-1')
  clip.retention_state = 'deletion_pending'
  await expect(getClipPlayback({ clipId: 'clip-1', userId: 'user-a', repo,
    now: '2026-09-01T00:00:00.000Z' })).rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
})


test.each([
  ['uploading', null, 'participants'],
  ['failed', null, 'participants'],
  ['published', '2026-08-26T20:00:00.000Z', 'participants'],
  ['published', null, 'revoked'],
])('unavailable media state %s is never signed', async (uploadStatus, deletedAt, consentStatus) => {
  const repo = createRepository()
  const clip = await repo.getClip('clip-1')
  clip.upload_status = uploadStatus
  clip.deleted_at = deletedAt
  clip.consent_status = consentStatus
  repo.createPlaybackUrl = jest.fn()

  await expect(getClipPlayback({
    clipId: 'clip-1', userId: 'user-a', repo, now: '2026-09-01T00:00:00.000Z',
  }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(repo.createPlaybackUrl).not.toHaveBeenCalled()
})

test('untrusted semantic flags are not published through ordinary report or clip reads', async () => {
  const repo = createRepository()
  const session = await repo.getSession('session-1')
  Object.assign(session, { reviewed_projection: { approved: true }, highlight: true,
    _sync_publication: { table_id: 'private-table' } })
  session.shots[0].personal_score = 100
  const clip = await repo.getClip('clip-1')
  Object.assign(clip, { highlight: true, reviewed: true, personal_score: 100 })
  const report = await getSession({ sessionId: 'session-1', userId: 'user-a', repo })
  expect(report).not.toHaveProperty('reviewed_projection')
  expect(report).not.toHaveProperty('_sync_publication')
  expect(report.shots[0]).not.toHaveProperty('personal_score')
  const clips = await listClips({ sessionId: 'session-1', userId: 'user-a', repo })
  expect(clips[0]).not.toHaveProperty('highlight')
  expect(clips[0]).not.toHaveProperty('reviewed')
  await expect(getClipPlayback({
    clipId: 'clip-1', userId: 'user-a', repo, now: '2026-09-01T00:00:00.000Z',
  }))
    .resolves.toMatchObject({ expires_in_seconds: 300 })
})

test('inconsistent repository rows cannot escape the requested session boundary', async () => {
  const repo = createRepository()
  repo.listClips = async () => [{ clip_id: 'foreign', session_id: 'other' }]
  await expect(listClips({ sessionId: 'session-1', userId: 'user-a', repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  repo.getSession = async () => ({ session_id: 'other', status: 'ended', sync_status: 'published' })
  await expect(getSession({ sessionId: 'session-1', userId: 'user-a', repo }))
    .rejects.toMatchObject({ code: 'SESSION_NOT_JOINABLE' })
})

test('session list rechecks membership rather than trusting candidate rows', async () => {
  const repo = createRepository()
  repo.listSessions = async () => [{ session_id: 'other', status: 'ended', sync_status: 'published' }]
  await expect(listSessions({ userId: 'user-a', repo })).resolves.toEqual([])
})

test('clip list exposes only active published large-screen-saved media', async () => {
  const repo = createRepository()
  const valid = await repo.getClip('clip-1')
  const variants = [
    { ...valid, clip_id: 'uploading', upload_status: 'uploading' },
    { ...valid, clip_id: 'failed', upload_status: 'failed' },
    { ...valid, clip_id: 'legacy', publication_kind: undefined },
    { ...valid, clip_id: 'revoked', consent_status: 'revoked' },
    { ...valid, clip_id: 'deleted', deleted_at: '2026-09-01T00:00:00.000Z' },
    { ...valid, clip_id: 'missing-retention-state', retention_state: undefined },
    { ...valid, clip_id: 'unknown-retention-state', retention_state: 'unknown' },
    { ...valid, clip_id: 'inactive-retention-state', retention_state: 'inactive' },
    { ...valid, clip_id: 'pending-delete', retention_state: 'deletion_pending' },
    { ...valid, clip_id: 'released', retention_state: 'released' },
    { ...valid, clip_id: 'missing-storage', storage_file_id: '' },
    { ...valid, clip_id: 'unsafe-storage', storage_file_id: 'https://public.invalid/clip.mp4' },
    { ...valid, clip_id: 'expired', retention_expires_at: '2026-08-31T00:00:00.000Z' },
  ]
  repo.listClips = async () => [valid, ...variants]

  const rows = await listClips({
    sessionId: 'session-1', userId: 'user-a', repo,
    now: '2026-09-01T00:00:00.000Z',
  })
  expect(rows.map(row => row.clip_id)).toEqual(['clip-1'])
  expect(rows[0]).toMatchObject({ publication_kind: 'saved_postgame_clip' })
  expect(rows[0]).not.toHaveProperty('favorite')
  expect(rows[0]).not.toHaveProperty('favorite_count')
})

test('direct playback rejects legacy published media not selected on the large screen', async () => {
  const repo = createRepository()
  const clip = await repo.getClip('clip-1')
  delete clip.publication_kind
  repo.createPlaybackUrl = jest.fn()

  await expect(getClipPlayback({
    clipId: 'clip-1', userId: 'user-a', repo, now: '2026-09-01T00:00:00.000Z',
  })).rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(repo.createPlaybackUrl).not.toHaveBeenCalled()
})

test.each([
  ['missing', undefined],
  ['unknown', 'unknown'],
  ['inactive', 'inactive'],
  ['deletion pending', 'deletion_pending'],
  ['released', 'released'],
])('direct playback rejects %s retention state', async (_label, retentionState) => {
  const repo = createRepository()
  const clip = await repo.getClip('clip-1')
  if (retentionState === undefined) delete clip.retention_state
  else clip.retention_state = retentionState
  repo.createPlaybackUrl = jest.fn()

  await expect(getClipPlayback({
    clipId: 'clip-1', userId: 'user-a', repo, now: '2026-09-01T00:00:00.000Z',
  })).rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(repo.createPlaybackUrl).not.toHaveBeenCalled()
})

test.each([
  ['non-array', {}],
  ['null array item', [null]],
  ['array item', [['not', 'a', 'record']]],
  ['date item', [new Date('2026-09-01T00:00:00.000Z')]],
])('malformed clip repository result %s fails with a stable media error', async (_label, value) => {
  const repo = createRepository()
  repo.listClips = async () => value
  await expect(listClips({
    sessionId: 'session-1', userId: 'user-a', repo,
    now: '2026-09-01T00:00:00.000Z',
  })).rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
})
