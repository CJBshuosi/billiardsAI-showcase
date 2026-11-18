const crypto = require('crypto')
const { completeFieldCommand, createCloudDeviceFieldCommandRepository } = require('../../cloudfunctions/sync-api/lib/field-commands')
const { createCloudContentRepository, listClips, getClipPlayback } = require('../../cloudfunctions/client-api/lib/content')
const fixture = require('../../../../tests/fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json')
const { historyId } = require('../helpers/private-clips')
const hash = value => crypto.createHash('sha256').update(value).digest('hex')

function database(seed) {
  const rows = JSON.parse(JSON.stringify(seed))
  const queries = []
  function collection(name) {
    rows[name] ||= {}
    let filters = {}
    return {
      doc(id) { return {
        async get() { return { data: rows[name][id] || null } },
        async set({ data }) { rows[name][id] = data },
        async update({ data }) { Object.assign(rows[name][id], data) },
        async remove() { delete rows[name][id] },
      } },
      where(value) { filters = value; queries.push([name, value]); return this },
      orderBy() { return this }, limit() { return this },
      async get() { return { data: Object.values(rows[name]).filter(row =>
        Object.entries(filters).every(([key, value]) => row[key] === value)) } },
    }
  }
  return { rows, queries, collection, async runTransaction(fn) {
    const snapshot = structuredClone(rows)
    try { return await fn({ collection }) } catch (error) {
      for (const key of Object.keys(rows)) delete rows[key]
      Object.assign(rows, snapshot)
      throw error
    }
  } }
}
function commandDb(type = 'claim_role', session = {}) {
  return database({
    field_session_commands: { 'claim-red-1': {
      command_id: 'claim-red-1', command_type: type, session_id: 'session-1',
      table_id: 'table-1', actor_user_id: 'user-a', validated_payload: { role: 'red' },
      status: 'claimed', claimed_by_device_id: 'device-1',
    } },
    field_live_sessions: { 'session-1': {
      session_id: 'session-1', table_id: 'table-1', red_user_id: 'user-a', blue_user_id: 'user-b', ...session,
    } },
  })
}
function complete(db, revision = 1) {
  return completeFieldCommand({
    commandId: 'claim-red-1', tableId: 'table-1', deviceId: 'device-1', status: 'applied',
    publicResultCode: 'APPLIED', resultingRevision: revision, now: 'now',
    repo: createCloudDeviceFieldCommandRepository(db),
  })
}

test.each(['claim_role', 'release_role'])('applied %s writes immutable history atomically and retries without migration', async type => {
  const db = commandDb(type)
  await complete(db)
  expect(db.rows.session_role_binding_history[historyId()]).toEqual({
    session_id: 'session-1', role: 'red', user_id: type === 'claim_role' ? 'user-a' : null,
    resulting_revision: 1, command_id: 'claim-red-1', table_id: 'table-1', device_id: 'device-1',
  })
  const before = structuredClone(db.rows)
  await complete(db)
  expect(db.rows).toEqual(before)
  delete db.rows.session_role_binding_history[historyId()]
  await complete(db)
  expect(db.rows.session_role_binding_history[historyId()]).toBeUndefined()
})

test('history key conflicts roll back command and session writes', async () => {
  const db = commandDb()
  db.rows.session_role_binding_history = { [historyId()]: { user_id: 'intruder' } }
  await expect(complete(db)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
  expect(db.rows.field_session_commands['claim-red-1'].status).toBe('claimed')
})

test('release without exactly one prior actor role fails closed instead of defaulting to blue', async () => {
  const db = commandDb('release_role', { red_user_id: 'user-c' })
  await expect(complete(db)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})

test('unsafe resulting revision is rejected before command writes', async () => {
  await expect(complete(commandDb(), Number.MAX_SAFE_INTEGER + 1)).rejects.toMatchObject({ code: 'SYNC_INVALID' })
})

function privateDb(change = {}) {
  const d = fixture.descriptor
  return database({ clips: { [d.clip_id]: {
    ...d, table_id: 'table-1', _device_id: 'device-1', publication_kind: 'saved_postgame_clip', upload_status: 'published',
    privacy_policy: 'shot_owner_only_v1', consent_status: 'owner', retention_policy: 'saved_until_deleted_v1',
    retention_state: 'active', ordinary_expires_at: null, retention_expires_at: null,
    deleted_at: null, storage_file_id: 'cloud://private/clip.mp4', ...change,
  } }, field_shot_ownership: { [hash('session-1:shot-0001')]: {
    session_id: d.session_id, shot_id: d.shot_id, owner_user_id: d.owner_user_id,
    actor_role: d.actor_role, ownership_receipt: d.ownership_receipt,
    ownership_receipt_sha256: d.ownership_receipt_sha256, table_id: 'table-1', device_id: 'device-1',
  } }, session_participants: {} })
}

test('former owner finds private collection without session membership; other users cannot list or sign a known ID', async () => {
  const db = privateDb()
  const cloud = { getTempFileURL: jest.fn(async () => ({ fileList: [{ status: 0, tempFileURL: 'https://private.invalid' }] })) }
  const repo = createCloudContentRepository(db, cloud)
  const rows = await listClips({ userId: 'user-a', repo })
  expect(rows).toEqual([expect.objectContaining({ actor_role: 'red' })])
  expect(rows[0]).not.toHaveProperty('owner_user_id')
  expect(db.queries).toContainEqual(['clips', { owner_user_id: 'user-a' }])
  for (const userId of ['user-b', 'user-c']) {
    await expect(listClips({ userId, sessionId: 'session-1', repo })).resolves.toEqual([])
    await expect(getClipPlayback({ clipId: fixture.descriptor.clip_id, userId, repo }))
      .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  }
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
  await expect(getClipPlayback({ clipId: fixture.descriptor.clip_id, userId: 'user-a', repo }))
    .resolves.toHaveProperty('url')
})

test.each([
  { schema_version: '1.0' }, { privacy_policy: undefined }, { consent_status: 'participants' },
  { ownership_receipt_sha256: '0'.repeat(64) }, { owner_user_id: 'user-b' },
  { source_kind: 'development_replay' }, { save_receipt_sha256: '0'.repeat(64) },
])('legacy or corrupted private clip %p cannot list or mint a URL', async change => {
  const db = privateDb(change)
  const cloud = { getTempFileURL: jest.fn() }
  const repo = createCloudContentRepository(db, cloud)
  const userId = change.owner_user_id || 'user-a'
  await expect(listClips({ userId, repo })).resolves.toEqual([])
  await expect(getClipPlayback({ clipId: fixture.descriptor.clip_id, userId, repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
})

test('terminal replay detects conflicting existing history without rebuilding missing history', async () => {
  const db = commandDb()
  await complete(db)
  db.rows.session_role_binding_history[historyId()].user_id = 'user-c'
  await expect(complete(db)).rejects.toMatchObject({ code: 'SYNC_CONFLICT' })
})


test('unknown fields in the immutable ownership projection fail closed before signing', async () => {
  const db = privateDb()
  db.rows.field_shot_ownership[hash('session-1:shot-0001')].alternate_owner = 'user-c'
  const cloud = { getTempFileURL: jest.fn(async () => ({ fileList: [{ status: 0, tempFileURL: 'https://private.invalid' }] })) }
  const repo = createCloudContentRepository(db, cloud)
  await expect(getClipPlayback({ clipId: fixture.descriptor.clip_id, userId: 'user-a', repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
})
