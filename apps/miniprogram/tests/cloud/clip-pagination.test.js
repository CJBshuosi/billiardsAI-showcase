const { createCloudContentRepository, listClips, listClipsPage, getClipPlayback } = require('../../cloudfunctions/client-api/lib/content')
const { privateClip } = require('../helpers/private-clips')
const { documentId, ownershipProjection } = require('../../cloudfunctions/shared/shot-ownership')

function database(clips) {
  const records = { clips: Object.fromEntries(clips.map(clip => [clip.clip_id, clip])), field_shot_ownership: {} }
  for (const clip of clips) records.field_shot_ownership[documentId(`${clip.session_id}:${clip.shot_id}`)] = ownershipProjection(clip, 'table-1', 'device-1')
  const queries = []
  function matches(row, condition) {
    if (condition.op === 'and') return condition.values.every(value => matches(row, value))
    if (condition.op === 'or') return condition.values.some(value => matches(row, value))
    return Object.entries(condition).every(([key, value]) => value && value.op === 'lt'
      ? row[key] < value.value : row[key] === value)
  }
  return {
    queries, records,
    async runTransaction(fn) { return fn(this) },
    command: { and: values => ({ op: 'and', values }), or: values => ({ op: 'or', values }), lt: value => ({ op: 'lt', value }) },
    collection(name) {
      let filter = {}; const order = []; let limit = Infinity
      return {
        doc(id) { return { async update({ data }) { Object.assign(records[name][id], data) }, async get() { return { data: records[name]?.[id] || null } } } },
        where(value) { filter = value; return this },
        orderBy(field, direction) { order.push([field, direction]); return this },
        limit(value) { limit = value; return this },
        async get() {
          queries.push({ name, filter, order, limit })
          const data = Object.values(records[name] || {}).filter(row => matches(row, filter)).sort((a, b) => {
            for (const [field, direction] of order) {
              const difference = a[field] < b[field] ? -1 : a[field] > b[field] ? 1 : 0
              if (difference) return direction === 'desc' ? -difference : difference
            }
            return 0
          }).slice(0, limit)
          return { data }
        },
      }
    },
  }
}
function clips(count, change = {}) {
  return Array.from({ length: count }, (_, i) => privateClip({
    clip_id: `clip-${String(i).padStart(4, '0')}`, session_id: `session-${i}`, shot_id: 'shot-1', sequence: 1,
    published_at: new Date(Date.UTC(2026, 9, 28, 14, 0, Math.floor(i / 3))).toISOString(), ...change,
  }))
}
async function traverse(repo, options = {}) {
  const results = []; let cursor; let pageCount = 0
  do {
    const page = await listClipsPage({ userId: 'user-a', ...options, cursor, repo })
    expect(page.schema_version).toBe('1.0')
    results.push(...page.items); cursor = page.next_cursor; pageCount += 1
    expect(pageCount).toBeLessThan(20)
  } while (cursor)
  return results
}

test('production owner repository exposes all 101 clips through stable chronological keysets including ties', async () => {
  expect(typeof listClipsPage).toBe('function')
  const owned = clips(101)
  const db = database(owned)
  const repo = createCloudContentRepository(db, {})
  const result = await traverse(repo)
  const expected = [...owned].sort((a, b) => b.published_at.localeCompare(a.published_at) || b.clip_id.localeCompare(a.clip_id))
  expect(result.map(clip => clip.clip_id)).toEqual(expected.map(clip => clip.clip_id))
  expect(new Set(result.map(clip => clip.clip_id)).size).toBe(101)
  expect(db.queries).toHaveLength(5)
  for (const query of db.queries) {
    expect(query.limit).toBe(25)
    expect(query.order).toEqual([['published_at', 'desc'], ['clip_id', 'desc']])
  }
  await expect(listClips({ userId: 'user-a', repo })).resolves.toHaveLength(100)
})

test('a fully filtered candidate page advances past deleted and invalid rows to later valid owner clips', async () => {
  expect(typeof listClipsPage).toBe('function')
  const invalid = clips(25, { published_at: '2026-10-29T00:00:00.000Z' }).map((clip, i) => ({
    ...clip, clip_id: `invalid-${i}`, ...(i % 2 ? { deleted_at: '2026-10-29T01:00:00.000Z' } : { privacy_policy: 'legacy' }),
  }))
  const db = database([...invalid, ...clips(101)])
  const repo = createCloudContentRepository(db, {})
  const first = await listClipsPage({ userId: 'user-a', repo })
  expect(first.items).toEqual([])
  expect(first.next_cursor).toEqual(expect.any(String))
  expect(await traverse(repo)).toHaveLength(101)
})

test('cursor scope rejects a different owner or optional session before querying', async () => {
  expect(typeof listClipsPage).toBe('function')
  const db = database(clips(26, { session_id: 'same-session', shot_id: 'same-shot' }))
  const repo = createCloudContentRepository(db, {})
  const page = await listClipsPage({ userId: 'user-a', sessionId: 'same-session', repo })
  const queryCount = db.queries.length
  for (const options of [{ userId: 'user-b', sessionId: 'same-session' }, { userId: 'user-a', sessionId: 'other-session' }, { userId: 'user-a' }]) {
    await expect(listClipsPage({ ...options, cursor: page.next_cursor, repo })).rejects.toMatchObject({ code: 'REQUEST_INVALID' })
  }
  expect(db.queries).toHaveLength(queryCount)
  const result = await traverse(repo, { sessionId: 'same-session' })
  expect(result).toHaveLength(26)
})

test('paged candidates never gain playback authority or trust a client-supplied owner', async () => {
  expect(typeof listClipsPage).toBe('function')
  const db = database(clips(1))
  const cloud = { getTempFileURL: jest.fn() }
  const repo = createCloudContentRepository(db, cloud)
  await expect(listClipsPage({ userId: 'user-b', owner_user_id: 'user-a', repo })).resolves.toMatchObject({ items: [], next_cursor: null })
  await expect(getClipPlayback({ clipId: 'clip-0000', userId: 'user-b', repo })).rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
  expect(cloud.getTempFileURL).not.toHaveBeenCalled()
})

test.each(['garbage', '', 10, 'a'.repeat(2049)])('malformed page cursor %p fails closed', async cursor => {
  expect(typeof listClipsPage).toBe('function')
  const db = database(clips(1)); const repo = createCloudContentRepository(db, {})
  await expect(listClipsPage({ userId: 'user-a', cursor, repo })).rejects.toMatchObject({ code: 'REQUEST_INVALID' })
  expect(db.queries).toHaveLength(0)
})

test('same-session clip outside legacy first 100 returns owner-authorized direct metadata with playback', async () => {
  expect(typeof listClipsPage).toBe('function')
  const owned = clips(101, { session_id: 'long-session' })
  const db = database(owned)
  const cloud = { getTempFileURL: jest.fn(async () => ({ fileList: [{ status: 0, tempFileURL: 'https://private.invalid/last.mp4' }] })) }
  const repo = createCloudContentRepository(db, cloud)
  const legacy = await listClips({ userId: 'user-a', sessionId: 'long-session', repo })
  expect(legacy.some(clip => clip.clip_id === 'clip-0100')).toBe(false)
  const playback = await getClipPlayback({ userId: 'user-a', clipId: 'clip-0100', repo })
  expect(playback).toMatchObject({ url: expect.any(String), clip: { clip_id: 'clip-0100', actor_role: 'red' } })
  expect(playback.clip).not.toHaveProperty('owner_user_id')
  expect(playback.clip).not.toHaveProperty('ownership_receipt')
  expect(cloud.getTempFileURL).toHaveBeenCalledTimes(1)
})

test('clips.listPage route takes owner from authenticated identity and preserves the legacy array route', async () => {
  const db = database(clips(26))
  const previousEnvironment = process.env.RANKING_RUNTIME_ENVIRONMENT
  process.env.RANKING_RUNTIME_ENVIRONMENT = 'development'
  jest.doMock('wx-server-sdk', () => ({
    init() {}, DYNAMIC_CURRENT_ENV: 'synthetic', database: () => db,
    getWXContext: () => ({ OPENID: 'owner-openid' }),
  }))
  jest.doMock('../../cloudfunctions/client-api/lib/identity', () => ({
    resolveUserId: async () => ({ user_id: 'user-a' }),
  }))
  try {
    let main
    jest.isolateModules(() => { main = require('../../cloudfunctions/client-api').main })
    const event = { schema_version: '1.0', action: 'clips.listPage', payload: { owner_user_id: 'user-b' } }
    const first = await main(event)
    expect(first.ok).toBe(true)
    expect(first.data.items).toHaveLength(25)
    const second = await main({ ...event, payload: { cursor: first.data.next_cursor } })
    expect(second.data.items).toHaveLength(1)
    const legacy = await main({ schema_version: '1.0', action: 'clips.list', payload: {} })
    expect(Array.isArray(legacy.data)).toBe(true)
    expect(legacy.data).toHaveLength(26)
  } finally {
    jest.dontMock('wx-server-sdk')
    jest.dontMock('../../cloudfunctions/client-api/lib/identity')
    if (previousEnvironment === undefined) delete process.env.RANKING_RUNTIME_ENVIRONMENT
    else process.env.RANKING_RUNTIME_ENVIRONMENT = previousEnvironment
  }
})

test('paged service independently rejects foreign candidate owners before projection access', async () => {
  const candidate = clips(1)[0]
  const repo = {
    listClipPageCandidates: async () => [candidate],
    getShotOwnership: jest.fn(),
  }
  await expect(listClipsPage({ userId: 'user-b', repo })).resolves.toMatchObject({ items: [], next_cursor: null })
  expect(repo.getShotOwnership).not.toHaveBeenCalled()
})
