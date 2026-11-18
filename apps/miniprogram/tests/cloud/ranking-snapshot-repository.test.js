const { createRankingSnapshotRepository } = require('../../cloudfunctions/shared/ranking-snapshot-repository')


function fakeDatabase() {
  const names = ['ranking_snapshot_heads', 'ranking_snapshots', 'ranking_snapshot_entries']
  const maps = Object.fromEntries(names.map(name => [name, new Map()]))
  const collection = name => ({
    doc(id) { return {
      async get() { return { data: maps[name].get(id) || null } },
      async set({ data }) { maps[name].set(id, { _id: id, ...structuredClone(data) }) },
      async update({ data }) { maps[name].set(id, { ...maps[name].get(id), ...structuredClone(data) }) },
    } },
  })
  return { maps, collection, async runTransaction(callback) { return callback({ collection }) } }
}


function payload(override = {}) {
  const snapshot = (id, kind, entries) => ({
    schema_version: '1.0', snapshot_id: id, snapshot_version: 1,
    ranking_scope_id: 'zhoukou-urban', is_test: false,
    kind, season_id: kind === 'season' ? 's1' : null,
    as_of: '2026-09-02T12:00:00.000Z', total_entries: entries.length, entries,
  })
  return {
    schema_version: '1.0', snapshot_set_id: 'set-1', ranking_scope_id: 'zhoukou-urban',
    is_test: false,
    season_id: 's1', as_of: '2026-09-02T12:00:00.000Z',
    long_term: snapshot('long-1', 'long_term', [
      { rank: 1, user_id: 'u1', nickname: '甲', display_rating: 1300, recent_sample_limited: false },
    ]),
    season: snapshot('season-1', 'season', []),
    ...override,
  }
}


test('snapshot repository stages entries then publishes one shared monotonic version idempotently', async () => {
  const db = fakeDatabase()
  const repo = createRankingSnapshotRepository(db, { now: () => '2026-09-02T12:00:01.000Z' })
  const first = await repo.publish(payload())
  const retry = await repo.publish(payload())
  expect(first).toEqual({ snapshot_set_id: 'set-1', snapshot_version: 1 })
  expect(retry).toEqual(first)
  expect([...db.maps.ranking_snapshots.values()]).toEqual(expect.arrayContaining([
    expect.objectContaining({ snapshot_id: 'long-1', snapshot_version: 1, status: 'published' }),
    expect.objectContaining({ snapshot_id: 'season-1', snapshot_version: 1, status: 'published' }),
  ]))
  expect([...db.maps.ranking_snapshot_entries.values()][0]).toMatchObject({
    snapshot_id: 'long-1', position: 1, rank: 1, user_id: 'u1', is_test: false,
  })
  expect([...db.maps.ranking_snapshot_heads.values()][0]).toMatchObject({ is_test: false })
  expect([...db.maps.ranking_snapshots.values()].every(document => document.is_test === false)).toBe(true)
})


test('snapshot head rejects changing the test classification of one scope', async () => {
  const db = fakeDatabase()
  const repo = createRankingSnapshotRepository(db, { now: () => '2026-09-02T12:00:01.000Z' })
  await repo.publish(payload())

  await expect(repo.publish(payload({ snapshot_set_id: 'set-2', is_test: true })))
    .rejects.toThrow('RANKING_ENVIRONMENT_SCOPE_MISMATCH')
})
