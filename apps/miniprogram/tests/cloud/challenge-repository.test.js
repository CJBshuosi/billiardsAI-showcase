const { createCloudChallengeRepository } = require('../../cloudfunctions/shared/challenge-repository')


function fakeDatabase() {
  const maps = Object.fromEntries(['challenges', 'challenge_pair_slots', 'challenge_request_keys']
    .map(name => [name, new Map()]))
  const collection = name => ({
    doc(id) { return {
      async get() { return { data: maps[name].get(id) || null } },
      async set({ data }) { maps[name].set(id, { _id: id, ...structuredClone(data) }) },
      async remove() { maps[name].delete(id) },
    } },
  })
  return { maps, collection, async runTransaction(callback) { return callback({ collection }) } }
}


function challenge(id = 'challenge-1') {
  return {
    schema_version: '1.0', challenge_id: id, challenger_user_id: 'a', target_user_id: 'b',
    venue_id: 'venue-1', ranking_scope_id: 'zhoukou-urban', proposed_at: '2026-09-02T12:00:00.000Z',
    format: 'single', race_target: null, status: 'pending', created_at: '2026-09-01T12:00:00.000Z',
    expires_at: '2026-09-03T12:00:00.000Z', onsite_scan_required: true,
  }
}


test('cloud repository atomically persists request, pair slot and challenge', async () => {
  const db = fakeDatabase()
  const repo = createCloudChallengeRepository(db)
  await expect(repo.create({
    retryKey: 'a:key-1', requestDigest: 'intent-1', pairKey: 'a:b', challenge: challenge(),
  })).resolves.toEqual(challenge())
  await expect(repo.findRequest('a:key-1')).resolves.toMatchObject({
    request_digest: 'intent-1', challenge: { challenge_id: 'challenge-1' },
  })
  await expect(repo.create({
    retryKey: 'b:key-2', requestDigest: 'intent-2', pairKey: 'a:b', challenge: challenge('challenge-2'),
  })).rejects.toMatchObject({ code: 'CHALLENGE_PAIR_PENDING' })
})


test('terminal transition releases the unordered pair slot', async () => {
  const db = fakeDatabase()
  const repo = createCloudChallengeRepository(db)
  await repo.create({ retryKey: 'a:key', requestDigest: 'intent', pairKey: 'a:b', challenge: challenge() })
  await expect(repo.transition({
    challengeId: 'challenge-1', pairKey: 'a:b', nextStatus: 'accepted',
    respondedAt: '2026-09-01T13:00:00.000Z',
  })).resolves.toMatchObject({ status: 'accepted' })
  expect(db.maps.challenge_pair_slots.size).toBe(0)
  await expect(repo.transition({
    challengeId: 'challenge-1', pairKey: 'a:b', nextStatus: 'rejected',
    respondedAt: '2026-09-01T14:00:00.000Z',
  })).rejects.toMatchObject({ code: 'CHALLENGE_NOT_PENDING' })
})
