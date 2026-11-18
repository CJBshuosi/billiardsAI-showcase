const {
  PublicProfileError,
  createPublicRankingProfileRepository,
  normalizePublicNickname,
} = require('../../cloudfunctions/shared/public-ranking-profile')


function fakeDatabase() {
  const maps = { public_ranking_profiles: new Map(), public_nickname_claims: new Map() }
  const collection = name => ({
    doc(id) { return {
      async get() { return { data: maps[name].get(id) || null } },
      async set({ data }) { maps[name].set(id, { _id: id, ...structuredClone(data) }) },
      async remove() { maps[name].delete(id) },
    } },
  })
  return { maps, collection, async runTransaction(callback) { return callback({ collection }) } }
}


test('nickname normalization closes case, width and whitespace aliases', () => {
  expect(normalizePublicNickname('  球王 ＡＢ  ')).toEqual({
    nickname: '球王 AB', nickname_key: '球王 ab',
  })
})


test.each(['甲', 'a'.repeat(17), '球\n王', '   '])('invalid public nickname %p is rejected', nickname => {
  expect(() => normalizePublicNickname(nickname)).toThrow(expect.objectContaining({
    name: 'PublicProfileError', code: 'PUBLIC_NICKNAME_INVALID',
  }))
})


test('normalized nickname is unique across concurrent-style profile writes', async () => {
  const db = fakeDatabase()
  const repository = createPublicRankingProfileRepository(db, { now: () => '2026-09-01T12:00:00.000Z' })

  await repository.saveProfile({
    userId: 'user-1', nickname: '球王 AB', isPublic: true, acceptChallenges: true,
  })
  await expect(repository.saveProfile({
    userId: 'user-2', nickname: '球王 ＡＢ', isPublic: true, acceptChallenges: false,
  })).rejects.toMatchObject({ code: 'PUBLIC_NICKNAME_TAKEN' })
  expect(db.maps.public_ranking_profiles.size).toBe(1)
})


test('privacy and challenge permissions are separate and idempotent', async () => {
  const db = fakeDatabase()
  const repository = createPublicRankingProfileRepository(db, { now: () => '2026-09-01T12:00:00.000Z' })
  const first = await repository.saveProfile({
    userId: 'user-1', nickname: '周口球友', isPublic: false, acceptChallenges: true,
  })
  const publicProfile = await repository.saveProfile({
    userId: 'user-1', nickname: '周口球友', isPublic: true, acceptChallenges: true,
  })

  expect(first).toMatchObject({ is_public: false, accept_challenges: true })
  expect(publicProfile).toMatchObject({ is_public: true, accept_challenges: true })
  expect(db.maps.public_nickname_claims.size).toBe(1)
})


test('changing nickname atomically releases the old normalized claim', async () => {
  const db = fakeDatabase()
  const repository = createPublicRankingProfileRepository(db, { now: () => '2026-09-01T12:00:00.000Z' })
  await repository.saveProfile({
    userId: 'user-1', nickname: '周口老名字', isPublic: true, acceptChallenges: false,
  })
  await repository.saveProfile({
    userId: 'user-1', nickname: '周口新名字', isPublic: true, acceptChallenges: false,
  })

  await expect(repository.saveProfile({
    userId: 'user-2', nickname: '周口老名字', isPublic: true, acceptChallenges: false,
  })).resolves.toMatchObject({ user_id: 'user-2', nickname: '周口老名字' })
})


test('stable public profile errors do not expose claim documents', () => {
  expect(new PublicProfileError('PUBLIC_NICKNAME_TAKEN')).toMatchObject({
    message: 'PUBLIC_NICKNAME_TAKEN', code: 'PUBLIC_NICKNAME_TAKEN',
  })
})
