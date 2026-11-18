const {
  publicUser,
  resolveUserId,
} = require('../../cloudfunctions/client-api/lib/identity')


function createFixture() {
  const users = new Map()
  const identities = new Map()
  const usersRepo = {
    async get(userId) {
      return users.get(userId) || null
    },
  }
  const identityRepo = {
    async resolveOrCreate(digest, create) {
      if (identities.has(digest)) {
        return identities.get(digest)
      }
      const user = await create()
      users.set(user.user_id, user)
      identities.set(digest, user)
      return user
    },
  }
  return {
    openid: 'synthetic-openid-value',
    usersRepo,
    identityRepo,
    idFactory: () => 'user-synthetic-0001',
    now: () => '2026-08-26T14:00:00.000Z',
  }
}


test('same openid resolves to one internal user without exposing openid', async () => {
  const fixture = createFixture()

  const first = await resolveUserId(fixture)
  const second = await resolveUserId(fixture)

  expect(first.user_id).toBe('user-synthetic-0001')
  expect(first).toEqual(second)
  expect(JSON.stringify(first)).not.toContain('synthetic-openid-value')
})


test('public identity has a stable anonymous name and avatar variant', () => {
  const user = {
    user_id: 'user-synthetic-0001',
    status: 'active',
    profile: { nickname: null, avatar_file_id: null },
  }

  expect(publicUser(user).identity).toEqual({
    avatar_variant: 7,
    display_name: '球友 2FC250',
  })
  expect(publicUser(user).identity).toEqual(publicUser({ ...user }).identity)
})


test('missing trusted identity fails closed', async () => {
  const fixture = { ...createFixture(), openid: '' }

  await expect(resolveUserId(fixture))
    .rejects.toMatchObject({ code: 'AUTH_REQUIRED' })
})


test('identity mapping never accepts a caller supplied user id', async () => {
  const fixture = { ...createFixture(), claimedUserId: 'user-attacker' }

  const resolved = await resolveUserId(fixture)

  expect(resolved.user_id).toBe('user-synthetic-0001')
  expect(JSON.stringify(resolved)).not.toContain('user-attacker')
})
