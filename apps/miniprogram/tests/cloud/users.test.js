const {
  createCloudRepositories,
  validateProfile,
} = require('../../cloudfunctions/client-api/lib/users')


test('profile fields are optional and normalized', () => {
  expect(validateProfile({ nickname: '  Test User  ', avatar_file_id: null }))
    .toEqual({ nickname: 'Test User', avatar_file_id: null })
})


test('profile rejects control characters and non-cloud avatars', () => {
  expect(() => validateProfile({ nickname: 'bad\u0000name' }))
    .toThrow('PROFILE_INVALID')
  expect(() => validateProfile({ avatar_file_id: 'https://example.invalid/avatar.png' }))
    .toThrow('PROFILE_INVALID')
})


test('first WeChat login creates a user when the identity document does not exist', async () => {
  const writes = new Map()
  const transaction = {
    collection(name) {
      return {
        doc(id) {
          return {
            async get() {
              if (name === 'wechat_identities') {
                throw new Error(`document.get:fail document with _id ${id} does not exist`)
              }
              return { data: writes.get(`${name}/${id}`) || null, errMsg: 'document.get:ok' }
            },
            async set({ data }) {
              writes.set(`${name}/${id}`, data)
              return { errMsg: 'document.set:ok' }
            },
          }
        },
      }
    },
  }
  const db = { runTransaction: async (callback) => callback(transaction) }
  const repositories = createCloudRepositories(db)
  const user = {
    schema_version: '1.0',
    user_id: 'user-first-login',
    status: 'active',
    profile: { nickname: null, avatar_file_id: null },
    created_at: '2026-08-27T10:34:55.000Z',
    updated_at: '2026-08-27T10:34:55.000Z',
  }

  await expect(repositories.identityRepo.resolveOrCreate('identity-digest', async () => user))
    .resolves.toEqual(user)
  expect(writes.get('users/user-first-login')).toEqual(user)
  expect(writes.get('wechat_identities/identity-digest')).toEqual({
    schema_version: '1.0',
    identity_digest: 'identity-digest',
    user_id: 'user-first-login',
    created_at: '2026-08-27T10:34:55.000Z',
  })
})
