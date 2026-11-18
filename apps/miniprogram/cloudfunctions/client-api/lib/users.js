const { ClientApiError, stableError } = require('./errors')


async function getOptionalDocument(reference) {
  try {
    return await reference.get()
  } catch (error) {
    const message = error && error.message ? error.message : String(error)
    if (/document with _id .+ does not exist/i.test(message)) {
      return { data: null }
    }
    throw error
  }
}


function validateProfile(payload) {
  const nickname = payload.nickname == null ? null : String(payload.nickname).trim()
  const avatar = payload.avatar_file_id == null ? null : String(payload.avatar_file_id).trim()
  if (nickname && (nickname.length > 40 || /[\u0000-\u001f\u007f]/u.test(nickname))) {
    throw new ClientApiError('PROFILE_INVALID')
  }
  if (avatar && (avatar.length > 512 || !avatar.startsWith('cloud://'))) {
    throw new ClientApiError('PROFILE_INVALID')
  }
  return { nickname: nickname || null, avatar_file_id: avatar || null }
}


function createCloudRepositories(db) {
  return {
    identityRepo: {
      async resolveOrCreate(digest, create) {
        return db.runTransaction(async (transaction) => {
          const identity = transaction.collection('wechat_identities').doc(digest)
          const existing = await getOptionalDocument(identity)
          if (existing.data && existing.data.user_id) {
            const user = await getOptionalDocument(
              transaction.collection('users').doc(existing.data.user_id),
            )
            if (!user.data) throw new ClientApiError('IDENTITY_CONFLICT')
            return user.data
          }
          const user = await create()
          await transaction.collection('users').doc(user.user_id).set({ data: user })
          await identity.set({
            data: {
              schema_version: '1.0',
              identity_digest: digest,
              user_id: user.user_id,
              created_at: user.created_at,
            },
          })
          return user
        })
      },
    },
    usersRepo: {
      async updateProfile(userId, payload, now) {
        const profile = validateProfile(payload)
        await db.collection('users').doc(userId).update({
          data: { profile, updated_at: now },
        })
        return { schema_version: '1.0', user_id: userId, profile, status: 'active' }
      },
    },
  }
}


module.exports = { createCloudRepositories, getOptionalDocument, stableError, validateProfile }
