const crypto = require('crypto')

const { ClientApiError } = require('./errors')


function identityDigest(openid) {
  return crypto.createHash('sha256').update(openid, 'utf8').digest('hex')
}


function anonymousIdentity(userId) {
  const digest = identityDigest(userId)
  return {
    avatar_variant: Number.parseInt(digest.slice(0, 2), 16) % 8,
    display_name: `球友 ${digest.slice(0, 6).toUpperCase()}`,
  }
}


function publicUser(user) {
  if (!user || typeof user.user_id !== 'string' || !user.user_id) {
    throw new ClientApiError('IDENTITY_CONFLICT')
  }
  return {
    schema_version: '1.0',
    user_id: user.user_id,
    status: user.status,
    identity: anonymousIdentity(user.user_id),
    profile: {
      nickname: user.profile.nickname,
      avatar_file_id: user.profile.avatar_file_id,
    },
  }
}


async function resolveUserId({ openid, identityRepo, idFactory, now }) {
  if (typeof openid !== 'string' || !openid.trim()) {
    throw new ClientApiError('AUTH_REQUIRED')
  }
  if (!identityRepo || typeof identityRepo.resolveOrCreate !== 'function') {
    throw new ClientApiError('INTERNAL_ERROR')
  }
  const digest = identityDigest(openid)
  const user = await identityRepo.resolveOrCreate(digest, async () => ({
    schema_version: '1.0',
    user_id: idFactory(),
    status: 'active',
    profile: { nickname: null, avatar_file_id: null },
    created_at: now(),
    updated_at: now(),
  }))
  return publicUser(user)
}


module.exports = { anonymousIdentity, identityDigest, publicUser, resolveUserId }
