const crypto = require('crypto')


class PublicProfileError extends Error {
  constructor(code) {
    super(code)
    this.name = 'PublicProfileError'
    this.code = code
  }
}


function fail(code) {
  throw new PublicProfileError(code)
}


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


function normalizePublicNickname(input) {
  if (typeof input !== 'string' || /[\p{Cc}\p{Cf}]/u.test(input)) fail('PUBLIC_NICKNAME_INVALID')
  const nickname = input.normalize('NFKC').trim().replace(/\s+/gu, ' ')
  const length = [...nickname].length
  if (length < 2 || length > 16) fail('PUBLIC_NICKNAME_INVALID')
  return { nickname, nickname_key: nickname.toLowerCase() }
}


async function optional(ref) {
  try {
    const result = await ref.get()
    return result && result.data ? result.data : null
  } catch (error) {
    if (String(error && error.message).includes('does not exist')) return null
    throw error
  }
}


function withoutId(value) {
  const { _id: _documentId, nickname_key: _nicknameKey, ...profile } = value
  return structuredClone(profile)
}


function createPublicRankingProfileRepository(db, { now }) {
  return {
    async saveProfile({ userId, nickname: inputNickname, isPublic, acceptChallenges }) {
      if (typeof userId !== 'string' || userId.length === 0
          || typeof isPublic !== 'boolean' || typeof acceptChallenges !== 'boolean') {
        fail('PUBLIC_PROFILE_INVALID')
      }
      const { nickname, nickname_key: nicknameKey } = normalizePublicNickname(inputNickname)
      const updatedAt = now()
      return db.runTransaction(async (transaction) => {
        const profileRef = transaction.collection('public_ranking_profiles').doc(digest(userId))
        const claimRef = transaction.collection('public_nickname_claims').doc(digest(nicknameKey))
        const [existingProfile, existingClaim] = await Promise.all([
          optional(profileRef), optional(claimRef),
        ])
        if (existingClaim && existingClaim.user_id !== userId) fail('PUBLIC_NICKNAME_TAKEN')

        if (existingProfile && existingProfile.nickname_key !== nicknameKey) {
          const oldClaimRef = transaction.collection('public_nickname_claims')
            .doc(digest(existingProfile.nickname_key))
          const oldClaim = await optional(oldClaimRef)
          if (!oldClaim || oldClaim.user_id !== userId) fail('PUBLIC_PROFILE_CONFLICT')
          await oldClaimRef.remove()
        }

        const createdAt = existingProfile ? existingProfile.created_at : updatedAt
        const profile = {
          schema_version: '1.0', user_id: userId, nickname, nickname_key: nicknameKey,
          is_public: isPublic, accept_challenges: acceptChallenges,
          created_at: createdAt, updated_at: updatedAt,
        }
        await claimRef.set({ data: {
          schema_version: '1.0', nickname_key: nicknameKey, user_id: userId,
          claimed_at: existingClaim ? existingClaim.claimed_at : updatedAt,
        } })
        await profileRef.set({ data: profile })
        return withoutId(profile)
      })
    },
  }
}


module.exports = {
  PublicProfileError,
  createPublicRankingProfileRepository,
  normalizePublicNickname,
}
