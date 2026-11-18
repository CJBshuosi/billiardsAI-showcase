const crypto = require('crypto')

const { ChallengeError } = require('./challenge-service')


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
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


function clean(value) {
  if (!value) return null
  const { _id: _documentId, ...document } = value
  return structuredClone(document)
}


function createCloudChallengeRepository(db) {
  return {
    async findRequest(retryKey) {
      const request = await optional(db.collection('challenge_request_keys').doc(digest(retryKey)))
      if (!request) return null
      const challenge = await optional(db.collection('challenges').doc(digest(request.challenge_id)))
      if (!challenge) throw new ChallengeError('CHALLENGE_CONFLICT')
      return { request_digest: request.request_digest, challenge: clean(challenge) }
    },

    async create({ retryKey, requestDigest, pairKey, challenge }) {
      return db.runTransaction(async transaction => {
        const requestRef = transaction.collection('challenge_request_keys').doc(digest(retryKey))
        const pairRef = transaction.collection('challenge_pair_slots').doc(digest(pairKey))
        const challengeRef = transaction.collection('challenges').doc(digest(challenge.challenge_id))
        const [existingRequest, existingPair, existingChallenge] = await Promise.all([
          optional(requestRef), optional(pairRef), optional(challengeRef),
        ])
        if (existingRequest) {
          if (existingRequest.request_digest !== requestDigest) {
            throw new ChallengeError('CHALLENGE_IDEMPOTENCY_CONFLICT')
          }
          const stored = await optional(
            transaction.collection('challenges').doc(digest(existingRequest.challenge_id)),
          )
          if (!stored) throw new ChallengeError('CHALLENGE_CONFLICT')
          return clean(stored)
        }
        if (existingPair) throw new ChallengeError('CHALLENGE_PAIR_PENDING')
        if (existingChallenge) throw new ChallengeError('CHALLENGE_CONFLICT')
        await challengeRef.set({ data: structuredClone(challenge) })
        await pairRef.set({ data: {
          schema_version: '1.0', pair_key: pairKey,
          challenge_id: challenge.challenge_id, created_at: challenge.created_at,
        } })
        await requestRef.set({ data: {
          schema_version: '1.0', challenger_user_id: challenge.challenger_user_id,
          idempotency_key_digest: digest(retryKey.split(':').slice(1).join(':')),
          request_digest: requestDigest, challenge_id: challenge.challenge_id,
          created_at: challenge.created_at,
        } })
        return structuredClone(challenge)
      })
    },

    async get(challengeId) {
      return clean(await optional(db.collection('challenges').doc(digest(challengeId))))
    },

    async listForUser(userId) {
      const [outgoing, incoming] = await Promise.all([
        db.collection('challenges').where({ challenger_user_id: userId })
          .orderBy('created_at', 'desc').limit(50).get(),
        db.collection('challenges').where({ target_user_id: userId })
          .orderBy('created_at', 'desc').limit(50).get(),
      ])
      const byId = new Map()
      for (const item of [...(outgoing.data || []), ...(incoming.data || [])]) {
        byId.set(item.challenge_id, clean(item))
      }
      return [...byId.values()].sort((left, right) => right.created_at.localeCompare(left.created_at))
    },

    async transition({ challengeId, pairKey, nextStatus, respondedAt }) {
      return db.runTransaction(async transaction => {
        const challengeRef = transaction.collection('challenges').doc(digest(challengeId))
        const challenge = await optional(challengeRef)
        if (!challenge) throw new ChallengeError('CHALLENGE_NOT_FOUND')
        if (challenge.status !== 'pending') throw new ChallengeError('CHALLENGE_NOT_PENDING')
        const updated = { ...clean(challenge), status: nextStatus, responded_at: respondedAt }
        await challengeRef.set({ data: updated })
        const pairRef = transaction.collection('challenge_pair_slots').doc(digest(pairKey))
        const slot = await optional(pairRef)
        if (!slot || slot.challenge_id !== challengeId) throw new ChallengeError('CHALLENGE_CONFLICT')
        await pairRef.remove()
        return structuredClone(updated)
      })
    },
  }
}


module.exports = { createCloudChallengeRepository }
