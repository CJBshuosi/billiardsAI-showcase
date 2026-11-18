const crypto = require('node:crypto')


class ChallengeError extends Error {
  constructor(code) {
    super(code)
    this.name = 'ChallengeError'
    this.code = code
  }
}


function clone(value) {
  return JSON.parse(JSON.stringify(value))
}


function digest(value) {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex')
}


function canonicalTime(value, code = 'CHALLENGE_INVALID') {
  const milliseconds = Date.parse(value)
  if (!Number.isFinite(milliseconds)) throw new ChallengeError(code)
  return new Date(milliseconds).toISOString()
}


function validateRequest(input, currentTime) {
  if (!input || typeof input !== 'object') throw new ChallengeError('CHALLENGE_INVALID')
  const requiredStrings = [
    input.idempotencyKey,
    input.challengerUserId,
    input.targetUserId,
    input.venueId,
  ]
  if (requiredStrings.some(value => typeof value !== 'string' || value.trim() === '')) {
    throw new ChallengeError('CHALLENGE_INVALID')
  }
  if (input.challengerUserId === input.targetUserId) throw new ChallengeError('CHALLENGE_INVALID')
  if (!['single', 'race'].includes(input.format)) throw new ChallengeError('CHALLENGE_INVALID')
  if (input.format === 'single' && input.raceTarget != null) throw new ChallengeError('CHALLENGE_INVALID')
  if (input.format === 'race' && (!Number.isInteger(input.raceTarget) || input.raceTarget < 2 || input.raceTarget > 99)) {
    throw new ChallengeError('CHALLENGE_INVALID')
  }
  const proposedAt = canonicalTime(input.proposedAt)
  if (Date.parse(proposedAt) <= Date.parse(currentTime)) throw new ChallengeError('CHALLENGE_INVALID')
  return {
    idempotency_key: input.idempotencyKey,
    challenger_user_id: input.challengerUserId,
    target_user_id: input.targetUserId,
    venue_id: input.venueId,
    proposed_at: proposedAt,
    format: input.format,
    race_target: input.format === 'race' ? input.raceTarget : null,
  }
}


function createMemoryChallengeRepository() {
  const challenges = new Map()
  const pendingPairs = new Map()
  const requests = new Map()
  return {
    async findRequest(retryKey) {
      const request = requests.get(retryKey)
      return request ? {
        request_digest: request.request_digest,
        challenge: clone(challenges.get(request.challenge_id)),
      } : null
    },
    async create({ retryKey, requestDigest, pairKey, challenge }) {
      const existing = requests.get(retryKey)
      if (existing) {
        if (existing.request_digest !== requestDigest) {
          throw new ChallengeError('CHALLENGE_IDEMPOTENCY_CONFLICT')
        }
        return clone(challenges.get(existing.challenge_id))
      }
      if (pendingPairs.has(pairKey)) throw new ChallengeError('CHALLENGE_PAIR_PENDING')
      challenges.set(challenge.challenge_id, clone(challenge))
      pendingPairs.set(pairKey, challenge.challenge_id)
      requests.set(retryKey, { request_digest: requestDigest, challenge_id: challenge.challenge_id })
      return clone(challenge)
    },
    async get(challengeId) {
      return challenges.has(challengeId) ? clone(challenges.get(challengeId)) : null
    },
    async transition({ challengeId, pairKey, nextStatus, respondedAt }) {
      const challenge = challenges.get(challengeId)
      if (!challenge) throw new ChallengeError('CHALLENGE_NOT_FOUND')
      if (challenge.status !== 'pending') throw new ChallengeError('CHALLENGE_NOT_PENDING')
      const updated = { ...challenge, status: nextStatus, responded_at: respondedAt }
      challenges.set(challengeId, updated)
      pendingPairs.delete(pairKey)
      return clone(updated)
    },
  }
}


function createChallengeService({
  readProfile, readVenue, now = () => new Date().toISOString(),
  repository = createMemoryChallengeRepository(),
}) {
  if (typeof readProfile !== 'function' || typeof readVenue !== 'function') {
    throw new ChallengeError('CHALLENGE_CONFIGURATION_INVALID')
  }
  for (const method of ['findRequest', 'create', 'get', 'transition']) {
    if (!repository || typeof repository[method] !== 'function') {
      throw new ChallengeError('CHALLENGE_CONFIGURATION_INVALID')
    }
  }

  function readCurrentTime() {
    return canonicalTime(now(), 'CHALLENGE_CLOCK_INVALID')
  }

  function pairKey(firstUserId, secondUserId) {
    return [firstUserId, secondUserId].sort().join(':')
  }

  function requestKey(challengerUserId, idempotencyKey) {
    return `${challengerUserId}:${digest(idempotencyKey)}`
  }

  async function requirePending(challengeId) {
    const challenge = await repository.get(challengeId)
    if (!challenge) throw new ChallengeError('CHALLENGE_NOT_FOUND')
    if (challenge.status !== 'pending') throw new ChallengeError('CHALLENGE_NOT_PENDING')
    return challenge
  }

  function complete(challenge, status, currentTime) {
    return repository.transition({
      challengeId: challenge.challenge_id,
      pairKey: pairKey(challenge.challenger_user_id, challenge.target_user_id),
      nextStatus: status,
      respondedAt: currentTime,
    })
  }

  async function create(input) {
    const currentTime = readCurrentTime()
    const normalized = validateRequest(input, currentTime)
    const retryKey = requestKey(normalized.challenger_user_id, normalized.idempotency_key)
    const requestDigest = digest(normalized)
    const existingRequest = await repository.findRequest(retryKey)
    if (existingRequest) {
      if (existingRequest.request_digest !== requestDigest) {
        throw new ChallengeError('CHALLENGE_IDEMPOTENCY_CONFLICT')
      }
      return clone(existingRequest.challenge)
    }

    const [targetProfile, venue] = await Promise.all([
      readProfile(normalized.target_user_id),
      readVenue(normalized.venue_id),
    ])
    if (!targetProfile || targetProfile.is_public !== true || targetProfile.accept_challenges !== true) {
      throw new ChallengeError('CHALLENGE_TARGET_UNAVAILABLE')
    }
    if (!venue || venue.status !== 'approved' || !venue.ranking_scope_id) {
      throw new ChallengeError('CHALLENGE_VENUE_UNAVAILABLE')
    }

    const normalizedPairKey = pairKey(normalized.challenger_user_id, normalized.target_user_id)
    const challengeId = `challenge_${digest(`${normalized.challenger_user_id}:${normalized.idempotency_key}`).slice(0, 24)}`
    const challenge = {
      schema_version: '1.0',
      challenge_id: challengeId,
      challenger_user_id: normalized.challenger_user_id,
      target_user_id: normalized.target_user_id,
      venue_id: normalized.venue_id,
      ranking_scope_id: venue.ranking_scope_id,
      proposed_at: normalized.proposed_at,
      format: normalized.format,
      race_target: normalized.race_target,
      status: 'pending',
      created_at: currentTime,
      expires_at: new Date(Date.parse(currentTime) + 48 * 60 * 60 * 1000).toISOString(),
      onsite_scan_required: true,
    }
    return repository.create({
      retryKey, requestDigest, pairKey: normalizedPairKey, challenge,
    })
  }

  async function accept({ challengeId, actorUserId }) {
    const challenge = await requirePending(challengeId)
    if (actorUserId !== challenge.target_user_id) throw new ChallengeError('CHALLENGE_UNAUTHORIZED')
    return complete(challenge, 'accepted', readCurrentTime())
  }

  async function reject({ challengeId, actorUserId }) {
    const challenge = await requirePending(challengeId)
    if (actorUserId !== challenge.target_user_id) throw new ChallengeError('CHALLENGE_UNAUTHORIZED')
    return complete(challenge, 'rejected', readCurrentTime())
  }

  async function cancel({ challengeId, actorUserId }) {
    const challenge = await requirePending(challengeId)
    if (actorUserId !== challenge.challenger_user_id) throw new ChallengeError('CHALLENGE_UNAUTHORIZED')
    return complete(challenge, 'cancelled', readCurrentTime())
  }

  async function expire({ challengeId }) {
    const challenge = await requirePending(challengeId)
    const currentTime = readCurrentTime()
    if (Date.parse(currentTime) < Date.parse(challenge.expires_at)) {
      throw new ChallengeError('CHALLENGE_TOO_EARLY')
    }
    return complete(challenge, 'expired', currentTime)
  }

  return { create, accept, reject, cancel, expire }
}


module.exports = { ChallengeError, createChallengeService, createMemoryChallengeRepository }
