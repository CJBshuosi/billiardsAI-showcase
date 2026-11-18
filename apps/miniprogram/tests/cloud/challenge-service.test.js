const { ChallengeError, createChallengeService } = require('../../cloudfunctions/shared/challenge-service')


const NOW = '2026-09-01T12:00:00.000Z'


function dependencies() {
  const profiles = new Map([
    ['user-a', { user_id: 'user-a', is_public: true, accept_challenges: true }],
    ['user-b', { user_id: 'user-b', is_public: true, accept_challenges: true }],
    ['private', { user_id: 'private', is_public: false, accept_challenges: true }],
    ['closed', { user_id: 'closed', is_public: true, accept_challenges: false }],
  ])
  return {
    readProfile: async userId => profiles.get(userId) || null,
    readVenue: async venueId => venueId === 'venue-approved'
      ? { venue_id: venueId, status: 'approved', ranking_scope_id: 'zhoukou-urban' } : null,
  }
}


function request(override = {}) {
  return {
    idempotencyKey: 'request-001', challengerUserId: 'user-a', targetUserId: 'user-b',
    venueId: 'venue-approved', proposedAt: '2026-09-02T11:00:00.000Z',
    format: 'single', raceTarget: null, ...override,
  }
}


test('challenge creation has a 48-hour response deadline and no match authority', async () => {
  const service = createChallengeService({ ...dependencies(), now: () => NOW })

  const challenge = await service.create(request())

  expect(challenge).toMatchObject({
    challenger_user_id: 'user-a', target_user_id: 'user-b', venue_id: 'venue-approved',
    format: 'single', race_target: null, status: 'pending',
    created_at: NOW, expires_at: '2026-09-03T12:00:00.000Z',
    onsite_scan_required: true,
  })
  expect(challenge).not.toHaveProperty('match_id')
  expect(challenge).not.toHaveProperty('rating_event_id')
})


test.each([
  ['private', 'CHALLENGE_TARGET_UNAVAILABLE'],
  ['closed', 'CHALLENGE_TARGET_UNAVAILABLE'],
])('target %s cannot receive a challenge', async (targetUserId, code) => {
  const service = createChallengeService({ ...dependencies(), now: () => NOW })
  await expect(service.create(request({ targetUserId }))).rejects.toMatchObject({ code })
})


test('only approved venues and supported match intentions are accepted', async () => {
  const service = createChallengeService({ ...dependencies(), now: () => NOW })
  await expect(service.create(request({ venueId: 'venue-unknown' })))
    .rejects.toMatchObject({ code: 'CHALLENGE_VENUE_UNAVAILABLE' })
  await expect(service.create(request({ format: 'race', raceTarget: 1 })))
    .rejects.toMatchObject({ code: 'CHALLENGE_INVALID' })
})


test('one unordered pair has only one pending challenge and request retries are idempotent', async () => {
  const service = createChallengeService({ ...dependencies(), now: () => NOW })
  const first = await service.create(request())
  await expect(service.create(request())).resolves.toEqual(first)
  await expect(service.create(request({
    idempotencyKey: 'request-002', challengerUserId: 'user-b', targetUserId: 'user-a',
  }))).rejects.toMatchObject({ code: 'CHALLENGE_PAIR_PENDING' })
})


test.each([
  ['accept', 'user-b', 'accepted'],
  ['reject', 'user-b', 'rejected'],
  ['cancel', 'user-a', 'cancelled'],
])('%s enforces the correct actor and reaches %s', async (action, actorUserId, status) => {
  const service = createChallengeService({ ...dependencies(), now: () => NOW })
  const challenge = await service.create(request())

  const result = await service[action]({ challengeId: challenge.challenge_id, actorUserId })

  expect(result).toMatchObject({ status, onsite_scan_required: true })
  expect(result).not.toHaveProperty('match_id')
})


test('wrong actors cannot respond and expiry occurs only when due', async () => {
  let current = NOW
  const service = createChallengeService({ ...dependencies(), now: () => current })
  const challenge = await service.create(request())
  await expect(service.accept({ challengeId: challenge.challenge_id, actorUserId: 'user-a' }))
    .rejects.toMatchObject({ code: 'CHALLENGE_UNAUTHORIZED' })
  await expect(service.expire({ challengeId: challenge.challenge_id }))
    .rejects.toMatchObject({ code: 'CHALLENGE_TOO_EARLY' })

  current = '2026-09-03T12:00:00.000Z'
  await expect(service.expire({ challengeId: challenge.challenge_id }))
    .resolves.toMatchObject({ status: 'expired' })
})


test('stable challenge errors expose no profile records', () => {
  expect(new ChallengeError('CHALLENGE_TARGET_UNAVAILABLE')).toMatchObject({
    message: 'CHALLENGE_TARGET_UNAVAILABLE', code: 'CHALLENGE_TARGET_UNAVAILABLE',
  })
})
