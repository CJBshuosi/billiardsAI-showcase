const { ClientApiError } = require('./errors')


async function listUserChallenges({ userId, repository, readProfile, readVenue }) {
  if (typeof userId !== 'string' || userId.length === 0) throw new ClientApiError('CHALLENGE_INVALID')
  const challenges = await repository.listForUser(userId)
  return Promise.all(challenges.map(async challenge => {
    const incoming = challenge.target_user_id === userId
    const counterpartUserId = incoming ? challenge.challenger_user_id : challenge.target_user_id
    const [profile, venue] = await Promise.all([
      readProfile(counterpartUserId), readVenue(challenge.venue_id),
    ])
    return {
      challenge_id: challenge.challenge_id,
      direction: incoming ? 'incoming' : 'outgoing',
      counterpart_user_id: counterpartUserId,
      counterpart_nickname: profile && profile.nickname || '球友',
      venue_id: challenge.venue_id,
      venue_name: venue && venue.display_name || '已约球房',
      proposed_at: challenge.proposed_at,
      format: challenge.format,
      race_target: challenge.race_target,
      status: challenge.status,
      expires_at: challenge.expires_at,
      onsite_scan_required: true,
    }
  }))
}


module.exports = { listUserChallenges }
