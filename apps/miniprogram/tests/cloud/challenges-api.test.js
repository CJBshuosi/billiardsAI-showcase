const { listUserChallenges } = require('../../cloudfunctions/client-api/lib/challenges')


test('challenge inbox exposes direction and counterpart nickname without match or rating authority', async () => {
  const result = await listUserChallenges({
    userId: 'me',
    repository: { async listForUser() { return [
      {
        challenge_id: 'c-in', challenger_user_id: 'friend', target_user_id: 'me',
        status: 'pending', venue_id: 'venue-1', proposed_at: '2026-09-02T11:30:00.000Z',
        format: 'single', race_target: null, expires_at: '2026-09-03T12:00:00.000Z',
        onsite_scan_required: true,
      },
      {
        challenge_id: 'c-out', challenger_user_id: 'me', target_user_id: 'other',
        status: 'accepted', venue_id: 'venue-1', proposed_at: '2026-09-04T11:30:00.000Z',
        format: 'race', race_target: 3, expires_at: '2026-09-03T13:00:00.000Z',
        onsite_scan_required: true,
      },
    ] } },
    readProfile: async userId => ({ user_id: userId, nickname: `昵称-${userId}` }),
    readVenue: async () => ({ display_name: '周口测试球房' }),
  })
  expect(result).toEqual([
    expect.objectContaining({ challenge_id: 'c-in', direction: 'incoming', counterpart_nickname: '昵称-friend' }),
    expect.objectContaining({ challenge_id: 'c-out', direction: 'outgoing', counterpart_nickname: '昵称-other' }),
  ])
  expect(JSON.stringify(result)).not.toContain('rating_event_id')
  expect(JSON.stringify(result)).not.toContain('match_id')
})
