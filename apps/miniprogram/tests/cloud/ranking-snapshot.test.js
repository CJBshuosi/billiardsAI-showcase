const {
  createRankingSnapshot,
  createSeason,
  inheritSeasonSkill,
} = require('../../cloudfunctions/shared/ranking-snapshot')
const { initialSkillState } = require('../../cloudfunctions/shared/rating-engine')


const AS_OF = '2026-09-01T12:00:00.000Z'


function player(userId, nickname, rating, override = {}) {
  return {
    user_id: userId,
    public_profile: { is_public: true, nickname },
    long_term: {
      ...initialSkillState(userId), rating, deviation: 100, rated_matches: 5,
      distinct_opponent_ids: ['a', 'b', 'c'], last_rated_at: '2026-08-31T12:00:00.000Z',
    },
    season: null,
    ...override,
  }
}


test('a season lasts exactly 60 days and inherits current long-term skill', () => {
  const season = createSeason({ seasonId: '2026-s01', startsAt: '2026-09-01T00:00:00.000Z' })
  expect(season).toEqual({
    schema_version: '1.0', season_id: '2026-s01',
    starts_at: '2026-09-01T00:00:00.000Z', ends_at: '2026-10-31T00:00:00.000Z',
    duration_days: 60,
  })

  const longTerm = player('user-a', '甲', 1234).long_term
  expect(inheritSeasonSkill({ season, longTerm })).toMatchObject({
    season_id: '2026-s01', user_id: 'user-a', rating: 1234,
    deviation: 100, rated_matches: 0, distinct_opponent_ids: [], inherited_from_rating: 1234,
  })
})


test('public long-term snapshot filters privacy and placement then uses 1,2,2,4 ranks', () => {
  const privatePlayer = player('private', '不公开', 1500)
  privatePlayer.public_profile.is_public = false
  const unplaced = player('unplaced', '未定级', 1600)
  unplaced.long_term.rated_matches = 2
  unplaced.long_term.distinct_opponent_ids = ['a', 'b']
  const snapshot = createRankingSnapshot({
    snapshotId: 'snapshot-1', rankingScopeId: 'zhoukou-urban', kind: 'long_term',
    asOf: AS_OF,
    players: [
      player('u1', '第一', 1300.4),
      player('u2', '并列甲', 1200.49),
      player('u3', '并列乙', 1200.4),
      player('u4', '第四', 1100.2),
      privatePlayer,
      unplaced,
    ],
  })

  expect(snapshot.entries.map(item => [item.rank, item.nickname, item.display_rating]))
    .toEqual([[1, '第一', 1300], [2, '并列甲', 1200], [2, '并列乙', 1200], [4, '第四', 1100]])
  expect(JSON.stringify(snapshot)).not.toContain('不公开')
  expect(JSON.stringify(snapshot)).not.toContain('未定级')
  expect(snapshot).toMatchObject({ snapshot_id: 'snapshot-1', snapshot_version: 1, total_entries: 4 })
})


test('season snapshot requires overall placement and one current-season match', () => {
  const season = createSeason({ seasonId: '2026-s01', startsAt: '2026-09-01T00:00:00.000Z' })
  const eligible = player('eligible', '赛季选手', 1200)
  eligible.season = { ...inheritSeasonSkill({ season, longTerm: eligible.long_term }), rated_matches: 1, rating: 1210 }
  const noSeasonMatch = player('no-match', '本季未赛', 1400)
  noSeasonMatch.season = inheritSeasonSkill({ season, longTerm: noSeasonMatch.long_term })

  const snapshot = createRankingSnapshot({
    snapshotId: 'season-snapshot', rankingScopeId: 'zhoukou-urban',
    kind: 'season', season, asOf: AS_OF, players: [eligible, noSeasonMatch],
  })

  expect(snapshot.entries).toHaveLength(1)
  expect(snapshot.entries[0]).toMatchObject({ user_id: 'eligible', display_rating: 1210 })
})


test('ninety days without a formal match adds a sample hint without changing rank or rating', () => {
  const stale = player('stale', '久未比赛', 1250)
  stale.long_term.last_rated_at = '2026-05-01T00:00:00.000Z'
  const current = player('current', '近期比赛', 1200)
  const snapshot = createRankingSnapshot({
    snapshotId: 'snapshot-stale', rankingScopeId: 'zhoukou-urban', kind: 'long_term',
    asOf: AS_OF, players: [stale, current],
  })

  expect(snapshot.entries[0]).toMatchObject({
    user_id: 'stale', rank: 1, display_rating: 1250, recent_sample_limited: true,
  })
})
