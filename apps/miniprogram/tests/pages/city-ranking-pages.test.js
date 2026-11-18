function pageAt(name) {
  let definition
  global.Page = value => { definition = value }
  jest.isolateModules(() => require(`../../miniprogram/pages/${name}/index`))
  return { ...definition, data: { ...definition.data }, setData(update) { Object.assign(this.data, update) } }
}


afterEach(() => { delete global.Page; delete global.wx; delete global.getApp })


test('ranking page loads a published snapshot for a signed-in user', async () => {
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async ({ data }) => ({ result: {
    ok: true,
    data: {
      availability: 'published', snapshot_id: 'snapshot-1', snapshot_version: 1,
      kind: data.payload.kind, as_of: '2026-09-01T12:00:00.000Z', total_entries: 1,
      ranking_scope: {
        ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
        mode: 'development_acceptance', is_test: true,
      },
      entries: [{ rank: 1, user_id: 'friend', nickname: '周口第一杆', display_rating: 1288 }],
      my_position: null,
      my_status: { status: 'unplaced', rated_matches: 2, distinct_opponents: 2, required_matches: 3, required_opponents: 3 },
    },
  } }) } }
  const page = pageAt('ranking')
  await page.onShow()
  expect(page.data.available).toBe(true)
  expect(page.data.entries[0]).toMatchObject({ nickname: '周口第一杆', display_rating: 1288 })
  expect(page.data.myStatus).toMatchObject({ status: 'unplaced', rated_matches: 2 })
  expect(page.data.myStatus).not.toHaveProperty('display_rating')
  expect(page.data).toMatchObject({
    isTest: true, rankingTitle: '周口开发测试榜',
    scopeNotice: '仅用于开发验收，不代表周口正式排名',
  })
})


test('ranking page keeps the approved unavailable state when no snapshot exists', async () => {
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async () => ({ result: { ok: true, data: {
    availability: 'unavailable', reason_code: 'RANKING_NOT_PUBLISHED', entries: [],
    my_position: null,
    my_status: { status: 'unplaced', rated_matches: 0, distinct_opponents: 0, required_matches: 3, required_opponents: 3 },
  } } }) } }
  const page = pageAt('ranking')
  await page.onShow()
  expect(page.data.available).toBe(false)
  expect(page.data.entries).toEqual([])
  expect(page.data.reasonCode).toBe('RANKING_NOT_PUBLISHED')
})


test('ranking page appends the next page only from the same snapshot version', async () => {
  const requests = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async ({ data }) => {
    requests.push(data)
    const page = data.payload.page
    return { result: { ok: true, data: {
      availability: 'published', snapshot_id: 'snapshot-1', snapshot_version: 1,
      kind: 'long_term', page, page_size: 1, has_more: page === 1,
      as_of: '2026-09-01T12:00:00.000Z', total_entries: 2,
      entries: [{ rank: page, user_id: `friend-${page}`, nickname: `球友${page}`, display_rating: 1301 - page }],
      my_position: null, my_status: { status: 'placed', display_rating: 1100 },
    } } }
  } } }
  const page = pageAt('ranking')
  await page.onShow()
  await page.loadMore()

  expect(requests.map(item => item.payload.page)).toEqual([1, 2])
  expect(page.data.entries.map(item => item.user_id)).toEqual(['friend-1', 'friend-2'])
  expect(page.data.hasMore).toBe(false)
})


test('ranking rows navigate to a public player page without carrying scores in the URL', () => {
  const paths = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { navigateTo: value => paths.push(value.url), cloud: { callFunction: jest.fn() } }
  const page = pageAt('ranking')
  page.setData({ snapshotId: 'snapshot-9' })
  page.openPlayer({ currentTarget: { dataset: { userId: 'friend-1' } } })
  expect(paths).toEqual(['/pages/ranking-player/index?user_id=friend-1&snapshot_id=snapshot-9'])
  expect(paths[0]).not.toContain('rating')
})


test('public player page shows unplaced status without hidden rating', async () => {
  const requests = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async ({ data }) => { requests.push(data); return ({ result: { ok: true, data: {
    user_id: 'friend', nickname: '新球友', accept_challenges: true,
    ranking_status: 'unplaced', rated_matches: 2, distinct_opponents: 2,
    required_matches: 3, required_opponents: 3,
    ranking_scope: {
      ranking_scope_id: 'zhoukou-urban-dev', display_name: '周口开发测试榜',
      mode: 'development_acceptance', is_test: true,
    },
  } } }) } } }
  const page = pageAt('ranking-player')
  await page.onLoad({ user_id: 'friend', snapshot_id: 'snapshot-9' })
  expect(page.data.player.ranking_status).toBe('unplaced')
  expect(page.data.player).not.toHaveProperty('display_rating')
  expect(page.data).toMatchObject({
    isTest: true, rankingTitle: '周口开发测试榜',
    scopeNotice: '仅用于开发验收，不代表周口正式排名',
  })
  expect(requests[0]).toMatchObject({
    action: 'ranking.player.get', payload: { user_id: 'friend', snapshot_id: 'snapshot-9' },
  })
})

test('ranking settings save public visibility and challenge permission separately', async () => {
  const requests = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async ({ data }) => {
    requests.push(data)
    if (data.action === 'ranking.profile.get') return { result: { ok: true, data: null } }
    return { result: { ok: true, data: {
      nickname: data.payload.nickname, is_public: data.payload.is_public,
      accept_challenges: data.payload.accept_challenges,
    } } }
  } } }
  const page = pageAt('ranking-settings')
  await page.onLoad()
  page.updateNickname({ detail: { value: '周口新球友' } })
  page.togglePublic({ detail: { value: true } })
  page.toggleChallenges({ detail: { value: false } })
  await page.save()
  expect(requests.at(-1)).toMatchObject({
    action: 'ranking.profile.save',
    payload: { nickname: '周口新球友', is_public: true, accept_challenges: false },
  })
  expect(page.data.saved).toBe(true)
})

test('challenge page submits a neutral invitation with an approved venue and no rating fields', async () => {
  const requests = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = {
    cloud: { callFunction: async ({ data }) => {
      requests.push(data)
      if (data.action === 'ranking.player.get') return { result: { ok: true, data: {
        user_id: 'friend', nickname: '球友 0316', ranking_status: 'placed',
        display_rating: 1200, rank: 8, accept_challenges: true,
      } } }
      if (data.action === 'ranking.venues.list') return { result: { ok: true, data: [
        { venue_id: 'venue-1', display_name: '周口测试球房' },
      ] } }
      return { result: { ok: true, data: { challenge_id: 'challenge-1', status: 'pending', onsite_scan_required: true } } }
    } },
  }
  const page = pageAt('challenge')
  await page.onLoad({ user_id: 'friend' })
  page.setData({ venueIndex: 0, proposedDate: '2026-09-02', proposedTime: '19:30' })
  await page.submit()
  const payload = requests.at(-1).payload
  expect(requests.at(-1).action).toBe('challenge.create')
  expect(payload).toMatchObject({
    target_user_id: 'friend', venue_id: 'venue-1', format: 'single', race_target: null,
    proposed_at: '2026-09-02T11:30:00.000Z',
  })
  expect(payload).not.toHaveProperty('rating_weight')
  expect(page.data.submitted).toBe(true)
})

test('challenge inbox allows only the role-appropriate pending action', async () => {
  const requests = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async ({ data }) => {
    requests.push(data)
    if (data.action === 'challenge.list') return { result: { ok: true, data: [{
      challenge_id: 'c-1', direction: 'incoming', counterpart_nickname: '球友甲',
      venue_name: '周口测试球房', proposed_at: '2026-09-02T11:30:00.000Z',
      format: 'single', race_target: null, status: 'pending', onsite_scan_required: true,
    }] } }
    return { result: { ok: true, data: { challenge_id: 'c-1', status: 'accepted' } } }
  } } }
  const page = pageAt('challenges')
  await page.onShow()
  await page.act({ currentTarget: { dataset: { id: 'c-1', action: 'accept' } } })
  expect(requests.some(item => item.action === 'challenge.accept'
    && item.payload.challenge_id === 'c-1')).toBe(true)
})


test('an accepted challenge routes to onsite table scanning without creating a match', () => {
  const paths = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { navigateTo: input => paths.push(input.url), cloud: { callFunction: jest.fn() } }
  const page = pageAt('challenges')

  page.openScanner()

  expect(paths).toEqual(['/pages/scan/index'])
})

test('pending result page confirms the exact candidate result without showing estimated points', async () => {
  const requests = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async ({ data }) => {
    requests.push(data)
    if (data.action === 'ranking.results.pending') return { result: { ok: true, data: [{
      match_id: 'match-1', status: 'awaiting_result_confirmations', actor_role: 'red',
      result: { winner_role: 'red', red_score: 1, blue_score: 0, result_revision: 1 },
      actor_result_confirmed: false, result_confirmation_deadline_at: '2026-09-02T12:00:00.000Z',
    }] } }
    return { result: { ok: true, data: { match_id: 'match-1', status: 'fact_package_ready' } } }
  } } }
  const page = pageAt('pending-results')
  await page.onShow()
  expect(JSON.stringify(page.data.matches)).not.toContain('rating_delta')
  await page.confirm({ currentTarget: { dataset: { id: 'match-1' } } })
  expect(requests.at(-1)).toEqual(expect.objectContaining({
    action: 'ranking.result.confirm', payload: { match_id: 'match-1' },
  }))
})


test('session report shows neutral review status and lets the participant appeal this match', async () => {
  const requests = []
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = {
    showActionSheet({ success }) { success({ tapIndex: 0 }) },
    cloud: { callFunction: async ({ data }) => {
      requests.push(data)
      if (data.action === 'sessions.get') return { result: { ok: true, data: {
        session_id: 'session-1', status: 'ended', shots: [], limitations: [],
      } } }
      if (data.action === 'clips.list') return { result: { ok: true, data: [] } }
      if (data.action === 'ranking.match.get') return { result: { ok: true, data: {
        match_id: 'match-1', status: 'fact_package_ready',
        result: { winner_role: 'red', red_score: 1, blue_score: 0 },
        ranking_outcome: { status: 'review_pending', can_appeal: true, actor_appealed: false },
      } } }
      return { result: { ok: true, data: { status: 'review_hold' } } }
    } },
  }
  const page = pageAt('session-detail')
  await page.onLoad({ session_id: 'session-1' })
  await new Promise(setImmediate)

  expect(page.data.rankingOutcome).toEqual({
    status: 'review_pending', can_appeal: true, actor_appealed: false,
  })
  await page.appealRankingReview()
  await new Promise(setImmediate)
  expect(requests.at(-1)).toMatchObject({
    action: 'ranking.review.appeal',
    payload: { match_id: 'match-1', reason_code: 'result_is_valid' },
  })
  expect(page.data.rankingOutcome.actor_appealed).toBe(true)
})


test('session report replaces the old candidate score with the append-only corrected result', async () => {
  global.getApp = () => ({ globalData: { currentUser: { user_id: 'me' } } })
  global.wx = { cloud: { callFunction: async ({ data }) => {
    if (data.action === 'sessions.get') return { result: { ok: true, data: {
      session_id: 'session-1', status: 'ended', shots: [], limitations: [],
    } } }
    if (data.action === 'clips.list') return { result: { ok: true, data: [] } }
    return { result: { ok: true, data: {
      match_id: 'match-1', status: 'fact_package_ready',
      result: { winner_role: 'red', red_score: 1, blue_score: 0 },
      ranking_outcome: {
        status: 'corrected', can_appeal: true, actor_appealed: false,
        corrected_result: { winner_role: 'blue', red_score: 0, blue_score: 1 },
      },
    } } }
  } } }
  const page = pageAt('session-detail')
  await page.onLoad({ session_id: 'session-1' })

  expect(page.data.rankingResult).toEqual({ winner_role: 'blue', red_score: 0, blue_score: 1 })
})
