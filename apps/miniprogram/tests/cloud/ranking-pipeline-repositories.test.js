const crypto = require('crypto')

const {
  createRankingAuthorityRepository,
  createRankingDecisionRepositories,
} = require('../../cloudfunctions/shared/ranking-pipeline-repositories')


const digest = value => crypto.createHash('sha256').update(value).digest('hex')


function fakeDatabase(seed = {}) {
  const names = [
    'ranking_scopes', 'ranking_venues', 'ranking_tables', 'field_live_sessions',
    'sessions', 'session_role_bindings', 'match_qualification_decisions',
    'match_risk_assessments',
  ]
  const maps = Object.fromEntries(names.map(name => [name, new Map(
    Object.entries(seed[name] || {}).map(([id, value]) => [id, { _id: id, ...structuredClone(value) }]),
  )]))
  const collection = name => ({
    doc(id) { return {
      async get() { return { data: maps[name].get(id) || null } },
      async set({ data }) { maps[name].set(id, { _id: id, ...structuredClone(data) }) },
    } },
  })
  return { maps, collection, async runTransaction(callback) { return callback({ collection }) } }
}


test('authority repository joins approved scope, venue, table, session and role bindings', async () => {
  const db = fakeDatabase({
    ranking_scopes: { [digest('zhoukou-urban')]: {
      ranking_scope_id: 'zhoukou-urban', status: 'approved', configuration_version: 'v1',
    } },
    ranking_venues: { [digest('venue-1')]: {
      venue_id: 'venue-1', ranking_scope_id: 'zhoukou-urban', status: 'approved',
    } },
    ranking_tables: { [digest('table-1')]: {
      table_id: 'table-1', venue_id: 'venue-1', status: 'approved',
      device_id: 'device-1', device_deployment_status: 'approved',
    } },
    field_live_sessions: { 'session-1': {
      session_id: 'session-1', table_id: 'table-1', source_kind: 'live_capture',
      environment: 'production', device_id: 'device-1',
      started_at: '2026-09-01T10:00:00.000Z',
    } },
    sessions: { 'session-1': {
      session_id: 'session-1', sync_status: 'published',
      _sync_publication: {
        table_id: 'table-1', device_id: 'device-1', content_sha256: 'a'.repeat(64),
      },
      _evidence_source_kind: 'usb_live', limitations: [],
    } },
    session_role_bindings: {
      [digest('session-1:red')]: {
        session_id: 'session-1', role: 'red', user_id: 'red', confirmed_at: '2026-09-01T09:59:00.000Z',
      },
      [digest('session-1:blue')]: {
        session_id: 'session-1', role: 'blue', user_id: 'blue', confirmed_at: '2026-09-01T09:59:30.000Z',
      },
    },
  })
  const authority = await createRankingAuthorityRepository(db).readForMatch({
    ranking_scope_id: 'zhoukou-urban', venue_id: 'venue-1', table_id: 'table-1', session_id: 'session-1',
  })

  expect(authority).toMatchObject({
    configuration_version: 'v1',
    scope: { ranking_scope_id: 'zhoukou-urban' }, venue: { venue_id: 'venue-1' },
    table: { table_id: 'table-1' },
    session: {
      evidence_source_kind: 'usb_live',
      role_bindings: { red: { user_id: 'red' }, blue: { user_id: 'blue' } },
    },
  })
})


test('authority ignores spoofed lifecycle source and derives replay from published evidence', async () => {
  const db = fakeDatabase({
    field_live_sessions: { 'session-1': {
      session_id: 'session-1', table_id: 'table-1', source_kind: 'live_capture',
      environment: 'production', device_id: 'device-1',
    } },
    sessions: { 'session-1': {
      session_id: 'session-1', sync_status: 'published',
      _sync_publication: {
        table_id: 'table-1', device_id: 'device-1', content_sha256: 'a'.repeat(64),
      },
      _evidence_source_kind: 'development_replay',
      limitations: ['development_replay_media'],
    } },
  })

  const authority = await createRankingAuthorityRepository(db).readForMatch({
    ranking_scope_id: 'scope', venue_id: 'venue', table_id: 'table-1',
    session_id: 'session-1',
  })

  expect(authority.session).toMatchObject({
    source_kind: 'live_capture', evidence_source_kind: 'development_replay',
  })
})


test('qualification and risk decisions are idempotent and conflicting rewrites fail closed', async () => {
  const db = fakeDatabase()
  const repositories = createRankingDecisionRepositories(db, {
    now: () => '2026-09-01T12:00:00.000Z',
    detectRiskSignals: async () => ({ hardConflicts: [], softSignals: ['repeated_opponent'] }),
  })
  const qualification = {
    schema_version: '1.0', match_id: 'match-1', result_revision: 1,
    ranking_scope_id: 'zhoukou-urban', configuration_version: 'v1',
    status: 'eligible', reason_codes: ['ELIGIBLE'],
  }
  await repositories.qualificationRepository.save(qualification)
  await repositories.qualificationRepository.save(structuredClone(qualification))
  expect(db.maps.match_qualification_decisions.size).toBe(1)
  await expect(repositories.qualificationRepository.save({
    ...qualification, status: 'not_ranked', reason_codes: ['RANKING_OPTED_OUT'],
  })).rejects.toMatchObject({ code: 'RANKING_DECISION_CONFLICT' })

  await expect(repositories.riskRepository.collectSignals({ match: {}, authority: {} }))
    .resolves.toEqual({ hardConflicts: [], softSignals: ['repeated_opponent'] })
  await repositories.riskRepository.save({
    schema_version: '1.0', match_id: 'match-1', result_revision: 1,
    ranking_scope_id: 'zhoukou-urban', status: 'clear_with_signals',
    hard_conflicts: [], soft_signals: ['repeated_opponent'], public_status: 'normal',
  })
  expect(db.maps.match_risk_assessments.size).toBe(1)
})
