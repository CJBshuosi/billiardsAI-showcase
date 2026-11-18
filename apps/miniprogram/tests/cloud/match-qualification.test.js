const { evaluateMatchEligibility } = require('../../cloudfunctions/shared/match-qualification')


function confirmedProjection() {
  return {
    match_id: 'match-001', session_id: 'session-001', ranking_scope_id: 'zhoukou-urban',
    venue_id: 'venue-001', table_id: 'table-001', status: 'fact_package_ready',
    event_count: 8, latest_sequence: 8,
    participants: { red: 'user-red', blue: 'user-blue' },
    settings: {
      settings_revision: 1, mode: 'observation', format: 'single', race_target: null,
      first_breaker: 'red', ranking_intent: true,
    },
    settings_confirmations: ['blue', 'red'],
    started_at: '2026-09-01T00:00:05.000Z',
    result: {
      result_revision: 1, winner_role: 'red', red_score: 1, blue_score: 0,
      finish_reason: 'normal',
    },
    result_confirmations: ['blue', 'red'],
    result_proposed_at: '2026-09-01T00:00:06.000Z',
    result_confirmation_deadline_at: '2026-09-02T00:00:06.000Z',
    dispute: null, confirmation_expired_at: null, interruption: null,
  }
}


function approvedAuthority() {
  return {
    schema_version: '1.0',
    configuration_version: 'zhoukou-ranking-v1',
    scope: {
      ranking_scope_id: 'zhoukou-urban', status: 'approved', mode: 'formal', is_test: false,
    },
    venue: { venue_id: 'venue-001', ranking_scope_id: 'zhoukou-urban', status: 'approved' },
    table: { table_id: 'table-001', venue_id: 'venue-001', status: 'approved' },
    session: {
      session_id: 'session-001', table_id: 'table-001',
      environment: 'production', source_kind: 'live_capture',
      evidence_source_kind: 'usb_live',
      device_deployment_status: 'approved', started_at: '2026-09-01T00:00:05.000Z',
      role_bindings: {
        red: { user_id: 'user-red', confirmed_at: '2026-09-01T00:00:03.000Z' },
        blue: { user_id: 'user-blue', confirmed_at: '2026-09-01T00:00:04.000Z' },
      },
    },
  }
}


function approvedDevelopmentMatch() {
  const projection = confirmedProjection()
  projection.ranking_scope_id = 'zhoukou-urban-dev'
  const authority = approvedAuthority()
  authority.scope = {
    ranking_scope_id: 'zhoukou-urban-dev', status: 'approved',
    mode: 'development_acceptance', is_test: true,
  }
  authority.venue.ranking_scope_id = 'zhoukou-urban-dev'
  authority.session.environment = 'development'
  return { projection, authority }
}


test('a confirmed live match on an approved Zhoukou table is eligible', () => {
  expect(evaluateMatchEligibility({
    projection: confirmedProjection(),
    authority: approvedAuthority(),
  })).toEqual({
    schema_version: '1.0',
    match_id: 'match-001',
    result_revision: 1,
    ranking_scope_id: 'zhoukou-urban',
    is_test: false,
    configuration_version: 'zhoukou-ranking-v1',
    status: 'eligible',
    reason_codes: ['ELIGIBLE'],
  })
})


test('a confirmed live match in the isolated development scope is eligible test data', () => {
  const result = evaluateMatchEligibility(approvedDevelopmentMatch())

  expect(result).toMatchObject({
    ranking_scope_id: 'zhoukou-urban-dev',
    is_test: true,
    status: 'eligible',
    reason_codes: ['ELIGIBLE'],
  })
})


test('a development session cannot settle into the formal scope', () => {
  const authority = approvedAuthority()
  authority.session.environment = 'development'

  expect(evaluateMatchEligibility({ projection: confirmedProjection(), authority }))
    .toMatchObject({
      status: 'not_ranked', reason_codes: ['RANKING_ENVIRONMENT_SCOPE_MISMATCH'],
    })
})


test.each([
  [undefined, 'SESSION_SOURCE_UNVERIFIED'],
  ['development_replay', 'SESSION_NOT_USB_LIVE'],
  ['mixed', 'SESSION_NOT_USB_LIVE'],
])('join lifecycle source cannot substitute for immutable evidence source %p', (source, reason) => {
  const authority = approvedAuthority()
  delete authority.session.evidence_source_kind
  if (source !== undefined) authority.session.evidence_source_kind = source
  authority.session.source_kind = 'live_capture'

  expect(evaluateMatchEligibility({ projection: confirmedProjection(), authority }))
    .toMatchObject({ status: 'not_ranked', reason_codes: [reason] })
})


test.each([
  ['awaiting_result_confirmations', 'pending_confirmation', 'RESULT_NOT_CONFIRMED'],
  ['disputed', 'review_pending', 'RESULT_DISPUTED'],
  ['confirmation_expired', 'not_ranked', 'RESULT_CONFIRMATION_EXPIRED'],
  ['interrupted', 'not_ranked', 'MATCH_INTERRUPTED'],
])('%s produces %s instead of a zero-point settlement', (matchStatus, status, reason) => {
  const projection = confirmedProjection()
  projection.status = matchStatus

  expect(evaluateMatchEligibility({ projection, authority: approvedAuthority() }))
    .toMatchObject({ status, reason_codes: [reason] })
})


test.each([
  ['mode', 'assistance', 'ASSISTANCE_MODE'],
  ['ranking_intent', false, 'RANKING_OPTED_OUT'],
])('match setting %s=%p is not ranked', (field, value, reason) => {
  const projection = confirmedProjection()
  projection.settings[field] = value

  expect(evaluateMatchEligibility({ projection, authority: approvedAuthority() }))
    .toMatchObject({ status: 'not_ranked', reason_codes: [reason] })
})


test.each([
  ['scope.status', 'disabled', 'SCOPE_NOT_APPROVED'],
  ['venue.status', 'pending', 'VENUE_NOT_APPROVED'],
  ['table.status', 'disabled', 'TABLE_NOT_APPROVED'],
  ['session.evidence_source_kind', 'development_replay', 'SESSION_NOT_USB_LIVE'],
  ['session.device_deployment_status', 'unapproved', 'DEVICE_DEPLOYMENT_NOT_APPROVED'],
])('%s=%s fails with %s', (path, value, reason) => {
  const authority = approvedAuthority()
  const [parent, field] = path.split('.')
  authority[parent][field] = value

  expect(evaluateMatchEligibility({ projection: confirmedProjection(), authority }))
    .toMatchObject({ status: 'not_ranked', reason_codes: [reason] })
})


test.each([
  ['session.session_id', 'session-other'],
  ['session.table_id', 'table-other'],
  ['venue.ranking_scope_id', 'scope-other'],
  ['table.venue_id', 'venue-other'],
])('%s mismatch is rejected as inconsistent authority', (path, value) => {
  const authority = approvedAuthority()
  const [parent, field] = path.split('.')
  authority[parent][field] = value

  expect(evaluateMatchEligibility({ projection: confirmedProjection(), authority }))
    .toMatchObject({ status: 'not_ranked', reason_codes: ['SESSION_MISMATCH'] })
})


test('missing start evidence or unverified account role bindings are not ranked', () => {
  const noStart = confirmedProjection()
  noStart.started_at = null
  expect(evaluateMatchEligibility({ projection: noStart, authority: approvedAuthority() }))
    .toMatchObject({ status: 'not_ranked', reason_codes: ['START_EVIDENCE_MISSING'] })

  const wrongBinding = approvedAuthority()
  wrongBinding.session.role_bindings.blue.user_id = 'user-other'
  expect(evaluateMatchEligibility({ projection: confirmedProjection(), authority: wrongBinding }))
    .toMatchObject({ status: 'not_ranked', reason_codes: ['ROLE_BINDINGS_UNVERIFIED'] })

  const lateBinding = approvedAuthority()
  lateBinding.session.role_bindings.red.confirmed_at = '2026-09-01T00:00:05.001Z'
  expect(evaluateMatchEligibility({ projection: confirmedProjection(), authority: lateBinding }))
    .toMatchObject({ status: 'not_ranked', reason_codes: ['ROLE_BINDINGS_UNVERIFIED'] })
})


test('invitation origin and wristband coverage cannot change qualification', () => {
  const ordinary = confirmedProjection()
  ordinary.match_origin = 'ordinary'
  const challenge = confirmedProjection()
  challenge.match_origin = 'challenge'
  const withWristbandFailure = approvedAuthority()
  withWristbandFailure.session.wristband_identity_state = 'unavailable'

  const baseline = evaluateMatchEligibility({ projection: ordinary, authority: approvedAuthority() })
  expect(evaluateMatchEligibility({ projection: challenge, authority: approvedAuthority() })).toEqual(baseline)
  expect(evaluateMatchEligibility({ projection: ordinary, authority: withWristbandFailure })).toEqual(baseline)
})


test('qualification evaluation does not mutate its inputs', () => {
  const projection = confirmedProjection()
  const authority = approvedAuthority()
  const before = structuredClone({ projection, authority })

  evaluateMatchEligibility({ projection, authority })

  expect({ projection, authority }).toEqual(before)
})
