const { deriveMatchRiskSignals } = require('../../cloudfunctions/shared/match-risk-detector')


function context() {
  return {
    match: { projection: {
      match_id: 'match-1', started_at: '2026-09-01T12:00:00.000Z',
      result_proposed_at: '2026-09-01T12:00:20.000Z', table_id: 'table-1',
      participants: { red: 'red', blue: 'blue' }, settings: { format: 'single' },
    } },
    authority: { session: { wristband_identity_status: 'conflict', location_device_switch: false } },
    concurrentMatches: [],
    relatedMatches: Array.from({ length: 5 }, (_, index) => ({
      match_id: `old-${index}`, red_user_id: 'red', blue_user_id: 'blue',
    })),
    reports: [
      { reporter_user_id: 'r1', reason_code: 'suspected_substitution' },
      { reporter_user_id: 'r2', reason_code: 'suspected_substitution' },
    ],
  }
}


test('detector emits independent closed signals without treating repeated opponents as a hard cap', () => {
  expect(deriveMatchRiskSignals(context())).toEqual({
    hardConflicts: [],
    softSignals: [
      'implausible_duration', 'independent_substitution_reports',
      'repeated_opponent', 'wristband_identity_conflict',
    ],
  })
})


test('overlapping user and table intervals are derived from persisted match facts', () => {
  const value = context()
  value.authority.session = {}
  value.reports = []
  value.relatedMatches = []
  value.match.projection.result_proposed_at = '2026-09-01T13:00:00.000Z'
  value.concurrentMatches = [
    { projection: {
      match_id: 'other-user', table_id: 'table-2',
      participants: { red: 'red', blue: 'third' },
      started_at: '2026-09-01T12:30:00.000Z', result_proposed_at: '2026-09-01T13:30:00.000Z',
    } },
    { projection: {
      match_id: 'other-table', table_id: 'table-1',
      participants: { red: 'fourth', blue: 'fifth' },
      started_at: '2026-09-01T11:50:00.000Z', result_proposed_at: '2026-09-01T12:10:00.000Z',
    } },
  ]

  expect(deriveMatchRiskSignals(value).hardConflicts).toEqual([
    'overlapping_table_matches', 'overlapping_user_sessions',
  ])
})


test('server evidence flags hard conflicts but a lone repeated opponent signal remains soft', () => {
  const value = context()
  value.authority.session = { hard_conflicts: ['overlapping_user_sessions'] }
  value.reports = []
  value.relatedMatches = value.relatedMatches.slice(0, 1)
  value.match.projection.result_proposed_at = '2026-09-01T12:10:00.000Z'
  expect(deriveMatchRiskSignals(value)).toEqual({
    hardConflicts: ['overlapping_user_sessions'], softSignals: [],
  })
})
