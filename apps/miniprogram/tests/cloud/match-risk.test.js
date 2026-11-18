const {
  MatchRiskError,
  evaluateMatchRisk,
} = require('../../cloudfunctions/shared/match-risk')


test('no signals clears the match for settlement', () => {
  expect(evaluateMatchRisk({ hardConflicts: [], softSignals: [] })).toEqual({
    status: 'clear',
    hard_conflicts: [],
    soft_signals: [],
    public_status: 'normal',
  })
})


test('hard factual conflicts block settlement without declaring cheating', () => {
  const result = evaluateMatchRisk({
    hardConflicts: ['result_version_mismatch'],
    softSignals: [],
  })

  expect(result).toEqual({
    status: 'blocked',
    hard_conflicts: ['result_version_mismatch'],
    soft_signals: [],
    public_status: 'not_ranked',
  })
  expect(JSON.stringify(result)).not.toMatch(/cheat|作弊/iu)
})


test('one soft signal cannot hold, ban or revoke a match', () => {
  expect(evaluateMatchRisk({
    hardConflicts: [], softSignals: ['wristband_identity_conflict'],
  })).toEqual({
    status: 'clear_with_signals',
    hard_conflicts: [],
    soft_signals: ['wristband_identity_conflict'],
    public_status: 'normal',
  })
})


test('two independent soft signal categories create a neutral review hold', () => {
  expect(evaluateMatchRisk({
    hardConflicts: [],
    softSignals: ['repeated_opponent', 'implausible_duration', 'repeated_opponent'],
  })).toEqual({
    status: 'review_hold',
    hard_conflicts: [],
    soft_signals: ['implausible_duration', 'repeated_opponent'],
    public_status: 'review_pending',
  })
})


test('repeated opponents alone never create a fixed match cap', () => {
  expect(evaluateMatchRisk({
    hardConflicts: [],
    softSignals: Array.from({ length: 20 }, () => 'repeated_opponent'),
  })).toMatchObject({ status: 'clear_with_signals', public_status: 'normal' })
})


test('technical duplicate delivery is excluded from user risk', () => {
  expect(evaluateMatchRisk({
    hardConflicts: [],
    softSignals: ['technical_duplicate', 'technical_duplicate'],
  })).toEqual({
    status: 'clear', hard_conflicts: [], soft_signals: [], public_status: 'normal',
  })
})


test('unknown signals fail closed instead of becoming accusations', () => {
  expect(() => evaluateMatchRisk({ hardConflicts: [], softSignals: ['looks_suspicious'] }))
    .toThrow(expect.objectContaining({ name: 'MatchRiskError', code: 'MATCH_RISK_INVALID' }))
  expect(new MatchRiskError('MATCH_RISK_INVALID')).toMatchObject({
    message: 'MATCH_RISK_INVALID', code: 'MATCH_RISK_INVALID',
  })
})
