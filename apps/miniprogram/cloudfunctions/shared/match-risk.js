const HARD_CONFLICTS = new Set([
  'duplicate_match_result',
  'overlapping_user_sessions',
  'overlapping_table_matches',
  'invalid_race_completion',
  'participant_changed_after_start',
  'revoked_or_test_reuse',
  'result_version_mismatch',
])

const SOFT_SIGNALS = new Set([
  'concentrated_one_way_losses',
  'new_account_loss_cluster',
  'implausible_duration',
  'repeated_trace_similarity',
  'loser_confirmation_avoidance',
  'wristband_identity_conflict',
  'location_device_switch',
  'independent_substitution_reports',
  'repeated_opponent',
  'technical_duplicate',
])


class MatchRiskError extends Error {
  constructor(code) {
    super(code)
    this.name = 'MatchRiskError'
    this.code = code
  }
}


function invalid() {
  throw new MatchRiskError('MATCH_RISK_INVALID')
}


function evaluateMatchRisk({ hardConflicts, softSignals }) {
  if (!Array.isArray(hardConflicts) || !Array.isArray(softSignals)
      || hardConflicts.some(item => !HARD_CONFLICTS.has(item))
      || softSignals.some(item => !SOFT_SIGNALS.has(item))) invalid()
  const hard = [...new Set(hardConflicts)].sort()
  const soft = [...new Set(softSignals.filter(item => item !== 'technical_duplicate'))].sort()
  if (hard.length > 0) {
    return {
      status: 'blocked', hard_conflicts: hard, soft_signals: soft, public_status: 'not_ranked',
    }
  }
  if (soft.length >= 2) {
    return {
      status: 'review_hold', hard_conflicts: [], soft_signals: soft,
      public_status: 'review_pending',
    }
  }
  if (soft.length === 1) {
    return {
      status: 'clear_with_signals', hard_conflicts: [], soft_signals: soft,
      public_status: 'normal',
    }
  }
  return { status: 'clear', hard_conflicts: [], soft_signals: [], public_status: 'normal' }
}


module.exports = {
  HARD_CONFLICTS,
  MatchRiskError,
  SOFT_SIGNALS,
  evaluateMatchRisk,
}
