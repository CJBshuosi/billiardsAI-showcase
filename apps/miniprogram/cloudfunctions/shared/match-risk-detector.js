const { HARD_CONFLICTS } = require('./match-risk')


function interval(projection) {
  const start = Date.parse(projection && projection.started_at)
  const end = Date.parse(projection && projection.result_proposed_at)
  return Number.isFinite(start) ? { start, end: Number.isFinite(end) ? end : Infinity } : null
}


function intervalsOverlap(left, right) {
  return left && right && left.start < right.end && right.start < left.end
}


function deriveMatchRiskSignals({
  match, authority, relatedMatches = [], concurrentMatches = [], reports = [],
}) {
  const projection = match && match.projection
  if (!projection || !projection.participants || !projection.settings) {
    throw new Error('MATCH_RISK_EVIDENCE_INVALID')
  }
  const session = authority && authority.session || {}
  const hardConflicts = Array.isArray(session.hard_conflicts)
    ? [...new Set(session.hard_conflicts.filter(code => HARD_CONFLICTS.has(code)))].sort()
    : []
  const hard = new Set(hardConflicts)
  const currentInterval = interval(projection)
  const currentUsers = new Set(Object.values(projection.participants))
  for (const item of concurrentMatches) {
    const other = item && item.projection || item
    if (!other || other.match_id === projection.match_id
        || !intervalsOverlap(currentInterval, interval(other))) continue
    if (other.table_id === projection.table_id) hard.add('overlapping_table_matches')
    const otherUsers = other.participants ? Object.values(other.participants) : []
    if (otherUsers.some(userId => currentUsers.has(userId))) hard.add('overlapping_user_sessions')
  }
  const soft = new Set()
  const started = Date.parse(projection.started_at)
  const finished = Date.parse(projection.result_proposed_at)
  if (Number.isFinite(started) && Number.isFinite(finished)) {
    const minimum = projection.settings.format === 'race'
      ? Math.max(60, projection.settings.race_target * 20) * 1000 : 30 * 1000
    if (finished - started < minimum) soft.add('implausible_duration')
  }
  if (session.wristband_identity_status === 'conflict') soft.add('wristband_identity_conflict')
  if (session.location_device_switch === true) soft.add('location_device_switch')

  const { red, blue } = projection.participants
  const pairMatches = relatedMatches.filter(item => item.match_id !== projection.match_id
    && ((item.red_user_id === red && item.blue_user_id === blue)
      || (item.red_user_id === blue && item.blue_user_id === red)))
  if (pairMatches.length >= 5) soft.add('repeated_opponent')
  const substitutionReporters = new Set(reports
    .filter(item => item.reason_code === 'suspected_substitution')
    .map(item => item.reporter_user_id))
  if (substitutionReporters.size >= 2) soft.add('independent_substitution_reports')
  return { hardConflicts: [...hard].sort(), softSignals: [...soft].sort() }
}


function createCloudMatchRiskDetector(db) {
  return async ({ match, authority }) => {
    const projection = match.projection
    const { red, blue } = projection.participants
    const recent = query => db.collection('matches').where(query)
      .orderBy('started_at', 'desc').limit(50).get()
    const [redAsRed, redAsBlue, blueAsRed, blueAsBlue, sameTable, reports] = await Promise.all([
      recent({ red_user_id: red }), recent({ blue_user_id: red }),
      recent({ red_user_id: blue }), recent({ blue_user_id: blue }),
      recent({ table_id: projection.table_id }),
      db.collection('user_match_reports').where({ match_id: projection.match_id }).limit(50).get(),
    ])
    const documents = new Map()
    for (const item of [
      ...(redAsRed.data || []), ...(redAsBlue.data || []),
      ...(blueAsRed.data || []), ...(blueAsBlue.data || []), ...(sameTable.data || []),
    ]) documents.set(item.match_id, item)
    const concurrentMatches = [...documents.values()]
    const relatedMatches = concurrentMatches.map(item => ({
      match_id: item.match_id, red_user_id: item.red_user_id, blue_user_id: item.blue_user_id,
    }))
    return deriveMatchRiskSignals({
      match, authority, relatedMatches, concurrentMatches,
      reports: reports.data || [],
    })
  }
}


module.exports = { createCloudMatchRiskDetector, deriveMatchRiskSignals }
