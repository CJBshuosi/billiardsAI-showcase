const { assertRankingScopePolicy } = require('./ranking-scope-policy')


function decision(projection, authority, status, reasonCode) {
  return {
    schema_version: '1.0',
    match_id: projection.match_id,
    result_revision: projection.result ? projection.result.result_revision : null,
    ranking_scope_id: projection.ranking_scope_id,
    is_test: authority && authority.scope && authority.scope.is_test,
    configuration_version: authority && authority.configuration_version || null,
    status,
    reason_codes: [reasonCode],
  }
}


function validTime(value) {
  return typeof value === 'string' && Number.isFinite(Date.parse(value))
}


function verifiedBindings(projection, session) {
  const bindings = session && session.role_bindings
  if (!bindings || !bindings.red || !bindings.blue || !validTime(projection.started_at)) return false
  return bindings.red.user_id === projection.participants.red
    && bindings.blue.user_id === projection.participants.blue
    && projection.participants.red !== projection.participants.blue
    && validTime(bindings.red.confirmed_at)
    && validTime(bindings.blue.confirmed_at)
    && Date.parse(bindings.red.confirmed_at) <= Date.parse(projection.started_at)
    && Date.parse(bindings.blue.confirmed_at) <= Date.parse(projection.started_at)
}


function evaluateMatchEligibility({ projection, authority }) {
  if (projection.status === 'disputed') {
    return decision(projection, authority, 'review_pending', 'RESULT_DISPUTED')
  }
  if (projection.status === 'confirmation_expired') {
    return decision(projection, authority, 'not_ranked', 'RESULT_CONFIRMATION_EXPIRED')
  }
  if (projection.status === 'interrupted') {
    return decision(projection, authority, 'not_ranked', 'MATCH_INTERRUPTED')
  }
  if (projection.status !== 'fact_package_ready') {
    return decision(projection, authority, 'pending_confirmation', 'RESULT_NOT_CONFIRMED')
  }
  if (projection.settings.mode !== 'observation') {
    return decision(projection, authority, 'not_ranked', 'ASSISTANCE_MODE')
  }
  if (!projection.settings.ranking_intent) {
    return decision(projection, authority, 'not_ranked', 'RANKING_OPTED_OUT')
  }

  const scope = authority && authority.scope
  const venue = authority && authority.venue
  const table = authority && authority.table
  const session = authority && authority.session
  if (!scope || scope.status !== 'approved' || scope.ranking_scope_id !== projection.ranking_scope_id) {
    return decision(projection, authority, 'not_ranked', 'SCOPE_NOT_APPROVED')
  }
  try {
    assertRankingScopePolicy({ environment: session && session.environment, scope })
  } catch (error) {
    return decision(projection, authority, 'not_ranked', error.message)
  }
  if (!venue || venue.status !== 'approved' || venue.venue_id !== projection.venue_id) {
    return decision(projection, authority, 'not_ranked', 'VENUE_NOT_APPROVED')
  }
  if (!table || table.status !== 'approved' || table.table_id !== projection.table_id) {
    return decision(projection, authority, 'not_ranked', 'TABLE_NOT_APPROVED')
  }
  if (venue.ranking_scope_id !== projection.ranking_scope_id
      || table.venue_id !== projection.venue_id
      || !session || session.session_id !== projection.session_id
      || session.table_id !== projection.table_id) {
    return decision(projection, authority, 'not_ranked', 'SESSION_MISMATCH')
  }
  if (!session.evidence_source_kind) {
    return decision(projection, authority, 'not_ranked', 'SESSION_SOURCE_UNVERIFIED')
  }
  if (session.evidence_source_kind !== 'usb_live') {
    return decision(projection, authority, 'not_ranked', 'SESSION_NOT_USB_LIVE')
  }
  if (session.device_deployment_status !== 'approved') {
    return decision(projection, authority, 'not_ranked', 'DEVICE_DEPLOYMENT_NOT_APPROVED')
  }
  if (!validTime(projection.started_at) || !validTime(session.started_at)) {
    return decision(projection, authority, 'not_ranked', 'START_EVIDENCE_MISSING')
  }
  if (!verifiedBindings(projection, session)) {
    return decision(projection, authority, 'not_ranked', 'ROLE_BINDINGS_UNVERIFIED')
  }
  return decision(projection, authority, 'eligible', 'ELIGIBLE')
}


module.exports = { evaluateMatchEligibility }
