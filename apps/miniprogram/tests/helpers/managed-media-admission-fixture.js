const { createJoinAdmin } = require('../../cloudfunctions/sync-api/lib/join-admin')
const { ownerDeleteDb } = require('./owner-delete-db')

const birthId = '12345678-1234-4234-8234-123456789abc'
const sessionId = `managed-${birthId}`
const now = '2026-11-01T10:00:00.000Z'
const input = { openRequestId: '98765432-1234-4234-9234-123456789abc', tableId: 'table-1', deviceId: 'device-1' }
// Actual trusted constructor/allocation boundary only. This fixture does not prove
// command history, publication eligibility, provider inventory or erasability.
function managedMediaBirthFixture(options = {}, dbOptions = {}) {
  const db = ownerDeleteDb({ table_join_codes: {
    'table-1': { table_id: 'table-1', status: 'active', current_session_id: null },
  } }, { strictTransactions: true, ...dbOptions })
  const calls = { ids: 0, clocks: 0 }
  const admin = createJoinAdmin(db, { managedSessionsEnabled: true,
    managedMediaAdmissionEnabled: true,
    idFactory() { calls.ids++; return birthId }, nowIso() { calls.clocks++; return now }, ...options })
  return { db, admin, calls, input: { ...input }, sessionId, birthId, now }
}
module.exports = { managedMediaBirthFixture, birthId, sessionId, now, input }
