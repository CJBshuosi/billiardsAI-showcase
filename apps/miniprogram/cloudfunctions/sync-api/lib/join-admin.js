const crypto = require('crypto')

const { SyncApiError } = require('./auth')
const { MANAGED_MEDIA_ADMISSION_CONTRACT, isManagedSessionId, readManagedCommandContext } = require('../../shared/managed-command-authority')

const ROLES = new Set(['red', 'blue'])

const BIRTH_CONTRACT = 'managed_session_birth_v1'
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu
const isUuid = value => typeof value === 'string' && UUID_V4.test(value)
const isScopeId = value => typeof value === 'string' && /^[A-Za-z0-9._-]{1,128}$/u.test(value)
const isTime = value => typeof value === 'string' && !Number.isNaN(Date.parse(value))
  && new Date(value).toISOString() === value
const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value)
function exactKeys(value, keys) {
  return isRecord(value) && (!Object.hasOwn(value, '_id') || typeof value._id === 'string')
    && Object.keys(value).filter(key => key !== '_id').length === keys.length
    && keys.every(key => Object.hasOwn(value, key))
}
function conflictUnless(condition) {
  if (!condition) throw new SyncApiError('SYNC_CONFLICT')
}
function validIdentity(value) {
  return isUuid(value.birth_id) && value.session_id === `managed-${value.birth_id}`
    && isScopeId(value.table_id) && isScopeId(value.device_id)
}
function validRequest(value) {
  return exactKeys(value, ['contract', 'open_request_id', 'session_id', 'birth_id', 'table_id', 'device_id'])
    && value.contract === 'managed_session_open_request_v1' && isUuid(value.open_request_id)
    && validIdentity(value)
}
async function requireEndedPredecessor(transaction, sessionId, tableId) {
  if (isManagedSessionId(sessionId)) {
    const context = await readManagedCommandContext(transaction, { sessionId, tableId })
    conflictUnless(context && context.head.coverage === 'sealed' && context.live.status === 'ended')
    return
  }
  const current = (await getOptionalDocument(transaction.collection('field_live_sessions').doc(sessionId))).data
  conflictUnless(current && current.table_id === tableId && current.status === 'ended')
}
function publicManagedSession(identity, status) {
  return { schema_version: '1.0', session_id: identity.session_id, birth_id: identity.birth_id,
    birth_contract: BIRTH_CONTRACT, table_id: identity.table_id, device_id: identity.device_id, status }
}


function withoutDocumentId(document) {
  if (!document || typeof document !== 'object') return document
  const { _id: _documentId, ...data } = document
  return data
}

async function getOptionalDocument(ref) {
  try {
    return await ref.get()
  } catch (error) {
    const message = error && error.message ? error.message : String(error)
    if (!/document with _id .+ does not exist/iu.test(message)) throw error
    return { data: null }
  }
}

function publicPendingRequest(request) {
  return {
    schema_version: '1.0', request_id: request.request_id,
    display_name: request.display_name, status: request.status, role: request.role || null,
  }
}

function createJoinAdmin(db, {
  nowIso = () => new Date().toISOString(), environment = 'development',
  managedSessionsEnabled = false, managedMediaAdmissionEnabled = false, idFactory = crypto.randomUUID,
} = {}) {
  if (typeof managedMediaAdmissionEnabled !== 'boolean'
      || (managedMediaAdmissionEnabled && managedSessionsEnabled !== true)) {
    throw new SyncApiError('SYNC_CONFIGURATION_REQUIRED')
  }
  const requireTable = async (transaction, tableId, errorCode) => {
    const result = await getOptionalDocument(transaction.collection('table_join_codes').doc(tableId))
    if (!result.data) throw new SyncApiError(errorCode)
    return result.data
  }
  return {
    async createManagedSession({ openRequestId, tableId, deviceId }) {
      if (managedSessionsEnabled !== true) throw new SyncApiError('SYNC_CONFIGURATION_REQUIRED')
      if (!isUuid(openRequestId) || !isScopeId(tableId) || !isScopeId(deviceId)) {
        throw new SyncApiError('SYNC_INVALID')
      }
      return db.runTransaction(async (transaction) => {
        const table = await requireTable(transaction, tableId, 'SYNC_CONFIGURATION_REQUIRED')
        if (table.status !== 'active' || table.table_id !== tableId) {
          throw new SyncApiError('SYNC_CONFIGURATION_REQUIRED')
        }
        conflictUnless(table.current_session_id == null || isScopeId(table.current_session_id))
        const requestKey = crypto.createHash('sha256').update(deviceId + ':' + openRequestId).digest('hex')
        const requestRef = transaction.collection('managed_session_open_requests').doc(requestKey)
        const existing = (await getOptionalDocument(requestRef)).data
        if (existing) {
          conflictUnless(validRequest(existing) && existing.open_request_id === openRequestId
            && existing.table_id === tableId && existing.device_id === deviceId)
          const sessionId = existing.session_id
          const context = await readManagedCommandContext(transaction, { sessionId, tableId, deviceId })
          conflictUnless(context && context.birth.birth_id === existing.birth_id)
          const { head, live } = context
          conflictUnless(head.coverage === 'sealed' || table.current_session_id === sessionId)
          return publicManagedSession(existing, live.status)
        }
        if (table.current_session_id) {
          await requireEndedPredecessor(transaction, table.current_session_id, tableId)
        }
        const birthId = idFactory()
        conflictUnless(isUuid(birthId))
        const sessionId = 'managed-' + birthId
        for (const name of ['managed_session_births', 'field_live_sessions', 'sessions', 'session_command_authority']) {
          conflictUnless(!(await getOptionalDocument(transaction.collection(name).doc(sessionId))).data)
        }
        const now = nowIso()
        conflictUnless(isTime(now) && typeof environment === 'string' && environment.length > 0)
        const identity = { session_id: sessionId, birth_id: birthId, table_id: tableId, device_id: deviceId }
        await transaction.collection('managed_session_births').doc(sessionId).set({ data: {
          contract: BIRTH_CONTRACT, ...identity, opened_at: now,
          ...(managedMediaAdmissionEnabled ? { media_admission_contract: MANAGED_MEDIA_ADMISSION_CONTRACT } : {}),
        } })
        await requestRef.set({ data: {
          contract: 'managed_session_open_request_v1', open_request_id: openRequestId, ...identity,
        } })
        await transaction.collection('session_command_authority').doc(sessionId).set({ data: {
          contract: 'managed_command_authority_v1', ...identity, contiguous_ack_sequence: 0,
          last_resulting_revision: 0, last_applied_server_sequence: 0,
          role_tails: { red: null, blue: null }, coverage: 'open', seal: null, revision: 1,
          issued_next: 1, closed_target: null,
        } })
        await transaction.collection('field_live_sessions').doc(sessionId).set({ data: {
          schema_version: '1.0', ...identity, status: 'open', opened_at: now, updated_at: now,
          environment, birth_contract: BIRTH_CONTRACT,
        } })
        await transaction.collection('table_join_codes').doc(tableId).update({ data: {
          current_session_id: sessionId, updated_at: now,
        } })
        return publicManagedSession(identity, 'open')
      })
    },
    async closeManagedSession({ sessionId, tableId, deviceId }) {
      if (managedSessionsEnabled !== true) throw new SyncApiError('SYNC_CONFIGURATION_REQUIRED')
      if (!isManagedSessionId(sessionId) || !isScopeId(sessionId)
          || !isScopeId(tableId) || !isScopeId(deviceId)) throw new SyncApiError('SYNC_INVALID')
      return db.runTransaction(async transaction => {
        const context = await readManagedCommandContext(transaction, { sessionId, tableId, deviceId })
        conflictUnless(context)
        const { birth, head, live } = context
        const table = await requireTable(transaction, tableId, 'SYNC_CONFLICT')
        conflictUnless(table.table_id === tableId && table.status === 'active')
        conflictUnless(head.coverage === 'sealed' || table.current_session_id === sessionId)
        let closed = withoutDocumentId(head)
        if (head.coverage === 'open') {
          const now = nowIso()
          conflictUnless(isTime(now) && head.revision < Number.MAX_SAFE_INTEGER)
          const target = head.issued_next - 1
          const sealed = head.contiguous_ack_sequence === target
          closed = { ...closed, closed_target: target, coverage: sealed ? 'sealed' : 'closing',
            seal: sealed ? { sealed_at: now } : null, revision: head.revision + 1 }
          await transaction.collection('session_command_authority').doc(sessionId).set({ data: closed })
          await transaction.collection('field_live_sessions').doc(sessionId).set({ data: {
            ...withoutDocumentId(live), status: 'ended', ended_at: live.ended_at || now, updated_at: now,
          } })
        } else conflictUnless(live.status === 'ended')
        return { schema_version: '1.0', session_id: sessionId, birth_id: birth.birth_id,
          birth_contract: BIRTH_CONTRACT, table_id: tableId, device_id: deviceId,
          coverage: closed.coverage, closed_target: closed.closed_target,
          contiguous_ack_sequence: closed.contiguous_ack_sequence }
      })
    },
    async openSession({ sessionId, tableId, operatorId = null }) {
      if (typeof sessionId === 'string' && sessionId.startsWith('managed-')) throw new SyncApiError('SYNC_CONFLICT')
      return db.runTransaction(async (transaction) => {
        const table = await requireTable(transaction, tableId, 'SYNC_CONFIGURATION_REQUIRED')
        if (table.status !== 'active') throw new SyncApiError('SYNC_CONFIGURATION_REQUIRED')
        if (table.current_session_id && table.current_session_id !== sessionId) {
          await requireEndedPredecessor(transaction, table.current_session_id, tableId)
        }
        const sessionRef = transaction.collection('field_live_sessions').doc(sessionId)
        const existing = await getOptionalDocument(sessionRef)
        if (existing.data && existing.data.table_id !== tableId) {
          throw new SyncApiError('SYNC_UNAUTHORIZED')
        }
        const now = nowIso()
        const session = existing.data && existing.data.status === 'open'
          ? withoutDocumentId(existing.data)
          : {
              schema_version: '1.0', session_id: sessionId, table_id: tableId,
              status: 'open', opened_at: now, updated_at: now,
              environment,
              device_id: typeof operatorId === 'string' ? operatorId : null,
            }
        await sessionRef.set({ data: session })
        await transaction.collection('table_join_codes').doc(tableId).update({
          data: { current_session_id: sessionId, updated_at: now },
        })
        return session
      })
    },
    async closeSession({ sessionId, tableId }) {
      if (typeof sessionId === 'string' && sessionId.startsWith('managed-')) throw new SyncApiError('SYNC_CONFLICT')
      return db.runTransaction(async (transaction) => {
        const table = await requireTable(transaction, tableId, 'SYNC_UNAUTHORIZED')
        const sessionRef = transaction.collection('field_live_sessions').doc(sessionId)
        const existing = await sessionRef.get()
        if (!existing.data || existing.data.table_id !== tableId) {
          throw new SyncApiError('SYNC_UNAUTHORIZED')
        }
        if (table.current_session_id && table.current_session_id !== sessionId) {
          throw new SyncApiError('SYNC_CONFLICT')
        }
        const now = nowIso()
        if (existing.data.status === 'ended') {
          if (table.current_session_id === sessionId) {
            await transaction.collection('table_join_codes').doc(tableId).update({
              data: { current_session_id: null, updated_at: now },
            })
          }
          return existing.data
        }
        const session = {
          ...withoutDocumentId(existing.data), status: 'ended', ended_at: now, updated_at: now,
        }
        await sessionRef.set({ data: session })
        await transaction.collection('table_join_codes').doc(tableId).update({
          data: { current_session_id: null, updated_at: now },
        })
        return session
      })
    },
    async listRequests({ sessionId, tableId }) {
      const session = await db.collection('field_live_sessions').doc(sessionId).get()
      if (!session.data || session.data.table_id !== tableId || session.data.status !== 'open') {
        throw new SyncApiError('SYNC_UNAUTHORIZED')
      }
      const result = await db.collection('session_join_requests')
        .where({ session_id: sessionId })
        .orderBy('created_at', 'asc').limit(20).get()
      return {
        schema_version: '1.0',
        requests: (result.data || [])
          .filter((request) => ['pending', 'confirmed'].includes(request.status))
          .map(publicPendingRequest),
      }
    },
    async confirm({ sessionId, requestId, role, tableId, operatorId }) {
      if (!ROLES.has(role)) throw new SyncApiError('SYNC_INVALID')
      if (isManagedSessionId(sessionId)) throw new SyncApiError('SYNC_CONFLICT')
      return db.runTransaction(async (transaction) => {
        const session = await transaction.collection('field_live_sessions').doc(sessionId).get()
        if (!session.data || session.data.table_id !== tableId || session.data.status !== 'open') {
          throw new SyncApiError('SYNC_UNAUTHORIZED')
        }
        const requestRef = transaction.collection('session_join_requests').doc(requestId)
        const requestResult = await requestRef.get()
        const request = requestResult.data
        if (!request || request.session_id !== sessionId || request.status === 'ended') {
          throw new SyncApiError('SYNC_CONFLICT')
        }
        if (request.status === 'confirmed') {
          if (request.role !== role) throw new SyncApiError('SYNC_CONFLICT')
          return request
        }
        const roleId = crypto.createHash('sha256').update(`${sessionId}:${role}`).digest('hex')
        const roleRef = transaction.collection('session_role_bindings').doc(roleId)
        const existing = await roleRef.get()
        if (existing.data && existing.data.user_id !== request.user_id) {
          throw new SyncApiError('SYNC_CONFLICT')
        }
        const now = nowIso()
        await roleRef.set({ data: {
          schema_version: '1.0', session_id: sessionId, request_id: requestId,
          user_id: request.user_id, role, operator_id: operatorId, confirmed_at: now,
        } })
        await requestRef.update({ data: { status: 'confirmed', role, updated_at: now } })
        return { ...request, status: 'confirmed', role, updated_at: now }
      })
    },
  }
}

module.exports = { createJoinAdmin, publicPendingRequest }
