const crypto = require('crypto')

const { ClientApiError } = require('./errors')
const { anonymousIdentity } = require('./identity')
const { isManagedSessionId } = require('../../shared/managed-command-authority')

const ROLES = new Set(['red', 'blue'])
const TABLE_CODE_PATTERN = /^[A-Za-z0-9_-]{32}$/u

async function getOptionalDocument(reference) {
  try {
    return await reference.get()
  } catch (error) {
    const message = error && error.message ? error.message : String(error)
    if (/document with _id .+ does not exist/i.test(message)) return { data: null }
    throw error
  }
}

function digestTableCode(tableCode) {
  if (typeof tableCode !== 'string' || !TABLE_CODE_PATTERN.test(tableCode)) {
    throw new ClientApiError('TABLE_CODE_INVALID')
  }
  return crypto.createHash('sha256').update(tableCode, 'utf8').digest('hex')
}

function publicJoinRequest(request) {
  return {
    schema_version: '1.0',
    request_id: request.request_id,
    session_id: request.session_id,
    display_name: request.display_name || anonymousIdentity(request.user_id).display_name,
    status: request.status,
    role: request.role || null,
  }
}

async function requestJoin({ tableCode, userId, now, repo }) {
  if (typeof userId !== 'string' || !userId || typeof now !== 'string') {
    throw new ClientApiError('REQUEST_INVALID')
  }
  const request = await repo.resolveRequest({
    codeDigest: digestTableCode(tableCode),
    userId,
    displayName: anonymousIdentity(userId).display_name,
    now,
  })
  return publicJoinRequest(request)
}

async function getJoinStatus({ requestId, userId, repo }) {
  if (typeof requestId !== 'string' || !requestId || typeof userId !== 'string' || !userId) {
    throw new ClientApiError('REQUEST_INVALID')
  }
  const request = await repo.getRequest({ requestId, userId })
  if (!request) throw new ClientApiError('SESSION_NOT_JOINABLE')
  return publicJoinRequest(request)
}

async function confirmRole({ requestId, role, operatorId, repo }) {
  if (!ROLES.has(role) || typeof requestId !== 'string' || !requestId
      || typeof operatorId !== 'string' || !operatorId) {
    throw new ClientApiError('REQUEST_INVALID')
  }
  return repo.confirmRole({ requestId, role, operatorId })
}

function createCloudJoinRepository(db, { idFactory = () => `join-${crypto.randomUUID()}` } = {}) {
  const collection = (transaction, name) => transaction.collection(name)
  return {
    async resolveRequest({ codeDigest, userId, displayName, now }) {
      const codeQuery = await db.collection('table_join_codes')
        .where({ code_digest: codeDigest }).limit(2).get()
      const codes = codeQuery.data || []
      if (codes.length !== 1) throw new ClientApiError('TABLE_CODE_DISABLED')
      const tableId = codes[0].table_id
      return db.runTransaction(async (transaction) => {
        const codeResult = await collection(transaction, 'table_join_codes').doc(tableId).get()
        const code = codeResult.data
        if (!code || code.code_digest !== codeDigest || code.status !== 'active') {
          throw new ClientApiError('TABLE_CODE_DISABLED')
        }
        if (!code.current_session_id) throw new ClientApiError('SESSION_NOT_JOINABLE')
        const sessionResult = await collection(transaction, 'field_live_sessions')
          .doc(code.current_session_id).get()
        const session = sessionResult.data
        if (!session || session.table_id !== tableId || session.status !== 'open') {
          throw new ClientApiError('SESSION_NOT_JOINABLE')
        }
        const bindingId = crypto.createHash('sha256')
          .update(`${session.session_id}:${userId}`, 'utf8').digest('hex')
        const bindingRef = collection(transaction, 'session_user_bindings').doc(bindingId)
        const bindingResult = await getOptionalDocument(bindingRef)
        if (bindingResult.data && bindingResult.data.request_id) {
          const existing = await collection(transaction, 'session_join_requests')
            .doc(bindingResult.data.request_id).get()
          if (!existing.data) throw new ClientApiError('SYNC_CONFLICT')
          return existing.data
        }
        const rateId = crypto.createHash('sha256')
          .update(`${tableId}:${userId}`, 'utf8').digest('hex')
        const rateRef = collection(transaction, 'join_rate_limits').doc(rateId)
        const rateResult = await getOptionalDocument(rateRef)
        if (rateResult.data && rateResult.data.blocked_until > now) {
          throw new ClientApiError('JOIN_RATE_LIMITED')
        }
        const request = {
          schema_version: '1.0', request_id: idFactory(), session_id: session.session_id,
          table_id: tableId, user_id: userId, display_name: displayName,
          status: 'pending', role: null, created_at: now, updated_at: now,
        }
        await collection(transaction, 'session_join_requests').doc(request.request_id)
          .set({ data: request })
        await bindingRef.set({ data: {
          schema_version: '1.0', session_id: session.session_id, table_id: tableId,
          user_id: userId, request_id: request.request_id, role: null, created_at: now,
        } })
        await rateRef.set({ data: {
          schema_version: '1.0', table_id: tableId, user_id: userId,
          last_request_at: now, blocked_until: now,
        } })
        return request
      })
    },
    async getRequest({ requestId, userId }) {
      const result = await db.collection('session_join_requests').doc(requestId).get()
      if (!result.data || result.data.user_id !== userId) return null
      return result.data
    },
    async confirmRole({ requestId, role, operatorId }) {
      return db.runTransaction(async (transaction) => {
        const requestRef = collection(transaction, 'session_join_requests').doc(requestId)
        const requestResult = await requestRef.get()
        const request = requestResult.data
        if (!request || request.status === 'ended') throw new ClientApiError('SESSION_NOT_JOINABLE')
        if (isManagedSessionId(request.session_id)) throw new ClientApiError('COMMAND_FORBIDDEN')
        if (request.status === 'confirmed') {
          if (request.role !== role) throw new ClientApiError('ROLE_CONFLICT')
          return request
        }
        const roleId = crypto.createHash('sha256')
          .update(`${request.session_id}:${role}`, 'utf8').digest('hex')
        const roleRef = collection(transaction, 'session_role_bindings').doc(roleId)
        const roleResult = await roleRef.get()
        if (roleResult.data && roleResult.data.user_id !== request.user_id) {
          throw new ClientApiError('ROLE_CONFLICT')
        }
        const updatedAt = new Date().toISOString()
        await roleRef.set({ data: {
          schema_version: '1.0', session_id: request.session_id, user_id: request.user_id,
          request_id: requestId, role, operator_id: operatorId, confirmed_at: updatedAt,
        } })
        await requestRef.update({ data: { status: 'confirmed', role, updated_at: updatedAt } })
        return { ...request, status: 'confirmed', role, updated_at: updatedAt }
      })
    },
  }
}

module.exports = {
  confirmRole,
  createCloudJoinRepository,
  digestTableCode,
  getJoinStatus,
  publicJoinRequest,
  requestJoin,
}
