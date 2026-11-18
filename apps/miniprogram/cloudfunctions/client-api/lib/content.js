const crypto = require('crypto')

const { ClientApiError } = require('./errors')
const { documentId, isActivePrivateSavedClip, isPrivateClipOwner } = require('../../shared/shot-ownership')
const { getOwnerClipTombstone } = require('../../shared/owner-delete')
const { getOptionalDocument } = require('./users')
const { isSavedUntilDeletedClip } = require('../../shared/retention-policy')


function requireId(value) {
  if (typeof value !== 'string' || !value || value.length > 128) {
    throw new ClientApiError('REQUEST_INVALID')
  }
  return value
}


function pick(value, keys) {
  return Object.fromEntries(keys.filter(key => Object.hasOwn(value, key)).map(key => [key, value[key]]))
}

function publicSession(session) {
  // Ordinary field facts are not a reviewed report. Internal receipts, future
  // projections and arbitrary semantic flags require a separate trusted join.
  const value = pick(session, ['schema_version', 'session_id', 'artifact_id', 'status',
    'created_at_utc', 'ended_at_utc', 'breaker', 'active_role_at_end', 'limitations',
    'sync_status', 'published_at'])
  if (session.participants) value.participants = session.participants.map(p =>
    pick(p, ['schema_version', 'user_id', 'role', 'group']))
  if (session.shots) value.shots = session.shots.map(shot => {
    const projected = pick(shot, ['schema_version', 'shot_id', 'sequence', 'actor_role',
      'started_ms', 'settled_ms', 'recommendation_id', 'limitations'])
    if (Object.hasOwn(shot, 'power')) projected.power = shot.power == null ? null
      : pick(shot.power, ['schema_version', 'band', 'approximate_percent', 'personalized'])
    return projected
  })
  return value
}

function publicAnalysis(projection) {
  return pick(projection, [
    'schema_version', 'session_id', 'revision', 'status', 'total_shots',
    'available_shots', 'retrying_shots', 'report_available', 'deadline_at', 'updated_at',
    'live_shots_recorded',
  ])
}

async function readSessionAnalysis(sessionId, repo, { allowLegacy = false } = {}) {
  const [projection, liveProgress] = await Promise.all([
    repo.getSessionAnalysis(sessionId),
    repo.getLiveShotProgress ? repo.getLiveShotProgress(sessionId) : null,
  ])
  if (liveProgress && (liveProgress.session_id !== sessionId
      || !Number.isInteger(liveProgress.shot_count) || liveProgress.shot_count < 0)) {
    throw new ClientApiError('SESSION_NOT_JOINABLE')
  }
  if (projection) {
    if (projection.session_id !== sessionId) throw new ClientApiError('SESSION_NOT_JOINABLE')
    return publicAnalysis({
      ...projection,
      live_shots_recorded: liveProgress ? liveProgress.shot_count : undefined,
    })
  }
  const ended = await repo.getEndedFieldSession(sessionId)
  if (!ended || ended.session_id !== sessionId || ended.status !== 'ended') {
    if (allowLegacy) return null
    throw new ClientApiError('SESSION_NOT_JOINABLE')
  }
  const endedAt = ended.ended_at || ended.ended_at_utc
  const timestamp = Date.parse(endedAt)
  if (!Number.isFinite(timestamp)) throw new ClientApiError('SESSION_NOT_JOINABLE')
  return {
    schema_version: '1.0', session_id: sessionId, revision: 1,
    status: 'PROCESSING', total_shots: null, available_shots: null,
    retrying_shots: null, report_available: false,
    deadline_at: new Date(timestamp + 120000).toISOString(),
    updated_at: ended.updated_at || endedAt,
  }
}


function publicClip(clip) {
  const value = pick(clip, ['schema_version', 'clip_id', 'session_id', 'shot_id', 'artifact_id',
    'source_kind', 'start_ms', 'end_ms', 'duration_ms', 'content_sha256', 'byte_size',
    'media_type', 'sequence', 'upload_status', 'consent_status', 'deleted_at', 'published_at',
    'publication_kind', 'actor_role'])
  if (Object.hasOwn(value, 'source_kind')) {
    if (value.source_kind === 'development_presentation') value.source_kind = 'development_replay'
    else if (!['development_replay', 'usb_live'].includes(value.source_kind)) {
      throw new ClientApiError('MEDIA_NOT_AVAILABLE')
    }
  }
  if (isSavedUntilDeletedClip(clip)) {
    value.retention_policy = clip.retention_policy
    value.available_until = null
  } else if (clip.retention_expires_at) value.available_until = clip.retention_expires_at
  return value
}


function legacyFavoriteResult(action) {
  return ['favorites.list', 'favorites.set'].includes(action)
    ? { ok: false, error: { code: 'ACTION_UNSUPPORTED' } }
    : null
}


function isPlainRecord(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}


async function hasImmutableOwnership(clip, userId, repo) {
  if (clip.owner_user_id !== userId || !isActivePrivateSavedClip(clip)) return false
  if (await repo.getOwnerClipTombstone(clip.clip_id) !== null) return false
  const projection = await repo.getShotOwnership(clip.session_id, clip.shot_id)
  return isPrivateClipOwner(clip, userId, projection)
}


async function requireParticipant({ sessionId, userId, repo }) {
  requireId(sessionId)
  requireId(userId)
  if (!await repo.hasParticipant(sessionId, userId)) {
    throw new ClientApiError('FORBIDDEN')
  }
}


async function listSessions({ userId, repo }) {
  requireId(userId)
  const sessions = await repo.listSessions(userId)
  const authorized = await Promise.all(sessions.map(async session => {
    if (!await repo.hasParticipant(session.session_id, userId)) return null
    const value = publicSession(session)
    value.analysis = await readSessionAnalysis(session.session_id, repo, { allowLegacy: true })
    return value
  }))
  return authorized.filter(Boolean)
}


async function getSession({ sessionId, userId, repo }) {
  await requireParticipant({ sessionId, userId, repo })
  const session = await repo.getSession(sessionId)
  if (!session || session.session_id !== sessionId) throw new ClientApiError('SESSION_NOT_JOINABLE')
  const value = publicSession(session)
  value.analysis = await readSessionAnalysis(sessionId, repo, { allowLegacy: true })
  return value
}


async function getSessionAnalysis({ sessionId, userId, repo }) {
  requireId(sessionId)
  requireId(userId)
  const finalParticipant = await repo.hasParticipant(sessionId, userId)
  const confirmedBinding = finalParticipant
    ? false : await repo.hasConfirmedSessionBinding(sessionId, userId)
  if (!finalParticipant && !confirmedBinding) throw new ClientApiError('FORBIDDEN')
  return readSessionAnalysis(sessionId, repo)
}


async function listClips({ sessionId, userId, repo }) {
  requireId(userId)
  if (sessionId !== undefined && sessionId !== null) requireId(sessionId)
  const clips = await repo.listClips(sessionId || undefined, userId)
  if (!Array.isArray(clips)
      || clips.some(clip => !isPlainRecord(clip) || (sessionId && clip.session_id !== sessionId))) {
    throw new ClientApiError('MEDIA_NOT_AVAILABLE')
  }
  const authorized = await Promise.all(clips.map(async clip =>
    await hasImmutableOwnership(clip, userId, repo) ? publicClip(clip) : null))
  return authorized.filter(Boolean)
}

const CLIP_PAGE_SIZE = 25
const CLIP_CURSOR_FIELDS = ['schema_version', 'scope', 'published_at', 'clip_id']

function clipCursorScope(userId, sessionId) {
  return crypto.createHash('sha256').update(JSON.stringify([userId, sessionId || null]), 'utf8').digest('hex')
}
function validClipPageKey(value) {
  return value && typeof value.published_at === 'string'
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u.test(value.published_at)
    && Number.isFinite(Date.parse(value.published_at))
    && typeof value.clip_id === 'string' && /^[A-Za-z0-9._-]{1,128}$/u.test(value.clip_id)
}
function decodeClipCursor(cursor, scope) {
  if (cursor === undefined || cursor === null) return null
  try {
    if (typeof cursor !== 'string' || !cursor || cursor.length > 2048
        || Buffer.from(cursor, 'base64url').toString('base64url') !== cursor) throw new Error('invalid cursor')
    const value = JSON.parse(Buffer.from(cursor, 'base64url').toString('utf8'))
    if (!isPlainRecord(value) || Object.keys(value).length !== CLIP_CURSOR_FIELDS.length
        || Object.keys(value).some(key => !CLIP_CURSOR_FIELDS.includes(key))
        || value.schema_version !== '1.0' || value.scope !== scope || !validClipPageKey(value)) {
      throw new Error('invalid cursor')
    }
    return { published_at: value.published_at, clip_id: value.clip_id }
  } catch (_error) { throw new ClientApiError('REQUEST_INVALID') }
}

async function listClipsPage({ sessionId, userId, cursor, repo }) {
  requireId(userId)
  if (sessionId !== undefined && sessionId !== null) requireId(sessionId)
  const scope = clipCursorScope(userId, sessionId)
  const after = decodeClipCursor(cursor, scope)
  const candidates = await repo.listClipPageCandidates({ sessionId: sessionId || undefined, userId, after })
  if (!Array.isArray(candidates) || candidates.length > CLIP_PAGE_SIZE
      || candidates.some(clip => !isPlainRecord(clip) || !validClipPageKey(clip)
        || (sessionId && clip.session_id !== sessionId))) throw new ClientApiError('MEDIA_NOT_AVAILABLE')
  const authorized = await Promise.all(candidates.map(async clip =>
    await hasImmutableOwnership(clip, userId, repo) ? publicClip(clip) : null))
  const last = candidates[candidates.length - 1]
  // Advance over every candidate, including deleted/invalid private rows. A
  // full filtered page still has a continuation; the next query may be empty.
  const nextCursor = candidates.length === CLIP_PAGE_SIZE ? Buffer.from(JSON.stringify({
    schema_version: '1.0', scope, published_at: last.published_at, clip_id: last.clip_id,
  }), 'utf8').toString('base64url') : null
  return { schema_version: '1.0', items: authorized.filter(Boolean), next_cursor: nextCursor }
}

async function getClipPlayback({ clipId, userId, repo }) {
  requireId(clipId)
  requireId(userId)
  // The repository commits a grant ordered against the permanent tombstone.
  // Sign only that snapshot; do not re-read a stale list or mutable clip row.
  const clip = await repo.authorizePlaybackForOwner({ clipId, userId })
  return { ...await repo.createPlaybackUrl(clip.storage_file_id, clip.playback_authorized_until), clip: publicClip(clip) }
}


function createCloudContentRepository(db, cloud, { clock = () => Date.now() } = {}) {
  // This dependency is server-owned. Request payloads cannot select grant/signing time.
  function serverTime() {
    const time = clock()
    if (typeof time !== 'number' || !Number.isFinite(time) || !Number.isFinite(new Date(time).getTime())) {
      throw new ClientApiError('MEDIA_NOT_AVAILABLE')
    }
    return time
  }
  function remainingAuthority(deadline) {
    const until = typeof deadline === 'string' ? Date.parse(deadline) : NaN
    const seconds = Math.min(300, Math.floor((until - serverTime()) / 1000))
    if (!Number.isFinite(seconds) || seconds < 1) throw new ClientApiError('MEDIA_NOT_AVAILABLE')
    return seconds
  }
  return {
    async getOwnerClipTombstone(clipId) { return getOwnerClipTombstone(db, clipId) },
    async authorizePlaybackForOwner({ clipId, userId }) {
      return db.runTransaction(async transaction => {
        if (await getOwnerClipTombstone(transaction, clipId) !== null) throw new ClientApiError('MEDIA_NOT_AVAILABLE')
        const ref = transaction.collection('clips').doc(clipId)
        const clip = (await getOptionalDocument(ref)).data
        const projection = clip && (await getOptionalDocument(transaction.collection('field_shot_ownership')
          .doc(documentId(clip.session_id + ':' + clip.shot_id)))).data
        if (!clip || clip.clip_id !== clipId || !isPrivateClipOwner(clip, userId, projection)) {
          throw new ClientApiError('MEDIA_NOT_AVAILABLE')
        }
        const existing = clip.playback_authorized_until == null ? 0 : Date.parse(clip.playback_authorized_until)
        if (!Number.isFinite(existing)) throw new ClientApiError('MEDIA_NOT_AVAILABLE')
        const time = serverTime()
        const grantUntil = new Date(time + 300000).getTime()
        if (!Number.isFinite(grantUntil)) throw new ClientApiError('MEDIA_NOT_AVAILABLE')
        const until = new Date(Math.max(existing, grantUntil)).toISOString()
        await ref.update({ data: { playback_authorized_until: until } })
        return { ...clip, playback_authorized_until: until }
      })
    },
    async hasParticipant(sessionId, userId) {
      const result = await db.collection('session_participants')
        .where({ session_id: sessionId, user_id: userId })
        .limit(1)
        .get()
      return Boolean(result.data && result.data.length)
    },

    async getSession(sessionId) {
      const result = await getOptionalDocument(db.collection('sessions').doc(sessionId))
      return result.data || null
    },

    async getSessionAnalysis(sessionId) {
      const result = await getOptionalDocument(
        db.collection('session_analysis_projections').doc(sessionId),
      )
      return result.data || null
    },

    async getLiveShotProgress(sessionId) {
      const result = await getOptionalDocument(
        db.collection('field_live_shot_progress').doc(sessionId),
      )
      return result.data || null
    },

    async hasConfirmedSessionBinding(sessionId, userId) {
      const bindingId = crypto.createHash('sha256')
        .update(`${sessionId}:${userId}`, 'utf8').digest('hex')
      const binding = await getOptionalDocument(
        db.collection('session_user_bindings').doc(bindingId),
      )
      if (!binding.data || binding.data.session_id !== sessionId
          || binding.data.user_id !== userId || !['red', 'blue'].includes(binding.data.role)) return false
      const roleId = crypto.createHash('sha256')
        .update(`${sessionId}:${binding.data.role}`, 'utf8').digest('hex')
      const role = await getOptionalDocument(db.collection('session_role_bindings').doc(roleId))
      return Boolean(role.data && role.data.session_id === sessionId
        && role.data.user_id === userId && role.data.role === binding.data.role
        && role.data.confirmed_at)
    },

    async getEndedFieldSession(sessionId) {
      const result = await getOptionalDocument(db.collection('field_live_sessions').doc(sessionId))
      return result.data || null
    },

    async listSessions(userId) {
      const memberships = await db.collection('session_participants')
        .where({ user_id: userId })
        .orderBy('created_at_utc', 'desc')
        .limit(50)
        .get()
      const sessions = await Promise.all((memberships.data || []).map(async (membership) => {
        const result = await getOptionalDocument(db.collection('sessions').doc(membership.session_id))
        return result.data || null
      }))
      return sessions.filter(Boolean)
    },

    async getClip(clipId) {
      const result = await getOptionalDocument(db.collection('clips').doc(clipId))
      return result.data || null
    },

    async getShotOwnership(sessionId, shotId) {
      const result = await getOptionalDocument(db.collection('field_shot_ownership')
        .doc(documentId(`${sessionId}:${shotId}`)))
      return result.data || null
    },

    async listClips(sessionId, userId) {
      const filter = { owner_user_id: userId }
      if (sessionId) filter.session_id = sessionId
      const result = await db.collection('clips')
        .where(filter)
        .orderBy('sequence', 'asc')
        .limit(100)
        .get()
      return result.data || []
    },

    async listClipPageCandidates({ sessionId, userId, after }) {
      const ownerFilter = { owner_user_id: userId }
      if (sessionId) ownerFilter.session_id = sessionId
      // published_at is the server-written immutable ISO completion timestamp;
      // clip_id uniquely orders all ties across sessions. Cursor scope only
      // selects a position: these owner predicates remain the authority.
      const filter = after ? db.command.and([
        ownerFilter,
        db.command.or([
          { published_at: db.command.lt(after.published_at) },
          { published_at: after.published_at, clip_id: db.command.lt(after.clip_id) },
        ]),
      ]) : ownerFilter
      const result = await db.collection('clips').where(filter)
        .orderBy('published_at', 'desc').orderBy('clip_id', 'desc')
        .limit(CLIP_PAGE_SIZE).get()
      return result.data || []
    },

    async createPlaybackUrl(fileId, authorizedUntil) {
      const maxAge = remainingAuthority(authorizedUntil)
      const result = await cloud.getTempFileURL({
        fileList: [{ fileID: fileId, maxAge }],
      })
      // A provider response can arrive after the application's grant has closed.
      // Discard its URL then; provider lifetime/physical closure need separate evidence.
      const remaining = Math.min(maxAge, remainingAuthority(authorizedUntil))
      const item = result.fileList && result.fileList[0]
      if (!item || (item.status !== 0 && item.code !== 'SUCCESS') || !item.tempFileURL) {
        throw new ClientApiError('MEDIA_NOT_AVAILABLE')
      }
      return {
        schema_version: '1.0',
        url: item.tempFileURL,
        expires_in_seconds: remaining,
      }
    },

  }
}


module.exports = {
  createCloudContentRepository,
  getClipPlayback,
  getSession,
  getSessionAnalysis,
  legacyFavoriteResult,
  listClips,
  listClipsPage,
  listSessions,
}
