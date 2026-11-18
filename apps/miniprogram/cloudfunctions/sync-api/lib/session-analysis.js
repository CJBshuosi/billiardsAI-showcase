const crypto = require('crypto')

const { SyncApiError } = require('./auth')

const HASH_PATTERN = /^[0-9a-f]{64}$/u
const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u
const PUBLIC_FIELDS = [
  'schema_version', 'session_id', 'revision', 'status', 'total_shots',
  'available_shots', 'retrying_shots', 'report_available', 'deadline_at', 'updated_at',
]
const INPUT_FIELDS = [
  ...PUBLIC_FIELDS, 'frozen_shots_digest', 'frozen_shot_ids', 'published_shots',
  'published_clips',
]
const LIVE_SHOT_FIELDS = [
  'session_id', 'shot_id', 'sequence', 'start_frame_id', 'settle_frame_id',
  'mode', 'actor_role', 'actor_user_id', 'observed_outcome', 'rule_revision',
  'recording_reference', 'idempotency_key',
]

function fail(code = 'SYNC_INVALID') { throw new SyncApiError(code) }

function isTimestamp(value) {
  return typeof value === 'string' && Number.isFinite(Date.parse(value))
}

function canonicalDigest(value) {
  const canonical = item => Array.isArray(item) ? item.map(canonical)
    : item && typeof item === 'object'
      ? Object.fromEntries(Object.keys(item).sort().map(key => [key, canonical(item[key])]))
      : item
  return crypto.createHash('sha256').update(JSON.stringify(canonical(value))).digest('hex')
}

function publicProjection(value) {
  return Object.fromEntries(PUBLIC_FIELDS.map(key => [key, value[key]]))
}

function publicLiveShotFact(value) {
  return Object.fromEntries(LIVE_SHOT_FIELDS.map(key => [key, value[key]]))
}

function validateLiveShotFact(sessionId, shotId, fact, idempotencyKey) {
  if (!fact || Array.isArray(fact) || typeof fact !== 'object'
      || Object.keys(fact).some(key => !LIVE_SHOT_FIELDS.includes(key))
      || Object.keys(fact).length !== LIVE_SHOT_FIELDS.length
      || fact.session_id !== sessionId || fact.shot_id !== shotId
      || !ID_PATTERN.test(sessionId) || !ID_PATTERN.test(shotId)
      || !Number.isInteger(fact.sequence) || fact.sequence < 1
      || !Number.isInteger(fact.start_frame_id) || fact.start_frame_id < 0
      || !Number.isInteger(fact.settle_frame_id)
      || fact.settle_frame_id <= fact.start_frame_id
      || !['assistance', 'observation'].includes(fact.mode)
      || !['red', 'blue'].includes(fact.actor_role)
      || !ID_PATTERN.test(fact.actor_user_id || '')
      || typeof fact.observed_outcome !== 'string' || !fact.observed_outcome
      || !Number.isInteger(fact.rule_revision) || fact.rule_revision < 0
      || typeof fact.recording_reference !== 'string' || !fact.recording_reference
      || fact.idempotency_key !== idempotencyKey) fail()
  const expected = `live-shot-${canonicalDigest({
    session_id: fact.session_id,
    shot_id: fact.shot_id,
    sequence: fact.sequence,
    start_frame_id: fact.start_frame_id,
    settle_frame_id: fact.settle_frame_id,
  })}`
  if (idempotencyKey !== expected) fail('SYNC_CONFLICT')
}

async function putLiveShotFact({
  sessionId, shotId, fact, tableId, deviceId, idempotencyKey, repo, now,
}) {
  validateLiveShotFact(sessionId, shotId, fact, idempotencyKey)
  const digest = canonicalDigest(fact)
  const work = async transaction => {
    const authority = await transaction.getSessionAuthority(sessionId)
    if (!authority || authority.session_id !== sessionId
        || !['open', 'ended'].includes(authority.status)
        || authority.table_id !== tableId || authority.device_id !== deviceId) {
      fail('SYNC_CONFLICT')
    }
    const existing = await transaction.getLiveShotFact(idempotencyKey)
    if (existing) {
      if (existing._content_digest !== digest || existing.session_id !== sessionId
          || existing.shot_id !== shotId || existing._table_id !== tableId
          || existing._device_id !== deviceId) fail('SYNC_CONFLICT')
      return publicLiveShotFact(existing)
    }
    const stored = {
      ...fact,
      _table_id: tableId,
      _device_id: deviceId,
      _content_digest: digest,
    }
    await transaction.saveLiveShotFact(stored)
    const currentProgress = await transaction.getLiveShotProgress(sessionId)
    if (currentProgress && (currentProgress.session_id !== sessionId
        || currentProgress._table_id !== tableId || currentProgress._device_id !== deviceId)) {
      fail('SYNC_CONFLICT')
    }
    const advancesLastShot = !currentProgress || fact.sequence > currentProgress.last_sequence
    await transaction.saveLiveShotProgress({
      schema_version: '1.0', session_id: sessionId,
      shot_count: (currentProgress && currentProgress.shot_count || 0) + 1,
      last_sequence: advancesLastShot ? fact.sequence : currentProgress.last_sequence,
      last_shot_id: advancesLastShot ? fact.shot_id : currentProgress.last_shot_id,
      updated_at: now || new Date().toISOString(),
      _table_id: tableId, _device_id: deviceId,
    })
    return publicLiveShotFact(stored)
  }
  return repo.runTransaction ? repo.runTransaction(work) : work(repo)
}

function validateProjection(sessionId, value) {
  if (value && typeof value === 'object' && value.session_id !== sessionId) {
    fail('SYNC_CONFLICT')
  }
  if (!value || Array.isArray(value) || typeof value !== 'object'
      || Object.keys(value).some(key => !INPUT_FIELDS.includes(key))
      || value.schema_version !== '1.0' || value.session_id !== sessionId
      || !ID_PATTERN.test(value.session_id || '') || !Number.isInteger(value.revision)
      || value.revision < 1 || !HASH_PATTERN.test(value.frozen_shots_digest || '')
      || !isTimestamp(value.deadline_at) || !isTimestamp(value.updated_at)
      || !['PROCESSING', 'AVAILABLE', 'PARTIAL', 'FAILED_RETRYING'].includes(value.status)
      || typeof value.report_available !== 'boolean'
      || !Array.isArray(value.frozen_shot_ids) || !Array.isArray(value.published_shots)
      || !Array.isArray(value.published_clips)) fail()

  const frozen = new Set(value.frozen_shot_ids)
  if (frozen.size !== value.frozen_shot_ids.length
      || value.frozen_shot_ids.some(item => !ID_PATTERN.test(item || ''))) fail()
  if (canonicalDigest(value.frozen_shot_ids) !== value.frozen_shots_digest) {
    fail('SYNC_CONFLICT')
  }
  const counts = [value.total_shots, value.available_shots, value.retrying_shots]
  const absent = counts.filter(item => item == null).length
  if (absent && (absent !== counts.length || value.status !== 'PROCESSING')) fail()
  if (absent && (value.published_shots.length || value.published_clips.length)) fail()
  if (!absent && counts.some(item => !Number.isInteger(item) || item < 0)) fail()
  if (!absent && (value.total_shots !== frozen.size
      || value.available_shots + value.retrying_shots > value.total_shots)) fail()

  const publishedShotIds = new Set()
  for (const shot of value.published_shots) {
    if (!shot || shot.session_id !== sessionId || !frozen.has(shot.shot_id)
        || publishedShotIds.has(shot.shot_id)) fail('SYNC_CONFLICT')
    publishedShotIds.add(shot.shot_id)
  }
  const clipIds = new Set()
  for (const clip of value.published_clips) {
    if (!clip || clip.session_id !== sessionId || !frozen.has(clip.shot_id)
        || !publishedShotIds.has(clip.shot_id) || !ID_PATTERN.test(clip.clip_id || '')
        || clipIds.has(clip.clip_id)) fail('SYNC_CONFLICT')
    clipIds.add(clip.clip_id)
  }
  if (!absent && publishedShotIds.size !== value.available_shots) fail()
  if (value.status === 'PROCESSING' && (value.report_available || (!absent
      && (value.available_shots !== 0 || value.retrying_shots !== 0)))) fail()
  if (value.status === 'AVAILABLE' && (value.available_shots !== value.total_shots
      || value.retrying_shots !== 0 || !value.report_available)) fail()
  if (value.status === 'PARTIAL' && (value.available_shots < 1
      || value.available_shots >= value.total_shots || !value.report_available)) fail()
  if (value.status === 'FAILED_RETRYING' && (value.available_shots !== 0
      || value.retrying_shots < 1 || value.report_available)) fail()
}

async function putSessionAnalysis({
  sessionId, projection, tableId, deviceId, idempotencyKey, repo,
}) {
  if (!ID_PATTERN.test(idempotencyKey || '') || !ID_PATTERN.test(tableId || '')
      || !ID_PATTERN.test(deviceId || '')) fail()
  validateProjection(sessionId, projection)
  const candidateDigest = canonicalDigest(projection)
  const work = async transaction => {
    const authority = await transaction.getSessionAuthority(sessionId)
    if (!authority || authority.session_id !== sessionId || authority.status !== 'ended'
        || authority.table_id !== tableId || authority.device_id !== deviceId) {
      fail('SYNC_CONFLICT')
    }
    if (projection.report_available || projection.published_shots.length
        || projection.published_clips.length) {
      const publishedSession = await transaction.getPublishedSession(sessionId)
      if (!publishedSession || publishedSession.session_id !== sessionId
          || publishedSession.sync_status !== 'published'
          || !ID_PATTERN.test(publishedSession.artifact_id || '')
          || !Array.isArray(publishedSession.shots)) fail('SYNC_CONFLICT')
      const canonicalShots = new Set(publishedSession.shots.map(shot => shot.shot_id))
      if (canonicalShots.size !== projection.frozen_shot_ids.length
          || projection.frozen_shot_ids.some(shotId => !canonicalShots.has(shotId))) {
        fail('SYNC_CONFLICT')
      }
      const clipsByShot = new Set()
      for (const reference of projection.published_clips) {
        const clip = await transaction.getPublishedClip(reference.clip_id)
        if (!clip || clip.clip_id !== reference.clip_id || clip.session_id !== sessionId
            || clip.shot_id !== reference.shot_id
            || clip.artifact_id !== publishedSession.artifact_id
            || clip.upload_status !== 'published') fail('SYNC_CONFLICT')
        clipsByShot.add(reference.shot_id)
      }
      if (projection.published_shots.some(reference => !canonicalShots.has(reference.shot_id)
          || !clipsByShot.has(reference.shot_id))) fail('SYNC_CONFLICT')
    }
    const existing = await transaction.getProjection(sessionId)
    if (existing) {
      if (existing._table_id !== tableId || existing._device_id !== deviceId
          || projection.revision < existing.revision
          || projection.frozen_shots_digest !== existing.frozen_shots_digest) {
        fail('SYNC_CONFLICT')
      }
      if (projection.revision === existing.revision) {
        if (existing._content_digest !== candidateDigest) fail('SYNC_CONFLICT')
        return publicProjection(existing)
      }
    }
    const stored = {
      ...projection,
      _table_id: tableId,
      _device_id: deviceId,
      _idempotency_key: idempotencyKey,
      _content_digest: candidateDigest,
    }
    await transaction.saveProjection(stored)
    return publicProjection(stored)
  }
  return repo.runTransaction ? repo.runTransaction(work) : work(repo)
}

async function optional(reference) {
  try { return await reference.get() } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error && error.message)) return { data: null }
    throw error
  }
}

function createCloudSessionAnalysisRepository(db) {
  const adapter = transaction => ({
    async getSessionAuthority(sessionId) {
      return (await optional(transaction.collection('field_live_sessions').doc(sessionId))).data || null
    },
    async getProjection(sessionId) {
      return (await optional(transaction.collection('session_analysis_projections').doc(sessionId))).data || null
    },
    async getPublishedSession(sessionId) {
      return (await optional(transaction.collection('sessions').doc(sessionId))).data || null
    },
    async getPublishedClip(clipId) {
      return (await optional(transaction.collection('clips').doc(clipId))).data || null
    },
    async getLiveShotFact(idempotencyKey) {
      return (await optional(transaction.collection('field_live_shot_facts')
        .doc(idempotencyKey))).data || null
    },
    async getLiveShotProgress(sessionId) {
      return (await optional(transaction.collection('field_live_shot_progress')
        .doc(sessionId))).data || null
    },
    async saveLiveShotFact(value) {
      await transaction.collection('field_live_shot_facts')
        .doc(value.idempotency_key).set({ data: value })
      return value
    },
    async saveLiveShotProgress(value) {
      await transaction.collection('field_live_shot_progress')
        .doc(value.session_id).set({ data: value })
      return value
    },
    async saveProjection(value) {
      await transaction.collection('session_analysis_projections').doc(value.session_id).set({ data: value })
      return value
    },
  })
  return {
    runTransaction: callback => db.runTransaction(transaction => callback(adapter(transaction))),
  }
}

module.exports = {
  createCloudSessionAnalysisRepository,
  publicProjection,
  putSessionAnalysis,
  putLiveShotFact,
  publicLiveShotFact,
  validateLiveShotFact,
  validateProjection,
}
