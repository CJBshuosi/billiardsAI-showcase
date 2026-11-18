const crypto = require('crypto')

const { placementProgress } = require('../../shared/rating-engine')
const {
  assertRankingScopePolicy,
  resolveRuntimeRankingScope,
} = require('../../shared/ranking-scope-policy')
const { ClientApiError } = require('./errors')


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


async function optional(ref) {
  try {
    const result = await ref.get()
    return result && result.data ? result.data : null
  } catch (error) {
    if (String(error && error.message).includes('does not exist')) return null
    throw error
  }
}


function clean(value) {
  if (!value) return null
  const { _id: _documentId, nickname_key: _nicknameKey, ...document } = value
  return structuredClone(document)
}


function placement(skill) {
  if (!skill) {
    return {
      status: 'unplaced', rated_matches: 0, distinct_opponents: 0,
      required_matches: 3, required_opponents: 3,
    }
  }
  const progress = placementProgress(skill)
  if (!progress.is_placed) {
    return {
      status: 'unplaced', rated_matches: progress.rated_matches,
      distinct_opponents: progress.distinct_opponents,
      required_matches: 3,
      required_opponents: 3,
    }
  }
  return { status: 'placed', display_rating: Math.round(skill.rating) }
}


function validateScopeAndKind(rankingScopeId, kind) {
  if (typeof rankingScopeId !== 'string' || rankingScopeId.length === 0
      || !['long_term', 'season'].includes(kind)) {
    throw new ClientApiError('RANKING_REQUEST_INVALID')
  }
}


function scopeMetadata(scope) {
  if (!scope) return undefined
  return {
    ranking_scope_id: scope.ranking_scope_id,
    display_name: scope.display_name,
    mode: scope.mode,
    is_test: scope.is_test,
  }
}


async function loadRuntimeRankingScope({ environment, repo }) {
  const expected = resolveRuntimeRankingScope({ environment })
  const scope = await repo.getScope(expected.ranking_scope_id)
  assertRankingScopePolicy({ environment, scope })
  return scope
}


async function getCityRanking({
  userId, rankingScopeId, rankingScope, kind, page = 1, pageSize = 50, repo,
}) {
  validateScopeAndKind(rankingScopeId, kind)
  if (!Number.isInteger(page) || page < 1 || !Number.isInteger(pageSize)
      || pageSize < 1 || pageSize > 50) throw new ClientApiError('RANKING_REQUEST_INVALID')
  const [snapshot, skill] = await Promise.all([
    repo.getLatestSnapshot(rankingScopeId, kind, { offset: (page - 1) * pageSize, limit: pageSize }),
    repo.getSkillState(rankingScopeId, userId),
  ])
  const myStatus = placement(skill)
  if (!snapshot) {
    return {
      availability: 'unavailable', reason_code: 'RANKING_NOT_PUBLISHED',
      ranking_scope_id: rankingScopeId, kind, entries: [], my_position: null,
      my_status: myStatus, ranking_scope: scopeMetadata(rankingScope),
    }
  }
  if (snapshot.ranking_scope_id !== rankingScopeId || snapshot.kind !== kind
      || !Array.isArray(snapshot.entries)) {
    throw new ClientApiError('RANKING_SNAPSHOT_INVALID')
  }
  const entries = snapshot.entries.map(entry => ({
    rank: entry.rank, user_id: entry.user_id, nickname: entry.nickname,
    display_rating: entry.display_rating,
    recent_sample_limited: entry.recent_sample_limited === true,
  }))
  const pagePosition = entries.find(entry => entry.user_id === userId) || null
  const storedPosition = !pagePosition && typeof repo.getSnapshotEntry === 'function'
    ? await repo.getSnapshotEntry(snapshot.snapshot_id, userId) : null
  const myPosition = pagePosition || (storedPosition ? {
    rank: storedPosition.rank, user_id: storedPosition.user_id,
    nickname: storedPosition.nickname, display_rating: storedPosition.display_rating,
    recent_sample_limited: storedPosition.recent_sample_limited === true,
  } : null)
  return {
    availability: 'published', snapshot_id: snapshot.snapshot_id,
    snapshot_version: snapshot.snapshot_version, ranking_scope_id: rankingScopeId,
    kind, season_id: snapshot.season_id || null, as_of: snapshot.as_of,
    total_entries: snapshot.total_entries, page, page_size: pageSize,
    has_more: page * pageSize < snapshot.total_entries, entries,
    my_position: myPosition ? structuredClone(myPosition) : null,
    my_status: myPosition
      ? { status: 'placed', display_rating: myPosition.display_rating }
      : myStatus,
    ranking_scope: scopeMetadata(rankingScope),
  }
}


async function getPublicRankingPlayer({
  targetUserId, rankingScopeId, rankingScope, snapshotId, repo,
}) {
  if (typeof targetUserId !== 'string' || targetUserId.length === 0
      || typeof rankingScopeId !== 'string' || rankingScopeId.length === 0
      || (snapshotId != null && (typeof snapshotId !== 'string' || snapshotId.length === 0))) {
    throw new ClientApiError('RANKING_REQUEST_INVALID')
  }
  const [profile, skill] = await Promise.all([
    repo.getPublicProfile(targetUserId),
    repo.getSkillState(rankingScopeId, targetUserId),
  ])
  if (!profile || profile.is_public !== true) throw new ClientApiError('RANKING_PLAYER_UNAVAILABLE')
  const progress = placement(skill)
  const result = {
    user_id: targetUserId, nickname: profile.nickname,
    accept_challenges: profile.accept_challenges === true,
    ranking_status: progress.status,
    ranking_scope: scopeMetadata(rankingScope),
  }
  if (progress.status === 'unplaced') {
    return {
      ...result, rated_matches: progress.rated_matches,
      distinct_opponents: progress.distinct_opponents,
      required_matches: progress.required_matches,
      required_opponents: progress.required_opponents,
    }
  }
  const snapshot = snapshotId
    ? await repo.getSnapshotById(snapshotId) : await repo.getLatestSnapshot(rankingScopeId, 'long_term')
  if (snapshot && (snapshot.ranking_scope_id !== rankingScopeId
      || snapshot.kind !== 'long_term' || (snapshot.status && snapshot.status !== 'published'))) {
    throw new ClientApiError('RANKING_SNAPSHOT_INVALID')
  }
  const pageEntry = snapshot && snapshot.entries.find(item => item.user_id === targetUserId)
  const entry = pageEntry || (snapshot && typeof repo.getSnapshotEntry === 'function'
    ? await repo.getSnapshotEntry(snapshot.snapshot_id, targetUserId) : null)
  if (!entry) throw new ClientApiError('RANKING_NOT_PUBLISHED')
  return {
    ...result, rank: entry.rank, display_rating: entry.display_rating,
    recent_sample_limited: entry.recent_sample_limited === true,
    snapshot_id: snapshot.snapshot_id, snapshot_version: snapshot.snapshot_version,
  }
}


function createCloudRankingRepository(db) {
  async function withEntries(snapshot, { offset = 0, limit = 50 } = {}) {
    if (!snapshot) return null
    const entriesResult = await db.collection('ranking_snapshot_entries')
      .where({ snapshot_id: snapshot.snapshot_id }).orderBy('position', 'asc')
      .skip(offset).limit(limit).get()
    return { ...clean(snapshot), entries: (entriesResult.data || []).map(clean) }
  }
  return {
    async getScope(rankingScopeId) {
      return clean(await optional(
        db.collection('ranking_scopes').doc(digest(rankingScopeId)),
      ))
    },
    async getTable(tableId) {
      return clean(await optional(db.collection('ranking_tables').doc(digest(tableId))))
    },
    async getVenue(venueId) {
      return clean(await optional(db.collection('ranking_venues').doc(digest(venueId))))
    },
    async listApprovedVenues(rankingScopeId) {
      const result = await db.collection('ranking_venues')
        .where({ ranking_scope_id: rankingScopeId, status: 'approved' })
        .orderBy('display_name', 'asc').get()
      return (result.data || []).map(item => {
        const venue = clean(item)
        return { venue_id: venue.venue_id, display_name: venue.display_name }
      })
    },
    async getLatestSnapshot(rankingScopeId, kind, paging) {
      const result = await db.collection('ranking_snapshots')
        .where({ ranking_scope_id: rankingScopeId, kind, status: 'published' })
        .orderBy('as_of', 'desc').limit(1).get()
      const snapshot = result.data && result.data[0]
      return withEntries(snapshot, paging)
    },
    async getSnapshotById(snapshotId) {
      return withEntries(await optional(
        db.collection('ranking_snapshots').doc(digest(snapshotId)),
      ))
    },
    async getSnapshotEntry(snapshotId, userId) {
      const result = await db.collection('ranking_snapshot_entries')
        .where({ snapshot_id: snapshotId, user_id: userId }).limit(1).get()
      return clean(result.data && result.data[0])
    },
    async getSkillState(rankingScopeId, userId) {
      return clean(await optional(
        db.collection('player_skill_states').doc(digest(`${rankingScopeId}:${userId}`)),
      ))
    },
    async getPublicProfile(userId) {
      return clean(await optional(
        db.collection('public_ranking_profiles').doc(digest(userId)),
      ))
    },
  }
}


module.exports = {
  createCloudRankingRepository,
  getCityRanking,
  getPublicRankingPlayer,
  loadRuntimeRankingScope,
}
