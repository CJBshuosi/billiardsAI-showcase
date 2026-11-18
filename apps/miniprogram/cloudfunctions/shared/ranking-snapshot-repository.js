const crypto = require('crypto')


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
  const { _id: _documentId, ...document } = value
  return structuredClone(document)
}


async function queryAll(buildQuery) {
  const values = []
  for (let offset = 0; ; offset += 100) {
    const result = await buildQuery().skip(offset).limit(100).get()
    const page = result.data || []
    values.push(...page.map(clean))
    if (page.length < 100) return values
  }
}


function snapshotDocument(snapshot, snapshotSetId, version, status, now) {
  const { entries: _entries, ...metadata } = snapshot
  return {
    ...metadata, snapshot_version: version, snapshot_set_id: snapshotSetId,
    status, updated_at: now,
  }
}


function createRankingSnapshotRepository(db, { now }) {
  return {
    async readPlayers(rankingScopeId, isTest) {
      const profiles = await queryAll(() => db.collection('public_ranking_profiles')
        .where({ is_public: true }))
      const players = []
      for (const profile of profiles) {
        const state = await optional(db.collection('player_skill_states')
          .doc(digest(`${rankingScopeId}:${profile.user_id}`)))
        if (state && state.is_test === isTest) players.push({
          user_id: profile.user_id, public_profile: profile, long_term: clean(state),
        })
      }
      return players
    },

    async readRatingEvents(rankingScopeId, isTest) {
      return queryAll(() => db.collection('rating_events')
        .where({ ranking_scope_id: rankingScopeId, is_test: isTest, status: 'active' })
        .orderBy('scope_sequence', 'asc'))
    },

    async publish(payload) {
      const timestamp = now()
      const headId = digest(payload.ranking_scope_id)
      const reserved = await db.runTransaction(async transaction => {
        const headRef = transaction.collection('ranking_snapshot_heads').doc(headId)
        const head = await optional(headRef)
        if (head && head.is_test !== payload.is_test) {
          throw new Error('RANKING_ENVIRONMENT_SCOPE_MISMATCH')
        }
        if (head && head.published_snapshot_set_id === payload.snapshot_set_id) {
          return { already_published: true, version: head.published_version }
        }
        if (head && head.building_snapshot_set_id
            && head.building_snapshot_set_id !== payload.snapshot_set_id) {
          throw new Error('RANKING_SNAPSHOT_BUILD_CONFLICT')
        }
        const version = head && head.building_version
          || head && head.next_version || 1
        await headRef.set({ data: {
          schema_version: '1.0', ranking_scope_id: payload.ranking_scope_id,
          is_test: payload.is_test,
          next_version: Math.max(version + 1, head && head.next_version || 1),
          building_snapshot_set_id: payload.snapshot_set_id, building_version: version,
          published_snapshot_set_id: head && head.published_snapshot_set_id || null,
          published_version: head && head.published_version || null,
          updated_at: timestamp,
        } })
        for (const snapshot of [payload.long_term, payload.season]) {
          await transaction.collection('ranking_snapshots').doc(digest(snapshot.snapshot_id)).set({
            data: snapshotDocument(snapshot, payload.snapshot_set_id, version, 'building', timestamp),
          })
        }
        return { already_published: false, version }
      })
      if (reserved.already_published) {
        return { snapshot_set_id: payload.snapshot_set_id, snapshot_version: reserved.version }
      }

      for (const snapshot of [payload.long_term, payload.season]) {
        for (const [index, entry] of snapshot.entries.entries()) {
          await db.collection('ranking_snapshot_entries')
            .doc(digest(`${snapshot.snapshot_id}:${entry.user_id}`)).set({ data: {
              schema_version: '1.0', snapshot_id: snapshot.snapshot_id,
              snapshot_version: reserved.version, is_test: payload.is_test,
              position: index + 1,
              ...structuredClone(entry),
            } })
        }
      }

      await db.runTransaction(async transaction => {
        const headRef = transaction.collection('ranking_snapshot_heads').doc(headId)
        const head = await optional(headRef)
        if (!head || head.building_snapshot_set_id !== payload.snapshot_set_id
            || head.building_version !== reserved.version) {
          throw new Error('RANKING_SNAPSHOT_BUILD_CONFLICT')
        }
        for (const snapshot of [payload.long_term, payload.season]) {
          await transaction.collection('ranking_snapshots').doc(digest(snapshot.snapshot_id)).update({
            data: { status: 'published', updated_at: timestamp },
          })
        }
        await headRef.set({ data: {
          ...clean(head), building_snapshot_set_id: null, building_version: null,
          published_snapshot_set_id: payload.snapshot_set_id,
          published_version: reserved.version, updated_at: timestamp,
        } })
      })
      return { snapshot_set_id: payload.snapshot_set_id, snapshot_version: reserved.version }
    },
  }
}


module.exports = { createRankingSnapshotRepository }
