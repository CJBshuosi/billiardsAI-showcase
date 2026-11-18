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
  const { _id: _documentId, ...document } = value
  return structuredClone(document)
}


function createRatingReplayRepository(db, { now }) {
  return {
    async claimNext(rankingScopeId) {
      const result = await db.collection('rating_replay_requests')
        .where({ ranking_scope_id: rankingScopeId, status: 'pending' })
        .orderBy('from_scope_sequence', 'asc').limit(1).get()
      const candidate = result.data && result.data[0]
      if (!candidate) return null
      return db.runTransaction(async transaction => {
        const requestRef = transaction.collection('rating_replay_requests')
          .doc(candidate.replay_request_id)
        const request = await optional(requestRef)
        if (!request || request.status !== 'pending') return null
        const headRef = transaction.collection('rating_scope_heads').doc(digest(rankingScopeId))
        const head = await optional(headRef)
        if (!head || head.active_generation_id !== request.source_generation_id
            || head.replay_status === 'building') throw new Error('RATING_REPLAY_CONFLICT')
        const claimedAt = now()
        await requestRef.update({ data: { status: 'processing', claimed_at: claimedAt } })
        await headRef.update({ data: {
          replay_status: 'building', replay_request_id: request.replay_request_id,
          updated_at: claimedAt,
        } })
        return { ...clean(request), status: 'processing', claimed_at: claimedAt }
      })
    },

    async loadSourceMatches(request) {
      const result = await db.collection('rating_events')
        .where({ generation_id: request.source_generation_id })
        .orderBy('scope_sequence', 'asc').limit(1000).get()
      return (result.data || []).map(event => ({
        schema_version: '1.0', match_id: event.match_id,
        result_revision: event.result_revision, ranking_scope_id: event.ranking_scope_id,
        scope_sequence: event.scope_sequence, settled_at: event.settled_at,
        started_at: event.started_at,
        red_user_id: event.red_user_id, blue_user_id: event.blue_user_id,
        winner_role: event.winner_role, format: event.format,
        status: event.status === 'revoked' ? 'revoked' : 'active',
      }))
    },

    async commit({ request, generation_id: generationId, replay, matches }) {
      const completedAt = now()
      return db.runTransaction(async transaction => {
        const requestRef = transaction.collection('rating_replay_requests')
          .doc(request.replay_request_id)
        const storedRequest = await optional(requestRef)
        const headRef = transaction.collection('rating_scope_heads')
          .doc(digest(request.ranking_scope_id))
        const head = await optional(headRef)
        if (!storedRequest || storedRequest.status !== 'processing' || !head
            || head.replay_request_id !== request.replay_request_id) {
          throw new Error('RATING_REPLAY_CONFLICT')
        }
        await transaction.collection('rating_generations').doc(generationId).set({ data: {
          schema_version: '1.0', generation_id: generationId,
          ranking_scope_id: request.ranking_scope_id,
          algorithm_version: replay.algorithm_version, status: 'active',
          source_generation_id: request.source_generation_id,
          replay_request_id: request.replay_request_id, created_at: completedAt,
        } })
        const storedEvents = new Map()
        for (const event of replay.events) {
          const ratingEventId = digest(`${generationId}:${event.rating_event_id}`)
          const stored = { ...event, rating_event_id: ratingEventId, generation_id: generationId }
          storedEvents.set(event.match_id, stored)
          await transaction.collection('rating_events').doc(ratingEventId).set({ data: stored })
        }
        for (const [userId, state] of Object.entries(replay.states)) {
          await transaction.collection('player_skill_states')
            .doc(digest(`${request.ranking_scope_id}:${userId}`)).set({ data: {
              ...state, ranking_scope_id: request.ranking_scope_id,
              generation_id: generationId,
              updated_sequence: Math.max(0, ...replay.events.map(event => event.scope_sequence)),
            } })
        }
        for (const match of matches) {
          const consumptionRef = transaction.collection('rating_match_consumptions')
            .doc(digest(`${request.ranking_scope_id}:${match.match_id}`))
          const consumption = await optional(consumptionRef)
          if (!consumption) throw new Error('RATING_REPLAY_CONFLICT')
          const event = storedEvents.get(match.match_id)
          await consumptionRef.set({ data: {
            ...clean(consumption), generation_id: generationId,
            rating_event_id: event ? event.rating_event_id : null,
            result_revision: match.result_revision,
            status: event ? 'active' : 'revoked',
            replay_request_id: request.replay_request_id,
            replayed_at: completedAt,
          } })
        }
        const oldGenerationRef = transaction.collection('rating_generations')
          .doc(request.source_generation_id)
        await oldGenerationRef.update({ data: { status: 'superseded', superseded_at: completedAt } })
        await headRef.set({ data: {
          ...clean(head), active_generation_id: generationId,
          replay_status: 'idle', replay_request_id: null, updated_at: completedAt,
        } })
        await requestRef.update({ data: {
          status: 'completed', target_generation_id: generationId, completed_at: completedAt,
        } })
        return { generation_id: generationId }
      })
    },

    async fail(request, error) {
      const failedAt = now()
      await db.runTransaction(async transaction => {
        await transaction.collection('rating_replay_requests').doc(request.replay_request_id).update({
          data: { status: 'failed', failure_code: error.code || error.message, failed_at: failedAt },
        })
        const headRef = transaction.collection('rating_scope_heads')
          .doc(digest(request.ranking_scope_id))
        const head = await optional(headRef)
        if (head && head.replay_request_id === request.replay_request_id) {
          await headRef.update({ data: { replay_status: 'idle', replay_request_id: null, updated_at: failedAt } })
        }
      })
    },
  }
}


function createRatingReviewRepository(db) {
  return {
    async findActive({ rankingScopeId, matchId }) {
      const head = await optional(db.collection('rating_scope_heads').doc(digest(rankingScopeId)))
      if (!head) return null
      const result = await db.collection('rating_events').where({
        generation_id: head.active_generation_id, match_id: matchId,
      }).limit(2).get()
      const events = result.data || []
      if (events.length > 1) throw new Error('RATING_REVIEW_CONFLICT')
      return events[0] ? clean(events[0]) : null
    },
    async saveReplayRequest(request) {
      return db.runTransaction(async transaction => {
        const ref = transaction.collection('rating_replay_requests').doc(request.replay_request_id)
        const existing = await optional(ref)
        if (existing) return clean(existing)
        await ref.set({ data: structuredClone(request) })
        return structuredClone(request)
      })
    },
  }
}


module.exports = { createRatingReplayRepository, createRatingReviewRepository }
