const crypto = require('crypto')

const { reduceReviewEvents } = require('./review-events')


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


function createMatchReviewRepository(db, { now }) {
  const repository = {
    async getState(matchId, resultRevision) {
      const stored = await optional(db.collection('match_review_states')
        .doc(digest(`${matchId}:${resultRevision}`)))
      return stored ? structuredClone(stored.state) : null
    },
    async open({ match_id: matchId, result_revision: resultRevision, reason_codes: reasonCodes }) {
      if (typeof matchId !== 'string' || !Number.isInteger(resultRevision)
          || !Array.isArray(reasonCodes) || reasonCodes.length < 2) {
        throw new Error('MATCH_REVIEW_INVALID')
      }
      const reasons = [...new Set(reasonCodes)].sort()
      const stateId = digest(`${matchId}:${resultRevision}`)
      return db.runTransaction(async transaction => {
        const stateRef = transaction.collection('match_review_states').doc(stateId)
        const existing = await optional(stateRef)
        if (existing) {
          if (JSON.stringify(existing.reason_codes) !== JSON.stringify(reasons)) {
            throw new Error('MATCH_REVIEW_CONFLICT')
          }
          return clean(existing).state
        }
        const occurredAt = now()
        const event = {
          schema_version: '1.0',
          event_id: `review-${digest(`${matchId}:${resultRevision}:opened`).slice(0, 48)}`,
          match_id: matchId, sequence: 1, event_type: 'review_opened', actor_id: 'system',
          occurred_at: occurredAt, payload: { reason_codes: reasons },
        }
        const state = reduceReviewEvents([event])
        await transaction.collection('match_review_events').doc(digest(event.event_id)).set({
          data: { ...event, result_revision: resultRevision, recorded_at: occurredAt },
        })
        await stateRef.set({ data: {
          schema_version: '1.0', match_id: matchId, result_revision: resultRevision,
          reason_codes: reasons, event_ids: [event.event_id], state, updated_at: occurredAt,
        } })
        return structuredClone(state)
      })
    },
    async appendEvent({ matchId, resultRevision, eventType, actorId, payload }) {
      const stateId = digest(`${matchId}:${resultRevision}`)
      return db.runTransaction(async transaction => {
        const stateRef = transaction.collection('match_review_states').doc(stateId)
        const storedState = await optional(stateRef)
        if (!storedState || storedState.match_id !== matchId
            || storedState.result_revision !== resultRevision) {
          throw new Error('MATCH_REVIEW_NOT_FOUND')
        }
        const events = []
        for (const storedEventId of storedState.event_ids) {
          const document = await optional(
            transaction.collection('match_review_events').doc(digest(storedEventId)),
          )
          if (!document) throw new Error('MATCH_REVIEW_CONFLICT')
          const {
            _id: _documentId, result_revision: _resultRevision,
            recorded_at: _recordedAt, ...event
          } = document
          events.push(event)
        }
        const eventDigest = digest(JSON.stringify({ matchId, resultRevision, eventType, actorId, payload }))
        const eventId = `review-${eventDigest.slice(0, 48)}`
        const duplicate = events.find(item => item.event_id === eventId)
        if (duplicate) return structuredClone(storedState.state)
        const occurredAt = now()
        const event = {
          schema_version: '1.0', event_id: eventId, match_id: matchId,
          sequence: events.length + 1, event_type: eventType, actor_id: actorId,
          occurred_at: occurredAt, payload: structuredClone(payload),
        }
        const state = reduceReviewEvents([...events, event])
        await transaction.collection('match_review_events').doc(digest(eventId)).set({
          data: { ...event, result_revision: resultRevision, recorded_at: occurredAt },
        })
        await stateRef.set({ data: {
          ...clean(storedState), event_ids: [...storedState.event_ids, eventId],
          state, updated_at: occurredAt,
        } })
        return structuredClone(state)
      })
    },
  }
  return repository
}


module.exports = { createMatchReviewRepository }
