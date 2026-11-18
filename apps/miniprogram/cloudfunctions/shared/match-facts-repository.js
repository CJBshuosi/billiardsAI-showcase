const crypto = require('crypto')

const {
  MatchFactsError,
  reduceMatchEvents,
} = require('./match-facts')


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


function matchDocumentId(matchId) {
  return digest(matchId)
}


function eventDocumentId(eventId) {
  return digest(eventId)
}


function clone(value) {
  return structuredClone(value)
}


function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`
  if (value !== null && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${stableJson(value[key])}`
    )).join(',')}}`
  }
  return JSON.stringify(value)
}


function conflict() {
  throw new MatchFactsError('MATCH_FACTS_CONFLICT')
}


function authorityTimestamp(value) {
  if (typeof value !== 'string'
      || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u.test(value)) {
    throw new MatchFactsError('MATCH_FACTS_INVALID')
  }
  const milliseconds = Date.parse(value)
  if (!Number.isFinite(milliseconds) || new Date(milliseconds).toISOString() !== value) {
    throw new MatchFactsError('MATCH_FACTS_INVALID')
  }
  return value
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


function eventFromDocument(document) {
  const { _id: _documentId, recorded_at: _recordedAt, ...event } = document
  return clone(event)
}


function projectionFromDocument(document) {
  return clone(document.projection)
}


function indexFields(projection) {
  return {
    red_user_id: projection.participants.red,
    blue_user_id: projection.participants.blue,
    table_id: projection.table_id,
    started_at: projection.started_at,
    status: projection.status,
    result_confirmation_deadline_at: projection.result_confirmation_deadline_at,
  }
}


async function loadMatch(scope, matchId) {
  const match = await optional(scope.collection('matches').doc(matchDocumentId(matchId)))
  if (!match) return null
  if (match.schema_version !== '1.0' || match.match_id !== matchId
      || !Array.isArray(match.event_ids) || match.event_ids.length === 0
      || match.latest_sequence !== match.event_ids.length) conflict()
  const events = []
  for (const eventId of match.event_ids || []) {
    const document = await optional(scope.collection('match_events').doc(eventDocumentId(eventId)))
    if (!document) conflict()
    const event = eventFromDocument(document)
    if (event.event_id !== eventId || event.match_id !== matchId) conflict()
    events.push(event)
  }
  const replayed = reduceMatchEvents(events)
  if (replayed.latest_sequence !== match.latest_sequence
      || replayed.session_id !== match.session_id
      || stableJson(replayed) !== stableJson(match.projection)) conflict()
  return { projection: projectionFromDocument(match), events }
}


function createMatchFactsRepository(db, { now }) {
  return {
    async appendEvent(inputEvent) {
      const event = clone(inputEvent)
      const recordedAt = authorityTimestamp(now())
      return db.runTransaction(async (transaction) => {
        const eventRef = transaction.collection('match_events').doc(eventDocumentId(event.event_id))
        const matchRef = transaction.collection('matches').doc(matchDocumentId(event.match_id))
        const existingEvent = await optional(eventRef)
        if (existingEvent) {
          if (stableJson(eventFromDocument(existingEvent)) !== stableJson(event)) conflict()
          const existingMatch = await loadMatch(transaction, event.match_id)
          if (!existingMatch) conflict()
          return clone(existingMatch.projection)
        }

        const currentMatchDocument = await optional(matchRef)
        if (currentMatchDocument) {
          if (event.sequence === 1) conflict()
          const current = await loadMatch(transaction, event.match_id)
          const projection = reduceMatchEvents([...current.events, event])
          await eventRef.set({ data: { ...event, recorded_at: recordedAt } })
          await matchRef.set({ data: {
            schema_version: '1.0',
            match_id: event.match_id,
            session_id: projection.session_id,
            event_ids: [...currentMatchDocument.event_ids, event.event_id],
            latest_sequence: projection.latest_sequence,
            ...indexFields(projection),
            projection,
            created_at: currentMatchDocument.created_at,
            updated_at: recordedAt,
          } })
          return clone(projection)
        }

        const projection = reduceMatchEvents([event])
        await eventRef.set({ data: { ...event, recorded_at: recordedAt } })
        await matchRef.set({ data: {
          schema_version: '1.0',
          match_id: event.match_id,
          session_id: projection.session_id,
          event_ids: [event.event_id],
          latest_sequence: projection.latest_sequence,
          ...indexFields(projection),
          projection,
          created_at: recordedAt,
          updated_at: recordedAt,
        } })
        return clone(projection)
      })
    },

    async getMatch(matchId) {
      return db.runTransaction(transaction => loadMatch(transaction, matchId))
    },

    async getBySession(sessionId) {
      const result = await db.collection('matches').where({ session_id: sessionId }).limit(2).get()
      const documents = result.data || []
      if (documents.length === 0) return null
      if (documents.length !== 1 || documents[0].session_id !== sessionId) conflict()
      return db.runTransaction(transaction => loadMatch(transaction, documents[0].match_id))
    },

    async listPendingForUser(userId) {
      const [red, blue] = await Promise.all([
        db.collection('matches').where({
          red_user_id: userId, status: 'awaiting_result_confirmations',
        }).limit(50).get(),
        db.collection('matches').where({
          blue_user_id: userId, status: 'awaiting_result_confirmations',
        }).limit(50).get(),
      ])
      const documents = new Map()
      for (const document of [...(red.data || []), ...(blue.data || [])]) {
        documents.set(document.match_id, document)
      }
      const matches = []
      for (const document of documents.values()) {
        matches.push(await db.runTransaction(transaction => loadMatch(transaction, document.match_id)))
      }
      return matches.sort((left, right) => (
        left.projection.result_confirmation_deadline_at
          .localeCompare(right.projection.result_confirmation_deadline_at)
      ))
    },
  }
}


module.exports = {
  createMatchFactsRepository,
  eventDocumentId,
  matchDocumentId,
}
