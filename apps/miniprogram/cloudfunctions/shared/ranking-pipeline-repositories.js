const crypto = require('crypto')


class RankingDecisionError extends Error {
  constructor(code) {
    super(code)
    this.name = 'RankingDecisionError'
    this.code = code
  }
}


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


function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}


function createRankingAuthorityRepository(db) {
  return {
    async readForMatch(projection) {
      const roleId = role => digest(`${projection.session_id}:${role}`)
      const [scope, venue, table, session, publishedSession, red, blue] = await Promise.all([
        optional(db.collection('ranking_scopes').doc(digest(projection.ranking_scope_id))),
        optional(db.collection('ranking_venues').doc(digest(projection.venue_id))),
        optional(db.collection('ranking_tables').doc(digest(projection.table_id))),
        optional(db.collection('field_live_sessions').doc(projection.session_id)),
        optional(db.collection('sessions').doc(projection.session_id)),
        optional(db.collection('session_role_bindings').doc(roleId('red'))),
        optional(db.collection('session_role_bindings').doc(roleId('blue'))),
      ])
      const sessionDocument = clean(session)
      if (sessionDocument) {
        const published = clean(publishedSession)
        const replayLimited = published
          && Array.isArray(published.limitations)
          && published.limitations.includes('development_replay_media')
        const publicationMatches = published
          && published.session_id === projection.session_id
          && published.sync_status === 'published'
          && published._sync_publication
          && published._sync_publication.table_id === projection.table_id
          && published._sync_publication.device_id === sessionDocument.device_id
          && /^[0-9a-f]{64}$/u.test(published._sync_publication.content_sha256 || '')
        sessionDocument.evidence_source_kind = publicationMatches
          && ((published._evidence_source_kind === 'usb_live' && !replayLimited)
            || (published._evidence_source_kind === 'development_replay' && replayLimited))
          ? published._evidence_source_kind : null
        sessionDocument.role_bindings = { red: clean(red), blue: clean(blue) }
        sessionDocument.device_deployment_status = table
          && table.device_deployment_status === 'approved'
          && typeof table.device_id === 'string'
          && table.device_id === sessionDocument.device_id
          ? 'approved' : 'unapproved'
      }
      return {
        configuration_version: scope && scope.configuration_version || null,
        scope: clean(scope), venue: clean(venue), table: clean(table), session: sessionDocument,
      }
    },
  }
}


function createDecisionRepository(db, { collectionName, now }) {
  return {
    async save(decision) {
      if (!decision || typeof decision.match_id !== 'string'
          || !Number.isInteger(decision.result_revision)) {
        throw new RankingDecisionError('RANKING_DECISION_INVALID')
      }
      const documentId = digest(`${decision.match_id}:${decision.result_revision}`)
      return db.runTransaction(async transaction => {
        const ref = transaction.collection(collectionName).doc(documentId)
        const existing = await optional(ref)
        if (existing) {
          const { _id: _id, decided_at: _decidedAt, assessed_at: _assessedAt, ...stored } = existing
          if (stable(stored) !== stable(decision)) {
            throw new RankingDecisionError('RANKING_DECISION_CONFLICT')
          }
          return clean(existing)
        }
        const timestampKey = collectionName === 'match_risk_assessments' ? 'assessed_at' : 'decided_at'
        const stored = { ...structuredClone(decision), [timestampKey]: now() }
        await ref.set({ data: stored })
        return structuredClone(stored)
      })
    },
  }
}


function createRankingDecisionRepositories(db, { now, detectRiskSignals }) {
  if (typeof now !== 'function' || typeof detectRiskSignals !== 'function') {
    throw new RankingDecisionError('RANKING_DECISION_CONFIGURATION_INVALID')
  }
  return {
    qualificationRepository: createDecisionRepository(db, {
      collectionName: 'match_qualification_decisions', now,
    }),
    riskRepository: {
      ...createDecisionRepository(db, { collectionName: 'match_risk_assessments', now }),
      async collectSignals(context) {
        const signals = await detectRiskSignals(context)
        return structuredClone(signals)
      },
    },
  }
}


module.exports = {
  RankingDecisionError,
  createRankingAuthorityRepository,
  createRankingDecisionRepositories,
}
