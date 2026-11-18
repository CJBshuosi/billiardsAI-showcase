const crypto = require('crypto')

const {
  ALGORITHM_VERSION,
  initialSkillState,
  updateMatchRatings,
} = require('./rating-engine')
const { ratingEventId } = require('./rating-ledger')


class RatingSettlementError extends Error {
  constructor(code) {
    super(code)
    this.name = 'RatingSettlementError'
    this.code = code
  }
}


function fail(code) {
  throw new RatingSettlementError(code)
}


function digest(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex')
}


function canonicalInput(qualification, match) {
  return JSON.stringify({
    match_id: match.match_id,
    result_revision: match.result_revision,
    ranking_scope_id: match.ranking_scope_id,
    is_test: match.is_test,
    red_user_id: match.red_user_id,
    blue_user_id: match.blue_user_id,
    winner_role: match.winner_role,
    format: match.format,
    started_at: match.started_at,
    settled_at: match.settled_at,
    configuration_version: qualification.configuration_version,
  })
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


function withoutId(value) {
  const { _id: _documentId, ...document } = value
  return structuredClone(document)
}


function skillFromDocument(value) {
  const {
    _id: _documentId,
    ranking_scope_id: _rankingScopeId,
    is_test: _isTest,
    generation_id: _generationId,
    updated_sequence: _updatedSequence,
    ...state
  } = value
  return structuredClone(state)
}


function createRatingLedgerRepository(db, { now }) {
  return {
    async settleEligibleMatch({ qualification, match }) {
      if (!qualification || qualification.status !== 'eligible'
          || qualification.match_id !== match.match_id
          || qualification.result_revision !== match.result_revision
          || qualification.ranking_scope_id !== match.ranking_scope_id
          || typeof qualification.is_test !== 'boolean'
          || qualification.is_test !== match.is_test) {
        fail('RATING_SETTLEMENT_NOT_ELIGIBLE')
      }
      const createdAt = now()
      const inputDigest = digest(canonicalInput(qualification, match))
      return db.runTransaction(async (transaction) => {
        const scopeId = match.ranking_scope_id
        const headRef = transaction.collection('rating_scope_heads').doc(digest(scopeId))
        const consumptionRef = transaction.collection('rating_match_consumptions')
          .doc(digest(`${scopeId}:${match.match_id}`))
        const [existingConsumption, currentHead] = await Promise.all([
          optional(consumptionRef), optional(headRef),
        ])
        if (currentHead && currentHead.replay_status === 'building') {
          fail('RATING_SETTLEMENT_REPLAY_IN_PROGRESS')
        }
        if (currentHead && currentHead.is_test !== match.is_test) {
          fail('RANKING_TEST_GENERATION_INVALID')
        }
        if (existingConsumption) {
          if (existingConsumption.is_test !== match.is_test) {
            fail('RANKING_TEST_GENERATION_INVALID')
          }
          if (existingConsumption.status === 'revoked') fail('RATING_SETTLEMENT_REVOKED')
          if (existingConsumption.input_digest !== inputDigest) fail('RATING_SETTLEMENT_CONFLICT')
          const existingEvent = await optional(
            transaction.collection('rating_events').doc(existingConsumption.rating_event_id),
          )
          if (!existingEvent) fail('RATING_SETTLEMENT_CONFLICT')
          return withoutId(existingEvent)
        }

        const generationId = currentHead
          ? currentHead.active_generation_id
          : digest(`${scopeId}:${ALGORITHM_VERSION}:generation-1`)
        const scopeSequence = currentHead && Number.isInteger(currentHead.next_sequence)
          ? currentHead.next_sequence : 1
        const redRef = transaction.collection('player_skill_states')
          .doc(digest(`${scopeId}:${match.red_user_id}`))
        const blueRef = transaction.collection('player_skill_states')
          .doc(digest(`${scopeId}:${match.blue_user_id}`))
        const [storedRed, storedBlue] = await Promise.all([optional(redRef), optional(blueRef)])
        const red = storedRed ? skillFromDocument(storedRed) : initialSkillState(match.red_user_id)
        const blue = storedBlue ? skillFromDocument(storedBlue) : initialSkillState(match.blue_user_id)
        const update = updateMatchRatings({
          red,
          blue,
          winnerRole: match.winner_role,
          format: match.format,
          settledAt: match.settled_at,
        })
        const ledgerMatch = { ...match, scope_sequence: scopeSequence, status: 'active' }
        const event = {
          schema_version: '1.0',
          rating_event_id: ratingEventId(ledgerMatch),
          generation_id: generationId,
          algorithm_version: ALGORITHM_VERSION,
          ranking_scope_id: scopeId,
          is_test: match.is_test,
          scope_sequence: scopeSequence,
          match_id: match.match_id,
          result_revision: match.result_revision,
          qualification_configuration_version: qualification.configuration_version,
          started_at: match.started_at, settled_at: match.settled_at,
          format: match.format,
          winner_role: match.winner_role,
          red_user_id: match.red_user_id,
          blue_user_id: match.blue_user_id,
          before: update.before,
          expected: update.expected,
          outcome: update.outcome,
          after: update.after,
          status: 'active',
          created_at: createdAt,
        }

        if (!currentHead) {
          await transaction.collection('rating_generations').doc(generationId).set({ data: {
            schema_version: '1.0', generation_id: generationId,
            ranking_scope_id: scopeId, algorithm_version: ALGORITHM_VERSION,
            is_test: match.is_test, status: 'active', created_at: createdAt,
          } })
        }
        await transaction.collection('rating_events').doc(event.rating_event_id).set({ data: event })
        await consumptionRef.set({ data: {
          schema_version: '1.0', ranking_scope_id: scopeId, match_id: match.match_id,
          is_test: match.is_test,
          result_revision: match.result_revision, generation_id: generationId,
          rating_event_id: event.rating_event_id, input_digest: inputDigest,
          scope_sequence: scopeSequence, status: 'active', consumed_at: createdAt,
        } })
        await redRef.set({ data: {
          ...update.after.red, ranking_scope_id: scopeId,
          is_test: match.is_test,
          generation_id: generationId, updated_sequence: scopeSequence,
        } })
        await blueRef.set({ data: {
          ...update.after.blue, ranking_scope_id: scopeId,
          is_test: match.is_test,
          generation_id: generationId, updated_sequence: scopeSequence,
        } })
        await headRef.set({ data: {
          schema_version: '1.0', ranking_scope_id: scopeId,
          is_test: match.is_test,
          active_generation_id: generationId, next_sequence: scopeSequence + 1,
          updated_at: createdAt,
        } })
        return structuredClone(event)
      })
    },
  }
}


module.exports = {
  RatingSettlementError,
  createRatingLedgerRepository,
}
