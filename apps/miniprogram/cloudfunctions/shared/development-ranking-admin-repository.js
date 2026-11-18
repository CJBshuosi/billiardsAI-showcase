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


function createDevelopmentRankingAdminRepository(db) {
  return {
    async getReset(resetId) {
      return clean(await optional(
        db.collection('ranking_test_resets').doc(digest(resetId)),
      ))
    },

    async readState(rankingScopeId) {
      const [head, seasons] = await Promise.all([
        optional(db.collection('rating_scope_heads').doc(digest(rankingScopeId))),
        db.collection('ranking_seasons').where({
          ranking_scope_id: rankingScopeId, is_test: true, status: 'active',
        }).orderBy('starts_at', 'desc').limit(2).get(),
      ])
      const active = seasons.data || []
      if (active.length !== 1) return null
      return { head: clean(head), season: clean(active[0]) }
    },

    async commitReset(input) {
      return db.runTransaction(async transaction => {
        const resetRef = transaction.collection('ranking_test_resets')
          .doc(digest(input.resetId))
        const existing = await optional(resetRef)
        if (existing) return structuredClone(existing.result)
        const headRef = transaction.collection('rating_scope_heads')
          .doc(digest(input.nextHead.ranking_scope_id))
        const previousSeasonRef = transaction.collection('ranking_seasons')
          .doc(digest(input.expectedSeasonId))
        const [head, season] = await Promise.all([
          optional(headRef), optional(previousSeasonRef),
        ])
        if (!head || !season || head.is_test !== true || season.is_test !== true
            || head.active_generation_id !== input.expectedGenerationId
            || season.season_id !== input.expectedSeasonId || season.status !== 'active') {
          const error = new Error('RANKING_TEST_GENERATION_INVALID')
          error.code = 'RANKING_TEST_GENERATION_INVALID'
          throw error
        }
        await previousSeasonRef.set({ data: structuredClone(input.previousSeason) })
        await transaction.collection('ranking_seasons')
          .doc(digest(input.nextSeason.season_id)).set({ data: structuredClone(input.nextSeason) })
        await transaction.collection('rating_generations')
          .doc(input.nextGeneration.generation_id).set({ data: structuredClone(input.nextGeneration) })
        await headRef.set({ data: structuredClone(input.nextHead) })
        await resetRef.set({ data: {
          schema_version: '1.0', reset_id: input.resetId,
          ranking_scope_id: input.nextHead.ranking_scope_id, is_test: true,
          result: structuredClone(input.result),
        } })
        return structuredClone(input.result)
      })
    },
  }
}


module.exports = { createDevelopmentRankingAdminRepository }
