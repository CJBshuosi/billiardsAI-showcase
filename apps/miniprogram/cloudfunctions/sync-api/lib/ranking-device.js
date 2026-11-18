const { SyncApiError } = require('./auth')


const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u


async function completeRankingStart({ command, matchSetupService }) {
  if (!command || command.status !== 'applied' || command.command_type !== 'set_breaker') return null
  return matchSetupService.startFromDevice({ sessionId: command.session_id })
}


async function proposeDeviceResult({
  tableId, sessionId, winnerRole, redScore, blueScore, finishReason,
  matchRepository, matchSetupService,
}) {
  if (!ID_PATTERN.test(tableId || '') || !ID_PATTERN.test(sessionId || '')
      || !['red', 'blue'].includes(winnerRole)
      || !Number.isInteger(redScore) || !Number.isInteger(blueScore)
      || !['normal', 'concession'].includes(finishReason)) {
    throw new SyncApiError('SYNC_INVALID')
  }
  const match = await matchRepository.getBySession(sessionId)
  if (!match || !match.projection || match.projection.table_id !== tableId) {
    throw new SyncApiError('SYNC_CONFLICT')
  }
  const projection = match.projection
  if (projection.status !== 'in_progress') {
    const result = projection.result
    if (result
        && result.winner_role === winnerRole
        && result.red_score === redScore
        && result.blue_score === blueScore
        && result.finish_reason === finishReason) {
      return structuredClone(projection)
    }
    throw new SyncApiError('SYNC_CONFLICT')
  }
  return matchSetupService.proposeResultFromDevice({
    sessionId, winnerRole, redScore, blueScore, finishReason,
  })
}


module.exports = { completeRankingStart, proposeDeviceResult }
