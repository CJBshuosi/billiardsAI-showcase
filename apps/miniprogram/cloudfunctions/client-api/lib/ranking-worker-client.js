async function processConfirmedMatch({ match, token, enqueueMatch, callWorker }) {
  if (!match || match.status !== 'fact_package_ready') return { settlement_status: 'not_ready' }
  await enqueueMatch(match.match_id)
  try {
    const response = await callWorker({
      name: 'ranking-worker',
      data: {
        schema_version: '1.0', token,
        action: 'process.match', match_id: match.match_id,
      },
    })
    const result = response && response.result
    const status = result && result.ok && result.data && result.data.settlement
      && result.data.settlement.status
    return { settlement_status: status || 'pending_processing' }
  } catch (_error) {
    return { settlement_status: 'pending_processing' }
  }
}


module.exports = { processConfirmedMatch }
