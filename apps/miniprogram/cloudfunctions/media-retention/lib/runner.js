const {
  executeClaimedCloudDeletion,
  prepareDeletionJob,
} = require('./deletion')


async function runSyntheticRetentionBatch({ repo, storage, env, workerId, now, limit = 25 }) {
  const summary = { enabled: false, scanned: 0, awaiting_edge: 0, held: 0, failed: 0 }
  if (!env || env.MEDIA_RETENTION_EXECUTION_ENABLED !== 'synthetic-only') return summary
  summary.enabled = true
  const authorities = await repo.listCandidateAuthorities({ now, limit })
  summary.scanned = authorities.length
  for (const authority of authorities) {
    let pending
    try {
      pending = prepareDeletionJob({ ...authority, now })
      if (!pending) {
        summary.held += 1
        continue
      }
      pending = await repo.createPending(pending, authority)
      const claimed = await repo.claim(pending, { workerId, now, leaseSeconds: 300 })
      const cloudDone = await executeClaimedCloudDeletion({
        job: claimed, clip: authority.clip, env, storage, now,
      })
      await repo.saveCloudDeletion(cloudDone)
      summary.awaiting_edge += 1
    } catch (error) {
      summary.failed += 1
      if (repo.markFailed) await repo.markFailed(pending, error)
    }
  }
  return summary
}


module.exports = { runSyntheticRetentionBatch }
