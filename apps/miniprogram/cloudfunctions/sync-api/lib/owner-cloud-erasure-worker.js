const { canonicalJson } = require('../../shared/shot-ownership')
// Internal composition only. Native requests never run in document callbacks.
function createOwnerCloudErasureWorker({ repository, provider }) {
  return { async runOnce({ jobId } = {}) {
    const claim = await repository.claim({ jobId, leaseSeconds: 30 })
    if (!claim) return { status: 'pending', receipt: null }
    if (claim.contract === 'owner_cloud_erasure_replay_v1') {
      const receipt = await repository.finalize(claim)
      return { status: receipt ? 'complete' : 'pending', receipt }
    }
    const step = await repository.loadStep(claim)
    if (step) {
      let result
      try { result = await provider.step({ ...step, beforeRequest: async () => {
        const current = await repository.loadStep(claim)
        return current !== null && canonicalJson(current) === canonicalJson(step)
      } }) } catch { return { status: 'pending', receipt: null } }
      if (!await repository.recordStep(claim, result)) return { status: 'pending', receipt: null }
    }
    const receipt = await repository.finalize(claim)
    return { status: receipt ? 'complete' : 'pending', receipt }
  } }
}
module.exports = { createOwnerCloudErasureWorker }
