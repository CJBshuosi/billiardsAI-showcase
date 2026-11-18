const crypto = require('crypto')
const cloud = require('wx-server-sdk')
const { createShotProjectionWorkerHandler } = require('./lib/handler')
const { createCloudShotProjectionRepository } = require('../shared/shot-projection')

function configuredToken(value) {
  return typeof value === 'string' && /^[\x21-\x7e]{1,4096}$/u.test(value)
    && !/\{\{|\}\}|\$\{|^<.*>$/u.test(value)
}

function verifyToken(candidate) {
  const expected = process.env.SHOT_PROJECTION_WORKER_TOKEN
  if (!configuredToken(expected) || !configuredToken(candidate)) return false
  // Fixed-size digests allow timingSafeEqual even when input lengths differ.
  const digest = value => crypto.createHash('sha256').update(value, 'utf8').digest()
  return crypto.timingSafeEqual(digest(candidate), digest(expected))
}

let repository
function getRepository() {
  if (!repository) {
    cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })
    repository = createCloudShotProjectionRepository(cloud.database())
  }
  return repository
}

exports.main = createShotProjectionWorkerHandler({
  verifyToken,
  repo: {
    findPending: () => getRepository().findPending(),
    projectBatch: input => getRepository().projectBatch(input),
  },
})
