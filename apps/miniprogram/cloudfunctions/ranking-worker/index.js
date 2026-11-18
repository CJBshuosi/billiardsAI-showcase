const crypto = require('crypto')
const cloud = require('wx-server-sdk')

const { createRankingWorkerHandler } = require('./lib/handler')
const { createMatchFactsRepository } = require('../shared/match-facts-repository')
const { createCloudMatchRiskDetector } = require('../shared/match-risk-detector')
const { createMatchReviewRepository } = require('../shared/match-review-repository')
const { createRankingSettlementPipeline } = require('../shared/ranking-settlement-pipeline')
const {
  createRankingAuthorityRepository,
  createRankingDecisionRepositories,
} = require('../shared/ranking-pipeline-repositories')
const { createRatingLedgerRepository } = require('../shared/rating-ledger-repository')
const {
  createRatingReplayRepository,
  createRatingReviewRepository,
} = require('../shared/rating-replay-repository')
const { createRatingReplayWorker } = require('../shared/rating-replay-worker')
const { createReviewResolutionService } = require('../shared/review-resolution-service')
const {
  createCloudRankingExpiryRepository,
  createRankingExpiryService,
} = require('../shared/ranking-expiry-service')
const { createResultConfirmationService } = require('../shared/result-confirmation-service')
const { createChallengeService } = require('../shared/challenge-service')
const { createCloudChallengeRepository } = require('../shared/challenge-repository')
const { createRankingSnapshotPublisher } = require('../shared/ranking-snapshot-publisher')
const { createRankingSnapshotRepository } = require('../shared/ranking-snapshot-repository')
const { createRankingSettlementQueue } = require('../shared/ranking-settlement-queue')
const {
  createDevelopmentRankingResetService,
} = require('../shared/development-ranking-reset-service')
const {
  createDevelopmentRankingAdminRepository,
} = require('../shared/development-ranking-admin-repository')
const {
  assertRankingScopePolicy,
  resolveRuntimeRankingScope,
} = require('../shared/ranking-scope-policy')


cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })
const db = cloud.database()
const now = () => new Date().toISOString()
const runtimeEnvironment = process.env.RANKING_RUNTIME_ENVIRONMENT
const runtimeScope = resolveRuntimeRankingScope({ environment: runtimeEnvironment })
const matchRepository = createMatchFactsRepository(db, { now })
const decisions = createRankingDecisionRepositories(db, {
  now, detectRiskSignals: createCloudMatchRiskDetector(db),
})
const snapshotRepository = createRankingSnapshotRepository(db, { now })
const reviewRepository = createMatchReviewRepository(db, { now })
const settlementQueue = createRankingSettlementQueue(db, { now })
const pipeline = createRankingSettlementPipeline({
  matchFactsRepository: matchRepository,
  authorityRepository: createRankingAuthorityRepository(db),
  qualificationRepository: decisions.qualificationRepository,
  riskRepository: decisions.riskRepository,
  reviewRepository,
  ratingRepository: createRatingLedgerRepository(db, { now }),
})


function safeToken(candidate) {
  const expected = process.env.RANKING_WORKER_TOKEN || ''
  if (typeof candidate !== 'string' || !expected) return false
  const left = Buffer.from(candidate)
  const right = Buffer.from(expected)
  return left.length === right.length && crypto.timingSafeEqual(left, right)
}


async function readDocument(collection, id) {
  try {
    const result = await db.collection(collection).doc(id).get()
    return result.data || null
  } catch (error) {
    if (String(error && error.message).includes('does not exist')) return null
    throw error
  }
}


const handler = createRankingWorkerHandler({
  rankingScopeId: runtimeScope.ranking_scope_id,
  developmentResetService: createDevelopmentRankingResetService({
    repository: createDevelopmentRankingAdminRepository(db), now,
  }),
  verifyToken: safeToken,
  settlementQueue,
  pipeline,
  snapshotPublisher: createRankingSnapshotPublisher({ repository: snapshotRepository }),
  replayWorker: createRatingReplayWorker({
    repository: createRatingReplayRepository(db, { now }),
  }),
  reviewResolutionService: createReviewResolutionService({
    reviewRepository, matchRepository,
    ratingEventRepository: createRatingReviewRepository(db), pipeline,
  }),
  expiryService: createRankingExpiryService({
    repository: createCloudRankingExpiryRepository(db, { now }),
    resultConfirmationService: createResultConfirmationService({ repository: matchRepository, now }),
    challengeService: createChallengeService({
      repository: createCloudChallengeRepository(db), now,
      readProfile: async () => null, readVenue: async () => null,
    }),
  }),
  readScope: async () => {
    const id = crypto.createHash('sha256').update(runtimeScope.ranking_scope_id, 'utf8').digest('hex')
    const scope = await readDocument('ranking_scopes', id)
    if (!scope) throw Object.assign(new Error('scope missing'), { code: 'WORKER_SCOPE_NOT_CONFIGURED' })
    assertRankingScopePolicy({ environment: runtimeEnvironment, scope })
    return scope
  },
  readSeason: async () => {
    const result = await db.collection('ranking_seasons')
      .where({ ranking_scope_id: runtimeScope.ranking_scope_id, status: 'active' })
      .orderBy('starts_at', 'desc').limit(1).get()
    const season = result.data && result.data[0]
    if (!season) throw Object.assign(new Error('season missing'), { code: 'WORKER_SEASON_NOT_CONFIGURED' })
    return season
  },
  now,
  reportError: diagnostic => console.error('ranking-worker failure', diagnostic),
})


exports.main = handler
exports.createRankingWorkerHandler = createRankingWorkerHandler
