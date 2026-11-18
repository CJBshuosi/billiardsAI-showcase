const crypto = require('crypto')
const cloud = require('wx-server-sdk')

const { ClientApiError, stableError } = require('./lib/errors')
const {
  createCloudContentRepository,
  getClipPlayback,
  getSession,
  getSessionAnalysis,
  legacyFavoriteResult,
  listClips,
  listClipsPage,
  listSessions,
} = require('./lib/content')
const { resolveUserId } = require('./lib/identity')
const { deleteOwnerClip } = require('./lib/clip-deletion')
const {
  createCloudFieldCommandRepository,
  getCurrentField,
  getFieldCommandStatus,
  submitFieldCommand,
} = require('./lib/field-commands')
const { createCloudJoinRepository, getJoinStatus, requestJoin } = require('./lib/join')
const { listUserChallenges } = require('./lib/challenges')
const { createRankingMatchApi } = require('./lib/ranking-matches')
const {
  createCloudRankingMatchOutcomeRepository,
} = require('./lib/ranking-match-outcome')
const {
  createCloudMatchReportRepository,
  submitMatchReport,
  submitReviewAppeal,
} = require('./lib/ranking-reports')
const { processConfirmedMatch } = require('./lib/ranking-worker-client')
const {
  createCloudRankingRepository,
  getCityRanking,
  getPublicRankingPlayer,
  loadRuntimeRankingScope,
} = require('./lib/ranking')
const { createCloudRepositories } = require('./lib/users')
const { ChallengeError, createChallengeService } = require('../shared/challenge-service')
const { createCloudChallengeRepository } = require('../shared/challenge-repository')
const {
  PublicProfileError,
  createPublicRankingProfileRepository,
} = require('../shared/public-ranking-profile')
const { createMatchFactsRepository } = require('../shared/match-facts-repository')
const { MatchSetupError, createMatchSetupService } = require('../shared/match-setup-service')
const { createMatchReviewRepository } = require('../shared/match-review-repository')
const { createRankingSettlementQueue } = require('../shared/ranking-settlement-queue')
const { resolveRuntimeRankingScope } = require('../shared/ranking-scope-policy')
const {
  ResultConfirmationError,
  createResultConfirmationService,
} = require('../shared/result-confirmation-service')

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })
const db = cloud.database()


function newUserId() {
  return `user-${crypto.randomUUID()}`
}


exports.main = async (event) => {
  try {
    if (!event || event.schema_version !== '1.0' || typeof event.action !== 'string') {
      throw new ClientApiError('REQUEST_INVALID')
    }
    const context = cloud.getWXContext()
    const repositories = createCloudRepositories(db)
    const contentRepo = createCloudContentRepository(db, cloud)
    const joinRepo = createCloudJoinRepository(db)
    const commandRepo = createCloudFieldCommandRepository(db)
    const rankingRepo = createCloudRankingRepository(db)
    const runtimeEnvironment = process.env.RANKING_RUNTIME_ENVIRONMENT
    const runtimeScope = resolveRuntimeRankingScope({ environment: runtimeEnvironment })
    let loadedRankingScope
    const getRankingScope = async () => {
      if (!loadedRankingScope) {
        loadedRankingScope = await loadRuntimeRankingScope({
          environment: runtimeEnvironment, repo: rankingRepo,
        })
      }
      return loadedRankingScope
    }
    const publicProfileRepo = createPublicRankingProfileRepository(db, {
      now: () => new Date().toISOString(),
    })
    const matchRepository = createMatchFactsRepository(db, {
      now: () => new Date().toISOString(),
    })
    const matchSetupService = createMatchSetupService({
      repository: matchRepository, now: () => new Date().toISOString(),
    })
    const resultService = createResultConfirmationService({
      repository: matchRepository, now: () => new Date().toISOString(),
    })
    const matchReviewRepository = createMatchReviewRepository(db, {
      now: () => new Date().toISOString(),
    })
    const matchReportRepository = createCloudMatchReportRepository(db, {
      now: () => new Date().toISOString(),
    })
    const settlementQueue = createRankingSettlementQueue(db, {
      now: () => new Date().toISOString(),
    })
    const rankingMatchApi = createRankingMatchApi({
      matchRepository, fieldRepository: commandRepo, rankingRepository: rankingRepo,
      outcomeRepository: createCloudRankingMatchOutcomeRepository(db),
      setupService: matchSetupService, resultService,
    })
    const challengeService = createChallengeService({
      repository: createCloudChallengeRepository(db),
      now: () => new Date().toISOString(),
      readProfile: targetUserId => rankingRepo.getPublicProfile(targetUserId),
      readVenue: venueId => rankingRepo.getVenue(venueId),
    })
    const user = await resolveUserId({
      openid: context.OPENID,
      identityRepo: repositories.identityRepo,
      idFactory: newUserId,
      now: () => new Date().toISOString(),
    })
    if (event.action === 'auth.me') {
      return { ok: true, data: user }
    }
    if (event.action === 'profile.update') {
      const updated = await repositories.usersRepo.updateProfile(
        user.user_id,
        event.payload || {},
        new Date().toISOString(),
      )
      return { ok: true, data: updated }
    }
    if (event.action === 'join.request') {
      const joined = await requestJoin({
        tableCode: event.payload && event.payload.table_code,
        userId: user.user_id,
        now: new Date().toISOString(),
        repo: joinRepo,
      })
      return { ok: true, data: joined }
    }
    if (event.action === 'join.status') {
      const status = await getJoinStatus({
        requestId: event.payload && event.payload.request_id,
        userId: user.user_id,
        repo: joinRepo,
      })
      return { ok: true, data: status }
    }
    if (event.action === 'field.command.submit') {
      return {
        ok: true,
        data: await submitFieldCommand({
          actorUserId: user.user_id,
          sessionId: event.payload && event.payload.session_id,
          commandType: event.payload && event.payload.command_type,
          payload: event.payload && event.payload.payload,
          expectedRevision: event.payload && event.payload.expected_revision,
          idempotencyKey: event.payload && event.payload.idempotency_key,
          now: new Date().toISOString(),
        }, commandRepo),
      }
    }
    if (event.action === 'field.current') {
      return {
        ok: true,
        data: await getCurrentField({
          sessionId: event.payload && event.payload.session_id,
          actorUserId: user.user_id,
          repo: commandRepo,
        }),
      }
    }
    if (event.action === 'field.command.status') {
      return {
        ok: true,
        data: await getFieldCommandStatus({
          commandId: event.payload && event.payload.command_id,
          actorUserId: user.user_id,
          repo: commandRepo,
        }),
      }
    }
    if (event.action === 'sessions.list') {
      return { ok: true, data: await listSessions({ userId: user.user_id, repo: contentRepo }) }
    }
    if (event.action === 'sessions.get') {
      return {
        ok: true,
        data: await getSession({
          sessionId: event.payload && event.payload.session_id,
          userId: user.user_id,
          repo: contentRepo,
        }),
      }
    }
    if (event.action === 'sessions.analysis.get') {
      return {
        ok: true,
        data: await getSessionAnalysis({
          sessionId: event.payload && event.payload.session_id,
          userId: user.user_id,
          repo: contentRepo,
        }),
      }
    }
    if (event.action === 'clips.list') {
      return {
        ok: true,
        data: await listClips({
          sessionId: event.payload ? event.payload.session_id : undefined,
          userId: user.user_id,
          repo: contentRepo,
          now: new Date().toISOString(),
        }),
      }
    }
    if (event.action === 'clips.listPage') {
      return {
        ok: true,
        data: await listClipsPage({
          sessionId: event.payload ? event.payload.session_id : undefined,
          cursor: event.payload ? event.payload.cursor : undefined,
          userId: user.user_id,
          repo: contentRepo,
        }),
      }
    }
    if (event.action === 'clips.getPlayback') {
      return {
        ok: true,
        data: await getClipPlayback({
          clipId: event.payload && event.payload.clip_id,
          userId: user.user_id,
          repo: contentRepo,
        }),
      }
    }
    if (event.action === 'clips.delete') {
      return { ok: true, data: await deleteOwnerClip({
        payload: event.payload, userId: user.user_id, db, now: new Date().toISOString(),
      }) }
    }
    if (['favorites.list', 'favorites.set'].includes(event.action)) {
      return legacyFavoriteResult(event.action)
    }
    if (event.action === 'ranking.list') {
      const rankingScope = await getRankingScope()
      return {
        ok: true,
        data: await getCityRanking({
          userId: user.user_id,
          rankingScopeId: runtimeScope.ranking_scope_id,
          rankingScope,
          kind: event.payload && event.payload.kind || 'long_term',
          page: event.payload && event.payload.page != null ? event.payload.page : 1,
          pageSize: event.payload && event.payload.page_size != null ? event.payload.page_size : 50,
          repo: rankingRepo,
        }),
      }
    }
    if (event.action === 'ranking.player.get') {
      const rankingScope = await getRankingScope()
      return {
        ok: true,
        data: await getPublicRankingPlayer({
          targetUserId: event.payload && event.payload.user_id,
          rankingScopeId: runtimeScope.ranking_scope_id,
          rankingScope,
          snapshotId: event.payload && event.payload.snapshot_id,
          repo: rankingRepo,
        }),
      }
    }
    if (event.action === 'ranking.venues.list') {
      await getRankingScope()
      return {
        ok: true,
        data: await rankingRepo.listApprovedVenues(runtimeScope.ranking_scope_id),
      }
    }
    if (event.action === 'ranking.profile.save') {
      return {
        ok: true,
        data: await publicProfileRepo.saveProfile({
          userId: user.user_id,
          nickname: event.payload && event.payload.nickname,
          isPublic: event.payload && event.payload.is_public,
          acceptChallenges: event.payload && event.payload.accept_challenges,
        }),
      }
    }
    if (event.action === 'ranking.profile.get') {
      return { ok: true, data: await rankingRepo.getPublicProfile(user.user_id) }
    }
    if (event.action === 'ranking.match.get') {
      return { ok: true, data: await rankingMatchApi.get({
        sessionId: event.payload && event.payload.session_id, actorUserId: user.user_id,
      }) }
    }
    if (event.action === 'ranking.match.settings.propose') {
      return { ok: true, data: await rankingMatchApi.proposeSettings({
        sessionId: event.payload && event.payload.session_id,
        actorUserId: user.user_id,
        mode: event.payload && event.payload.mode,
        format: event.payload && event.payload.format,
        raceTarget: event.payload && event.payload.race_target,
        firstBreaker: event.payload && event.payload.first_breaker,
        rankingIntent: event.payload && event.payload.ranking_intent,
      }) }
    }
    if (event.action === 'ranking.match.settings.confirm') {
      return { ok: true, data: await rankingMatchApi.confirmSettings({
        sessionId: event.payload && event.payload.session_id, actorUserId: user.user_id,
      }) }
    }
    if (event.action === 'ranking.results.pending') {
      return { ok: true, data: await rankingMatchApi.listPendingResults({ actorUserId: user.user_id }) }
    }
    if (event.action === 'ranking.result.confirm') {
      const match = await rankingMatchApi.confirmResult({
        matchId: event.payload && event.payload.match_id, actorUserId: user.user_id,
      })
      const processing = await processConfirmedMatch({
        match, token: process.env.RANKING_WORKER_TOKEN,
        enqueueMatch: matchId => settlementQueue.enqueue({
          matchId, rankingScopeId: runtimeScope.ranking_scope_id,
        }),
        callWorker: input => cloud.callFunction(input),
      })
      return { ok: true, data: { ...match, ...processing } }
    }
    if (event.action === 'ranking.result.dispute') {
      return { ok: true, data: await rankingMatchApi.disputeResult({
        matchId: event.payload && event.payload.match_id,
        actorUserId: user.user_id, reason: event.payload && event.payload.reason,
      }) }
    }
    if (event.action === 'ranking.match.report') {
      return { ok: true, data: await submitMatchReport({
        matchId: event.payload && event.payload.match_id,
        actorUserId: user.user_id,
        reasonCode: event.payload && event.payload.reason_code,
        matchRepository, reportRepository: matchReportRepository,
      }) }
    }
    if (event.action === 'ranking.review.appeal') {
      return { ok: true, data: await submitReviewAppeal({
        matchId: event.payload && event.payload.match_id,
        actorUserId: user.user_id,
        reasonCode: event.payload && event.payload.reason_code,
        matchRepository, reviewRepository: matchReviewRepository,
      }) }
    }
    if (event.action === 'challenge.create') {
      return {
        ok: true,
        data: await challengeService.create({
          idempotencyKey: event.payload && event.payload.idempotency_key,
          challengerUserId: user.user_id,
          targetUserId: event.payload && event.payload.target_user_id,
          venueId: event.payload && event.payload.venue_id,
          proposedAt: event.payload && event.payload.proposed_at,
          format: event.payload && event.payload.format,
          raceTarget: event.payload && event.payload.race_target,
        }),
      }
    }
    if (event.action === 'challenge.list') {
      return {
        ok: true,
        data: await listUserChallenges({
          userId: user.user_id,
          repository: createCloudChallengeRepository(db),
          readProfile: targetUserId => rankingRepo.getPublicProfile(targetUserId),
          readVenue: venueId => rankingRepo.getVenue(venueId),
        }),
      }
    }
    if (['challenge.accept', 'challenge.reject', 'challenge.cancel'].includes(event.action)) {
      const method = event.action.split('.')[1]
      return {
        ok: true,
        data: await challengeService[method]({
          challengeId: event.payload && event.payload.challenge_id,
          actorUserId: user.user_id,
        }),
      }
    }
    throw new ClientApiError('REQUEST_INVALID')
  } catch (error) {
    if (error instanceof ChallengeError || error instanceof PublicProfileError
        || error instanceof MatchSetupError || error instanceof ResultConfirmationError) {
      return stableError(new ClientApiError(error.code))
    }
    return stableError(error)
  }
}
