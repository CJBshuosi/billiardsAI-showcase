const crypto = require('crypto')
const { enqueueReadyOwnerDeviceDeletion } = require('../shared/owner-device-deletion-dispatch')
const { createCloudOwnerDeviceDeliveryRepository, exact, LIMIT: OWNER_DEVICE_LIMIT } = require('./lib/owner-device-delivery')
const cloudbase = require('@cloudbase/node-sdk')
const cloud = require('wx-server-sdk')

const {
  SyncApiError,
  authenticate,
  createCloudAuthRepositories,
  getHeader,
} = require('./lib/auth')
const {
  authorizeMedia,
  authorizeSavedClip,
  completeSavedClipJob,
  completeSyncJob,
  createCloudSavedClipRepository,
  createCloudSyncRepository,
  createSavedClipJob,
  createSyncJob,
} = require('./lib/ingest')
const { createCloudBaseStorageProvider, createStoragePort } = require('./lib/storage')
const { createJoinAdmin } = require('./lib/join-admin')
const { validManagedUploadAttempt } = require('../shared/managed-upload-attempt')
const { createOwnerCloudErasureRepository, evaluateSynchronousNativeAdmission } = require('../shared/owner-cloud-erasure')
const { createCloudOwnerDeviceTaskRepository } = require('../shared/owner-device-deletion')
const { createOwnerCloudErasureWorker } = require('./lib/owner-cloud-erasure-worker')
const { createOwnerErasureProvider } = require('./lib/owner-erasure-provider')
const { validManagedUploadRequest } = require('../shared/managed-upload-request')
const { validUploadReceipt } = require('../shared/upload-terminal-receipts')
const {
  claimNextFieldCommand,
  completeFieldCommand,
  createCloudDeviceFieldCommandRepository,
} = require('./lib/field-commands')
const { completeRankingStart, proposeDeviceResult } = require('./lib/ranking-device')
const { createMatchFactsRepository } = require('../shared/match-facts-repository')
const { createMatchSetupService } = require('../shared/match-setup-service')
const {
  claimNextEdgeDeletion,
  completeEdgeDeletion,
  createCloudMediaDeletionRepository,
} = require('./lib/media-retention')
const {
  createCloudSessionAnalysisRepository,
  putLiveShotFact,
  putSessionAnalysis,
} = require('./lib/session-analysis')


function response(statusCode, payload) {
  return {
    statusCode,
    headers: { 'content-type': 'application/json; charset=utf-8' },
    body: JSON.stringify(payload),
  }
}


function errorResponse(error, logger = console) {
  if (!(error instanceof SyncApiError)) {
    logger.error('sync-api unexpected error', {
      name: error && error.name,
      code: error && error.code,
      errCode: error && error.errCode,
      errMsg: error && error.errMsg,
      message: error && error.message,
    })
  }
  const code = error instanceof SyncApiError ? error.code : 'INTERNAL_ERROR'
  const statusCode = code === 'SYNC_UNAUTHORIZED' || code === 'SYNC_REPLAYED'
    ? 401
    : code === 'SYNC_CONFLICT'
      ? 409
      : code === 'SYNC_STORAGE_UNAVAILABLE' || code === 'SYNC_PARTIAL'
        ? 503
        : 400
  return response(statusCode, { ok: false, error: { code } })
}


function parseJson(body) {
  try {
    const value = JSON.parse(body)
    if (!value || Array.isArray(value) || typeof value !== 'object') {
      throw new Error('invalid body')
    }
    return value
  } catch (_error) {
    throw new SyncApiError('SYNC_INVALID')
  }
}


function runtimeAttestationFromEnvironment(environment) {
  return Object.freeze({
    schema_version: '1.0',
    runtime_environment: environment.RANKING_RUNTIME_ENVIRONMENT || 'unknown',
    cloudbase_environment_id: environment.CLOUDBASE_ENV_ID
      || environment.TCB_ENV || environment.SCF_NAMESPACE || 'unknown',
  })
}


function createHandler(dependencies) {
  return async (event) => {
    let ownerDeviceRoute = false
    try {
      const method = String(event.httpMethod || event.method || '').toUpperCase()
      const path = event.path || (event.requestContext && event.requestContext.path) || ''
      ownerDeviceRoute = path === '/v1/owner-media-deletions/claim' || /^\/v1\/owner-media-deletions\/[A-Za-z0-9._-]+\/complete$/u.test(path)
      const rawBody = event.isBase64Encoded
        ? Buffer.from(event.body || '', 'base64')
        : Buffer.from(String(event.body || ''), 'utf8')
      if (ownerDeviceRoute && rawBody.length > OWNER_DEVICE_LIMIT) throw new SyncApiError('SYNC_INVALID')
      const body = event.isBase64Encoded ? rawBody.toString('utf8') : String(event.body || '')
      if (ownerDeviceRoute && !Buffer.from(body, 'utf8').equals(rawBody)) throw new SyncApiError('SYNC_INVALID')
      const request = { method, path, body, headers: event.headers || {} }
      const scope = await authenticate(request, dependencies.auth)
      const payload = parseJson(body)

      if (method === 'POST' && path === '/v1/runtime-attestation') {
        return response(200, { ok: true, data: dependencies.runtimeAttestation })
      }

      if (method === 'POST' && path === '/v2/field-sessions/open') {
        if (Object.keys(payload).length !== 2 || payload.schema_version !== '1.0'
            || typeof payload.open_request_id !== 'string'
            || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu.test(payload.open_request_id)) {
          throw new SyncApiError('SYNC_INVALID')
        }
        const data = await dependencies.joinAdmin.createManagedSession({
          openRequestId: payload.open_request_id, tableId: scope.table_id, deviceId: scope.device_id,
        })
        return response(200, { ok: true, data })
      }

      const managedCloseMatch = path.match(/^\/v2\/field-sessions\/(managed-[A-Za-z0-9._-]+)\/close$/u)
      if (method === 'POST' && managedCloseMatch) {
        if (Object.keys(payload).length !== 1 || payload.schema_version !== '1.0') {
          throw new SyncApiError('SYNC_INVALID')
        }
        const data = await dependencies.joinAdmin.closeManagedSession({
          sessionId: managedCloseMatch[1], tableId: scope.table_id, deviceId: scope.device_id,
        })
        return response(200, { ok: true, data })
      }

      const analysisMatch = path.match(
        /^\/v1\/field-sessions\/([A-Za-z0-9._-]+)\/analysis$/u,
      )
      if (method === 'PUT' && analysisMatch) {
        const data = await putSessionAnalysis({
          sessionId: analysisMatch[1], projection: payload,
          tableId: scope.table_id, deviceId: scope.device_id,
          idempotencyKey: getHeader(event.headers, 'x-idempotency-key'),
          repo: dependencies.analysisRepo,
        })
        return response(200, { ok: true, data })
      }

      const liveShotMatch = path.match(
        /^\/v1\/field-sessions\/([A-Za-z0-9._-]+)\/shots\/([A-Za-z0-9._-]+)$/u,
      )
      if (method === 'PUT' && liveShotMatch) {
        const data = await putLiveShotFact({
          sessionId: liveShotMatch[1], shotId: liveShotMatch[2], fact: payload,
          tableId: scope.table_id, deviceId: scope.device_id,
          idempotencyKey: getHeader(event.headers, 'x-idempotency-key'),
          repo: dependencies.analysisRepo, now: dependencies.nowIso(),
        })
        return response(200, { ok: true, data })
      }

      if (method === 'POST' && ownerDeviceRoute) {
        let data
        if (path === '/v1/owner-media-deletions/claim') {
          if (!exact(payload, ['contract']) || payload.contract !== 'owner_delete_device_poll_v1') throw new SyncApiError('SYNC_INVALID')
          data = await dependencies.ownerDeviceDeliveryRepo.poll({ tableId: scope.table_id, deviceId: scope.device_id })
        } else {
          const jobId = path.split('/')[3]
          data = await dependencies.ownerDeviceDeliveryRepo.complete({ jobId, tableId: scope.table_id, deviceId: scope.device_id, body: payload })
        }
        const result = response(200, { ok: true, data })
        if (Buffer.byteLength(result.body, 'utf8') > OWNER_DEVICE_LIMIT) throw new SyncApiError('SYNC_CONFLICT')
        return result
      }

      if (method === 'POST' && path === '/v1/media-deletions/claim') {
        const data = await claimNextEdgeDeletion({
          tableId: scope.table_id, deviceId: scope.device_id,
          now: dependencies.nowIso(), repo: dependencies.mediaDeletionRepo,
        })
        return response(200, { ok: true, data })
      }

      const deletionCompleteMatch = path.match(
        /^\/v1\/media-deletions\/([A-Za-z0-9._-]+)\/complete$/u,
      )
      if (method === 'POST' && deletionCompleteMatch) {
        const data = await completeEdgeDeletion({
          jobId: deletionCompleteMatch[1], tableId: scope.table_id,
          deviceId: scope.device_id, claimToken: payload.claim_token,
          clipId: payload.clip_id, outcome: payload.outcome,
          now: dependencies.nowIso(), repo: dependencies.mediaDeletionRepo,
        })
        return response(200, { ok: true, data })
      }

      if (method === 'POST' && path === '/v1/field-commands/claim') {
        const data = await claimNextFieldCommand({
          tableId: scope.table_id, deviceId: scope.device_id,
          now: dependencies.nowIso(), repo: dependencies.fieldCommandRepo,
        })
        return response(200, { ok: true, data })
      }

      const commandCompleteMatch = path.match(
        /^\/v1\/field-commands\/([A-Za-z0-9._-]+)\/complete$/u,
      )
      if (method === 'POST' && commandCompleteMatch) {
        const data = await completeFieldCommand({
          commandId: commandCompleteMatch[1], tableId: scope.table_id,
          deviceId: scope.device_id, status: payload.status,
          publicResultCode: payload.public_result_code,
          resultingRevision: payload.resulting_revision,
          now: dependencies.nowIso(), repo: dependencies.fieldCommandRepo,
        })
        await completeRankingStart({ command: data, matchSetupService: dependencies.matchSetupService })
        return response(200, { ok: true, data })
      }

      const rankingResultMatch = path.match(
        /^\/v1\/field-sessions\/([A-Za-z0-9._-]+)\/ranking-result$/u,
      )
      if (method === 'POST' && rankingResultMatch) {
        const data = await proposeDeviceResult({
          tableId: scope.table_id, sessionId: rankingResultMatch[1],
          winnerRole: payload.winner_role, redScore: payload.red_score,
          blueScore: payload.blue_score, finishReason: payload.finish_reason,
          matchRepository: dependencies.matchRepository,
          matchSetupService: dependencies.matchSetupService,
        })
        return response(200, { ok: true, data })
      }

      const sessionLifecycleMatch = path.match(
        /^\/v1\/field-sessions\/([A-Za-z0-9._-]+)\/(open|close)$/u,
      )
      if (method === 'POST' && sessionLifecycleMatch) {
        const operation = sessionLifecycleMatch[2] === 'open' ? 'openSession' : 'closeSession'
        const data = await dependencies.joinAdmin[operation]({
          sessionId: sessionLifecycleMatch[1], tableId: scope.table_id,
          operatorId: scope.device_id,
        })
        return response(200, { ok: true, data })
      }

      const joinQueryMatch = path.match(
        /^\/v1\/field-sessions\/([A-Za-z0-9._-]+)\/join-requests\/query$/u,
      )
      if (method === 'POST' && joinQueryMatch) {
        const data = await dependencies.joinAdmin.listRequests({
          sessionId: joinQueryMatch[1], tableId: scope.table_id,
        })
        return response(200, { ok: true, data })
      }

      const joinConfirmMatch = path.match(
        /^\/v1\/field-sessions\/([A-Za-z0-9._-]+)\/join-requests\/([A-Za-z0-9._-]+)\/confirm$/u,
      )
      if (method === 'POST' && joinConfirmMatch) {
        const data = await dependencies.joinAdmin.confirm({
          sessionId: joinConfirmMatch[1],
          requestId: joinConfirmMatch[2],
          role: payload.role,
          tableId: scope.table_id,
          operatorId: scope.device_id,
        })
        return response(200, { ok: true, data })
      }

      if (method === 'POST' && path === '/v1/sync/saved-clips') {
        const data = await createSavedClipJob({
          descriptor: payload,
          idempotencyKey: getHeader(event.headers, 'x-idempotency-key'),
          tableId: scope.table_id,
          deviceId: scope.device_id,
          repo: dependencies.savedClipRepo,
          idFactory: dependencies.idFactory,
          now: dependencies.nowIso(),
        })
        return response(200, { ok: true, data })
      }

      const savedAuthorizationMatch = path.match(
        /^\/v1\/sync\/saved-clips\/([A-Za-z0-9._-]+)\/media-authorization$/u,
      )
      if (method === 'POST' && savedAuthorizationMatch) {
        if (Object.keys(payload).some(key => key !== 'upload_request')
            || (Object.hasOwn(payload, 'upload_request') && !validManagedUploadRequest(payload.upload_request))) {
          throw new SyncApiError('SYNC_INVALID')
        }
        const data = await authorizeSavedClip({
          jobId: savedAuthorizationMatch[1],
          ...(Object.hasOwn(payload, 'upload_request') ? { uploadRequest: payload.upload_request } : {}),
          tableId: scope.table_id,
          deviceId: scope.device_id,
          repo: dependencies.savedClipRepo,
          storage: dependencies.storage,
        })
        return response(200, { ok: true, data })
      }

      const savedReceiptMatch = path.match(
        /^\/v1\/sync\/saved-clips\/([A-Za-z0-9._-]+)\/upload-receipt$/u,
      )
      if (method === 'POST' && savedReceiptMatch) {
        if (!validUploadReceipt(payload)) throw new SyncApiError('SYNC_INVALID')
        if (typeof dependencies.savedClipRepo?.recordTrackedUploadReceipt !== 'function') throw new SyncApiError('SYNC_STORAGE_UNAVAILABLE')
        const data = await dependencies.savedClipRepo.recordTrackedUploadReceipt({
          jobId: savedReceiptMatch[1], tableId: scope.table_id, deviceId: scope.device_id, receipt: payload,
        })
        return response(200, { ok: true, data })
      }

      const savedCompleteMatch = path.match(
        /^\/v1\/sync\/saved-clips\/([A-Za-z0-9._-]+)\/complete$/u,
      )
      if (method === 'POST' && savedCompleteMatch) {
        if (Object.keys(payload).some(key => key !== 'upload_attempt')
            || (Object.hasOwn(payload, 'upload_attempt') && !validManagedUploadAttempt(payload.upload_attempt))) {
          throw new SyncApiError('SYNC_INVALID')
        }
        const data = await completeSavedClipJob({
          jobId: savedCompleteMatch[1],
          uploadAttempt: payload.upload_attempt,
          tableId: scope.table_id,
          deviceId: scope.device_id,
          repo: dependencies.savedClipRepo,
          storage: dependencies.storage,
          now: dependencies.nowIso(),
        })
        return response(200, { ok: true, data })
      }

      if (method === 'POST' && path === '/v1/sync/jobs') {
        const job = await createSyncJob({
          envelope: payload,
          idempotencyKey: getHeader(event.headers, 'x-idempotency-key'),
          manifestHash: getHeader(event.headers, 'x-manifest-sha256'),
          tableId: scope.table_id,
          deviceId: scope.device_id,
          repo: dependencies.repo,
          idFactory: dependencies.idFactory,
          now: dependencies.nowIso(),
        })
        const { _device_id: _privateDeviceId, ...publicJob } = job
        return response(200, { ok: true, data: publicJob })
      }

      const mediaMatch = path.match(/^\/v1\/sync\/jobs\/([A-Za-z0-9._-]+)\/media-authorizations$/u)
      if (method === 'POST' && mediaMatch) {
        const authorization = await authorizeMedia({
          jobId: mediaMatch[1],
          clipId: payload.clip_id,
          tableId: scope.table_id,
          deviceId: scope.device_id,
          repo: dependencies.repo,
          storage: dependencies.storage,
        })
        return response(200, { ok: true, data: authorization })
      }

      const completeMatch = path.match(/^\/v1\/sync\/jobs\/([A-Za-z0-9._-]+)\/complete$/u)
      if (method === 'POST' && completeMatch) {
        const job = await completeSyncJob({
          jobId: completeMatch[1],
          tableId: scope.table_id,
          deviceId: scope.device_id,
          repo: dependencies.repo,
          storage: dependencies.storage,
          now: dependencies.nowIso(),
        })
        const { _device_id: _privateDeviceId, ...publicJob } = job
        return response(200, { ok: true, data: publicJob })
      }
      throw new SyncApiError('SYNC_INVALID')
    } catch (error) {
      // Auth/nonce storage and dispatch share this boundary. A rolled-back
      // operational failure must remain retryable for the original device.
      const safeError = ownerDeviceRoute && !(error instanceof SyncApiError)
        ? new SyncApiError('SYNC_STORAGE_UNAVAILABLE') : error
      return errorResponse(safeError, dependencies.logger || console)
    }
  }
}


function createRuntimeDependencies({
  db, storageCloud, environment = process.env, managedSessionsEnabled = false,
  managedMediaAdmissionEnabled = false, clock = Date.now, registeredStorageOptions = {}, ownerCloudErasure = null,
}) {
  // Trusted internal construction only; environment flags/public payloads cannot
  // activate erasure or supply native operational acceptance evidence.
  const environmentId = runtimeAttestationFromEnvironment(environment).cloudbase_environment_id
  const ownerDeviceTaskRepo = createCloudOwnerDeviceTaskRepository(db, { environmentId: environmentId === 'unknown' ? null : environmentId, clock })
  let ownerCloudErasureWorker = null
  if (ownerCloudErasure && typeof ownerCloudErasure.nativeContractAdmission === 'function'
      && ownerCloudErasure.providerOptions && typeof ownerCloudErasure.providerOptions === 'object'
      && environmentId !== 'unknown' && ownerCloudErasure.providerOptions.environmentId === environmentId) {
    const { nativeContractAdmission, providerOptions } = ownerCloudErasure
    const binding = { environment_id: providerOptions.environmentId, bucket: providerOptions.bucket, region: providerOptions.region,
      provider_policy: 'cos_exact_target_absence_v1' }
    const admission = input => {
      const value = evaluateSynchronousNativeAdmission(nativeContractAdmission, input)
      return value && Object.entries(binding).every(([key, expected]) => value[key] === expected) ? value : null
    }
    ownerCloudErasureWorker = createOwnerCloudErasureWorker({
      repository: createOwnerCloudErasureRepository(db, { clock, nativeContractAdmission: admission, async onFinalizedReceipt(tx, receipt) {
        await enqueueReadyOwnerDeviceDeletion(tx, await ownerDeviceTaskRepo.issueVerified(tx, receipt))
      } }),
      provider: createOwnerErasureProvider({ ...providerOptions, clock }),
    })
  }
  const nowIso = () => new Date(clock()).toISOString()
  const authRepositories = createCloudAuthRepositories(db)
  const matchRepository = createMatchFactsRepository(db, { now: nowIso })
  const matchSetupService = createMatchSetupService({ repository: matchRepository, now: nowIso })
  return {
    ownerCloudErasureWorker,
    ownerDeviceTaskRepo,
    ownerDeviceDeliveryRepo: createCloudOwnerDeviceDeliveryRepository(db, { environmentId: environmentId === 'unknown' ? null : environmentId, taskRepository: ownerDeviceTaskRepo, cloudErasureWorker: ownerCloudErasureWorker, clock }),
    auth: { ...authRepositories, nowSeconds: () => Math.floor(clock() / 1000) },
    idFactory: () => `sync-${crypto.randomUUID()}`,
    nowIso,
    joinAdmin: createJoinAdmin(db, { nowIso,
      environment: environment.RANKING_RUNTIME_ENVIRONMENT || 'development',
      managedSessionsEnabled, managedMediaAdmissionEnabled,
    }),
    fieldCommandRepo: createCloudDeviceFieldCommandRepository(db),
    matchRepository,
    matchSetupService,
    mediaDeletionRepo: createCloudMediaDeletionRepository(db),
    analysisRepo: createCloudSessionAnalysisRepository(db),
    repo: createCloudSyncRepository(db),
    savedClipRepo: createCloudSavedClipRepository(db),
    storage: createStoragePort(createCloudBaseStorageProvider(storageCloud, registeredStorageOptions)),
    runtimeAttestation: runtimeAttestationFromEnvironment(environment),
  }
}

cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })
const storageCloud = cloudbase.init({ env: cloudbase.SYMBOL_CURRENT_ENV })
const db = cloud.database()

exports.createHandler = createHandler
exports.errorResponse = errorResponse
exports.runtimeAttestationFromEnvironment = runtimeAttestationFromEnvironment
exports.createRuntimeDependencies = createRuntimeDependencies
exports.main = createHandler(createRuntimeDependencies({ db, storageCloud }))
