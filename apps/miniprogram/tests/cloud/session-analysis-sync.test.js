const crypto = require('crypto')

const { canonicalSignatureInput } = require('../../cloudfunctions/sync-api/lib/auth')
const { createHandler } = require('../../cloudfunctions/sync-api/index')
const { putLiveShotFact } = require('../../cloudfunctions/sync-api/lib/session-analysis')

const SECRET = 'development-device-secret'
const NOW_SECONDS = 1788696000
const frozenDigest = ids => crypto.createHash('sha256').update(JSON.stringify(ids)).digest('hex')
const DIGEST = frozenDigest(['shot-1', 'shot-2'])

function projection(change = {}) {
  return {
    schema_version: '1.0', session_id: 'session-1', revision: 2,
    status: 'PARTIAL', total_shots: 2, available_shots: 1, retrying_shots: 1,
    report_available: true, deadline_at: '2026-09-06T12:02:00.000Z',
    updated_at: '2026-09-06T12:01:00.000Z', frozen_shots_digest: DIGEST,
    frozen_shot_ids: ['shot-1', 'shot-2'],
    published_shots: [{ session_id: 'session-1', shot_id: 'shot-1' }],
    published_clips: [{ session_id: 'session-1', shot_id: 'shot-1', clip_id: 'clip-1' }],
    ...change,
  }
}

function repository() {
  const records = new Map()
  const publishedClips = new Map([['clip-1', {
    clip_id: 'clip-1', session_id: 'session-1', shot_id: 'shot-1',
    artifact_id: 'artifact-1', upload_status: 'published',
  }]])
  return {
    records,
    async getSessionAuthority(sessionId) {
      return sessionId === 'session-1'
        ? { session_id: 'session-1', table_id: 'table-1', device_id: 'device-1', status: 'ended' }
        : null
    },
    async getPublishedSession(sessionId) {
      return sessionId === 'session-1' ? {
        session_id: 'session-1', artifact_id: 'artifact-1', sync_status: 'published',
        shots: [{ shot_id: 'shot-1' }, { shot_id: 'shot-2' }],
      } : null
    },
    async getPublishedClip(clipId) { return publishedClips.get(clipId) || null },
    async getProjection(sessionId) { return records.get(sessionId) || null },
    async saveProjection(value) { records.set(value.session_id, value); return value },
  }
}

function signedEvent(payload, { signature = true } = {}) {
  const method = 'PUT'
  const path = '/v1/field-sessions/session-1/analysis'
  const body = JSON.stringify(payload)
  const timestamp = String(NOW_SECONDS)
  const nonce = `nonce-${crypto.randomUUID()}`
  const bodyHash = crypto.createHash('sha256').update(body).digest('hex')
  const digest = crypto.createHmac('sha256', SECRET).update(canonicalSignatureInput({
    method, path, timestamp, nonce, bodyHash,
  })).digest('hex')
  return {
    httpMethod: method, path, body,
    headers: {
      'x-device-id': 'device-1', 'x-timestamp': timestamp, 'x-nonce': nonce,
      'x-body-sha256': bodyHash, 'x-signature': signature ? digest : '0'.repeat(64),
      'x-idempotency-key': 'analysis-session-1-revision-2',
    },
  }
}

function handler(repo) {
  return createHandler({
    auth: {
      nowSeconds: () => NOW_SECONDS,
      deviceRepo: { getActiveDevice: async () => ({
        device_id: 'device-1', table_id: 'table-1', status: 'active', secret: SECRET,
      }) },
      nonceRepo: { consume: async () => true },
    },
    analysisRepo: repo,
    nowIso: () => '2026-09-06T12:01:00.000Z',
  })
}

test('authenticated table device writes a session-isolated analysis projection', async () => {
  const repo = repository()

  const response = await handler(repo)(signedEvent(projection()))

  expect(response.statusCode).toBe(200)
  expect(JSON.parse(response.body).data).toMatchObject({
    session_id: 'session-1', revision: 2, status: 'PARTIAL',
  })
  expect(repo.records.get('session-1')).toMatchObject({
    _table_id: 'table-1', _device_id: 'device-1', frozen_shots_digest: DIGEST,
  })
})

test('same revision and digest is idempotent while changed content conflicts', async () => {
  const repo = repository()
  const run = handler(repo)
  expect((await run(signedEvent(projection()))).statusCode).toBe(200)
  expect((await run(signedEvent(projection()))).statusCode).toBe(200)

  const changed = projection({ frozen_shots_digest: 'b'.repeat(64) })
  const response = await run(signedEvent(changed))
  expect(response.statusCode).toBe(409)
  expect(JSON.parse(response.body).error.code).toBe('SYNC_CONFLICT')
})

test('live shot fact upserts once by trusted session shot and retry key without video', async () => {
  const records = new Map()
  const progress = new Map()
  const identity = {
    session_id: 'session-1', shot_id: 'shot-0001', sequence: 1,
    start_frame_id: 100, settle_frame_id: 220,
  }
  const idempotencyKey = `live-shot-${crypto.createHash('sha256')
    .update(JSON.stringify(Object.fromEntries(Object.entries(identity).sort())))
    .digest('hex')}`
  const fact = {
    ...identity, mode: 'observation', actor_role: 'red', actor_user_id: 'user-red',
    observed_outcome: 'unknown', rule_revision: 3,
    recording_reference: 'session-recording:epoch-1:100-220',
    idempotency_key: idempotencyKey,
  }
  const repo = {
    getSessionAuthority: async () => ({
      session_id: 'session-1', table_id: 'table-1', device_id: 'device-1', status: 'open',
    }),
    getLiveShotFact: async key => records.get(key) || null,
    saveLiveShotFact: async value => { records.set(value.idempotency_key, value); return value },
    getLiveShotProgress: async sessionId => progress.get(sessionId) || null,
    saveLiveShotProgress: async value => { progress.set(value.session_id, value); return value },
  }

  const first = await putLiveShotFact({
    sessionId: 'session-1', shotId: 'shot-0001', fact,
    tableId: 'table-1', deviceId: 'device-1', idempotencyKey, repo,
  })
  const second = await putLiveShotFact({
    sessionId: 'session-1', shotId: 'shot-0001', fact,
    tableId: 'table-1', deviceId: 'device-1', idempotencyKey, repo,
  })

  expect(second).toEqual(first)
  expect(records).toHaveProperty('size', 1)
  expect(progress.get('session-1')).toMatchObject({
    session_id: 'session-1', shot_count: 1, last_sequence: 1, last_shot_id: 'shot-0001',
  })
  expect(JSON.stringify(first)).not.toMatch(/video|media_bytes|local_path/u)
})

test('out-of-order unique live facts advance recorded count without regressing last sequence', async () => {
  const records = new Map()
  const progress = new Map()
  const repo = {
    getSessionAuthority: async () => ({
      session_id: 'session-1', table_id: 'table-1', device_id: 'device-1', status: 'open',
    }),
    getLiveShotFact: async key => records.get(key) || null,
    saveLiveShotFact: async value => { records.set(value.idempotency_key, value); return value },
    getLiveShotProgress: async sessionId => progress.get(sessionId) || null,
    saveLiveShotProgress: async value => { progress.set(value.session_id, value); return value },
  }
  const write = async sequence => {
    const identity = {
      session_id: 'session-1', shot_id: `shot-${sequence}`, sequence,
      start_frame_id: sequence * 100, settle_frame_id: sequence * 100 + 50,
    }
    const idempotencyKey = `live-shot-${crypto.createHash('sha256')
      .update(JSON.stringify(Object.fromEntries(Object.entries(identity).sort())))
      .digest('hex')}`
    return putLiveShotFact({
      sessionId: 'session-1', shotId: identity.shot_id,
      fact: {
        ...identity, mode: 'assistance', actor_role: 'red', actor_user_id: 'user-red',
        observed_outcome: 'unknown', rule_revision: 1,
        recording_reference: `session-recording:epoch-1:${identity.start_frame_id}-${identity.settle_frame_id}`,
        idempotency_key: idempotencyKey,
      },
      tableId: 'table-1', deviceId: 'device-1', idempotencyKey, repo,
    })
  }

  await write(2)
  await write(1)

  expect(progress.get('session-1')).toMatchObject({
    shot_count: 2, last_sequence: 2, last_shot_id: 'shot-2',
  })
})

test.each([
  ['bad device signature', projection(), { signature: false }, 401],
  ['URL and body session mismatch', projection({ session_id: 'session-2' }), {}, 409],
  ['cross-session clip', projection({
    published_clips: [{ session_id: 'session-2', shot_id: 'shot-1', clip_id: 'clip-1' }],
  }), {}, 409],
  ['clip outside frozen shots', projection({
    published_clips: [{ session_id: 'session-1', shot_id: 'shot-9', clip_id: 'clip-9' }],
  }), {}, 409],
  ['processing without counts cannot claim published content', projection({
    status: 'PROCESSING', total_shots: null, available_shots: null,
    retrying_shots: null, report_available: false,
  }), {}, 400],
])('%s is rejected without replacing the projection', async (_label, payload, options, status) => {
  const repo = repository()
  const response = await handler(repo)(signedEvent(payload, options))
  expect(response.statusCode).toBe(status)
  expect(repo.records.size).toBe(0)
})

test.each([
  ['legacy authority without device identity', async repo => {
    repo.getSessionAuthority = async () => ({
      session_id: 'session-1', table_id: 'table-1', status: 'ended',
    })
  }],
  ['different authoritative device', async repo => {
    repo.getSessionAuthority = async () => ({
      session_id: 'session-1', table_id: 'table-1', device_id: 'device-2', status: 'ended',
    })
  }],
  ['digest not bound to frozen shot IDs', async (_repo, payload) => {
    payload.frozen_shots_digest = 'b'.repeat(64)
  }],
  ['published shot missing from canonical session', async (_repo, payload) => {
    payload.frozen_shot_ids = ['shot-1', 'shot-9']
    payload.frozen_shots_digest = frozenDigest(payload.frozen_shot_ids)
    payload.published_shots = [{ session_id: 'session-1', shot_id: 'shot-9' }]
    payload.published_clips = [{
      session_id: 'session-1', shot_id: 'shot-9', clip_id: 'clip-1',
    }]
  }],
  ['published clip is absent', async repo => {
    repo.getPublishedClip = async () => null
  }],
  ['report is not actually published', async repo => {
    repo.getPublishedSession = async () => null
  }],
])('%s cannot advance analysis state', async (_label, mutate) => {
  const repo = repository()
  const payload = projection()
  await mutate(repo, payload)
  const response = await handler(repo)(signedEvent(payload))
  expect(response.statusCode).toBe(409)
  expect(JSON.parse(response.body).error.code).toBe('SYNC_CONFLICT')
  expect(repo.records.size).toBe(0)
})
