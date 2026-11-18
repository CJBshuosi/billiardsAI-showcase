const {
  createCloudJoinRepository,
  digestTableCode,
  requestJoin,
} = require('../../cloudfunctions/client-api/lib/join')


const TABLE_CODE = 'DevelopmentTableCode_00000000001'


function openTableRepository() {
  const requests = new Map()
  return {
    calls: [],
    async resolveRequest(input) {
      this.calls.push(input)
      const key = `${input.codeDigest}:${input.userId}`
      if (!requests.has(key)) {
        requests.set(key, {
          schema_version: '1.0',
          request_id: `request-${input.userId}`,
          session_id: 'session-0001',
          user_id: input.userId,
          display_name: input.displayName,
          status: 'pending',
          role: null,
        })
      }
      return requests.get(key)
    },
  }
}


test('permanent table code resolves the current open session idempotently', async () => {
  const repo = openTableRepository()
  const first = await requestJoin({
    tableCode: TABLE_CODE,
    userId: 'user-a',
    now: '2026-08-27T10:00:00.000Z',
    repo,
  })
  const second = await requestJoin({
    tableCode: TABLE_CODE,
    userId: 'user-a',
    now: '2026-08-27T10:00:01.000Z',
    repo,
  })

  expect(second.request_id).toBe(first.request_id)
  expect(first).toMatchObject({
    session_id: 'session-0001',
    display_name: '球友 FC9529',
    status: 'pending',
    role: null,
  })
  expect(repo.calls[0].codeDigest).toMatch(/^[a-f0-9]{64}$/u)
  expect(JSON.stringify(first)).not.toContain(TABLE_CODE)
  expect(JSON.stringify(repo.calls)).not.toContain(TABLE_CODE)
})


test('invalid table code fails before repository access', async () => {
  const repo = { resolveRequest: jest.fn() }

  await expect(requestJoin({
    tableCode: 'short', userId: 'user-a', now: 'now', repo,
  })).rejects.toMatchObject({ code: 'TABLE_CODE_INVALID' })
  expect(repo.resolveRequest).not.toHaveBeenCalled()
})


test.each([
  ['TABLE_CODE_DISABLED'],
  ['SESSION_NOT_JOINABLE'],
  ['JOIN_RATE_LIMITED'],
])('%s remains a stable public error', async (code) => {
  const error = Object.assign(new Error(code), { code })
  const repo = { resolveRequest: jest.fn().mockRejectedValue(error) }

  await expect(requestJoin({
    tableCode: TABLE_CODE,
    userId: 'user-a',
    now: '2026-08-27T10:00:00.000Z',
    repo,
  })).rejects.toMatchObject({ code })
})


test('table code digest is deterministic without exposing the source code', () => {
  expect(digestTableCode(TABLE_CODE)).toMatch(/^[a-f0-9]{64}$/u)
  expect(digestTableCode(TABLE_CODE)).toBe(digestTableCode(TABLE_CODE))
  expect(digestTableCode(TABLE_CODE)).not.toContain(TABLE_CODE)
})


test('first join tolerates missing binding and rate-limit documents in cloud transactions', async () => {
  const codeDigest = digestTableCode(TABLE_CODE)
  const collections = new Map([
    ['table_join_codes', new Map([['table-1', {
      table_id: 'table-1', code_digest: codeDigest, status: 'active',
      current_session_id: 'session-1',
    }]])],
    ['field_live_sessions', new Map([['session-1', {
      session_id: 'session-1', table_id: 'table-1', status: 'open',
    }]])],
    ['session_user_bindings', new Map()],
    ['session_join_requests', new Map()],
    ['join_rate_limits', new Map()],
  ])
  const collection = (name) => ({
    where(query) {
      return {
        limit() {
          return {
            async get() {
              return {
                data: [...collections.get(name).values()].filter((item) => (
                  Object.entries(query).every(([key, value]) => item[key] === value)
                )),
              }
            },
          }
        },
      }
    },
    doc(id) {
      return {
        async get() {
          const data = collections.get(name).get(id)
          if (!data) throw new Error(`document.get:fail document with _id ${id} does not exist`)
          return { data }
        },
        async set({ data }) { collections.get(name).set(id, data) },
      }
    },
  })
  const db = {
    collection,
    async runTransaction(callback) { return callback({ collection }) },
  }
  const repo = createCloudJoinRepository(db, { idFactory: () => 'join-1' })

  await expect(requestJoin({
    tableCode: TABLE_CODE,
    userId: 'user-a',
    now: '2026-08-30T14:49:00.000Z',
    repo,
  })).resolves.toMatchObject({
    request_id: 'join-1', session_id: 'session-1', status: 'pending',
  })
})
