const {
  createIndexDefinition,
  ensureIndexes,
  missingIndexDefinitions,
} = require('../../scripts/apply-development-indexes')


test('development indexes preserve field order, direction and uniqueness', () => {
  expect(createIndexDefinition({
    collection: 'ranking_settlement_jobs',
    fields: [['ranking_scope_id', 'asc'], ['status', 'asc'], ['created_at', 'asc']],
    unique: false,
  })).toMatchObject({
    IndexName: expect.stringMatching(/^idx_[a-f0-9]{16}$/),
    MgoKeySchema: {
      MgoIndexKeys: [
        { Name: 'ranking_scope_id', Direction: '1' },
        { Name: 'status', Direction: '1' },
        { Name: 'created_at', Direction: '1' },
      ],
      MgoIsUnique: false,
      MgoIsSparse: false,
    },
  })
})


test('development indexes add only signatures missing from cloud state', () => {
  const desired = [
    { collection: 'matches', fields: [['session_id', 'asc']], unique: true },
    { collection: 'matches', fields: [['started_at', 'desc']], unique: false },
  ]
  const existing = [{
    Name: 'existing', Unique: true,
    Keys: [{ Name: 'session_id', Direction: '1' }],
  }]

  const missing = missingIndexDefinitions(desired, existing)

  expect(missing).toHaveLength(1)
  expect(missing[0].MgoKeySchema).toMatchObject({
    MgoIndexKeys: [{ Name: 'started_at', Direction: '-1' }],
    MgoIsUnique: false,
  })
})


test('development indexes re-read cloud state after a transient update timeout', () => {
  const desired = {
    collection: 'match_events', fields: [['event_id', 'asc']], unique: true,
  }
  let described = 0
  let updated = 0
  const waits = []
  const run = (_command, args) => {
    const action = args[2]
    if (action === 'DescribeTable') {
      described += 1
      const indexes = described === 1 ? [] : [{
        Name: 'created-in-background', Unique: true,
        Keys: [{ Name: 'event_id', Direction: '1' }],
      }]
      return { status: 0, stdout: JSON.stringify({ data: { Indexes: indexes } }) }
    }
    updated += 1
    return { status: 1, stderr: 'FailedOperation.Timeout: temporarily busy' }
  }

  const result = ensureIndexes({
    environmentId: 'development-env', indexes: [desired], run,
    wait: milliseconds => waits.push(milliseconds),
  })

  expect(result).toEqual({ added: 0, existing: 1 })
  expect({ described, updated, waits }).toEqual({ described: 2, updated: 1, waits: [3000] })
})
