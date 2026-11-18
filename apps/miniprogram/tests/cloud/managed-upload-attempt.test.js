let api = {}
try { api = require('../../cloudfunctions/shared/managed-upload-attempt') } catch (error) { if (error.code !== 'MODULE_NOT_FOUND') throw error }
const { MANAGED_UPLOAD_ATTEMPT_CONTRACT, makeManagedUploadAttempt, validManagedUploadAttempt } = api
const operationId = '01234567-89ab-4cde-8fab-0123456789ab'
const attempt = { contract: 'managed_upload_attempt_v1', operation_id: operationId, sequence: 1 }
const ticket = { clip_id: 'clip-1', operation_id: operationId, sequence: 1,
  epoch: 'abcdefab-cdef-4abc-9def-abcdefabcdef', identity_sha256: 'a'.repeat(64),
  target: { object_id: `key-${'b'.repeat(64)}`, kind: 'staging', cloud_path: 'private-target' } }

test('projects a private ticket to a strict public correlation without disclosing private authority', () => {
  expect(MANAGED_UPLOAD_ATTEMPT_CONTRACT).toBe('managed_upload_attempt_v1')
  const value = makeManagedUploadAttempt(ticket)
  expect(value).toEqual(attempt); expect(validManagedUploadAttempt(value)).toBe(true)
  value.sequence = 2; expect(ticket.sequence).toBe(1)
  expect(makeManagedUploadAttempt({ ...ticket, sequence: 64 })).toEqual({ ...attempt, sequence: 64 })
  expect(validManagedUploadAttempt({ ...attempt, sequence: 64 })).toBe(true)
  expect(makeManagedUploadAttempt({ operation_id: operationId, sequence: 1 })).toEqual(attempt)
})

test.each([
  null, undefined, [], 'secret', {}, { ...attempt, contract: 'other' },
  { ...attempt, sequence: 0 }, { ...attempt, sequence: 65 }, { ...attempt, sequence: 1.5 },
  { ...attempt, sequence: '1' }, { ...attempt, sequence: true }, { ...attempt, sequence: NaN },
  { ...attempt, sequence: Number.MAX_SAFE_INTEGER + 1 },
  { ...attempt, operation_id: operationId.toUpperCase() },
  { ...attempt, operation_id: '01234567-89ab-5cde-8fab-0123456789ab' },
  { ...attempt, operation_id: '01234567-89ab-4cde-7fab-0123456789ab' },
  { ...attempt, operation_id: [operationId] }, { ...attempt, operation_id: 1 },
  { ...attempt, operation_id: `${operationId}\n` },
  { ...attempt, token: 'secret' }, { ...attempt, file_id: 'cloud://private' },
  { ...attempt, epoch: ticket.epoch }, { contract: attempt.contract, operation_id: operationId },
])('rejects malformed or expanded public correlation without coercion: %p', value => {
  expect(validManagedUploadAttempt(value)).toBe(false)
})

test.each([null, undefined, [], 'token=private', {}, { ...ticket, operation_id: ['secret'] },
  Object.create({ operation_id: operationId, sequence: 1 }),
  { ...ticket, operation_id: operationId.toUpperCase() }, { ...ticket, sequence: '1' },
  { ...ticket, sequence: true }, { ...ticket, sequence: 0 }, { ...ticket, sequence: 65 }])(
  'invalid ticket projection throws a fixed safe TypeError: %p', value => {
    let thrown
    try { makeManagedUploadAttempt(value) } catch (error) { thrown = error }
    expect(thrown).toBeInstanceOf(TypeError)
    expect(thrown.message).toBe('Invalid managed upload attempt')
  })
