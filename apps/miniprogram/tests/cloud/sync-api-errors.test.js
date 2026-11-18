const { errorResponse } = require('../../cloudfunctions/sync-api')
const { SyncApiError } = require('../../cloudfunctions/sync-api/lib/auth')


test('unexpected sync API failures are logged without exposing details', () => {
  const logger = { error: jest.fn() }
  const response = errorResponse(Object.assign(new Error('wxacode unavailable'), {
    code: 'OPENAPI_FAILED',
    errCode: -604101,
    errMsg: 'permission denied',
  }), logger)

  expect(JSON.parse(response.body)).toEqual({
    ok: false, error: { code: 'INTERNAL_ERROR' },
  })
  expect(logger.error).toHaveBeenCalledWith('sync-api unexpected error', {
    name: 'Error',
    code: 'OPENAPI_FAILED',
    errCode: -604101,
    errMsg: 'permission denied',
    message: 'wxacode unavailable',
  })
})


test('expected sync API failures are not logged as internal errors', () => {
  const logger = { error: jest.fn() }
  const response = errorResponse(new SyncApiError('SYNC_CONFIGURATION_REQUIRED'), logger)

  expect(JSON.parse(response.body)).toEqual({
    ok: false, error: { code: 'SYNC_CONFIGURATION_REQUIRED' },
  })
  expect(logger.error).not.toHaveBeenCalled()
})


test('storage verification races remain retryable without exposing details', () => {
  const result = errorResponse(new SyncApiError('SYNC_PARTIAL'))

  expect(result.statusCode).toBe(503)
  expect(JSON.parse(result.body)).toEqual({ ok: false, error: { code: 'SYNC_PARTIAL' } })
})
