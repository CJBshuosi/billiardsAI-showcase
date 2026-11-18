const { ClientApiError, stableError } = require('../../cloudfunctions/client-api/lib/errors')


test('unexpected client API failures are logged but not exposed to the mini program', () => {
  const logger = { error: jest.fn() }
  const result = stableError(Object.assign(new Error('database unavailable'), {
    code: 'DATABASE_ERROR',
  }), logger)

  expect(result).toEqual({ ok: false, error: { code: 'INTERNAL_ERROR' } })
  expect(logger.error).toHaveBeenCalledWith('client-api unexpected error', {
    name: 'Error',
    code: 'DATABASE_ERROR',
    message: 'database unavailable',
  })
})


test('expected stable client API failures are not logged as internal errors', () => {
  const logger = { error: jest.fn() }
  expect(stableError(new ClientApiError('AUTH_REQUIRED'), logger)).toEqual({
    ok: false,
    error: { code: 'AUTH_REQUIRED' },
  })
  expect(logger.error).not.toHaveBeenCalled()
})

test('legacy favorite actions use a stable unsupported result code', () => {
  const { normalizeResult } = require('../../miniprogram/utils/result')
  const { legacyFavoriteResult } = require('../../cloudfunctions/client-api/lib/content')
  for (const action of ['favorites.list', 'favorites.set']) {
    expect(legacyFavoriteResult(action)).toEqual({
      ok: false,
      error: { code: 'ACTION_UNSUPPORTED' },
    })
  }
  expect(legacyFavoriteResult('clips.list')).toBeNull()
  expect(normalizeResult({ ok: false, error: { code: 'ACTION_UNSUPPORTED' } })).toEqual({
    ok: false,
    error: { code: 'ACTION_UNSUPPORTED' },
  })
})
