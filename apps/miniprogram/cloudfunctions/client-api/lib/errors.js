class ClientApiError extends Error {
  constructor(code) {
    super(code)
    this.name = 'ClientApiError'
    this.code = code
  }
}

function stableError(error, logger = console) {
  if (error instanceof ClientApiError) {
    return { ok: false, error: { code: error.code } }
  }
  const details = {
    name: error && typeof error.name === 'string' ? error.name : 'UnknownError',
    code: error && typeof error.code === 'string' ? error.code : null,
    message: error && typeof error.message === 'string'
      ? error.message.slice(0, 500)
      : 'unknown internal failure',
  }
  logger.error('client-api unexpected error', details)
  return { ok: false, error: { code: 'INTERNAL_ERROR' } }
}

module.exports = { ClientApiError, stableError }
