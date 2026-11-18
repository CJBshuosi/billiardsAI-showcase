const { ClientApiError } = require('./errors')
const {
  OwnerClipDeletionError,
  createCloudOwnerDeleteRepository,
  requestOwnerClipDeletion,
} = require('../../shared/owner-delete')

const ERROR_CODES = {
  OWNER_DELETE_INVALID: 'REQUEST_INVALID',
  OWNER_DELETE_UNAVAILABLE: 'MEDIA_NOT_AVAILABLE',
  OWNER_DELETE_CONFLICT: 'REQUEST_CONFLICT',
}

async function deleteOwnerClip({ payload, userId, db, now }) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)
      || Object.keys(payload).length !== 2
      || !Object.hasOwn(payload, 'clip_id') || !Object.hasOwn(payload, 'request_id')) {
    throw new ClientApiError('REQUEST_INVALID')
  }
  try {
    return await requestOwnerClipDeletion({
      clipId: payload.clip_id, requestId: payload.request_id, userId,
      repo: createCloudOwnerDeleteRepository(db), now,
    })
  } catch (error) {
    if (error instanceof OwnerClipDeletionError && ERROR_CODES[error.code]) {
      throw new ClientApiError(ERROR_CODES[error.code])
    }
    throw error
  }
}

module.exports = { deleteOwnerClip }
