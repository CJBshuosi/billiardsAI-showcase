const MANAGED_UPLOAD_ATTEMPT_CONTRACT = 'managed_upload_attempt_v1'
const FIELDS = ['contract', 'operation_id', 'sequence']
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/u

function validCorrelation(value) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value)
    && Object.hasOwn(value, 'operation_id') && Object.hasOwn(value, 'sequence')
    && typeof value.operation_id === 'string' && UUID.test(value.operation_id)
    && Number.isSafeInteger(value.sequence) && value.sequence >= 1 && value.sequence <= 64)
}
function makeManagedUploadAttempt(ticket) {
  if (!validCorrelation(ticket)) throw new TypeError('Invalid managed upload attempt')
  return { contract: MANAGED_UPLOAD_ATTEMPT_CONTRACT, operation_id: ticket.operation_id, sequence: ticket.sequence }
}
function validManagedUploadAttempt(value) {
  return validCorrelation(value) && value.contract === MANAGED_UPLOAD_ATTEMPT_CONTRACT
    && Object.keys(value).length === FIELDS.length && FIELDS.every(field => Object.hasOwn(value, field))
    && Object.keys(value).every(field => FIELDS.includes(field))
}
module.exports = { MANAGED_UPLOAD_ATTEMPT_CONTRACT, makeManagedUploadAttempt, validManagedUploadAttempt }
