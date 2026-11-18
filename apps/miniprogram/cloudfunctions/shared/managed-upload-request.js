const { digest } = require('./shot-ownership')

const MANAGED_UPLOAD_REQUEST_CONTRACT = 'managed_upload_request_v1'
const MANAGED_UPLOAD_REQUEST_BINDING_CONTRACT = 'managed_upload_request_binding_v1'
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/u
const HASH = /^[a-f0-9]{64}$/u
const ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,128}$/u
const BINDING_FIELDS = ['contract', 'request_id', 'table_id', 'device_id', 'job_id', 'descriptor_sha256', 'identity', 'identity_sha256', 'ticket']
const TICKET_FIELDS = ['clip_id', 'operation_id', 'sequence', 'epoch', 'identity_sha256', 'target']
function exact(value, fields, metadata = false) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value)
    && Reflect.ownKeys(value).filter(key => !(metadata && key === '_id')).length === fields.length
    && fields.every(field => Object.hasOwn(value, field))
    && Reflect.ownKeys(value).every(key => fields.includes(key) || (metadata && key === '_id')))
}
function validManagedUploadRequest(value) {
  return exact(value, ['contract', 'request_id']) && value.contract === MANAGED_UPLOAD_REQUEST_CONTRACT
    && typeof value.request_id === 'string' && UUID.test(value.request_id)
}
function managedUploadRequestId({ tableId, deviceId, uploadRequest }) {
  return digest({ contract: MANAGED_UPLOAD_REQUEST_CONTRACT, table_id: tableId, device_id: deviceId, request_id: uploadRequest.request_id })
}
// Shape/provenance only. The repository must validate the full identity and
// reserved ticket against the authority's complete original inventory.
function validManagedUploadRequestBinding(value) {
  return Boolean(exact(value, BINDING_FIELDS, true) && value.contract === MANAGED_UPLOAD_REQUEST_BINDING_CONTRACT
    && typeof value.request_id === 'string' && UUID.test(value.request_id)
    && ['table_id', 'device_id', 'job_id'].every(field => typeof value[field] === 'string' && ID.test(value[field]))
    && typeof value.descriptor_sha256 === 'string' && HASH.test(value.descriptor_sha256)
    && typeof value.identity_sha256 === 'string' && HASH.test(value.identity_sha256)
    && value.identity && value.identity.publication_kind === 'saved_private'
    && value.identity.table_id === value.table_id && value.identity.device_id === value.device_id
    && digest(value.identity) === value.identity_sha256 && exact(value.ticket, TICKET_FIELDS))
}
module.exports = { MANAGED_UPLOAD_REQUEST_CONTRACT, MANAGED_UPLOAD_REQUEST_BINDING_CONTRACT,
  validManagedUploadRequest, managedUploadRequestId, validManagedUploadRequestBinding }
