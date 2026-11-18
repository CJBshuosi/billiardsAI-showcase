const https = require('https')
const COS = require('cos-nodejs-sdk-v5')
const { parseString } = require('xml2js')
const { canonicalJson, digest, documentId } = require('../../shared/shot-ownership')

const DEADLINE_MS = 10000
const METADATA_BYTES = 256 * 1024
const ERROR_BYTES = 8 * 1024
const HEADER_BYTES = 8 * 1024
const PAGE_SIZE = 16
const CURSOR_WINDOW = 128
const HASH = /^[a-f0-9]{64}$/u
const UUID = '[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}'
const TARGET_FIELDS = ['sequence', 'operation_id', 'object_id', 'kind', 'cloud_path', 'file_id', 'writer_state', 'receipt_sha256']
const CHECKPOINT_FIELDS = ['contract', 'target_sha256', 'mode', 'phase', 'cursor', 'cursor_sha256', 'pages', 'versions', 'transcript_sha256']
const MODES = ['NeverEnabled', 'Enabled', 'Suspended']
const PHASES = ['versioning', 'simple_delete', 'simple_probe', 'list', 'delete', 'version_probe', 'recheck']
class OwnerErasureProviderError extends Error {
  constructor(suffix) { const code = `OWNER_ERASURE_${suffix}`; super(code); this.name = 'OwnerErasureProviderError'; this.code = code }
}
function fail(code) { throw new OwnerErasureProviderError(code) }
function exact(value, fields) {
  return Boolean(value && [Object.prototype, null].includes(Object.getPrototypeOf(value))
    && Reflect.ownKeys(value).length === fields.length && fields.every(key => Object.hasOwn(value, key)
      && Object.getOwnPropertyDescriptor(value, key).enumerable && Object.hasOwn(Object.getOwnPropertyDescriptor(value, key), 'value')))
}
function text(value, max = 1024, empty = false) {
  return typeof value === 'string' && (empty || value.length > 0) && Buffer.byteLength(value) <= max
    && !/[\u0000-\u001f\u007f]/u.test(value) && Buffer.from(value).toString('utf8') === value
}
const dense = (value, max) => Array.isArray(value) && value.length <= max && Reflect.ownKeys(value).length === value.length + 1
  && Array.from({ length: value.length }, (_, i) => i).every(i => Object.hasOwn(value, i)
    && Object.hasOwn(Object.getOwnPropertyDescriptor(value, i), 'value') && Object.getOwnPropertyDescriptor(value, i).enumerable)
function validTarget(target, environmentId, bucket) {
  if (!exact(target, TARGET_FIELDS) || !Number.isSafeInteger(target.sequence) || target.sequence < 1 || target.sequence > 64
      || !text(target.operation_id) || !new RegExp(`^${UUID}$`, 'u').test(target.operation_id)
      || !['staging', 'published'].includes(target.kind) || !text(target.cloud_path)
      || !['upload_succeeded', 'upload_retired', 'copy_succeeded'].includes(target.writer_state)
      || !(target.receipt_sha256 === null || (typeof target.receipt_sha256 === 'string' && HASH.test(target.receipt_sha256)))) return false
  const namespace = target.kind === 'staging' ? 'managed-clips-v1/[a-f0-9]{64}/staging/' : 'published/managed-clips-v1/[a-f0-9]{64}/'
  return new RegExp(`^${namespace}${target.operation_id}\\.mp4$`, 'u').test(target.cloud_path)
    && target.object_id === `key-${documentId(target.cloud_path)}`
    && target.file_id === `cloud://${environmentId}.${bucket}/${target.cloud_path}`
}
function fresh(targetHash) {
  return { contract: 'owner_erasure_provider_v1', target_sha256: targetHash, mode: null, phase: 'versioning', cursor: null,
    cursor_sha256: [], pages: 0, versions: [], transcript_sha256: digest({ contract: 'owner_erasure_transcript_v1', target_sha256: targetHash }) }
}
function validCheckpoint(cp, targetHash, key) {
  if (!exact(cp, CHECKPOINT_FIELDS) || cp.contract !== 'owner_erasure_provider_v1' || cp.target_sha256 !== targetHash
      || !PHASES.includes(cp.phase) || !(cp.mode === null || MODES.includes(cp.mode)) || typeof cp.transcript_sha256 !== 'string' || !HASH.test(cp.transcript_sha256)
      || !Number.isSafeInteger(cp.pages) || cp.pages < 0 || cp.pages > CURSOR_WINDOW
      || !dense(cp.cursor_sha256, CURSOR_WINDOW) || cp.cursor_sha256.length !== cp.pages
      || !cp.cursor_sha256.every(h => typeof h === 'string' && HASH.test(h)) || new Set(cp.cursor_sha256).size !== cp.pages
      || !dense(cp.versions, PAGE_SIZE) || !cp.versions.every(v => text(v)) || new Set(cp.versions).size !== cp.versions.length) return false
  if (cp.cursor !== null && (!exact(cp.cursor, ['key', 'version']) || !text(cp.cursor.key) || !cp.cursor.key.startsWith(key)
      || !text(cp.cursor.version, 1024, true) || cp.cursor_sha256.at(-1) !== digest(cp.cursor))) return false
  if ((cp.cursor === null) !== (cp.pages === 0) || (cp.phase === 'delete') !== (cp.versions.length > 0)) return false
  if (cp.phase === 'versioning') return cp.mode === null && cp.pages === 0
  if (cp.mode === null) return false
  if (['simple_delete', 'simple_probe'].includes(cp.phase) && cp.mode !== 'NeverEnabled') return false
  if (['list', 'delete', 'version_probe'].includes(cp.phase) && cp.mode === 'NeverEnabled') return false
  return !(['simple_delete', 'simple_probe', 'recheck'].includes(cp.phase) && cp.pages > 0)
}

function safeXmlReferences(source) {
  // Do not permit the parser's broader named-entity table. Standard references
  // expand to at most one code point, bounded by the existing raw-body limit.
  for (let offset = source.indexOf('&'); offset !== -1;) {
    const end = source.indexOf(';', offset + 1)
    if (end === -1) fail('XML')
    const reference = source.slice(offset + 1, end)
    if (!['amp', 'lt', 'gt', 'quot', 'apos'].includes(reference)) {
      if (!/^#(?:[0-9]+|x[0-9a-fA-F]+)$/u.test(reference)) fail('XML')
      const point = reference[1] === 'x' ? Number.parseInt(reference.slice(2), 16) : Number.parseInt(reference.slice(1), 10)
      if (!(point === 9 || point === 10 || point === 13 || (point >= 0x20 && point <= 0xD7FF)
          || (point >= 0xE000 && point <= 0xFFFD) || (point >= 0x10000 && point <= 0x10FFFF))) fail('XML')
    }
    offset = source.indexOf('&', end + 1)
  }
}
// Bounded raw XML is parsed by pinned xml2js. Reject DTD/custom declarations and
// excessive structure before SAX; permit standard escapes, never normalize roots.
function xml(body, root) {
  const source = body.toString('utf8')
  if (!source || !Buffer.from(source).equals(body) || /<!/u.test(source)
      || /<\?(?!xml\s[^?]*\?>)/u.test(source)) fail('XML')
  safeXmlReferences(source)
  let depth = 0; let nodes = 0
  for (const token of source.matchAll(/<[^>]*>/gu)) {
    const tag = token[0]
    if (tag.startsWith('<?')) continue
    if (tag.startsWith('</')) depth--
    else { nodes++; if (!tag.endsWith('/>')) depth++ }
    if (depth < 0 || depth > 16 || nodes > 512) fail('XML')
  }
  let result; let error
  parseString(source, { explicitArray: true, explicitRoot: true, strict: true, trim: false, normalize: false, async: false }, (err, value) => { error = err; result = value })
  if (error || !exact(result, [root])) fail('XML')
  let value = result[root]
  if (typeof value === 'string' && value.trim() === '') value = {}
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('XML')
  if (Object.hasOwn(value, '$')) {
    if (!exact(value.$, ['xmlns']) || value.$.xmlns !== 'http://doc.s3.amazonaws.com/2006-03-01') fail('XML')
    delete value.$
  }
  return value
}
function fields(value, required, optional = []) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || !required.every(k => Object.hasOwn(value, k)) || Object.keys(value).some(k => !required.includes(k) && !optional.includes(k))) fail('XML')
}
function scalar(value, name, empty = false) {
  const rows = value[name]
  if (!Array.isArray(rows) || rows.length !== 1 || !text(rows[0], 1024, empty)) fail('XML')
  return rows[0]
}
function versioning(body) {
  const value = xml(body, 'VersioningConfiguration'); fields(value, [], ['Status'])
  if (!Object.hasOwn(value, 'Status')) return 'NeverEnabled'
  const mode = scalar(value, 'Status')
  if (!['Enabled', 'Suspended'].includes(mode)) fail('XML')
  return mode
}
function versions(body, cp, bucket, key) {
  const value = xml(body, 'ListVersionsResult')
  fields(value, ['Name', 'Prefix', 'KeyMarker', 'VersionIdMarker', 'MaxKeys', 'IsTruncated'], ['Version', 'DeleteMarker', 'NextKeyMarker', 'NextVersionIdMarker'])
  if (scalar(value, 'Name') !== bucket || scalar(value, 'Prefix') !== key || scalar(value, 'KeyMarker', true) !== (cp.cursor?.key || '')
      || scalar(value, 'VersionIdMarker', true) !== (cp.cursor?.version || '') || scalar(value, 'MaxKeys') !== '16') fail('XML')
  const truncated = scalar(value, 'IsTruncated')
  if (!['true', 'false'].includes(truncated)) fail('XML')
  const ids = []; const seen = new Set(); let count = 0; let greatestKey = cp.cursor?.key || key
  for (const kind of ['Version', 'DeleteMarker']) {
    const rows = value[kind] || []
    if (!Array.isArray(rows) || rows.length > PAGE_SIZE) fail('XML')
    for (const row of rows) {
      fields(row, ['Key', 'VersionId', 'IsLatest', 'LastModified'], kind === 'Version' ? ['Owner', 'ETag', 'Size', 'StorageClass', 'StorageTier'] : ['Owner'])
      const rowKey = scalar(row, 'Key'); const id = scalar(row, 'VersionId')
      if (!rowKey.startsWith(key) || !['true', 'false'].includes(scalar(row, 'IsLatest')) || !Number.isFinite(Date.parse(scalar(row, 'LastModified')))) fail('XML')
      if (cp.cursor && Buffer.compare(Buffer.from(rowKey), Buffer.from(cp.cursor.key)) < 0) fail('CURSOR')
      if (Buffer.compare(Buffer.from(rowKey), Buffer.from(greatestKey)) > 0) greatestKey = rowKey
      if (Object.hasOwn(row, 'Owner')) {
        if (!Array.isArray(row.Owner) || row.Owner.length !== 1) fail('XML')
        fields(row.Owner[0], ['ID'], ['DisplayName']); scalar(row.Owner[0], 'ID')
        if (Object.hasOwn(row.Owner[0], 'DisplayName')) scalar(row.Owner[0], 'DisplayName', true)
      }
      for (const name of ['ETag', 'Size', 'StorageClass']) if (Object.hasOwn(row, name)) scalar(row, name)
      if (Object.hasOwn(row, 'StorageTier') && (scalar(row, 'StorageClass') !== 'INTELLIGENT_TIERING'
          || !['FREQUENT', 'INFREQUENT'].includes(scalar(row, 'StorageTier')))) fail('XML')
      if (Object.hasOwn(row, 'Size') && !/^\d{1,20}$/u.test(scalar(row, 'Size'))) fail('XML')
      const identity = canonicalJson([rowKey, id]); if (seen.has(identity)) fail('XML'); seen.add(identity)
      if (++count > PAGE_SIZE) fail('XML')
      if (rowKey === key) ids.push(id)
    }
  }
  let cursor = null
  if (truncated === 'true') {
    cursor = { key: scalar(value, 'NextKeyMarker'), version: scalar(value, 'NextVersionIdMarker', true) }
    if (!cursor.key.startsWith(key) || Buffer.compare(Buffer.from(cursor.key), Buffer.from(greatestKey)) < 0
        || cp.cursor_sha256.includes(digest(cursor))) fail('CURSOR')
  } else if (Object.hasOwn(value, 'NextKeyMarker') || Object.hasOwn(value, 'NextVersionIdMarker')) fail('XML')
  return { ids, cursor }
}
function absent(body, key) {
  const value = xml(body, 'Error'); fields(value, ['Code', 'Resource'], ['Message', 'RequestId', 'TraceId'])
  if (scalar(value, 'Code') !== 'NoSuchKey' || scalar(value, 'Resource') !== `/${key}`) fail('NATIVE')
  for (const name of ['Message', 'RequestId', 'TraceId']) if (Object.hasOwn(value, name)) scalar(value, name, true)
}

// request has the node:https.request signature. It is the only network seam; no COS
// client instance or SDK transport exists. Timer includes lazy IAM and full response.
function exchange({ request, getCredentials, beforeRequest, host, method, path, query, probe, observed }) {
  return new Promise((resolve, reject) => {
    let settled = false; let outgoing; let incoming; const started = Date.now()
    const finish = (error, value) => {
      if (settled) return
      settled = true; clearTimeout(timer)
      if (error || value?.present) { incoming?.destroy(); outgoing?.destroy() }
      if (error) reject(error); else resolve(value)
    }
    const timer = setTimeout(() => finish(new OwnerErasureProviderError('TIMEOUT')), DEADLINE_MS)
    Promise.resolve().then(getCredentials).then(async credentials => {
      if (settled) return
      if (!credentials || !text(credentials.SecretId, 256) || !text(credentials.SecretKey, 256)
          || !(credentials.Token === undefined || text(credentials.Token, 4096))) fail('CREDENTIALS')
      const headers = { Host: host, ...(credentials.Token ? { 'x-cos-security-token': credentials.Token } : {}), ...(probe ? { Range: 'bytes=0-0' } : {}) }
      const seconds = Math.floor(observed / 1000)
      headers.Authorization = COS.getAuthorization({ SecretId: credentials.SecretId, SecretKey: credentials.SecretKey, Method: method,
        Pathname: path, Query: query, Headers: headers, KeyTime: `${seconds};${seconds + 60}` })
      const encoded = Object.entries(query).map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`).join('&')
      // Local authorization is re-read after lazy IAM, immediately before issue.
      // Already-issued calls still need native stale-request/configuration proof.
      let admitted
      try { admitted = beforeRequest === undefined ? true : await beforeRequest() } catch { fail('FENCE') }
      if (settled) return
      if (Date.now() - started >= DEADLINE_MS) fail('TIMEOUT')
      if (admitted !== true) fail('FENCE')
      outgoing = request({ protocol: 'https:', hostname: host, port: 443, method, path: path + (encoded ? `?${encoded}` : ''), headers,
        agent: false, rejectUnauthorized: true, minVersion: 'TLSv1.2', maxHeaderSize: HEADER_BYTES, highWaterMark: ERROR_BYTES }, response => {
        incoming = response
        response.on('error', () => finish(new OwnerErasureProviderError('TRANSPORT')))
        response.once('aborted', () => finish(new OwnerErasureProviderError('TRANSPORT')))
        response.once('close', () => { if (!response.readableEnded) finish(new OwnerErasureProviderError('TRANSPORT')) })
        if (settled) { response.destroy(); return }
        const status = response.statusCode
        const cap = status >= 300 ? ERROR_BYTES : METADATA_BYTES
        const headerSize = Math.max(Buffer.byteLength((response.rawHeaders || []).join('\r\n')),
          Buffer.byteLength(Object.entries(response.headers || {}).map(([k, v]) => `${k}: ${v}`).join('\r\n')))
        const length = response.headers?.['content-length']; const encoding = response.headers?.['content-encoding']
        if (!Number.isInteger(status) || status < 100 || status > 599 || headerSize > HEADER_BYTES
            || (encoding && encoding !== 'identity') || (length !== undefined && (typeof length !== 'string' || !/^\d{1,12}$/u.test(length)))) {
          finish(new OwnerErasureProviderError('BOUNDS')); return
        }
        if (probe && [200, 206, 416].includes(status)) { finish(null, { status, body: Buffer.alloc(0), present: true }); return }
        if (length !== undefined && Number(length) > cap) { finish(new OwnerErasureProviderError('BOUNDS')); return }
        // A fixed allocation avoids one retained Buffer per tiny network chunk.
        const buffer = Buffer.allocUnsafe(cap); let bytes = 0
        response.on('data', chunk => {
          if (settled) return
          if (!Buffer.isBuffer(chunk) && !(chunk instanceof Uint8Array)) { finish(new OwnerErasureProviderError('TRANSPORT')); return }
          if (chunk.length > cap - bytes) { finish(new OwnerErasureProviderError('BOUNDS')); return }
          buffer.set(chunk, bytes); bytes += chunk.length
        })
        response.once('end', () => {
          if (length !== undefined && Number(length) !== bytes) { finish(new OwnerErasureProviderError('TRANSPORT')); return }
          finish(null, { status, body: buffer.subarray(0, bytes), present: false })
        })
      })
      outgoing.once('error', () => finish(new OwnerErasureProviderError('TRANSPORT')))
      outgoing.end()
    }).catch(error => finish(error instanceof OwnerErasureProviderError ? error : new OwnerErasureProviderError('TRANSPORT')))
  })
}
function createOwnerErasureProvider({ environmentId, bucket, region, getCredentials = () => { fail('CREDENTIALS') }, request = https.request, clock = Date.now } = {}) {
  if (!text(environmentId, 128) || !/^[A-Za-z0-9][A-Za-z0-9_-]*$/u.test(environmentId)
      || !text(bucket, 128) || !/^[a-z0-9][a-z0-9-]{1,62}-[0-9]+$/u.test(bucket)
      || !text(region, 64) || !/^[a-z]{2,8}-[a-z]+(?:-[a-z]+)?[0-9]?$/u.test(region)
      || typeof getCredentials !== 'function' || typeof request !== 'function' || typeof clock !== 'function') fail('INVALID_CONFIG')
  const host = `${bucket}.cos.${region}.myqcloud.com`
  return {
    async step({ target, checkpoint = null, beforeRequest } = {}) {
      if (!validTarget(target, environmentId, bucket)) fail('INVALID_TARGET')
      const targetHash = digest(target); const key = target.cloud_path
      if (checkpoint !== null && !validCheckpoint(checkpoint, targetHash, key)) fail('INVALID_CHECKPOINT')
      const cp = checkpoint === null ? fresh(targetHash) : JSON.parse(canonicalJson(checkpoint))
      const observed = clock()
      if (!Number.isSafeInteger(observed) || observed < 0 || observed > 8640000000000000) fail('INVALID_CONFIG')
      const evidence = code => ({ target_sha256: targetHash, versioning_mode: cp.mode, observed_at: new Date(observed).toISOString(), transcript_sha256: cp.transcript_sha256, ...(code ? { code } : {}) })
      const pending = code => ({ status: 'pending', checkpoint: cp, evidence: evidence(code) })
      const phase = cp.phase
      const query = {}; let method = 'GET'; let path = `/${key}`; let probe = false
      if (['versioning', 'recheck'].includes(phase)) { path = '/'; query.versioning = '' }
      else if (phase === 'list') {
        path = '/'; Object.assign(query, { versions: '', prefix: key, 'max-keys': '16' })
        if (cp.cursor) Object.assign(query, { 'key-marker': cp.cursor.key, 'version-id-marker': cp.cursor.version })
      } else if (['simple_delete', 'delete'].includes(phase)) { method = 'DELETE'; if (phase === 'delete') query.versionId = cp.versions[0] }
      else probe = true
      const started = Date.now()
      try {
        const response = await exchange({ request, getCredentials, beforeRequest, host, method, path, query, probe, observed })
        const next = JSON.parse(canonicalJson(cp)); let done = false
        if (phase === 'versioning' || phase === 'recheck') {
          if (response.status !== 200) fail('NATIVE')
          const mode = versioning(response.body)
          if (phase === 'recheck' && mode === cp.mode) done = true
          else {
            if (cp.mode && cp.mode !== 'NeverEnabled' && mode === 'NeverEnabled') fail('NATIVE')
            next.mode = mode; next.phase = mode === 'NeverEnabled' ? 'simple_delete' : 'list'
            next.cursor = null; next.cursor_sha256 = []; next.pages = 0; next.versions = []
          }
        } else if (phase === 'simple_delete' || phase === 'delete') {
          if (![200, 204].includes(response.status) || response.body.length !== 0) fail('NATIVE')
          if (phase === 'simple_delete') next.phase = 'simple_probe'
          else {
            next.versions.shift()
            if (!next.versions.length) { next.phase = 'list'; next.cursor = null; next.cursor_sha256 = []; next.pages = 0 }
          }
        } else if (phase === 'list') {
          if (response.status !== 200) fail('NATIVE')
          const page = versions(response.body, cp, bucket, key)
          if (page.ids.length) { next.phase = 'delete'; next.versions = page.ids }
          else if (page.cursor) {
            // Only the recent-loop witness rolls; the exact continuation cursor is
            // retained. There is no page cap that freezes valid larger histories.
            next.cursor = page.cursor; next.cursor_sha256.push(digest(page.cursor))
            if (next.cursor_sha256.length > CURSOR_WINDOW) next.cursor_sha256.shift()
            next.pages = next.cursor_sha256.length
          } else { next.phase = 'version_probe'; next.cursor = null; next.cursor_sha256 = []; next.pages = 0 }
        } else {
          if (response.present) fail('PRESENT')
          if (response.status !== 404) fail('NATIVE')
          absent(response.body, key); next.phase = 'recheck'
        }
        if (Date.now() - started >= DEADLINE_MS) fail('TIMEOUT')
        next.transcript_sha256 = digest({ previous: cp.transcript_sha256, phase, status: response.status, body_sha256: digest(response.body.toString('base64')) })
        Object.assign(cp, next)
        return { status: done ? 'absent' : 'pending', checkpoint: cp, evidence: evidence() }
      } catch (error) { return pending(error instanceof OwnerErasureProviderError ? error.code : 'OWNER_ERASURE_TRANSPORT') }
    },
  }
}
module.exports = { createOwnerErasureProvider, OwnerErasureProviderError }
