const { observeStorageGrant } = require('../../scripts/lib/storage-grant-observation')
const mode = process.argv[2]
let text = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', value => { text += value })
process.stdin.on('end', () => {
  const input = JSON.parse(text)
  if (mode === 'hang') { process.on('SIGTERM', () => {}); setTimeout(() => {}, 60000); return }
  process.stdout.write(input.credentials.secretId + '\n' + input.credentials.token + '\nCANARY_SDK_STDOUT\n')
  process.stderr.write(input.credentials.secretKey + '\nCANARY_SDK_STDERR\n')
  if (mode === 'crash') { process.stderr.write('CANARY_CRASH'); process.exitCode = 1; return }
  if (mode === 'malformed') { process.stdout.write('BILLIARDS_NATIVE_STORAGE_REPORT:{CANARY_BAD_JSON\n'); return }
  if (mode === 'error_unknown') { process.stdout.write('BILLIARDS_NATIVE_STORAGE_ERROR:CANARY_UNKNOWN\n'); return }
  if (mode === 'aggregate') { process.stdout.write('A'.repeat(600000) + '\n'); process.stderr.write('B'.repeat(600000)); }
  if (mode === 'oversize') { process.stdout.write('X'.repeat(1048577)); return }
  if (mode === 'error') { process.stdout.write('BILLIARDS_NATIVE_STORAGE_ERROR:sdk_unavailable\n'); process.exitCode = 2; return }
  const report = observeStorageGrant({ ...input, metadata: { data: { url: 'https://cos.ap-test.myqcloud.com', token: 'CANARY_GRANT', authorization: 'CANARY_AUTH', fileId: 'cloud://CANARY_FILE', cosFileId: 'CANARY_COS' } } })
  if (mode === 'unknown') report.extra = 'CANARY'
  if (mode === 'authority') report.erasable = true
  if (mode === 'context') report.observed_at = '2000-01-01T00:00:00.000Z'
  const line = 'BILLIARDS_NATIVE_STORAGE_REPORT:' + JSON.stringify(report) + '\n'
  process.stdout.write(line)
  if (mode === 'mixed') process.stdout.write('BILLIARDS_NATIVE_STORAGE_ERROR:sdk_unavailable\n')
  if (mode === 'nonzero') process.exitCode = 1
  if (mode === 'duplicate') process.stdout.write(line)
})
