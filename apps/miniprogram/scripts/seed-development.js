const fs = require('fs')
const path = require('path')


function main(argv) {
  const environmentIndex = argv.indexOf('--environment')
  const outputIndex = argv.indexOf('--output')
  const environment = environmentIndex >= 0 ? argv[environmentIndex + 1] : ''
  const output = outputIndex >= 0 ? argv[outputIndex + 1] : ''
  if (environment !== 'development') {
    throw new Error('seed refuses every environment except explicit development')
  }
  if (!output) throw new Error('seed output path is required')
  const payload = {
    schema_version: '1.0',
    environment: 'development',
    synthetic_only: true,
    users: [
      { user_id: 'user-synthetic-red', nickname: 'Synthetic Red' },
      { user_id: 'user-synthetic-blue', nickname: 'Synthetic Blue' },
    ],
    media_labels: [{ source_kind: 'development_replay', real_user_data: false }],
  }
  const target = path.resolve(output)
  fs.mkdirSync(path.dirname(target), { recursive: true })
  fs.writeFileSync(target, `${JSON.stringify(payload, null, 2)}\n`, { flag: 'wx' })
  process.stdout.write(`${target}\n`)
}


if (require.main === module) {
  try {
    main(process.argv.slice(2))
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  }
}


module.exports = { main }
