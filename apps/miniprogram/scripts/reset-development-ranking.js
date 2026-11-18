const { spawnSync } = require('child_process')


function value(argv, name) {
  const index = argv.indexOf(name)
  return index >= 0 ? argv[index + 1] : undefined
}


function parseResetArguments(argv, environment = process.env) {
  const environmentId = value(argv, '--environment-id')
  const confirmedScope = value(argv, '--confirm-scope')
  const resetId = value(argv, '--reset-id')
  const token = environment.RANKING_WORKER_TOKEN
  if (confirmedScope !== 'zhoukou-urban-dev') {
    throw new Error('RANKING_TEST_RESET_FORBIDDEN')
  }
  if (!environmentId || !/^[a-z0-9][a-z0-9-]{2,127}$/iu.test(environmentId)
      || !resetId || !/^[A-Za-z0-9._-]{1,128}$/u.test(resetId) || !token) {
    throw new Error('RANKING_TEST_CONFIGURATION_INVALID')
  }
  return { environmentId, resetId, token }
}


function main(argv, environment = process.env, run = spawnSync) {
  const { environmentId, resetId, token } = parseResetArguments(argv, environment)
  const payload = {
    schema_version: '1.0', action: 'development.reset', token, reset_id: resetId,
  }
  const result = run('tcb', [
    '-e', environmentId, 'fn', 'invoke', 'ranking-worker',
    '--params', JSON.stringify(payload), '--json',
  ], { encoding: 'utf8' })
  if (!result || result.status !== 0) {
    const detail = [result && result.stderr, result && result.stdout].filter(Boolean).join('\n')
    throw new Error(`RANKING_TEST_RESET_FAILED: ${detail.trim() || 'unknown tcb error'}`)
  }
  process.stdout.write('PASS: development ranking reset completed\n')
}


if (require.main === module) {
  try {
    main(process.argv.slice(2))
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  }
}


module.exports = { main, parseResetArguments }
