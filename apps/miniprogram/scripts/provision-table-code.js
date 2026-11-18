const crypto = require('crypto')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { spawnSync } = require('child_process')

function defaultRun(command, args, options = {}) {
  return spawnSync(command, args, { encoding: 'utf8', cwd: options.cwd })
}

function defaultTemporaryDirectory() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'billiardsAI-table-code-'))
}

function safeSegment(value, label) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(value)) {
    throw new Error(`${label} is invalid`)
  }
  return value
}

function ownerOnlyDirectory(directory) {
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 })
  fs.chmodSync(directory, 0o700)
}

function writeOwnerOnly(file, content) {
  ownerOnlyDirectory(path.dirname(file))
  fs.writeFileSync(file, content, { mode: 0o600 })
  fs.chmodSync(file, 0o600)
}

function commandArgs(environmentId, configFile, action, functionName, extra = []) {
  return [
    '--env-id', environmentId,
    '--config-file', configFile,
    '--yes',
    'fn', action, functionName,
    ...extra,
  ]
}

function deletionSucceeded(result) {
  if (result && result.status === 0) return true
  try {
    const envelope = JSON.parse(String(result && result.stdout || '').trim())
    const code = envelope && envelope.error && envelope.error.code
    return code === 'ResourceNotFound.FunctionName'
      || code === 'ResourceNotFound.Function'
      || code === 'FUNCTION_NOT_FOUND'
  } catch (_error) {
    return false
  }
}

function sanitizeDiagnostic(value, secrets = []) {
  let sanitized = String(value || '')
  for (const secret of secrets) {
    if (secret) sanitized = sanitized.split(String(secret)).join('REDACTED')
  }
  return sanitized.replace(/[A-Za-z0-9_-]{32,}/gu, 'REDACTED')
}

function commandFailure(result, includeStdout = false, secrets = []) {
  const stdout = String(result && result.stdout || '').trim()
  let cloudbaseMessage = ''
  const lines = stdout.split('\n').filter(Boolean)
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const payload = JSON.parse(lines[index])
      cloudbaseMessage = payload && payload.error && payload.error.message || ''
      if (cloudbaseMessage) break
    } catch (_error) {
      // CloudBase may print progress before its final JSON error envelope.
    }
  }
  const unstructuredStdout = includeStdout && !cloudbaseMessage ? stdout : ''
  return sanitizeDiagnostic(
    [result && result.error && result.error.message, cloudbaseMessage, unstructuredStdout, result && result.stderr]
      .filter(Boolean).join('\n').trim(),
    secrets,
  )
}

function parseInvocation(stdout, expectedStatus = 'REGISTERED') {
  const text = String(stdout || '').trim()
  const lines = text.split('\n').filter(Boolean)
  let envelope
  const firstBrace = text.indexOf('{')
  const lastBrace = text.lastIndexOf('}')
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    try {
      envelope = JSON.parse(text.slice(firstBrace, lastBrace + 1))
    } catch (_error) {
      // Fall back to single-line envelopes used by older CLI versions.
    }
  }
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    if (envelope) break
    try {
      envelope = JSON.parse(lines[index])
      break
    } catch (_error) {
      // CloudBase may print progress before the final JSON envelope.
    }
  }
  if (!envelope) throw new Error('TABLE_CODE_PROVISION_RESPONSE_INVALID')
  const rawPayload = envelope.result !== undefined
    ? envelope.result
    : (envelope.data && envelope.data.RetMsg !== undefined ? envelope.data.RetMsg : envelope)
  const payload = typeof rawPayload === 'string' ? JSON.parse(rawPayload) : rawPayload
  if (!payload || payload.status !== expectedStatus) {
    const rawDetail = payload && (
      payload.errorMessage || payload.error_message || payload.errMsg || payload.message || payload.errorCode
    ) || (typeof payload === 'string' ? payload : JSON.stringify(payload || {}))
    const detail = String(rawDetail)
      .replace(/[A-Za-z0-9_-]{32,}/gu, 'REDACTED')
      .replace(/[^A-Za-z0-9 ._:/-]/gu, '')
      .slice(0, 240)
    throw new Error(`TABLE_CODE_PROVISION_RESPONSE_INVALID${detail ? `: ${detail}` : ''}`)
  }
  return payload
}

function validatedImage(payload) {
  const image = Buffer.from(payload.image_base64 || '', 'base64')
  const signature = image.subarray(0, 8)
  const png = signature.equals(Buffer.from('\x89PNG\r\n\x1a\n', 'binary'))
  const jpeg = image[0] === 0xff && image[1] === 0xd8
  const actual = crypto.createHash('sha256').update(image).digest('hex')
  if ((!png && !jpeg) || actual !== payload.image_sha256) {
    throw new Error('TABLE_CODE_IMAGE_INVALID')
  }
  return image
}

function loadReusableState(options) {
  if (options.rotate || fs.existsSync(`${options.statePath}.invalid`) || !fs.existsSync(options.statePath)) return null
  const state = JSON.parse(fs.readFileSync(options.statePath, 'utf8'))
  const expectedCodeKind = options.permanent ? 'permanent' : 'development_preview'
  if (state.code_kind !== expectedCodeKind
      || state.environment_id !== options.environmentId
      || state.device_id !== options.deviceId) {
    return null
  }
  if (!/^[A-Za-z0-9_-]{32}$/u.test(state.table_code || '')
      || typeof state.image_path !== 'string' || !fs.existsSync(state.image_path)) {
    throw new Error('TABLE_CODE_LOCAL_STATE_INVALID')
  }
  const image = fs.readFileSync(state.image_path)
  validatedImage({
    image_base64: image.toString('base64'),
    image_sha256: crypto.createHash('sha256').update(image).digest('hex'),
  })
  writeOwnerOnly(options.outputPath, image)
  return { image, tableCode: state.table_code, reused: true }
}

function generateDevelopmentPreview(options, dependencies = {}) {
  const run = dependencies.run || defaultRun
  const projectRoot = path.resolve(options.projectRoot)
  const privateConfigPath = path.join(projectRoot, 'project.private.config.json')
  const projectConfigPath = path.join(projectRoot, 'project.config.json')
  const previewEntryPath = path.join(projectRoot, 'miniprogram', 'config', 'preview-entry.local.js')
  const existed = fs.existsSync(privateConfigPath)
  const original = existed ? fs.readFileSync(privateConfigPath) : null
  const originalProject = fs.readFileSync(projectConfigPath)
  const originalPreviewEntry = fs.existsSync(previewEntryPath) ? fs.readFileSync(previewEntryPath) : null
  const config = existed ? JSON.parse(original.toString('utf8')) : {}
  const projectConfig = JSON.parse(originalProject.toString('utf8'))
  if (typeof config.appid !== 'string' || !/^wx[A-Za-z0-9]+$/u.test(config.appid)) {
    throw new Error('MINIPROGRAM_APP_ID_INVALID')
  }
  const condition = {
    miniprogram: {
      list: [{
        id: -1,
        name: 'development-table-join',
        pathName: 'pages/join/index',
        query: `table_code=${options.tableCode}`,
        scene: null,
      }],
    },
  }
  config.condition = condition
  projectConfig.condition = condition
  writeOwnerOnly(privateConfigPath, JSON.stringify(config, null, 2))
  fs.writeFileSync(projectConfigPath, JSON.stringify(projectConfig, null, 2))
  writeOwnerOnly(previewEntryPath, `module.exports = { tableCode: '${options.tableCode}' }\n`)
  ownerOnlyDirectory(path.dirname(path.resolve(options.outputPath)))
  try {
    run(options.cliPath, ['close', '--project', projectRoot], { cwd: projectRoot })
    const result = run(options.cliPath, [
      'preview',
      '--project', projectRoot,
      '--qr-format', 'image',
      '--qr-output', path.resolve(options.outputPath),
    ], { cwd: projectRoot })
    if (!result || result.status !== 0) {
      throw new Error(`TABLE_CODE_PREVIEW_FAILED: ${commandFailure(result, true)}`)
    }
    return validatedImage({
      image_base64: fs.readFileSync(options.outputPath).toString('base64'),
      image_sha256: crypto.createHash('sha256').update(fs.readFileSync(options.outputPath)).digest('hex'),
    })
  } finally {
    fs.writeFileSync(projectConfigPath, originalProject)
    if (originalPreviewEntry) writeOwnerOnly(previewEntryPath, originalPreviewEntry)
    else fs.rmSync(previewEntryPath, { force: true })
    if (original) writeOwnerOnly(privateConfigPath, original)
    else fs.rmSync(privateConfigPath, { force: true })
  }
}

function refreshDevelopmentPreview(options, dependencies = {}) {
  const generatePreview = dependencies.generatePreview || generateDevelopmentPreview
  const temporaryDirectory = dependencies.temporaryDirectory || defaultTemporaryDirectory
  const environmentId = safeSegment(options.environmentId, 'environment ID')
  const deviceId = safeSegment(options.deviceId, 'device ID')
  if (fs.existsSync(`${options.statePath}.invalid`)
      || !fs.existsSync(options.statePath)
      || fs.lstatSync(options.statePath).isSymbolicLink()) {
    throw new Error('TABLE_CODE_LOCAL_STATE_INVALID')
  }
  const state = JSON.parse(fs.readFileSync(options.statePath, 'utf8'))
  if (state.code_kind !== 'development_preview'
      || state.environment_id !== environmentId
      || state.device_id !== deviceId
      || !/^[A-Za-z0-9_-]{32}$/u.test(state.table_code || '')) {
    throw new Error('TABLE_CODE_LOCAL_STATE_INVALID')
  }
  const temporaryRoot = path.resolve(temporaryDirectory())
  const previewPath = path.join(temporaryRoot, 'table-join-preview.png')
  let image
  try {
    ownerOnlyDirectory(temporaryRoot)
    image = generatePreview({
      projectRoot: options.projectRoot,
      outputPath: previewPath,
      tableCode: state.table_code,
      cliPath: options.cliPath || '/Applications/wechatwebdevtools.app/Contents/MacOS/cli',
    })
    validatedImage({
      image_base64: image.toString('base64'),
      image_sha256: crypto.createHash('sha256').update(image).digest('hex'),
    })
  } catch (error) {
    throw new Error(`TABLE_CODE_PREVIEW_FAILED: ${sanitizeDiagnostic(error && error.message, [state.table_code])}`)
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true })
  }
  writeOwnerOnly(options.outputPath, image)
  writeOwnerOnly(options.statePath, JSON.stringify({
    schema_version: '1.0',
    code_kind: 'development_preview',
    environment_id: environmentId,
    device_id: deviceId,
    table_code: state.table_code,
    image_path: path.resolve(options.outputPath),
    updated_at: new Date().toISOString(),
  }))
  return { image, tableCode: state.table_code, reused: true }
}

async function provisionTableCode(options, dependencies = {}) {
  const run = dependencies.run || defaultRun
  const generatePreview = dependencies.generatePreview || generateDevelopmentPreview
  const randomBytes = dependencies.randomBytes || crypto.randomBytes
  const temporaryDirectory = dependencies.temporaryDirectory || defaultTemporaryDirectory
  const environmentId = safeSegment(options.environmentId, 'environment ID')
  const deviceId = safeSegment(options.deviceId, 'device ID')
  const reusable = loadReusableState({ ...options, environmentId, deviceId })
  if (reusable) return reusable

  const tableCode = randomBytes(24).toString('base64url')
  const authorization = randomBytes(32).toString('hex')
  const authorizationDigest = crypto.createHash('sha256').update(authorization).digest('hex')
  const functionName = `billiards-provision-${randomBytes(8).toString('hex')}`
  const temporaryRoot = path.resolve(temporaryDirectory())
  const functionRoot = path.join(temporaryRoot, 'functions', functionName)
  const configFile = path.join(temporaryRoot, 'cloudbaserc.json')
  const eventFile = path.join(temporaryRoot, 'event.json')
  const previewPath = path.join(temporaryRoot, 'table-join-preview.png')
  const stateInvalidationPath = `${options.statePath}.invalid`
  const pendingInvalidation = fs.existsSync(stateInvalidationPath)
    ? JSON.parse(fs.readFileSync(stateInvalidationPath, 'utf8'))
    : null
  let deployAttempted = false
  let operationError = null
  let image = null
  try {
    ownerOnlyDirectory(temporaryRoot)
    if (pendingInvalidation && pendingInvalidation.cleanup_function_name) {
      if (pendingInvalidation.environment_id !== environmentId
          || pendingInvalidation.device_id !== deviceId) {
        throw new Error('TABLE_CODE_PENDING_CLEANUP_SCOPE_INVALID')
      }
      const pendingFunctionName = safeSegment(
        pendingInvalidation.cleanup_function_name,
        'pending cleanup function name',
      )
      writeOwnerOnly(configFile, JSON.stringify({
        version: '2.0',
        envId: environmentId,
        functionRoot: './functions',
        functions: [{ name: pendingFunctionName, handler: 'index.main' }],
      }))
      const pendingDeletion = run(
        'tcb',
        commandArgs(environmentId, configFile, 'delete', pendingFunctionName, ['--json']),
        { cwd: temporaryRoot },
      )
      if (!deletionSucceeded(pendingDeletion)) {
        throw new Error(
          `TABLE_CODE_CLEANUP_FAILED: ${commandFailure(pendingDeletion, true, [tableCode, authorization])}`,
        )
      }
      writeOwnerOnly(stateInvalidationPath, JSON.stringify({
        schema_version: '1.0',
        environment_id: environmentId,
        device_id: deviceId,
        reason: 'cloud_registration_requires_replacement',
        invalidated_at: new Date().toISOString(),
      }))
    }
    ownerOnlyDirectory(functionRoot)
    fs.copyFileSync(path.resolve(__dirname, '../provisioning/table-code/index.js'), path.join(functionRoot, 'handler.js'))
    fs.copyFileSync(path.resolve(__dirname, '../provisioning/table-code/package.json'), path.join(functionRoot, 'package.json'))
    const handlerName = options.permanent ? 'createProvisioningHandler' : 'createRegistrationHandler'
    const exportedHandler = options.permanent
      ? "exports.main = async (event) => { try { return await handler(event) } catch (error) { return { schema_version: '1.0', status: 'FAILED', errorCode: Number(error && (error.errCode || error.code)) || null, errorMessage: String(error && (error.errMsg || error.message) || 'WXACODE_FAILED') } } }"
      : 'exports.main = async (event) => handler(event)'
    const wrapper = [
      "const cloud = require('wx-server-sdk')",
      `const { ${handlerName} } = require('./handler')`,
      'cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })',
      `const handler = ${handlerName}({ cloud, expectedAuthDigest: '${authorizationDigest}', now: () => new Date().toISOString() })`,
      exportedHandler,
      '',
    ].join('\n')
    writeOwnerOnly(path.join(functionRoot, 'index.js'), wrapper)
    writeOwnerOnly(configFile, JSON.stringify({
      version: '2.0',
      envId: environmentId,
      functionRoot: './functions',
      functions: [{
        name: functionName,
        handler: 'index.main',
        runtime: 'Nodejs20.19',
        timeout: 20,
        memorySize: 256,
        installDependency: true,
      }],
    }))
    writeOwnerOnly(eventFile, JSON.stringify({
      authorization,
      table_code: tableCode,
      table_id: deviceId,
    }))

    if (!options.permanent) {
      try {
        image = generatePreview({
          projectRoot: options.projectRoot,
          outputPath: previewPath,
          tableCode,
          cliPath: options.cliPath || '/Applications/wechatwebdevtools.app/Contents/MacOS/cli',
        })
      } catch (error) {
        throw new Error(`TABLE_CODE_PREVIEW_FAILED: ${sanitizeDiagnostic(error && error.message, [tableCode, authorization])}`)
      }
      validatedImage({
        image_base64: image.toString('base64'),
        image_sha256: crypto.createHash('sha256').update(image).digest('hex'),
      })
    }

    writeOwnerOnly(stateInvalidationPath, JSON.stringify({
      schema_version: '1.0',
      environment_id: environmentId,
      device_id: deviceId,
      cleanup_function_name: functionName,
      reason: 'cloud_registration_in_progress',
      invalidated_at: new Date().toISOString(),
    }))
    deployAttempted = true
    const deploy = run('tcb', commandArgs(environmentId, configFile, 'deploy', functionName), { cwd: temporaryRoot })
    if (!deploy || deploy.status !== 0) {
      throw new Error(`TABLE_CODE_PROVISION_FAILED: ${commandFailure(deploy, true, [tableCode, authorization])}`)
    }
    const invoke = run('tcb', commandArgs(
      environmentId,
      configFile,
      'invoke',
      functionName,
      ['-d', `@${eventFile}`, '--json'],
    ), { cwd: temporaryRoot })
    if (!invoke || invoke.status !== 0) {
      throw new Error(`TABLE_CODE_PROVISION_FAILED: ${commandFailure(invoke, false, [tableCode, authorization])}`)
    }
    const payload = parseInvocation(invoke.stdout, options.permanent ? 'PROVISIONED' : 'REGISTERED')
    if (options.permanent) image = validatedImage(payload)
  } catch (error) {
    operationError = new Error(sanitizeDiagnostic(error && error.message, [tableCode, authorization]))
    throw operationError
  } finally {
    let cleanupError = null
    if (deployAttempted) {
      try {
        const deletion = run(
          'tcb',
          commandArgs(environmentId, configFile, 'delete', functionName, ['--json']),
          { cwd: temporaryRoot },
        )
        if (!deletionSucceeded(deletion)) {
          cleanupError = new Error(
            `TABLE_CODE_CLEANUP_FAILED: ${commandFailure(deletion, true, [tableCode, authorization])}`,
          )
        } else if (fs.existsSync(stateInvalidationPath)) {
          writeOwnerOnly(stateInvalidationPath, JSON.stringify({
            schema_version: '1.0',
            environment_id: environmentId,
            device_id: deviceId,
            reason: 'cloud_registration_requires_local_commit',
            invalidated_at: new Date().toISOString(),
          }))
        }
      } catch (error) {
        cleanupError = new Error(
          `TABLE_CODE_CLEANUP_FAILED: ${sanitizeDiagnostic(error && error.message, [tableCode, authorization])}`,
        )
      }
    }
    fs.rmSync(temporaryRoot, { recursive: true, force: true })
    if (cleanupError) {
      if (operationError) operationError.message = `${operationError.message}\n${cleanupError.message}`
      else throw cleanupError
    }
  }
  writeOwnerOnly(options.outputPath, image)
  writeOwnerOnly(options.statePath, JSON.stringify({
    schema_version: '1.0',
    code_kind: options.permanent ? 'permanent' : 'development_preview',
    environment_id: environmentId,
    device_id: deviceId,
    table_code: tableCode,
    image_path: path.resolve(options.outputPath),
    updated_at: new Date().toISOString(),
  }))
  fs.rmSync(stateInvalidationPath, { force: true })
  return { image, tableCode, reused: false }
}

function argumentValue(argv, name) {
  const index = argv.indexOf(name)
  return index >= 0 ? argv[index + 1] : undefined
}

async function main(argv, dependencies = {}) {
  if (argumentValue(argv, '--environment') !== 'development') {
    throw new Error('table-code provisioning requires --environment development')
  }
  const outputPath = argumentValue(argv, '--output')
  const statePath = argumentValue(argv, '--state')
  if (!outputPath || !statePath) throw new Error('table-code output and state paths are required')
  const provision = dependencies.provision || provisionTableCode
  const permanent = argv.includes('--permanent-table-code')
  const refresh = argv.includes('--refresh-development-preview')
  if (refresh && (permanent || argv.includes('--rotate-table-code'))) {
    throw new Error('preview refresh cannot rotate or provision a permanent table code')
  }
  const options = {
    environmentId: argumentValue(argv, '--environment-id'),
    deviceId: argumentValue(argv, '--device-id'),
    outputPath,
    statePath,
    projectRoot: path.resolve(__dirname, '..'),
    rotate: argv.includes('--rotate-table-code'),
    permanent,
  }
  const result = refresh
    ? await (dependencies.refreshPreview || refreshDevelopmentPreview)(options)
    : await provision(options)
  const payload = {
    status: 'PASS',
    image_path: path.resolve(outputPath),
    image_sha256: crypto.createHash('sha256').update(result.image).digest('hex'),
    code_kind: permanent ? 'permanent' : 'development_preview',
    reused: Boolean(result.reused),
  }
  const write = dependencies.write || ((value) => process.stdout.write(value))
  write(`${JSON.stringify(payload)}\n`)
  return payload
}

if (require.main === module) {
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  })
}

module.exports = {
  generateDevelopmentPreview,
  main,
  provisionTableCode,
  refreshDevelopmentPreview,
}
