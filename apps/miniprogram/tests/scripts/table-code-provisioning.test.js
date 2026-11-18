const crypto = require('crypto')
const fs = require('fs')
const os = require('os')
const path = require('path')

const {
  generateDevelopmentPreview,
  main,
  provisionTableCode,
  refreshDevelopmentPreview,
} = require('../../scripts/provision-table-code')

const png = Buffer.from('\x89PNG\r\n\x1a\npermanent-code', 'binary')
const tableCode = 'AbCdEfGhIjKlMnOpQrStUvWxYz012345'

function deterministicRandom(length) {
  return Buffer.alloc(length, length)
}

function fixtureRun({ failDeploy = false, failInvoke = false, failDelete = false } = {}) {
  const calls = []
  const cwds = []
  const run = (command, args, options) => {
    calls.push([command, ...args])
    cwds.push(options && options.cwd)
    const action = args[args.indexOf('fn') + 1]
    if (action === 'deploy' && failDeploy) {
      return { status: 1, stdout: '', stderr: 'deploy uncertain' }
    }
    if (action === 'invoke') {
      if (failInvoke) return { status: 1, stdout: '', stderr: 'invoke failed' }
      const payload = {
        schema_version: '1.0',
        status: 'REGISTERED',
      }
      return {
        status: 0,
        stdout: JSON.stringify({ data: { InvokeResult: 0, RetMsg: JSON.stringify(payload), ErrMsg: '' } }, null, 2),
        stderr: '',
      }
    }
    if (action === 'delete' && failDelete) {
      return { status: 1, stdout: '', stderr: 'delete refused' }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }
  return { calls, cwds, run }
}

function fixturePreview() {
  const calls = []
  const generate = ({ outputPath, tableCode }) => {
    calls.push({ outputPath, tableCode })
    fs.mkdirSync(path.dirname(outputPath), { recursive: true })
    fs.writeFileSync(outputPath, png)
    return png
  }
  return { calls, generate }
}

test('deployment failure reports the CloudBase error emitted on stdout', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-stdout-error-'))
  const run = (_command, args) => {
    const action = args[args.indexOf('fn') + 1]
    if (action === 'deploy') {
      return {
        status: 1,
        stdout: 'function directory not found',
        stderr: '- Loading data...',
      }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: path.join(temporaryRoot, 'table-join.png'),
    statePath: path.join(temporaryRoot, 'state.json'),
    rotate: true,
  }, {
    run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('function directory not found')
})

test('orchestrator deploys invokes and always deletes a random temporary function', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-test-'))
  const output = path.join(temporaryRoot, 'output', 'table-join.png')
  const state = path.join(temporaryRoot, 'state', 'raytheon-01.json')
  const fixture = fixtureRun()
  const preview = fixturePreview()

  const result = await provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: output,
    statePath: state,
    rotate: true,
  }, {
    run: fixture.run,
    generatePreview: preview.generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })

  expect(result.image).toEqual(png)
  expect(result.tableCode).toMatch(/^[A-Za-z0-9_-]{32}$/)
  expect(fs.readFileSync(output)).toEqual(png)
  expect(fs.existsSync(path.join(temporaryRoot, 'ephemeral'))).toBe(false)
  expect(fixture.calls.map((call) => call[call.indexOf('fn') + 1]))
    .toEqual(['deploy', 'invoke', 'delete'])
  expect(fixture.calls.every((call) => call.includes('--env-id'))).toBe(true)
  expect(fixture.calls.every((call) => call.includes('--config-file'))).toBe(true)
  expect(fixture.cwds).toEqual([
    path.join(temporaryRoot, 'ephemeral'),
    path.join(temporaryRoot, 'ephemeral'),
    path.join(temporaryRoot, 'ephemeral'),
  ])
  expect(preview.calls).toHaveLength(1)
  const persisted = JSON.parse(fs.readFileSync(state, 'utf8'))
  expect(persisted.table_code).toBe(result.tableCode)
  expect(persisted.code_kind).toBe('development_preview')
  expect(JSON.stringify(fixture.calls)).not.toContain(result.tableCode)
})

test('permanent mode obtains the QR image from wxacode without using a preview short link', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-permanent-'))
  const output = path.join(temporaryRoot, 'output', 'table-join.png')
  const state = path.join(temporaryRoot, 'state', 'raytheon-01.json')
  const calls = []
  let deployedWrapper = ''
  const run = (_command, args, options) => {
    const action = args[args.indexOf('fn') + 1]
    calls.push(action)
    if (action === 'deploy') {
      const functionName = args[args.indexOf('fn') + 2]
      deployedWrapper = fs.readFileSync(path.join(options.cwd, 'functions', functionName, 'index.js'), 'utf8')
    }
    if (action === 'invoke') {
      const payload = {
        schema_version: '1.0', status: 'PROVISIONED',
        image_base64: png.toString('base64'),
        image_sha256: crypto.createHash('sha256').update(png).digest('hex'),
      }
      return { status: 0, stdout: JSON.stringify({ result: JSON.stringify(payload) }), stderr: '' }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  const result = await provisionTableCode({
    environmentId: 'cloud1-test', deviceId: 'raytheon-01',
    outputPath: output, statePath: state, rotate: true, permanent: true,
  }, {
    run,
    generatePreview: () => { throw new Error('preview must not run') },
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })

  expect(result.image).toEqual(png)
  expect(calls).toEqual(['deploy', 'invoke', 'delete'])
  expect(deployedWrapper).toContain('createProvisioningHandler')
  expect(deployedWrapper).not.toContain('createRegistrationHandler')
  expect(deployedWrapper).toContain("status: 'FAILED'")
  expect(deployedWrapper).toContain('errorCode')
  expect(JSON.parse(fs.readFileSync(state, 'utf8')).code_kind).toBe('permanent')
})

test('CLI forwards the explicit permanent table-code mode', async () => {
  const provision = jest.fn(async () => ({ image: png, tableCode, reused: false }))

  const result = await main([
    '--environment', 'development', '--environment-id', 'cloud1-test',
    '--device-id', 'raytheon-01', '--output', '/private/tmp/table-code.png',
    '--state', '/private/tmp/table-code.json', '--permanent-table-code',
  ], { provision, write: () => {} })

  expect(provision).toHaveBeenCalledWith(expect.objectContaining({ permanent: true }))
  expect(result.code_kind).toBe('permanent')
  expect(result.image_sha256).toMatch(/^[0-9a-f]{64}$/u)
})

test('development preview uses a temporary compile condition without putting the table code in CLI arguments', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-preview-project-'))
  const privateConfig = path.join(root, 'project.private.config.json')
  const projectConfig = path.join(root, 'project.config.json')
  const previewEntry = path.join(root, 'miniprogram', 'config', 'preview-entry.local.js')
  const original = { appid: 'wx0000000000000000', projectname: 'local-project' }
  const originalProject = { appid: 'touristappid', miniprogramRoot: 'miniprogram/' }
  fs.writeFileSync(privateConfig, JSON.stringify(original))
  fs.writeFileSync(projectConfig, JSON.stringify(originalProject))
  fs.mkdirSync(path.dirname(previewEntry), { recursive: true })
  const output = path.join(root, 'private', 'table-join.png')
  const observations = []
  const run = (command, args, options) => {
    observations.push({
      command,
      args,
      cwd: options.cwd,
      config: JSON.parse(fs.readFileSync(privateConfig, 'utf8')),
      projectConfig: JSON.parse(fs.readFileSync(projectConfig, 'utf8')),
      previewEntry: fs.existsSync(previewEntry) ? fs.readFileSync(previewEntry, 'utf8') : '',
    })
    if (args[0] === 'preview') {
      if (!fs.existsSync(path.dirname(output))) throw new Error('preview output directory missing')
      fs.writeFileSync(output, png)
    }
    return { status: 0, stdout: '', stderr: '' }
  }

  const image = generateDevelopmentPreview({
    projectRoot: root,
    outputPath: output,
    tableCode,
    cliPath: '/Applications/wechatwebdevtools.app/Contents/MacOS/cli',
  }, { run })

  expect(image).toEqual(png)
  expect(observations.map((item) => item.args[0])).toEqual(['close', 'preview'])
  expect(JSON.stringify(observations[1].args)).not.toContain(tableCode)
  expect(observations[1].config.condition.miniprogram.list).toEqual([{
    id: -1,
    name: 'development-table-join',
    pathName: 'pages/join/index',
    query: `table_code=${tableCode}`,
    scene: null,
  }])
  expect(observations[1].projectConfig.condition).toEqual(observations[1].config.condition)
  expect(observations[1].previewEntry).toContain(tableCode)
  expect(JSON.parse(fs.readFileSync(privateConfig, 'utf8'))).toEqual(original)
  expect(JSON.parse(fs.readFileSync(projectConfig, 'utf8'))).toEqual(originalProject)
  expect(fs.existsSync(previewEntry)).toBe(false)
})

test('refreshing a development preview preserves the registered table code and skips CloudBase', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-preview-refresh-'))
  const output = path.join(root, 'private', 'table-join.png')
  const state = path.join(root, 'state', 'raytheon-01.json')
  const previousImage = Buffer.from('\x89PNG\r\n\x1a\nexpired-preview', 'binary')
  const freshImage = Buffer.from('\x89PNG\r\n\x1a\nfresh-preview', 'binary')
  fs.mkdirSync(path.dirname(output), { recursive: true })
  fs.mkdirSync(path.dirname(state), { recursive: true })
  fs.writeFileSync(output, previousImage)
  fs.writeFileSync(state, JSON.stringify({
    schema_version: '1.0',
    code_kind: 'development_preview',
    environment_id: 'cloud1-test',
    device_id: 'raytheon-01',
    table_code: tableCode,
    image_path: output,
    updated_at: '2026-08-30T00:00:00.000Z',
  }))
  const previewCalls = []

  const result = refreshDevelopmentPreview({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: output,
    statePath: state,
    projectRoot: root,
    cliPath: '/Applications/wechatwebdevtools.app/Contents/MacOS/cli',
  }, {
    generatePreview: (options) => {
      previewCalls.push(options)
      fs.mkdirSync(path.dirname(options.outputPath), { recursive: true })
      fs.writeFileSync(options.outputPath, freshImage)
      return freshImage
    },
    temporaryDirectory: () => path.join(root, 'ephemeral'),
  })

  expect(result).toEqual({ image: freshImage, tableCode, reused: true })
  expect(previewCalls).toHaveLength(1)
  expect(previewCalls[0].tableCode).toBe(tableCode)
  expect(fs.readFileSync(output)).toEqual(freshImage)
  const persisted = JSON.parse(fs.readFileSync(state, 'utf8'))
  expect(persisted.table_code).toBe(tableCode)
  expect(persisted.code_kind).toBe('development_preview')
  expect(persisted.image_path).toBe(path.resolve(output))
  expect(fs.existsSync(path.join(root, 'ephemeral'))).toBe(false)
})

test('CLI can refresh a development preview without rotating its registered table code', async () => {
  const refreshPreview = jest.fn(() => ({ image: png, tableCode, reused: true }))

  const result = await main([
    '--environment', 'development', '--environment-id', 'cloud1-test',
    '--device-id', 'raytheon-01', '--output', '/private/tmp/table-code.png',
    '--state', '/private/tmp/table-code.json', '--refresh-development-preview',
  ], { refreshPreview, write: () => {} })

  expect(refreshPreview).toHaveBeenCalledWith(expect.objectContaining({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
  }))
  expect(result.code_kind).toBe('development_preview')
  expect(result.reused).toBe(true)
})

test('orchestrator deletes the temporary cloud function after invoke failure', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-failure-'))
  const fixture = fixtureRun({ failInvoke: true })
  const outputPath = path.join(temporaryRoot, 'table-join.png')
  const statePath = path.join(temporaryRoot, 'state.json')
  fs.writeFileSync(outputPath, png)
  fs.writeFileSync(statePath, JSON.stringify({
    schema_version: '1.0',
    code_kind: 'development_preview',
    environment_id: 'cloud1-test',
    device_id: 'raytheon-01',
    table_code: tableCode,
    image_path: outputPath,
  }))

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath,
    statePath,
    rotate: true,
  }, {
    run: fixture.run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('TABLE_CODE_PROVISION_FAILED')

  expect(fixture.calls.map((call) => call[call.indexOf('fn') + 1]))
    .toEqual(['deploy', 'invoke', 'delete'])
  expect(fs.readFileSync(outputPath)).toEqual(png)
  expect(fs.existsSync(`${statePath}.invalid`)).toBe(true)
  expect(fs.existsSync(path.join(temporaryRoot, 'ephemeral'))).toBe(false)

  const retry = fixtureRun()
  const result = await provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath,
    statePath,
    rotate: false,
  }, {
    run: retry.run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral-retry'),
  })
  expect(result.reused).toBe(false)
  expect(retry.calls.map((call) => call[call.indexOf('fn') + 1]))
    .toEqual(['deploy', 'invoke', 'delete'])
})

test('orchestrator reports a sanitized CloudBase function error response', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-response-error-'))
  const run = (_command, args) => {
    const action = args[args.indexOf('fn') + 1]
    if (action === 'invoke') {
      return {
        status: 0,
        stdout: JSON.stringify({ result: JSON.stringify({ errorCode: -1, errorMessage: 'DATABASE_NOT_READY' }) }),
        stderr: '',
      }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: path.join(temporaryRoot, 'table-join.png'),
    statePath: path.join(temporaryRoot, 'state.json'),
    rotate: true,
  }, {
    run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('DATABASE_NOT_READY')
})

test('orchestrator reports the safe shape of an unexpected invocation response', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-response-shape-'))
  const run = (_command, args) => {
    const action = args[args.indexOf('fn') + 1]
    if (action === 'invoke') {
      return { status: 0, stdout: JSON.stringify({ result: { requestId: 'request-123' } }), stderr: '' }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: path.join(temporaryRoot, 'table-join.png'),
    statePath: path.join(temporaryRoot, 'state.json'),
    rotate: true,
  }, {
    run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('requestId')
})

test('orchestrator attempts deletion after an uncertain deploy failure', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-deploy-failure-'))
  const fixture = fixtureRun({ failDeploy: true })

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: path.join(temporaryRoot, 'table-join.png'),
    statePath: path.join(temporaryRoot, 'state.json'),
    rotate: true,
  }, {
    run: fixture.run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('TABLE_CODE_PROVISION_FAILED')

  expect(fixture.calls.map((call) => call[call.indexOf('fn') + 1]))
    .toEqual(['deploy', 'delete'])
})

test('cleanup target is persisted before deployment starts', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-deploy-marker-'))
  const statePath = path.join(temporaryRoot, 'state.json')
  let markerDuringDeploy = null
  const run = (_command, args) => {
    const action = args[args.indexOf('fn') + 1]
    if (action === 'deploy') {
      markerDuringDeploy = JSON.parse(fs.readFileSync(`${statePath}.invalid`, 'utf8'))
      return { status: 1, stdout: '', stderr: 'deployment result uncertain' }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: path.join(temporaryRoot, 'table-join.png'),
    statePath,
    rotate: true,
  }, {
    run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('TABLE_CODE_PROVISION_FAILED')

  expect(markerDuringDeploy.cleanup_function_name).toMatch(/^billiards-provision-/)
})

test('unrelated not-found diagnostics do not count as successful function deletion', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-delete-diagnostic-'))
  const run = (_command, args) => {
    const action = args[args.indexOf('fn') + 1]
    if (action === 'invoke') {
      return {
        status: 0,
        stdout: JSON.stringify({ data: { RetMsg: JSON.stringify({ status: 'REGISTERED' }) } }),
        stderr: '',
      }
    }
    if (action === 'delete') {
      return { status: 1, stdout: '', stderr: 'config file not found' }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: path.join(temporaryRoot, 'table-join.png'),
    statePath: path.join(temporaryRoot, 'state.json'),
    rotate: true,
  }, {
    run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('TABLE_CODE_CLEANUP_FAILED')
})

test('structured CloudBase function-not-found is an idempotent deletion success', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-delete-missing-function-'))
  const statePath = path.join(temporaryRoot, 'state.json')
  const run = (_command, args) => {
    const action = args[args.indexOf('fn') + 1]
    if (action === 'deploy') {
      return { status: 5, stdout: '', stderr: 'deployment result uncertain' }
    }
    if (action === 'delete') {
      return {
        status: 4,
        stdout: JSON.stringify({
          error: {
            code: 'ResourceNotFound.FunctionName',
            message: 'the requested function is absent',
            exit_code: 4,
          },
        }),
        stderr: '',
      }
    }
    return { status: 0, stdout: '{}', stderr: '' }
  }

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: path.join(temporaryRoot, 'table-join.png'),
    statePath,
    rotate: true,
  }, {
    run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow(/^TABLE_CODE_PROVISION_FAILED:(?![\s\S]*TABLE_CODE_CLEANUP_FAILED)/)

  const marker = JSON.parse(fs.readFileSync(`${statePath}.invalid`, 'utf8'))
  expect(marker.cleanup_function_name).toBeUndefined()
})

test('preview failure leaves the previous local state intact and does not update CloudBase', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-preview-failure-'))
  const output = path.join(temporaryRoot, 'table-join.png')
  const state = path.join(temporaryRoot, 'state.json')
  fs.writeFileSync(output, png)
  fs.writeFileSync(state, JSON.stringify({
    schema_version: '1.0',
    code_kind: 'development_preview',
    environment_id: 'cloud1-test',
    device_id: 'raytheon-01',
    table_code: tableCode,
    image_path: output,
  }))
  const originalState = fs.readFileSync(state)
  const fixture = fixtureRun()
  let replacementCode = ''

  await expect(provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: output,
    statePath: state,
    rotate: true,
  }, {
    run: fixture.run,
    generatePreview: ({ tableCode: generatedCode }) => {
      replacementCode = generatedCode
      throw new Error(`preview rejected query table_code=${generatedCode}`)
    },
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('TABLE_CODE_PREVIEW_FAILED')

  expect(fixture.calls).toEqual([])
  expect(fs.readFileSync(output)).toEqual(png)
  expect(fs.readFileSync(state)).toEqual(originalState)
  expect(replacementCode).toMatch(/^[A-Za-z0-9_-]{32}$/)
})

test('legacy or mismatched state is replaced instead of reused', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-state-scope-'))
  const output = path.join(temporaryRoot, 'table-join.png')
  const state = path.join(temporaryRoot, 'state.json')
  fs.writeFileSync(output, png)
  fs.writeFileSync(state, JSON.stringify({
    schema_version: '1.0',
    code_kind: 'permanent',
    environment_id: 'cloud1-other',
    device_id: 'other-device',
    table_code: tableCode,
    image_path: output,
  }))
  const fixture = fixtureRun()
  const preview = fixturePreview()

  const result = await provisionTableCode({
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath: output,
    statePath: state,
    rotate: false,
  }, {
    run: fixture.run,
    generatePreview: preview.generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })

  expect(result.reused).toBe(false)
  expect(preview.calls).toHaveLength(1)
  const persisted = JSON.parse(fs.readFileSync(state, 'utf8'))
  expect(persisted.code_kind).toBe('development_preview')
  expect(persisted.environment_id).toBe('cloud1-test')
  expect(persisted.device_id).toBe('raytheon-01')
})

test('failure diagnostics redact the raw table code', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-redaction-'))
  let generatedCode = ''

  try {
    await provisionTableCode({
      environmentId: 'cloud1-test',
      deviceId: 'raytheon-01',
      outputPath: path.join(temporaryRoot, 'table-join.png'),
      statePath: path.join(temporaryRoot, 'state.json'),
      rotate: true,
    }, {
      run: fixtureRun().run,
      generatePreview: ({ tableCode: code }) => {
        generatedCode = code
        throw new Error(`preview echoed table_code=${code}`)
      },
      randomBytes: deterministicRandom,
      temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
    })
    throw new Error('expected provisioning to fail')
  } catch (error) {
    expect(error.message).toContain('TABLE_CODE_PREVIEW_FAILED')
    expect(error.message).toContain('REDACTED')
    expect(error.message).not.toContain(generatedCode)
  }
})

test('temporary function deletion failure prevents a PASS result', async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'table-code-delete-failure-'))
  const fixture = fixtureRun({ failDelete: true })
  const outputPath = path.join(temporaryRoot, 'table-join.png')
  const statePath = path.join(temporaryRoot, 'state.json')
  const previousImage = Buffer.from('\xff\xd8\xffprevious-preview', 'binary')
  fs.writeFileSync(outputPath, previousImage)
  fs.writeFileSync(statePath, JSON.stringify({
    schema_version: '1.0',
    code_kind: 'development_preview',
    environment_id: 'cloud1-test',
    device_id: 'raytheon-01',
    table_code: tableCode,
    image_path: outputPath,
  }))
  const options = {
    environmentId: 'cloud1-test',
    deviceId: 'raytheon-01',
    outputPath,
    statePath,
    rotate: true,
  }

  await expect(provisionTableCode(options, {
    run: fixture.run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral'),
  })).rejects.toThrow('TABLE_CODE_CLEANUP_FAILED')

  expect(fs.readFileSync(outputPath)).toEqual(previousImage)
  expect(fs.existsSync(statePath)).toBe(true)
  expect(fs.existsSync(`${statePath}.invalid`)).toBe(true)
  const pending = JSON.parse(fs.readFileSync(`${statePath}.invalid`, 'utf8'))
  expect(pending.environment_id).toBe('cloud1-test')
  expect(pending.cleanup_function_name).toMatch(/^billiards-provision-/)

  const retry = fixtureRun()
  const result = await provisionTableCode({ ...options, rotate: false }, {
    run: retry.run,
    generatePreview: fixturePreview().generate,
    randomBytes: deterministicRandom,
    temporaryDirectory: () => path.join(temporaryRoot, 'ephemeral-retry'),
  })

  expect(result.reused).toBe(false)
  expect(fs.existsSync(`${statePath}.invalid`)).toBe(false)
  expect(retry.calls.map((call) => call[call.indexOf('fn') + 1]))
    .toEqual(['delete', 'deploy', 'invoke', 'delete'])
})

test('CLI output exposes only the image path and reuse status', async () => {
  const writes = []
  const output = path.join(os.tmpdir(), 'table-code-cli-output.png')
  const result = await main([
    '--environment', 'development',
    '--environment-id', 'cloud1-test',
    '--device-id', 'raytheon-01',
    '--output', output,
    '--state', path.join(os.tmpdir(), 'table-code-cli-state.json'),
  ], {
    provision: async () => ({ image: png, tableCode: 'do-not-print-this-table-code-001' }),
    write: (value) => writes.push(value),
  })

  expect(result).toEqual({
    status: 'PASS',
    image_path: output,
    image_sha256: crypto.createHash('sha256').update(png).digest('hex'),
    code_kind: 'development_preview',
    reused: false,
  })
  expect(JSON.parse(writes.join(''))).toEqual(result)
  expect(writes.join('')).not.toContain('do-not-print-this-table-code-001')
})
