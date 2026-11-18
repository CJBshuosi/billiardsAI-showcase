#!/usr/bin/env node

const fs = require('fs')
const path = require('path')

const { buildRetentionInventory } = require('../cloudfunctions/media-retention/lib/inventory')


function main(argv) {
  if (argv.length !== 1 || path.extname(argv[0]) !== '.json') {
    throw new Error('usage: media-retention-dry-run.js synthetic-fixture.json')
  }
  const input = JSON.parse(fs.readFileSync(path.resolve(argv[0]), 'utf8'))
  const result = buildRetentionInventory(input)
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
}

try {
  main(process.argv.slice(2))
} catch (error) {
  process.stderr.write(`${error && error.message ? error.message : 'invalid input'}\n`)
  process.exitCode = 1
}
