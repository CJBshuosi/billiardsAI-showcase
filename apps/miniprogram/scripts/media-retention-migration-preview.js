#!/usr/bin/env node

const fs = require('fs')
const path = require('path')

const { buildLegacyMigrationPreview } = require('../cloudfunctions/media-retention/lib/migration')


function main(argv) {
  if (argv.length !== 1) throw new Error('usage: media-retention-migration-preview.js synthetic-input.json')
  const inputPath = path.resolve(argv[0])
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'))
  if (!input || input.synthetic_fixture !== true) throw new Error('only a local synthetic fixture is accepted')
  process.stdout.write(`${JSON.stringify(buildLegacyMigrationPreview(input), null, 2)}\n`)
}


if (require.main === module) {
  try { main(process.argv.slice(2)) } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 2
  }
}

module.exports = { main }
