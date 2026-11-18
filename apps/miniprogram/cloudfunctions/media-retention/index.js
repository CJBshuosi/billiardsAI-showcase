const crypto = require('crypto')
const cloudbase = require('@cloudbase/node-sdk')
const cloud = require('wx-server-sdk')

const { createCloudRetentionRepository } = require('./lib/cloud-repository')
const { runSyntheticRetentionBatch } = require('./lib/runner')
const {
  createCloudBaseStorageProvider,
  createStoragePort,
} = require('../sync-api/lib/storage')


cloud.init({ env: cloud.DYNAMIC_CURRENT_ENV })
const storageCloud = cloudbase.init({ env: cloudbase.SYMBOL_CURRENT_ENV })
const db = cloud.database()


exports.main = async () => runSyntheticRetentionBatch({
  repo: createCloudRetentionRepository(db),
  storage: createStoragePort(createCloudBaseStorageProvider(storageCloud)),
  env: process.env,
  workerId: `retention-${crypto.randomUUID()}`,
  now: new Date().toISOString(),
})
