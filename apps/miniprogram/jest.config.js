module.exports = {
  clearMocks: true,
  testEnvironment: 'node',
  // Tests resolve the same locked SDK packages as the cloud functions.
  modulePaths: ['<rootDir>/cloudfunctions/node_modules'],
  testMatch: ['<rootDir>/tests/**/*.test.js'],
}
