const {
  evidenceExpiry,
  favoriteExpiry,
  ordinaryExpiry,
  releaseGraceExpiry,
  retentionStatus,
  timestamp,
} = require('../../cloudfunctions/shared/retention-policy')


const NOW = '2026-09-01T00:00:00.000Z'


test('retention deadlines use exact UTC durations', () => {
  expect(ordinaryExpiry(NOW)).toBe('2026-09-03T00:00:00.000Z')
  expect(releaseGraceExpiry(NOW)).toBe('2026-09-02T00:00:00.000Z')
  expect(favoriteExpiry(NOW)).toBe('2026-10-01T00:00:00.000Z')
  expect(evidenceExpiry(NOW)).toBe('2026-11-30T00:00:00.000Z')
})

test.each([null, '', 'not-a-date', '2026-09-01'])('invalid authority timestamp %p fails closed', value => {
  expect(() => ordinaryExpiry(value)).toThrow('invalid retention timestamp')
})


test('retention status treats the exact deadline as expired', () => {
  expect(retentionStatus({ availableUntil: '2026-09-03T00:00:00.001Z', now: '2026-09-03T00:00:00.000Z' }))
    .toEqual({ state: 'active', remaining_seconds: 0 })
  expect(retentionStatus({ availableUntil: '2026-09-03T00:00:00.000Z', now: '2026-09-03T00:00:00.000Z' }))
    .toEqual({ state: 'expired', remaining_seconds: 0 })
})

test('canonical whole-second SQLite UTC timestamps have identical instants and deadlines', () => {
  const seconds = '2026-09-01T00:00:00Z'
  expect(timestamp(seconds)).toBe(timestamp(NOW))
  for (const expiry of [ordinaryExpiry, releaseGraceExpiry, favoriteExpiry, evidenceExpiry]) {
    expect(expiry(seconds)).toBe(expiry(NOW))
    expect(expiry(seconds)).toMatch(/\.000Z$/u)
  }
  expect(timestamp('2026-09-01T00:00:00.123Z') - timestamp(seconds)).toBe(123)
  expect(retentionStatus({ availableUntil: '2026-09-03T00:00:00Z', now: '2026-09-03T00:00:00.000Z' }))
    .toEqual({ state: 'expired', remaining_seconds: 0 })
  expect(retentionStatus({ availableUntil: '2026-09-03T00:00:00.001Z', now: '2026-09-03T00:00:00Z' }))
    .toEqual({ state: 'active', remaining_seconds: 0 })
})

test.each([0, NaN, Infinity, {}, [], '2026-02-30T00:00:00Z', '2026-02-30T00:00:00.000Z',
  '2026-09-01T00:00:00+00:00', '2026-09-01T00:00:00z', '2026-09-01t00:00:00Z',
  '2026-09-01T00:00:00.0Z', '2026-09-01T00:00:00.00Z', '2026-09-01T00:00:00.0000Z',
  '2026-09-01T24:00:00Z', '2026-09-01T00:00:60Z', '2026-09-01T00:00:00Z\n',
])('noncanonical or invalid timestamp %p remains rejected', value => {
  expect(() => timestamp(value)).toThrow('invalid retention timestamp')
})
