const { normalizeResult } = require('../miniprogram/utils/result')

test('normalizes cloud errors without leaking internal details', () => {
  expect(normalizeResult({ ok: false, error: { code: 'FORBIDDEN', stack: 'secret' } }))
    .toEqual({ ok: false, error: { code: 'FORBIDDEN' } })
})

test('normalizes successful cloud results', () => {
  expect(normalizeResult({ ok: true, data: { user_id: 'user-1' } }))
    .toEqual({ ok: true, data: { user_id: 'user-1' } })
})

test('rejects malformed cloud responses', () => {
  expect(normalizeResult({ value: 'unexpected' }))
    .toEqual({ ok: false, error: { code: 'INTERNAL_ERROR' } })
})
