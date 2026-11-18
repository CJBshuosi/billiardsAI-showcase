const { ownerDeleteDb } = require('../helpers/owner-delete-db')
const { documentId } = require('../../cloudfunctions/shared/shot-ownership')
const { privateClip } = require('../helpers/private-clips')
const { ownershipProjection } = require('../../cloudfunctions/shared/shot-ownership')
// Local byte-serving storage contract test, NOT CloudBase or Raytheon acceptance.
const http = require('http')
const { createCloudContentRepository, getClipPlayback } = require('../../cloudfunctions/client-api/lib/content')

test('signed playback retrieves bytes; outsider, revoked consent and expired URLs fail closed', async () => {
  const bytes = Buffer.from('synthetic-private-video-fixture')
  let expired = false
  const server = http.createServer((request, response) => {
    if (expired || request.url !== '/clip?token=fixture-private-token') {
      response.writeHead(403).end()
    } else response.writeHead(200, { 'content-type': 'video/mp4' }).end(bytes)
  })
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  try {
    const url = `http://127.0.0.1:${server.address().port}/clip?token=fixture-private-token`
    const db = ownerDeleteDb({ clips: { c: privateClip({ clip_id: 'c', storage_file_id: 'cloud://fixture/c' }) },
      field_shot_ownership: { [documentId('session-1:shot-0001')]: ownershipProjection(privateClip(), 'table-1', 'device-1') } })
    const repo = createCloudContentRepository(db, { async getTempFileURL({ fileList }) {
      expect(fileList).toEqual([{ fileID: 'cloud://fixture/c', maxAge: 300 }])
      return { fileList: [{ status: 0, tempFileURL: url }] }
    } }, { clock: () => Date.parse('2026-10-29T19:00:00.000Z') })
    const playback = await getClipPlayback({ clipId: 'c', userId: 'user-a', repo })
    const response = await fetch(playback.url)
    expect(response.status).toBe(200)
    expect(Buffer.from(await response.arrayBuffer())).toEqual(bytes)
    expect((await fetch(url.replace('fixture-private-token', 'wrong'))).status).toBe(403)
    await expect(getClipPlayback({ clipId: 'c', userId: 'outsider', repo }))
      .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
    db.rows.clips.c.consent_status = 'revoked'
    await expect(getClipPlayback({ clipId: 'c', userId: 'user-a', repo }))
      .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
    expired = true
    expect((await fetch(playback.url)).status).toBe(403)
  } finally {
    server.closeAllConnections()
    await new Promise(resolve => server.close(resolve))
  }
})
