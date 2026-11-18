const { privateClip } = require('../helpers/private-clips')
const { ownershipProjection } = require('../../cloudfunctions/shared/shot-ownership')
const {
  createCloudContentRepository,
  getClipPlayback,
} = require('../../cloudfunctions/client-api/lib/content')


function playbackRepository({ clip = {}, favoriteCollectionError } = {}) {
  const collectionsRead = []
  const db = {
    async runTransaction(fn) { return fn(db) },
    collection(name) {
      collectionsRead.push(name)
      if (name === 'clip_favorites' && favoriteCollectionError) throw favoriteCollectionError
      return {
        doc() {
          return { async update() {}, async get() {
            if (name === 'owner_clip_tombstones') return { data: null }
            if (name === 'clips') return { data: privateClip({ clip_id: 'clip-1', ...clip }) }
            if (name === 'field_shot_ownership') return { data: ownershipProjection(privateClip(), 'table-1', 'device-1') }
            throw new Error(`Unexpected collection: ${name}`)
          } }
        },
        where() { return this },
        limit() { return this },
        async get() { return { data: [{ user_id: 'user-1', session_id: 'session-1' }] } },
      }
    },
  }
  const cloud = { async getTempFileURL({ fileList }) {
    if (fileList[0].fileID !== 'cloud://private/clip.mp4') throw new Error('Unexpected file')
    return { fileList: [{ status: 0, tempFileURL: 'https://media.example/clip.mp4' }] }
  } }
  return { repo: createCloudContentRepository(db, cloud, { clock: () => Date.parse('2026-10-29T19:00:00.000Z') }), collectionsRead }
}


test('playback never reads the historical favorite collection', async () => {
  const favoriteError = new Error('database permission denied')
  const { repo, collectionsRead } = playbackRepository({ favoriteCollectionError: favoriteError })
  await expect(getClipPlayback({ clipId: 'clip-1', userId: 'user-a', repo }))
    .resolves.toMatchObject({
      schema_version: '1.0', url: 'https://media.example/clip.mp4', expires_in_seconds: 300,
    })
  expect(collectionsRead).not.toContain('clip_favorites')
  expect(repo).not.toHaveProperty('changeFavorite')
  expect(repo).not.toHaveProperty('isFavorite')
})


test('legacy published clip cannot bypass saved-media playback authority', async () => {
  const { repo } = playbackRepository({ clip: { publication_kind: undefined } })
  await expect(getClipPlayback({ clipId: 'clip-1', userId: 'user-a', repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
})


test('missing clip document returns media-not-available without signing a URL', async () => {
  const db = { async runTransaction(fn) { return fn(this) }, collection() { return { doc() { return { async get() {
    throw new Error('document.get:fail document with _id missing-clip does not exist')
  } } } } } }
  const repo = createCloudContentRepository(db, { getTempFileURL() { throw new Error('must not sign') } })
  await expect(getClipPlayback({ clipId: 'missing-clip', userId: 'user-a', repo }))
    .rejects.toMatchObject({ code: 'MEDIA_NOT_AVAILABLE' })
})
