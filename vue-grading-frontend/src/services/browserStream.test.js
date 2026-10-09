import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createBrowserStream } from './browserStream.js'
import { createBrowserInputQueue } from './browserInputQueue.js'

globalThis.window = { location: { href: 'https://portal.example/assistant' } }
const pause = ms => new Promise(resolve => setTimeout(resolve, ms))
async function until(predicate) {
  for (let i = 0; i < 100 && !predicate(); i++) await pause(5)
  assert.ok(predicate())
}
function fixture(options = {}) {
  const sockets = [], errors = [], frames = []
  const stream = createBrowserStream({
    url: () => '/api/rpa/jobs/job/stream', token: () => 'fixture-private-token',
    reconnectDelay: 5, onError: message => errors.push(message), onFrame: frame => frames.push(frame),
    socketFactory: url => {
      const socket = { url, readyState: 1, bufferedAmount: 0, sent: [],
        send(value) { this.sent.push(JSON.parse(value)) },
        close() { this.readyState = 3; this.onclose?.() },
        message(value) { this.onmessage({ data: JSON.stringify(value) }) },
      }
      sockets.push(socket)
      return socket
    }, ...options,
  })
  async function open() {
    stream.start()
    await until(() => sockets.length > 0)
    sockets.at(-1).onopen()
    sockets.at(-1).message({ type: 'ready' })
    return sockets.at(-1)
  }
  return { stream, sockets, errors, frames, open }
}

test('drag events are pipelined without acknowledgements and frames continue during drag', async t => {
  const f = fixture()
  t.after(() => f.stream.stop())
  const socket = await f.open()
  assert.equal(socket.url, 'wss://portal.example/api/rpa/jobs/job/stream')
  assert.deepEqual(socket.sent[0], { type: 'auth', token: 'fixture-private-token' })
  const queue = createBrowserInputQueue({ delay: 0, send: value => f.stream.send(value) })
  await queue.enqueue({ kind: 'pointer_down' })
  await queue.enqueue({ kind: 'pointer_move', x: 100 })
  socket.message({ type: 'frame', revision: 2 })
  await queue.enqueue({ kind: 'pointer_up' })
  assert.deepEqual(socket.sent.slice(1).map(value => value.action.kind), ['pointer_down', 'pointer_move', 'pointer_up'])
  assert.deepEqual(socket.sent.slice(1).map(value => value.id), [1, 2, 3])
  assert.equal(f.frames[0].revision, 2)
})

test('reconnecting clears unconfirmed input and ignores messages from the old socket', async t => {
  const f = fixture()
  t.after(() => f.stream.stop())
  const old = await f.open()
  f.stream.send({ kind: 'text', text: 'private-password' })
  old.close()
  await until(() => f.sockets.length === 2)
  const next = f.sockets[1]
  next.onopen()
  next.message({ type: 'ready' })
  old.message({ type: 'frame', revision: 100 })
  assert.equal(f.frames.length, 0)
  assert.equal(next.sent.length, 1)
  assert.equal(next.sent[0].type, 'auth')
  f.stream.send({ kind: 'key', key: 'Tab' })
  assert.equal(next.sent[1].id, 1)
})

test('socket backpressure stops input instead of buffering an unbounded drag', async t => {
  const f = fixture()
  t.after(() => f.stream.stop())
  const socket = await f.open()
  socket.bufferedAmount = 129 * 1024
  assert.throws(() => f.stream.send({ kind: 'pointer_move' }))
  assert.equal(socket.readyState, 3)
  assert.equal(socket.sent.length, 1)
  assert.equal(f.errors.length, 1)
})

test('stopping while authentication is pending cannot create a stale connection', async () => {
  let release
  const f = fixture({ prepare: () => new Promise(resolve => { release = resolve }) })
  f.stream.start()
  f.stream.stop()
  release()
  await pause(10)
  assert.equal(f.sockets.length, 0)
})
