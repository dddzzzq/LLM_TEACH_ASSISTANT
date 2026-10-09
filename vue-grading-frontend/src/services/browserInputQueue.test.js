import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createBrowserInputQueue } from './browserInputQueue.js'

const action = (kind, values = {}) => ({ kind, job_id: 'job', epoch: 1, page_id: '0', revision: 1, ...values })

test('typing is combined without crossing keyboard or click boundaries', async () => {
  const sent = []
  const queue = createBrowserInputQueue({ send: async payload => { sent.push(payload) } })
  const results = await Promise.all([
    queue.enqueue(action('text', { text: 'a' })),
    queue.enqueue(action('text', { text: 'b' })),
    queue.enqueue(action('key', { key: 'Tab' })),
    queue.enqueue(action('text', { text: 'c' })),
    queue.enqueue(action('pointer_down', { x: 1, y: 2 })),
    queue.enqueue(action('pointer_up', { x: 1, y: 2 })),
  ])
  assert.ok(results.every(Boolean))
  assert.equal(sent.length, 1)
  assert.deepEqual(sent[0].actions.map(item => item.text || item.key || item.kind), ['ab', 'Tab', 'c', 'pointer_down', 'pointer_up'])
})

test('slow transport coalesces motion and preserves release, with one request at a time', async () => {
  let unblock, started
  const ready = new Promise(resolve => { started = resolve })
  const gate = new Promise(resolve => { unblock = resolve })
  const sent = []
  const queue = createBrowserInputQueue({ delay: 0, send: async payload => {
    sent.push(payload)
    if (sent.length === 1) { started(); await gate }
  } })
  const down = queue.enqueue(action('pointer_down'))
  await ready
  const moves = Array.from({ length: 100 }, (_, x) => queue.enqueue(action('pointer_move', { x, y: 0 })))
  const up = queue.enqueue(action('pointer_up'))
  assert.equal(sent.length, 1)
  unblock()
  await Promise.all([down, ...moves, up])
  assert.equal(sent.length, 2)
  assert.deepEqual(sent[1].actions.map(item => item.kind), ['pointer_move', 'pointer_up'])
  assert.equal(sent[1].actions[0].x, 99)
})

test('transport failure drops pending events without replaying uncertain input', async () => {
  let errors = 0
  const sent = []
  const queue = createBrowserInputQueue({ send: async payload => {
    sent.push(payload)
    throw new Error('connection lost')
  }, onError: () => { errors++ } })
  assert.equal(await queue.enqueue(action('text', { text: 'secret' })), false)
  assert.equal(sent.length, 1)
  assert.equal(errors, 1)
})

test('control changes discard queued input', async () => {
  let sent = 0
  const queue = createBrowserInputQueue({ send: async () => { sent++ } })
  const typing = queue.enqueue(action('text', { text: 'secret' }))
  queue.clear()
  assert.equal(await typing, false)
  assert.equal(sent, 0)
})
