// Keep discrete events ordered; replace only adjacent motion or text events.
// Never retry input after a transport error: its execution may be uncertain.
export function createBrowserInputQueue({ send, onIdle, onError, delay = 12 }) {
  let pending = [], running = false, timer, generation = 0
  function clear() {
    generation++
    clearTimeout(timer)
    timer = undefined
    for (const entry of pending) entry.done.forEach(resolve => resolve(false))
    pending = []
  }
  async function flush() {
    timer = undefined
    if (running || !pending.length) return
    running = true
    const revision = generation
    const entries = pending.splice(0, 32)
    let ok = false
    try {
      const actions = entries.map(entry => entry.action)
      await send(actions.length === 1 ? actions[0] : { kind: 'batch', actions })
      ok = revision === generation
    } catch (error) {
      if (revision === generation) { clear(); onError?.(error) }
    } finally {
      entries.forEach(entry => entry.done.forEach(resolve => resolve(ok)))
      running = false
      if (pending.length) schedule(0)
      else if (revision === generation) onIdle?.()
    }
  }
  function schedule(wait = delay) {
    if (!running && timer === undefined) timer = setTimeout(flush, wait)
  }
  function enqueue(action) {
    return new Promise(resolve => {
      const previous = pending.at(-1)
      const last = previous?.action
      const sameFrame = last && ['job_id', 'epoch', 'page_id', 'revision'].every(key => last[key] === action[key])
      if (sameFrame && last.kind === action.kind && action.kind === 'pointer_move') {
        previous.action = action
        previous.done.push(resolve)
      } else if (sameFrame && last.kind === 'text' && action.kind === 'text' && last.text.length + action.text.length <= 2000) {
        last.text += action.text
        previous.done.push(resolve)
      } else if (sameFrame && last.kind === 'scroll' && action.kind === 'scroll'
          && Math.sign(last.delta_y) === Math.sign(action.delta_y) && Math.abs(last.delta_y + action.delta_y) <= 1000) {
        last.delta_y += action.delta_y
        previous.done.push(resolve)
      } else {
        pending.push({ action: { ...action }, done: [resolve] })
      }
      schedule()
    })
  }
  return { enqueue, clear }
}
