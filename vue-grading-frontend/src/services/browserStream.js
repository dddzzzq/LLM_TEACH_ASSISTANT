// Input is pipelined on one connection. Acknowledgements detect uncertainty;
// they never gate the next event and unconfirmed events are never replayed.
export function createBrowserStream({ url, token, prepare, onFrame, onError, onReset, onState,
  socketFactory = value => new WebSocket(value), reconnectDelay = 1000, heartbeatMs = 10000 }) {
  let socket, retry, heartbeat, active = false, ready = false, generation = 0, sequence = 0
  let lastMessage = 0
  const pending = new Map()
  function closeConnection() {
    ready = false
    clearInterval(heartbeat)
    pending.clear()
    const previous = socket
    socket = undefined
    previous?.close()
    onReset?.()
  }
  function stop() {
    active = false
    generation++
    clearTimeout(retry)
    closeConnection()
    onState?.('disconnected')
  }
  function failed(message) {
    onError?.(message)
    closeConnection()
    if (active) retry = setTimeout(connect, reconnectDelay)
    onState?.('connecting')
  }
  async function connect() {
    const current = ++generation
    onState?.('connecting')
    try {
      await prepare?.()
      if (!active || current !== generation) return
      const endpoint = new URL(url(), window.location.href)
      endpoint.protocol = endpoint.protocol === 'https:' ? 'wss:' : 'ws:'
      const connection = socketFactory(endpoint.href)
      socket = connection
      sequence = 0
      lastMessage = Date.now()
      connection.onopen = () => {
        if (socket !== connection) return
        connection.send(JSON.stringify({ type: 'auth', token: token() }))
        heartbeat = setInterval(() => {
          if (socket !== connection) return
          if (Date.now() - lastMessage > 35000 || [...pending.values()].some(sent => Date.now() - sent > 10000)) {
            failed('连接超时，操作结果未确认，请检查画面后重试')
          } else {
            connection.send(JSON.stringify({ type: 'ping' }))
          }
        }, heartbeatMs)
      }
      connection.onmessage = event => {
        if (socket !== connection) return
        let message
        try { message = JSON.parse(event.data) } catch { failed('浏览器画面数据无效'); return }
        lastMessage = Date.now()
        if (message.type === 'ready') { ready = true; onState?.('connected') }
        else if (message.type === 'frame') onFrame?.(message)
        else if (message.type === 'input_ack') pending.delete(message.id)
        else if (message.type === 'error') onError?.(message.message || '浏览器操作未完成')
        else if (message.type === 'ended') { stop(); onError?.(message.message) }
      }
      connection.onerror = () => { if (socket === connection) failed('浏览器连接失败，正在重连') }
      connection.onclose = () => { if (socket === connection) failed('连接已断开，未确认的操作不会重发，请检查画面后继续') }
    } catch {
      if (active && current === generation) failed('浏览器暂未就绪，正在重连')
    }
  }
  function start() {
    if (active) return
    active = true
    connect()
  }
  function send(action) {
    if (!ready || !socket || socket.readyState !== 1) throw new Error('浏览器尚未连接')
    if (socket.bufferedAmount > 128 * 1024 || pending.size >= 128) {
      failed('连接过慢，操作已停止，请检查画面后重试')
      throw new Error('浏览器连接拥塞')
    }
    const id = ++sequence
    pending.set(id, Date.now())
    try { socket.send(JSON.stringify({ type: 'input', id, action })) }
    catch (error) { failed('操作结果未确认，请检查画面后重试'); throw error }
  }
  return { start, stop, send }
}
