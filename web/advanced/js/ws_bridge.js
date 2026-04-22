/**
 * ws_bridge.js — resilient WebSocket client.
 *
 * Features:
 *   - Exponential backoff reconnect (250ms → 8s cap).
 *   - Heartbeat ping/pong every heartbeatMs.
 *   - Backpressure: drops oldest frames when inbound queue > maxQueue.
 *   - Tiny typed event emitter; no external dependencies.
 *
 * Usage:
 *     const ws = createWSBridge({ url, heartbeatMs: 15000, maxQueue: 100 });
 *     ws.on("state", st => ...);
 *     ws.on("frame", msg => ...);
 *     ws.connect();
 *     ws.send({ type: "user_input", text: "hi" });
 */

export function createWSBridge({ url, heartbeatMs = 15000, maxQueue = 100 } = {}) {
  const handlers = { state: new Set(), frame: new Set() };
  let socket = null;
  let backoff = 250;
  const backoffCap = 8000;
  let heartbeatTimer = null;
  let dropped = 0;
  const queue = [];
  let closedByUser = false;

  function emit(kind, payload) {
    for (const fn of handlers[kind]) {
      try { fn(payload); } catch (e) { console.error("[ws_bridge] handler error", e); }
    }
  }

  function on(kind, fn) {
    if (!handlers[kind]) throw new Error("unknown event: " + kind);
    handlers[kind].add(fn);
    return () => handlers[kind].delete(fn);
  }

  function setState(st) { emit("state", st); }

  function scheduleReconnect() {
    if (closedByUser) return;
    setTimeout(() => {
      backoff = Math.min(backoff * 2, backoffCap);
      connect();
    }, backoff);
  }

  function startHeartbeat() {
    stopHeartbeat();
    heartbeatTimer = setInterval(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        try { socket.send(JSON.stringify({ type: "ping", t: Date.now() })); }
        catch { /* swallow — socket may be mid-close */ }
      }
    }, heartbeatMs);
  }
  function stopHeartbeat() {
    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
  }

  function pushFrame(msg) {
    queue.push(msg);
    while (queue.length > maxQueue) { queue.shift(); dropped += 1; }
    // Drain one per micro-tick so the UI stays responsive under bursts.
    Promise.resolve().then(() => {
      const next = queue.shift();
      if (next !== undefined) emit("frame", next);
    });
  }

  function connect() {
    closedByUser = false;
    setState("connecting");
    try {
      socket = new WebSocket(url);
    } catch (e) {
      console.warn("[ws_bridge] constructor threw", e);
      setState("error");
      scheduleReconnect();
      return;
    }

    socket.addEventListener("open", () => {
      backoff = 250;
      setState("open");
      startHeartbeat();
    });

    socket.addEventListener("message", (ev) => {
      let data = ev.data;
      if (typeof data === "string") {
        try { data = JSON.parse(data); }
        catch { /* leave as string */ }
      }
      if (data && data.type === "pong") return; // silent
      pushFrame(data);
    });

    socket.addEventListener("error", () => {
      setState("error");
    });

    socket.addEventListener("close", () => {
      stopHeartbeat();
      setState("closed");
      scheduleReconnect();
    });
  }

  function send(obj) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    try {
      socket.send(typeof obj === "string" ? obj : JSON.stringify(obj));
      return true;
    } catch (e) {
      console.warn("[ws_bridge] send failed", e);
      return false;
    }
  }

  function close() {
    closedByUser = true;
    stopHeartbeat();
    if (socket) { try { socket.close(); } catch { /* ignore */ } }
  }

  function stats() { return { dropped, queueLength: queue.length, backoff }; }

  return { on, connect, send, close, stats };
}
