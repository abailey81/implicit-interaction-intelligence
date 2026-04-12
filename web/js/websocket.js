/**
 * WebSocket client with automatic reconnection and event routing.
 *
 * Provides a typed event system: incoming JSON messages with a `type` field
 * are dispatched to registered handlers. Reconnection uses exponential
 * backoff with jitter.
 */

class WebSocketClient {
    /**
     * @param {string} url  WebSocket endpoint URL.
     */
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.handlers = {};
        this.reconnectAttempts = 0;
        this.maxReconnect = 5;
        this._reconnectTimer = null;
    }

    /**
     * Establish the WebSocket connection.
     */
    connect() {
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }

        try {
            this.ws = new WebSocket(this.url);
        } catch (err) {
            console.error('[WS] Failed to create WebSocket:', err);
            this._reconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('[WS] Connected to', this.url);
            this.reconnectAttempts = 0;
            this._trigger('connected');
        };

        this.ws.onmessage = (event) => {
            // SEC: Reject non-string frames (binary). We only speak JSON over WS.
            if (typeof event.data !== 'string') {
                console.warn('[WS] Ignoring non-string frame');
                return;
            }
            // SEC: Cap incoming frame size to prevent UI lockup from a hostile
            // or buggy server. 256 KiB is generous for our payloads.
            if (event.data.length > 262144) {
                console.warn('[WS] Dropping oversized frame:', event.data.length);
                return;
            }
            let data;
            try {
                data = JSON.parse(event.data);
            } catch (err) {
                // SEC: Malformed JSON is silently dropped (with log) — we never
                // pass raw event.data to the DOM.
                console.error('[WS] Failed to parse message:', err);
                return;
            }
            // SEC: Validate envelope shape before dispatch.
            if (!data || typeof data !== 'object' || Array.isArray(data)) {
                console.warn('[WS] Ignoring non-object payload');
                return;
            }
            if (data.type !== undefined && typeof data.type !== 'string') {
                console.warn('[WS] Ignoring payload with non-string type');
                return;
            }
            if (data.type) {
                this._trigger(data.type, data);
            } else {
                this._trigger('message', data);
            }
        };

        this.ws.onclose = (event) => {
            console.log('[WS] Connection closed:', event.code, event.reason);
            this._trigger('disconnected');
            this._reconnect();
        };

        this.ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            this._trigger('error', err);
        };
    }

    /**
     * Send a JSON-serialisable object.
     * @param {object} data  Payload to send.
     */
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('[WS] Cannot send -- connection not open.');
        }
    }

    /**
     * Register an event handler.
     * @param {string}   type     Event type (matches `data.type` from server).
     * @param {function} handler  Callback receiving the parsed data object.
     */
    on(type, handler) {
        if (!this.handlers[type]) {
            this.handlers[type] = [];
        }
        this.handlers[type].push(handler);
    }

    /**
     * Dispatch an event to all registered handlers.
     * @private
     */
    _trigger(type, data) {
        const list = this.handlers[type];
        if (list) {
            for (const handler of list) {
                try {
                    handler(data);
                } catch (err) {
                    console.error(`[WS] Handler error for "${type}":`, err);
                }
            }
        }
    }

    /**
     * Exponential backoff reconnection with jitter.
     * @private
     */
    _reconnect() {
        if (this.reconnectAttempts >= this.maxReconnect) {
            console.warn('[WS] Max reconnection attempts reached.');
            this._trigger('reconnect_failed');
            return;
        }

        this.reconnectAttempts++;
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s + random jitter
        const baseDelay = Math.pow(2, this.reconnectAttempts - 1) * 1000;
        const jitter = Math.random() * 500;
        const delay = baseDelay + jitter;

        console.log(
            `[WS] Reconnecting in ${Math.round(delay)}ms ` +
            `(attempt ${this.reconnectAttempts}/${this.maxReconnect})`
        );

        this._trigger('reconnecting', {
            attempt: this.reconnectAttempts,
            maxAttempts: this.maxReconnect,
            delayMs: delay,
        });

        this._reconnectTimer = setTimeout(() => this.connect(), delay);
    }

    /**
     * Cleanly close the WebSocket without triggering reconnection.
     */
    disconnect() {
        this.maxReconnect = 0;  // prevent reconnect
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }
    }

    /**
     * @returns {boolean} True if the WebSocket is currently open.
     */
    get isConnected() {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
}
