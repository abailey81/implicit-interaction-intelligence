/**
 * Main application module for Implicit Interaction Intelligence (I3).
 *
 * Orchestrates the WebSocket connection, chat interface, dashboard,
 * embedding visualisation, keystroke monitor, and interaction diary.
 */

// =========================================================================
// KeystrokeMonitor -- captures implicit behavioural signals from typing
// =========================================================================

class KeystrokeMonitor {
    /**
     * @param {WebSocketClient} wsClient  WebSocket client for streaming events.
     */
    constructor(wsClient) {
        this.ws = wsClient;
        this.lastKeyTime = null;
        this.keyTimings = [];          // inter-key intervals (ms)
        this.backspaceCount = 0;
        this.totalKeystrokes = 0;
        this.compositionStartTime = null;
        this.lastKeystrokeTime = null;
        this._attached = false;
    }

    /**
     * Attach keystroke listeners to an input element.
     * @param {HTMLInputElement} inputElement
     */
    attach(inputElement) {
        if (this._attached) return;
        this._attached = true;

        inputElement.addEventListener('keydown', (e) => {
            const now = performance.now();

            // Start composition timer on first keystroke
            if (this.compositionStartTime === null) {
                this.compositionStartTime = now;
            }

            this.totalKeystrokes++;
            this.lastKeystrokeTime = now;

            // Track inter-key interval
            if (this.lastKeyTime !== null) {
                const iki = now - this.lastKeyTime;
                this.keyTimings.push(iki);
            }
            this.lastKeyTime = now;

            // Track backspaces
            if (e.key === 'Backspace' || e.key === 'Delete') {
                this.backspaceCount++;
            }

            // Stream keystroke event to server (sampled -- not every key)
            if (this.totalKeystrokes % 3 === 0) {
                this.ws.send({
                    type: 'keystroke',
                    timestamp: Date.now() / 1000,
                    key_type: this._classifyKey(e.key),
                    iki_ms: this.keyTimings.length > 0
                        ? this.keyTimings[this.keyTimings.length - 1]
                        : 0,
                });
            }
        });
    }

    /**
     * Get composition metrics for the current message.
     * @returns {object}
     */
    getCompositionMetrics() {
        const now = performance.now();
        const compositionTime = this.compositionStartTime !== null
            ? now - this.compositionStartTime
            : 0;

        const pauseBeforeSend = this.lastKeystrokeTime !== null
            ? now - this.lastKeystrokeTime
            : 0;

        let meanIki = 0;
        let stdIki = 0;
        if (this.keyTimings.length > 0) {
            const sum = this.keyTimings.reduce((a, b) => a + b, 0);
            meanIki = sum / this.keyTimings.length;
            if (this.keyTimings.length > 1) {
                const variance = this.keyTimings.reduce(
                    (acc, v) => acc + Math.pow(v - meanIki, 2), 0
                ) / (this.keyTimings.length - 1);
                stdIki = Math.sqrt(variance);
            }
        }

        return {
            composition_time_ms: Math.round(compositionTime),
            total_keystrokes: this.totalKeystrokes,
            backspace_count: this.backspaceCount,
            mean_iki: Math.round(meanIki * 100) / 100,
            std_iki: Math.round(stdIki * 100) / 100,
            pause_before_send: Math.round(pauseBeforeSend),
            keystroke_timings: this.keyTimings.slice(-50),  // Last 50 for bandwidth
        };
    }

    /**
     * Reset all tracking state for a new message.
     */
    reset() {
        this.lastKeyTime = null;
        this.keyTimings = [];
        this.backspaceCount = 0;
        this.totalKeystrokes = 0;
        this.compositionStartTime = null;
        this.lastKeystrokeTime = null;
    }

    /**
     * Classify a key event into a category.
     * @param {string} key
     * @returns {string}
     * @private
     */
    _classifyKey(key) {
        if (key === 'Backspace' || key === 'Delete') return 'backspace';
        if (key === 'Enter') return 'enter';
        if (key.length === 1) return 'char';
        return 'modifier';
    }
}

// =========================================================================
// DiaryPanel -- collapsible interaction diary
// =========================================================================

class DiaryPanel {
    /**
     * @param {HTMLElement} container  The .diary-panel element.
     */
    constructor(container) {
        this.container = container;
        this.entriesEl = container.querySelector('.diary-entries');
        this.entryCount = 0;

        // Toggle expand/collapse
        const header = container.querySelector('.diary-header');
        if (header) {
            header.addEventListener('click', () => this.toggle());
        }
    }

    /**
     * Add a diary entry.
     * @param {object} entry  { timestamp, summary, emotion, topics }
     */
    addEntry(entry) {
        if (!this.entriesEl) return;

        const el = document.createElement('div');
        el.className = 'diary-entry';

        // Timestamp
        const timeEl = document.createElement('span');
        timeEl.className = 'entry-time';
        const d = entry.timestamp ? new Date(entry.timestamp * 1000) : new Date();
        timeEl.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Content
        const contentEl = document.createElement('span');
        contentEl.className = 'entry-content';
        contentEl.textContent = entry.summary || entry.text || 'State change detected';

        // Tags
        if (entry.emotion) {
            const tag = document.createElement('span');
            tag.className = 'entry-tag';
            tag.textContent = entry.emotion;
            contentEl.appendChild(tag);
        }

        if (entry.topics && entry.topics.length > 0) {
            for (const topic of entry.topics.slice(0, 2)) {
                const tag = document.createElement('span');
                tag.className = 'entry-tag';
                tag.textContent = topic;
                contentEl.appendChild(tag);
            }
        }

        el.appendChild(timeEl);
        el.appendChild(contentEl);
        this.entriesEl.appendChild(el);
        this.entryCount++;

        // Auto-expand on first entry
        if (this.entryCount === 1) {
            this.container.classList.add('expanded');
        }

        // Scroll to latest
        this.entriesEl.scrollTop = this.entriesEl.scrollHeight;
    }

    /**
     * Toggle the diary panel open/closed.
     */
    toggle() {
        this.container.classList.toggle('expanded');
    }
}

// =========================================================================
// I3App -- main application orchestrator
// =========================================================================

class I3App {
    constructor() {
        this.userId = 'demo_user';

        // Determine WebSocket URL
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/${this.userId}`;

        // Initialise modules
        this.wsClient = new WebSocketClient(wsUrl);
        this.keystrokeMonitor = new KeystrokeMonitor(this.wsClient);

        this.chat = new ChatInterface(
            document.getElementById('chat-messages'),
            this.wsClient,
            this.keystrokeMonitor
        );

        this.dashboard = new Dashboard(
            document.getElementById('dashboard')
        );

        this.embeddingViz = new EmbeddingViz(
            document.getElementById('embedding-canvas')
        );

        this.diary = new DiaryPanel(
            document.getElementById('diary')
        );

        // Wire up chat input
        const inputEl = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        this.chat.setupInput(inputEl, sendBtn);
        this.chat.setTypingIndicator(document.getElementById('typing-indicator'));

        // Connection status bar
        this.connectionBar = document.getElementById('connection-bar');

        // Set up event handlers
        this.setupHandlers();

        // Connect
        this.wsClient.connect();
    }

    setupHandlers() {
        // --- Connection events ---
        this.wsClient.on('connected', () => {
            this._showConnectionStatus('Connected', true);
            setTimeout(() => this._hideConnectionStatus(), 2000);

            // Send session_start
            this.wsClient.send({
                type: 'session_start',
                user_id: this.userId,
                timestamp: Date.now() / 1000,
            });
        });

        this.wsClient.on('disconnected', () => {
            this._showConnectionStatus('Disconnected -- reconnecting...', false);
        });

        this.wsClient.on('reconnecting', (data) => {
            this._showConnectionStatus(
                `Reconnecting (${data.attempt}/${data.maxAttempts})...`,
                false
            );
        });

        this.wsClient.on('reconnect_failed', () => {
            this._showConnectionStatus('Connection lost. Refresh to retry.', false);
        });

        // --- AI response ---
        this.wsClient.on('response', (data) => {
            this.chat.addMessage(data.text || data.response_text, 'ai', {
                route: data.route || data.route_chosen,
                latency: data.latency_ms || data.latency,
                emotional_tone: data.adaptation?.emotional_tone,
            });
        });

        // --- State update (dashboard + embedding) ---
        this.wsClient.on('state_update', (data) => {
            this.dashboard.update(data);
            if (data.user_state_embedding_2d) {
                this.embeddingViz.update(data.user_state_embedding_2d);
            }
        });

        // --- Diary entry ---
        this.wsClient.on('diary_entry', (data) => {
            this.diary.addEntry(data.entry || data);
        });

        // --- Error from server ---
        this.wsClient.on('error_response', (data) => {
            this.chat.addMessage(
                `[Error] ${data.message || 'An error occurred'}`,
                'ai'
            );
        });
    }

    /** @private */
    _showConnectionStatus(text, isConnected) {
        if (!this.connectionBar) return;
        this.connectionBar.textContent = text;
        this.connectionBar.classList.add('visible');
        this.connectionBar.classList.toggle('connected', isConnected);
    }

    /** @private */
    _hideConnectionStatus() {
        if (!this.connectionBar) return;
        this.connectionBar.classList.remove('visible');
    }
}

// =========================================================================
// Bootstrap
// =========================================================================

document.addEventListener('DOMContentLoaded', () => {
    window.app = new I3App();
});
