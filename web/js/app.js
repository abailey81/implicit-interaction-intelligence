/**
 * Main application module for Implicit Interaction Intelligence (I3).
 *
 * Orchestrates the WebSocket connection, chat interface, dashboard,
 * embedding visualisation, keystroke monitor, and interaction diary.
 */

// Global slot for the most recent server-supplied reasoning trace.
// The explain panel reads this on every open; the WS handlers below
// update it on every response/response_done frame.  Initialised to null
// so the panel can detect "no message yet".
if (typeof window !== 'undefined' && window.__i3LastReasoningTrace === undefined) {
    window.__i3LastReasoningTrace = null;
}

/**
 * Iter 20 (2026-04-26): render the active conversation topic in the
 * chat hero pill.  Updates on every response / state_update frame.
 * Hides itself when the entity tracker has no anchored topic.  Pure
 * DOM, no framework.  Soft-fails silently if the pill isn't mounted
 * (e.g. on tabs other than chat).
 */
/**
 * Iter 26 (2026-04-26): render the topic-history breadcrumb in the
 * chat hero — up to 4 distinct ORG/TOPIC/PLACE entities the user
 * has previously discussed in this session, excluding the current
 * active topic.
 */
function renderTopicHistory(history) {
    const root = document.getElementById('topic-history-breadcrumb');
    if (!root) return;
    const list = root.querySelector('.topic-history-list');
    if (!list) return;
    if (!Array.isArray(history) || history.length === 0) {
        root.classList.add('is-hidden');
        list.textContent = '';
        return;
    }
    // Rebuild the inline list with one .crumb span per topic.  Pure
    // textContent — no innerHTML — to keep server surface forms safe.
    // Iter 28 (2026-04-26): each crumb is clickable — on click,
    // pre-fill the chat input with "back to {topic}" and submit, so
    // the user can navigate the conversation history with a tap.
    list.textContent = '';
    const inputEl = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    for (const t of history) {
        if (!t || !t.surface) continue;
        const span = document.createElement('span');
        span.className = 'crumb';
        span.textContent = String(t.surface);
        span.setAttribute('role', 'button');
        span.setAttribute('tabindex', '0');
        span.setAttribute('aria-label', `Pivot back to ${t.surface}`);
        const fire = () => {
            if (!inputEl || !sendBtn) return;
            const surf = String(t.surface).trim();
            if (!surf) return;
            inputEl.value = `back to ${surf}`;
            inputEl.focus();
            sendBtn.click();
        };
        span.addEventListener('click', fire);
        span.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') {
                ev.preventDefault();
                fire();
            }
        });
        list.appendChild(span);
    }
    root.classList.remove('is-hidden');
}

function renderActiveTopic(topic) {
    const pill = document.getElementById('active-topic-pill');
    if (!pill) return;
    if (!topic || !topic.canonical) {
        pill.classList.add('is-hidden');
        return;
    }
    const label = pill.querySelector('.active-topic-label');
    const value = pill.querySelector('.active-topic-value');
    if (value) {
        // SEC: textContent (NEVER innerHTML) — server-supplied surface
        // forms could in theory contain HTML.
        value.textContent = String(topic.surface || topic.canonical || '—');
    }
    if (label) {
        const kind = String(topic.kind || '').toLowerCase();
        label.textContent = (
            kind === 'org' ? 'Topic · org'
            : kind === 'person' ? 'Topic · person'
            : kind === 'place' ? 'Topic · place'
            : 'Topic'
        );
    }
    if (topic.user_anchored) {
        pill.classList.add('is-anchored');
    } else {
        pill.classList.remove('is-anchored');
    }
    pill.classList.remove('is-hidden');
}

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
                // SEC: Cap the buffer so a long composition session cannot
                // cause unbounded memory growth. We only ever transmit the
                // last 50 timings anyway (see getCompositionMetrics).
                if (this.keyTimings.length > 500) {
                    this.keyTimings.splice(0, this.keyTimings.length - 500);
                }
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
        // SEC: Reject non-object payloads.
        if (!entry || typeof entry !== 'object') return;

        // SEC: Helper — coerce arbitrary server values to a length-capped string.
        const safeStr = (v, max = 200) => {
            if (v === null || v === undefined) return '';
            const s = String(v);
            return s.length > max ? s.slice(0, max) : s;
        };

        const el = document.createElement('div');
        el.className = 'diary-entry';

        // Timestamp
        const timeEl = document.createElement('span');
        timeEl.className = 'entry-time';
        // SEC: Coerce timestamp to number; reject NaN.
        const tsNum = Number(entry.timestamp);
        const d = Number.isFinite(tsNum) ? new Date(tsNum * 1000) : new Date();
        timeEl.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Content
        const contentEl = document.createElement('span');
        contentEl.className = 'entry-content';
        // SEC: textContent is the only DOM sink; server-supplied summary/text
        // is rendered as plain text and cannot inject HTML.
        contentEl.textContent =
            safeStr(entry.summary) || safeStr(entry.text) || 'State change detected';

        // Tags
        if (entry.emotion) {
            const tag = document.createElement('span');
            tag.className = 'entry-tag';
            tag.textContent = safeStr(entry.emotion, 50);
            contentEl.appendChild(tag);
        }

        if (Array.isArray(entry.topics) && entry.topics.length > 0) {
            for (const topic of entry.topics.slice(0, 2)) {
                const tag = document.createElement('span');
                tag.className = 'entry-tag';
                // SEC: Topics are server-controlled — textContent only, capped.
                tag.textContent = safeStr(topic, 50);
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
// StateBadge -- renders the live discrete user-state label in the nav
// =========================================================================

/**
 * Apply a {state, confidence, secondary_state, contributing_signals}
 * payload to the .state-badge element.  Pure DOM; safe to call from
 * both response and state_badge handlers (idempotent).
 *
 * @param {object} payload  Server-shaped state label.
 */
function renderStateBadge(payload) {
    const el = document.getElementById('state-badge');
    if (!el || !payload || typeof payload !== 'object') return;

    // Whitelist the state name → CSS class to avoid CSS injection via
    // a crafted payload.
    const ALLOWED = new Set([
        'calm', 'focused', 'stressed', 'tired', 'distracted', 'warming up',
    ]);
    const stateRaw = String(payload.state || '').toLowerCase();
    const state = ALLOWED.has(stateRaw) ? stateRaw : 'warming up';
    const cssClass = state.replace(' ', '-');  // "warming up" → "warming-up"

    const confNum = Number(payload.confidence);
    const conf = Number.isFinite(confNum) ? Math.max(0, Math.min(1, confNum)) : 0;

    el.classList.remove('calm', 'focused', 'stressed', 'tired', 'distracted', 'warming-up');
    el.classList.add(cssClass);
    el.textContent = `${state} ${conf.toFixed(2)}`;

    // Tooltip = top contributing signals, joined by ' · '.
    const signals = Array.isArray(payload.contributing_signals)
        ? payload.contributing_signals.slice(0, 3).map((s) => String(s))
        : [];
    const secondary = payload.secondary_state ? String(payload.secondary_state) : '';
    let tip = signals.length ? signals.join(' · ') : 'no strong signals';
    if (secondary && secondary !== state) {
        tip += ` (runner-up: ${secondary})`;
    }
    el.setAttribute('title', tip);
}

// =========================================================================
// IdentityLock -- typing-biometric continuous-auth pill in the nav
// =========================================================================
//
// Cites Monrose & Rubin (1997) and Killourhy & Maxion (2009) -- the
// templating + matching logic lives server-side in
// i3.biometric.keystroke_auth.  This module is the visible HMI handle.

/**
 * Apply a server-shaped BiometricMatch payload to the Identity Lock pill.
 * Pure DOM; safe to call from response / response_done / state_update
 * handlers (idempotent).
 *
 * @param {object} payload  Server-shaped biometric state.
 */
function renderBiometric(payload) {
    const el = document.getElementById('identity-lock');
    const txt = document.getElementById('lock-text');
    const dotsHost = document.getElementById('lock-progress');
    if (!el || !txt) return;
    if (!payload || typeof payload !== 'object') return;

    // Whitelist the state name → CSS class to prevent CSS injection.
    const ALLOWED = new Set([
        'unregistered', 'registering', 'registered', 'verifying', 'mismatch',
    ]);
    const stateRaw = String(payload.state || 'unregistered').toLowerCase();
    const state = ALLOWED.has(stateRaw) ? stateRaw : 'unregistered';

    const confNum = Number(payload.confidence);
    const conf = Number.isFinite(confNum) ? Math.max(0, Math.min(1, confNum)) : 0;
    const isOwner = !!payload.is_owner;
    const driftAlert = !!payload.drift_alert;
    const progress = Math.max(0, Math.min(5, Number(payload.enrolment_progress) || 0));
    const target = Math.max(1, Math.min(20, Number(payload.enrolment_target) || 5));

    // Reset all classes, then apply the active one.
    el.classList.remove(
        'unregistered', 'registering', 'registered', 'mismatch',
        'drift', 'is-owner',
    );
    if (state === 'verifying') {
        el.classList.add('registered');
        if (isOwner) el.classList.add('is-owner');
    } else {
        el.classList.add(state);
        if (state === 'registered' && isOwner) el.classList.add('is-owner');
    }
    if (driftAlert) {
        el.classList.add('drift');
        // Brief shake animation on the rising-edge of a drift event.
        if (window.__i3LastBiometricDrift !== true) {
            el.classList.add('shake');
            setTimeout(() => el.classList.remove('shake'), 350);
        }
    }
    window.__i3LastBiometricDrift = driftAlert;

    // Text per the brief.
    let label;
    switch (state) {
        case 'unregistered':
            label = 'register';
            break;
        case 'registering':
            label = `learning · ${progress}/${target}`;
            break;
        case 'registered':
        case 'verifying':
            if (driftAlert) {
                label = `⚠ drift · ${conf.toFixed(2)}`;
            } else if (isOwner) {
                label = `You · ${conf.toFixed(2)}`;
            } else {
                label = `verifying · ${conf.toFixed(2)}`;
            }
            break;
        case 'mismatch':
            label = `⚠ not you · ${conf.toFixed(2)}`;
            break;
        default:
            label = state;
    }
    txt.textContent = label;

    // Tooltip = the explanatory notes string.
    const notes = String(payload.notes || '');
    el.setAttribute('title',
        notes || 'Typing biometric authentication (Monrose & Rubin 1997, Killourhy & Maxion 2009)');

    // Light up the registration progress dots (5 total).
    if (dotsHost) {
        const dots = dotsHost.querySelectorAll('.lock-dot');
        dots.forEach((d, i) => {
            if (i < progress) {
                d.classList.add('filled');
            } else {
                d.classList.remove('filled');
            }
        });
    }

    // Stash for the explain panel / profile tab to consume.
    window.__i3LastBiometric = payload;
    try {
        window.dispatchEvent(new CustomEvent(
            'i3:biometric',
            { detail: payload },
        ));
    } catch (_e) { /* noop */ }
}

// =========================================================================
// AccessibilityUI -- strip + body class + toggle button
// =========================================================================

/**
 * Apply an accessibility-mode payload (matching the controller's
 * AccessibilityModeState dict) to the visible UI: toggles the strip,
 * the body.accessibility-mode class, and the [A] toggle's pressed
 * state.  Reads tts_rate_multiplier into a window global so the TTS
 * player module can pick it up.
 *
 * @param {object} payload  Server-shaped accessibility state.
 */
function applyAccessibilityState(payload) {
    if (!payload || typeof payload !== 'object') return;

    const active = !!payload.active;
    const reason = String(payload.reason || '');

    document.body.classList.toggle('accessibility-mode', active);

    const strip = document.getElementById('access-strip');
    const reasonEl = document.getElementById('access-strip-reason');
    if (strip) {
        strip.hidden = !active;
        // Only update the reason text when the strip is visible to
        // avoid spurious DOM writes between turns.
        if (active && reasonEl) {
            reasonEl.textContent = reason || 'sustained motor-difficulty signals';
        }
    }

    const toggle = document.getElementById('access-toggle');
    if (toggle) {
        toggle.classList.toggle('active', active);
        toggle.setAttribute('aria-pressed', active ? 'true' : 'false');
    }

    // Expose the TTS rate multiplier on the window so the speech
    // module (loaded later) can read it without a fragile import.
    const rateNum = Number(payload.tts_rate_multiplier);
    window.__i3TtsRateMultiplier = Number.isFinite(rateNum) ? rateNum : 1.0;

    // Stash the latest accessibility payload so other modules can
    // query it without a round-trip.
    window.__i3LastAccessibility = payload;
}

// =========================================================================
// I3App -- main application orchestrator
// =========================================================================

class I3App {
    constructor() {
        this.userId = 'demo_user';
        this.sessionId = null;

        // Determine WebSocket URL
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/${this.userId}`;

        // Initialise modules
        this.wsClient = new WebSocketClient(wsUrl);
        this.keystrokeMonitor = new KeystrokeMonitor(this.wsClient);

        // Voice-prosody flagship #2 — pure-WebAudio prosodic feature
        // extractor.  Mic is OFF by default; the user clicks the
        // toggle button to enable.  ``window.I3VoiceProsody.mount``
        // is defined in ``web/js/voice_prosody.js`` (loaded ahead of
        // ``app.js`` in index.html) so the global is always present.
        // We mount the UI lazily inside ``setupHandlers`` once the
        // chat input is in the DOM.
        this.voiceProsody = null;

        this.chat = new ChatInterface(
            document.getElementById('chat-messages'),
            this.wsClient,
            this.keystrokeMonitor
        );

        // Apple-style layout no longer wraps everything in a single
        // ``#dashboard`` element — pass null and let Dashboard pick
        // its sections up by id from the document root.  The legacy
        // wrapper is still found if it's there.
        this.dashboard = new Dashboard(
            document.getElementById('dashboard')
        );

        this.embeddingViz = new EmbeddingViz(
            document.getElementById('embedding-canvas')
        );

        this.diary = new DiaryPanel(
            document.getElementById('diary-panel') || document.getElementById('diary')
        );

        // Wire up chat input
        const inputEl = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        this.chat.setupInput(inputEl, sendBtn);
        this.chat.setTypingIndicator(document.getElementById('typing-indicator'));

        // Suggestion-chip wiring (apple26): tap to pre-fill + send.
        // Hides the chip strip after the first user-sent message so
        // the chat hero doesn't keep them around once the demo's
        // started.
        const chipStrip = document.getElementById('suggestion-chips');
        if (chipStrip && inputEl && sendBtn) {
            chipStrip.querySelectorAll('.suggestion-chip').forEach((btn) => {
                btn.addEventListener('click', () => {
                    const prompt = btn.getAttribute('data-prompt') || '';
                    if (!prompt) return;
                    inputEl.value = prompt;
                    inputEl.focus();
                    sendBtn.click();
                });
            });
            const hideChips = () => {
                chipStrip.classList.add('is-hidden');
                sendBtn.removeEventListener('click', hideChips);
                inputEl.removeEventListener('keydown', onEnterDown);
            };
            const onEnterDown = (ev) => {
                if (ev.key === 'Enter' && (inputEl.value || '').trim()) {
                    hideChips();
                }
            };
            sendBtn.addEventListener('click', hideChips);
            inputEl.addEventListener('keydown', onEnterDown);
        }

        // Mount the voice-prosody mic toggle next to the send button.
        // The monitor stays disabled until the user clicks; chat.js
        // reads from ``this.voiceProsody`` on every send.
        try {
            if (typeof window !== 'undefined' &&
                window.I3VoiceProsody &&
                typeof window.I3VoiceProsody.mount === 'function') {
                const wrap = inputEl ? inputEl.closest('.chat-input-wrap')
                                     : null;
                this.voiceProsody = window.I3VoiceProsody.mount({
                    host: wrap || document.body,
                    sendButton: sendBtn,
                });
                this.chat.setVoiceProsody(this.voiceProsody);
            }
        } catch (e) {
            // Voice prosody is non-essential — never block chat init.
            console.warn('[I3] Voice prosody mount failed:', e);
        }

        // Mount the gaze-capture camera toggle (vision fine-tuning
        // flagship).  Off by default; chat.js reads from
        // ``this.gazeCapture`` on every send and attaches the
        // aggregate gaze_features dict to the WS frame when enabled.
        try {
            if (typeof window !== 'undefined' &&
                window.I3GazeCapture &&
                typeof window.I3GazeCapture.mount === 'function') {
                const wrap = inputEl ? inputEl.closest('.chat-input-wrap')
                                     : null;
                this.gazeCapture = window.I3GazeCapture.mount({
                    host: wrap || document.body,
                    sendButton: sendBtn,
                });
                this.chat.setGazeCapture(this.gazeCapture);
            }
        } catch (e) {
            // Gaze capture is non-essential — never block chat init.
            console.warn('[I3] Gaze capture mount failed:', e);
        }

        // Connection status bar
        this.connectionBar = document.getElementById('connection-bar');

        // Set up event handlers
        this.setupHandlers();

        // SEC: Cleanly close the WebSocket on page unload to prevent the
        // browser from holding a half-open socket and to stop reconnection
        // logic from firing during navigation.
        window.addEventListener('beforeunload', () => {
            try {
                this.wsClient.disconnect();
            } catch (e) { /* swallow — page is going away */ }
        });

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

        // --- AI response (single-frame retrieval / tool / OOD path) ---
        this.wsClient.on('response', (data) => {
            // Stash the latest reasoning trace so the explain panel can
            // render it on demand without a second round-trip.
            if (data && data.reasoning_trace) {
                window.__i3LastReasoningTrace = data.reasoning_trace;
                try {
                    window.dispatchEvent(new CustomEvent(
                        'i3:reasoning-trace',
                        { detail: data.reasoning_trace }
                    ));
                } catch (_e) { /* noop */ }
            }
            this.chat.addMessage(data.text || data.response_text, 'ai', {
                route: data.route || data.route_chosen,
                latency: data.latency_ms || data.latency,
                emotional_tone: data.adaptation?.emotional_tone,
                response_path: data.response_path,
                retrieval_score: data.retrieval_score,
                adaptation_changes: data.adaptation_changes,
                affect_shift: data.affect_shift,
                critique: data.critique,
                coreference_resolution: data.coreference_resolution,
                multimodal: data.multimodal,
                routing_decision: data.routing_decision,
                privacy_budget: data.privacy_budget,
                // Iteration 12 (2026-04-26): pass the FULL adaptation
                // snapshot + biometric verdict + explain-plan through
                // to chat.js so the per-bubble Decision Trace and
                // Reasoning chain expanders can render without a
                // second API call.
                adaptation: data.adaptation,
                biometric: data.biometric,
                explain_plan: data.explain_plan,
            });
            if (data.active_topic !== undefined) {
                renderActiveTopic(data.active_topic);
            }
            if (data.topic_history !== undefined) {
                renderTopicHistory(data.topic_history);
            }
            // Forward the privacy_budget snapshot to the Privacy tab
            // immediately on response so the counters reflect the
            // call this turn fired.
            if (data.privacy_budget) {
                this._updatePrivacyBudget(data.privacy_budget);
            }
            if (data.routing_decision) {
                this._renderRoutingDecision(data.routing_decision);
            }
            // Update the live state badge on every response so it
            // tracks both per-turn (here) and mid-typing (state_badge
            // frames).
            if (data.user_state_label) {
                renderStateBadge(data.user_state_label);
            }
            if (data.accessibility) {
                applyAccessibilityState(data.accessibility);
            }
            if (data.biometric) {
                renderBiometric(data.biometric);
            }
            // Light up the header pipeline ribbon per the path the
            // server actually took.  retrieval → green, slm → red
            // glow, ood → neutral.
            this._pulsePipeline(data.response_path || 'retrieval');
        });

        // --- AI streaming token (SLM-path per-token delta) ---
        this.wsClient.on('token', (data) => {
            if (!data || typeof data.delta !== 'string') return;
            this.chat.appendTokenDelta(data.delta);
        });

        // --- AI streaming complete (server-confirmed final text) ---
        this.wsClient.on('response_done', (data) => {
            if (data && data.reasoning_trace) {
                window.__i3LastReasoningTrace = data.reasoning_trace;
                try {
                    window.dispatchEvent(new CustomEvent(
                        'i3:reasoning-trace',
                        { detail: data.reasoning_trace }
                    ));
                } catch (_e) { /* noop */ }
            }
            this.chat.finaliseStreaming(data.text || data.response_text, {
                route: data.route || data.route_chosen,
                latency: data.latency_ms || data.latency,
                emotional_tone: data.adaptation?.emotional_tone,
                response_path: data.response_path,
                retrieval_score: data.retrieval_score,
                adaptation_changes: data.adaptation_changes,
                affect_shift: data.affect_shift,
                critique: data.critique,
                coreference_resolution: data.coreference_resolution,
                multimodal: data.multimodal,
                routing_decision: data.routing_decision,
                privacy_budget: data.privacy_budget,
                // Iteration 12 (2026-04-26): pass the FULL adaptation
                // snapshot + biometric verdict + explain-plan through
                // to chat.js so the per-bubble Decision Trace and
                // Reasoning chain expanders can render without a
                // second API call.
                adaptation: data.adaptation,
                biometric: data.biometric,
                explain_plan: data.explain_plan,
            });
            if (data.active_topic !== undefined) {
                renderActiveTopic(data.active_topic);
            }
            if (data.topic_history !== undefined) {
                renderTopicHistory(data.topic_history);
            }
            if (data.privacy_budget) {
                this._updatePrivacyBudget(data.privacy_budget);
            }
            if (data.routing_decision) {
                this._renderRoutingDecision(data.routing_decision);
            }
            if (data.user_state_label) {
                renderStateBadge(data.user_state_label);
            }
            if (data.accessibility) {
                applyAccessibilityState(data.accessibility);
            }
            if (data.biometric) {
                renderBiometric(data.biometric);
            }
            this._pulsePipeline(data.response_path || 'slm');
        });

        // --- Biometric one-shot rising-edge event ---
        // Fires once per state transition (registered, drift_alert,
        // mismatch).  We log in DevTools so the demo is visibly
        // auditable, and apply a brief celebratory flash on
        // 'registered' or a shake on drift / mismatch.
        this.wsClient.on('biometric_event', (data) => {
            if (!data || typeof data !== 'object') return;
            const evt = String(data.event || '');
            // eslint-disable-next-line no-console
            console.log('[I3 biometric_event]', evt, data);
            const lock = document.getElementById('identity-lock');
            if (!lock) return;
            if (evt === 'registered') {
                lock.classList.add('flash-registered');
                lock.setAttribute('title', 'Registered! Future turns will be verified against this rhythm.');
                setTimeout(() => lock.classList.remove('flash-registered'), 2000);
            } else if (evt === 'drift_alert' || evt === 'mismatch') {
                lock.classList.add('shake');
                setTimeout(() => lock.classList.remove('shake'), 350);
            }
        });

        // --- Live State Badge mid-typing update ---
        this.wsClient.on('state_badge', (data) => {
            renderStateBadge(data);
        });

        // --- Accessibility-mode rising / falling edge ---
        this.wsClient.on('accessibility_change', (data) => {
            applyAccessibilityState(data);
        });

        // --- State update (dashboard + embedding) ---
        this.wsClient.on('state_update', (data) => {
            this.dashboard.update(data);
            if (data.user_state_embedding_2d) {
                this.embeddingViz.update(data.user_state_embedding_2d);
            }
            if (data.user_state_label) {
                renderStateBadge(data.user_state_label);
            }
            if (data.accessibility) {
                applyAccessibilityState(data.accessibility);
            }
            // Iter 51: re-broadcast as a DOM event so the new
            // huawei_tabs.js can pick up personal_facts / intent_result
            // for the Personal Facts and Intent panels.
            try {
                document.dispatchEvent(new CustomEvent('i3:state_update', {
                    detail: data,
                }));
            } catch (_e) { /* never break dashboard on listener wiring */ }
            if (data.biometric) {
                renderBiometric(data.biometric);
            }
            // Iter 20 (2026-04-26): active-topic pill.
            if (data.active_topic !== undefined) {
                renderActiveTopic(data.active_topic);
            }
            if (data.topic_history !== undefined) {
                renderTopicHistory(data.topic_history);
            }
            // Cloud-route consent + privacy-budget surfaces.  The WS
            // layer ships a privacy_budget snapshot on every turn so
            // the counters update live without a separate REST call.
            if (data.privacy_budget) {
                this._updatePrivacyBudget(data.privacy_budget);
            }
            if (data.routing_decision) {
                this._renderRoutingDecision(data.routing_decision);
            }
        });

        // --- Session id from server (used by the accessibility toggle) ---
        this.wsClient.on('session_started', (data) => {
            if (data && data.session_id) {
                this.sessionId = data.session_id;
            }
        });

        // Wire up the Identity Lock pill: clicking opens / toggles the
        // popover, and each action button hits the corresponding REST
        // endpoint (or jumps to the Profile tab for "View profile").
        const lockBtn = document.getElementById('identity-lock');
        const lockPop = document.getElementById('identity-lock-popover');
        if (lockBtn && lockPop) {
            lockBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const open = !lockPop.hidden;
                lockPop.hidden = open;
                lockBtn.setAttribute('aria-expanded', open ? 'false' : 'true');
            });
            // Close on outside click.
            document.addEventListener('click', (e) => {
                if (!lockPop.hidden && !lockPop.contains(e.target) && e.target !== lockBtn) {
                    lockPop.hidden = true;
                    lockBtn.setAttribute('aria-expanded', 'false');
                }
            });
            lockPop.querySelectorAll('.lock-action').forEach((btn) => {
                btn.addEventListener('click', async () => {
                    lockPop.hidden = true;
                    lockBtn.setAttribute('aria-expanded', 'false');
                    const action = btn.dataset.action;
                    if (action === 'reregister') {
                        await this._postBiometric('reset');
                        await this._postBiometric('force-register');
                    } else if (action === 'force') {
                        await this._postBiometric('force-register');
                    } else if (action === 'profile') {
                        // Jump to the Profile tab.
                        const profileLink = document.querySelector('a[data-tab="profile"]');
                        if (profileLink) profileLink.click();
                    }
                });
            });
        }

        // Wire up the accessibility-mode manual toggle in the nav.
        const accessToggle = document.getElementById('access-toggle');
        if (accessToggle) {
            accessToggle.addEventListener('click', () => {
                const isActive = document.body.classList.contains('accessibility-mode');
                this._postAccessibilityToggle(isActive ? false : true);
            });
        }
        // Wire up the [exit] button inside the strip — issues
        // force=false so the mode stays off until the auto rule
        // *and* the manual override are both cleared.
        const exitBtn = document.getElementById('access-strip-exit');
        if (exitBtn) {
            exitBtn.addEventListener('click', () => {
                this._postAccessibilityToggle(false);
            });
        }

        // Wire up the cloud-route consent toggle in the nav.  Default
        // OFF; flipping POSTs to /api/routing/cloud-consent/{user_id}.
        // We also bootstrap the live counter from
        // /api/routing/budget/{user_id}/{session_id} on page load so
        // the pill text reflects the server-side state on refresh.
        const cloudToggle = document.getElementById('cloud-consent-toggle');
        if (cloudToggle) {
            cloudToggle.addEventListener('click', () => {
                const currentlyOn = cloudToggle.classList.contains('on');
                this._postCloudConsent(!currentlyOn);
            });
        }
        // --- Routing decision frames (per-turn cloud/edge audit) ---
        this.wsClient.on('routing_decision', (data) => {
            this._renderRoutingDecision(data);
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

    /**
     * Hit a biometric action endpoint and apply the server's response.
     * @param {string} action  one of 'reset', 'force-register'.
     * @private
     */
    async _postBiometric(action) {
        try {
            const url = `/api/biometric/${encodeURIComponent(this.userId)}/${action}`;
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const payload = await res.json();
            renderBiometric(payload);
        } catch (e) {
            // Local fallback so the click still does *something* visible
            // when the server isn't reachable.
            // eslint-disable-next-line no-console
            console.warn('[I3] biometric ' + action + ' failed:', e);
        }
    }

    /**
     * Issue a POST /api/accessibility/{user_id}/toggle and apply the
     * server's response.  Falls back to a local toggle when the
     * fetch fails so the UI is never left out-of-sync with the body
     * class.
     * @param {boolean|null} force  true=on, false=off, null=auto.
     * @private
     */
    async _postAccessibilityToggle(force) {
        try {
            const res = await fetch(
                `/api/accessibility/${encodeURIComponent(this.userId)}/toggle`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'same-origin',
                    body: JSON.stringify({
                        session_id: this.sessionId || 'manual',
                        force: force,
                    }),
                }
            );
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const payload = await res.json();
            applyAccessibilityState(payload);
        } catch (e) {
            // Local fallback so the click still does something visible.
            applyAccessibilityState({
                active: !!force,
                activated_this_turn: !!force,
                deactivated_this_turn: !force,
                confidence: 1.0,
                reason: force ? 'manual override' : 'manual exit',
                sentence_cap: force ? 1 : 3,
                simplify_vocab: !!force,
                tts_rate_multiplier: force ? 0.6 : 1.0,
                font_scale: force ? 1.25 : 1.0,
            });
        }
    }

    /**
     * POST /api/routing/cloud-consent/{user_id} — flip the cloud-route
     * consent flag.  Updates the toggle button + the privacy-budget
     * status pill.  Falls back to a local toggle when the server is
     * unreachable so the click is never a no-op.
     * @private
     */
    async _postCloudConsent(enabled) {
        const toggleEl = document.getElementById('cloud-consent-toggle');
        const apply = (state) => {
            if (!toggleEl) return;
            toggleEl.classList.toggle('on', !!state);
            toggleEl.classList.toggle('off', !state);
            toggleEl.setAttribute('aria-pressed', String(!!state));
            const txt = toggleEl.querySelector('.cloud-text');
            if (txt) txt.textContent = state ? 'cloud · on' : 'cloud · off';
            const status = document.getElementById('privacy-budget-status');
            if (status) {
                status.classList.toggle('on', !!state);
                status.classList.toggle('off', !state);
                status.innerHTML = state
                    ? 'Cloud route consent: <strong>ON</strong> (you opted in for this session)'
                    : 'Cloud route consent: <strong>OFF</strong>';
            }
        };
        try {
            const res = await fetch(
                `/api/routing/cloud-consent/${encodeURIComponent(this.userId)}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'same-origin',
                    body: JSON.stringify({ enabled: !!enabled }),
                }
            );
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const payload = await res.json();
            apply(!!payload.enabled);
        } catch (e) {
            // eslint-disable-next-line no-console
            console.warn('[I3] cloud-consent toggle failed:', e);
            apply(!!enabled);
        }
    }

    /**
     * Update the cloud-route toggle text + the Privacy-tab counters
     * from a privacy_budget snapshot dict (the shape produced by
     * :class:`PrivacyBudgetSnapshot`).
     * @private
     */
    _updatePrivacyBudget(snapshot) {
        if (!snapshot || typeof snapshot !== 'object') return;
        // Top-of-page toggle: when consent is on, show remaining call
        // budget directly on the pill so the user can see they have N
        // cloud calls left without opening the Privacy tab.
        const toggleEl = document.getElementById('cloud-consent-toggle');
        if (toggleEl) {
            const consent = !!snapshot.consent_enabled;
            toggleEl.classList.toggle('on', consent);
            toggleEl.classList.toggle('off', !consent);
            toggleEl.setAttribute('aria-pressed', String(consent));
            const txt = toggleEl.querySelector('.cloud-text');
            const remaining = Number(snapshot.budget_remaining_calls ?? 0);
            if (txt) {
                txt.textContent = consent
                    ? `cloud · on (${Number.isFinite(remaining) ? remaining : '—'} calls left)`
                    : 'cloud · off';
            }
        }
        // Privacy panel counters.
        const callsEl = document.getElementById('privacy-counter-calls');
        const callsSubEl = document.getElementById('privacy-counter-calls-sub');
        if (callsEl) callsEl.textContent = String(snapshot.cloud_calls_total ?? 0);
        if (callsSubEl && snapshot.cloud_calls_max) {
            callsSubEl.textContent = `of ${snapshot.cloud_calls_max} budget`;
        }
        const piiEl = document.getElementById('privacy-counter-pii');
        if (piiEl) piiEl.textContent = String(snapshot.pii_redactions_total ?? 0);
        const bytesEl = document.getElementById('privacy-counter-bytes');
        const bytesSubEl = document.getElementById('privacy-counter-bytes-sub');
        if (bytesEl) {
            bytesEl.textContent = String(snapshot.bytes_transmitted_total ?? 0);
        }
        if (bytesSubEl && snapshot.bytes_transmitted_max) {
            bytesSubEl.textContent = `of ${Number(snapshot.bytes_transmitted_max).toLocaleString()} byte budget`;
        }
        const redEl = document.getElementById('privacy-counter-redacted');
        if (redEl) redEl.textContent = String(snapshot.bytes_redacted_total ?? 0);

        // Status pill in the panel.
        const status = document.getElementById('privacy-budget-status');
        if (status) {
            const consent = !!snapshot.consent_enabled;
            status.classList.toggle('on', consent);
            status.classList.toggle('off', !consent);
            status.innerHTML = consent
                ? 'Cloud route consent: <strong>ON</strong> (you opted in for this session)'
                : 'Cloud route consent: <strong>OFF</strong>';
        }

        // Per-category bar updates.
        const cats = snapshot.sensitive_categories || {};
        const total = Math.max(
            1,
            Object.values(cats).reduce(
                (a, b) => a + (Number(b) || 0),
                0
            )
        );
        const rows = document.querySelectorAll('.privacy-budget-cat-row');
        rows.forEach((row) => {
            const k = row.getAttribute('data-cat');
            const n = Number((cats || {})[k] || 0);
            const fill = row.querySelector('.cat-bar-fill');
            const count = row.querySelector('.cat-count');
            if (fill) {
                const pct = total > 0 ? Math.min(100, (n / total) * 100) : 0;
                fill.style.width = `${pct}%`;
            }
            if (count) count.textContent = String(n);
        });
    }

    /**
     * Render a per-turn routing decision dict on the existing routing
     * surfaces.  Updates the routing tab's scatter plot and the
     * routing-decision label.
     * @private
     */
    _renderRoutingDecision(data) {
        try {
            if (window.RoutingScatter && typeof window.RoutingScatter.append === 'function') {
                window.RoutingScatter.append(data);
            }
        } catch (e) {
            // eslint-disable-next-line no-console
            console.warn('[I3] routing-scatter append failed:', e);
        }
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

    /**
     * Pulse the pipeline-ribbon LEDs in the header to indicate which
     * on-device components just carried the turn.  Every response walks
     * through encoder → adaptation → router; the last step is either
     * retrieval (common prompt) or SLM generation (novel prompt).
     * @private
     * @param {string} route
     */
    _pulsePipeline(responsePath) {
        const ribbon = document.getElementById('pipeline-ribbon');
        if (!ribbon) return;
        const dots = ribbon.querySelectorAll('.pipe-dot');
        // Every response walks through encoder + adaptation + router.
        // The last stage depends on the response_path the server
        // actually took.
        const activeStages = new Set(['encoder', 'adapt', 'router']);
        const path = String(responsePath || '').toLowerCase();
        if (path === 'retrieval' || path === 'retrieval_borderline') {
            activeStages.add('retrieval');
        } else if (path === 'slm') {
            activeStages.add('slm');
        } else if (path === 'ood') {
            // Out-of-distribution: no final-stage match; leave both dim.
        } else {
            activeStages.add('retrieval');
        }
        dots.forEach((d) => {
            d.classList.remove('active', 'ok');
            const stage = d.dataset.stage;
            if (activeStages.has(stage)) {
                d.classList.add('active');
                setTimeout(() => {
                    d.classList.remove('active');
                    d.classList.add('ok');
                }, 500);
            }
        });
    }
}

// =========================================================================
// Bootstrap
// =========================================================================

// Populate the on-device-stack panel from /api/stack.  The panel exists
// to surface the "from scratch" narrative (the custom AdaptiveSLM, TCN
// encoder, LinUCB bandit, ONNX export) so a reader can see at a glance
// that the demo ships without HuggingFace transformers or any other
// heavy framework as the generation backbone.
async function populateStackPanel() {
    try {
        const res = await fetch('/api/stack', {
            credentials: 'same-origin',
            cache: 'no-store',
        });
        if (!res.ok) return;
        const s = await res.json();

        const slm = s.slm || {};
        const enc = s.encoder || {};
        const rtr = s.router || {};
        const dev = s.device || {};

        const setText = (id, text, title) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.textContent = text || '—';
            if (title) el.setAttribute('title', title);
        };

        if (slm.loaded) {
            setText(
                'stack-slm',
                `${slm.params_human || '—'} · ${slm.d_model || '?'}d · ${slm.n_layers || '?'}L · ${slm.vocab_size || '?'} vocab`,
                slm.implementation || '',
            );
            setText(
                'stack-slm-meta',
                `${slm.implementation || 'custom transformer'} · 0 HF deps`,
            );
        } else {
            setText('stack-slm', 'rule-based fallback');
            setText('stack-slm-meta', 'train-slm has not produced a checkpoint yet');
        }

        if (enc.loaded) {
            setText(
                'stack-encoder',
                `${enc.input_dim || '?'}→${enc.embedding_dim || '?'} · k=${enc.kernel_size || '?'} · ${(enc.dilations || []).join(',') || '?'}`,
                enc.implementation || '',
            );
            setText(
                'stack-encoder-meta',
                `${enc.implementation || 'TCN'} · window=${enc.window_size || '?'}`,
            );
        } else {
            setText('stack-encoder', 'zero-embedding fallback');
        }

        if (rtr.loaded) {
            setText(
                'stack-router',
                `${rtr.name || 'bandit'} · arms=${(rtr.arms || []).length}`,
                rtr.implementation || '',
            );
            setText(
                'stack-router-meta',
                rtr.implementation || 'contextual bandit',
            );
        }

        setText(
            'stack-device',
            `${dev.name || 'CPU'} · ${(dev.vram_gb || 0).toFixed ? dev.vram_gb.toFixed(1) : dev.vram_gb} GB`,
        );
        const edge = s.edge || {};
        setText(
            'stack-device-meta',
            edge.onnx_exported
                ? `ONNX exported · ${edge.onnx_size_mb} MB · edge-ready`
                : 'edge-first · ONNX export pending',
        );

        const depsPill = document.getElementById('stack-deps');
        if (depsPill) {
            const total = (slm.hf_dependencies || 0) +
                (enc.hf_dependencies || 0) +
                (rtr.hf_dependencies || 0);
            depsPill.textContent = `${total} HF deps`;
        }
    } catch (e) { /* panel is decorative; swallow */ }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new I3App();
    populateStackPanel();
});
