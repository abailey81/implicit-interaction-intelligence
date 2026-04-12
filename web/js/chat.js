/**
 * Chat interface module.
 *
 * Renders user and AI messages with metadata badges (route, latency),
 * slide-in animations, and auto-scroll. Integrates with the keystroke
 * monitor for implicit signal capture.
 */

class ChatInterface {
    /**
     * @param {HTMLElement}      container        The .chat-messages container.
     * @param {WebSocketClient}  wsClient         WebSocket client for sending messages.
     * @param {KeystrokeMonitor} keystrokeMonitor Keystroke capture instance.
     */
    constructor(container, wsClient, keystrokeMonitor) {
        this.container = container;
        this.ws = wsClient;
        this.ksMonitor = keystrokeMonitor;
        this.messageCount = 0;
        this.typingIndicator = null;
    }

    /**
     * Append a message to the chat log.
     *
     * @param {string} text          Message content.
     * @param {string} sender        "user" or "ai".
     * @param {object} [metadata={}] Optional metadata: { route, latency }.
     */
    addMessage(text, sender, metadata = {}) {
        // Hide typing indicator
        this._hideTyping();

        // SEC: Coerce text to string. Server may occasionally return null/undefined
        // or non-string types; textContent assignment below would still be safe
        // (DOM coerces it), but explicit coercion makes the contract obvious.
        const safeText = (text === null || text === undefined) ? '' : String(text);

        // SEC: Whitelist sender to prevent CSS class injection via crafted metadata.
        const safeSender = (sender === 'user' || sender === 'ai') ? sender : 'ai';

        const msgEl = document.createElement('div');
        msgEl.className = `message ${safeSender}`;

        // Avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (safeSender === 'user') {
            avatar.textContent = 'U';
        } else {
            avatar.textContent = 'I\u00B3';
            // Shift avatar colour based on emotional tone if available
            if (metadata.emotional_tone !== undefined) {
                // SEC: Clamp tone to [0, 1] so that an attacker-controlled value
                // cannot push rgb() out of range or inject CSS via NaN/Infinity.
                const toneRaw = Number(metadata.emotional_tone);
                const tone = Number.isFinite(toneRaw)
                    ? Math.max(0, Math.min(1, toneRaw))
                    : 0.5;
                // Warm amber (high tone) to cool blue (low tone)
                const r = Math.round(233 * tone + 58 * (1 - tone));
                const g = Math.round(69 * tone + 120 * (1 - tone));
                const b = Math.round(96 * tone + 210 * (1 - tone));
                avatar.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
            }
        }

        // Body
        const body = document.createElement('div');
        body.className = 'message-body';

        // Text bubble
        const textEl = document.createElement('div');
        textEl.className = 'message-text';
        // SEC: textContent (NEVER innerHTML) — this is the AI/user text sink.
        // Even if the AI returns HTML/JS, it will be rendered as plain text.
        textEl.textContent = safeText;

        body.appendChild(textEl);

        // Metadata badges
        if (metadata.route || metadata.latency !== undefined) {
            const meta = document.createElement('div');
            meta.className = 'message-meta';

            if (metadata.route) {
                // SEC: Coerce to string and use textContent. We compute a fixed
                // label ("Edge SLM" / "Cloud LLM") so user-controlled route
                // strings are never written to the DOM verbatim.
                const routeStr = String(metadata.route).toLowerCase();
                const routeBadge = document.createElement('span');
                const isEdge = routeStr.includes('local') ||
                               routeStr.includes('edge') ||
                               routeStr.includes('slm');
                // SEC: Use classList (whitelist) instead of string-concat className
                routeBadge.classList.add('meta-badge');
                routeBadge.classList.add(isEdge ? 'edge' : 'cloud');
                routeBadge.textContent = isEdge ? 'Edge SLM' : 'Cloud LLM';
                meta.appendChild(routeBadge);
            }

            if (metadata.latency !== undefined) {
                // SEC: Coerce latency to number; reject NaN/Infinity.
                const latNum = Number(metadata.latency);
                if (Number.isFinite(latNum)) {
                    const latBadge = document.createElement('span');
                    latBadge.className = 'meta-badge';
                    latBadge.textContent = `${Math.round(latNum)}ms`;
                    meta.appendChild(latBadge);
                }
            }

            body.appendChild(meta);
        }

        msgEl.appendChild(avatar);
        msgEl.appendChild(body);

        this.container.appendChild(msgEl);
        this.messageCount++;

        // Auto-scroll to bottom
        this._scrollToBottom();
    }

    /**
     * Wire up the chat input field and send button.
     *
     * @param {HTMLInputElement} inputElement The text input field.
     * @param {HTMLButtonElement} sendButton  The send button.
     */
    setupInput(inputElement, sendButton) {
        this.inputEl = inputElement;

        // Attach keystroke monitor
        if (this.ksMonitor) {
            this.ksMonitor.attach(inputElement);
        }

        // SEC: Defensive client-side cap on outgoing message length.
        // Server is the source of truth for sanitisation/limits, but capping
        // here prevents accidentally sending megabyte payloads.
        const MAX_INPUT_CHARS = 4000;

        const doSend = () => {
            let text = inputElement.value.trim();
            if (!text) return;
            if (text.length > MAX_INPUT_CHARS) {
                // SEC: Truncate rather than reject so the user gets feedback.
                text = text.slice(0, MAX_INPUT_CHARS);
            }

            // Gather composition metrics before resetting
            const metrics = this.ksMonitor ? this.ksMonitor.getCompositionMetrics() : {};

            // Add user message to chat
            this.addMessage(text, 'user');

            // Send to server
            this.ws.send({
                type: 'message',
                text: text,
                timestamp: Date.now() / 1000,
                composition_metrics: metrics,
            });

            // Clear input and reset monitor
            inputElement.value = '';
            if (this.ksMonitor) {
                this.ksMonitor.reset();
            }

            // Show typing indicator
            this._showTyping();
        };

        // Enter to send
        inputElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                doSend();
            }
        });

        // Send button click
        if (sendButton) {
            sendButton.addEventListener('click', doSend);
        }
    }

    /**
     * Set up the typing indicator element.
     * @param {HTMLElement} indicator The .typing-indicator element.
     */
    setTypingIndicator(indicator) {
        this.typingIndicator = indicator;
    }

    /** @private */
    _showTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.classList.add('visible');
            this._scrollToBottom();
        }
    }

    /** @private */
    _hideTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.classList.remove('visible');
        }
    }

    /** @private */
    _scrollToBottom() {
        requestAnimationFrame(() => {
            this.container.scrollTop = this.container.scrollHeight;
        });
    }
}
