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

        const msgEl = document.createElement('div');
        msgEl.className = `message ${sender}`;

        // Avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (sender === 'user') {
            avatar.textContent = 'U';
        } else {
            avatar.textContent = 'I\u00B3';
            // Shift avatar colour based on emotional tone if available
            if (metadata.emotional_tone !== undefined) {
                const tone = metadata.emotional_tone;
                // Warm amber (high tone) to cool blue (low tone)
                const r = Math.round(233 * tone + 58 * (1 - tone));
                const g = Math.round(69 * tone + 120 * (1 - tone));
                const b = Math.round(96 * tone + 210 * (1 - tone));
                avatar.style.background = `rgb(${r}, ${g}, ${b})`;
            }
        }

        // Body
        const body = document.createElement('div');
        body.className = 'message-body';

        // Text bubble
        const textEl = document.createElement('div');
        textEl.className = 'message-text';
        textEl.textContent = text;

        body.appendChild(textEl);

        // Metadata badges
        if (metadata.route || metadata.latency) {
            const meta = document.createElement('div');
            meta.className = 'message-meta';

            if (metadata.route) {
                const routeBadge = document.createElement('span');
                const isEdge = metadata.route.toLowerCase().includes('local') ||
                               metadata.route.toLowerCase().includes('edge') ||
                               metadata.route.toLowerCase().includes('slm');
                routeBadge.className = `meta-badge ${isEdge ? 'edge' : 'cloud'}`;
                routeBadge.textContent = isEdge ? 'Edge SLM' : 'Cloud LLM';
                meta.appendChild(routeBadge);
            }

            if (metadata.latency !== undefined) {
                const latBadge = document.createElement('span');
                latBadge.className = 'meta-badge';
                latBadge.textContent = `${Math.round(metadata.latency)}ms`;
                meta.appendChild(latBadge);
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

        const doSend = () => {
            const text = inputElement.value.trim();
            if (!text) return;

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
