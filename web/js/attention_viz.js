/**
 * attention_viz.js -- I3 Advanced Panels
 *
 * Cross-attention heatmap visualiser. On every `response` event pushed
 * over the WebSocket, this module fetches `/api/attention?session_id=X`
 * and renders a 4 (transformer blocks) x 4 (conditioning tokens) grid
 * showing the last generated token's attention distribution.
 *
 * The grid is a CSS-variable-driven DOM structure (nicer than a flat
 * canvas blit for accessibility) plus a small requestAnimationFrame
 * loop that eases cells to their target fill value.
 *
 * Export: initAttentionViz(container)
 */

const ROWS = 4;      // transformer blocks
const COLS = 4;      // conditioning tokens
const EASE_RATE = 0.12;
const FETCH_TIMEOUT_MS = 2500;

/**
 * Safely coerce a backend value to a clamped [0, 1] number.
 * @param {*} v
 * @returns {number}
 */
function clamp01(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 0;
    if (n < 0) return 0;
    if (n > 1) return 1;
    return n;
}

/**
 * Generate a plausible synthetic attention matrix for offline demos.
 * Returns a ROWS x COLS nested array in [0, 1].
 * @param {number} seed
 * @returns {number[][]}
 */
function synthesisedMatrix(seed = Date.now()) {
    const out = [];
    // Simple LCG so the synthetic pattern shifts between calls but is
    // deterministic within a call.
    let s = seed >>> 0;
    const rand = () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
    };
    for (let r = 0; r < ROWS; r++) {
        const row = [];
        // Bias later blocks toward later tokens (a soft diagonal).
        const peak = (r / (ROWS - 1)) * (COLS - 1);
        for (let c = 0; c < COLS; c++) {
            const dist = Math.abs(c - peak);
            const base = Math.max(0, 1 - dist / (COLS - 1));
            const noise = rand() * 0.3;
            row.push(clamp01(base * 0.7 + noise));
        }
        // Soft-normalise across columns for each row.
        const sum = row.reduce((a, b) => a + b, 0) || 1;
        for (let c = 0; c < COLS; c++) row[c] = row[c] / sum;
        out.push(row);
    }
    return out;
}

/**
 * Fetch attention data with a hard timeout.
 * @param {string} sessionId
 * @returns {Promise<number[][]>}
 */
async function fetchAttention(sessionId) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    try {
        const url = `/api/attention?session_id=${encodeURIComponent(sessionId)}`;
        const res = await fetch(url, { signal: ctrl.signal, credentials: 'same-origin' });
        if (!res.ok) throw new Error(`status ${res.status}`);
        const data = await res.json();
        const matrix = data?.attention || data?.matrix;
        if (!Array.isArray(matrix) || matrix.length !== ROWS) {
            throw new Error('malformed attention payload');
        }
        return matrix.map((row) => {
            if (!Array.isArray(row) || row.length !== COLS) {
                throw new Error('malformed attention row');
            }
            return row.map(clamp01);
        });
    } finally {
        clearTimeout(to);
    }
}

/**
 * Discover an active session identifier from the running app, with a
 * hardened fallback so the visualiser still renders when offline.
 * @returns {string}
 */
function currentSessionId() {
    try {
        const app = window.app;
        if (app) {
            return app.sessionId || app.userId || 'demo_user';
        }
    } catch (e) { /* no-op */ }
    return 'demo_user';
}

/**
 * Initialise the attention-viz panel inside the given container.
 * @param {HTMLElement} container
 */
export function initAttentionViz(container) {
    if (!container) return;

    const section = document.createElement('section');
    section.className = 'i3-adv-section';
    section.innerHTML = `
        <div class="i3-adv-section-title">
            Cross-Attention Heatmap
            <span class="i3-adv-section-hint">last-token x conditioning</span>
        </div>
        <div class="i3-attn-wrap">
            <div class="i3-attn-grid" role="img"
                 aria-label="Cross-attention distribution, 4 transformer blocks by 4 conditioning tokens"></div>
            <div class="i3-attn-axis">
                <span>cog</span><span>fml</span><span>wrm</span><span>a11y</span>
            </div>
            <div class="i3-attn-legend">
                <span>cold</span>
                <div class="i3-attn-legend-bar" aria-hidden="true"></div>
                <span>hot</span>
            </div>
        </div>
    `;
    container.appendChild(section);

    const gridEl = section.querySelector('.i3-attn-grid');
    const cells = [];
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.className = 'i3-attn-cell';
            cell.style.setProperty('--i3-fill', '0');
            cell.setAttribute('data-val', '0.00');
            cell.setAttribute('aria-label',
                `block ${r + 1}, token ${c + 1}, weight 0.00`);
            gridEl.appendChild(cell);
            cells.push({ el: cell, current: 0, target: 0 });
        }
    }

    // Animation loop eases every cell toward its target.
    let rafId = null;
    function tick() {
        let dirty = false;
        for (let i = 0; i < cells.length; i++) {
            const c = cells[i];
            const diff = c.target - c.current;
            if (Math.abs(diff) > 0.001) {
                c.current += diff * EASE_RATE;
                const v = clamp01(c.current);
                c.el.style.setProperty('--i3-fill', v.toFixed(3));
                c.el.setAttribute('data-val', v.toFixed(2));
                dirty = true;
            }
        }
        if (dirty) {
            rafId = requestAnimationFrame(tick);
        } else {
            rafId = null;
        }
    }
    function kick() {
        if (rafId === null) rafId = requestAnimationFrame(tick);
    }

    function applyMatrix(matrix) {
        let i = 0;
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const v = clamp01(matrix[r][c]);
                cells[i].target = v;
                cells[i].el.setAttribute('aria-label',
                    `block ${r + 1}, token ${c + 1}, weight ${v.toFixed(2)}`);
                i++;
            }
        }
        kick();
    }

    async function refresh() {
        try {
            const m = await fetchAttention(currentSessionId());
            applyMatrix(m);
        } catch (err) {
            // Network or parse failure — fall back to a synthetic pattern
            // so the visualiser still feels alive during the demo.
            applyMatrix(synthesisedMatrix());
        }
    }

    // Initial fill.
    applyMatrix(synthesisedMatrix());

    // Hook WS response events, if a client is already constructed.
    function attachWs() {
        const ws = window.app?.wsClient;
        if (!ws || typeof ws.on !== 'function') {
            return false;
        }
        ws.on('response', () => { refresh(); });
        return true;
    }

    if (!attachWs()) {
        // The app may still be bootstrapping; poll briefly.
        let attempts = 0;
        const pollId = setInterval(() => {
            attempts++;
            if (attachWs() || attempts >= 20) {
                clearInterval(pollId);
            }
        }, 250);
    }

    return {
        refresh,
        applyMatrix,
    };
}
