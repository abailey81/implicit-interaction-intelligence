/**
 * whatif.js -- I3 Advanced Panels
 *
 * Counter-factual adaptation comparator. When the user clicks
 * "Show alternative adaptations", this module takes the most recent
 * outgoing message and POSTs it to `/whatif/compare` along with three
 * hand-crafted alternative AdaptationVectors, then renders a side-by-
 * side grid of the canonical response and the three counter-factuals.
 *
 * Export: initWhatIfPanel(container)
 */

const FETCH_TIMEOUT_MS = 6000;

const ALT_VECTORS = [
    {
        id: 'low_cog',
        label: 'Low cognitive load',
        vector: {
            verbosity: 0.25,
            formality: 0.3,
            directness: 0.85,
            warmth: 0.55,
            cognitive_accessibility: 0.9,
            emotional_tone: 0.55,
            technical_depth: 0.2,
        },
    },
    {
        id: 'high_warmth',
        label: 'High warmth',
        vector: {
            verbosity: 0.55,
            formality: 0.25,
            directness: 0.45,
            warmth: 0.95,
            cognitive_accessibility: 0.6,
            emotional_tone: 0.85,
            technical_depth: 0.35,
        },
    },
    {
        id: 'a11y',
        label: 'Accessibility-elevated',
        vector: {
            verbosity: 0.35,
            formality: 0.55,
            directness: 0.8,
            warmth: 0.6,
            cognitive_accessibility: 1.0,
            emotional_tone: 0.5,
            technical_depth: 0.25,
        },
    },
];

/**
 * Length-capped string coercion for server-supplied content.
 */
function safeStr(v, max = 800) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return s.length > max ? s.slice(0, max) : s;
}

/**
 * POST with an AbortController-driven hard timeout.
 */
async function postWithTimeout(url, body, ms = FETCH_TIMEOUT_MS) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), ms);
    try {
        const res = await fetch(url, {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`status ${res.status}`);
        return await res.json();
    } finally {
        clearTimeout(to);
    }
}

/**
 * Find the most recent outbound user message displayed in the chat.
 * Falls back to a bland prompt if nothing is found.
 */
function lastUserMessage() {
    try {
        const nodes = document.querySelectorAll('#chat-messages .message.user .message-text');
        if (nodes.length === 0) return '';
        return nodes[nodes.length - 1].textContent.trim();
    } catch (e) {
        return '';
    }
}

/**
 * Find the most recent AI response for the canonical column.
 */
function lastAiMessage() {
    try {
        const nodes = document.querySelectorAll('#chat-messages .message.ai .message-text');
        if (nodes.length === 0) return '';
        return nodes[nodes.length - 1].textContent.trim();
    } catch (e) {
        return '';
    }
}

/**
 * Render a single column in the comparator grid.
 */
function renderColumn(grid, label, text, meta = {}, canonical = false) {
    const col = document.createElement('div');
    col.className = 'i3-whatif-col' + (canonical ? ' canonical' : '');

    const lab = document.createElement('div');
    lab.className = 'i3-whatif-col-label';
    lab.textContent = label;

    const body = document.createElement('div');
    body.className = 'i3-whatif-col-text';
    body.textContent = safeStr(text) || '(no response)';

    col.appendChild(lab);
    col.appendChild(body);

    if (meta && Object.keys(meta).length > 0) {
        const metaEl = document.createElement('div');
        metaEl.className = 'i3-whatif-col-meta';
        for (const [k, v] of Object.entries(meta)) {
            const span = document.createElement('span');
            span.textContent = `${k}:${v}`;
            metaEl.appendChild(span);
        }
        col.appendChild(metaEl);
    }

    grid.appendChild(col);
}

/**
 * Initialise the what-if comparator panel.
 * @param {HTMLElement} container
 */
export function initWhatIfPanel(container) {
    if (!container) return;

    const section = document.createElement('section');
    section.className = 'i3-adv-section';
    section.innerHTML = `
        <div class="i3-adv-section-title">
            What-If Comparator
            <span class="i3-adv-section-hint">3 counter-factual adaptations</span>
        </div>
        <button class="i3-whatif-btn" type="button">Show alternative adaptations</button>
        <div class="i3-whatif-grid" hidden></div>
    `;
    container.appendChild(section);

    const btn = section.querySelector('.i3-whatif-btn');
    const grid = section.querySelector('.i3-whatif-grid');

    btn.addEventListener('click', async () => {
        const prompt = lastUserMessage();
        if (!prompt) {
            grid.hidden = false;
            grid.innerHTML = '';
            const err = document.createElement('div');
            err.className = 'i3-whatif-err';
            err.textContent = 'Send a message first, then ask for alternatives.';
            grid.appendChild(err);
            return;
        }

        btn.disabled = true;
        grid.hidden = false;
        grid.innerHTML = '';
        const loading = document.createElement('div');
        loading.className = 'i3-whatif-loading';
        loading.textContent = 'Generating counter-factual responses...';
        grid.appendChild(loading);

        // Canonical column first, shown immediately from the DOM.
        const canonicalText = lastAiMessage();

        let alts = null;
        let err = null;
        try {
            const result = await postWithTimeout('/whatif/compare', {
                prompt,
                alternatives: ALT_VECTORS.map((a) => ({
                    id: a.id,
                    label: a.label,
                    adaptation: a.vector,
                })),
            });
            alts = Array.isArray(result?.responses) ? result.responses : null;
        } catch (e) {
            err = e;
        }

        grid.innerHTML = '';
        renderColumn(grid, 'Canonical', canonicalText, {}, true);

        if (alts && alts.length > 0) {
            for (const item of alts.slice(0, 3)) {
                const label = safeStr(item.label, 48) || 'Variant';
                const text = safeStr(item.text || item.response, 800);
                const meta = {};
                if (item.route) meta.route = safeStr(item.route, 8);
                if (item.latency_ms != null) {
                    const n = Number(item.latency_ms);
                    if (Number.isFinite(n)) meta.ms = Math.round(n);
                }
                renderColumn(grid, label, text, meta);
            }
        } else {
            // Graceful fallback: explain the three variants textually.
            for (const a of ALT_VECTORS) {
                renderColumn(grid, a.label,
                    '(backend /whatif/compare unavailable — placeholder)', {});
            }
            if (err) {
                const e = document.createElement('div');
                e.className = 'i3-whatif-err';
                e.textContent = `Request failed: ${safeStr(err.message || err, 120)}`;
                grid.appendChild(e);
            }
        }

        btn.disabled = false;
    });

    return { section };
}
