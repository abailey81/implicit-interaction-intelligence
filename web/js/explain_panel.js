/**
 * explain_panel.js  --  I3 visible reasoning-trace panel.
 *
 * Renders a collapsible "Why this response?" section into the advanced-
 * panels mount (`#i3-advanced-panels`).
 *
 * Primary view: the server-supplied per-turn reasoning trace.  Each
 * response/response_done WebSocket frame carries a `reasoning_trace`
 * field built by `i3.explain.reasoning_trace.build_reasoning_trace`.
 * The trace is stashed on `window.__i3LastReasoningTrace` by app.js
 * and read here on toggle-open, then rendered as:
 *
 *   1. 3-5 short narrative paragraphs in plain English.
 *   2. A horizontal strip of {label, value} signal chips with hover
 *      hints from the `hint` field.
 *   3. A vertical decision-chain list (Encoder → Adaptation → Routing
 *      → Retrieval/SLM → Rewriting).
 *
 * Secondary view (collapsed by default): the existing MC-Dropout
 * `μ`/`σ` per-dimension bars, fetched from `/api/explain/adaptation`
 * on the same toggle.  We keep the raw uncertainty surface so ML
 * reviewers can still audit the underlying numerics — it is just no
 * longer the primary thing the panel shows.
 *
 * Soft-fail policy:
 *   - If no trace has arrived yet, show a "send a message" prompt.
 *   - If `/api/explain/adaptation` fails or times out, keep the
 *     narrative trace and surface a one-line diagnostic in the raw
 *     uncertainty `<details>` section.
 *   - Never throw to the console as an unhandled rejection.
 */

const FETCH_TIMEOUT_MS = 6000;
const ENDPOINT = '/api/explain/adaptation';
const USER_ID = 'demo_user'; // Matches the userId in app.js (I3App).

/**
 * Coerce an arbitrary value to a safe, length-capped string.
 * @param {unknown} v
 * @param {number} max
 * @returns {string}
 */
function safeStr(v, max = 400) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return s.length > max ? s.slice(0, max) : s;
}

/**
 * POST a JSON body with a hard AbortController timeout.
 * @param {string} url
 * @param {object} body
 * @param {number} ms
 * @returns {Promise<object>}
 */
async function postWithTimeout(url, body, ms = FETCH_TIMEOUT_MS) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), ms);
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
        clearTimeout(timer);
    }
}

// =========================================================================
// Primary view — narrative reasoning trace
// =========================================================================

/**
 * Render the server-supplied reasoning trace into a container element.
 * @param {HTMLElement} body
 * @param {object|null} trace
 */
function renderReasoningTrace(body, trace) {
    body.innerHTML = '';

    if (!trace || typeof trace !== 'object') {
        const empty = document.createElement('div');
        empty.className = 'i3-trace-empty';
        empty.textContent = 'Send a message to see the reasoning trace.';
        body.appendChild(empty);
        return;
    }

    // ---- 1. Narrative paragraphs ----
    const paragraphs = Array.isArray(trace.narrative_paragraphs)
        ? trace.narrative_paragraphs
        : [];
    if (paragraphs.length > 0) {
        const narrative = document.createElement('div');
        narrative.className = 'i3-trace-narrative';
        for (const p of paragraphs) {
            const para = document.createElement('p');
            para.className = 'i3-trace-para';
            // SEC: textContent only — server-supplied prose is rendered
            // as plain text, never HTML.
            para.textContent = safeStr(p, 1200);
            narrative.appendChild(para);
        }
        body.appendChild(narrative);
    }

    // ---- 2. Signal chips ----
    const chips = Array.isArray(trace.signal_chips) ? trace.signal_chips : [];
    if (chips.length > 0) {
        const strip = document.createElement('div');
        strip.className = 'i3-trace-chips';
        for (const c of chips) {
            if (!c || typeof c !== 'object') continue;
            const chip = document.createElement('span');
            chip.className = 'i3-trace-chip';
            const label = document.createElement('span');
            label.className = 'i3-trace-chip-label';
            label.textContent = safeStr(c.label, 32);
            const value = document.createElement('span');
            value.className = 'i3-trace-chip-value';
            value.textContent = safeStr(c.value, 32);
            chip.appendChild(label);
            chip.appendChild(value);
            if (c.hint) chip.title = safeStr(c.hint, 200);
            strip.appendChild(chip);
        }
        body.appendChild(strip);
    }

    // ---- 3. Decision chain ----
    const chain = Array.isArray(trace.decision_chain) ? trace.decision_chain : [];
    if (chain.length > 0) {
        const ol = document.createElement('ol');
        ol.className = 'i3-trace-chain';
        for (const step of chain) {
            if (!step || typeof step !== 'object') continue;
            const li = document.createElement('li');
            li.className = 'i3-trace-chain-step';
            const head = document.createElement('div');
            head.className = 'i3-trace-chain-head';
            head.textContent = safeStr(step.step, 64);
            const what = document.createElement('div');
            what.className = 'i3-trace-chain-what';
            what.textContent = safeStr(step.what, 400);
            const why = document.createElement('div');
            why.className = 'i3-trace-chain-why';
            why.textContent = safeStr(step.why, 400);
            li.appendChild(head);
            li.appendChild(what);
            li.appendChild(why);
            ol.appendChild(li);
        }
        body.appendChild(ol);
    }
}

// =========================================================================
// Secondary view — MC-Dropout per-dimension bars
// =========================================================================

/**
 * Build the per-dimension confidence bar row.
 * @param {{dimension: string, mean: number, std: number, ci_lower: number, ci_upper: number, classification: string}} d
 */
function renderDimensionRow(d) {
    const row = document.createElement('div');
    row.className = 'i3-explain-dim-row';
    const isConfident = d.classification === 'confident';
    row.dataset.classification = isConfident ? 'confident' : 'uncertain';

    const label = document.createElement('span');
    label.className = 'i3-explain-dim-label';
    label.textContent = safeStr(d.dimension, 32);

    const barWrap = document.createElement('span');
    barWrap.className = 'i3-explain-bar-wrap';
    barWrap.setAttribute('role', 'progressbar');
    barWrap.setAttribute('aria-valuemin', '0');
    barWrap.setAttribute('aria-valuemax', '1');
    barWrap.setAttribute('aria-valuenow', String(d.mean ?? 0));

    const meanFrac = Math.max(0, Math.min(1, Number(d.mean) || 0));
    const stdFrac = Math.max(0, Math.min(0.5, Number(d.std) || 0));

    const meanBar = document.createElement('span');
    meanBar.className = 'i3-explain-bar-mean';
    meanBar.style.width = (meanFrac * 100).toFixed(1) + '%';

    const stdBar = document.createElement('span');
    stdBar.className = 'i3-explain-bar-std';
    stdBar.style.width = (stdFrac * 2 * 100).toFixed(1) + '%';
    stdBar.style.left = Math.max(0, (meanFrac - stdFrac) * 100).toFixed(1) + '%';

    barWrap.appendChild(meanBar);
    barWrap.appendChild(stdBar);

    const meta = document.createElement('span');
    meta.className = 'i3-explain-dim-meta';
    meta.textContent = `μ=${meanFrac.toFixed(2)} σ=${stdFrac.toFixed(2)}`;

    row.appendChild(label);
    row.appendChild(barWrap);
    row.appendChild(meta);

    if (!isConfident) {
        const badge = document.createElement('span');
        badge.className = 'i3-explain-refuse-badge';
        badge.textContent = 'refused';
        badge.title =
            'The system is not confident enough in this dimension to ' +
            'adapt; showing the neutral baseline instead.';
        row.appendChild(badge);
    }

    return row;
}

/**
 * Render the MC-Dropout payload into the raw-uncertainty details body.
 * @param {HTMLElement} body
 * @param {object} data
 */
function renderRawUncertainty(body, data) {
    body.innerHTML = '';

    if (!data || typeof data !== 'object') {
        const err = document.createElement('div');
        err.className = 'i3-explain-err';
        err.textContent = 'Malformed server response.';
        body.appendChild(err);
        return;
    }

    const perDim = Array.isArray(data.per_dimension) ? data.per_dimension : [];
    const rows = document.createElement('div');
    rows.className = 'i3-explain-dim-rows';
    for (const d of perDim) {
        rows.appendChild(renderDimensionRow(d));
    }
    body.appendChild(rows);

    const cfs = Array.isArray(data.counterfactuals) ? data.counterfactuals : [];
    if (cfs.length > 0) {
        const cf = cfs[0];
        const cfText =
            `If ${safeStr(cf.feature_name, 40)} had been ` +
            `${Number(cf.counterfactual_value).toFixed(3)} instead of ` +
            `${Number(cf.current_value).toFixed(3)}, the ` +
            `${safeStr(cf.dimension_affected, 40)} adaptation would have ` +
            `been ${Number(cf.counterfactual_dimension).toFixed(3)} ` +
            `instead of ${Number(cf.current_dimension).toFixed(3)}.`;
        const cfEl = document.createElement('div');
        cfEl.className = 'i3-explain-cf';
        cfEl.textContent = cfText;
        body.appendChild(cfEl);
    }

    const summary = document.createElement('div');
    summary.className = 'i3-explain-summary';
    summary.textContent = safeStr(data.natural_language, 600);
    body.appendChild(summary);

    const footer = document.createElement('div');
    footer.className = 'i3-explain-footer';
    const samples = Number(data.sample_count) || 0;
    const latency = Number(data.latency_ms) || 0;
    footer.textContent =
        `${samples} MC-Dropout samples - ${latency.toFixed(1)} ms`;
    body.appendChild(footer);
}

// =========================================================================
// Panel wiring
// =========================================================================

/**
 * Wire up the collapsible panel and click handler.
 * @param {HTMLElement} root
 */
function initExplainPanel(root) {
    if (!root) return;

    const section = document.createElement('section');
    section.className = 'i3-explain-section';
    section.innerHTML = `
        <button type="button" class="i3-explain-toggle" aria-expanded="false">
            <span class="i3-explain-toggle-caret" aria-hidden="true">&#9654;</span>
            Why this response?
        </button>
        <div class="i3-explain-body" hidden>
            <div class="i3-trace-mount"></div>
            <details class="i3-raw-uncertainty">
                <summary>Raw uncertainty (MC-Dropout)</summary>
                <div class="i3-raw-uncertainty-body">
                    <div class="i3-explain-loading">Click to load…</div>
                </div>
            </details>
        </div>
    `;
    root.appendChild(section);

    const toggle = section.querySelector('.i3-explain-toggle');
    const body = section.querySelector('.i3-explain-body');
    const caret = section.querySelector('.i3-explain-toggle-caret');
    const traceMount = section.querySelector('.i3-trace-mount');
    const rawDetails = section.querySelector('.i3-raw-uncertainty');
    const rawBody = section.querySelector('.i3-raw-uncertainty-body');

    let rawLoaded = false;
    let rawLoading = false;

    function refreshTrace() {
        const trace = (typeof window !== 'undefined')
            ? window.__i3LastReasoningTrace
            : null;
        renderReasoningTrace(traceMount, trace);
    }

    async function loadRawUncertainty() {
        if (rawLoaded || rawLoading) return;
        rawLoading = true;
        rawBody.innerHTML =
            '<div class="i3-explain-loading">Running Monte-Carlo-dropout estimator...</div>';
        try {
            const data = await postWithTimeout(ENDPOINT, {
                user_id: USER_ID,
                top_k: 3,
                confidence_threshold: 0.15,
            });
            renderRawUncertainty(rawBody, data);
            rawLoaded = true;
        } catch (e) {
            rawBody.innerHTML = '';
            const err = document.createElement('div');
            err.className = 'i3-explain-err';
            err.textContent = 'Raw uncertainty unavailable.';
            rawBody.appendChild(err);
            try {
                console.warn('[I3] explain panel:', e && e.message);
            } catch (_ignored) { /* noop */ }
        } finally {
            rawLoading = false;
        }
    }

    function open() {
        toggle.setAttribute('aria-expanded', 'true');
        if (caret) caret.textContent = '▼';
        body.hidden = false;
        refreshTrace();
    }

    function close() {
        toggle.setAttribute('aria-expanded', 'false');
        if (caret) caret.textContent = '▶';
        body.hidden = true;
    }

    toggle.addEventListener('click', () => {
        if (toggle.getAttribute('aria-expanded') === 'true') {
            close();
        } else {
            open();
        }
    });

    // Lazy-load raw uncertainty only when the user expands the
    // <details> element.  Avoids burning MC-Dropout cycles on every
    // panel open.
    if (rawDetails) {
        rawDetails.addEventListener('toggle', () => {
            if (rawDetails.open) loadRawUncertainty();
        });
    }

    // Live-update when a new reasoning trace arrives while the panel
    // is open.
    try {
        window.addEventListener('i3:reasoning-trace', () => {
            if (toggle.getAttribute('aria-expanded') === 'true') {
                refreshTrace();
            }
        });
    } catch (_e) { /* noop */ }

    return { section };
}

function bootstrap() {
    // apple21 cleanup: prefer the dedicated mount in the State tab
    // (#i3-explain-mount) so the reasoning trace lives next to the
    // attention map.  Fall back to the legacy advanced-panels host so
    // older HTML still mounts the panel without breaking.
    const root = document.getElementById('i3-explain-mount')
        || document.getElementById('i3-advanced-panels');
    if (!root) return;
    try {
        initExplainPanel(root);
    } catch (e) {
        try {
            console.warn('[I3] explain panel init failed:', e && e.message);
        } catch (_ignored) {
            /* noop */
        }
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
} else {
    bootstrap();
}

export { initExplainPanel };
