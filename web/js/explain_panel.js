/**
 * explain_panel.js  --  I3 uncertainty + counterfactual explainer panel.
 *
 * Renders a collapsible "Why this response?" section into the advanced-
 * panels mount (`#i3-advanced-panels`). On click, POSTs the current
 * user's id to `/api/explain/adaptation` and renders:
 *
 *   - Per-dimension confidence bars (green for confident,
 *     amber for uncertain, against the per-dim mean).
 *   - The top counterfactual as a single plain-English sentence.
 *   - A "refuse to adapt" badge for each dimension whose std is above
 *     the configured threshold.
 *
 * This module soft-fails silently:
 *   - If `#i3-advanced-panels` does not exist, nothing is rendered.
 *   - If the endpoint returns non-2xx or times out, the panel shows a
 *     one-line diagnostic but never throws.
 *
 * Palette follows the project's `#0f3460` + `#e94560` scheme (see
 * `web/css/explain_panel.css`).
 */

const FETCH_TIMEOUT_MS = 6000;
const ENDPOINT = '/api/explain/adaptation';
const USER_ID = 'demo'; // Matches the default user id in the web shell.

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
    meta.textContent =
        `μ=${meanFrac.toFixed(2)} σ=${stdFrac.toFixed(2)}`;

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
 * Render the response body into the collapsible panel body.
 * @param {HTMLElement} body
 * @param {object} data
 */
function renderPayload(body, data) {
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
        <div class="i3-explain-body" hidden></div>
    `;
    root.appendChild(section);

    const toggle = section.querySelector('.i3-explain-toggle');
    const body = section.querySelector('.i3-explain-body');
    const caret = section.querySelector('.i3-explain-toggle-caret');

    let loading = false;

    async function open() {
        toggle.setAttribute('aria-expanded', 'true');
        if (caret) caret.textContent = '▼'; // black down-pointing triangle
        body.hidden = false;

        if (loading) return;
        loading = true;
        body.innerHTML =
            '<div class="i3-explain-loading">Running Monte-Carlo-dropout estimator...</div>';

        try {
            const data = await postWithTimeout(ENDPOINT, {
                user_id: USER_ID,
                top_k: 3,
                confidence_threshold: 0.15,
            });
            renderPayload(body, data);
        } catch (e) {
            body.innerHTML = '';
            const err = document.createElement('div');
            err.className = 'i3-explain-err';
            err.textContent = 'Explanation unavailable.';
            body.appendChild(err);
            // Silent soft-fail: the endpoint may simply not be mounted
            // in this build. Log for developers but never throw.
            try {
                console.warn('[I3] explain panel:', e && e.message);
            } catch (_ignored) {
                /* noop */
            }
        } finally {
            loading = false;
        }
    }

    function close() {
        toggle.setAttribute('aria-expanded', 'false');
        if (caret) caret.textContent = '▶'; // black right-pointing triangle
        body.hidden = true;
    }

    toggle.addEventListener('click', () => {
        if (toggle.getAttribute('aria-expanded') === 'true') {
            close();
        } else {
            open();
        }
    });

    return { section };
}

function bootstrap() {
    const root = document.getElementById('i3-advanced-panels');
    if (!root) return;
    try {
        initExplainPanel(root);
    } catch (e) {
        // Absolute belt-and-braces: never propagate to the console as
        // an unhandled rejection; the panel is non-critical.
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
