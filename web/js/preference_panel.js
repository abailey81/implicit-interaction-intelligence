/**
 * preference_panel.js  --  I3 active-preference A/B prompt panel.
 *
 * Periodically polls `/api/preference/query/{user_id}` to ask the user
 * for an A/B preference.  The backend's ActivePreferenceSelector decides
 * whether the current turn is "informative enough" to warrant a query;
 * when `should_query` is true the panel renders a side-by-side prompt
 * and POSTs the user's choice back to `/api/preference/record`.
 *
 * The panel appears at most once every `COOLDOWN_MS` (default 30 s) and
 * soft-fails silently — if the endpoint is absent (older servers) the
 * panel simply never shows.  This mirrors the behaviour of
 * `explain_panel.js` and `browser_inference.js`.
 */

const FETCH_TIMEOUT_MS = 6000;
const POLL_INTERVAL_MS = 20000;
const COOLDOWN_MS = 30000;
const USER_ID = 'demo';
const QUERY_ENDPOINT = '/api/preference/query/';
const RECORD_ENDPOINT = '/api/preference/record';
const MAX_STR_LEN = 400;

/**
 * Coerce an arbitrary value to a length-capped string.
 * @param {unknown} v
 * @param {number} max
 * @returns {string}
 */
function safeStr(v, max = MAX_STR_LEN) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return s.length > max ? s.slice(0, max) : s;
}

/**
 * GET JSON with a hard AbortController timeout.
 * @param {string} url
 * @param {number} ms
 * @returns {Promise<object>}
 */
async function getWithTimeout(url, ms = FETCH_TIMEOUT_MS) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), ms);
    try {
        const res = await fetch(url, {
            method: 'GET',
            credentials: 'same-origin',
            signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`status ${res.status}`);
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

/**
 * POST JSON with a hard timeout.
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
 * Render the A/B card and return a dismiss handle.
 * @param {HTMLElement} root
 * @param {object} query
 * @param {(winner: string) => Promise<void>} onChoose
 */
function renderCard(root, query, onChoose) {
    const card = document.createElement('section');
    card.className = 'i3-preference-card';
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-label', 'Preference prompt');

    const title = document.createElement('div');
    title.className = 'i3-preference-title';
    title.textContent = 'Which response feels more natural for how you are typing right now?';
    card.appendChild(title);

    const prompt = document.createElement('div');
    prompt.className = 'i3-preference-prompt';
    prompt.textContent = safeStr(query.prompt, 300);
    card.appendChild(prompt);

    const grid = document.createElement('div');
    grid.className = 'i3-preference-grid';

    const mkOption = (label, text, winnerCode) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'i3-preference-option';
        btn.setAttribute('aria-label', `Choose ${label}`);
        const hd = document.createElement('div');
        hd.className = 'i3-preference-option-label';
        hd.textContent = label;
        const bd = document.createElement('div');
        bd.className = 'i3-preference-option-text';
        bd.textContent = safeStr(text, 500);
        btn.appendChild(hd);
        btn.appendChild(bd);
        btn.addEventListener('click', () => {
            onChoose(winnerCode).catch(() => {
                /* swallow — soft-fail */
            });
        });
        return btn;
    };

    grid.appendChild(mkOption('Response A', query.response_a, 'a'));
    grid.appendChild(mkOption('Response B', query.response_b, 'b'));
    card.appendChild(grid);

    const tieRow = document.createElement('div');
    tieRow.className = 'i3-preference-tie-row';
    const tieBtn = document.createElement('button');
    tieBtn.type = 'button';
    tieBtn.className = 'i3-preference-tie';
    tieBtn.textContent = 'Both feel equivalent';
    tieBtn.addEventListener('click', () => {
        onChoose('tie').catch(() => {
            /* swallow */
        });
    });
    tieRow.appendChild(tieBtn);
    card.appendChild(tieRow);

    const footer = document.createElement('div');
    footer.className = 'i3-preference-footer';
    const ig = Number(query.information_gain) || 0;
    footer.textContent = `Info gain ${ig.toFixed(3)} — ${safeStr(query.reason, 120)}`;
    card.appendChild(footer);

    root.appendChild(card);
    return {
        dismiss: () => {
            if (card.parentNode) card.parentNode.removeChild(card);
        },
    };
}

/**
 * Top-level panel state — one active card at a time.
 */
class PreferencePanel {
    /**
     * @param {HTMLElement} root
     */
    constructor(root) {
        this.root = root;
        this.activeHandle = null;
        this.lastShown = 0;
        this.timer = null;
    }

    start() {
        // Stagger the first poll so we don't race explain_panel/tts init.
        setTimeout(() => this.tick(), 5000);
        this.timer = setInterval(() => this.tick(), POLL_INTERVAL_MS);
    }

    async tick() {
        if (this.activeHandle) return;
        if (Date.now() - this.lastShown < COOLDOWN_MS) return;
        try {
            const q = await getWithTimeout(QUERY_ENDPOINT + encodeURIComponent(USER_ID));
            if (!q || !q.should_query) return;
            this.show(q);
        } catch (e) {
            // Endpoint absent or error — soft-fail.
            try {
                console.debug('[I3] preference panel poll failed:', e && e.message);
            } catch (_ignored) {
                /* noop */
            }
        }
    }

    show(query) {
        this.lastShown = Date.now();
        this.activeHandle = renderCard(this.root, query, (winner) => this.submit(query, winner));
    }

    async submit(query, winner) {
        try {
            await postWithTimeout(RECORD_ENDPOINT, {
                user_id: USER_ID,
                prompt: safeStr(query.prompt, 2000) || 'unknown',
                response_a: safeStr(query.response_a, 2000) || 'A',
                response_b: safeStr(query.response_b, 2000) || 'B',
                winner: winner,
                context: Array.isArray(query.context) ? query.context : [],
                response_a_features: Array.isArray(query.response_a_features)
                    ? query.response_a_features
                    : [],
                response_b_features: Array.isArray(query.response_b_features)
                    ? query.response_b_features
                    : [],
            });
        } catch (e) {
            try {
                console.debug('[I3] preference record failed:', e && e.message);
            } catch (_ignored) {
                /* noop */
            }
        } finally {
            if (this.activeHandle) {
                this.activeHandle.dismiss();
                this.activeHandle = null;
            }
        }
    }
}

/**
 * Bootstrap the panel — silently returns when the mount point is absent.
 */
function bootstrap() {
    const root = document.getElementById('i3-advanced-panels');
    if (!root) return;
    const host = document.createElement('div');
    host.id = 'i3-preference-panel';
    host.className = 'i3-preference-host';
    root.appendChild(host);
    try {
        const panel = new PreferencePanel(host);
        panel.start();
    } catch (e) {
        try {
            console.warn('[I3] preference panel init failed:', e && e.message);
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

export { PreferencePanel };
