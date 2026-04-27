/**
 * preference_panel.js  --  I3 floating "How am I doing?" toast.
 *
 * apple21 cleanup (2026-04-25): the inline A/B preference card has been
 * relocated to a small floating toast in the bottom-right corner so the
 * chat tab can be JUST chat.  Behaviour:
 *
 *   - Shows only after the user has received ≥5 AI messages this session
 *     (driven by the `i3:ai-message-landed` CustomEvent dispatched from
 *     chat.js).
 *   - Polls `/api/preference/query/{user_id}` to ask whether the current
 *     turn is "informative enough" to warrant an A/B query; renders
 *     only if `should_query` is true.
 *   - Sticks to the bottom-right corner with a subtle drop-shadow,
 *     dismissible by an explicit ✕ button.
 *   - Dismissals are persisted in `localStorage['i3:pref-dismissed']`
 *     for 24 hours; once dismissed it won't show again until then or
 *     until the user explicitly invokes it (future work).
 *   - Cooldown of 60 s between toast appearances inside one session.
 *
 * Soft-fails silently if the preference endpoint is absent.
 */

const FETCH_TIMEOUT_MS = 6000;
const POLL_INTERVAL_MS = 30000;          // 30 s — check after each new chat
const COOLDOWN_MS = 60000;               // 60 s between toast appearances
const DISMISS_TTL_MS = 24 * 60 * 60 * 1000;
const MIN_AI_MESSAGES = 5;
const USER_ID = 'demo';
const QUERY_ENDPOINT = '/api/preference/query/';
const RECORD_ENDPOINT = '/api/preference/record';
const MAX_STR_LEN = 400;
const DISMISS_KEY = 'i3:pref-dismissed';

function safeStr(v, max = MAX_STR_LEN) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return s.length > max ? s.slice(0, max) : s;
}

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

function isDismissed() {
    try {
        const raw = localStorage.getItem(DISMISS_KEY);
        if (!raw) return false;
        const ts = Number(raw);
        if (!Number.isFinite(ts)) return false;
        return Date.now() - ts < DISMISS_TTL_MS;
    } catch (_e) {
        return false;
    }
}

function markDismissed() {
    try {
        localStorage.setItem(DISMISS_KEY, String(Date.now()));
    } catch (_e) { /* ignore */ }
}

/**
 * Build and inject the toast into the global root.
 * Returns a {dismiss} handle.
 */
function renderToast(root, query, onChoose, onDismiss) {
    const toast = document.createElement('section');
    toast.className = 'i3-preference-toast';
    toast.setAttribute('role', 'dialog');
    toast.setAttribute('aria-label', 'Quick preference check');

    const header = document.createElement('div');
    header.className = 'i3-preference-toast-head';
    const title = document.createElement('div');
    title.className = 'i3-preference-toast-title';
    title.textContent = 'How am I doing?';
    const dismissBtn = document.createElement('button');
    dismissBtn.type = 'button';
    dismissBtn.className = 'i3-preference-toast-dismiss';
    dismissBtn.setAttribute('aria-label', 'Dismiss preference prompt');
    dismissBtn.textContent = '✕';
    dismissBtn.addEventListener('click', () => {
        markDismissed();
        if (typeof onDismiss === 'function') onDismiss();
    });
    header.appendChild(title);
    header.appendChild(dismissBtn);
    toast.appendChild(header);

    const sub = document.createElement('div');
    sub.className = 'i3-preference-toast-sub';
    sub.textContent = 'Which response feels more natural for how you’re typing right now?';
    toast.appendChild(sub);

    const grid = document.createElement('div');
    grid.className = 'i3-preference-toast-grid';
    const mkOption = (label, text, code) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'i3-preference-toast-option';
        btn.setAttribute('aria-label', `Choose ${label}`);
        const hd = document.createElement('div');
        hd.className = 'i3-preference-toast-option-label';
        hd.textContent = label;
        const bd = document.createElement('div');
        bd.className = 'i3-preference-toast-option-text';
        bd.textContent = safeStr(text, 220);
        btn.appendChild(hd);
        btn.appendChild(bd);
        btn.addEventListener('click', () => {
            onChoose(code).catch(() => { /* swallow */ });
        });
        return btn;
    };
    grid.appendChild(mkOption('A', query.response_a, 'a'));
    grid.appendChild(mkOption('B', query.response_b, 'b'));
    toast.appendChild(grid);

    const tieRow = document.createElement('button');
    tieRow.type = 'button';
    tieRow.className = 'i3-preference-toast-tie';
    tieRow.textContent = 'Both feel equivalent';
    tieRow.addEventListener('click', () => {
        onChoose('tie').catch(() => { /* swallow */ });
    });
    toast.appendChild(tieRow);

    root.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('is-visible'));
    return {
        dismiss: () => {
            toast.classList.remove('is-visible');
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 280);
        },
    };
}

class PreferenceToast {
    constructor(root) {
        this.root = root;
        this.activeHandle = null;
        this.lastShown = 0;
        this.aiMessageCount = 0;
        this.checking = false;

        // Listen for AI messages landing in the chat — only check the
        // server once we have ≥MIN_AI_MESSAGES.
        window.addEventListener('i3:ai-message-landed', (ev) => {
            const c = ev && ev.detail && ev.detail.messageCount;
            if (Number.isFinite(c)) this.aiMessageCount = c;
            // Wait briefly for the trace + post-processing to settle.
            setTimeout(() => this.maybeCheck(), 1200);
        });
    }

    async maybeCheck() {
        if (this.activeHandle) return;
        if (this.aiMessageCount < MIN_AI_MESSAGES) return;
        if (Date.now() - this.lastShown < COOLDOWN_MS) return;
        if (isDismissed()) return;
        if (this.checking) return;
        this.checking = true;
        try {
            const q = await getWithTimeout(
                QUERY_ENDPOINT + encodeURIComponent(USER_ID)
            );
            if (!q || !q.should_query) return;
            this.show(q);
        } catch (e) {
            try {
                console.debug('[I3] preference toast poll failed:', e && e.message);
            } catch (_ignored) { /* noop */ }
        } finally {
            this.checking = false;
        }
    }

    show(query) {
        this.lastShown = Date.now();
        this.activeHandle = renderToast(
            this.root,
            query,
            (winner) => this.submit(query, winner),
            () => {
                if (this.activeHandle) {
                    this.activeHandle.dismiss();
                    this.activeHandle = null;
                }
            },
        );
    }

    async submit(query, winner) {
        let resp = null;
        try {
            resp = await postWithTimeout(RECORD_ENDPOINT, {
                user_id: USER_ID,
                prompt: safeStr(query.prompt, 2000) || 'unknown',
                response_a: safeStr(query.response_a, 2000) || 'A',
                response_b: safeStr(query.response_b, 2000) || 'B',
                winner: winner,
                context: Array.isArray(query.context) ? query.context : [],
                response_a_features: Array.isArray(query.response_a_features)
                    ? query.response_a_features : [],
                response_b_features: Array.isArray(query.response_b_features)
                    ? query.response_b_features : [],
            });
        } catch (e) {
            try {
                console.debug('[I3] preference record failed:', e && e.message);
            } catch (_ignored) { /* noop */ }
        } finally {
            if (this.activeHandle) {
                this.activeHandle.dismiss();
                this.activeHandle = null;
            }
        }
        try {
            if (resp && resp.personalisation) {
                showPersonalisationToast(resp.personalisation);
            }
        } catch (e) {
            try {
                console.debug('[I3] personalisation toast failed:', e && e.message);
            } catch (_ignored) { /* noop */ }
        }
    }
}

function showPersonalisationToast(update) {
    const direction = String(update.direction || 'unknown');
    const delta = Number(update.delta || 0);
    const n = Number(update.n_updates_total || 0);
    const sign = delta >= 0 ? '+' : '';
    const toast = document.createElement('div');
    toast.className = 'lora-update-toast';
    toast.setAttribute('role', 'status');
    toast.setAttribute('aria-live', 'polite');
    const icon = document.createElement('span');
    icon.className = 'lora-update-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '\u{1F9E0}';
    const text = document.createElement('span');
    text.className = 'lora-update-text';
    text.textContent = `Personalisation updated · ${direction} ${sign}${delta.toFixed(3)}`;
    const meta = document.createElement('span');
    meta.className = 'lora-update-meta';
    meta.textContent = `N=${n} total updates · drift bounded`;
    toast.appendChild(icon);
    toast.appendChild(text);
    toast.appendChild(meta);
    document.body.appendChild(toast);
    requestAnimationFrame(() => {
        toast.classList.add('lora-update-toast-visible');
    });
    setTimeout(() => {
        toast.classList.remove('lora-update-toast-visible');
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 400);
    }, 4000);
}

function bootstrap() {
    const root = document.getElementById('i3-preference-toast-root');
    if (!root) return;
    try {
        new PreferenceToast(root);
    } catch (e) {
        try {
            console.warn('[I3] preference toast init failed:', e && e.message);
        } catch (_ignored) { /* noop */ }
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
} else {
    bootstrap();
}

export { PreferenceToast };
