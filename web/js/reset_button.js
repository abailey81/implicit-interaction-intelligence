/**
 * reset_button.js -- I3 Advanced Panels
 *
 * Floating "Reset session" pill in the top-right of the chat panel.
 * On click it shows a 1.5-second confirmation overlay; if not cancelled
 * it POSTs to `/admin/reset` with the admin bearer token from
 * `localStorage.i3AdminToken`, then reloads the page on success.
 *
 * Export: initResetButton()
 */

const HOLD_MS = 1500;
const FETCH_TIMEOUT_MS = 4000;

function adminToken() {
    try {
        return localStorage.getItem('i3AdminToken') || '';
    } catch (e) {
        return '';
    }
}

async function postReset() {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    try {
        const headers = { 'Content-Type': 'application/json' };
        const tok = adminToken();
        if (tok) headers['Authorization'] = `Bearer ${tok}`;
        const res = await fetch('/admin/reset', {
            method: 'POST',
            credentials: 'same-origin',
            headers,
            body: JSON.stringify({}),
            signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`status ${res.status}`);
        return await res.json().catch(() => ({}));
    } finally {
        clearTimeout(to);
    }
}

/**
 * Build and show the confirmation overlay. Resolves true on confirm,
 * false on cancel or timeout.
 * @returns {Promise<boolean>}
 */
function showConfirmOverlay() {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'i3-reset-confirm-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.innerHTML = `
            <div class="i3-reset-confirm-box">
                <h3>Reset session</h3>
                <p>This clears working memory and persona bias for the current session.</p>
                <div class="i3-reset-confirm-countdown">1.5</div>
                <div class="i3-reset-confirm-actions">
                    <button type="button" class="cancel">Cancel</button>
                    <button type="button" class="confirm">Confirm</button>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        requestAnimationFrame(() => overlay.classList.add('visible'));

        const countdownEl = overlay.querySelector('.i3-reset-confirm-countdown');
        const confirmBtn = overlay.querySelector('.confirm');
        const cancelBtn = overlay.querySelector('.cancel');

        const start = performance.now();
        let rafId = null;
        let done = false;

        const finish = (value) => {
            if (done) return;
            done = true;
            if (rafId !== null) cancelAnimationFrame(rafId);
            overlay.classList.remove('visible');
            setTimeout(() => overlay.remove(), 220);
            resolve(value);
        };

        function tick() {
            const elapsed = performance.now() - start;
            const remaining = Math.max(0, HOLD_MS - elapsed);
            countdownEl.textContent = (remaining / 1000).toFixed(1);
            if (remaining <= 0) {
                // After the hold window, auto-enable confirm animation.
                countdownEl.textContent = '0.0';
                return;
            }
            rafId = requestAnimationFrame(tick);
        }
        rafId = requestAnimationFrame(tick);

        confirmBtn.addEventListener('click', () => {
            const elapsed = performance.now() - start;
            if (elapsed < HOLD_MS) {
                // Reject early clicks — the whole point of the 1.5s hold.
                confirmBtn.animate(
                    [{ transform: 'translateX(0)' }, { transform: 'translateX(-4px)' },
                     { transform: 'translateX(4px)' }, { transform: 'translateX(0)' }],
                    { duration: 180 }
                );
                return;
            }
            finish(true);
        });
        cancelBtn.addEventListener('click', () => finish(false));
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) finish(false);
        });
        const keyHandler = (e) => {
            if (e.key === 'Escape') finish(false);
        };
        document.addEventListener('keydown', keyHandler, { once: true });
    });
}

/**
 * Attach the floating reset button.
 */
export function initResetButton() {
    if (document.querySelector('.i3-reset-btn')) return;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'i3-reset-btn';
    btn.textContent = 'Reset session';
    btn.setAttribute('aria-label', 'Reset the current interaction session');
    document.body.appendChild(btn);

    btn.addEventListener('click', async () => {
        const confirmed = await showConfirmOverlay();
        if (!confirmed) return;
        btn.disabled = true;
        btn.textContent = 'Resetting...';
        try {
            await postReset();
            // Small delay so the user sees the confirmation before reload.
            setTimeout(() => window.location.reload(), 350);
        } catch (err) {
            btn.textContent = 'Reset failed';
            setTimeout(() => {
                btn.disabled = false;
                btn.textContent = 'Reset session';
            }, 1800);
        }
    });

    return btn;
}
