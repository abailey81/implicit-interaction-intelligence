/**
 * advanced_init.js -- I3 Advanced Panels entry point
 *
 * Imports and initialises the optional advanced insights UI:
 *   - Cross-attention heatmap (attention_viz.js)
 *   - What-if comparator       (whatif.js)
 *   - Persona switcher         (persona_switcher.js)
 *   - Floating reset pill      (reset_button.js)
 *   - WCAG 2.2 audit script    (wcag_audit.js)
 *
 * The panels can be disabled at runtime by setting
 *   localStorage.setItem('i3AdvancedUI', '0')
 * (default is enabled). Press Alt+A to toggle the drawer.
 */

import { initAttentionViz } from '/static/js/attention_viz.js';
import { initWhatIfPanel } from '/static/js/whatif.js';
import { initPersonaSwitcher } from '/static/js/persona_switcher.js';
import { initResetButton } from '/static/js/reset_button.js';
// wcag_audit is self-running; the side-effect import is sufficient.
import '/static/js/wcag_audit.js';

const STORAGE_KEY = 'i3AdvancedUI';

function isEnabled() {
    try {
        const v = localStorage.getItem(STORAGE_KEY);
        return v === null || v === '1' || v === 'true';
    } catch (e) {
        return true;
    }
}

function buildDrawer(root) {
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'i3-adv-toggle';
    toggle.textContent = 'INSIGHTS';
    toggle.setAttribute('aria-label', 'Open advanced insights drawer');
    toggle.setAttribute('aria-expanded', 'false');
    root.appendChild(toggle);

    const drawer = document.createElement('aside');
    drawer.className = 'i3-adv-drawer';
    drawer.setAttribute('role', 'complementary');
    drawer.setAttribute('aria-label', 'Advanced insights');
    drawer.innerHTML = `
        <div class="i3-adv-drawer-header">
            <h2>Advanced insights</h2>
            <button type="button" class="i3-adv-drawer-close" aria-label="Close insights drawer">Close</button>
        </div>
        <div class="i3-adv-drawer-body"></div>
    `;
    root.appendChild(drawer);

    const body = drawer.querySelector('.i3-adv-drawer-body');
    const closeBtn = drawer.querySelector('.i3-adv-drawer-close');

    function open() {
        drawer.classList.add('open');
        toggle.setAttribute('aria-expanded', 'true');
    }
    function close() {
        drawer.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
    }
    function toggleDrawer() {
        if (drawer.classList.contains('open')) close();
        else open();
    }

    toggle.addEventListener('click', toggleDrawer);
    closeBtn.addEventListener('click', close);

    // Alt+A keyboard shortcut.
    document.addEventListener('keydown', (e) => {
        if (e.altKey && (e.key === 'a' || e.key === 'A')) {
            e.preventDefault();
            toggleDrawer();
        }
    });

    return { drawer, body, toggle, open, close, toggleDrawer };
}

function bootstrap() {
    if (!isEnabled()) {
        return;
    }
    const root = document.getElementById('i3-advanced-panels');
    if (!root) return;

    const { body } = buildDrawer(root);

    // Order: attention first (centrepiece), then what-if, then persona.
    try { initAttentionViz(body); } catch (e) { console.warn('[I3] attention viz failed', e); }
    try { initWhatIfPanel(body); } catch (e) { console.warn('[I3] whatif failed', e); }
    try { initPersonaSwitcher(body); } catch (e) { console.warn('[I3] persona switcher failed', e); }
    try { initResetButton(); } catch (e) { console.warn('[I3] reset button failed', e); }

    // Convenience API for the demo operator.
    window.i3Advanced = {
        enable() {
            try { localStorage.setItem(STORAGE_KEY, '1'); } catch (e) {}
            window.location.reload();
        },
        disable() {
            try { localStorage.setItem(STORAGE_KEY, '0'); } catch (e) {}
            window.location.reload();
        },
    };
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
} else {
    bootstrap();
}
