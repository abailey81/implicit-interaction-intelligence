/**
 * persona_switcher.js -- I3 Advanced Panels
 *
 * Chip row that temporarily biases the user profile toward one of four
 * archetypes. Clicking a chip POSTs to `/admin/persona/{name}` with the
 * admin bearer token from localStorage (if present) and marks the chip
 * active. The backend endpoint is a stub elsewhere in the project; this
 * frontend wires the call regardless.
 *
 * Export: initPersonaSwitcher(container)
 */

const PERSONAS = [
    { id: 'neutral', label: 'Neutral' },
    { id: 'energetic', label: 'Energetic' },
    { id: 'fatigued', label: 'Fatigued' },
    { id: 'a11y_sensitive', label: 'Accessibility-sensitive' },
];

const FETCH_TIMEOUT_MS = 4000;

function adminToken() {
    try {
        return localStorage.getItem('i3AdminToken') || '';
    } catch (e) {
        return '';
    }
}

async function postPersona(name) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    try {
        const headers = { 'Content-Type': 'application/json' };
        const tok = adminToken();
        if (tok) headers['Authorization'] = `Bearer ${tok}`;
        const res = await fetch(`/admin/persona/${encodeURIComponent(name)}`, {
            method: 'POST',
            credentials: 'same-origin',
            headers,
            body: JSON.stringify({ persona: name }),
            signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`status ${res.status}`);
        return await res.json().catch(() => ({}));
    } finally {
        clearTimeout(to);
    }
}

/**
 * Initialise the persona chip row inside the given container.
 * @param {HTMLElement} container
 */
export function initPersonaSwitcher(container) {
    if (!container) return;

    const section = document.createElement('section');
    section.className = 'i3-adv-section';
    section.innerHTML = `
        <div class="i3-adv-section-title">
            Persona Override
            <span class="i3-adv-section-hint">temporary profile bias</span>
        </div>
        <div class="i3-persona-row" role="group" aria-label="Persona selector"></div>
        <div class="i3-persona-status" aria-live="polite"
             style="font-family:var(--font-mono);font-size:10px;color:var(--text-dim);margin-top:8px;"></div>
    `;
    container.appendChild(section);

    const row = section.querySelector('.i3-persona-row');
    const status = section.querySelector('.i3-persona-status');

    const chips = {};
    let activeId = 'neutral';

    for (const p of PERSONAS) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'i3-persona-chip';
        chip.textContent = p.label;
        chip.setAttribute('data-persona', p.id);
        chip.setAttribute('aria-pressed', 'false');
        if (p.id === activeId) {
            chip.classList.add('active');
            chip.setAttribute('aria-pressed', 'true');
        }
        chip.addEventListener('click', async () => {
            if (chip.disabled) return;
            Object.values(chips).forEach((c) => {
                c.disabled = true;
            });
            status.textContent = `applying ${p.label}...`;
            try {
                await postPersona(p.id);
                activeId = p.id;
                for (const [id, el] of Object.entries(chips)) {
                    const on = id === activeId;
                    el.classList.toggle('active', on);
                    el.setAttribute('aria-pressed', on ? 'true' : 'false');
                }
                status.textContent = `active: ${p.label}`;
            } catch (err) {
                status.textContent = `failed: ${String(err.message || err).slice(0, 80)}`;
            } finally {
                Object.values(chips).forEach((c) => {
                    c.disabled = false;
                });
            }
        });
        chips[p.id] = chip;
        row.appendChild(chip);
    }

    return { section, chips };
}
