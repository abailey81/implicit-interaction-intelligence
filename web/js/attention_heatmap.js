/**
 * Attention heatmap visualiser (State tab).
 *
 * Fetches real self-attention weights from ``GET /api/attention`` and
 * renders them as an HTML-table heatmap. A layer selector lets the user
 * switch between transformer layers (last layer is the default — it's
 * typically the most semantic).
 *
 * The endpoint does a CPU-only forward pass through the custom SLM so
 * this panel works while the GPU is busy with training. To avoid running
 * the model on every page visit, the endpoint caches the most recent
 * result; the "Compute" button is what actually re-runs the model.
 */

(function () {
    'use strict';

    // SEC: Restrict the prompt to the same bounded-length regime used by
    // the server so an oversized client input is truncated before fetch.
    const MAX_PROMPT_CHARS = 256;

    // Current cached payload so the layer selector can re-render without
    // refetching.
    let current = null;

    function $(id) { return document.getElementById(id); }

    function setStatus(text, kind) {
        const el = $('attn-status');
        if (!el) return;
        el.textContent = text || '';
        el.classList.remove('attn-status-error', 'attn-status-busy', 'attn-status-ok');
        if (kind) el.classList.add(`attn-status-${kind}`);
    }

    function setCaption(text) {
        const el = $('attn-caption-meta');
        if (el) el.textContent = text || '—';
    }

    /**
     * Render one layer of the attention heatmap.
     * @param {object} payload  Response body from /api/attention.
     * @param {number} layerIdx Index into payload.mean_per_layer.
     */
    function renderGrid(payload, layerIdx) {
        const grid = $('attn-grid');
        if (!grid) return;
        grid.innerHTML = '';

        const tokens = Array.isArray(payload.tokens) ? payload.tokens : [];
        const mean = Array.isArray(payload.mean_per_layer) ? payload.mean_per_layer : [];
        const layer = (layerIdx >= 0 && layerIdx < mean.length) ? mean[layerIdx] : null;

        if (!layer || !tokens.length) {
            grid.textContent = '(no attention data)';
            return;
        }

        const seqLen = tokens.length;

        // Header row: blank corner + one column per token.
        const headRow = document.createElement('div');
        headRow.className = 'attn-row attn-row-head';
        const corner = document.createElement('div');
        corner.className = 'attn-cell attn-corner';
        headRow.appendChild(corner);
        for (let j = 0; j < seqLen; j++) {
            const c = document.createElement('div');
            c.className = 'attn-cell attn-col-label';
            // SEC: textContent only — token strings come from the server but
            // are rendered as plain text.
            c.textContent = String(tokens[j] || '').slice(0, 16);
            c.title = String(tokens[j] || '');
            headRow.appendChild(c);
        }
        grid.appendChild(headRow);

        // One row per query token.
        for (let i = 0; i < seqLen; i++) {
            const row = document.createElement('div');
            row.className = 'attn-row';
            const rowLabel = document.createElement('div');
            rowLabel.className = 'attn-cell attn-row-label';
            rowLabel.textContent = String(tokens[i] || '').slice(0, 16);
            rowLabel.title = String(tokens[i] || '');
            row.appendChild(rowLabel);
            const weights = Array.isArray(layer[i]) ? layer[i] : [];
            for (let j = 0; j < seqLen; j++) {
                const wRaw = Number(weights[j]);
                // SEC: Clamp to [0, 1] — softmax rows are already in range
                // but a network hiccup could produce NaN/negative values.
                const w = Number.isFinite(wRaw) ? Math.max(0, Math.min(1, wRaw)) : 0;
                const cell = document.createElement('div');
                cell.className = 'attn-cell attn-value';
                // Apple-blue scaled by weight — same palette used elsewhere.
                cell.style.backgroundColor = `rgba(41, 151, 255, ${w.toFixed(3)})`;
                cell.title = `${tokens[i]} → ${tokens[j]}: ${w.toFixed(3)}`;
                // Only draw the number for non-trivial weights to keep the
                // grid readable at small sizes.
                if (w >= 0.12) {
                    cell.textContent = w.toFixed(2);
                }
                row.appendChild(cell);
            }
            grid.appendChild(row);
        }
    }

    /**
     * Populate the layer selector, defaulting to the last layer.
     * @param {object} payload
     */
    function populateLayerSelect(payload) {
        const sel = $('attn-layer');
        if (!sel) return;
        const nLayers = Number(payload.n_layers) || 0;
        const previousIdx = Number(sel.value);
        sel.innerHTML = '';
        for (let i = 0; i < nLayers; i++) {
            const opt = document.createElement('option');
            opt.value = String(i);
            opt.textContent = `Layer ${i}${i === nLayers - 1 ? ' (final)' : ''}`;
            sel.appendChild(opt);
        }
        // Prefer last layer (semantic) on first render, but preserve the
        // user's pick across re-renders when possible.
        const defaultIdx = nLayers > 0 ? nLayers - 1 : 0;
        const wanted = Number.isFinite(previousIdx) && previousIdx >= 0 && previousIdx < nLayers
            ? previousIdx
            : defaultIdx;
        sel.value = String(wanted);
    }

    function refreshCaption(payload) {
        if (!payload) { setCaption('—'); return; }
        const bits = [];
        if (payload.synthetic) {
            bits.push('synthetic fallback (SLM not loaded)');
        } else {
            if (payload.n_layers) bits.push(`${payload.n_layers} layers`);
            if (payload.n_heads) bits.push(`${payload.n_heads} heads`);
            if (payload.seq_len) bits.push(`seq ${payload.seq_len}`);
        }
        setCaption(bits.join(' · '));
    }

    async function fetchAttention({ compute }) {
        const input = $('attn-prompt');
        let text = (input && input.value) ? String(input.value).slice(0, MAX_PROMPT_CHARS) : '';
        if (!text.trim()) text = 'Hello, how are you?';

        const qs = new URLSearchParams({
            text,
            compute: compute ? 'true' : 'false',
        });
        setStatus(compute ? 'Computing on CPU…' : 'Loading…', 'busy');

        try {
            const res = await fetch(`/api/attention?${qs.toString()}`, {
                credentials: 'same-origin',
                cache: 'no-store',
            });
            if (!res.ok) {
                setStatus(`Failed (${res.status})`, 'error');
                return;
            }
            const payload = await res.json();
            current = payload;
            populateLayerSelect(payload);
            const sel = $('attn-layer');
            const layerIdx = sel ? Number(sel.value) : 0;
            renderGrid(payload, Number.isFinite(layerIdx) ? layerIdx : 0);
            refreshCaption(payload);
            setStatus(
                payload.synthetic
                    ? 'Showing synthetic fallback — the SLM checkpoint is not loaded.'
                    : 'Attention weights extracted on CPU.',
                payload.synthetic ? 'error' : 'ok',
            );
        } catch (err) {
            console.error('[attn] fetch failed', err);
            setStatus('Network error', 'error');
        }
    }

    function init() {
        const wrap = $('attn-viz');
        if (!wrap) return;  // panel not present on this page

        const btn = $('attn-compute');
        if (btn) {
            btn.addEventListener('click', () => fetchAttention({ compute: true }));
        }

        const input = $('attn-prompt');
        if (input) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    fetchAttention({ compute: true });
                }
            });
        }

        const sel = $('attn-layer');
        if (sel) {
            sel.addEventListener('change', () => {
                if (!current) return;
                const idx = Number(sel.value);
                renderGrid(current, Number.isFinite(idx) ? idx : 0);
            });
        }

        // Read-through load on first paint so the grid is never empty.
        fetchAttention({ compute: false });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
