/**
 * Live system-architecture Flow dashboard.
 *
 * Third flagship surface for I3 — renders an animated diagram of every
 * pipeline stage firing in real time as the user types.  Every box
 * pulses when its component fires; every arrow shows the data being
 * passed (with real measurements: latency, dimensions, scores).
 *
 * Driven by the WebSocket ``response`` / ``response_done`` frames the
 * server now ships with a ``pipeline_trace`` field.  No polling, no
 * synthetic timings, no chart libs — pure SVG + DOM nodes painted
 * inside ``requestAnimationFrame`` so the main chat thread stays free.
 *
 * Layout (4 rows, fixed positions so the SVG arrow geometry is stable):
 *
 *   Row 1 (input):       interaction · multimodal_fusion · biometric
 *   Row 2 (encoders):    encoder
 *   Row 3 (state):       adaptation · personalisation · entity_tracker ·
 *                        state_classifier · affect_shift · accessibility
 *   Row 4 (decision):    router · generation · critique · postprocess
 */

(function () {
    'use strict';

    // ------------------------------------------------------------------
    // Stage layout — keyed by stage_id from the server-side
    // PipelineTraceCollector.  Coordinates are in a 1000x600 viewBox
    // so the SVG scales proportionally to the panel width.
    // ------------------------------------------------------------------
    const STAGE_LAYOUT = {
        // Row 1 — input signals
        interaction:         { x:  80, y:  60, w: 180, h: 70, row: 1 },
        multimodal_fusion:   { x: 410, y:  60, w: 180, h: 70, row: 1 },
        biometric:           { x: 740, y:  60, w: 180, h: 70, row: 1 },
        // Row 2 — encoders
        encoder:             { x: 240, y: 180, w: 180, h: 70, row: 2 },
        // Row 3 — state
        adaptation:          { x:  20, y: 300, w: 160, h: 70, row: 3 },
        personalisation:     { x: 200, y: 300, w: 160, h: 70, row: 3 },
        entity_tracker:      { x: 380, y: 300, w: 160, h: 70, row: 3 },
        state_classifier:    { x: 560, y: 300, w: 160, h: 70, row: 3 },
        affect_shift:        { x: 740, y: 300, w: 100, h: 70, row: 3 },
        accessibility:       { x: 850, y: 300, w: 130, h: 70, row: 3 },
        // Row 4 — decision
        router:              { x:  60, y: 460, w: 180, h: 70, row: 4 },
        generation:          { x: 290, y: 460, w: 240, h: 70, row: 4 },
        critique:            { x: 580, y: 460, w: 160, h: 70, row: 4 },
        postprocess:         { x: 770, y: 460, w: 180, h: 70, row: 4 },
    };

    // Default labels used when the trace dict omits one (defensive).
    const DEFAULT_LABELS = {
        interaction: 'Interaction monitor',
        encoder: 'TCN encoder',
        multimodal_fusion: 'Multimodal fusion',
        adaptation: 'Adaptation controller',
        personalisation: 'Personal LoRA',
        biometric: 'Identity Lock',
        state_classifier: 'State classifier',
        affect_shift: 'Affect shift',
        accessibility: 'Accessibility',
        entity_tracker: 'Entity tracker',
        router: 'Router (LinUCB)',
        generation: 'Generation',
        critique: 'Self-critique',
        postprocess: 'Post-process',
    };

    const SVG_NS = 'http://www.w3.org/2000/svg';

    let lastTrace = null;
    let recentTraces = [];
    let pulseSeq = 0;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    function el(tag, attrs, children) {
        const node = document.createElement(tag);
        if (attrs) {
            for (const k of Object.keys(attrs)) {
                if (k === 'class') node.className = attrs[k];
                else if (k === 'text') node.textContent = attrs[k];
                else node.setAttribute(k, attrs[k]);
            }
        }
        if (children) {
            for (const c of children) {
                if (c) node.appendChild(c);
            }
        }
        return node;
    }

    function svgEl(tag, attrs) {
        const node = document.createElementNS(SVG_NS, tag);
        if (attrs) {
            for (const k of Object.keys(attrs)) {
                node.setAttribute(k, attrs[k]);
            }
        }
        return node;
    }

    function fmtMs(v) {
        if (typeof v !== 'number' || !isFinite(v)) return '— ms';
        if (v < 1) return v.toFixed(2) + ' ms';
        if (v < 10) return v.toFixed(2) + ' ms';
        if (v < 100) return v.toFixed(1) + ' ms';
        return Math.round(v) + ' ms';
    }

    function escapeHtml(s) {
        if (s == null) return '';
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // ------------------------------------------------------------------
    // SVG renderer
    // ------------------------------------------------------------------

    function ensureCanvas(container) {
        let svg = container.querySelector('svg.flow-svg');
        if (svg) return svg;

        svg = document.createElementNS(SVG_NS, 'svg');
        svg.setAttribute('class', 'flow-svg');
        svg.setAttribute('viewBox', '0 0 1000 580');
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        svg.setAttribute('aria-label', 'Live pipeline flow diagram');

        // Arrowhead marker (drawn once, referenced by every arrow).
        const defs = svgEl('defs');
        const marker = svgEl('marker', {
            id: 'flow-arrow-head',
            viewBox: '0 0 10 10',
            refX: '9', refY: '5',
            markerWidth: '6', markerHeight: '6',
            orient: 'auto-start-reverse',
        });
        marker.appendChild(svgEl('path', {
            d: 'M 0 0 L 10 5 L 0 10 z',
            fill: 'rgba(170, 200, 255, 0.7)',
        }));
        defs.appendChild(marker);
        svg.appendChild(defs);

        // Layer groups so arrows always sit beneath nodes.
        const arrowsLayer = svgEl('g', { class: 'flow-arrows' });
        const nodesLayer = svgEl('g', { class: 'flow-nodes' });
        svg.appendChild(arrowsLayer);
        svg.appendChild(nodesLayer);

        container.appendChild(svg);
        return svg;
    }

    function renderEmpty(container, message) {
        container.innerHTML =
            '<div class="flow-empty">' + escapeHtml(message) + '</div>';
    }

    /**
     * Paint the entire flow diagram from a pipeline_trace dict.
     */
    function renderTrace(trace) {
        const canvas = document.getElementById('flow-canvas');
        if (!canvas) return;

        if (!trace || !trace.stages) {
            renderEmpty(canvas, 'Send a message to see the pipeline fire.');
            updateStatsBar(null);
            return;
        }

        // Wipe and rebuild — diagrams are tiny so this is cheaper
        // than diffing.
        canvas.innerHTML = '';
        const svg = ensureCanvas(canvas);
        const arrowsLayer = svg.querySelector('g.flow-arrows');
        const nodesLayer = svg.querySelector('g.flow-nodes');

        // Build a quick lookup of stage records.
        const byId = {};
        for (const s of trace.stages) {
            byId[s.stage_id] = s;
        }

        // ---- Draw arrows first (under the nodes) -------------------
        const arrows = trace.arrow_flows || [];
        for (const a of arrows) {
            const from = STAGE_LAYOUT[a.from];
            const to = STAGE_LAYOUT[a.to];
            if (!from || !to) continue;

            // Connect from bottom-centre / right-centre of source to
            // top-centre / left-centre of target depending on relative
            // row numbers.
            const fromRec = byId[a.from];
            const toRec = byId[a.to];
            const fired = fromRec && fromRec.fired && toRec && toRec.fired;

            const path = svgEl('path', {
                d: arrowPath(from, to),
                class: 'flow-arrow' + (fired ? ' fired' : ''),
                'marker-end': 'url(#flow-arrow-head)',
            });
            arrowsLayer.appendChild(path);

            // Animate the dashes if the arrow fired.
            if (fired) {
                animateArrow(path, pulseSeq++);
            }

            // Label at the midpoint.
            if (a.payload_summary) {
                const mid = midpoint(from, to);
                const label = svgEl('text', {
                    x: mid.x, y: mid.y - 4,
                    class: 'flow-arrow-label' + (fired ? ' fired' : ''),
                    'text-anchor': 'middle',
                });
                label.textContent = a.payload_summary;
                arrowsLayer.appendChild(label);
            }
        }

        // ---- Draw stage nodes -------------------------------------
        const orderedStages = trace.stages.slice().sort(
            (a, b) => (a.started_at_ms || 0) - (b.started_at_ms || 0)
        );

        for (const stage of trace.stages) {
            const layout = STAGE_LAYOUT[stage.stage_id];
            if (!layout) continue;
            const node = renderStageNode(stage, layout);
            nodesLayer.appendChild(node);
        }

        // Draw any stages we know about that weren't in the trace as
        // dimmed placeholders so the layout never collapses.
        for (const stageId of Object.keys(STAGE_LAYOUT)) {
            if (byId[stageId]) continue;
            const placeholder = renderStageNode(
                {
                    stage_id: stageId,
                    label: DEFAULT_LABELS[stageId] || stageId,
                    fired: false,
                    latency_ms: 0,
                    inputs: {}, outputs: {}, notes: 'not run this turn',
                    is_tool: false,
                },
                STAGE_LAYOUT[stageId],
            );
            nodesLayer.appendChild(placeholder);
        }

        // ---- Animated firing sequence -----------------------------
        // Pulse each fired stage in the order they actually started,
        // staggered so the eye can follow the wave through the diagram.
        // We use rAF + setTimeout for the stagger so we don't block.
        requestAnimationFrame(() => {
            let delay = 0;
            for (const stage of orderedStages) {
                if (!stage.fired) continue;
                const g = nodesLayer.querySelector(
                    'g.flow-node[data-stage="' + stage.stage_id + '"]'
                );
                if (!g) continue;
                setTimeout(() => {
                    g.classList.remove('pulsing');
                    // Trigger reflow so the animation restarts.
                    void g.getBoundingClientRect();
                    g.classList.add('pulsing');
                }, delay);
                delay += 80;
            }
        });

        updateStatsBar(trace);
    }

    function renderStageNode(stage, layout) {
        const fired = !!stage.fired;
        const isTool = !!stage.is_tool;
        const cls =
            'flow-node' +
            (fired ? ' fired' : ' dimmed') +
            (isTool ? ' tool' : '');

        const g = svgEl('g', {
            class: cls,
            transform: 'translate(' + layout.x + ',' + layout.y + ')',
            'data-stage': stage.stage_id,
            tabindex: '0',
            role: 'button',
            'aria-label':
                (stage.label || DEFAULT_LABELS[stage.stage_id] ||
                 stage.stage_id) +
                ' — ' + (fired ? fmtMs(stage.latency_ms) : 'not run'),
        });

        // Outer background rect.
        g.appendChild(svgEl('rect', {
            x: 0, y: 0,
            width: layout.w, height: layout.h,
            rx: 12, ry: 12,
            class: 'flow-node-bg',
        }));

        // Title text (label).
        const titleText = stage.label ||
                          DEFAULT_LABELS[stage.stage_id] ||
                          stage.stage_id;
        const title = svgEl('text', {
            x: 12, y: 22,
            class: 'flow-node-title',
        });
        title.textContent = titleText;
        g.appendChild(title);

        // Latency text.
        const lat = svgEl('text', {
            x: 12, y: 42,
            class: 'flow-node-lat',
        });
        lat.textContent = fired ? fmtMs(stage.latency_ms) : 'idle';
        g.appendChild(lat);

        // One-line summary on the right (output dimensions / score / state).
        const summary = svgEl('text', {
            x: layout.w - 12, y: 42,
            class: 'flow-node-summary',
            'text-anchor': 'end',
        });
        summary.textContent = oneLineSummary(stage);
        g.appendChild(summary);

        // Click handler -> reveal detail panel.
        g.addEventListener('click', () => showStageDetail(stage));
        g.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') {
                ev.preventDefault();
                showStageDetail(stage);
            }
        });

        return g;
    }

    function oneLineSummary(stage) {
        const o = stage.outputs || {};
        if (stage.stage_id === 'encoder') {
            return (o.embedding_dim || 64) + '-d';
        }
        if (stage.stage_id === 'adaptation') {
            return (o.axes || 8) + ' axes';
        }
        if (stage.stage_id === 'router') {
            return o.route ? '→ ' + String(o.route) : '';
        }
        if (stage.stage_id === 'generation') {
            const path = o.path || '';
            return path ? path.replace(/^tool:/, 'tool: ') : '';
        }
        if (stage.stage_id === 'biometric') {
            return o.state ? String(o.state) : '';
        }
        if (stage.stage_id === 'state_classifier') {
            return o.state ? String(o.state) : '';
        }
        if (stage.stage_id === 'multimodal_fusion') {
            return o.prosody_active ? 'prosody on' : '96-d';
        }
        if (stage.stage_id === 'personalisation') {
            return o.applied ? 'applied' : 'base';
        }
        if (stage.stage_id === 'critique') {
            const sc = typeof o.score === 'number'
                ? o.score.toFixed(2) : '';
            return sc ? 'score=' + sc : '';
        }
        if (stage.stage_id === 'affect_shift') {
            return o.detected ? 'shift!' : 'stable';
        }
        if (stage.stage_id === 'accessibility') {
            return o.active ? 'active' : 'off';
        }
        return '';
    }

    /**
     * Compute SVG path data for an arrow from box ``a`` to box ``b``.
     * Uses a simple cubic Bézier with vertical control offsets so
     * arrows curve gently between rows and hop neatly within a row.
     */
    function arrowPath(a, b) {
        const ax = a.x + a.w / 2;
        const ay = a.y + a.h;       // bottom of source
        const bx = b.x + b.w / 2;
        const by = b.y;             // top of target

        if (a.row === b.row) {
            // Horizontal hop — exit right, enter left.
            const ax2 = a.x + a.w;
            const ay2 = a.y + a.h / 2;
            const bx2 = b.x;
            const by2 = b.y + b.h / 2;
            const c1x = ax2 + 30;
            const c1y = ay2;
            const c2x = bx2 - 30;
            const c2y = by2;
            return [
                'M', ax2, ay2,
                'C', c1x, c1y, c2x, c2y, bx2, by2,
            ].join(' ');
        }

        // Cross-row: exit bottom, enter top with a vertical Bézier.
        const c1x = ax;
        const c1y = ay + 60;
        const c2x = bx;
        const c2y = by - 60;
        return [
            'M', ax, ay,
            'C', c1x, c1y, c2x, c2y, bx, by,
        ].join(' ');
    }

    function midpoint(a, b) {
        if (a.row === b.row) {
            return {
                x: (a.x + a.w + b.x) / 2,
                y: a.y + a.h / 2,
            };
        }
        return {
            x: (a.x + a.w / 2 + b.x + b.w / 2) / 2,
            y: (a.y + a.h + b.y) / 2,
        };
    }

    function animateArrow(path, idx) {
        // Simple dashed-stroke animation kicked off by toggling the
        // class so the CSS keyframes restart.  The stagger keeps the
        // wave readable.
        const delay = (idx % 12) * 60;
        setTimeout(() => {
            path.classList.remove('flowing');
            void path.getBoundingClientRect();
            path.classList.add('flowing');
        }, delay);
    }

    // ------------------------------------------------------------------
    // Detail panel
    // ------------------------------------------------------------------

    function showStageDetail(stage) {
        const stats = document.getElementById('flow-stats');
        if (!stats) return;
        const o = stage.outputs || {};
        const i = stage.inputs || {};
        const rows = [];
        rows.push(
            '<div class="flow-detail">' +
            '<div class="flow-detail-head"><span class="flow-detail-id">' +
            escapeHtml(stage.stage_id) + '</span> <span class="flow-detail-label">' +
            escapeHtml(stage.label || '') + '</span></div>' +
            '<div class="flow-detail-row">' +
            '<span class="flow-detail-key">latency</span>' +
            '<span class="flow-detail-val">' + escapeHtml(fmtMs(stage.latency_ms)) + '</span>' +
            '</div>' +
            '<div class="flow-detail-row">' +
            '<span class="flow-detail-key">started</span>' +
            '<span class="flow-detail-val">' +
            escapeHtml(fmtMs(stage.started_at_ms)) + ' from turn start</span>' +
            '</div>' +
            '<div class="flow-detail-row">' +
            '<span class="flow-detail-key">fired</span>' +
            '<span class="flow-detail-val">' + (stage.fired ? 'yes' : 'no') + '</span>' +
            '</div>'
        );
        if (stage.notes) {
            rows.push(
                '<div class="flow-detail-row notes">' +
                escapeHtml(stage.notes) + '</div>'
            );
        }
        if (Object.keys(i).length) {
            rows.push('<div class="flow-detail-sub">inputs</div>');
            for (const k of Object.keys(i)) {
                rows.push(
                    '<div class="flow-detail-row">' +
                    '<span class="flow-detail-key">' + escapeHtml(k) + '</span>' +
                    '<span class="flow-detail-val">' + escapeHtml(JSON.stringify(i[k])) + '</span>' +
                    '</div>'
                );
            }
        }
        if (Object.keys(o).length) {
            rows.push('<div class="flow-detail-sub">outputs</div>');
            for (const k of Object.keys(o)) {
                rows.push(
                    '<div class="flow-detail-row">' +
                    '<span class="flow-detail-key">' + escapeHtml(k) + '</span>' +
                    '<span class="flow-detail-val">' + escapeHtml(JSON.stringify(o[k])) + '</span>' +
                    '</div>'
                );
            }
        }
        rows.push('</div>');
        stats.innerHTML = rows.join('');
    }

    // ------------------------------------------------------------------
    // Stats bar (totals + history)
    // ------------------------------------------------------------------

    function updateStatsBar(trace) {
        const stats = document.getElementById('flow-stats');
        if (!stats) return;
        if (!trace) {
            stats.innerHTML =
                '<p class="flow-empty-stats">' +
                'Click on a stage to inspect its inputs / outputs.</p>';
            return;
        }
        const fired = trace.stages.filter(s => s.fired).length;
        const total = trace.stages.length;
        const arrowsCount = (trace.arrow_flows || []).length;
        const html =
            '<div class="flow-summary">' +
            '<div class="flow-summary-card">' +
            '<div class="flow-summary-label">total latency</div>' +
            '<div class="flow-summary-val">' +
                escapeHtml(fmtMs(trace.total_latency_ms)) + '</div>' +
            '</div>' +
            '<div class="flow-summary-card">' +
            '<div class="flow-summary-label">stages fired</div>' +
            '<div class="flow-summary-val">' +
                fired + ' / ' + total + '</div>' +
            '</div>' +
            '<div class="flow-summary-card">' +
            '<div class="flow-summary-label">arrows</div>' +
            '<div class="flow-summary-val">' + arrowsCount + '</div>' +
            '</div>' +
            '<div class="flow-summary-card">' +
            '<div class="flow-summary-label">turn id</div>' +
            '<div class="flow-summary-val mono">' +
                escapeHtml((trace.turn_id || '').slice(0, 8)) + '</div>' +
            '</div>' +
            '</div>' +
            '<p class="flow-tip">Click on any stage box to inspect inputs / outputs.</p>';
        stats.innerHTML = html;
    }

    // ------------------------------------------------------------------
    // Recent-turns history table
    // ------------------------------------------------------------------

    function renderHistory() {
        const tbody = document.getElementById('flow-history-body');
        if (!tbody) return;
        if (!recentTraces.length) {
            tbody.innerHTML =
                '<tr><td colspan="5" class="flow-history-empty">No turns yet.</td></tr>';
            return;
        }
        const rows = recentTraces.slice(0, 10).map((t, idx) => {
            const fired = (t.stages || []).filter(s => s.fired).length;
            const total = (t.stages || []).length;
            // Derive the response_path from the generation stage outputs.
            const gen = (t.stages || []).find(s => s.stage_id === 'generation');
            const path = gen && gen.outputs ? (gen.outputs.path || '—') : '—';
            const tid = escapeHtml((t.turn_id || '').slice(0, 8));
            return (
                '<tr data-turn-id="' + escapeHtml(t.turn_id || '') + '">' +
                '<td>' + tid + '</td>' +
                '<td>' + escapeHtml(fmtMs(t.total_latency_ms)) + '</td>' +
                '<td>' + fired + ' / ' + total + '</td>' +
                '<td>' + escapeHtml(path) + '</td>' +
                '<td><button class="flow-replay-btn" data-idx="' + idx + '">Replay</button></td>' +
                '</tr>'
            );
        });
        tbody.innerHTML = rows.join('');

        // Wire replay buttons.
        const buttons = tbody.querySelectorAll('.flow-replay-btn');
        for (const btn of buttons) {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.getAttribute('data-idx'), 10);
                if (!Number.isNaN(idx) && recentTraces[idx]) {
                    renderTrace(recentTraces[idx]);
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // WS frame intake
    // ------------------------------------------------------------------

    function ingestFrame(data) {
        if (!data || !data.pipeline_trace) return;
        const trace = data.pipeline_trace;
        lastTrace = trace;
        recentTraces.unshift(trace);
        if (recentTraces.length > 50) recentTraces.length = 50;
        // Only render immediately if the Flow tab is currently visible
        // OR has never been rendered.  The first trace lands, then
        // subsequent ones repaint when the user opens the tab.
        const panel = document.getElementById('tab-flow');
        if (panel && !panel.hidden) {
            renderTrace(trace);
            renderHistory();
        }
    }

    function paintIfNeeded() {
        // Triggered when the Flow tab becomes visible.
        if (!lastTrace) {
            renderEmpty(
                document.getElementById('flow-canvas'),
                'Send a message to see the pipeline fire.'
            );
            updateStatsBar(null);
        } else {
            renderTrace(lastTrace);
        }
        renderHistory();
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------

    function init() {
        const canvas = document.getElementById('flow-canvas');
        if (!canvas) return;

        renderEmpty(canvas, 'Send a message to see the pipeline fire.');
        updateStatsBar(null);
        renderHistory();

        // Subscribe to WS frames.  ``window.app.wsClient`` is the
        // shared WebSocket client created in app.js — we hook into
        // both 'response' and 'response_done' so retrieval / SLM
        // paths both repaint.
        const tryHook = () => {
            const app = window.app;
            const ws = app && app.wsClient;
            if (!ws || typeof ws.on !== 'function') {
                setTimeout(tryHook, 200);
                return;
            }
            ws.on('response', ingestFrame);
            ws.on('response_done', ingestFrame);
        };
        tryHook();

        // Repaint when the user activates the tab.  We listen to the
        // standard tab_router CustomEvent + a hashchange fallback.
        window.addEventListener('hashchange', () => {
            if ((location.hash || '').replace('#', '') === 'flow') {
                paintIfNeeded();
            }
        });
        document.addEventListener('i3:tab-changed', (ev) => {
            try {
                if (ev && ev.detail && ev.detail.tab === 'flow') {
                    paintIfNeeded();
                }
            } catch (_e) { /* noop */ }
        });

        // Also repaint when the nav-link is clicked directly (covers
        // setups where the tab router doesn't dispatch the event).
        const navLink = document.querySelector('.nav-link[data-tab="flow"]');
        if (navLink) {
            navLink.addEventListener('click', () => {
                requestAnimationFrame(paintIfNeeded);
            });
        }

        // First-load fetch of recent traces (in case the user opens
        // the Flow tab before sending any message in the page session
        // — the deque might still have entries from a previous load).
        fetch('/api/flow/recent?n=10')
            .then((r) => r.ok ? r.json() : null)
            .then((data) => {
                if (data && Array.isArray(data.traces) && data.traces.length) {
                    recentTraces = data.traces.slice();
                    if (!lastTrace) lastTrace = data.traces[0];
                    renderHistory();
                    if ((location.hash || '').replace('#', '') === 'flow') {
                        paintIfNeeded();
                    }
                }
            })
            .catch(() => { /* noop — first paint will fall back to empty state */ });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
