/**
 * Routing-decision scatter plot for the Routing tab.
 *
 * Renders the last 50 routing decisions as a 2-D scatter:
 *   X axis: per-prompt complexity score in [0, 1]
 *   Y axis: top retrieval cosine score in [0, 1]
 *   Colour: arm picked (green = edge_slm, orange = cloud_llm)
 *
 * A dashed vertical line marks the LinUCB cloud threshold at
 * complexity 0.65.  All rendering is hand-rolled SVG — no chart
 * library — so the bundle stays small and the page loads fast.
 *
 * Public API:
 *   window.RoutingScatter.append(decision) — push one decision dict
 *   window.RoutingScatter.render()         — re-render from current state
 *   window.RoutingScatter.replaceAll(arr)  — replace the entire window
 */

(function () {
    'use strict';

    const SVG_NS = 'http://www.w3.org/2000/svg';
    const MAX_DECISIONS = 50;
    const VIEW_W = 720;
    const VIEW_H = 320;
    const PAD_L = 48;
    const PAD_R = 24;
    const PAD_T = 16;
    const PAD_B = 32;
    const CLOUD_THRESHOLD = 0.65;

    const state = {
        decisions: [],
    };

    function _decisionToPoint(d) {
        if (!d || typeof d !== 'object') return null;
        const complexity =
            (d.complexity && typeof d.complexity === 'object' && d.complexity.score)
            || (typeof d.complexity === 'number' ? d.complexity : null);
        const retrieval = d.retrieval_top_score;
        // X: complexity score (0..1).  Default 0 if missing.
        let x = Number(complexity);
        if (!Number.isFinite(x)) x = 0;
        x = Math.max(0, Math.min(1, x));
        // Y: retrieval top score.  Default 0 if missing — UI will plot
        // it on the bottom axis.
        let y = Number(retrieval);
        if (!Number.isFinite(y)) y = 0;
        y = Math.max(0, Math.min(1, y));
        const arm = String(d.arm || d.route || '').toLowerCase();
        const isCloud = arm === 'cloud_llm';
        return { x, y, arm: isCloud ? 'cloud' : 'edge', d };
    }

    function _xToPx(x) {
        return PAD_L + x * (VIEW_W - PAD_L - PAD_R);
    }
    function _yToPx(y) {
        // Invert: y=1 at top, y=0 at bottom.
        return PAD_T + (1 - y) * (VIEW_H - PAD_T - PAD_B);
    }

    function _ensureDefs(svg) {
        // No-op for now; placeholder if we ever add gradients.
    }

    function _render() {
        const svg = document.getElementById('routing-scatter-svg');
        if (!svg) return;
        // Clear existing children.
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        _ensureDefs(svg);

        // ---- Axes ----
        const xAxis = document.createElementNS(SVG_NS, 'line');
        xAxis.setAttribute('class', 'axis-line');
        xAxis.setAttribute('x1', _xToPx(0));
        xAxis.setAttribute('y1', _yToPx(0));
        xAxis.setAttribute('x2', _xToPx(1));
        xAxis.setAttribute('y2', _yToPx(0));
        svg.appendChild(xAxis);

        const yAxis = document.createElementNS(SVG_NS, 'line');
        yAxis.setAttribute('class', 'axis-line');
        yAxis.setAttribute('x1', _xToPx(0));
        yAxis.setAttribute('y1', _yToPx(0));
        yAxis.setAttribute('x2', _xToPx(0));
        yAxis.setAttribute('y2', _yToPx(1));
        svg.appendChild(yAxis);

        // ---- Gridlines + tick labels ----
        for (let i = 0; i <= 4; i++) {
            const v = i / 4;
            // X tick
            const xt = document.createElementNS(SVG_NS, 'line');
            xt.setAttribute('class', 'axis-tick');
            xt.setAttribute('x1', _xToPx(v));
            xt.setAttribute('y1', _yToPx(0));
            xt.setAttribute('x2', _xToPx(v));
            xt.setAttribute('y2', _yToPx(1));
            svg.appendChild(xt);

            const xl = document.createElementNS(SVG_NS, 'text');
            xl.setAttribute('class', 'axis-label');
            xl.setAttribute('x', _xToPx(v));
            xl.setAttribute('y', _yToPx(0) + 14);
            xl.setAttribute('text-anchor', 'middle');
            xl.textContent = v.toFixed(2);
            svg.appendChild(xl);

            // Y tick
            const yt = document.createElementNS(SVG_NS, 'line');
            yt.setAttribute('class', 'axis-tick');
            yt.setAttribute('x1', _xToPx(0));
            yt.setAttribute('y1', _yToPx(v));
            yt.setAttribute('x2', _xToPx(1));
            yt.setAttribute('y2', _yToPx(v));
            svg.appendChild(yt);

            const yl = document.createElementNS(SVG_NS, 'text');
            yl.setAttribute('class', 'axis-label');
            yl.setAttribute('x', _xToPx(0) - 8);
            yl.setAttribute('y', _yToPx(v) + 3);
            yl.setAttribute('text-anchor', 'end');
            yl.textContent = v.toFixed(2);
            svg.appendChild(yl);
        }

        // X axis title
        const xt = document.createElementNS(SVG_NS, 'text');
        xt.setAttribute('class', 'axis-label');
        xt.setAttribute('x', (_xToPx(0) + _xToPx(1)) / 2);
        xt.setAttribute('y', _yToPx(0) + 28);
        xt.setAttribute('text-anchor', 'middle');
        xt.textContent = 'prompt complexity →';
        svg.appendChild(xt);

        // Y axis title
        const yt = document.createElementNS(SVG_NS, 'text');
        yt.setAttribute('class', 'axis-label');
        yt.setAttribute(
            'transform',
            `rotate(-90 ${_xToPx(0) - 36} ${(_yToPx(0) + _yToPx(1)) / 2})`
        );
        yt.setAttribute('x', _xToPx(0) - 36);
        yt.setAttribute('y', (_yToPx(0) + _yToPx(1)) / 2);
        yt.setAttribute('text-anchor', 'middle');
        yt.textContent = 'retrieval top score →';
        svg.appendChild(yt);

        // ---- Threshold line at complexity = 0.65 ----
        const thr = document.createElementNS(SVG_NS, 'line');
        thr.setAttribute('class', 'threshold-line');
        thr.setAttribute('x1', _xToPx(CLOUD_THRESHOLD));
        thr.setAttribute('y1', _yToPx(0));
        thr.setAttribute('x2', _xToPx(CLOUD_THRESHOLD));
        thr.setAttribute('y2', _yToPx(1));
        svg.appendChild(thr);

        const thrLabel = document.createElementNS(SVG_NS, 'text');
        thrLabel.setAttribute('class', 'threshold-label');
        thrLabel.setAttribute('x', _xToPx(CLOUD_THRESHOLD) + 6);
        thrLabel.setAttribute('y', _yToPx(1) + 4);
        thrLabel.setAttribute('text-anchor', 'start');
        thrLabel.textContent = `cloud threshold ${CLOUD_THRESHOLD.toFixed(2)}`;
        svg.appendChild(thrLabel);

        // ---- Dots ----
        for (const p of state.decisions) {
            const point = _decisionToPoint(p);
            if (!point) continue;
            const c = document.createElementNS(SVG_NS, 'circle');
            c.setAttribute(
                'class',
                `scatter-dot ${point.arm}`
            );
            c.setAttribute('cx', _xToPx(point.x));
            c.setAttribute('cy', _yToPx(point.y));
            c.setAttribute('r', '4.5');
            const reason = (point.d && point.d.reason) || '';
            const title = document.createElementNS(SVG_NS, 'title');
            title.textContent =
                `${point.arm.toUpperCase()} · complexity ${point.x.toFixed(2)} `
                + `· retrieval ${point.y.toFixed(2)}` + (reason ? `\n${reason}` : '');
            c.appendChild(title);
            svg.appendChild(c);
        }
    }

    function append(decision) {
        if (!decision || typeof decision !== 'object') return;
        state.decisions.push(decision);
        if (state.decisions.length > MAX_DECISIONS) {
            state.decisions.splice(0, state.decisions.length - MAX_DECISIONS);
        }
        _render();
    }

    function replaceAll(arr) {
        if (!Array.isArray(arr)) return;
        state.decisions = arr.slice(-MAX_DECISIONS);
        _render();
    }

    async function _bootstrapFromServer() {
        try {
            const res = await fetch('/api/routing/decision/recent?n=50');
            if (!res.ok) return;
            const payload = await res.json();
            if (payload && Array.isArray(payload.decisions)) {
                replaceAll(payload.decisions);
            }
        } catch (e) {
            // Quiet — empty state is fine.
        }
    }

    function _init() {
        _render();
        _bootstrapFromServer();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _init);
    } else {
        _init();
    }

    window.RoutingScatter = {
        append,
        replaceAll,
        render: _render,
    };
})();
