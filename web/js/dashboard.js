/**
 * Real-time dashboard module.
 *
 * Renders adaptation gauge bars, routing confidence indicators, and
 * engagement metrics. All values animate smoothly via CSS transitions.
 */

class Dashboard {
    /**
     * @param {HTMLElement} container  The #dashboard element.
     */
    constructor(container) {
        this.container = container;
        this.gauges = {};
        this.routingDecision = null;
        this.engagementCards = {};
        this._previousValues = {};
        this.init(container);
    }

    /**
     * Build all dashboard sections.
     *
     * The Apple-style redesign moved each dashboard panel out of a
     * shared ``#dashboard`` wrapper and into top-level ``<section
     * id="adaptation">``, ``<section id="routing">``, and so on.  We
     * therefore look up containers either inside *container* (legacy
     * layout) or at document root by id (current layout) and bind to
     * whichever is found first.
     * @param {HTMLElement|null} container  Optional legacy wrapper.
     */
    init(container) {
        const queryFirst = (selectors) => {
            for (const sel of selectors) {
                const node = (container && container.querySelector(sel))
                    || document.querySelector(sel);
                if (node) return node;
            }
            return null;
        };

        // --- Adaptation gauges ---
        const adaptGauges = queryFirst([
            '.dash-section.adaptation .gauge-container',
            '#tab-adaptation .gauge-container',
            '#adaptation .gauge-container',
            'section#adaptation .gauge-container',
        ]);
        if (adaptGauges) this._createAdaptationGauges(adaptGauges);

        // --- Routing gauges + decision chip ---
        const routeGauges = queryFirst([
            '.dash-section.routing .gauge-container',
            '#tab-routing .gauge-container',
            '#routing .gauge-container',
            'section#routing .gauge-container',
        ]);
        if (routeGauges) this._createRoutingGauges(routeGauges);
        this.routingDecision = queryFirst([
            '.dash-section.routing .routing-decision',
            '#tab-routing .routing-decision',
            '#routing .routing-decision',
            'section#routing .routing-decision',
            '.routing-decision',
        ]);

        // --- Engagement cards ---
        const engRoot = queryFirst([
            '.dash-section.engagement',
            '.engagement-grid',
        ]);
        if (engRoot) {
            this.engagementCards = {
                score:     engRoot.querySelector('[data-metric="score"] .card-value'),
                deviation: engRoot.querySelector('[data-metric="deviation"] .card-value'),
                messages:  engRoot.querySelector('[data-metric="messages"] .card-value'),
                baseline:  engRoot.querySelector('[data-metric="baseline"] .card-value'),
            };
        }
    }

    /** @private */
    _createAdaptationGauges(container) {
        // Iter 51 phase 19: clear the skeleton placeholder rendered
        // in the static HTML so the live gauges replace it cleanly.
        container.innerHTML = '';
        const dims = [
            { key: 'cognitive_load', label: 'Cog. Load', cssClass: 'cognitive' },
            { key: 'formality',      label: 'Formality', cssClass: 'formality' },
            { key: 'verbosity',      label: 'Verbosity', cssClass: 'verbosity' },
            { key: 'emotionality',   label: 'Emotional', cssClass: 'emotionality' },
            { key: 'directness',     label: 'Directness', cssClass: 'directness' },
            { key: 'emotional_tone', label: 'Tone',      cssClass: 'tone' },
            { key: 'accessibility',  label: 'Access.',   cssClass: 'accessibility' },
        ];

        for (const dim of dims) {
            const row = this._buildGaugeRow(dim.label, 0, dim.cssClass);
            container.appendChild(row.element);
            this.gauges[dim.key] = row;
        }
    }

    /** @private */
    _createRoutingGauges(container) {
        const edgeRow = this._buildGaugeRow('Edge SLM', 0.5, 'edge');
        const cloudRow = this._buildGaugeRow('Cloud LLM', 0.5, 'cloud');

        container.appendChild(edgeRow.element);
        container.appendChild(cloudRow.element);

        this.gauges['route_edge'] = edgeRow;
        this.gauges['route_cloud'] = cloudRow;
    }

    /**
     * Build a single gauge bar row.
     * @param {string} label
     * @param {number} value  Initial value in [0, 1].
     * @param {string} cssClass
     * @returns {{ element: HTMLElement, fill: HTMLElement, valueEl: HTMLElement }}
     * @private
     */
    _buildGaugeRow(label, value, cssClass) {
        const row = document.createElement('div');
        row.className = 'gauge-row';

        const labelEl = document.createElement('div');
        labelEl.className = 'gauge-label';
        labelEl.textContent = label;

        const track = document.createElement('div');
        track.className = 'gauge-track';

        const fill = document.createElement('div');
        fill.className = `gauge-fill ${cssClass}`;
        fill.style.width = `${(value * 100).toFixed(1)}%`;

        track.appendChild(fill);

        const valueEl = document.createElement('div');
        valueEl.className = 'gauge-value';
        valueEl.textContent = value.toFixed(2);

        row.appendChild(labelEl);
        row.appendChild(track);
        row.appendChild(valueEl);

        return { element: row, fill, valueEl };
    }

    /**
     * Update all dashboard values from a state_update payload.
     *
     * @param {object} data  Server state_update payload containing:
     *   - adaptation: { cognitive_load, formality, verbosity, emotionality,
     *                    directness, emotional_tone, accessibility }
     *   - routing_confidence: { local_slm, cloud_llm }
     *   - engagement_score: number
     *   - deviation_from_baseline: number
     *   - messages_in_session: number
     *   - baseline_established: boolean
     */
    update(data) {
        // --- Adaptation gauges ---
        if (data.adaptation) {
            const adapt = data.adaptation;
            const keys = [
                'cognitive_load', 'formality', 'verbosity', 'emotionality',
                'directness', 'emotional_tone', 'accessibility',
            ];
            for (const key of keys) {
                if (adapt[key] !== undefined && this.gauges[key]) {
                    this._updateGauge(key, adapt[key]);
                }
            }
        }

        // --- Routing gauges ---
        if (data.routing_confidence) {
            const rc = data.routing_confidence;
            const edgeVal = rc.local_slm !== undefined ? rc.local_slm : (rc.arm_0 || 0.5);
            const cloudVal = rc.cloud_llm !== undefined ? rc.cloud_llm : (rc.arm_1 || 0.5);
            this._updateGauge('route_edge', edgeVal);
            this._updateGauge('route_cloud', cloudVal);

            // Routing decision text
            if (this.routingDecision) {
                // SEC: data.route_chosen is server-controlled. Coerce to string and
                // build DOM with textContent to prevent XSS via injected HTML.
                const chosenRaw = data.route_chosen ||
                    (edgeVal >= cloudVal ? 'local_slm' : 'cloud_llm');
                const chosen = String(chosenRaw);
                const label = chosen.includes('local') || chosen.includes('slm')
                    ? 'Edge SLM'
                    : 'Cloud LLM';
                const confidencePct = (Math.max(edgeVal, cloudVal) * 100).toFixed(0);

                // SEC: Replace innerHTML with safe DOM construction (textContent only)
                this.routingDecision.textContent = '';
                this.routingDecision.appendChild(
                    document.createTextNode('Decision: ')
                );
                const labelSpan = document.createElement('span');
                labelSpan.className = 'route-label';
                labelSpan.textContent = label;
                this.routingDecision.appendChild(labelSpan);
                this.routingDecision.appendChild(
                    document.createTextNode(` (${confidencePct}% confidence)`)
                );
            }
        }

        // --- Engagement cards ---
        // SEC: Coerce all server-supplied numerics to Number() and reject NaN
        // before calling .toFixed(), which would otherwise throw on a string
        // and break the update loop.
        if (data.engagement_score !== undefined && this.engagementCards.score) {
            const score = Number(data.engagement_score);
            if (Number.isFinite(score)) {
                this.engagementCards.score.textContent = score.toFixed(2);
            }
        }

        if (data.deviation_from_baseline !== undefined && this.engagementCards.deviation) {
            const dev = Number(data.deviation_from_baseline);
            if (Number.isFinite(dev)) {
                const sign = dev >= 0 ? '+' : '';
                this.engagementCards.deviation.textContent = `${sign}${dev.toFixed(3)}`;
            }
        }

        if (data.messages_in_session !== undefined && this.engagementCards.messages) {
            const count = Number(data.messages_in_session);
            if (Number.isFinite(count)) {
                // SEC: textContent accepts a number safely; coerce explicitly
                // to avoid passing a server-controlled object/string into the DOM.
                this.engagementCards.messages.textContent = String(Math.round(count));
            }
        }

        if (data.baseline_established !== undefined && this.engagementCards.baseline) {
            // SEC: Replace innerHTML with DOM construction to satisfy strict CSP
            // (script-src 'self') and avoid the innerHTML sink entirely.
            const baselineEl = this.engagementCards.baseline;
            baselineEl.textContent = '';
            const span = document.createElement('span');
            if (data.baseline_established) {
                span.className = 'baseline-yes';
                span.textContent = 'Established';
            } else {
                span.className = 'baseline-no';
                span.textContent = 'Warming up...';
            }
            baselineEl.appendChild(span);
        }
    }

    /**
     * Smoothly update a gauge bar + value label.
     * @param {string} key    Gauge identifier.
     * @param {number} value  New value in [0, 1].
     * @private
     */
    _updateGauge(key, value) {
        const gauge = this.gauges[key];
        if (!gauge) return;

        // SEC: Coerce to Number and reject NaN/Infinity before clamp.
        // This prevents NaN% widths and broken textContent on bad payloads.
        const num = Number(value);
        if (!Number.isFinite(num)) return;
        const clamped = Math.max(0, Math.min(1, num));

        gauge.fill.style.width = `${(clamped * 100).toFixed(1)}%`;
        gauge.valueEl.textContent = clamped.toFixed(2);

        // Highlight significant changes
        const prev = this._previousValues[key] || 0;
        const delta = Math.abs(clamped - prev);
        if (delta > 0.1) {
            gauge.valueEl.classList.add('highlight');
            setTimeout(() => gauge.valueEl.classList.remove('highlight'), 800);
        }
        this._previousValues[key] = clamped;
    }
}
