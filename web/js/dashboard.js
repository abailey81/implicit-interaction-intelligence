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
     * @param {HTMLElement} container
     */
    init(container) {
        // --- Adaptation section ---
        const adaptSection = container.querySelector('.dash-section.adaptation');
        if (adaptSection) {
            const gaugeContainer = adaptSection.querySelector('.gauge-container');
            if (gaugeContainer) {
                this._createAdaptationGauges(gaugeContainer);
            }
        }

        // --- Routing section ---
        const routeSection = container.querySelector('.dash-section.routing');
        if (routeSection) {
            const gaugeContainer = routeSection.querySelector('.gauge-container');
            if (gaugeContainer) {
                this._createRoutingGauges(gaugeContainer);
            }
            this.routingDecision = routeSection.querySelector('.routing-decision');
        }

        // --- Engagement section ---
        const engSection = container.querySelector('.dash-section.engagement');
        if (engSection) {
            this.engagementCards = {
                score: engSection.querySelector('[data-metric="score"] .card-value'),
                deviation: engSection.querySelector('[data-metric="deviation"] .card-value'),
                messages: engSection.querySelector('[data-metric="messages"] .card-value'),
                baseline: engSection.querySelector('[data-metric="baseline"] .card-value'),
            };
        }
    }

    /** @private */
    _createAdaptationGauges(container) {
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
                const chosen = data.route_chosen || (edgeVal >= cloudVal ? 'local_slm' : 'cloud_llm');
                const label = chosen.includes('local') || chosen.includes('slm') ? 'Edge SLM' : 'Cloud LLM';
                this.routingDecision.innerHTML =
                    `Decision: <span class="route-label">${label}</span>` +
                    ` (${(Math.max(edgeVal, cloudVal) * 100).toFixed(0)}% confidence)`;
            }
        }

        // --- Engagement cards ---
        if (data.engagement_score !== undefined && this.engagementCards.score) {
            this.engagementCards.score.textContent = data.engagement_score.toFixed(2);
        }

        if (data.deviation_from_baseline !== undefined && this.engagementCards.deviation) {
            const dev = data.deviation_from_baseline;
            const sign = dev >= 0 ? '+' : '';
            this.engagementCards.deviation.textContent = `${sign}${dev.toFixed(3)}`;
        }

        if (data.messages_in_session !== undefined && this.engagementCards.messages) {
            this.engagementCards.messages.textContent = data.messages_in_session;
        }

        if (data.baseline_established !== undefined && this.engagementCards.baseline) {
            if (data.baseline_established) {
                this.engagementCards.baseline.innerHTML = '<span class="baseline-yes">Established</span>';
            } else {
                this.engagementCards.baseline.innerHTML = '<span class="baseline-no">Warming up...</span>';
            }
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

        const clamped = Math.max(0, Math.min(1, value));

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
