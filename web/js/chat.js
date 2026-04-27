/**
 * Chat interface module.
 *
 * Renders user and AI messages with metadata badges (route, latency),
 * slide-in animations, and auto-scroll. Integrates with the keystroke
 * monitor for implicit signal capture.
 */

class ChatInterface {
    /**
     * @param {HTMLElement}      container        The .chat-messages container.
     * @param {WebSocketClient}  wsClient         WebSocket client for sending messages.
     * @param {KeystrokeMonitor} keystrokeMonitor Keystroke capture instance.
     */
    constructor(container, wsClient, keystrokeMonitor) {
        this.container = container;
        this.ws = wsClient;
        this.ksMonitor = keystrokeMonitor;
        this.messageCount = 0;
        this.typingIndicator = null;
        // Streaming state — populated by appendTokenDelta and drained by
        // finaliseStreaming. Only one streaming bubble can exist at a time.
        this._streamState = null;
        // Voice-prosody monitor (flagship feature #2).  Set via
        // setVoiceProsody() once the toggle button is mounted.  When
        // the user has enabled the mic and frames have been collected,
        // the eight prosodic scalars are attached to every outgoing
        // message frame.  When null / disabled / no frames yet, the
        // ``prosody_features`` field is OMITTED from the WS frame
        // (rather than sent as zeros), so the server's null branch
        // fires correctly.
        this.voiceProsody = null;
        // Vision-gaze flagship.  Set via setGazeCapture() once the
        // camera-toggle button is mounted.  When non-null + enabled,
        // the predicted gaze label + 5 numeric scalars are attached to
        // every outgoing message frame.  Raw video NEVER crosses the
        // wire — see web/js/gaze_capture.js for the privacy contract.
        this.gazeCapture = null;
    }

    /**
     * Inject the ``VoiceProsodyMonitor`` instance after construction.
     * Called by app.js right after mounting the mic-toggle button.
     */
    setVoiceProsody(monitor) {
        this.voiceProsody = monitor || null;
    }

    /**
     * Inject the ``GazeCaptureMonitor`` instance after construction.
     * Called by app.js right after mounting the camera-toggle button.
     */
    setGazeCapture(monitor) {
        this.gazeCapture = monitor || null;
    }

    /**
     * Append an incremental token delta to the in-progress streaming
     * bubble, creating the bubble on first call.
     * @param {string} delta  Text chunk to append.
     */
    appendTokenDelta(delta) {
        if (typeof delta !== 'string' || !delta) return;

        if (!this._streamState) {
            // First token → build a bubble now and cache the refs.
            this._hideTyping();

            const msgEl = document.createElement('div');
            msgEl.className = 'message ai streaming';

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = 'I³';

            const body = document.createElement('div');
            body.className = 'message-body';

            const textEl = document.createElement('div');
            textEl.className = 'message-text streaming-text';
            textEl.textContent = '';

            // Blinking caret for liveness.
            const caretEl = document.createElement('span');
            caretEl.className = 'streaming-caret';
            caretEl.textContent = '▌';  // ▌

            body.appendChild(textEl);
            body.appendChild(caretEl);
            msgEl.appendChild(avatar);
            msgEl.appendChild(body);
            this.container.appendChild(msgEl);

            this._streamState = {
                msgEl,
                body,
                textEl,
                caretEl,
                buffer: '',
            };
        }

        this._streamState.buffer += delta;
        // SEC: textContent (not innerHTML) — server tokens never reach
        // the DOM as HTML.
        this._streamState.textEl.textContent = this._streamState.buffer;
        this._scrollToBottom();
    }

    /**
     * Finalise the currently-streaming bubble: replace its text with the
     * server-confirmed full string, attach metadata chips, and clear the
     * streaming state so the next response starts fresh.
     * @param {string} fullText
     * @param {object} metadata
     */
    finaliseStreaming(fullText, metadata = {}) {
        if (!this._streamState) {
            // No bubble was created (no tokens arrived) — fall back to
            // the single-frame addMessage path for robustness.
            this.addMessage(fullText, 'ai', metadata);
            return;
        }

        const { msgEl, body, textEl, caretEl, buffer } = this._streamState;

        // SEC: Coerce full text; prefer the server's confirmed final
        // string, fall back to the accumulated token buffer.
        const safeText = (fullText === null || fullText === undefined)
            ? (buffer || '')
            : String(fullText);
        textEl.textContent = safeText;
        textEl.classList.remove('streaming-text');
        if (caretEl && caretEl.parentNode) {
            caretEl.parentNode.removeChild(caretEl);
        }
        msgEl.classList.remove('streaming');

        // Attach the usual metadata chips so the streamed message looks
        // identical to a non-streamed one once complete.
        this._attachMeta(body, metadata);

        // Iter 51 phase 11: streamed responses (token frames +
        // response_done) used to skip the side-chip row entirely
        // because ``finaliseStreaming`` only called ``_attachMeta``.
        // The three per-arm indicator chips + "Used: X" winner badge
        // now render here too, mirroring the addMessage path.
        try {
            this._appendSideChips(msgEl, metadata);
        } catch (_e) { /* never let chip render kill the bubble */ }

        // Per-bubble TTS button — apple21 cleanup moves the speaker
        // out of the inline panel and into a small icon on each AI
        // bubble (ChatGPT-style).  Only added once the response is
        // finalised so we don't read the streaming-caret as text.
        this._appendBubbleTts(msgEl, body);

        this._streamState = null;
        this.messageCount++;
        this._scrollToBottom();

        // Tell the floating "How am I doing?" toast that another AI
        // message landed.  The toast itself decides whether to show.
        try {
            window.dispatchEvent(new CustomEvent('i3:ai-message-landed', {
                detail: { messageCount: this.messageCount },
            }));
        } catch (_e) { /* no-op */ }
    }

    /**
     * @private
     * Build and append the ``.message-meta`` badge row to a bubble body.
     * Extracted so both ``addMessage`` and ``finaliseStreaming`` can
     * share the logic without drift.
     */
    _attachMeta(body, metadata) {
        if (!metadata) return;
        const hasAffectShift = metadata.affect_shift
            && metadata.affect_shift.detected;
        const hasCritique = metadata.critique
            && typeof metadata.critique === 'object';
        const hasCoref = metadata.coreference_resolution
            && typeof metadata.coreference_resolution === 'object'
            && metadata.coreference_resolution.used_entity;
        const hasSafety = metadata.safety
            && typeof metadata.safety === 'object'
            && (metadata.safety.verdict === 'refuse'
                || metadata.safety.verdict === 'review');
        if (!metadata.route && metadata.latency === undefined
            && !metadata.response_path
            && !(Array.isArray(metadata.adaptation_changes)
                 && metadata.adaptation_changes.length)
            && !hasAffectShift
            && !hasCritique
            && !hasCoref
            && !hasSafety) {
            return;
        }
        const meta = document.createElement('div');
        meta.className = 'message-meta';

        if (metadata.route) {
            const routeStr = String(metadata.route).toLowerCase();
            const routeBadge = document.createElement('span');
            const isEdge = routeStr.includes('local')
                || routeStr.includes('edge')
                || routeStr.includes('slm');
            routeBadge.classList.add('meta-badge');
            routeBadge.classList.add(isEdge ? 'edge' : 'cloud');
            // Re-skin as the new orange/green "route" chip with a
            // tooltip pulled from the routing_decision dict so the
            // user can see exactly why this turn was routed where.
            routeBadge.classList.add(
                isEdge ? 'meta-badge-edge-route' : 'meta-badge-cloud-route'
            );
            routeBadge.textContent = isEdge
                ? 'route · edge SLM'
                : 'route · cloud LLM';
            const rd = metadata.routing_decision || null;
            if (rd && typeof rd === 'object' && rd.reason) {
                routeBadge.title = String(rd.reason);
            }
            meta.appendChild(routeBadge);
        }

        // PII-redacted chip — only on cloud-routed turns where
        // sanitiser actually fired.
        const rd = metadata.routing_decision || null;
        if (rd && typeof rd === 'object'
            && Number(rd.pii_redactions) > 0) {
            const piiBadge = document.createElement('span');
            piiBadge.className = 'meta-badge meta-badge-pii';
            piiBadge.textContent = `pii redacted · ${Number(rd.pii_redactions)}`;
            piiBadge.title =
                `${Number(rd.pii_redactions)} PII tokens redacted before the cloud call`
                + (Number(rd.bytes_redacted) > 0
                    ? ` (${Number(rd.bytes_redacted)} bytes saved from the wire)`
                    : '');
            meta.appendChild(piiBadge);
        }

        if (metadata.latency !== undefined) {
            const latNum = Number(metadata.latency);
            if (Number.isFinite(latNum)) {
                const latBadge = document.createElement('span');
                latBadge.className = 'meta-badge latency';
                latBadge.textContent = `${Math.round(latNum)} ms`;
                meta.appendChild(latBadge);
            }
        }

        if (metadata.response_path) {
            const pathStr = String(metadata.response_path).toLowerCase();
            const pathBadge = document.createElement('span');
            pathBadge.classList.add('meta-badge');
            let pathLabel = '';
            if (pathStr === 'retrieval') pathLabel = 'retrieval';
            else if (pathStr === 'retrieval_borderline') pathLabel = 'retrieval (low conf.)';
            else if (pathStr === 'retrieval_consistent') {
                pathLabel = 'retrieval (consistency: 2/3 agreed)';
                pathBadge.classList.add('meta-badge-tool');
            }
            else if (pathStr === 'explain_decomposed') {
                pathLabel = 'explain · decomposed';
                pathBadge.classList.add('meta-badge-tool');
            }
            else if (pathStr === 'slm') pathLabel = 'SLM generation';
            else if (pathStr === 'ood') pathLabel = 'out-of-domain';
            else if (pathStr === 'tool:math') {
                pathLabel = 'tool · math solver';
                pathBadge.classList.add('meta-badge-tool');
            } else if (pathStr === 'tool:refuse') {
                pathLabel = 'tool · hostility guard';
                pathBadge.classList.add('meta-badge-tool');
            } else if (pathStr === 'tool:safety') {
                pathLabel = 'tool · constitutional safety';
                pathBadge.classList.add('meta-badge-tool');
            } else if (pathStr === 'tool:entity') {
                pathLabel = 'tool · entity';
                pathBadge.classList.add('meta-badge-tool');
            } else if (pathStr === 'tool:graph_compose') {
                pathLabel = 'tool · knowledge graph';
                pathBadge.classList.add('meta-badge-tool');
            } else if (pathStr === 'tool:clarify') {
                pathLabel = 'tool · clarify';
                pathBadge.classList.add('meta-badge-tool');
            } else if (pathStr.startsWith('tool:')) {
                pathLabel = `tool · ${pathStr.slice(5)}`;
                pathBadge.classList.add('meta-badge-tool');
            }
            if (pathLabel) {
                pathBadge.textContent = pathLabel;
                meta.appendChild(pathBadge);
            }
            const scoreNum = Number(metadata.retrieval_score);
            if (Number.isFinite(scoreNum) && scoreNum > 0) {
                const confBadge = document.createElement('span');
                confBadge.className = 'meta-badge';
                confBadge.textContent = `conf ${scoreNum.toFixed(2)}`;
                meta.appendChild(confBadge);
            }
        }

        if (Array.isArray(metadata.adaptation_changes)
            && metadata.adaptation_changes.length) {
            for (const ch of metadata.adaptation_changes) {
                if (!ch || typeof ch !== 'object') continue;
                const adaptBadge = document.createElement('span');
                adaptBadge.className = 'meta-badge meta-badge-adapt';
                const axis = String(ch.axis || '').replace(/_/g, ' ');
                const change = String(ch.change || '');
                const val = ch.value !== undefined ? String(ch.value) : '';
                adaptBadge.textContent = val
                    ? `${axis} ${val} · ${change}`
                    : `${axis} · ${change}`;
                adaptBadge.title = `Adaptation rewrite triggered by ${axis} = ${val}`;
                meta.appendChild(adaptBadge);
            }
        }

        // Co-reference resolution chip — surfaces the multi-turn
        // pronoun resolution that rewrote the user's follow-up before
        // retrieval (e.g. "where are they located?" → resolved 'they'
        // to 'huawei').  Reuses the adaptation purple so it visually
        // groups with the other implicit-rewrite chips.  Only shown
        // when an entity was actually used — no-op turns stay quiet.
        this._appendCorefBadge(meta, metadata.coreference_resolution);

        // Affect-shift chip — surfaces the proactive check-in (the
        // pitch piece for the Huawei HMI Lab demo).  Only shown when
        // a shift was detected; the chip text reads
        // ``affect-shift · rising_load · 1.6σ`` so a reviewer can see
        // both the direction and the magnitude at a glance.
        const ashift = metadata.affect_shift;
        if (ashift && ashift.detected) {
            const dir = String(ashift.direction || 'neutral');
            const magNum = Number(ashift.magnitude);
            const magStr = Number.isFinite(magNum)
                ? `${magNum.toFixed(1)}σ`
                : '—';
            const affectBadge = document.createElement('span');
            affectBadge.className = 'meta-badge meta-badge-affect';
            affectBadge.textContent = `affect-shift · ${dir} · ${magStr}`;
            const suggestion = String(ashift.suggestion || '').trim();
            affectBadge.title = suggestion
                ? `Mid-conversation affect shift; appended check-in: "${suggestion}"`
                : `Mid-conversation affect shift detected (${dir}, ${magStr}).`;
            meta.appendChild(affectBadge);
        }

        // Self-critique chip — Phase 7 HMI piece.  Only present on
        // SLM-path turns (the engine nulls the critique field for
        // retrieval / tool / OOD).  Two variants:
        //   * accepted-first-try → "self-critique · accepted 0.79"
        //   * regenerated        → "self-critique · regenerated 0.41 → 0.79"
        // The expandable <details> trace below the message bubble
        // narrates each attempt's sub-scores so reviewers can audit
        // the regenerate decision.
        const critique = metadata.critique;
        if (critique && typeof critique === 'object') {
            this._appendCritiqueBadge(meta, critique);
        }

        // Constitutional safety chip — char-CNN classifier.  Renders
        // ``safety · refuse`` (red) or ``safety · review`` (orange) when
        // the classifier flagged the inbound message.  Tooltip surfaces
        // the constitutional principle that was violated.  Bai et al.
        // (2022) Constitutional AI is the citation; the implementation
        // is a from-scratch char-CNN — see i3/safety/classifier.py.
        const safety = metadata.safety;
        if (safety && typeof safety === 'object'
            && (safety.verdict === 'refuse' || safety.verdict === 'review')) {
            const sb = document.createElement('span');
            sb.className = 'meta-badge meta-badge-safety';
            sb.dataset.verdict = safety.verdict;
            const conf = Number(safety.confidence);
            const confStr = Number.isFinite(conf) ? conf.toFixed(2) : '—';
            sb.textContent = `safety · ${safety.verdict} ${confStr}`;
            const reasons = Array.isArray(safety.reasons) ? safety.reasons : [];
            const principle = String(safety.constitutional_principle || '');
            sb.title =
                `Char-CNN safety classifier (Bai et al. 2022, Constitutional AI). ` +
                `Verdict: ${safety.verdict.toUpperCase()} ` +
                `(reasons: ${reasons.join(', ') || '—'}). ` +
                principle;
            meta.appendChild(sb);
        }

        // Voice-prosody multimodal-fusion chip — flagship feature #2.
        // Only renders when the user enabled the mic on this turn AND
        // the server confirmed the prosody features were validated +
        // fused into the 96-d multimodal embedding.
        this._appendMultimodalBadge(meta, metadata.multimodal);

        // Vision-gaze chip — fine-tuned MobileNetV3 head.  Only renders
        // when the camera was on AND the server returned a gaze label
        // for this turn.
        this._appendGazeBadge(meta, metadata.gaze);

        body.appendChild(meta);

        if (critique && typeof critique === 'object') {
            this._appendCritiqueTrace(body, critique);
        }

        // Phase B.3 (2026-04-25): explain decomposition plan as a
        // collapsible <details> "Reasoning chain" element under the
        // response.  Only rendered when the engine produced a plan
        // (i.e. on "explain X" / "tell me about X" turns).
        const explainPlan = metadata.explain_plan;
        if (explainPlan && typeof explainPlan === 'object'
            && Array.isArray(explainPlan.sub_questions)
            && explainPlan.sub_questions.length > 0) {
            this._appendExplainPlan(body, explainPlan);
        }

        // Iteration 12 (2026-04-26): per-bubble Decision Trace
        // expander — consolidates the FULL adaptation snapshot,
        // biometric verdict, and routing decision behind a single
        // <details> click so power-users / recruiters can audit
        // exactly which signals shaped this answer without leaving
        // the chat tab.
        this._appendDecisionTrace(body, metadata);
    }

    /**
     * @private
     * Render a compact "Decision trace" <details> under each AI bubble
     * showing: the 8-axis adaptation vector, biometric similarity +
     * verdict, routing decision (route/path/score), and timing.  All
     * read directly from the per-turn metadata bag — no extra fetch.
     */
    _appendDecisionTrace(body, metadata) {
        if (!metadata || typeof metadata !== 'object') return;
        const adapt = metadata.adaptation || metadata.adaptation_snapshot;
        const bio = metadata.biometric;
        const route = metadata.route;
        const path = metadata.response_path;
        const score = metadata.retrieval_score;
        // Only render when at least one of the trace inputs is present
        // — avoids an empty expander on errored / OOD turns.
        const hasContent = !!(adapt || bio || route || path);
        if (!hasContent) return;

        const det = document.createElement('details');
        det.className = 'decision-trace';
        det.style.marginTop = '6px';
        det.style.fontSize = '0.82em';
        det.style.opacity = '0.85';

        const sum = document.createElement('summary');
        sum.style.cursor = 'pointer';
        sum.style.userSelect = 'none';
        const summaryBits = [];
        if (path) summaryBits.push(path);
        if (route) summaryBits.push(route);
        if (typeof score === 'number' && score > 0) {
            summaryBits.push(`conf ${score.toFixed(2)}`);
        }
        sum.textContent = `Decision trace · ${summaryBits.join(' · ')}`;
        det.appendChild(sum);

        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = '110px 1fr 50px';
        grid.style.gap = '4px 10px';
        grid.style.marginTop = '6px';
        grid.style.padding = '8px 10px';
        grid.style.background = 'rgba(255, 255, 255, 0.03)';
        grid.style.border = '1px solid var(--hairline)';
        grid.style.borderRadius = '8px';

        // Adaptation vector — 8 axes as small horizontal bars.
        const renderRow = (label, value, kind) => {
            const lab = document.createElement('span');
            lab.textContent = label;
            lab.style.color = 'var(--text-subtle)';
            grid.appendChild(lab);
            const barWrap = document.createElement('div');
            barWrap.style.position = 'relative';
            barWrap.style.height = '8px';
            barWrap.style.background = 'rgba(255,255,255,0.05)';
            barWrap.style.borderRadius = '4px';
            barWrap.style.alignSelf = 'center';
            const fill = document.createElement('div');
            const pct = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
            fill.style.position = 'absolute';
            fill.style.left = '0';
            fill.style.top = '0';
            fill.style.height = '100%';
            fill.style.width = `${pct.toFixed(0)}%`;
            fill.style.borderRadius = '4px';
            fill.style.background = kind === 'bio'
                ? 'rgba(94, 232, 162, 0.65)'
                : 'rgba(94, 165, 232, 0.65)';
            barWrap.appendChild(fill);
            grid.appendChild(barWrap);
            const num = document.createElement('span');
            num.textContent = (Number(value) || 0).toFixed(2);
            num.style.color = 'var(--text-subtle)';
            num.style.textAlign = 'right';
            num.style.fontVariantNumeric = 'tabular-nums';
            grid.appendChild(num);
        };

        if (adapt && typeof adapt === 'object') {
            const sm = adapt.style_mirror || {};
            renderRow('cognitive load', adapt.cognitive_load);
            renderRow('formality', sm.formality);
            renderRow('verbosity', sm.verbosity);
            renderRow('emotionality', sm.emotionality);
            renderRow('directness', sm.directness);
            renderRow('emotional tone', adapt.emotional_tone);
            renderRow('accessibility', adapt.accessibility);
        }

        if (bio && typeof bio === 'object') {
            // Visual separator
            const hr = document.createElement('div');
            hr.style.gridColumn = '1 / -1';
            hr.style.borderTop = '1px solid var(--hairline)';
            hr.style.margin = '4px 0';
            grid.appendChild(hr);
            renderRow('biometric', bio.similarity, 'bio');
            // Verdict label + state
            const lab = document.createElement('span');
            lab.textContent = 'state';
            lab.style.color = 'var(--text-subtle)';
            grid.appendChild(lab);
            const stateSpan = document.createElement('span');
            stateSpan.textContent = String(bio.state || '—');
            stateSpan.style.color = 'var(--text)';
            stateSpan.style.gridColumn = '2 / -1';
            grid.appendChild(stateSpan);
        }

        det.appendChild(grid);
        body.appendChild(det);
    }

    /**
     * @private
     * Append the expandable <details> "Reasoning chain" element under
     * a chat reply.  Shows the multi-step decomposition the engine
     * built when answering an "explain X" query so the reviewer can
     * see the model thinking in steps.
     */
    _appendExplainPlan(body, plan) {
        try {
            const det = document.createElement('details');
            det.className = 'reasoning-chain';
            det.style.marginTop = '6px';
            det.style.fontSize = '0.85em';
            det.style.opacity = '0.85';
            const sum = document.createElement('summary');
            sum.style.cursor = 'pointer';
            const subQs = Array.isArray(plan.sub_questions)
                ? plan.sub_questions : [];
            const topic = String(plan.topic || '').trim();
            sum.textContent = `Reasoning chain · ${subQs.length} steps`
                + (topic ? ` · ${topic}` : '');
            det.appendChild(sum);
            const ol = document.createElement('ol');
            ol.style.marginTop = '4px';
            ol.style.paddingLeft = '18px';
            const subAs = Array.isArray(plan.sub_answers)
                ? plan.sub_answers : [];
            for (let i = 0; i < subQs.length; i++) {
                const li = document.createElement('li');
                li.style.marginBottom = '6px';
                const q = document.createElement('strong');
                q.textContent = String(subQs[i] || '');
                li.appendChild(q);
                const sa = subAs[i] || {};
                const src = String(sa.source || 'unanswered');
                const conf = Number(sa.confidence) || 0;
                const meta = document.createElement('span');
                meta.style.opacity = '0.7';
                meta.style.fontSize = '0.85em';
                meta.textContent = ` · ${src} · ${conf.toFixed(2)}`;
                li.appendChild(meta);
                if (sa.text) {
                    const p = document.createElement('div');
                    p.textContent = String(sa.text);
                    p.style.marginTop = '2px';
                    li.appendChild(p);
                }
                ol.appendChild(li);
            }
            det.appendChild(ol);
            body.appendChild(det);
        } catch (e) {
            // Decorative — never break the response render.
        }
    }

    /**
     * @private
     * Build the voice-prosody multimodal chip
     * (``voice prosody · active``) and append it to the meta row.
     *
     * Only renders when the server confirmed
     * ``multimodal.prosody_active === true`` for this turn.  The chip
     * uses the existing teal palette via .meta-badge-multimodal so it
     * groups with the other modality / signal chips.  Hover tooltip
     * states the privacy contract: audio never left the device.
     */
    _appendMultimodalBadge(meta, multimodal) {
        if (!multimodal || typeof multimodal !== 'object') return;
        if (!multimodal.prosody_active) return;
        const captured = Number(multimodal.captured_seconds);
        const fusedDim = Number(multimodal.fused_dim) || 96;
        const samples = Number(multimodal.samples_count) || 0;
        const badge = document.createElement('span');
        badge.className = 'meta-badge meta-badge-multimodal';
        badge.textContent = 'voice prosody · active';
        const dur = Number.isFinite(captured) ? `${captured.toFixed(1)} s` : '—';
        badge.title =
            `Voice prosody fused into a ${fusedDim}-d multimodal embedding ` +
            `(${samples} frames, ${dur} of audio). ` +
            `Audio never left this device — only 8 numeric features were sent.`;
        meta.appendChild(badge);
    }

    /**
     * @private
     * Build the gaze chip (e.g. `gaze · at-screen 0.91`) and append it
     * to the meta row.  Only renders when the camera was on this turn.
     * Uses the violet palette via .meta-badge-gaze so it groups with
     * the other multimodal flagship chips while staying visually
     * distinct from the teal prosody chip.  The hover tooltip surfaces
     * the privacy contract.
     */
    _appendGazeBadge(meta, gaze) {
        if (!gaze || typeof gaze !== 'object') return;
        const label = String(gaze.label || '');
        if (!label) return;
        const conf = Number(gaze.confidence) || 0;
        const presence = !!gaze.presence;
        const badge = document.createElement('span');
        badge.className = 'meta-badge meta-badge-gaze';
        const pretty = label.replace('_', '-');
        badge.textContent = `gaze · ${pretty} ${conf.toFixed(2)}`;
        const note = String(gaze.gaze_aware_note || '').trim();
        let tooltip =
            'Fine-tuned MobileNetV3-small head (75 k params over a ' +
            '5.4 M frozen backbone). Only a 64×48 grayscale fingerprint ' +
            'reached the server — the raw frame was discarded on-device.';
        if (!presence && note) tooltip += ' ' + note;
        badge.title = tooltip;
        meta.appendChild(badge);
    }

    /**
     * @private
     * Build the co-reference resolution chip (`coref · they → huawei`)
     * and append it to the meta row.  Reuses the adaptation purple
     * (.meta-badge-adapt) so it groups visually with other
     * implicit-rewrite chips.  Only renders when an entity was
     * actually substituted — no-op turns produce nothing.
     *
     * Defensive about unexpected shapes; ALL strings go through
     * textContent / title so model-controlled prose can never become
     * innerHTML.
     */
    _appendCorefBadge(meta, coref) {
        if (!coref || typeof coref !== 'object') return;
        const ent = coref.used_entity;
        const pron = coref.used_pronoun;
        if (!ent || typeof ent !== 'object') return;
        if (!pron || !ent.canonical) return;

        const badge = document.createElement('span');
        badge.className = 'meta-badge meta-badge-adapt';
        badge.classList.add('meta-badge-coref');
        const pronStr = String(pron);
        const canonStr = String(ent.canonical);
        badge.textContent = `coref · ${pronStr} → ${canonStr}`;
        const reasoning = String(coref.reasoning || '').trim();
        const original = String(coref.original_query || '').trim();
        const resolved = String(coref.resolved_query || '').trim();
        const confNum = Number(coref.confidence);
        const confStr = Number.isFinite(confNum)
            ? ` (confidence ${confNum.toFixed(2)})` : '';
        const titleParts = [];
        if (reasoning) titleParts.push(reasoning + confStr);
        if (original && resolved && original !== resolved) {
            titleParts.push(`Rewrote ${JSON.stringify(original)} → ${JSON.stringify(resolved)} before retrieval.`);
        }
        if (!titleParts.length) {
            titleParts.push(`Resolved '${pronStr}' to '${canonStr}'${confStr}.`);
        }
        badge.title = titleParts.join(' ');
        meta.appendChild(badge);
    }

    /**
     * @private
     * Build the .meta-badge-critique chip and append it to the meta row.
     * Defensive about missing fields — the chip only renders when at
     * least one attempt is present; otherwise the critic didn't run
     * and there's nothing to surface.
     */
    _appendCritiqueBadge(meta, critique) {
        const attempts = Array.isArray(critique.attempts)
            ? critique.attempts : [];
        if (!attempts.length) return;
        const finalScoreNum = Number(critique.final_score);
        const finalScore = Number.isFinite(finalScoreNum) ? finalScoreNum : 0;
        const regenerated = Boolean(critique.regenerated);
        const accepted = Boolean(critique.accepted);
        const rejected = Boolean(critique.rejected);
        const firstScoreNum = Number((attempts[0] || {}).score);
        const firstScore = Number.isFinite(firstScoreNum) ? firstScoreNum : 0;

        const badge = document.createElement('span');
        badge.className = 'meta-badge meta-badge-critique';
        if (regenerated) badge.classList.add('regenerated');

        let label = '';
        if (regenerated) {
            label = `self-critique · regenerated ${firstScore.toFixed(2)} → ${finalScore.toFixed(2)}`;
        } else if (accepted) {
            label = `self-critique · accepted ${finalScore.toFixed(2)}`;
        } else if (rejected) {
            label = `self-critique · rejected ${finalScore.toFixed(2)}`;
        } else {
            label = `self-critique · ${finalScore.toFixed(2)}`;
        }
        badge.textContent = label;

        const thresholdNum = Number(critique.threshold);
        const threshold = Number.isFinite(thresholdNum) ? thresholdNum : 0.65;
        // SEC: title attribute renders as plain tooltip, not HTML —
        // no escape needed for textContent paths but we still avoid
        // letting model-controlled prose end up as innerHTML below.
        const titleParts = [
            `Composite score ${finalScore.toFixed(2)} vs threshold ${threshold.toFixed(2)}.`,
        ];
        if (regenerated) {
            titleParts.push(
                `First attempt scored ${firstScore.toFixed(2)} and was rejected; ` +
                `the SLM regenerated with tighter sampling and the better ` +
                `attempt (${finalScore.toFixed(2)}) is shown.`
            );
        } else if (accepted) {
            titleParts.push('Accepted on the first SLM draft.');
        }
        badge.title = titleParts.join(' ');
        meta.appendChild(badge);
    }

    /**
     * @private
     * Append the expandable <details> "Self-critique trace" element
     * beneath the message body.  Each attempt becomes one row with
     * its score, sub-scores, sampling parameters and reasons.  Uses
     * textContent everywhere — the SLM output and critic reasons are
     * model-controlled strings and must NEVER touch innerHTML.
     */
    _appendCritiqueTrace(body, critique) {
        const attempts = Array.isArray(critique.attempts)
            ? critique.attempts : [];
        if (!attempts.length) return;

        const details = document.createElement('details');
        details.className = 'critique-trace';

        const summary = document.createElement('summary');
        summary.textContent = 'Self-critique trace';
        details.appendChild(summary);

        const finalScoreNum = Number(critique.final_score);
        const finalScore = Number.isFinite(finalScoreNum) ? finalScoreNum : 0;
        const thresholdNum = Number(critique.threshold);
        const threshold = Number.isFinite(thresholdNum) ? thresholdNum : 0.65;
        const intro = document.createElement('div');
        intro.style.marginTop = '6px';
        intro.style.fontSize = '11.5px';
        intro.textContent =
            `Composite ${finalScore.toFixed(2)} / threshold ${threshold.toFixed(2)} · ` +
            `${attempts.length} attempt${attempts.length === 1 ? '' : 's'} · ` +
            (critique.regenerated
                ? 'regenerated with tighter sampling and the better attempt was kept.'
                : 'accepted on the first draft.');
        details.appendChild(intro);

        // Identify which attempt is the kept one — we display the
        // attempt with the higher score as "kept" (this matches the
        // engine's tie-break: regen wins only when strictly better).
        let bestIdx = 0;
        let bestScore = -1;
        for (let i = 0; i < attempts.length; i++) {
            const s = Number((attempts[i] || {}).score);
            if (Number.isFinite(s) && s > bestScore) {
                bestScore = s;
                bestIdx = i;
            }
        }

        for (let i = 0; i < attempts.length; i++) {
            const a = attempts[i] || {};
            const row = document.createElement('div');
            row.className = 'critique-attempt';
            if (i === bestIdx) row.classList.add('kept');

            const scoreNum = Number(a.score);
            const scoreVal = Number.isFinite(scoreNum) ? scoreNum : 0;
            const sp = a.sampling_params && typeof a.sampling_params === 'object'
                ? a.sampling_params : {};
            const tNum = Number(sp.temperature);
            const tStr = Number.isFinite(tNum) ? `T=${tNum.toFixed(2)}` : 'T=?';
            const repNum = Number(sp.rep_pen);
            const repStr = Number.isFinite(repNum) ? `rep ${repNum.toFixed(2)}` : '';
            const samplingStr = [tStr, repStr].filter(Boolean).join(' · ');

            const head = document.createElement('div');
            head.className = 'critique-attempt-head';
            head.textContent =
                `Attempt ${i + 1} · score ${scoreVal.toFixed(2)} · ${samplingStr}` +
                (i === bestIdx ? '  (kept)' : '');
            row.appendChild(head);

            // Sub-scores row — one inline-comma-separated string so
            // the trace stays compact even with five rubrics.
            const subScores = a.sub_scores && typeof a.sub_scores === 'object'
                ? a.sub_scores : {};
            const subParts = [];
            for (const k of Object.keys(subScores)) {
                const v = Number(subScores[k]);
                if (Number.isFinite(v)) {
                    subParts.push(`${k}=${v.toFixed(2)}`);
                }
            }
            if (subParts.length) {
                const subEl = document.createElement('span');
                subEl.className = 'critique-subscores';
                subEl.textContent = subParts.join(' · ');
                row.appendChild(subEl);
            }

            // Reasons — sit on their own line so they wrap naturally.
            const reasons = Array.isArray(a.reasons) ? a.reasons : [];
            const reasonText = reasons
                .map(r => String(r || '').trim())
                .filter(Boolean)
                .join(' · ');
            if (reasonText) {
                const rEl = document.createElement('span');
                rEl.className = 'critique-reasons';
                rEl.textContent = reasonText;
                row.appendChild(rEl);
            }

            details.appendChild(row);
        }

        body.appendChild(details);
    }

    /**
     * Append a message to the chat log.
     *
     * @param {string} text          Message content.
     * @param {string} sender        "user" or "ai".
     * @param {object} [metadata={}] Optional metadata: { route, latency }.
     */
    addMessage(text, sender, metadata = {}) {
        // Hide typing indicator
        this._hideTyping();

        // SEC: Coerce text to string. Server may occasionally return null/undefined
        // or non-string types; textContent assignment below would still be safe
        // (DOM coerces it), but explicit coercion makes the contract obvious.
        const safeText = (text === null || text === undefined) ? '' : String(text);

        // SEC: Whitelist sender to prevent CSS class injection via crafted metadata.
        const safeSender = (sender === 'user' || sender === 'ai') ? sender : 'ai';

        const msgEl = document.createElement('div');
        msgEl.className = `message ${safeSender}`;

        // Avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (safeSender === 'user') {
            avatar.textContent = 'U';
        } else {
            avatar.textContent = 'I\u00B3';
            // Shift avatar colour based on emotional tone if available
            if (metadata.emotional_tone !== undefined) {
                // SEC: Clamp tone to [0, 1] so that an attacker-controlled value
                // cannot push rgb() out of range or inject CSS via NaN/Infinity.
                const toneRaw = Number(metadata.emotional_tone);
                const tone = Number.isFinite(toneRaw)
                    ? Math.max(0, Math.min(1, toneRaw))
                    : 0.5;
                // Warm amber (high tone) to cool blue (low tone)
                const r = Math.round(233 * tone + 58 * (1 - tone));
                const g = Math.round(69 * tone + 120 * (1 - tone));
                const b = Math.round(96 * tone + 210 * (1 - tone));
                avatar.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
            }
        }

        // Body
        const body = document.createElement('div');
        body.className = 'message-body';

        // Text bubble
        const textEl = document.createElement('div');
        textEl.className = 'message-text';
        // SEC: textContent (NEVER innerHTML) — this is the AI/user text sink.
        // Even if the AI returns HTML/JS, it will be rendered as plain text.
        textEl.textContent = safeText;

        body.appendChild(textEl);

        // Metadata badges
        if (metadata.route || metadata.latency !== undefined) {
            const meta = document.createElement('div');
            meta.className = 'message-meta';

            if (metadata.route) {
                // SEC: Coerce to string and use textContent. We compute a fixed
                // label ("route · edge SLM" / "route · cloud LLM") so user-
                // controlled route strings are never written to the DOM verbatim.
                const routeStr = String(metadata.route).toLowerCase();
                const routeBadge = document.createElement('span');
                const isEdge = routeStr.includes('local') ||
                               routeStr.includes('edge') ||
                               routeStr.includes('slm');
                // SEC: Use classList (whitelist) instead of string-concat className
                routeBadge.classList.add('meta-badge');
                routeBadge.classList.add(isEdge ? 'edge' : 'cloud');
                routeBadge.classList.add(
                    isEdge ? 'meta-badge-edge-route' : 'meta-badge-cloud-route'
                );
                routeBadge.textContent = isEdge
                    ? 'route · edge SLM'
                    : 'route · cloud LLM';
                const rdInner = metadata.routing_decision || null;
                if (rdInner && typeof rdInner === 'object' && rdInner.reason) {
                    routeBadge.title = String(rdInner.reason);
                }
                meta.appendChild(routeBadge);
            }

            // PII-redacted chip — only on cloud-routed turns where the
            // sanitiser actually fired.
            const rdInner = metadata.routing_decision || null;
            if (rdInner && typeof rdInner === 'object'
                && Number(rdInner.pii_redactions) > 0) {
                const piiBadge = document.createElement('span');
                piiBadge.className = 'meta-badge meta-badge-pii';
                piiBadge.textContent =
                    `pii redacted · ${Number(rdInner.pii_redactions)}`;
                piiBadge.title =
                    `${Number(rdInner.pii_redactions)} PII tokens redacted before the cloud call`
                    + (Number(rdInner.bytes_redacted) > 0
                        ? ` (${Number(rdInner.bytes_redacted)} bytes saved from the wire)`
                        : '');
                meta.appendChild(piiBadge);
            }

            if (metadata.latency !== undefined) {
                // SEC: Coerce latency to number; reject NaN/Infinity.
                const latNum = Number(metadata.latency);
                if (Number.isFinite(latNum)) {
                    const latBadge = document.createElement('span');
                    latBadge.className = 'meta-badge latency';
                    latBadge.textContent = `${Math.round(latNum)} ms`;
                    meta.appendChild(latBadge);
                }
            }

            // Response-path chip tells the reader which sub-path of
            // the hybrid stack carried the turn.  Makes the
            // retrieval-vs-generation story visible per message.
            if (metadata.response_path) {
                const pathStr = String(metadata.response_path).toLowerCase();
                const pathBadge = document.createElement('span');
                pathBadge.classList.add('meta-badge');
                let pathLabel = '';
                if (pathStr === 'retrieval') pathLabel = 'retrieval';
                else if (pathStr === 'retrieval_borderline') pathLabel = 'retrieval (low conf.)';
                else if (pathStr === 'slm') pathLabel = 'SLM generation';
                else if (pathStr === 'ood') pathLabel = 'out-of-domain';
                else if (pathStr === 'tool:math') {
                    pathLabel = 'tool · math solver';
                    pathBadge.classList.add('meta-badge-tool');
                } else if (pathStr === 'tool:refuse') {
                    pathLabel = 'tool · hostility guard';
                    pathBadge.classList.add('meta-badge-tool');
                } else if (pathStr === 'tool:entity') {
                    pathLabel = 'tool · entity';
                    pathBadge.classList.add('meta-badge-tool');
                } else if (pathStr.startsWith('tool:')) {
                    pathLabel = `tool · ${pathStr.slice(5)}`;
                    pathBadge.classList.add('meta-badge-tool');
                }
                if (pathLabel) {
                    pathBadge.textContent = pathLabel;
                    meta.appendChild(pathBadge);
                }
                // Retrieval confidence chip.
                const scoreNum = Number(metadata.retrieval_score);
                if (Number.isFinite(scoreNum) && scoreNum > 0) {
                    const confBadge = document.createElement('span');
                    confBadge.className = 'meta-badge';
                    confBadge.textContent = `conf ${scoreNum.toFixed(2)}`;
                    meta.appendChild(confBadge);
                }
            }

            // Adaptation-change chips — surfaces exactly how the live
            // AdaptationVector reshaped this specific reply.  Each
            // entry is `{axis, value, change}` from the engine's
            // ResponsePostProcessor.adapt_with_log().  Without this the
            // adaptation pathway is invisible to the user.
            if (Array.isArray(metadata.adaptation_changes)
                && metadata.adaptation_changes.length) {
                for (const ch of metadata.adaptation_changes) {
                    if (!ch || typeof ch !== 'object') continue;
                    const adaptBadge = document.createElement('span');
                    adaptBadge.className = 'meta-badge meta-badge-adapt';
                    const axis = String(ch.axis || '').replace(/_/g, ' ');
                    const change = String(ch.change || '');
                    const val = ch.value !== undefined ? String(ch.value) : '';
                    adaptBadge.textContent = val
                        ? `${axis} ${val} · ${change}`
                        : `${axis} · ${change}`;
                    adaptBadge.title = `Adaptation rewrite triggered by ${axis} = ${val}`;
                    meta.appendChild(adaptBadge);
                }
            }

            // Co-reference chip on the legacy direct-render path
            // (mirrors the logic in ``_attachMeta``).
            this._appendCorefBadge(meta, metadata.coreference_resolution);

            // Affect-shift chip on the legacy direct-render path
            // (mirrors the logic in ``_attachMeta``).
            const ashift = metadata.affect_shift;
            if (ashift && ashift.detected) {
                const dir = String(ashift.direction || 'neutral');
                const magNum = Number(ashift.magnitude);
                const magStr = Number.isFinite(magNum)
                    ? `${magNum.toFixed(1)}σ`
                    : '—';
                const affectBadge = document.createElement('span');
                affectBadge.className = 'meta-badge meta-badge-affect';
                affectBadge.textContent = `affect-shift · ${dir} · ${magStr}`;
                const suggestion = String(ashift.suggestion || '').trim();
                affectBadge.title = suggestion
                    ? `Mid-conversation affect shift; appended check-in: "${suggestion}"`
                    : `Mid-conversation affect shift detected (${dir}, ${magStr}).`;
                meta.appendChild(affectBadge);
            }

            // Self-critique chip on the legacy direct-render path
            // (mirrors the logic in ``_attachMeta``).
            const critique = metadata.critique;
            if (critique && typeof critique === 'object') {
                this._appendCritiqueBadge(meta, critique);
            }

            // Voice-prosody multimodal-fusion chip on the legacy
            // direct-render path (mirrors the logic in ``_attachMeta``).
            this._appendMultimodalBadge(meta, metadata.multimodal);

            // Vision-gaze chip (mirrors the logic above).
            this._appendGazeBadge(meta, metadata.gaze);

            body.appendChild(meta);

            if (critique && typeof critique === 'object') {
                this._appendCritiqueTrace(body, critique);
            }

            // Iteration 12 (2026-04-26): mirror the explain-plan and
            // Decision Trace expanders that ``_attachMeta`` (used by
            // the streaming finaliseStreaming path) renders, so the
            // direct-render addMessage path produces identical UI.
            const explainPlanLegacy = metadata.explain_plan;
            if (explainPlanLegacy && typeof explainPlanLegacy === 'object'
                && Array.isArray(explainPlanLegacy.sub_questions)
                && explainPlanLegacy.sub_questions.length > 0) {
                this._appendExplainPlan(body, explainPlanLegacy);
            }
            this._appendDecisionTrace(body, metadata);
        }

        msgEl.appendChild(avatar);
        msgEl.appendChild(body);

        // Iter 51: side-channel chips — affect-shift, safety caveat,
        // adaptation summary, intent-result.  These used to be
        // appended into the chat-bubble text; now rendered as small
        // pills next to the bubble so the chat stays clean.
        if (safeSender === 'ai') {
            try {
                this._appendSideChips(msgEl, metadata);
            } catch (_e) { /* never let chip render kill the bubble */ }
        }

        // Per-bubble TTS — only on AI bubbles, never user bubbles.
        if (safeSender === 'ai') {
            this._appendBubbleTts(msgEl, body);
        }

        this.container.appendChild(msgEl);
        this.messageCount++;

        // Notify floating preference toast that another AI msg landed.
        if (safeSender === 'ai') {
            try {
                window.dispatchEvent(new CustomEvent('i3:ai-message-landed', {
                    detail: { messageCount: this.messageCount },
                }));
            } catch (_e) { /* no-op */ }
        }

        // Auto-scroll to bottom
        this._scrollToBottom();
    }

    /**
     * @private
     * Append a small "speak" icon button to the AI message body.
     * Hides the bulky inline TTS player; the click handler triggers the
     * existing /api/tts pipeline via a custom event the global TTS
     * module listens for.  Soft-fail: never breaks message rendering.
     */
    /**
     * Iter 51 (2026-04-27) — render side-channel chips next to the
     * bubble.  These used to be appended into the chat-bubble text
     * (creating noise on benign turns); now they're small pills the
     * user can scan without polluting the answer.
     *
     * Chips rendered (when present in metadata):
     *   - .chip-affect  : metadata.affect_shift.suggestion
     *   - .chip-safety  : metadata.safety_caveat
     *   - .chip-adapt   : metadata.adaptation_changes summary
     *   - .chip-intent  : metadata.intent_result.action (if WS shipped one)
     */
    _appendSideChips(msgEl, metadata) {
        if (!metadata || typeof metadata !== 'object') return;
        const chips = [];
        // Affect shift.
        if (metadata.affect_shift && metadata.affect_shift.detected
                && metadata.affect_shift.suggestion) {
            chips.push({
                cls: 'chip-affect',
                title: String(metadata.affect_shift.suggestion).slice(0, 200),
                text: '⚡ interaction shift',
            });
        }
        // Safety caveat.
        if (metadata.safety_caveat && typeof metadata.safety_caveat === 'string') {
            chips.push({
                cls: 'chip-safety',
                title: metadata.safety_caveat.slice(0, 220),
                text: 'ⓘ moderation note',
            });
        }
        // Adaptation summary (top 2 axes that moved).
        if (Array.isArray(metadata.adaptation_changes)
                && metadata.adaptation_changes.length) {
            metadata.adaptation_changes.slice(0, 2).forEach(function (c) {
                if (c && c.axis) {
                    chips.push({
                        cls: 'chip-adapt',
                        title: 'Adaptation rewrite: ' + c.axis +
                               (c.change ? ' — ' + c.change : ''),
                        text: '↻ ' + String(c.axis).replace(/_/g, ' '),
                    });
                }
            });
        }
        // Intent-parser result, if the engine attached one.
        if (metadata.intent_result && metadata.intent_result.action) {
            const ir = metadata.intent_result;
            chips.push({
                cls: 'chip-intent',
                title: 'Intent: ' + ir.action +
                       ' params: ' + JSON.stringify(ir.params || {}),
                text: '◆ ' + ir.action,
            });
        }
        // Iter 51 phase 10: per-arm indicator chips.  Show all THREE
        // arms (200M from-scratch SLM, Qwen 1.7B+LoRA, Gemini cloud)
        // on every message, lit green when fired and grey when not,
        // so the recruiter watches the cascade light up in real time
        // rather than guessing from a single winner chip.
        const rd0 = metadata.route_decision;
        if (rd0 && typeof rd0 === 'object' && rd0.arms_used) {
            const arms = rd0.arms_used;
            const tipBase = (label, used) =>
                used ? `${label}: USED on this turn` : `${label}: not used on this turn`;
            chips.push({
                cls: 'chip-arm chip-arm-indicator chip-arm-slm '
                     + (arms.slm ? 'chip-arm-on' : 'chip-arm-off'),
                title: tipBase('SLM (204M from-scratch transformer + retrieval)', arms.slm),
                text: (arms.slm ? '● ' : '○ ') + 'SLM 204M',
            });
            chips.push({
                cls: 'chip-arm chip-arm-indicator chip-arm-qwen '
                     + (arms.qwen ? 'chip-arm-on' : 'chip-arm-off'),
                title: tipBase('Qwen 1.7B + LoRA intent parser', arms.qwen),
                text: (arms.qwen ? '● ' : '○ ') + 'Qwen 1.7B',
            });
            chips.push({
                cls: 'chip-arm chip-arm-indicator chip-arm-gemini '
                     + (arms.gemini ? 'chip-arm-on' : 'chip-arm-off'),
                title: tipBase('Gemini 2.5 Flash (cloud)', arms.gemini),
                text: (arms.gemini ? '● ' : '○ ') + 'Gemini 2.5',
            });
        }
        // Iter 51 phase 11: prominent "Used: X" badge at the end of
        // the chip row.  The three indicator chips above show the
        // on/off state; this badge names the winner explicitly so
        // the recruiter sees "Used: Qwen LoRA" / "Used: Gemini 2.5
        // Flash" / etc. without having to interpret the bullet colours.
        if (rd0 && typeof rd0 === 'object' && rd0.arm) {
            const usedLabelMap = {
                'slm':              'Used: from-scratch SLM (204M)',
                'slm+retrieval':    'Used: from-scratch SLM + retrieval',
                'qwen-lora':        'Used: Qwen 1.7B + LoRA',
                'gemini-backup':    'Used: Qwen → Gemini (backup)',
                'gemini-chat':      'Used: Gemini 2.5 Flash (cloud)',
                'diary':            'Used: encrypted diary (no LLM)',
                'hostility-guard':  'Used: hostility guard (no LLM)',
                'ood':              'Used: safe fallback (no LLM)',
            };
            const usedClassMap = {
                'slm':              'chip-used-slm',
                'slm+retrieval':    'chip-used-slm',
                'qwen-lora':        'chip-used-qwen',
                'gemini-backup':    'chip-used-gemini',
                'gemini-chat':      'chip-used-gemini',
                'diary':            'chip-used-tool',
                'hostility-guard':  'chip-used-tool',
                'ood':              'chip-used-tool',
            };
            const usedLabel = usedLabelMap[rd0.arm] || ('Used: ' + rd0.arm);
            const usedCls   = usedClassMap[rd0.arm] || 'chip-used-tool';
            const usedTip = [
                'arm: ' + rd0.arm,
                'model: ' + (rd0.model || '—'),
                'class: ' + (rd0.query_class || '—'),
                'reason: ' + (rd0.reason || '—'),
                'threshold: ' + (rd0.threshold || '—'),
            ].join('\n');
            chips.push({
                cls: 'chip-used ' + usedCls,
                title: usedTip,
                text: usedLabel,
            });
        }

        // Legacy ``response_path``-only fallback (kept for any client
        // that hasn't received a route_decision yet).
        if (!rd0 && metadata.response_path) {
            // Legacy fallback (kept until every server ships
            // route_decision).  Same minimal arm chip as before.
            const armMap = {
                'slm':         { label: 'A · SLM',          cls: 'chip-arm-a' },
                'tool:intent': { label: 'B · Qwen LoRA',    cls: 'chip-arm-b' },
                'cloud_chat':  { label: 'C · Gemini cloud', cls: 'chip-arm-c' },
                'cloud_llm':   { label: 'C · cloud',        cls: 'chip-arm-c' },
                'retrieval':   { label: 'R · retrieval',    cls: 'chip-arm-r' },
            };
            const path = metadata.response_path;
            let arm = armMap[path];
            if (!arm && typeof path === 'string' && path.startsWith('tool:')) {
                arm = { label: 'T · ' + path.replace('tool:', ''),
                        cls: 'chip-arm-t' };
            }
            if (!arm && typeof path === 'string' && path.startsWith('retrieval')) {
                arm = armMap['retrieval'];
            }
            if (arm) {
                chips.push({
                    cls: arm.cls + ' chip-arm',
                    title: 'Cascade arm: ' + path,
                    text: arm.label,
                });
            }
        }
        if (!chips.length) return;
        const wrap = document.createElement('div');
        wrap.className = 'message-chips';
        chips.forEach(function (chip) {
            const span = document.createElement('span');
            span.className = 'message-chip ' + chip.cls;
            span.title = chip.title;
            span.textContent = chip.text;
            wrap.appendChild(span);
        });
        msgEl.appendChild(wrap);
    }

    _appendBubbleTts(msgEl, body) {
        try {
            // Don't double-attach (defensive).
            if (body.querySelector(':scope > .i3-bubble-tts')) return;
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'i3-bubble-tts';
            btn.setAttribute('aria-label', 'Speak this response');
            btn.title = 'Speak this response';
            btn.innerHTML = '<span class="i3-bubble-tts-icon" aria-hidden="true">🔊</span>';
            btn.addEventListener('click', (ev) => {
                ev.stopPropagation();
                btn.classList.add('is-loading');
                try {
                    const textEl = body.querySelector('.message-text');
                    const text = textEl ? textEl.textContent.trim() : '';
                    window.dispatchEvent(new CustomEvent('i3:speak-bubble', {
                        detail: { text, button: btn },
                    }));
                } catch (e) {
                    btn.classList.remove('is-loading');
                }
            });
            body.appendChild(btn);
        } catch (_e) { /* never break the bubble */ }
    }

    /**
     * Wire up the chat input field and send button.
     *
     * @param {HTMLInputElement} inputElement The text input field.
     * @param {HTMLButtonElement} sendButton  The send button.
     */
    setupInput(inputElement, sendButton) {
        this.inputEl = inputElement;

        // Attach keystroke monitor
        if (this.ksMonitor) {
            this.ksMonitor.attach(inputElement);
        }

        // SEC: Defensive client-side cap on outgoing message length.
        // Server is the source of truth for sanitisation/limits, but capping
        // here prevents accidentally sending megabyte payloads.
        const MAX_INPUT_CHARS = 4000;

        const doSend = () => {
            let text = inputElement.value.trim();
            if (!text) return;
            if (text.length > MAX_INPUT_CHARS) {
                // SEC: Truncate rather than reject so the user gets feedback.
                text = text.slice(0, MAX_INPUT_CHARS);
            }

            // Gather composition metrics before resetting
            const metrics = this.ksMonitor ? this.ksMonitor.getCompositionMetrics() : {};

            // Voice-prosody features (flagship #2).  Attached only when
            // the user has the mic enabled AND we have at least one
            // frame buffered; otherwise the field is omitted so the
            // server takes the keystroke-only branch.  The audio buffer
            // never crosses the wire — only these eight scalars do.
            // See web/js/voice_prosody.js for the full privacy contract.
            let prosody_features = null;
            if (this.voiceProsody && this.voiceProsody.isEnabled()) {
                try {
                    prosody_features = this.voiceProsody.getCurrentFeatures();
                } catch (e) {
                    console.warn('[I3] prosody features failed:', e);
                    prosody_features = null;
                }
            }

            // Gaze features (vision flagship).  Attached only when the
            // user has the camera enabled AND we have at least one
            // inference completed; otherwise the field is omitted so
            // the server takes the camera-off branch.  Raw frames never
            // cross the wire — only the predicted label + 5 scalars.
            // See web/js/gaze_capture.js for the privacy contract.
            let gaze_features = null;
            if (this.gazeCapture && this.gazeCapture.isEnabled()) {
                try {
                    gaze_features = this.gazeCapture.getCurrentFeatures();
                } catch (e) {
                    console.warn('[I3] gaze features failed:', e);
                    gaze_features = null;
                }
            }

            // Add user message to chat
            this.addMessage(text, 'user');

            // Send to server
            const wsFrame = {
                type: 'message',
                text: text,
                timestamp: Date.now() / 1000,
                composition_metrics: metrics,
            };
            if (prosody_features) {
                wsFrame.prosody_features = prosody_features;
            }
            if (gaze_features) {
                wsFrame.gaze_features = gaze_features;
            }
            this.ws.send(wsFrame);

            // Clear input and reset monitor
            inputElement.value = '';
            if (this.ksMonitor) {
                this.ksMonitor.reset();
            }

            // Show typing indicator
            this._showTyping();
        };

        // Enter to send
        inputElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                doSend();
            }
        });

        // Send button click
        if (sendButton) {
            sendButton.addEventListener('click', doSend);
        }
    }

    /**
     * Set up the typing indicator element.
     * @param {HTMLElement} indicator The .typing-indicator element.
     */
    setTypingIndicator(indicator) {
        this.typingIndicator = indicator;
    }

    /** @private */
    _showTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.classList.add('visible');
            this._scrollToBottom();
        }
    }

    /** @private */
    _hideTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.classList.remove('visible');
        }
    }

    /** @private */
    _scrollToBottom() {
        requestAnimationFrame(() => {
            this.container.scrollTop = this.container.scrollHeight;
        });
    }
}
