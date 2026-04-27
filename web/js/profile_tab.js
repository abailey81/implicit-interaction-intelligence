/**
 * Profile tab renderer for Implicit Interaction Intelligence (I3).
 *
 * Reads /api/profile/{user_id}/{session_id} on tab activation and
 * refreshes on every WebSocket state_update so the tile values track
 * the live session.  All visualisations are hand-rolled SVG / CSS to
 * keep the page free of chart-library dependencies.
 *
 * Cites Monrose & Rubin (1997) and Killourhy & Maxion (2009) by way
 * of the biometric tile.
 */

(function () {
    'use strict';

    // Donut palette — colour-blind safe-ish set.  Same colour for the
    // same state across sessions so the user can build intuition.
    const STATE_COLOURS = {
        'calm':         '#2997ff',
        'focused':      '#30d158',
        'stressed':     '#ff6259',
        'tired':        '#b495d8',
        'distracted':   '#ff9f0a',
        'warming up':   '#888c95',
    };

    const FALLBACK_COLOUR = '#888c95';

    let _refreshTimer = null;
    let _lastFetchAt = 0;

    /** Get the current user / session ids from the I3App instance. */
    function _ids() {
        const app = window.app;
        if (!app) return { userId: 'demo_user', sessionId: '' };
        return {
            userId: app.userId || 'demo_user',
            sessionId: app.sessionId || '',
        };
    }

    /** Coerce to a finite number, fall back to 0. */
    function _num(v) {
        const n = Number(v);
        return Number.isFinite(n) ? n : 0;
    }

    /** Render the spark line at #id from a list of numbers. */
    function _sparkLine(id, values) {
        const svg = document.getElementById(id);
        if (!svg) return;
        // Clear previous polyline.
        svg.innerHTML = '';
        if (!Array.isArray(values) || values.length < 2) return;
        const xs = values.map(_num);
        const min = Math.min(...xs);
        const max = Math.max(...xs);
        const range = (max - min) || 1;
        const W = 120;
        const H = 30;
        const points = xs.map((v, i) => {
            const x = (i / (xs.length - 1)) * W;
            const y = H - ((v - min) / range) * (H - 4) - 2;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(' ');
        const polyline = document.createElementNS(
            'http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', points);
        svg.appendChild(polyline);
    }

    /** Render a 5-bar histogram for cognitive_load distribution. */
    function _histBars(id, fractions) {
        const host = document.getElementById(id);
        if (!host) return;
        host.innerHTML = '';
        const arr = Array.isArray(fractions) ? fractions : [];
        const max = Math.max(0.001, ...arr.map(_num));
        for (let i = 0; i < 5; i++) {
            const bar = document.createElement('div');
            bar.className = 'hist-bar';
            const f = _num(arr[i]);
            bar.style.height = `${Math.max(2, (f / max) * 36)}px`;
            bar.title = `${(f * 100).toFixed(0)}%`;
            host.appendChild(bar);
        }
    }

    /** Render a tiny donut chart for the discrete state distribution. */
    function _donut(svgEl, fractions) {
        if (!svgEl) return;
        svgEl.innerHTML = '';
        const entries = Object.entries(fractions || {});
        const total = entries.reduce((a, [, v]) => a + _num(v), 0) || 1;
        // Background ring.
        const bg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        bg.setAttribute('cx', '18');
        bg.setAttribute('cy', '18');
        bg.setAttribute('r', '15.915');
        bg.setAttribute('stroke', 'rgba(255,255,255,0.06)');
        bg.setAttribute('stroke-width', '4');
        bg.setAttribute('fill', 'transparent');
        svgEl.appendChild(bg);
        let acc = 0;
        const C = 2 * Math.PI * 15.915;  // circumference
        for (const [name, v] of entries) {
            const frac = _num(v) / total;
            if (frac <= 0) continue;
            const seg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            seg.setAttribute('cx', '18');
            seg.setAttribute('cy', '18');
            seg.setAttribute('r', '15.915');
            seg.setAttribute('stroke', STATE_COLOURS[name] || FALLBACK_COLOUR);
            seg.setAttribute('stroke-width', '4');
            seg.setAttribute('fill', 'transparent');
            seg.setAttribute('stroke-dasharray', `${(frac * C).toFixed(2)} ${C.toFixed(2)}`);
            seg.setAttribute('stroke-dashoffset', `${(-acc * C).toFixed(2)}`);
            seg.setAttribute('transform', 'rotate(-90 18 18)');
            seg.setAttribute('stroke-linecap', 'butt');
            svgEl.appendChild(seg);
            acc += frac;
        }
    }

    function _donutLegend(id, fractions) {
        const ul = document.getElementById(id);
        if (!ul) return;
        ul.innerHTML = '';
        const entries = Object.entries(fractions || {});
        if (!entries.length) {
            const li = document.createElement('li');
            li.textContent = 'no data yet';
            ul.appendChild(li);
            return;
        }
        // Sort by fraction desc, top 5.
        entries.sort((a, b) => _num(b[1]) - _num(a[1]));
        for (const [name, frac] of entries.slice(0, 5)) {
            const li = document.createElement('li');
            const sw = document.createElement('span');
            sw.className = 'legend-swatch';
            sw.style.background = STATE_COLOURS[name] || FALLBACK_COLOUR;
            li.appendChild(sw);
            const label = document.createElement('span');
            label.textContent = `${name} ${(_num(frac) * 100).toFixed(0)}%`;
            li.appendChild(label);
            ul.appendChild(li);
        }
    }

    function _styleAxes(id, prefs) {
        const host = document.getElementById(id);
        if (!host) return;
        host.innerHTML = '';
        const axes = [
            ['formality',     _num(prefs?.formality_avg)],
            ['verbosity',     _num(prefs?.verbosity_avg)],
            ['accessibility', _num(prefs?.accessibility_avg)],
        ];
        for (const [name, v] of axes) {
            const row = document.createElement('div');
            row.className = 'style-axis';

            const label = document.createElement('span');
            label.textContent = name;
            row.appendChild(label);

            const track = document.createElement('span');
            track.className = 'style-track';
            const fill = document.createElement('span');
            fill.className = 'style-fill';
            fill.style.width = `${Math.max(0, Math.min(1, v)) * 100}%`;
            track.appendChild(fill);
            row.appendChild(track);

            const val = document.createElement('span');
            val.className = 'style-value';
            val.textContent = v.toFixed(2);
            row.appendChild(val);

            host.appendChild(row);
        }
    }

    /** Apply a fetched snapshot to the tiles. */
    function applySnapshot(snap) {
        if (!snap || typeof snap !== 'object') return;

        // Tile 1: biometric.
        const bio = snap.biometric || {};
        const bioStateEl = document.getElementById('profile-bio-state');
        const bioMetaEl = document.getElementById('profile-bio-meta');
        if (bioStateEl) {
            const state = String(bio.state || 'unregistered');
            const conf = _num(bio.confidence);
            const sim = _num(bio.similarity);
            const owner = !!bio.is_owner;
            let label = state;
            if (state === 'verifying' && owner) label = `You · ${conf.toFixed(2)}`;
            else if (state === 'mismatch' || bio.drift_alert) label = `⚠ ${state}`;
            else if (state === 'registering') {
                label = `learning ${_num(bio.enrolment_progress)}/${_num(bio.enrolment_target) || 5}`;
            } else if (state === 'registered' && owner) label = 'Registered';
            bioStateEl.textContent = label;
        }
        if (bioMetaEl) {
            const drifts = _num(snap.biometric_drifts);
            bioMetaEl.textContent = bio.notes
                ? `${bio.notes} · drift events: ${drifts}`
                : `template · drift events: ${drifts}`;
        }

        // Tile 2: typing rhythm.
        const iki = snap.iki || {};
        const ikiHeadline = document.getElementById('profile-iki-headline');
        if (ikiHeadline) ikiHeadline.textContent = `${_num(iki.mean).toFixed(0)} ms`;
        const ikiMeta = document.getElementById('profile-iki-meta');
        if (ikiMeta) {
            const pct = _num(iki.vs_baseline_pct);
            const sign = pct >= 0 ? '+' : '';
            ikiMeta.textContent = `mean · ${sign}${pct.toFixed(0)}% vs baseline · σ ${_num(iki.std).toFixed(0)} ms`;
        }
        _sparkLine('profile-iki-spark', iki.history);

        // Tile 3: composition cadence.
        const comp = snap.composition || {};
        const compHeadline = document.getElementById('profile-comp-headline');
        if (compHeadline) compHeadline.textContent = `${(_num(comp.mean_ms) / 1000).toFixed(2)} s`;
        const compMeta = document.getElementById('profile-comp-meta');
        if (compMeta) {
            const fp = (_num(comp.fast_turn_pct) * 100).toFixed(0);
            compMeta.textContent = `avg compose · ${fp}% fast turns (<1.5s)`;
        }
        _sparkLine('profile-comp-spark', comp.history);

        // Tile 4: edits.
        const edits = snap.edits || {};
        const editsHeadline = document.getElementById('profile-edits-headline');
        if (editsHeadline) editsHeadline.textContent = `${_num(edits.mean_per_turn).toFixed(2)} / turn`;
        const editsMeta = document.getElementById('profile-edits-meta');
        if (editsMeta) editsMeta.textContent = `peak burst: ${_num(edits.peak_burst).toFixed(0)} edits in one turn`;
        _sparkLine('profile-edits-spark', edits.history);

        // Tile 5: cognitive load.
        const cl = snap.cognitive_load || {};
        const clHeadline = document.getElementById('profile-cl-headline');
        if (clHeadline) clHeadline.textContent = `${_num(cl.mean).toFixed(2)} avg`;
        _histBars('profile-cl-bars', cl.histogram);

        // Tile 6: style preferences.
        const sp = snap.style_preferences || {};
        const styleHeadline = document.getElementById('profile-style-headline');
        if (styleHeadline) {
            styleHeadline.textContent = `formality ${_num(sp.formality_avg).toFixed(2)}`;
        }
        _styleAxes('profile-style-bars', sp);

        // Tile 7: state distribution.
        const sd = snap.state_distribution || {};
        const stateHeadline = document.getElementById('profile-state-headline');
        if (stateHeadline) {
            const top = Object.entries(sd).sort((a, b) => _num(b[1]) - _num(a[1]))[0];
            stateHeadline.textContent = top
                ? `${top[0]} ${(_num(top[1]) * 100).toFixed(0)}%`
                : '— no data';
        }
        const donutSvg = document.querySelector('#profile-state-donut .donut-svg');
        _donut(donutSvg, sd);
        _donutLegend('profile-state-legend', sd);

        // Tile 8: affect shifts.
        const affect = snap.affect_shifts || {};
        const affectHeadline = document.getElementById('profile-affect-headline');
        if (affectHeadline) affectHeadline.textContent = `${_num(affect.total)}`;
        const affectMeta = document.getElementById('profile-affect-meta');
        if (affectMeta) {
            const ts = affect.last_ts;
            let last = '—';
            if (ts) {
                const d = new Date(_num(ts) * 1000);
                last = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            }
            affectMeta.textContent = `total · last event ${last}`;
        }

        // Recent activity log.
        _renderRecentLog(snap);
    }

    function _renderRecentLog(snap) {
        const ul = document.getElementById('profile-event-log');
        if (!ul) return;
        // Build a small synthetic log from the snapshot.
        const events = [];
        const session = _num(snap.session_messages);
        if (session > 0) {
            events.push({
                t: '',
                text: `Session has ${session} messages over ${_num(snap.session_duration_seconds).toFixed(0)}s.`,
            });
        }
        const drifts = _num(snap.biometric_drifts);
        if (drifts > 0) {
            events.push({ t: '', text: `Biometric drift detected ${drifts} time(s) this session.` });
        }
        const affect = snap.affect_shifts || {};
        if (_num(affect.total) > 0) {
            events.push({ t: '', text: `Affect-shift events: ${_num(affect.total)} (last surfaced as a check-in to the user).` });
        }
        const bio = snap.biometric || {};
        if (bio.state) {
            events.push({ t: '', text: `Biometric state: ${bio.state} (confidence ${_num(bio.confidence).toFixed(2)}).` });
        }
        ul.innerHTML = '';
        if (!events.length) {
            const li = document.createElement('li');
            li.className = 'profile-event-empty';
            li.textContent = 'No events yet. Send a few messages to populate this log.';
            ul.appendChild(li);
            return;
        }
        for (const e of events) {
            const li = document.createElement('li');
            const t = document.createElement('time');
            t.textContent = e.t || '';
            li.appendChild(t);
            const span = document.createElement('span');
            span.textContent = e.text;
            li.appendChild(span);
            ul.appendChild(li);
        }
    }

    /** Throttled fetch + apply. */
    async function refresh() {
        const now = Date.now();
        if (now - _lastFetchAt < 600) return;  // 600ms throttle
        _lastFetchAt = now;
        const { userId, sessionId } = _ids();
        if (!userId || !sessionId) return;
        try {
            const res = await fetch(
                `/api/profile/${encodeURIComponent(userId)}/${encodeURIComponent(sessionId)}`,
                { credentials: 'same-origin', cache: 'no-store' },
            );
            if (!res.ok) return;
            const snap = await res.json();
            applySnapshot(snap);
        } catch (e) {
            // Decorative — never throw into the user's chat path.
        }
        // Personalisation tile lives off a separate endpoint so the
        // existing /api/profile schema doesn't have to grow.
        try { refreshPersonalisation(); } catch (e) { /* decorative */ }
        // Gaze tile — same pattern, separate endpoint.
        try { refreshGaze(); } catch (e) { /* decorative */ }
    }

    /** Render the per-user gaze classifier tile. */
    async function refreshGaze() {
        const { userId } = _ids();
        if (!userId) return;
        let payload = null;
        try {
            const res = await fetch(
                `/api/gaze/${encodeURIComponent(userId)}/status`,
                { credentials: 'same-origin', cache: 'no-store' },
            );
            if (!res.ok) return;
            payload = await res.json();
        } catch (e) { return; }
        applyGaze(payload);
    }

    function applyGaze(p) {
        if (!p || typeof p !== 'object') return;
        const headline = document.getElementById('profile-gaze-headline');
        const meta = document.getElementById('profile-gaze-meta');
        const liveEl = document.getElementById('profile-gaze-live');
        const onScreenEl = document.getElementById('profile-gaze-onscreen');
        const stabilityEl = document.getElementById('profile-gaze-stability');
        const paramsEl = document.getElementById('profile-gaze-params');
        const cm = p.calibration_meta || {};
        const calibrated = !!p.calibrated;

        if (headline) {
            if (!calibrated) {
                headline.textContent = 'Not calibrated';
            } else {
                const acc = (_num(cm.val_accuracy) * 100).toFixed(0);
                const n = _num(cm.n_frames_used);
                headline.textContent = `Calibrated · ${acc}% val · ${n} frames`;
            }
        }
        if (meta) {
            if (!calibrated) {
                meta.textContent =
                    'enable the camera and press «Calibrate gaze» ' +
                    'to fine-tune the head';
            } else {
                let when = '';
                const ts = _num(cm.calibrated_at);
                if (ts > 0) {
                    const d = new Date(ts * 1000);
                    when = ' at ' + d.toLocaleTimeString([], {
                        hour: '2-digit', minute: '2-digit',
                    });
                }
                meta.textContent =
                    `frozen MobileNetV3-small backbone (5.4 M params) ` +
                    `+ fine-tuned head (75 k params)${when}`;
            }
        }

        // Try to surface a live label from the GazeCaptureMonitor if
        // it's been mounted.
        if (liveEl) {
            const monitor = window.app && window.app.gazeCapture;
            if (monitor && monitor.isEnabled && monitor.isEnabled()) {
                const live = monitor.getLiveDisplay();
                liveEl.textContent =
                    `${(live.label || 'unknown').replace('_', '-')} ` +
                    `${_num(live.confidence).toFixed(2)}`;
            } else {
                liveEl.textContent = 'camera off';
            }
        }
        if (onScreenEl) {
            // Aggregate from the monitor if available.
            const monitor = window.app && window.app.gazeCapture;
            if (monitor && monitor.isEnabled && monitor.isEnabled()) {
                const f = monitor.getCurrentFeatures();
                if (f && f.label_probs) {
                    onScreenEl.textContent =
                        (f.label_probs.at_screen * 100).toFixed(0) + '%';
                } else {
                    onScreenEl.textContent = '—';
                }
            } else {
                onScreenEl.textContent = '—';
            }
        }
        if (stabilityEl) {
            const monitor = window.app && window.app.gazeCapture;
            if (monitor && monitor.isEnabled && monitor.isEnabled()) {
                const f = monitor.getCurrentFeatures();
                if (f) {
                    stabilityEl.textContent = _num(f.head_stability).toFixed(2);
                } else stabilityEl.textContent = '—';
            } else stabilityEl.textContent = '—';
        }
        if (paramsEl) {
            const np = _num(p.n_head_params);
            paramsEl.textContent = np > 0 ? np.toLocaleString() : '74 372';
        }

        // Wire the re-calibrate button to call into the global monitor.
        const btn = document.getElementById('profile-gaze-recal');
        if (btn) {
            btn.onclick = () => {
                const monitor = window.app && window.app.gazeCapture;
                if (!monitor || !monitor.isEnabled || !monitor.isEnabled()) {
                    alert('Enable the camera first (the 📷 button next to send).');
                    return;
                }
                // Trigger the calibration overlay by clicking the
                // calibrate button mounted next to the chat input.
                const calBtn = document.querySelector('.gaze-calibrate-btn');
                if (calBtn) calBtn.click();
            };
        }
    }

    /** Render the per-biometric LoRA personalisation tile. */
    async function refreshPersonalisation() {
        const { userId } = _ids();
        if (!userId) return;
        let payload = null;
        try {
            const res = await fetch(
                `/api/personalisation/${encodeURIComponent(userId)}/status`,
                { credentials: 'same-origin', cache: 'no-store' },
            );
            if (!res.ok) return;
            payload = await res.json();
        } catch (e) {
            return; // decorative
        }
        applyPersonalisation(payload);
    }

    function applyPersonalisation(p) {
        if (!p || typeof p !== 'object') return;
        const headline = document.getElementById('profile-pers-headline');
        const meta = document.getElementById('profile-pers-meta');
        const paramsEl = document.getElementById('profile-pers-params');
        const storageEl = document.getElementById('profile-pers-storage');
        const rankEl = document.getElementById('profile-pers-rank');
        const driftHost = document.getElementById('profile-pers-drift');

        const registered = !!p.biometric_registered;
        const nUpdates = _num(p.n_updates);
        if (headline) {
            if (!registered) {
                headline.textContent = 'No biometric — base model';
            } else if (nUpdates === 0) {
                headline.textContent = 'Active (0 updates)';
            } else {
                headline.textContent = `Active (${nUpdates} update${nUpdates === 1 ? '' : 's'})`;
            }
        }
        if (meta) {
            if (!registered) {
                meta.textContent = 'register a typing template to activate per-user LoRA';
            } else {
                const userKey = String(p.user_key || '').slice(0, 12);
                meta.textContent = `keyed by SHA-256 of biometric template (${userKey || '—'}…)`;
            }
        }
        if (paramsEl) {
            const np = _num(p.num_parameters);
            paramsEl.textContent = np > 0 ? `${np}` : '288';
        }
        if (storageEl) {
            // 288 floats x ~12 bytes JSON each + overhead lands at 4 KB.
            storageEl.textContent = '~4 KB';
        }
        if (rankEl) {
            rankEl.textContent = String(_num(p.rank) || 4);
        }
        if (driftHost) {
            driftHost.innerHTML = '';
            const drift = (p.cumulative_drift && typeof p.cumulative_drift === 'object')
                ? p.cumulative_drift : {};
            const axes = ['cognitive_load', 'formality', 'verbosity', 'emotionality', 'directness', 'emotional_tone', 'accessibility'];
            for (const axis of axes) {
                const val = _num(drift[axis]);
                const row = document.createElement('div');
                row.className = 'pers-drift-row';
                const label = document.createElement('span');
                label.className = 'pers-drift-label';
                label.textContent = axis;
                const barWrap = document.createElement('div');
                barWrap.className = 'pers-drift-bar-wrap';
                const bar = document.createElement('div');
                bar.className = 'pers-drift-bar';
                // val is in roughly [-0.15, +0.15]; map to [-100%, +100%] of half-bar.
                const pct = Math.max(-1, Math.min(1, val / 0.15));
                const widthPct = Math.abs(pct * 50);  // half-bar is 50%
                bar.style.width = `${widthPct.toFixed(1)}%`;
                if (pct >= 0) {
                    bar.classList.add('pers-drift-positive');
                    bar.style.left = '50%';
                } else {
                    bar.classList.add('pers-drift-negative');
                    bar.style.right = '50%';
                }
                const valLabel = document.createElement('span');
                valLabel.className = 'pers-drift-value';
                valLabel.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(3)}`;
                barWrap.appendChild(bar);
                row.appendChild(label);
                row.appendChild(barWrap);
                row.appendChild(valLabel);
                driftHost.appendChild(row);
            }
        }

        // Wire the reset button (idempotent — re-binding is fine, the
        // previous handler is replaced by setting onclick).
        const btn = document.getElementById('profile-pers-reset');
        if (btn) {
            btn.onclick = async () => {
                const { userId } = _ids();
                if (!userId) return;
                btn.disabled = true;
                try {
                    await fetch(
                        `/api/personalisation/${encodeURIComponent(userId)}/reset`,
                        { method: 'POST', credentials: 'same-origin' },
                    );
                } catch (e) { /* decorative */ }
                btn.disabled = false;
                setTimeout(refreshPersonalisation, 100);
            };
        }
    }

    /** Schedule a debounced refresh after a state_update arrives. */
    function scheduleRefresh() {
        if (_refreshTimer) return;
        _refreshTimer = setTimeout(() => {
            _refreshTimer = null;
            refresh();
        }, 250);
    }

    function isProfileTabActive() {
        const tab = document.getElementById('tab-profile');
        return !!(tab && !tab.hidden);
    }

    document.addEventListener('DOMContentLoaded', () => {
        // Refresh when the Profile tab is selected.
        const profileLink = document.querySelector('a[data-tab="profile"]');
        if (profileLink) {
            profileLink.addEventListener('click', () => {
                // Allow tab_router to flip the panel hidden flag first.
                setTimeout(refresh, 50);
            });
        }
        // Also refresh on hash change in case the tab was opened via URL.
        window.addEventListener('hashchange', () => {
            if (location.hash === '#profile') setTimeout(refresh, 50);
        });
        if (location.hash === '#profile') setTimeout(refresh, 200);

        // Hook the WS state_update (set up via the I3App which fans
        // events out to handlers).  We listen on a custom CustomEvent
        // bus the I3App fires when the panel is visible.
        if (window.app && window.app.wsClient) {
            window.app.wsClient.on('state_update', () => {
                if (isProfileTabActive()) scheduleRefresh();
            });
            window.app.wsClient.on('response', () => {
                if (isProfileTabActive()) scheduleRefresh();
            });
            window.app.wsClient.on('response_done', () => {
                if (isProfileTabActive()) scheduleRefresh();
            });
        } else {
            // The app might bootstrap *after* this script.  Poll once.
            const tries = 20;
            let i = 0;
            const interval = setInterval(() => {
                i++;
                if (window.app && window.app.wsClient) {
                    window.app.wsClient.on('state_update', () => {
                        if (isProfileTabActive()) scheduleRefresh();
                    });
                    window.app.wsClient.on('response', () => {
                        if (isProfileTabActive()) scheduleRefresh();
                    });
                    window.app.wsClient.on('response_done', () => {
                        if (isProfileTabActive()) scheduleRefresh();
                    });
                    clearInterval(interval);
                } else if (i >= tries) {
                    clearInterval(interval);
                }
            }, 250);
        }
    });

    // Expose for manual debugging.
    window.__i3RefreshProfile = refresh;
})();
