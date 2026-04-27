/**
 * GazeCaptureMonitor — browser-side webcam capture for the I³ vision
 * fine-tuning showcase.  Mirrors the structure of voice_prosody.js so
 * a security review only needs to look at one pattern twice.
 *
 * ===========================================================================
 *                              PRIVACY CONTRACT
 * ===========================================================================
 *
 * 1. The webcam is OFF BY DEFAULT.  The user must explicitly click the
 *    camera-toggle button to call ``enable()``.
 * 2. The raw video stream is held only in this tab's memory (one
 *    ``<video>`` element + one hidden ``<canvas>`` per instance).  It
 *    is NEVER sent over the network — no MediaRecorder, no WebRTC peer
 *    connection, no fetch with a base64 image body of full-resolution
 *    frames.
 * 3. Every 250 ms we capture one frame onto a hidden 64×48 grayscale
 *    ``<canvas>`` (4 fps, enough for gaze stability).  The original
 *    full-resolution frame is then garbage-collected on the next
 *    capture; nothing keeps it.
 * 4. The 64×48 grayscale fingerprint (3072 bytes — too few for face
 *    biometrics) is what we ship to the server's calibration / infer
 *    endpoints.  The raw video frame NEVER crosses the wire.
 * 5. On send (``getCurrentFeatures()``, called by chat.js right before
 *    the WS message frame is built), we emit only the predicted
 *    gaze label + 5 numeric scalars (confidence, presence, blink rate,
 *    head stability, captured_seconds) plus the per-class softmax
 *    probabilities.  No image data crosses the WS frame.
 * 6. No external library is loaded — vanilla ``getUserMedia`` +
 *    ``<canvas>`` only.  No TensorFlow.js, no MediaPipe, no ONNX-Web.
 *
 * The camera toggle has a clear visual indicator (red dot + animated
 * camera-rec border) whenever video is being captured, in line with
 * WCAG and platform conventions.
 *
 * Architecture — split between browser and server
 * ================================================
 * The browser handles capture + privacy (downsampling, fingerprint
 * extraction, never shipping raw frames).  The SERVER handles
 * inference: a fine-tuned MobileNetV3-small backbone (Howard et al.
 * 2019, ImageNet-pre-trained, frozen) plus a tiny per-user
 * fine-tunable head (576 → 128 → 4).  See
 * ``i3/multimodal/gaze_classifier.py`` for the full design and the
 * citation chain.
 *
 * ===========================================================================
 */

(function () {
    'use strict';

    // ---------------------------------------------------------------------
    // Tuning constants — kept in one block so a reviewer can audit them.
    // ---------------------------------------------------------------------
    const FRAME_INTERVAL_MS = 250;          // 4 fps capture
    const FINGERPRINT_W = 64;
    const FINGERPRINT_H = 48;
    const FINGERPRINT_LEN = FINGERPRINT_W * FINGERPRINT_H;  // 3072
    const HISTORY_SECONDS = 30.0;           // rolling-window for aggregate
    const HISTORY_CAP = Math.ceil(HISTORY_SECONDS * 1000 / FRAME_INTERVAL_MS);
    const PRESENCE_CONF = 0.4;              // off-screen labels need this conf
    const CALIBRATION_FRAMES_PER_TARGET = 8; // 8 frames × 4 targets = 32
    const CALIBRATION_CAPTURE_MS = 2400;     // 2.4 s @ 4 fps = 8-9 frames
    const VIDEO_WIDTH = 320;
    const VIDEO_HEIGHT = 240;

    const GAZE_LABELS = ['at_screen', 'away_left', 'away_right', 'away_other'];

    // ---------------------------------------------------------------------
    // Pure helpers
    // ---------------------------------------------------------------------

    function clamp01(x) {
        if (!Number.isFinite(x)) return 0;
        if (x < 0) return 0;
        if (x > 1) return 1;
        return x;
    }

    function _modeLabel(arr) {
        if (!arr || !arr.length) return 'at_screen';
        const counts = {};
        for (const lbl of arr) counts[lbl] = (counts[lbl] || 0) + 1;
        let best = arr[0];
        let bestN = 0;
        for (const k in counts) {
            if (counts[k] > bestN) { best = k; bestN = counts[k]; }
        }
        return best;
    }

    function _meanProbs(samples) {
        const out = { at_screen: 0, away_left: 0, away_right: 0, away_other: 0 };
        if (!samples.length) return out;
        for (const s of samples) {
            for (const k of GAZE_LABELS) {
                out[k] += (s.label_probs && s.label_probs[k]) || 0;
            }
        }
        for (const k of GAZE_LABELS) out[k] /= samples.length;
        return out;
    }

    // =========================================================================
    // GazeCaptureMonitor
    // =========================================================================

    class GazeCaptureMonitor {
        constructor() {
            this._enabled = false;
            this._stream = null;
            this._video = null;
            this._canvas = null;
            this._ctx = null;
            this._timer = null;
            this._samples = [];     // ring of {t, label, confidence, label_probs, fingerprint}
            this._capacity = HISTORY_CAP;
            this._userId = (window.app && window.app.userId) || 'demo_user';
            this._listeners = new Set();
            // Last fingerprint sent to the server (cached for the
            // calibration flow).
            this._lastFingerprint = null;
            // Frame-to-frame motion proxy (head stability).
            this._lastMeanIntensity = null;
            // Inference backoff: don't fire if a previous infer is in flight.
            this._inferInFlight = false;
        }

        onFrame(fn) {
            if (typeof fn === 'function') this._listeners.add(fn);
            return () => this._listeners.delete(fn);
        }

        async enable() {
            if (this._enabled) return true;
            if (typeof navigator === 'undefined' ||
                !navigator.mediaDevices ||
                typeof navigator.mediaDevices.getUserMedia !== 'function') {
                console.warn('[GazeCapture] getUserMedia unavailable.');
                return false;
            }
            try {
                this._stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: VIDEO_WIDTH },
                        height: { ideal: VIDEO_HEIGHT },
                        facingMode: 'user',
                    },
                    audio: false,
                });
            } catch (err) {
                console.warn('[GazeCapture] camera permission denied:', err);
                return false;
            }

            try {
                this._video = document.createElement('video');
                this._video.autoplay = true;
                this._video.playsInline = true;
                this._video.muted = true;
                this._video.srcObject = this._stream;
                this._video.style.display = 'none';
                document.body.appendChild(this._video);

                this._canvas = document.createElement('canvas');
                this._canvas.width = FINGERPRINT_W;
                this._canvas.height = FINGERPRINT_H;
                this._canvas.style.display = 'none';
                document.body.appendChild(this._canvas);
                this._ctx = this._canvas.getContext('2d', {
                    willReadFrequently: true,
                });

                await new Promise((resolve) => {
                    if (this._video.readyState >= 2) {
                        resolve();
                    } else {
                        this._video.onloadedmetadata = () => resolve();
                        // Defensive timeout so we don't hang forever.
                        setTimeout(resolve, 1500);
                    }
                });

                this._samples = [];
                this._enabled = true;
                this._timer = setInterval(
                    () => this._tick(),
                    FRAME_INTERVAL_MS,
                );
                console.log('[GazeCapture] capture started.');
                return true;
            } catch (err) {
                console.error('[GazeCapture] init failed:', err);
                this.disable();
                return false;
            }
        }

        disable() {
            if (this._timer) {
                clearInterval(this._timer);
                this._timer = null;
            }
            try {
                if (this._stream) {
                    this._stream.getTracks().forEach((t) => t.stop());
                }
            } catch (_e) { /* swallow */ }
            try {
                if (this._video && this._video.parentNode) {
                    this._video.srcObject = null;
                    this._video.parentNode.removeChild(this._video);
                }
            } catch (_e) { /* swallow */ }
            try {
                if (this._canvas && this._canvas.parentNode) {
                    this._canvas.parentNode.removeChild(this._canvas);
                }
            } catch (_e) { /* swallow */ }
            // PRIVACY: discard every cached frame / fingerprint on the
            // way out so a paranoid follow-up cannot read stale image data.
            this._stream = null;
            this._video = null;
            this._canvas = null;
            this._ctx = null;
            this._samples = [];
            this._lastFingerprint = null;
            this._lastMeanIntensity = null;
            this._enabled = false;
        }

        isEnabled() { return this._enabled; }

        // -----------------------------------------------------------------
        // Core capture loop
        // -----------------------------------------------------------------

        /**
         * One 250 ms tick: capture frame → 64×48 grayscale fingerprint →
         * fire-and-forget POST to /api/gaze/infer → append to ring.
         */
        async _tick() {
            if (!this._enabled || !this._video || !this._ctx) return;

            // Draw the current video frame onto the 64×48 canvas.
            // The canvas downsample is what gives us privacy: the
            // resulting fingerprint is too coarse for face biometrics.
            try {
                this._ctx.drawImage(
                    this._video, 0, 0, FINGERPRINT_W, FINGERPRINT_H,
                );
            } catch (_e) {
                // Video not ready yet — try again next tick.
                return;
            }
            const img = this._ctx.getImageData(
                0, 0, FINGERPRINT_W, FINGERPRINT_H,
            );
            // Grayscale: luminance via Rec. 601 weights.  A single Uint8
            // per pixel = 3072 bytes total.
            const gray = new Uint8Array(FINGERPRINT_LEN);
            let sum = 0;
            for (let i = 0; i < FINGERPRINT_LEN; i++) {
                const r = img.data[i * 4];
                const g = img.data[i * 4 + 1];
                const b = img.data[i * 4 + 2];
                const y = (0.299 * r + 0.587 * g + 0.114 * b) | 0;
                gray[i] = y;
                sum += y;
            }
            const meanIntensity = sum / FINGERPRINT_LEN;
            this._lastFingerprint = gray;

            // Send a fire-and-forget inference request.  We don't await
            // it — the next tick will fire regardless.  We do gate
            // against "in-flight already" so we don't pile up requests
            // if the server is slow.
            if (!this._inferInFlight) {
                this._inferInFlight = true;
                this._postInfer(gray)
                    .then((snap) => {
                        if (snap) this._appendSample(snap);
                    })
                    .catch((e) => console.warn('[GazeCapture] infer failed:', e))
                    .finally(() => { this._inferInFlight = false; });
            }

            this._lastMeanIntensity = meanIntensity;
        }

        async _postInfer(gray) {
            try {
                const resp = await fetch('/api/gaze/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: this._userId,
                        // Send as Array (the server accepts list[int])
                        // — Uint8Array.toJSON would be quirky.
                        fingerprint: Array.from(gray),
                    }),
                });
                if (!resp.ok) return null;
                return await resp.json();
            } catch (e) {
                return null;
            }
        }

        _appendSample(snap) {
            const now = performance.now();
            this._samples.push({
                t: now,
                label: snap.label,
                confidence: snap.confidence,
                label_probs: snap.label_probs || {},
                presence: snap.presence,
            });
            while (this._samples.length > this._capacity) {
                this._samples.shift();
            }
            // Notify the live indicator listeners.
            this._listeners.forEach((fn) => {
                try { fn(snap); } catch (_e) { /* swallow */ }
            });
        }

        /**
         * Aggregate the last 30 s of inferences into a single
         * gaze_features object suitable for the WS frame.
         * Returns ``null`` when the camera is off OR no inferences
         * have completed yet.
         */
        getCurrentFeatures() {
            if (!this._enabled || !this._samples.length) return null;
            const samples = this._samples.slice();
            const labels = samples.map((s) => s.label);
            const modeLabel = _modeLabel(labels);
            const probs = _meanProbs(samples);
            const top = samples[samples.length - 1];

            // Presence: at_screen always counts; off-screen labels need
            // confidence ≥ PRESENCE_CONF.
            let presence = false;
            if (modeLabel === 'at_screen') presence = true;
            else if ((top.confidence || 0) >= PRESENCE_CONF) presence = true;

            // Blink rate proxy: fraction of frames where confidence
            // crashed (a likely eyelid-closure artefact).  Capped
            // at the natural blink rate (~0.5 Hz).
            let blinks = 0;
            for (let i = 1; i < samples.length; i++) {
                if (samples[i - 1].confidence > 0.6 &&
                    samples[i].confidence < 0.3) blinks++;
            }
            const captured_seconds = samples.length * FRAME_INTERVAL_MS / 1000;
            const blinkHz = captured_seconds > 0
                ? blinks / captured_seconds : 0;
            const blink_rate_norm = clamp01(blinkHz / 0.5);

            // Head stability proxy: 1 - fraction of label changes between
            // consecutive samples.
            let changes = 0;
            for (let i = 1; i < samples.length; i++) {
                if (samples[i].label !== samples[i - 1].label) changes++;
            }
            const head_stability = samples.length > 1
                ? clamp01(1 - changes / (samples.length - 1)) : 1.0;

            return {
                label: modeLabel,
                confidence: clamp01(probs[modeLabel] || top.confidence || 0),
                label_probs: probs,
                presence: !!presence,
                blink_rate_norm: blink_rate_norm,
                head_stability: head_stability,
                captured_seconds: captured_seconds,
                samples_count: samples.length,
            };
        }

        getLiveDisplay() {
            const last = this._samples.length
                ? this._samples[this._samples.length - 1] : null;
            return {
                enabled: this._enabled,
                label: last ? last.label : 'at_screen',
                confidence: last ? last.confidence : 0,
                presence: last ? !!last.presence : false,
            };
        }

        // -----------------------------------------------------------------
        // Calibration flow
        // -----------------------------------------------------------------

        /**
         * Capture ``CALIBRATION_FRAMES_PER_TARGET`` frames for one label.
         * Resolves with ``frames: Array<Array<int>>``.
         */
        captureCalibrationFor(label, captureMs = CALIBRATION_CAPTURE_MS) {
            if (!this._enabled) return Promise.reject(new Error('camera_off'));
            return new Promise((resolve) => {
                const frames = [];
                const start = performance.now();
                const grab = () => {
                    if (!this._enabled) {
                        resolve(frames);
                        return;
                    }
                    try {
                        this._ctx.drawImage(
                            this._video, 0, 0, FINGERPRINT_W, FINGERPRINT_H,
                        );
                        const img = this._ctx.getImageData(
                            0, 0, FINGERPRINT_W, FINGERPRINT_H,
                        );
                        const gray = new Array(FINGERPRINT_LEN);
                        for (let i = 0; i < FINGERPRINT_LEN; i++) {
                            const r = img.data[i * 4];
                            const g = img.data[i * 4 + 1];
                            const b = img.data[i * 4 + 2];
                            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) | 0;
                        }
                        frames.push(gray);
                    } catch (_e) { /* skip frame */ }
                    if (performance.now() - start < captureMs) {
                        setTimeout(grab, FRAME_INTERVAL_MS);
                    } else {
                        resolve(frames);
                    }
                };
                grab();
            });
        }

        /**
         * POST a populated ``calibration_frames`` dict to
         * ``/api/gaze/calibrate``.  Returns the server's JSON.
         */
        async submitCalibration(framesByLabel) {
            const body = {
                user_id: this._userId,
                calibration_frames: framesByLabel,
                epochs: 60,
                lr: 0.001,
            };
            const resp = await fetch('/api/gaze/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(`calibrate failed: ${resp.status} ${text}`);
            }
            return await resp.json();
        }
    }

    // -----------------------------------------------------------------
    // UI mounting helpers — vanilla DOM, no frameworks.
    // -----------------------------------------------------------------

    function _mountPrivacyAssertion(host) {
        if (host.querySelector('.gaze-capture-privacy')) return;
        const note = document.createElement('div');
        note.className = 'gaze-capture-privacy';
        note.setAttribute('role', 'note');
        note.textContent =
            'Camera frames never leave your device as full images. ' +
            'Only a 64×48 grayscale fingerprint reaches the server\'s ' +
            'gaze classifier; the original frame is discarded.';
        host.appendChild(note);
    }

    function _mountLiveIndicator(host, monitor) {
        if (host.querySelector('.gaze-live-indicator')) return null;
        const ind = document.createElement('div');
        ind.className = 'gaze-live-indicator hidden';
        ind.setAttribute('aria-hidden', 'true');
        ind.innerHTML = '<span class="gaze-icon">👁</span>' +
            '<span class="gaze-label">at-screen</span>' +
            '<span class="gaze-conf">0.00</span>';
        host.appendChild(ind);

        const labelEl = ind.querySelector('.gaze-label');
        const confEl = ind.querySelector('.gaze-conf');
        const iconEl = ind.querySelector('.gaze-icon');
        const ICONS = {
            at_screen: '👁',
            away_left: '←',
            away_right: '→',
            away_other: '↓',
        };
        monitor.onFrame((snap) => {
            if (!snap) return;
            iconEl.textContent = ICONS[snap.label] || '?';
            labelEl.textContent = (snap.label || 'unknown').replace('_', '-');
            confEl.textContent = (snap.confidence || 0).toFixed(2);
        });

        return {
            show() { ind.classList.remove('hidden'); },
            hide() { ind.classList.add('hidden'); },
        };
    }

    function _mountCalibrationUI(host, monitor) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'gaze-calibrate-btn';
        btn.textContent = 'Calibrate gaze';
        btn.title = 'Walk through 4 targets to fine-tune the head';
        btn.style.display = 'none';   // shown when camera is on
        host.appendChild(btn);

        const overlay = document.createElement('div');
        overlay.className = 'gaze-calibration-overlay';
        overlay.style.display = 'none';
        overlay.innerHTML =
            '<div class="gaze-cal-card">' +
            '<div class="gaze-cal-target" id="gaze-cal-target">●</div>' +
            '<div class="gaze-cal-text" id="gaze-cal-text">Look at the dot</div>' +
            '<div class="gaze-cal-progress" id="gaze-cal-progress"></div>' +
            '</div>';
        document.body.appendChild(overlay);

        async function runCalibration() {
            if (!monitor.isEnabled()) return;
            const steps = [
                { lbl: 'at_screen',  text: 'Look at the centre dot',
                  cls: 'centre' },
                { lbl: 'away_left',  text: 'Now look at the left edge of the screen',
                  cls: 'left' },
                { lbl: 'away_right', text: 'Now look at the right edge of the screen',
                  cls: 'right' },
                { lbl: 'away_other', text: 'Now look down (at your phone or lap)',
                  cls: 'down' },
            ];
            const captured = {};
            const dot = overlay.querySelector('#gaze-cal-target');
            const text = overlay.querySelector('#gaze-cal-text');
            const prog = overlay.querySelector('#gaze-cal-progress');
            overlay.style.display = 'flex';
            try {
                for (const step of steps) {
                    text.textContent = step.text;
                    dot.className = 'gaze-cal-target ' + step.cls;
                    prog.textContent = '3...';
                    await new Promise((r) => setTimeout(r, 1000));
                    prog.textContent = '2...';
                    await new Promise((r) => setTimeout(r, 1000));
                    prog.textContent = '1...';
                    await new Promise((r) => setTimeout(r, 1000));
                    prog.textContent = 'capturing';
                    captured[step.lbl] = await monitor.captureCalibrationFor(step.lbl);
                }
                text.textContent = 'Fine-tuning the head...';
                prog.textContent = '';
                const result = await monitor.submitCalibration(captured);
                text.textContent =
                    `Done. val_acc=${(result.val_accuracy || 0).toFixed(2)}, ` +
                    `${result.n_frames_used} frames.`;
                prog.textContent = '✓';
                await new Promise((r) => setTimeout(r, 1500));
            } catch (e) {
                text.textContent = 'Calibration failed: ' + e.message;
                prog.textContent = '';
                await new Promise((r) => setTimeout(r, 2500));
            } finally {
                overlay.style.display = 'none';
            }
        }

        btn.addEventListener('click', runCalibration);

        return {
            show() { btn.style.display = ''; },
            hide() { btn.style.display = 'none'; },
        };
    }

    function _mountToggleButton(sendButton, monitor, onChange) {
        if (!sendButton || !sendButton.parentNode) return null;
        const existing = document.getElementById('camera-toggle-btn');
        if (existing) return existing;

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.id = 'camera-toggle-btn';
        btn.className = 'camera-toggle-btn off';
        btn.setAttribute('aria-label',
            'Toggle gaze capture (camera frames stay on this device)');
        btn.setAttribute('title',
            'Capture gaze (off by default — only a 64×48 grayscale ' +
            'fingerprint crosses the wire)');
        btn.textContent = '📷';

        sendButton.parentNode.insertBefore(btn, sendButton);

        btn.addEventListener('click', async () => {
            if (monitor.isEnabled()) {
                monitor.disable();
                btn.classList.remove('on', 'pulse');
                btn.classList.add('off');
                btn.textContent = '📷';
                document.body.classList.remove('i3-camera-active');
                onChange && onChange(false);
            } else {
                btn.disabled = true;
                const ok = await monitor.enable();
                btn.disabled = false;
                if (ok) {
                    btn.classList.remove('off');
                    btn.classList.add('on', 'pulse');
                    btn.textContent = '🎥';
                    // Apple-clean: privacy text + calibrate button only
                    // appear once the camera is actually capturing, with
                    // the privacy text auto-fading after 5 s.
                    document.body.classList.add('i3-camera-active');
                    document.querySelectorAll('.gaze-capture-privacy')
                        .forEach((el) => {
                            el.classList.add('live-reveal');
                            setTimeout(() => el.classList.remove('live-reveal'), 5500);
                        });
                    document.querySelectorAll('.gaze-live-indicator')
                        .forEach((el) => el.classList.add('is-on'));
                    onChange && onChange(true);
                }
            }
        });
        return btn;
    }

    function mount(opts) {
        const monitor = new GazeCaptureMonitor();
        const host = (opts && opts.host) ? opts.host : document.body;
        const sendButton = (opts && opts.sendButton) ? opts.sendButton : null;
        const liveHandle = _mountLiveIndicator(host, monitor);
        const calHandle = _mountCalibrationUI(host, monitor);
        _mountPrivacyAssertion(host);
        _mountToggleButton(sendButton, monitor, (on) => {
            if (liveHandle) (on ? liveHandle.show() : liveHandle.hide());
            if (calHandle) (on ? calHandle.show() : calHandle.hide());
        });
        return monitor;
    }

    window.I3GazeCapture = {
        GazeCaptureMonitor,
        GAZE_LABELS,
        FINGERPRINT_W,
        FINGERPRINT_H,
        mount,
    };
})();
