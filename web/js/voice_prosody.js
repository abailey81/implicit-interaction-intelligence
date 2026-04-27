/**
 * VoiceProsodyMonitor — browser-side prosodic feature extractor for I³.
 *
 * ===========================================================================
 *                              PRIVACY CONTRACT
 * ===========================================================================
 *
 * 1. The microphone is OFF BY DEFAULT.  The user must explicitly click
 *    the mic-toggle button to call ``enable()``.
 * 2. The raw audio buffer is held only in this tab's memory (the
 *    AudioContext + AnalyserNode managed by this class).  It is NEVER
 *    sent over the network — no MediaRecorder, no fetch/XHR, no
 *    WebSocket frame ever carries audio bytes.
 * 3. Every 100 ms we read the current AnalyserNode snapshot, compute the
 *    eight prosodic scalars listed in PROSODY_FEATURE_KEYS, and append
 *    them to a 30-frame ring buffer.  The raw waveform is OVERWRITTEN
 *    on the next read; nothing keeps it.
 * 4. On send (``getCurrentFeatures()``, called by chat.js right before
 *    the WS message frame is built), we emit only the eight aggregate
 *    scalars + two metadata fields (samples_count, captured_seconds).
 * 5. No external library is loaded.  Everything below is vanilla
 *    WebAudio API + plain JS.  No Meyda, no librosa.js, no DSP libs —
 *    so a security review only has to read this file.
 *
 * The mic toggle has a clear visual indicator (red dot + animated
 * waveform) whenever audio is being captured, in line with WCAG and
 * platform conventions.
 *
 * References:
 *   Schuller, B. et al. (2009). The INTERSPEECH 2009 Emotion Challenge.
 *   Eyben, F. et al. (2010). openSMILE: The Munich versatile and fast
 *     open-source audio feature extractor. ACM Multimedia, 1459–1462.
 *
 * The eight features below are a deliberate subset of the openSMILE
 * GeMAPS feature set, chosen so a JS-side WebAudio worklet can compute
 * them in real time at <50 ms latency.
 *
 * ===========================================================================
 */

(function () {
    'use strict';

    // The eight feature keys, in the order documented in
    // i3/multimodal/prosody.py :: PROSODY_FEATURE_KEYS.  The Python
    // validator rejects any payload missing one of these keys.
    const PROSODY_FEATURE_KEYS = [
        'speech_rate_wpm_norm',
        'pitch_mean_norm',
        'pitch_variance_norm',
        'energy_mean_norm',
        'energy_variance_norm',
        'voiced_ratio',
        'pause_density',
        'spectral_centroid_norm',
    ];

    // Tuning constants — kept in one block so a reviewer can audit them.
    const FFT_SIZE = 2048;
    const SMOOTHING_TIME_CONSTANT = 0.4;
    const FRAME_INTERVAL_MS = 100;          // 10 Hz feature extraction
    const RING_SECONDS = 3.0;                // last 3 s of features kept
    const RING_CAPACITY = Math.ceil(RING_SECONDS * 1000 / FRAME_INTERVAL_MS);
    const SILENCE_RMS = 0.012;               // below this is "silence"
    const VOICED_PITCH_MIN_HZ = 70;          // human voice band (low)
    const VOICED_PITCH_MAX_HZ = 450;         // human voice band (high)
    const SPEECH_RATE_MAX_WPM = 300;         // saturation cap for normalisation
    const PITCH_MIN_NORM_HZ = 50;
    const PITCH_MAX_NORM_HZ = 400;
    const PITCH_VARIANCE_MAX_HZ = 80;        // σ saturation cap
    const PAUSE_DENSITY_MAX = 4;             // pauses/s saturation cap

    // -----------------------------------------------------------------
    // Pure helpers (no audio context, no DOM)
    // -----------------------------------------------------------------

    function clamp01(x) {
        if (!Number.isFinite(x)) return 0;
        if (x < 0) return 0;
        if (x > 1) return 1;
        return x;
    }

    function mean(arr) {
        if (!arr.length) return 0;
        let s = 0;
        for (let i = 0; i < arr.length; i++) s += arr[i];
        return s / arr.length;
    }

    function stddev(arr) {
        if (arr.length < 2) return 0;
        const m = mean(arr);
        let s = 0;
        for (let i = 0; i < arr.length; i++) {
            const d = arr[i] - m;
            s += d * d;
        }
        return Math.sqrt(s / arr.length);
    }

    /**
     * Estimate fundamental frequency (f0) via simplified autocorrelation.
     *
     * Good enough for prosody (pitch register and σ); not a phoneme
     * transcriber.  We search lag window [sampleRate / fmax,
     * sampleRate / fmin] and return the lag with the highest
     * normalised autocorrelation; the f0 is sampleRate / lag.
     *
     * Returns 0 when no clear pitch is found (silence / unvoiced /
     * noise) so callers can use 0 as the "unvoiced" sentinel.
     */
    function estimatePitchHz(timeDomain, sampleRate) {
        if (!timeDomain || timeDomain.length < 256) return 0;
        const minLag = Math.floor(sampleRate / VOICED_PITCH_MAX_HZ);
        const maxLag = Math.floor(sampleRate / VOICED_PITCH_MIN_HZ);
        if (maxLag >= timeDomain.length) return 0;

        // Energy of the signal (denominator of normalised autocorr).
        let energy0 = 0;
        for (let i = 0; i < timeDomain.length - maxLag; i++) {
            energy0 += timeDomain[i] * timeDomain[i];
        }
        if (energy0 < 1e-6) return 0;          // silence

        let bestLag = 0;
        let bestCorr = 0;
        for (let lag = minLag; lag <= maxLag; lag++) {
            let corr = 0;
            // Step by 2 to keep this O(N^2/2) instead of O(N^2).
            // For FFT_SIZE = 2048 and lag range ≈ 100..630, that's
            // ~330k mults per call — well under 1 ms on any modern CPU.
            const limit = timeDomain.length - lag;
            for (let i = 0; i < limit; i += 2) {
                corr += timeDomain[i] * timeDomain[i + lag];
            }
            if (corr > bestCorr) {
                bestCorr = corr;
                bestLag = lag;
            }
        }
        // Normalise; reject low-confidence pitches (corr ratio < 0.3
        // typically means noise / unvoiced).
        const corrRatio = bestCorr / (energy0 / 2);
        if (corrRatio < 0.3 || bestLag === 0) return 0;
        return sampleRate / bestLag;
    }

    /**
     * Compute the spectral centroid (brightness) from a magnitude
     * spectrum.  Returns the weighted mean bin frequency in Hz.
     */
    function spectralCentroidHz(freqDomain, sampleRate, fftSize) {
        const nyquist = sampleRate / 2;
        const binHz = nyquist / freqDomain.length;
        let weighted = 0;
        let total = 0;
        for (let i = 0; i < freqDomain.length; i++) {
            // freqDomain values are 0..255 (Uint8) or dB (Float32);
            // we standardise on a non-negative magnitude.
            const mag = freqDomain[i] / 255.0;
            weighted += mag * (i * binHz);
            total += mag;
        }
        if (total < 1e-6) return 0;
        return weighted / total;
    }

    /**
     * Estimate the RMS energy of a time-domain frame (Uint8 samples
     * centred at 128).
     */
    function rmsEnergy(timeDomain) {
        if (!timeDomain || !timeDomain.length) return 0;
        let s = 0;
        for (let i = 0; i < timeDomain.length; i++) {
            const v = (timeDomain[i] - 128) / 128.0;     // [-1, 1]
            s += v * v;
        }
        return Math.sqrt(s / timeDomain.length);
    }

    // =================================================================
    // VoiceProsodyMonitor
    // =================================================================

    class VoiceProsodyMonitor {
        constructor() {
            this._enabled = false;
            this._stream = null;
            this._audioCtx = null;
            this._analyser = null;
            this._sourceNode = null;
            this._timeBuffer = null;       // Uint8Array, time-domain samples
            this._freqBuffer = null;       // Uint8Array, frequency-domain bins
            this._timer = null;
            this._frames = [];             // ring buffer of per-frame feature dicts
            this._capacity = RING_CAPACITY;
            // Latest live values for the prosody bar (not the aggregate
            // sent to the server).  Updated every frame.
            this._live = {
                pitch_now_hz: 0,
                energy_now: 0,
                voiced_now: false,
            };
            this._listeners = new Set();
        }

        /**
         * Add a listener that fires on every 100 ms feature update.
         * Used by the live prosody bar.
         */
        onFrame(fn) {
            if (typeof fn === 'function') this._listeners.add(fn);
            return () => this._listeners.delete(fn);
        }

        /**
         * Request mic permission and start capturing.  Returns ``true``
         * on success, ``false`` if the user denied permission OR no
         * AudioContext / getUserMedia is available (covered browser
         * fallbacks).
         */
        async enable() {
            if (this._enabled) return true;
            if (typeof navigator === 'undefined' ||
                !navigator.mediaDevices ||
                typeof navigator.mediaDevices.getUserMedia !== 'function') {
                console.warn('[VoiceProsody] getUserMedia unavailable.');
                return false;
            }
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (typeof Ctx !== 'function') {
                console.warn('[VoiceProsody] AudioContext unavailable.');
                return false;
            }

            try {
                this._stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: false,
                    },
                });
            } catch (err) {
                console.warn('[VoiceProsody] mic permission denied:', err);
                return false;
            }

            try {
                this._audioCtx = new Ctx();
                this._analyser = this._audioCtx.createAnalyser();
                this._analyser.fftSize = FFT_SIZE;
                this._analyser.smoothingTimeConstant = SMOOTHING_TIME_CONSTANT;
                this._sourceNode =
                    this._audioCtx.createMediaStreamSource(this._stream);
                this._sourceNode.connect(this._analyser);
                // Note: we deliberately do NOT connect the analyser to
                // the destination — we only want to inspect samples,
                // not play them back.

                this._timeBuffer = new Uint8Array(this._analyser.fftSize);
                this._freqBuffer = new Uint8Array(this._analyser.frequencyBinCount);
                this._frames = [];
                this._enabled = true;

                this._timer = setInterval(
                    () => this._readFrame(),
                    FRAME_INTERVAL_MS,
                );
                console.log('[VoiceProsody] capture started.');
                return true;
            } catch (err) {
                console.error('[VoiceProsody] init failed:', err);
                this.disable();
                return false;
            }
        }

        /** Tear down the audio graph and release the mic. */
        disable() {
            if (this._timer) {
                clearInterval(this._timer);
                this._timer = null;
            }
            try {
                if (this._sourceNode) this._sourceNode.disconnect();
            } catch (_e) { /* swallow */ }
            try {
                if (this._analyser) this._analyser.disconnect();
            } catch (_e) { /* swallow */ }
            try {
                if (this._audioCtx) {
                    // ``close()`` returns a Promise; we don't await it
                    // because the caller already considers us torn down.
                    this._audioCtx.close().catch(() => { });
                }
            } catch (_e) { /* swallow */ }
            try {
                if (this._stream) {
                    this._stream.getTracks().forEach((t) => t.stop());
                }
            } catch (_e) { /* swallow */ }
            // PRIVACY: discard every audio buffer and feature snapshot
            // on the way out so a paranoid follow-up cannot read stale
            // mic data.
            this._stream = null;
            this._audioCtx = null;
            this._analyser = null;
            this._sourceNode = null;
            this._timeBuffer = null;
            this._freqBuffer = null;
            this._frames = [];
            this._live = { pitch_now_hz: 0, energy_now: 0, voiced_now: false };
            this._enabled = false;
        }

        isEnabled() { return this._enabled; }

        /**
         * One 100 ms read: sample the analyser, compute per-frame
         * scalars, append to the ring buffer.
         */
        _readFrame() {
            if (!this._enabled || !this._analyser) return;

            this._analyser.getByteTimeDomainData(this._timeBuffer);
            this._analyser.getByteFrequencyData(this._freqBuffer);

            const energy = rmsEnergy(this._timeBuffer);

            // Convert Uint8 [0,255] back to Float32 [-1,1] for the
            // pitch autocorrelation — that needs centred amplitudes.
            // We allocate in-place for speed; the array is short-lived.
            const sampleRate = this._audioCtx.sampleRate;
            const tdFloat = new Float32Array(this._timeBuffer.length);
            for (let i = 0; i < this._timeBuffer.length; i++) {
                tdFloat[i] = (this._timeBuffer[i] - 128) / 128.0;
            }

            const isLoudEnough = energy > SILENCE_RMS;
            let pitchHz = 0;
            if (isLoudEnough) {
                pitchHz = estimatePitchHz(tdFloat, sampleRate);
            }
            const isVoiced = isLoudEnough && pitchHz > 0;

            const centroidHz = spectralCentroidHz(
                this._freqBuffer, sampleRate, FFT_SIZE,
            );

            const frame = {
                t: performance.now(),
                voiced: isVoiced,
                pitch_hz: pitchHz,
                energy: energy,
                centroid_hz: centroidHz,
            };
            this._frames.push(frame);
            // Drop oldest beyond the 3 s window.
            if (this._frames.length > this._capacity) {
                this._frames.splice(0, this._frames.length - this._capacity);
            }

            this._live.pitch_now_hz = pitchHz;
            this._live.energy_now = energy;
            this._live.voiced_now = isVoiced;

            // Notify live listeners (the prosody bar).
            this._listeners.forEach((fn) => {
                try { fn(this._live); } catch (_e) { /* swallow */ }
            });
        }

        /**
         * Aggregate the buffered frames into the eight scalars expected
         * by the server-side validator.  Returns ``null`` when the mic
         * is off OR when no frames have been collected (so the caller
         * can omit the ``prosody_features`` field on the WS frame
         * rather than send zeros).
         */
        getCurrentFeatures() {
            if (!this._enabled || !this._frames.length) return null;

            const frames = this._frames.slice();
            const voicedFrames = frames.filter((f) => f.voiced);
            const totalFrames = frames.length;
            const voicedCount = voicedFrames.length;

            // Speech-rate estimator: count voiced→unvoiced transitions
            // (a rough proxy for syllable boundaries) and assume ~1.4
            // syllables per word in conversational English.  This is
            // intentionally crude — phoneme transcription is out of
            // scope.  The server normalises the cap at 300 wpm anyway.
            let voicedRuns = 0;
            for (let i = 1; i < frames.length; i++) {
                if (frames[i].voiced && !frames[i - 1].voiced) voicedRuns++;
            }
            const captured_seconds =
                (totalFrames * FRAME_INTERVAL_MS) / 1000.0;
            const wpm = captured_seconds > 0
                ? (voicedRuns / 1.4) * (60.0 / captured_seconds)
                : 0;

            // Pitch stats — only over voiced frames.
            const pitches = voicedFrames
                .map((f) => f.pitch_hz)
                .filter((p) => p >= VOICED_PITCH_MIN_HZ &&
                              p <= VOICED_PITCH_MAX_HZ);
            const pitchMeanHz = pitches.length ? mean(pitches) : 0;
            const pitchStdHz = pitches.length ? stddev(pitches) : 0;

            // Energy stats — over all frames so silence pulls the mean
            // down (a fair proxy for "engagement").
            const energies = frames.map((f) => f.energy);
            const energyMean = mean(energies);
            const energyStd = stddev(energies);

            // Pause density: count silent runs (≥150 ms each).
            let pauseCount = 0;
            let silentLen = 0;
            for (let i = 0; i < frames.length; i++) {
                if (!frames[i].voiced) {
                    silentLen++;
                } else {
                    if (silentLen * FRAME_INTERVAL_MS >= 150) pauseCount++;
                    silentLen = 0;
                }
            }
            // Trailing silence run.
            if (silentLen * FRAME_INTERVAL_MS >= 150) pauseCount++;
            const pauseDensity = captured_seconds > 0
                ? pauseCount / captured_seconds : 0;

            const centroids = frames.map((f) => f.centroid_hz);
            const centroidMeanHz = mean(centroids);

            // Normalise everything to [0, 1] for the server contract.
            const speech_rate_wpm_norm = clamp01(wpm / SPEECH_RATE_MAX_WPM);
            const pitch_mean_norm = clamp01(
                (pitchMeanHz - PITCH_MIN_NORM_HZ) /
                (PITCH_MAX_NORM_HZ - PITCH_MIN_NORM_HZ),
            );
            const pitch_variance_norm = clamp01(
                pitchStdHz / PITCH_VARIANCE_MAX_HZ,
            );
            const energy_mean_norm = clamp01(energyMean / 0.5);
            const energy_variance_norm = clamp01(energyStd / 0.25);
            const voiced_ratio = clamp01(
                totalFrames > 0 ? voicedCount / totalFrames : 0,
            );
            const pause_density = clamp01(pauseDensity / PAUSE_DENSITY_MAX);
            // Spectral centroid: cap at half the analysis nyquist (12 kHz)
            // for the [0,1] scale; voice usually sits 500 Hz – 4 kHz.
            const spectral_centroid_norm = clamp01(centroidMeanHz / 6000);

            return {
                speech_rate_wpm_norm,
                pitch_mean_norm,
                pitch_variance_norm,
                energy_mean_norm,
                energy_variance_norm,
                voiced_ratio,
                pause_density,
                spectral_centroid_norm,
                samples_count: totalFrames,
                captured_seconds: captured_seconds,
            };
        }

        /**
         * The live snapshot for the prosody bar.  Cheaper than
         * ``getCurrentFeatures`` because it doesn't aggregate.
         */
        getLiveDisplay() {
            return {
                pitch_now_hz: this._live.pitch_now_hz,
                energy_now: this._live.energy_now,
                voiced_now: this._live.voiced_now,
                enabled: this._enabled,
            };
        }
    }

    // -----------------------------------------------------------------
    // UI mounting helpers — built so app.js can opt in by calling
    // ``window.I3VoiceProsody.mount(...)`` once after the chat input
    // is in the DOM.  Vanilla DOM, no frameworks.
    // -----------------------------------------------------------------

    function _mountPrivacyAssertion(host) {
        if (host.querySelector('.voice-prosody-privacy')) return;
        const note = document.createElement('div');
        note.className = 'voice-prosody-privacy';
        note.setAttribute('role', 'note');
        note.textContent =
            'Audio never leaves your device. Only 8 numeric features ' +
            '(pace, pitch variance, energy) are sent.';
        host.appendChild(note);
    }

    function _mountLiveBar(host, monitor) {
        if (host.querySelector('.voice-prosody-bar')) return null;
        const bar = document.createElement('div');
        bar.className = 'voice-prosody-bar hidden';
        bar.setAttribute('aria-hidden', 'true');

        const pitchWrap = document.createElement('div');
        pitchWrap.className = 'voice-prosody-meter pitch';
        const pitchFill = document.createElement('div');
        pitchFill.className = 'voice-prosody-meter-fill';
        pitchWrap.appendChild(pitchFill);
        const pitchLabel = document.createElement('span');
        pitchLabel.className = 'voice-prosody-meter-label';
        pitchLabel.textContent = 'pitch';
        pitchWrap.appendChild(pitchLabel);

        const energyWrap = document.createElement('div');
        energyWrap.className = 'voice-prosody-meter energy';
        const energyFill = document.createElement('div');
        energyFill.className = 'voice-prosody-meter-fill';
        energyWrap.appendChild(energyFill);
        const energyLabel = document.createElement('span');
        energyLabel.className = 'voice-prosody-meter-label';
        energyLabel.textContent = 'energy';
        energyWrap.appendChild(energyLabel);

        bar.appendChild(pitchWrap);
        bar.appendChild(energyWrap);
        host.appendChild(bar);

        monitor.onFrame((live) => {
            const pitchPct = Math.max(0, Math.min(100,
                ((live.pitch_now_hz - 50) / 350) * 100));
            const energyPct = Math.max(0, Math.min(100,
                (live.energy_now / 0.4) * 100));
            pitchFill.style.width = pitchPct.toFixed(0) + '%';
            energyFill.style.width = energyPct.toFixed(0) + '%';
            bar.classList.toggle('voiced', live.voiced_now);
        });

        return {
            show() { bar.classList.remove('hidden'); },
            hide() { bar.classList.add('hidden'); },
        };
    }

    function _mountToggleButton(sendButton, monitor, onChange) {
        if (!sendButton || !sendButton.parentNode) return null;
        // Don't double-mount if a previous app.js init already added one.
        const existing = document.getElementById('mic-toggle-btn');
        if (existing) return existing;

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.id = 'mic-toggle-btn';
        btn.className = 'mic-toggle-btn off';
        btn.setAttribute('aria-label',
            'Toggle voice prosody capture (audio stays on this device)');
        btn.setAttribute('title',
            'Capture voice prosody (off by default — audio stays on device)');
        btn.textContent = '🎙';   // 🎙

        sendButton.parentNode.insertBefore(btn, sendButton);

        btn.addEventListener('click', async () => {
            if (monitor.isEnabled()) {
                monitor.disable();
                btn.classList.remove('on', 'pulse');
                btn.classList.add('off');
                btn.textContent = '🎙';
                document.body.classList.remove('i3-mic-active');
                onChange && onChange(false);
            } else {
                btn.disabled = true;
                const ok = await monitor.enable();
                btn.disabled = false;
                if (ok) {
                    btn.classList.remove('off');
                    btn.classList.add('on', 'pulse');
                    btn.textContent = '🔴';   // 🔴
                    // Apple-clean rule: privacy text is hidden until
                    // the toggle is ON, then shown for 5 s as a fade.
                    document.body.classList.add('i3-mic-active');
                    document.querySelectorAll('.voice-prosody-privacy')
                        .forEach((el) => {
                            el.classList.add('live-reveal');
                            // Remove the trigger after the animation so
                            // a re-toggle fires it fresh.
                            setTimeout(() => el.classList.remove('live-reveal'), 5500);
                        });
                    document.querySelectorAll('.voice-prosody-bar')
                        .forEach((el) => el.classList.add('is-on'));
                    onChange && onChange(true);
                }
            }
        });
        return btn;
    }

    /**
     * Public mount helper: wires the toggle button + live bar +
     * privacy note into the host element.  Idempotent.
     *
     * @param {object} opts
     * @param {HTMLElement} opts.host  Container to mount into
     *                                 (typically ``.chat-input-wrap`` or
     *                                 ``.hero-input``).
     * @param {HTMLElement} opts.sendButton  The existing send button —
     *     the toggle is inserted before it so it visually sits next to send.
     * @returns {VoiceProsodyMonitor}
     */
    function mount(opts) {
        const monitor = new VoiceProsodyMonitor();
        const host = opts && opts.host ? opts.host : document.body;
        const sendButton = opts && opts.sendButton ? opts.sendButton : null;
        const liveHandle = _mountLiveBar(host, monitor);
        _mountPrivacyAssertion(host);
        _mountToggleButton(sendButton, monitor, (on) => {
            if (!liveHandle) return;
            if (on) liveHandle.show(); else liveHandle.hide();
        });
        return monitor;
    }

    // Expose a stable global so app.js can call ``window.I3VoiceProsody.mount``.
    window.I3VoiceProsody = {
        VoiceProsodyMonitor,
        PROSODY_FEATURE_KEYS,
        mount,
    };
})();
