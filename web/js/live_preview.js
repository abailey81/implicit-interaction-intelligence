/**
 * live_preview.js — Iteration 13 (2026-04-26).
 *
 * Live keystroke→adaptation preview.  Hooks into the chat input box,
 * derives a quick CLIENT-SIDE adaptation estimate from the user's
 * typing rhythm, and pushes the estimate into the Dashboard so the
 * 7 axis gauges visibly track typing as it happens — no chat-send
 * needed.  When the user actually sends, the server's authoritative
 * estimate overwrites the preview via the usual state_update frame.
 *
 * The estimator is intentionally simple (we're emulating, not
 * replicating, the from-scratch TCN encoder + linguistic pipeline):
 *
 *   - cognitive_load:   slow IKI + many backspaces -> high load
 *   - verbosity:        long composition + many tokens -> high
 *   - formality:        avg word length + capitalisation share -> high
 *   - directness:       sentence-final punctuation density -> high
 *   - emotionality:     "!" / "?" / lowercase trailing -> high
 *   - emotional_tone:   smiley-emoji / positive lexicon density
 *   - accessibility:    short avg word length + simple punctuation
 *
 * These are heuristics chosen so the preview "feels" like the live
 * adaptation (so a recruiter typing fast vs. slow vs. formal vs.
 * casual sees the bars reshape in real time).  They are NOT a
 * substitute for the server-side TCN+linguistic pipeline.  When the
 * Send button fires, the server's authoritative estimate overwrites
 * the preview within a few hundred milliseconds.
 *
 * Pure DOM, no framework.  Soft-fails silently if the dashboard
 * isn't mounted yet (e.g. the user hasn't visited the Adaptation tab
 * since page load).
 */

(() => {
    'use strict';

    const INPUT_ID = 'chat-input';
    const DEBOUNCE_MS = 180;       // throttle to avoid layout thrash
    const RECENT_KEYSTROKE_WINDOW_MS = 4000;

    // Positive / casual lexicon — small list, only used for the
    // emotional_tone bias.  Trades coverage for transparency.
    const POSITIVE_WORDS = new Set([
        'good', 'great', 'cool', 'nice', 'love', 'like', 'happy',
        'awesome', 'fun', 'thanks', 'cheers', 'wow',
    ]);
    const CASUAL_WORDS = new Set([
        'yeah', 'yep', 'nope', 'gonna', 'wanna', 'lol', 'haha',
        'btw', 'imo', 'tbh', 'kinda', 'sorta', 'dude', 'bro',
    ]);

    const _keystrokeTimes = [];   // monotonic ms timestamps
    let _backspaceCount = 0;
    let _composingStartedAt = 0;
    let _scheduled = false;

    function _now() { return performance.now(); }

    function _trimWindow() {
        const cutoff = _now() - RECENT_KEYSTROKE_WINDOW_MS;
        while (_keystrokeTimes.length > 0 && _keystrokeTimes[0] < cutoff) {
            _keystrokeTimes.shift();
        }
    }

    function _meanInterKeyMs() {
        _trimWindow();
        if (_keystrokeTimes.length < 2) return null;
        let sum = 0;
        for (let i = 1; i < _keystrokeTimes.length; i++) {
            sum += _keystrokeTimes[i] - _keystrokeTimes[i - 1];
        }
        return sum / (_keystrokeTimes.length - 1);
    }

    function _clamp01(v) {
        if (!Number.isFinite(v)) return 0.5;
        if (v < 0) return 0;
        if (v > 1) return 1;
        return v;
    }

    function _estimate(text) {
        const trimmed = (text || '').trim();
        if (!trimmed) {
            return null;  // nothing to preview
        }

        const words = trimmed.split(/\s+/).filter(Boolean);
        const n = words.length;
        const totalChars = trimmed.length;

        // ---- cognitive_load ----
        // High when the user types slowly OR backspaces a lot.
        const meanIKI = _meanInterKeyMs();
        const ikiTerm = meanIKI !== null
            ? Math.min(1, Math.max(0, (meanIKI - 80) / 320))   // 80ms→0, 400ms→1
            : 0.5;
        const composingMs = _composingStartedAt > 0
            ? _now() - _composingStartedAt
            : 0;
        const composeTerm = Math.min(1, composingMs / 12000);   // 12s -> 1
        const editTerm = totalChars > 0
            ? Math.min(1, _backspaceCount / Math.max(8, totalChars))
            : 0;
        const cognitive_load = _clamp01(
            0.40 * ikiTerm + 0.30 * composeTerm + 0.30 * editTerm
        );

        // ---- verbosity ----
        // High when the message is long.  Saturates around 40 words.
        const verbosity = _clamp01(n / 40);

        // ---- formality ----
        // High when avg word length is large AND capital-share is meaningful.
        const avgWordLen = n > 0
            ? words.reduce((s, w) => s + w.length, 0) / n
            : 0;
        const formalityFromLen = Math.min(1, Math.max(0, (avgWordLen - 3) / 4)); // 3→0, 7→1
        const upperShare = totalChars > 0
            ? (trimmed.match(/[A-Z]/g) || []).length / totalChars
            : 0;
        const formalityFromCaps = Math.min(1, upperShare * 6);   // 1/6 of chars upper -> 1
        const lowerSet = new Set(words.map(w => w.toLowerCase()));
        const casualHits = [...lowerSet].filter(w => CASUAL_WORDS.has(w)).length;
        const casualPenalty = Math.min(0.5, casualHits * 0.15);
        const formality = _clamp01(
            0.55 * formalityFromLen + 0.45 * formalityFromCaps - casualPenalty
        );

        // ---- directness ----
        // Short sentences + period-density -> direct; long meandering -> low.
        const sentenceCount = Math.max(1, (trimmed.match(/[.!?]/g) || []).length);
        const wordsPerSentence = n / sentenceCount;
        const directness = _clamp01(
            1 - Math.min(1, Math.max(0, (wordsPerSentence - 6) / 18))
        );

        // ---- emotionality ----
        // Exclamation + question density.
        const excl = (trimmed.match(/!/g) || []).length;
        const ques = (trimmed.match(/\?/g) || []).length;
        const emotionality = _clamp01(
            Math.min(1, (excl * 0.18) + (ques * 0.10))
        );

        // ---- emotional_tone ----
        // Positive lexicon density + smileys.  0.5 baseline (neutral).
        const positiveHits = [...lowerSet].filter(w => POSITIVE_WORDS.has(w)).length;
        const smileyHits = (trimmed.match(/[:;]-?[)D]|<3/g) || []).length;
        const emotional_tone = _clamp01(
            0.5 + 0.10 * positiveHits + 0.15 * smileyHits
        );

        // ---- accessibility ----
        // High when the text uses short words and simple punctuation (a
        // proxy for "this user might benefit from simpler replies").
        // Currently low by default — surfaces only when the typing
        // pattern strongly suggests motor/cognitive difficulty.
        const punctDensity = totalChars > 0
            ? (trimmed.match(/[,;:()\-—]/g) || []).length / totalChars
            : 0;
        const shortWordShare = n > 0
            ? words.filter(w => w.length <= 3).length / n
            : 0;
        const accessibility = _clamp01(
            Math.max(0, shortWordShare - 0.4) * 1.5
            - Math.min(0.4, punctDensity * 4)
        );

        return {
            cognitive_load,
            formality,
            verbosity,
            emotionality,
            directness,
            emotional_tone,
            accessibility,
        };
    }

    function _push(estimate) {
        if (!estimate) return;
        const dash = window.app && window.app.dashboard;
        if (!dash || typeof dash.update !== 'function') return;
        // Build the same shape the server's state_update frame uses
        // so dashboard.update() doesn't need any branching.
        try {
            dash.update({
                adaptation: estimate,
                // Keep the engagement card untouched — this is a
                // typing preview, not a real per-turn signal.
            });
        } catch (_e) { /* ignore */ }
    }

    function _onKey(ev) {
        const t = _now();
        if (_composingStartedAt === 0) {
            _composingStartedAt = t;
        }
        _keystrokeTimes.push(t);
        if (ev && ev.key === 'Backspace') {
            _backspaceCount++;
        }
        if (_scheduled) return;
        _scheduled = true;
        setTimeout(() => {
            _scheduled = false;
            const inputEl = document.getElementById(INPUT_ID);
            if (!inputEl) return;
            const text = inputEl.value || '';
            if (!text.trim()) {
                // Reset composing window when the input goes empty.
                _composingStartedAt = 0;
                _backspaceCount = 0;
                _keystrokeTimes.length = 0;
                return;
            }
            const est = _estimate(text);
            _push(est);
        }, DEBOUNCE_MS);
    }

    function _onSendOrEnter() {
        // Reset state so the next message gets a clean window.
        _composingStartedAt = 0;
        _backspaceCount = 0;
        _keystrokeTimes.length = 0;
    }

    function init() {
        const inputEl = document.getElementById(INPUT_ID);
        if (!inputEl) return;
        inputEl.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' && !ev.shiftKey) {
                _onSendOrEnter();
                return;
            }
            _onKey(ev);
        });
        inputEl.addEventListener('input', () => _onKey(null));
        const sendBtn = document.getElementById('send-btn');
        if (sendBtn) {
            sendBtn.addEventListener('click', _onSendOrEnter);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
