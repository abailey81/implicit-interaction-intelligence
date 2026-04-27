/**
 * tts_player.js
 * ─────────────
 * Client-side "Speak response" player for the I³ web demo.
 *
 * Responsibilities
 *   * Renders a "Speak response" button (with aria-label) into
 *     `#i3-advanced-panels` (if present) or directly into the
 *     chat-panel footer as a fallback.
 *   * On click, reads the most recent AI message text from the
 *     chat DOM, POSTs it to `/api/tts`, and plays whatever the
 *     server returns:
 *       - `audio_wav_base64`  → an `<audio>` element.
 *       - `directive`         → `window.speechSynthesis.speak(new
 *                                SpeechSynthesisUtterance(...))`.
 *   * Surfaces the conditioning explanation as an aria-live="polite"
 *     caption below the button.
 *   * Gracefully disables itself when neither a server audio path
 *     nor the Web Speech API is available.
 *
 * This module is pure ES (no build step) and does NOT modify any of
 * `app.js`, `chat.js`, or the other existing web modules.
 */

const CONTAINER_FALLBACK_ID = 'i3-advanced-panels';
const PLAYER_ID = 'i3-tts-player';
const CAPTION_ID = 'i3-tts-caption';
const BUTTON_ID = 'i3-tts-speak-btn';
const AUDIO_ID = 'i3-tts-audio';
const ENDPOINT = '/api/tts';
const USER_ID = 'demo_user';
const MAX_TEXT_CHARS = 2000;

let _initialised = false;

/**
 * Read the latest AI message text from the chat panel.
 * @returns {string} the text (possibly empty).
 */
function getLatestAiMessage() {
    const messages = document.querySelectorAll('.message.ai .message-text');
    if (!messages || messages.length === 0) return '';
    const last = messages[messages.length - 1];
    const txt = (last && last.textContent) ? last.textContent.trim() : '';
    return txt.slice(0, MAX_TEXT_CHARS);
}

/**
 * Detect whether the Web Speech API is usable client-side.
 * @returns {boolean}
 */
function hasWebSpeechApi() {
    return typeof window !== 'undefined'
        && typeof window.speechSynthesis !== 'undefined'
        && typeof window.SpeechSynthesisUtterance !== 'undefined';
}

/**
 * Build the DOM: button + caption + hidden audio element.
 * @param {HTMLElement} mount
 */
function buildUi(mount) {
    const wrap = document.createElement('div');
    wrap.id = PLAYER_ID;
    wrap.className = 'i3-tts-wrap';

    const btn = document.createElement('button');
    btn.id = BUTTON_ID;
    btn.className = 'i3-tts-btn';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Speak the latest response');
    btn.setAttribute('aria-controls', CAPTION_ID);
    btn.innerHTML = '<span class="i3-tts-btn-icon" aria-hidden="true">▶</span>'
        + '<span class="i3-tts-btn-label">Speak response</span>';

    const caption = document.createElement('div');
    caption.id = CAPTION_ID;
    caption.className = 'i3-tts-caption';
    caption.setAttribute('role', 'status');
    caption.setAttribute('aria-live', 'polite');
    caption.textContent = 'Idle.';

    const audio = document.createElement('audio');
    audio.id = AUDIO_ID;
    audio.className = 'i3-tts-audio';
    audio.preload = 'none';
    audio.controls = false;

    wrap.appendChild(btn);
    wrap.appendChild(caption);
    wrap.appendChild(audio);
    mount.appendChild(wrap);

    return { wrap, btn, caption, audio };
}

/**
 * Disable the button with a user-visible reason.
 * @param {HTMLButtonElement} btn
 * @param {HTMLElement} caption
 * @param {string} reason
 */
function disableButton(btn, caption, reason) {
    btn.disabled = true;
    btn.classList.add('is-disabled');
    btn.setAttribute('aria-disabled', 'true');
    caption.textContent = reason;
}

/**
 * Decode a base64 audio payload into an object URL for the <audio> el.
 * @param {string} b64
 * @returns {string} object URL
 */
function base64WavToUrl(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'audio/wav' });
    return URL.createObjectURL(blob);
}

/**
 * Play the server-returned TTS output.
 * @param {HTMLAudioElement} audio
 * @param {HTMLButtonElement} btn
 * @param {HTMLElement} caption
 * @param {object} result
 */
async function playResult(audio, btn, caption, result) {
    btn.classList.add('is-speaking');
    caption.textContent = result.explanation || 'Speaking.';

    if (result.audio_wav_base64) {
        // Server produced WAV bytes — play them directly.  Browsers
        // gate ``audio.play()`` behind a user-gesture promise, but
        // since we're inside the click handler that requested TTS we
        // already have the gesture token.  If playback still fails
        // (e.g. the pyttsx3 WAV stream confuses the browser), fall
        // through to the Web Speech API instead of giving up — the
        // user can hear *something* either way.
        try {
            audio.src = base64WavToUrl(result.audio_wav_base64);
            audio.onended = () => { btn.classList.remove('is-speaking'); };
            audio.onerror = () => {
                console.warn('[tts] <audio> element errored; trying Web Speech API');
                speakViaWebSpeech(result, btn, caption);
            };
            await audio.play();
            return;
        } catch (err) {
            console.warn('[tts] audio.play() rejected:', err);
            // Fall through to the Web Speech path below.
        }
    }

    if (hasWebSpeechApi()) {
        speakViaWebSpeech(result, btn, caption);
        return;
    }

    if (result.directive && hasWebSpeechApi()) {
        speakViaWebSpeech(result, btn, caption);
        return;
    }

    btn.classList.remove('is-speaking');
    caption.textContent = 'No playback path available.';
}

/**
 * Web Speech API fallback path — used both when the server didn't
 * produce a WAV and when the WAV refused to play in this browser.
 * @param {object} result
 * @param {HTMLButtonElement} btn
 * @param {HTMLElement} caption
 */
function speakViaWebSpeech(result, btn, caption) {
    if (!hasWebSpeechApi()) {
        btn.classList.remove('is-speaking');
        caption.textContent = 'No playback path available.';
        return;
    }
    const d = result.directive || {};
    const text = String(
        d.text || result.text || result.spoken_text || result.message || ''
    ).trim() || 'Hello.';
    const utt = new SpeechSynthesisUtterance(text);
    if (typeof d.rate === 'number')   utt.rate   = d.rate;
    if (typeof d.pitch === 'number')  utt.pitch  = d.pitch;
    if (typeof d.volume === 'number') utt.volume = d.volume;
    utt.onend = () => { btn.classList.remove('is-speaking'); };
    utt.onerror = () => {
        btn.classList.remove('is-speaking');
        caption.textContent = 'Speech synthesis failed.';
    };
    try {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utt);
        caption.textContent = 'Speaking…';
    } catch (err) {
        btn.classList.remove('is-speaking');
        caption.textContent = 'Speech synthesis failed.';
    }
}

/**
 * Drive one TTS request end-to-end.
 * @param {HTMLButtonElement} btn
 * @param {HTMLElement} caption
 * @param {HTMLAudioElement} audio
 */
async function onSpeakClicked(btn, caption, audio) {
    if (btn.disabled) return;
    const text = getLatestAiMessage();
    if (!text) {
        caption.textContent = 'No AI message to speak yet.';
        return;
    }
    caption.textContent = 'Requesting synthesis…';
    try {
        const resp = await fetch(ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: USER_ID, text }),
        });
        if (!resp.ok) {
            if (resp.status === 503 && hasWebSpeechApi()) {
                // Server has no backend — degrade to a pure-client Web Speech utter.
                const fake = {
                    directive: { text, rate: 1.0, pitch: 1.0, volume: 1.0 },
                    explanation: 'Server TTS unavailable — using browser voice.',
                };
                await playResult(audio, btn, caption, fake);
                return;
            }
            caption.textContent = 'TTS request failed.';
            return;
        }
        const result = await resp.json();
        await playResult(audio, btn, caption, result);
    } catch (err) {
        caption.textContent = 'TTS request errored.';
    }
}

/**
 * Entry point — invoked once on DOMContentLoaded.
 * Idempotent.
 *
 * The original on-page "Speak response" button is still mounted (so
 * legacy diagnostics that look for #i3-tts-speak-btn keep working) but
 * the parent #i3-advanced-panels host is hidden by CSS in the apple21
 * layout.  The primary entry point is now a per-bubble icon button
 * dispatched as `i3:speak-bubble` — see chat.js _appendBubbleTts.
 */
export function initTtsPlayer() {
    if (_initialised) return;
    _initialised = true;

    let mount = document.getElementById(CONTAINER_FALLBACK_ID);
    if (!mount) {
        const chatPanel = document.querySelector('.chat-panel');
        if (!chatPanel) return;  // no place to live in this page layout
        mount = document.createElement('div');
        mount.className = 'i3-tts-mount';
        chatPanel.appendChild(mount);
    }

    const { btn, caption, audio } = buildUi(mount);

    if (!hasWebSpeechApi()) {
        caption.textContent = 'Browser has no Web Speech API — '
            + 'server audio will be used if available.';
    }

    btn.addEventListener('click', () => {
        onSpeakClicked(btn, caption, audio);
    });

    if (!hasWebSpeechApi() && typeof fetch !== 'function') {
        disableButton(btn, caption, 'TTS unavailable in this browser.');
    }

    // Listen for the per-bubble TTS event dispatched from chat.js.
    // We reuse the same audio element + caption + button-state machinery
    // so the global player stays the single source of truth.
    window.addEventListener('i3:speak-bubble', (ev) => {
        const detail = ev && ev.detail ? ev.detail : {};
        const txt = String(detail.text || '').trim();
        const bubbleBtn = detail.button || null;
        if (!txt) {
            if (bubbleBtn) bubbleBtn.classList.remove('is-loading');
            return;
        }
        speakArbitraryText(txt, btn, caption, audio, bubbleBtn);
    });
}

/**
 * Drive a TTS request for an arbitrary text payload (rather than reading
 * the latest message off the DOM).  Used by the bubble TTS buttons so
 * each bubble plays its own text.
 *
 * @param {string} text
 * @param {HTMLButtonElement} btn  the legacy mount button (used as the
 *     state machine for "is-speaking" pulse).
 * @param {HTMLElement} caption    the legacy caption element.
 * @param {HTMLAudioElement} audio shared audio element.
 * @param {HTMLButtonElement|null} bubbleBtn the per-bubble button to
 *     reset once playback starts.
 */
async function speakArbitraryText(text, btn, caption, audio, bubbleBtn) {
    if (btn.disabled) return;
    caption.textContent = 'Requesting synthesis…';
    if (bubbleBtn) {
        bubbleBtn.classList.add('is-loading');
        bubbleBtn.classList.remove('is-speaking');
    }
    const resetBubble = () => {
        if (!bubbleBtn) return;
        bubbleBtn.classList.remove('is-loading');
        bubbleBtn.classList.remove('is-speaking');
    };
    try {
        const resp = await fetch(ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: USER_ID, text: text.slice(0, MAX_TEXT_CHARS) }),
        });
        if (!resp.ok) {
            if (resp.status === 503 && hasWebSpeechApi()) {
                const fake = {
                    directive: { text, rate: 1.0, pitch: 1.0, volume: 1.0 },
                    explanation: 'Server TTS unavailable — using browser voice.',
                };
                if (bubbleBtn) {
                    bubbleBtn.classList.remove('is-loading');
                    bubbleBtn.classList.add('is-speaking');
                }
                await playResult(audio, btn, caption, fake);
                if (audio) audio.addEventListener('ended', resetBubble, { once: true });
                else resetBubble();
                return;
            }
            caption.textContent = 'TTS request failed.';
            resetBubble();
            return;
        }
        const result = await resp.json();
        if (bubbleBtn) {
            bubbleBtn.classList.remove('is-loading');
            bubbleBtn.classList.add('is-speaking');
        }
        await playResult(audio, btn, caption, result);
        if (audio) audio.addEventListener('ended', resetBubble, { once: true });
        else resetBubble();
    } catch (err) {
        caption.textContent = 'TTS request errored.';
        resetBubble();
    }
}

// Auto-init after DOMContentLoaded.
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTtsPlayer);
    } else {
        initTtsPlayer();
    }
}

export default initTtsPlayer;
