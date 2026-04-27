/**
 * Playground tab — manual override of every pipeline stage.
 *
 * Wires the controls in #tab-playground to the
 * POST /api/playground/whatif endpoint and renders the response
 * (and optional baseline comparison) into the right-hand cards.
 *
 * The endpoint is rate-limited at 100 calls per session by the
 * pipeline; this client maintains a local counter for the chip in the
 * controls column.  No build step, no framework — plain DOM.
 */

(() => {
    "use strict";

    const SLIDER_KEYS = [
        "cognitive_load", "formality", "verbosity", "emotionality",
        "directness", "emotional_tone", "accessibility", "reserved",
    ];

    let userId = null;

    function getUserId() {
        if (userId) return userId;
        // Try the existing app's userId, else fabricate.  window.app
        // surfaces the chat user id in the I3App instance constructed
        // by app.js; falling back to a random pg-* prefix means the
        // Playground tab works even if the chat WS hasn't connected.
        if (window.app && window.app.userId) {
            userId = window.app.userId;
            return userId;
        }
        userId = `pg-${Math.floor(Math.random() * 1e9).toString(36)}`;
        return userId;
    }

    function bindSlider(key) {
        const input = document.getElementById(`pg-adapt-${key}`);
        const output = document.getElementById(`pg-adapt-${key}-out`);
        if (!input || !output) return;
        const update = () => {
            output.textContent = parseFloat(input.value).toFixed(2);
        };
        input.addEventListener("input", update);
        update();
    }

    function readAdaptation() {
        // Convert the 8 sliders into the AdaptationVector dict shape.
        const v = (k) => parseFloat(
            document.getElementById(`pg-adapt-${k}`).value
        );
        return {
            cognitive_load: v("cognitive_load"),
            style_mirror: {
                formality: v("formality"),
                verbosity: v("verbosity"),
                emotionality: v("emotionality"),
                directness: v("directness"),
            },
            emotional_tone: v("emotional_tone"),
            accessibility: v("accessibility"),
        };
    }

    function radioValue(name) {
        const el = document.querySelector(`input[name="${name}"]:checked`);
        return el ? el.value : "auto";
    }

    function buildOverrides() {
        const out = {};
        if (document.getElementById("pg-adapt-enabled").checked) {
            out.adaptation = readAdaptation();
        }
        const bio = radioValue("pg-bio");
        if (bio !== "auto") out.biometric_state = bio;
        const access = radioValue("pg-access");
        if (access === "on") out.accessibility = true;
        else if (access === "off") out.accessibility = false;
        const route = radioValue("pg-route");
        if (route !== "auto") out.route = route;
        const critique = radioValue("pg-critique");
        if (critique === "off") out.critique = false;
        const coref = radioValue("pg-coref");
        if (coref === "off") out.coref = false;
        const safety = radioValue("pg-safety");
        if (safety === "off") out.safety = false;
        return out;
    }

    function renderResult(elText, elMeta, payload) {
        if (!payload) {
            elText.textContent = "(no response)";
            elMeta.textContent = "";
            return;
        }
        elText.textContent = String(payload.text || "");
        const chips = [];
        if (payload.route) chips.push(`route · ${payload.route}`);
        if (payload.response_path) chips.push(`path · ${payload.response_path}`);
        if (typeof payload.latency_ms === "number") {
            chips.push(`${payload.latency_ms.toFixed(0)} ms`);
        }
        if (payload.safety && payload.safety.verdict) {
            chips.push(`safety · ${payload.safety.verdict}`);
        }
        if (payload.accessibility && payload.accessibility.active) {
            chips.push("accessibility · on");
        }
        elMeta.innerHTML = chips
            .map((c) => `<span class="pg-chip">${c}</span>`)
            .join("");
    }

    async function send() {
        const promptEl = document.getElementById("pg-prompt");
        const sendBtn = document.getElementById("pg-send");
        const compareEl = document.getElementById("pg-compare");
        const capEl = document.getElementById("pg-cap");
        const resultText = document.getElementById("pg-result-text");
        const resultMeta = document.getElementById("pg-result-meta");
        const baseText = document.getElementById("pg-baseline-text");
        const baseMeta = document.getElementById("pg-baseline-meta");
        const baseCard = document.getElementById("pg-baseline-card");
        const message = (promptEl.value || "").trim();
        if (!message) {
            promptEl.focus();
            return;
        }
        sendBtn.disabled = true;
        sendBtn.textContent = "Running…";
        resultText.textContent = "Running…";
        resultMeta.textContent = "";
        const compareBaseline = !!compareEl.checked;
        baseCard.style.display = compareBaseline ? "block" : "none";
        if (compareBaseline) {
            baseText.textContent = "Running baseline…";
            baseMeta.textContent = "";
        }
        try {
            const body = {
                user_id: getUserId(),
                message,
                overrides: buildOverrides(),
                compare_baseline: compareBaseline,
            };
            const res = await fetch("/api/playground/whatif", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const txt = await res.text();
                resultText.textContent = `Error ${res.status}: ${txt.slice(0, 200)}`;
                return;
            }
            const data = await res.json();
            renderResult(resultText, resultMeta, data.result);
            if (compareBaseline && data.baseline) {
                renderResult(baseText, baseMeta, data.baseline);
            }
            if (typeof data.calls_remaining === "number") {
                capEl.textContent = `${data.calls_remaining} calls remaining this session.`;
            }
        } catch (err) {
            resultText.textContent = `Network error: ${err.message || err}`;
        } finally {
            sendBtn.disabled = false;
            sendBtn.textContent = "Send";
        }
    }

    function init() {
        SLIDER_KEYS.forEach(bindSlider);
        const sendBtn = document.getElementById("pg-send");
        const promptEl = document.getElementById("pg-prompt");
        if (sendBtn) sendBtn.addEventListener("click", send);
        if (promptEl) {
            promptEl.addEventListener("keydown", (ev) => {
                if (ev.key === "Enter" && !ev.shiftKey) {
                    ev.preventDefault();
                    send();
                }
            });
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
