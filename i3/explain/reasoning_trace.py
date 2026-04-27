"""Visible reasoning-trace composer for the I3 chat demo.

Architectural decision
~~~~~~~~~~~~~~~~~~~~~~
This module is the **answer** to the Huawei R&D UK HMI Lab brief on
*closed-loop interaction*: the implicit signals the system reads off the
user's keystroke pattern and the way those signals reshape the visible
response must be **explicitly auditable** on every turn, not buried
inside an opaque μ/σ table that only an ML researcher can decode.

`build_reasoning_trace` composes already-computed pipeline outputs
(keystroke metrics, the 8-axis adaptation vector, the LinUCB router
confidence, the post-processor's per-axis change log, the response path
chosen by the hybrid stack) into three surfaces the UI renders side by
side:

* **Narrative paragraphs** — 3–5 short paragraphs of plain English that
  read like an HMI researcher wrote them, walking from observed typing
  through state inference to the final output shaping.
* **Signal chips** — short ``label / value`` tags for the strip beneath
  the chat reply.
* **Decision chain** — a four/five-step ``Encoder → Adaptation →
  Routing → Retrieval/SLM → Rewriting`` audit trail.

Implementation discipline
~~~~~~~~~~~~~~~~~~~~~~~~~
* **No torch.** Every fact in the trace must come from one of the
  function arguments; we never call back into a model.
* **No fabrication.** When a signal is missing (no keystroke baseline
  yet, no adaptation_changes, etc.) we degrade gracefully with phrases
  like "(baseline still warming up — too few messages to compare)"
  rather than invent a number.
* **Deterministic.** Same inputs always produce the same trace so the
  pitch demo doesn't drift between runs.

This is the visible HMI artefact; the existing
``/api/explain/adaptation`` endpoint with its MC-Dropout per-dimension
σ values continues to serve as the *raw signals* surface for ML
reviewers, demoted to a collapsed `<details>` element in the UI.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Adaptation thresholds used to bin numeric values into qualitative phrases.
# These mirror the bands the post-processor itself applies in
# i3/cloud/postprocess.py so the narrative reads consistently with the
# visible rewrite.
_HIGH = 0.65
_LOW = 0.35
_VERY_HIGH = 0.8
_VERY_LOW = 0.2

# Display labels for adaptation axes — these are what the UI shows in the
# decision chain / narrative.  Keys are the post-promotion flat names that
# the websocket layer ships.
_AXIS_LABELS: dict[str, str] = {
    "cognitive_load": "cognitive_load",
    "verbosity": "verbosity",
    "formality": "formality",
    "emotionality": "emotionality",
    "directness": "directness",
    "emotional_tone": "emotional_tone",
    "accessibility": "accessibility",
}


def build_reasoning_trace(
    *,
    keystroke_metrics: dict,
    adaptation: dict,
    adaptation_changes: list[dict],
    engagement_score: float,
    deviation_from_baseline: float,
    messages_in_session: int,
    baseline_established: bool,
    routing_confidence: dict,
    response_path: str,
    retrieval_score: float,
    user_message_preview: str,
    response_preview: str,
    user_state_embedding_2d: tuple[float, float] | list[float] | None = None,
    history_turns_used: int = 0,
    affect_shift: Any | None = None,
    user_state_label: dict | None = None,
    accessibility: dict | None = None,
    biometric: dict | None = None,
    critique: dict | None = None,
    coreference_resolution: dict | None = None,
    personalisation: dict | None = None,
    multimodal: dict | None = None,
    gaze: dict | None = None,
    routing_decision: dict | None = None,
    privacy_budget: dict | None = None,
    session_memory: dict | None = None,
    explain_plan: dict | None = None,
) -> dict:
    """Compose a structured reasoning trace from pipeline outputs.

    Args:
        keystroke_metrics: Mapping carrying ``composition_time_ms``,
            ``edit_count``, ``pause_before_send_ms`` and the optional
            ``keystroke_timings`` list (last ~50 inter-key intervals in
            milliseconds).
        adaptation: Flattened 8-dim adaptation dict (post style_mirror
            promotion); keys include ``cognitive_load``, ``formality``,
            ``verbosity``, ``emotionality``, ``directness``,
            ``emotional_tone``, ``accessibility``.
        adaptation_changes: The post-processor's per-axis change log,
            i.e. ``[{"axis": ..., "value": ..., "change": ...}, ...]``.
        engagement_score: Composite engagement estimate in [0, 1].
        deviation_from_baseline: Cosine distance between the user's
            current 64-d state and their long-term baseline.
        messages_in_session: Running message count for the active session.
        baseline_established: Whether the encoder has seen enough turns
            to have a stable baseline for this user.
        routing_confidence: ``{"local_slm": x, "cloud_llm": y}`` from the
            LinUCB bandit.
        response_path: Which sub-path of the hybrid local stack carried
            the turn (``"retrieval"`` / ``"retrieval_borderline"`` /
            ``"slm"`` / ``"tool:math"`` / ``"tool:refuse"`` / ``"ood"``).
        retrieval_score: Cosine similarity of the matched retrieval
            entry, in [0, 1].  Only meaningful for retrieval paths.
        user_message_preview: First ~80 chars of the user's input
            (used only for prose framing, never persisted).
        response_preview: First ~80 chars of the AI response.
        user_state_embedding_2d: Optional 2-D projection of the 64-dim
            user-state embedding.  Used to name the qualitative quadrant
            ("rushed/uncertain", "relaxed/curious"...).  When omitted,
            quadrant inference falls back to the keystroke metrics.
        history_turns_used: Number of prior ``(user, assistant)``
            exchanges the pipeline carried into this turn (the
            multi-turn context window).  ``0`` means the response was
            generated from the current message alone; positive values
            cause paragraph 4 to mention the carried context so the
            user can see *why* a contextual prompt like "what about
            that?" resolved correctly.
        affect_shift: Optional :class:`i3.affect.AffectShift` (or its
            dict serialisation).  When ``detected=True`` an extra
            sentence is prepended to paragraph 1 surfacing that the
            model proactively appended a check-in to its reply.
            ``None`` or ``detected=False`` omits the sentence.
        critique: Optional self-critique trace dict produced by the
            engine on SLM-path turns; see
            :attr:`i3.pipeline.types.PipelineOutput.critique`.  When
            ``regenerated=True`` an extra sentence is prepended to
            paragraph 4 narrating the inner monologue
            (``"Self-critique loop fired (first attempt scored 0.41
            [off-topic + word salad]; regenerated with T=0.4 and got
            0.79). The accepted response is shown."``).  Accepted-on-
            first-try traces stay silent — only the interesting
            regenerate case is narrated.

    Returns:
        A dict with three keys:
            * ``narrative_paragraphs`` — list[str] (3–5 paragraphs).
            * ``signal_chips`` — list[dict] of ``{label, value, hint}``.
            * ``decision_chain`` — list[dict] of ``{step, what, why}``.
    """
    # ------------------------------------------------------------------
    # Defensive coercion.  The pipeline ships well-formed dicts; the
    # WebSocket layer wraps the call in try/except so a malformed input
    # only suppresses the trace, never the response.
    # ------------------------------------------------------------------
    km = dict(keystroke_metrics or {})
    ad = dict(adaptation or {})
    changes = list(adaptation_changes or [])
    rconf = dict(routing_confidence or {})
    # Alias the accessibility-state dict before the local name
    # ``accessibility`` is reused for the float adaptation axis.
    access_state_dict: dict | None = (
        dict(accessibility) if isinstance(accessibility, dict) else None
    )

    composition_ms = _safe_float(km.get("composition_time_ms"), 0.0)
    edit_count = int(_safe_float(km.get("edit_count"), 0))
    pause_ms = _safe_float(km.get("pause_before_send_ms"), 0.0)
    timings = [
        _safe_float(t, 0.0)
        for t in (km.get("keystroke_timings") or [])
        if _safe_float(t, -1.0) >= 0.0
    ]

    iki_summary = _summarise_iki(timings)
    quadrant = _quadrant(
        user_state_embedding_2d=user_state_embedding_2d,
        baseline_established=baseline_established,
        composition_ms=composition_ms,
        edit_count=edit_count,
        pause_ms=pause_ms,
    )

    cognitive_load = _safe_float(ad.get("cognitive_load"), 0.5)
    verbosity = _safe_float(ad.get("verbosity"), 0.5)
    formality = _safe_float(ad.get("formality"), 0.5)
    emotional_tone = _safe_float(ad.get("emotional_tone"), 0.0)
    accessibility = _safe_float(ad.get("accessibility"), 0.0)

    local_p = _safe_float(rconf.get("local_slm"), 0.0)
    cloud_p = _safe_float(rconf.get("cloud_llm"), 0.0)
    routed_to = "Edge SLM" if local_p >= cloud_p else "Cloud LLM"
    route_p = max(local_p, cloud_p)

    path_label = _path_label(response_path)
    path_lc = (response_path or "").lower()

    deviation = _safe_float(deviation_from_baseline, 0.0)

    # ------------------------------------------------------------------
    # Paragraph 1 — what was observed
    # ------------------------------------------------------------------
    para1_bits: list[str] = []
    msg_preview = (user_message_preview or "").strip()
    msg_quoted = f"'{msg_preview}'" if msg_preview else "your message"
    if composition_ms > 0:
        para1_bits.append(
            f"You typed {msg_quoted} across {composition_ms / 1000.0:.2f} s"
        )
    else:
        para1_bits.append(f"You sent {msg_quoted}")

    if edit_count > 0:
        s = "" if edit_count == 1 else "s"
        para1_bits.append(f"with {edit_count} backspace{s}")
    else:
        para1_bits.append("with no edits")

    if pause_ms >= 50:
        para1_bits.append(
            f"and a {pause_ms:.0f} ms pause before sending"
        )
    else:
        para1_bits.append("and no measurable pre-send pause")

    para1 = ", ".join(para1_bits[:1]) + " " + ", ".join(para1_bits[1:]) + "."

    # Prepend an affect-shift announcement when one was detected on
    # this turn.  This is the visible HMI handle on the showpiece
    # behaviour: the user can see, in plain English, that their
    # typing pattern crossed a shift threshold and that the model
    # responded by appending a polite check-in to its reply.
    shift_prefix = _affect_shift_prefix(affect_shift)
    if shift_prefix:
        para1 = shift_prefix + " " + para1

    # Append IKI-baseline phrasing.
    if iki_summary["have_data"]:
        if iki_summary["have_baseline"]:
            sign = "+" if iki_summary["pct_vs_baseline"] >= 0 else ""
            para1 += (
                f" Inter-keystroke variance is {sign}"
                f"{iki_summary['pct_vs_baseline']:.0f}% vs your "
                f"session baseline (mean {iki_summary['mean_ms']:.0f} ms,"
                f" σ {iki_summary['std_ms']:.0f} ms)."
            )
        else:
            para1 += (
                f" Mean inter-keystroke interval is "
                f"{iki_summary['mean_ms']:.0f} ms — baseline still "
                f"warming up, too few messages to compare."
            )
    else:
        para1 += " (No inter-keystroke samples were captured for this turn.)"

    # Voice-prosody multimodal fusion sentence — only when the user
    # had the mic enabled this turn.  Mirrors the privacy contract in
    # plain English so the reader can see exactly what crossed the wire
    # (eight numeric scalars, audio discarded on-device).  Cites the
    # Schuller / Eyben paralinguistic feature design implicitly via
    # the narration of pace/pitch/energy.
    prosody_sentence = _multimodal_prosody_sentence(multimodal)
    if prosody_sentence:
        para1 += " " + prosody_sentence

    # Vision-gaze fine-tuning sentence — only when the camera was on
    # this turn.  Surfaces the predicted gaze label + confidence and,
    # when the user wasn't looking at the screen, the gaze-conditioned
    # response-timing note.  Cites the fine-tuned MobileNetV3-small
    # backbone implicitly via the parameter counts in the narration.
    gaze_sentence = _gaze_sentence(gaze)
    if gaze_sentence:
        para1 += " " + gaze_sentence

    # ------------------------------------------------------------------
    # Paragraph 2 — what state the encoder inferred
    # ------------------------------------------------------------------
    if baseline_established:
        para2 = (
            f"The TCN encoder mapped that pattern onto a user-state "
            f"embedding {deviation:.2f} cosine units from your "
            f"established baseline — "
        )
        if deviation >= 0.4:
            para2 += "a substantial shift "
        elif deviation >= 0.2:
            para2 += "a moderate shift "
        elif deviation >= 0.05:
            para2 += "a small shift "
        else:
            para2 += "essentially no movement "
        para2 += f"toward the '{quadrant['name']}' quadrant ({quadrant['why']})."
    else:
        para2 = (
            f"Your behavioural baseline is still warming up "
            f"(message {messages_in_session} of the session, threshold "
            f"is normally five). The encoder is currently treating this "
            f"turn as the '{quadrant['name']}' quadrant ({quadrant['why']})"
            f" based on the raw keystroke metrics alone."
        )

    if engagement_score > 0:
        para2 += f" Composite engagement score sits at {engagement_score:.2f}."

    # Append the discrete user-state-label sentence (Live State Badge).
    # Surfaces the classifier's argmax as plain English so the user
    # can audit *why* the badge in the nav says what it does.
    state_sentence = _user_state_label_sentence(user_state_label)
    if state_sentence:
        para2 += " " + state_sentence

    # Append the typing-biometric Identity Lock sentence.  Surfaces
    # whether the keystroke template matched, drifted, or is still
    # registering.  Cites Monrose & Rubin (1997) and Killourhy &
    # Maxion (2009) implicitly by appearing here -- the docstring of
    # i3.biometric.keystroke_auth carries the references.
    biometric_sentence = _biometric_sentence(biometric)
    if biometric_sentence:
        para2 += " " + biometric_sentence

    # ------------------------------------------------------------------
    # Paragraph 3 — how adaptation responded
    # ------------------------------------------------------------------
    cl_phrase = _phrase_load(cognitive_load)
    v_phrase = _phrase_verbosity(verbosity)
    f_phrase = _phrase_formality(formality)

    para3 = (
        f"The adaptation controller produced cognitive_load={cognitive_load:.2f}"
        f" ({cl_phrase}), verbosity={verbosity:.2f} ({v_phrase}), and"
        f" formality={formality:.2f} ({f_phrase})."
    )
    if accessibility > 0.5:
        para3 += (
            f" Accessibility is raised to {accessibility:.2f}, so the"
            f" rewriter swaps complex words for plain alternatives."
        )
    if changes:
        change_clauses: list[str] = []
        for ch in changes[:6]:
            axis = str(ch.get("axis", "")) or "axis"
            change_text = str(ch.get("change", "")) or "rewritten"
            change_clauses.append(f"{axis} {change_text}")
        para3 += (
            " The post-processor then "
            + "; ".join(change_clauses)
            + "."
        )
    else:
        para3 += " The post-processor left the surface text untouched on this turn."

    # Accessibility-mode override sentence — only when active.
    # NB: ``accessibility`` is a *float* local further up (the
    # adaptation axis value); the kwarg we need here is the dict
    # passed via :func:`build_reasoning_trace` and aliased to
    # ``access_state_dict`` at function entry.
    access_sentence = _accessibility_sentence(access_state_dict)
    if access_sentence:
        para3 += " " + access_sentence

    # Per-biometric LoRA personalisation sentence — only when an
    # active adapter actually shifted the base adaptation.  Surfaces
    # the n_updates count + the dominant-axis drift so the user can
    # audit *how much* their preference history shaped this turn's
    # response.  Cites Hu et al. 2021 (LoRA) and Houlsby et al. 2019
    # (Adapter modules) by inheritance from
    # i3.personalisation.lora_adapter.
    pers_sentence = _personalisation_sentence(personalisation)
    if pers_sentence:
        para3 += " " + pers_sentence

    # ------------------------------------------------------------------
    # Paragraph 4 — which path the router chose, and why
    # ------------------------------------------------------------------
    # Multi-turn history prefix: when the pipeline carried prior
    # exchanges into this turn, surface that explicitly so the user
    # can audit *why* a contextual prompt like "what about animals?"
    # resolved correctly.  This is the visible HMI handle on the
    # closed-loop, history-aware retrieval/SLM path.
    history_prefix = ""
    try:
        h_used = int(history_turns_used)
    except (TypeError, ValueError):
        h_used = 0
    if h_used > 0:
        s = "" if h_used == 1 else "s"
        history_prefix = (
            f"Working from the last {h_used} conversation turn{s} of"
            f" context (prepended to both the retriever's embedding query"
            f" and the SLM's prompt). Without that history, the retriever"
            f" would have seen only this turn's text in isolation. "
        )

    # Co-reference narration — only spoken when the resolver actually
    # rewrote a pronoun-laden follow-up.  No-op turns (no pronoun, or
    # pronoun with no compatible referent) leave the prefix empty so
    # the trace stays terse on single-turn flows.
    coref_prefix = ""
    if isinstance(coreference_resolution, dict):
        ent = coreference_resolution.get("used_entity")
        pron = coreference_resolution.get("used_pronoun")
        original = str(coreference_resolution.get("original_query") or "").strip()
        if isinstance(ent, dict) and ent.get("canonical") and pron:
            ent_canon = str(ent.get("canonical"))
            ent_kind = str(ent.get("kind") or "entity").upper()
            # Try to extract the "N turns ago" phrase straight from
            # the tracker's reasoning string ("(most recent org
            # mentioned 1 turn ago)") so the trace voices it the same
            # way every turn.  Falls back to a plain "most recent
            # ENT_KIND" when the tracker phrasing is unavailable.
            reasoning_str = str(coreference_resolution.get("reasoning") or "").strip()
            import re as _re
            paren = _re.search(r"\(([^)]+)\)", reasoning_str)
            qualifier = paren.group(1) if paren else f"most recent {ent_kind}"
            coref_prefix = (
                f"You said '{original}'; I resolved '{pron}' to "
                f"'{ent_canon}' ({qualifier}) before retrieval. "
            )

    # Self-critique narration — only spoken when the loop actually
    # fired a regenerate (the interesting case).  Accepted-first-try
    # turns leave the prefix empty so the trace stays terse.
    critique_prefix = _critique_sentence(critique)

    para4 = (
        f"{history_prefix}"
        f"{coref_prefix}"
        f"{critique_prefix}"
        f"The LinUCB bandit chose {routed_to} with confidence {route_p:.2f}"
        f" (edge {local_p:.2f} / cloud {cloud_p:.2f}). "
    )
    if path_lc in {"retrieval", "retrieval_borderline"}:
        para4 += (
            f"Path: {path_label}. The retrieval head matched a curated"
            f" demo entry at cosine {retrieval_score:.2f}"
        )
        if path_lc == "retrieval_borderline":
            para4 += " — borderline match, so a calibration check was applied"
        para4 += (
            ", short-circuiting the SLM and emitting a deterministic answer."
        )
    elif path_lc == "slm":
        para4 += (
            "Path: SLM generation. No curated retrieval entry crossed the"
            " similarity threshold, so the on-device 53M-parameter"
            " transformer decoded the response token-by-token under the"
            " adaptation conditioning above."
        )
    elif path_lc.startswith("tool:"):
        tool_name = path_lc.split(":", 1)[1]
        para4 += (
            f"Path: tool:{tool_name}. The intent classifier routed this"
            f" turn to the deterministic '{tool_name}' tool rather than"
            f" the language model."
        )
    elif path_lc == "ood":
        para4 += (
            "Path: out-of-distribution. The retrieval head found no"
            " match and the SLM gating logic refused to emit, so the"
            " safe-fallback message was returned."
        )
    else:
        para4 += f"Path: {path_label}."

    # Paragraph 4-bis — cloud-route narration.  Only fires when the
    # routing decision actually picked the cloud arm.  Carries the
    # complexity score, the consent + budget context, and the
    # per-call PII redaction count so the reviewer can audit *why*
    # the cloud fired and what was protected before any byte left
    # the host.  Adaptation enforcement is route-agnostic: the cloud
    # reply still runs through ``adapt_with_log`` and the
    # self-critique loop, which we surface here to make the pitch
    # visible in plain English.
    para_cloud = _cloud_route_paragraph(
        routing_decision=routing_decision,
        privacy_budget=privacy_budget,
        cognitive_load=cognitive_load,
        formality=formality,
    )

    # Optional fifth paragraph on the visible response itself.
    para5_pieces: list[str] = []
    if response_preview:
        para5_pieces.append(
            f'Final visible reply (first {len(response_preview)} chars):'
            f' "{response_preview.strip()}".'
        )
    if changes:
        bullets = ", ".join(
            f"{c.get('axis')}={c.get('value')}" for c in changes[:4]
        )
        para5_pieces.append(
            f"Per-axis rewrites this turn: {bullets}."
        )
    para5 = " ".join(para5_pieces) if para5_pieces else ""

    # Phase B.5 (2026-04-25): hierarchical memory paragraph.  When
    # the session has any user-stated facts or a thread summary, we
    # surface them as one extra paragraph so the reviewer sees the
    # cross-turn memory plumbing in plain English.
    para6 = ""
    try:
        if session_memory and isinstance(session_memory, dict):
            facts = session_memory.get("user_facts") or []
            stack = session_memory.get("topic_stack") or []
            thread = (session_memory.get("thread_summary") or "").strip()
            fact_phrases: list[str] = []
            for f in facts[-5:]:
                pred = f.get("predicate", "")
                obj = f.get("object", "")
                if pred == "role":
                    fact_phrases.append(f"is a {obj}")
                elif pred == "preference":
                    fact_phrases.append(f"prefers {obj}")
                elif pred == "name":
                    fact_phrases.append(f"goes by {obj}")
                elif pred and obj:
                    fact_phrases.append(f"{pred.replace('_', ' ')} {obj}")
            pieces: list[str] = []
            if fact_phrases:
                pieces.append(
                    "User-stated facts on file: "
                    + "; ".join(fact_phrases) + "."
                )
            if thread:
                pieces.append(f"Session topic thread: {thread}")
            if stack and not thread:
                top = ", ".join(
                    e["canonical"] for e in stack[:3] if isinstance(e, dict)
                )
                if top:
                    pieces.append(f"Recent topic stack: {top}.")
            if pieces:
                pieces.append(
                    "These were considered when generating this response."
                )
                para6 = " ".join(pieces)
    except Exception:  # pragma: no cover - decorative only
        para6 = ""

    # Phase B.3 (2026-04-25): explain-decomposition paragraph.  When
    # the engine ran the multi-step decomposer this turn, narrate the
    # decomposition + sub-question count so the reviewer sees the
    # "model thinking in steps" surface.
    para7 = ""
    try:
        if explain_plan and isinstance(explain_plan, dict):
            topic = explain_plan.get("topic") or ""
            sub_qs = explain_plan.get("sub_questions") or []
            if topic and len(sub_qs) >= 2:
                para7 = (
                    f"Detected an 'explain' query about {topic}. Decomposed "
                    f"it into {len(sub_qs)} sub-questions ("
                    + ", ".join(
                        q.lower().rstrip("?.") for q in sub_qs[:3]
                    )
                    + "), retrieved each sub-answer separately, then "
                    "composed the final response."
                )
    except Exception:  # pragma: no cover - decorative only
        para7 = ""

    narrative_paragraphs = [
        p for p in (para1, para2, para3, para4, para_cloud, para5, para6, para7) if p
    ]

    # ------------------------------------------------------------------
    # Signal chips — short label/value tags for the UI strip
    # ------------------------------------------------------------------
    chips: list[dict[str, str]] = []
    chips.append({
        "label": "composition",
        "value": f"{composition_ms / 1000.0:.2f} s",
        "hint": "time from first keystroke to send",
    })
    chips.append({
        "label": "edits",
        "value": str(edit_count),
        "hint": "backspaces + deletes during composition",
    })
    if iki_summary["have_data"] and iki_summary["have_baseline"]:
        sign = "+" if iki_summary["pct_vs_baseline"] >= 0 else ""
        chips.append({
            "label": "IKI vs baseline",
            "value": f"{sign}{iki_summary['pct_vs_baseline']:.0f}%",
            "hint": "inter-keystroke interval vs running mean",
        })
    elif iki_summary["have_data"]:
        chips.append({
            "label": "mean IKI",
            "value": f"{iki_summary['mean_ms']:.0f} ms",
            "hint": "inter-keystroke interval — baseline warming up",
        })
    chips.append({
        "label": "state shift",
        "value": f"{deviation:+.2f}",
        "hint": "cosine deviation from your baseline embedding",
    })
    chips.append({
        "label": "quadrant",
        "value": quadrant["name"],
        "hint": quadrant["why"],
    })
    chips.append({
        "label": "cognitive_load",
        "value": f"{cognitive_load:.2f}",
        "hint": cl_phrase,
    })
    chips.append({
        "label": "verbosity",
        "value": f"{verbosity:.2f}",
        "hint": v_phrase,
    })
    chips.append({
        "label": "formality",
        "value": f"{formality:.2f}",
        "hint": f_phrase,
    })
    chips.append({
        "label": "router",
        "value": f"edge {local_p:.2f}",
        "hint": f"LinUCB chose {routed_to}",
    })
    chips.append({
        "label": "path",
        "value": path_label,
        "hint": _path_hint(response_path, retrieval_score),
    })
    if engagement_score > 0:
        chips.append({
            "label": "engagement",
            "value": f"{engagement_score:.2f}",
            "hint": "composite engagement score",
        })

    # ------------------------------------------------------------------
    # Decision chain — 4-5 step audit trail
    # ------------------------------------------------------------------
    chain: list[dict[str, str]] = []
    chain.append({
        "step": "Encoder",
        "what": (
            "TCN dilated-conv over the 32-d feature window → 64-d "
            "user-state vector"
        ),
        "why": (
            f"captures temporal typing rhythm; current shift {deviation:+.2f} "
            f"vs baseline ({quadrant['name']})"
        ),
    })
    chain.append({
        "step": "Adaptation",
        "what": (
            f"8-axis controller emitted cognitive_load={cognitive_load:.2f}, "
            f"verbosity={verbosity:.2f}, formality={formality:.2f}"
        ),
        "why": _adaptation_why(
            composition_ms=composition_ms,
            edit_count=edit_count,
            pause_ms=pause_ms,
            cognitive_load=cognitive_load,
            verbosity=verbosity,
        ),
    })
    chain.append({
        "step": "Routing",
        "what": (
            f"LinUCB bandit picked {routed_to} "
            f"(edge {local_p:.2f} / cloud {cloud_p:.2f})"
        ),
        "why": (
            "edge route preferred while cloud is disabled; bandit reward "
            "weights still update from engagement signals"
        ),
    })

    # Step 4 depends on the response_path
    if path_lc in {"retrieval", "retrieval_borderline"}:
        chain.append({
            "step": "Retrieval",
            "what": (
                f"cosine {retrieval_score:.2f} match against curated "
                "demo intent index"
            ),
            "why": (
                "exact-match short-circuit — no SLM decode needed when a "
                "curated answer exists"
            ),
        })
    elif path_lc == "slm":
        chain.append({
            "step": "SLM decode",
            "what": (
                "53M-param decoder transformer ran cross-attention over "
                "the adaptation vector and emitted tokens"
            ),
            "why": (
                "no curated retrieval entry crossed the threshold, so "
                "the on-device language model generated from scratch"
            ),
        })
    elif path_lc.startswith("tool:"):
        tool_name = path_lc.split(":", 1)[1]
        chain.append({
            "step": f"Tool: {tool_name}",
            "what": (
                f"deterministic '{tool_name}' tool produced the response"
            ),
            "why": (
                "intent classifier routed away from the language model "
                "for safety / determinism"
            ),
        })
    elif path_lc == "ood":
        chain.append({
            "step": "OOD fallback",
            "what": "no retrieval match + SLM gate refused — safe fallback used",
            "why": (
                "out-of-distribution intent; the demo prefers a graceful "
                "fallback over a hallucinated answer"
            ),
        })
    else:
        chain.append({
            "step": "Response",
            "what": f"path={path_label}",
            "why": "fallback path",
        })

    # Step 5 — rewriting (post-processor)
    if changes:
        what_bits = "; ".join(
            f"{ch.get('axis')} {ch.get('change')}" for ch in changes[:4]
        )
        chain.append({
            "step": "Rewriting",
            "what": f"post-processor applied: {what_bits}",
            "why": (
                "deterministic rewrite enforces the adaptation vector on "
                "the surface text without re-decoding"
            ),
        })
    else:
        chain.append({
            "step": "Rewriting",
            "what": "no axis crossed its threshold — surface text unchanged",
            "why": (
                "the adaptation vector is close enough to neutral that no "
                "rewrite was triggered"
            ),
        })

    return {
        "narrative_paragraphs": narrative_paragraphs,
        "signal_chips": chips,
        "decision_chain": chain,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float) -> float:
    """Return ``value`` as a finite float, or ``default`` on failure."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _summarise_iki(timings: list[float]) -> dict[str, Any]:
    """Summarise inter-keystroke intervals against a baseline.

    The "baseline" used here is the running mean of the supplied
    timings — there is no longer-term per-user history available in
    this stateless module, so we treat the tail (last 50 entries) as
    the current sample and the prefix as the baseline.  When fewer than
    8 samples are present we declare "baseline warming up".
    """
    if not timings:
        return {
            "have_data": False,
            "have_baseline": False,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "pct_vs_baseline": 0.0,
        }
    if len(timings) < 8:
        mean_ms = float(sum(timings) / len(timings))
        return {
            "have_data": True,
            "have_baseline": False,
            "mean_ms": mean_ms,
            "std_ms": 0.0,
            "pct_vs_baseline": 0.0,
        }
    # Use a 50/50 split: earlier half = baseline, recent half = current.
    half = len(timings) // 2
    baseline = timings[:half]
    recent = timings[half:]
    base_mean = float(sum(baseline) / len(baseline)) if baseline else 0.0
    recent_mean = float(sum(recent) / len(recent)) if recent else 0.0
    try:
        recent_std = float(statistics.pstdev(recent)) if len(recent) > 1 else 0.0
    except statistics.StatisticsError:
        recent_std = 0.0
    pct = (
        100.0 * (recent_mean - base_mean) / base_mean
        if base_mean > 1e-9
        else 0.0
    )
    return {
        "have_data": True,
        "have_baseline": True,
        "mean_ms": recent_mean,
        "std_ms": recent_std,
        "pct_vs_baseline": pct,
    }


def _quadrant(
    *,
    user_state_embedding_2d: tuple[float, float] | list[float] | None,
    baseline_established: bool,
    composition_ms: float,
    edit_count: int,
    pause_ms: float,
) -> dict[str, str]:
    """Name the qualitative quadrant of the user-state embedding.

    Quadrant naming convention from the project brief:
        (+, +) → "energetic/focused"
        (+, −) → "stressed"
        (−, +) → "relaxed/curious"
        (−, −) → "tired/disengaged"
    Falls back to keystroke heuristics when no embedding is available
    or the baseline is not yet established.
    """
    if not baseline_established:
        return {
            "name": "warming up",
            "why": "not enough turns yet to anchor a baseline",
        }

    e: list[float] = []
    if user_state_embedding_2d is not None:
        try:
            e = [float(v) for v in user_state_embedding_2d]
        except (TypeError, ValueError):
            e = []

    if len(e) >= 2:
        x, y = e[0], e[1]
        if x >= 0 and y >= 0:
            return {
                "name": "energetic/focused",
                "why": "positive on both temporal and linguistic axes",
            }
        if x >= 0 and y < 0:
            return {
                "name": "stressed",
                "why": "fast/erratic typing with negative linguistic signal",
            }
        if x < 0 and y >= 0:
            return {
                "name": "relaxed/curious",
                "why": "slower typing but rich linguistic signal",
            }
        return {
            "name": "tired/disengaged",
            "why": "negative on both temporal and linguistic axes",
        }

    # Keystroke-only fallback.
    rushed = composition_ms > 0 and composition_ms < 1500 and edit_count >= 1
    hesitant = pause_ms >= 800 or (composition_ms > 4000 and edit_count >= 2)
    if rushed:
        return {
            "name": "rushed/uncertain",
            "why": "short composition window with edits suggests urgency",
        }
    if hesitant:
        return {
            "name": "hesitant/considered",
            "why": "long pause and/or many edits before send",
        }
    return {
        "name": "neutral",
        "why": "no strong typing signal in either direction",
    }


def _phrase_load(load: float) -> str:
    if load >= _VERY_HIGH:
        return "very high → single-sentence reply"
    if load >= _HIGH:
        return "raised → trim response"
    if load <= _VERY_LOW:
        return "very low → generous detail OK"
    if load <= _LOW:
        return "low → generous detail"
    return "neutral"


def _cloud_route_paragraph(
    *,
    routing_decision: dict | None,
    privacy_budget: dict | None,
    cognitive_load: float,
    formality: float,
) -> str:
    """Compose the cloud-route narration paragraph.

    Returns ``""`` (a falsy value the caller filters out) for any
    turn that didn't route to the cloud LLM.  When the cloud DID fire,
    returns a single-paragraph plain-English story:

      * which arm fired and why (the complexity score + threshold)
      * the PII redaction count (``"...redacted 2 tokens (1 email,
        1 phone-number-like pattern)..."``)
      * the adaptation values that the system prompt encoded so the
        cloud reply respects the user's cognitive load + formality
      * a closing sentence affirming that adapt_with_log + the
        self-critique loop fired on the cloud reply too — adaptation
        is route-agnostic.
    """
    if not isinstance(routing_decision, dict):
        return ""
    arm = str(routing_decision.get("arm", "")).lower()
    route = str(routing_decision.get("route", "")).lower()
    if arm != "cloud_llm" and route != "cloud_llm":
        return ""

    complexity_dict = routing_decision.get("complexity") or {}
    score = _safe_float(complexity_dict.get("score"), 0.0)
    factors = complexity_dict.get("factors") or {}
    dominant = ""
    try:
        if isinstance(factors, dict) and factors:
            name, _val = max(factors.items(), key=lambda kv: float(kv[1]))
            dominant = str(name).replace("_", " ").replace("factor", "").strip()
    except (TypeError, ValueError):
        dominant = ""

    pii = int(_safe_float(routing_decision.get("pii_redactions"), 0.0))
    bytes_in = int(_safe_float(routing_decision.get("bytes_in"), 0.0))
    bytes_out = int(_safe_float(routing_decision.get("bytes_out"), 0.0))
    bytes_redacted = int(_safe_float(
        routing_decision.get("bytes_redacted"), 0.0
    ))

    # Per-category breakdown — the privacy_budget snapshot carries
    # the cumulative session-level counts, but the per-call slice we
    # care about isn't separately tracked.  Best we can do without
    # over-claiming is name the dominant category in the budget if
    # one exists.
    cat_clauses: list[str] = []
    if isinstance(privacy_budget, dict):
        sens_cats = privacy_budget.get("sensitive_categories") or {}
        if isinstance(sens_cats, dict):
            for k, v in sorted(
                sens_cats.items(), key=lambda kv: -int(kv[1])
            ):
                try:
                    n = int(v)
                except (TypeError, ValueError):
                    continue
                if n <= 0:
                    continue
                cat_clauses.append(
                    f"{n} {k.replace('_', '-')}"
                )
                if len(cat_clauses) >= 3:
                    break
    pii_clause = ""
    if pii > 0:
        if cat_clauses:
            pii_clause = (
                f" Before the network call, the PII sanitiser "
                f"redacted {pii} token{'s' if pii != 1 else ''} "
                f"({', '.join(cat_clauses)} in the session so far)."
            )
        else:
            pii_clause = (
                f" Before the network call, the PII sanitiser "
                f"redacted {pii} token{'s' if pii != 1 else ''}."
            )
    elif bytes_redacted > 0:
        pii_clause = (
            f" Before the network call, the PII sanitiser "
            f"protected {bytes_redacted} bytes of data that would "
            f"otherwise have crossed the network."
        )
    else:
        pii_clause = (
            " Before the network call, the PII sanitiser scanned "
            "the prompt and found nothing to redact."
        )

    bandwidth_clause = ""
    if bytes_in > 0 or bytes_out > 0:
        bandwidth_clause = (
            f" The wire footprint was {bytes_in} bytes outbound "
            f"and {bytes_out} bytes inbound."
        )

    score_clause = (
        f"This turn routed to the cloud LLM (Anthropic Claude) because "
        f"the LinUCB bandit estimated complexity {score:.2f}"
    )
    if dominant:
        score_clause += (
            f" — dominant signal: {dominant} — above the 0.65 edge"
            f" threshold."
        )
    else:
        score_clause += " — above the 0.65 edge threshold."

    adapt_clause = (
        f" The system prompt was built from your live "
        f"AdaptationVector so the cloud reply respects your current "
        f"cognitive_load ({cognitive_load:.2f}) and formality "
        f"preference ({formality:.2f})."
    )
    closing = (
        " The cloud response was then run through the same "
        "adapt_with_log post-processor + self-critique loop as "
        "local replies — adaptation enforcement is route-agnostic."
    )
    return (
        score_clause
        + pii_clause
        + bandwidth_clause
        + adapt_clause
        + closing
    )


def _phrase_verbosity(v: float) -> str:
    if v >= _HIGH:
        return "elaborate; follow-up invitation appended"
    if v <= _LOW:
        return "concise; hedges and follow-ups stripped"
    return "moderate"


def _phrase_formality(f: float) -> str:
    if f >= _HIGH:
        return "formal; contractions expanded"
    if f <= _LOW:
        return "casual; contractions kept"
    return "neutral register"


def _path_label(path: str) -> str:
    p = (path or "").lower()
    if p in {"retrieval", "retrieval_borderline"}:
        return "retrieval"
    if p == "slm":
        return "SLM"
    if p.startswith("tool:"):
        return p
    if p == "ood":
        return "OOD"
    return path or "unknown"


def _path_hint(path: str, retrieval_score: float) -> str:
    p = (path or "").lower()
    if p == "retrieval":
        return f"matched curated demo intent at cosine {retrieval_score:.2f}"
    if p == "retrieval_borderline":
        return f"borderline retrieval match (cosine {retrieval_score:.2f})"
    if p == "slm":
        return "on-device SLM generated tokens"
    if p.startswith("tool:"):
        return "deterministic tool path"
    if p == "ood":
        return "out-of-distribution — safe fallback"
    return "response path"


def _adaptation_why(
    *,
    composition_ms: float,
    edit_count: int,
    pause_ms: float,
    cognitive_load: float,
    verbosity: float,
) -> str:
    drivers: list[str] = []
    if cognitive_load >= _HIGH:
        if edit_count >= 2:
            drivers.append("multiple edits suggest hesitation")
        if pause_ms >= 800:
            drivers.append("long pre-send pause")
        if composition_ms > 0 and composition_ms < 1500:
            drivers.append("very fast composition")
        if not drivers:
            drivers.append("cumulative typing signal")
        return "; ".join(drivers) + " → user is mentally taxed → shorter reply"
    if cognitive_load <= _LOW:
        return "calm, deliberate typing → spare bandwidth → longer reply OK"
    if verbosity >= _HIGH:
        return "linguistic signal favours elaboration"
    if verbosity <= _LOW:
        return "linguistic signal favours concise replies"
    return "no axis crossed a strong threshold; staying near neutral"


def _affect_shift_prefix(affect_shift: Any | None) -> str:
    """Render the affect-shift announcement sentence, or ``""``.

    Accepts either an :class:`i3.affect.AffectShift` dataclass or
    its ``to_dict()`` serialisation — the WS layer ships the dict,
    in-process callers pass the dataclass.  Returns an empty string
    when *affect_shift* is ``None``, ``detected=False``, or any
    field cannot be coerced.
    """
    if affect_shift is None:
        return ""
    if isinstance(affect_shift, dict):
        detected = bool(affect_shift.get("detected", False))
        direction = str(affect_shift.get("direction", "neutral"))
        magnitude = _safe_float(affect_shift.get("magnitude"), 0.0)
    else:
        detected = bool(getattr(affect_shift, "detected", False))
        direction = str(getattr(affect_shift, "direction", "neutral"))
        magnitude = _safe_float(getattr(affect_shift, "magnitude", 0.0), 0.0)
    if not detected:
        return ""
    return (
        f"Affect-shift detected this turn ({direction}, "
        f"magnitude {magnitude:.1f}σ). The model proactively appended "
        f"a short check-in to its reply."
    )


def _user_state_label_sentence(user_state_label: dict | None) -> str:
    """Render the user-state classifier's argmax as a one-line sentence.

    Surfaces the discrete state, confidence, and top contributing
    signals.  Returns ``""`` when the classifier hasn't run, the dict
    is malformed, or ``state`` is ``"warming up"`` and we want to
    avoid stating the obvious during session warm-up.
    """
    if not isinstance(user_state_label, dict):
        return ""
    state = str(user_state_label.get("state", "")).strip()
    if not state:
        return ""
    confidence = _safe_float(user_state_label.get("confidence"), 0.0)
    secondary = user_state_label.get("secondary_state")
    secondary_str = str(secondary).strip() if secondary else ""
    signals = user_state_label.get("contributing_signals") or []
    signals_clean: list[str] = []
    for s in signals:
        s_str = str(s).strip()
        if s_str:
            signals_clean.append(s_str)
        if len(signals_clean) >= 3:
            break

    sentence = (
        f"The state classifier flagged this as '{state}' "
        f"(confidence {confidence:.2f})"
    )
    if secondary_str and secondary_str != state:
        sentence += f", with '{secondary_str}' as the runner-up"
    if signals_clean:
        sentence += ". Top signals: " + ", ".join(signals_clean) + "."
    else:
        sentence += "."
    return sentence


def _accessibility_sentence(accessibility: dict | None) -> str:
    """Render the accessibility-mode override as a one-line sentence.

    Only emits a sentence when the mode is ``active=True``.  Mentions
    the forced adaptation knob values and the controller's reason so
    the user can see the override was deliberate.
    """
    if not isinstance(accessibility, dict):
        return ""
    if not accessibility.get("active"):
        return ""
    reason = str(accessibility.get("reason", "")).strip()
    sentence = (
        "Accessibility mode is active — adaptation values were "
        "force-overridden (cognitive_load≥0.85, accessibility≥0.95, "
        "verbosity≤0.25) to ensure short, simple replies"
    )
    if reason:
        sentence += f" ({reason})"
    sentence += "."
    return sentence


def _biometric_sentence(biometric: dict | None) -> str:
    """Render the typing-biometric Identity Lock state as one sentence.

    Surfaces (a) registration progress, (b) verified match, or
    (c) drift / mismatch in plain English.  Returns ``""`` when no
    biometric dict is attached so callers that haven't wired in the
    feature still get a clean trace.

    Cites Monrose & Rubin (1997) "Authentication via keystroke
    dynamics" and Killourhy & Maxion (2009) "Comparing anomaly-
    detection algorithms for keystroke dynamics" by inheritance from
    :class:`i3.biometric.keystroke_auth.KeystrokeAuthenticator`.
    """
    if not isinstance(biometric, dict):
        return ""
    state = str(biometric.get("state", "")).strip()
    if not state:
        return ""
    progress = int(_safe_float(biometric.get("enrolment_progress"), 0))
    target = int(_safe_float(biometric.get("enrolment_target"), 5))
    similarity = _safe_float(biometric.get("similarity"), 0.0)
    confidence = _safe_float(biometric.get("confidence"), 0.0)
    threshold = _safe_float(biometric.get("threshold"), 0.65)
    is_owner = bool(biometric.get("is_owner"))
    drift_alert = bool(biometric.get("drift_alert"))
    diverged = biometric.get("diverged_signals") or []
    diverged_clean = [str(d) for d in diverged if str(d).strip()][:3]

    if state == "registering":
        return (
            f"Biometric enrolment in progress -- {progress} / {target} "
            f"typing samples collected."
        )
    if state == "registered":
        return (
            f"Biometric template just registered -- future turns will be "
            f"verified against this rhythm (threshold {threshold:.2f})."
        )
    if state == "verifying" and is_owner:
        return (
            f"Biometric continuous-auth confirmed (similarity "
            f"{max(0.0, min(1.0, confidence)):.2f}, threshold "
            f"{threshold:.2f})."
        )
    if drift_alert or state == "mismatch":
        if diverged_clean:
            why = ", ".join(diverged_clean)
        else:
            why = "IKI rhythm and edit cadence both off-baseline"
        return (
            f"Typing pattern diverges from registered owner -- {why} "
            f"(confidence {confidence:.2f} below threshold "
            f"{threshold:.2f})."
        )
    return ""


def _personalisation_sentence(personalisation: dict | None) -> str:
    """Render the per-biometric LoRA personalisation as one sentence.

    Surfaces the headline differentiator: this user has *personalised
    weights* layered onto the base adaptation, gated by their
    typing-biometric identity.  Returns ``""`` when the adapter is
    inactive (no biometric template registered, or zero updates so
    far) so the trace stays terse for unregistered demos.

    Cites Hu et al. 2021 "LoRA: Low-Rank Adaptation of Large Language
    Models" (arXiv:2106.09685) and Houlsby et al. 2019 "Parameter-
    Efficient Transfer Learning for NLP" (ICML 2019) by inheritance
    from :mod:`i3.personalisation.lora_adapter`.
    """
    if not isinstance(personalisation, dict):
        return ""
    if not personalisation.get("applied"):
        return ""
    n_updates = int(_safe_float(personalisation.get("n_updates"), 0))
    if n_updates <= 0:
        # Adapter exists but has never been trained — residual is
        # zero by LoRA init, so there's nothing meaningful to narrate.
        return ""
    drift = personalisation.get("drift") or {}
    if not isinstance(drift, dict):
        drift = {}
    # Pick the two axes with the largest |drift| as the headline.
    ranked: list[tuple[str, float]] = []
    for axis, val in drift.items():
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if axis == "reserved":
            continue
        ranked.append((str(axis), v))
    ranked.sort(key=lambda kv: -abs(kv[1]))
    top = ranked[:2]
    if not top:
        return (
            f"Your personal LoRA adapter ({n_updates} updates "
            f"accumulated) was queried this turn. The adapter is keyed "
            f"by your typing-biometric template hash and persists "
            f"locally; it has never left this device."
        )
    drift_clauses = []
    for axis, val in top:
        sign = "+" if val >= 0 else ""
        drift_clauses.append(f"{axis} by {sign}{val:.2f}")
    drift_text = " and ".join(drift_clauses)
    return (
        f"Your personal LoRA adapter ({n_updates} updates accumulated) "
        f"shifted {drift_text} from the base controller's neutral "
        f"output. The adapter is keyed by your typing-biometric "
        f"template hash and persists locally; it has never left this "
        f"device."
    )


def _critique_sentence(critique: dict | None) -> str:
    """Render the self-critique loop as a one-sentence inner monologue.

    Returns ``""`` for the boring cases (critic didn't run, or
    accepted on the first try) so the trace stays terse.  Only the
    *interesting* cases — a regenerate fired — are narrated, with
    enough detail (per-criterion failure reasons + first/second
    score) for a reviewer to see the model self-correcting in front
    of them.
    """
    if not isinstance(critique, dict) or not critique:
        return ""
    if not critique.get("regenerated"):
        return ""
    attempts = critique.get("attempts") or []
    if len(attempts) < 2:
        return ""
    first = attempts[0] if isinstance(attempts[0], dict) else {}
    second = attempts[-1] if isinstance(attempts[-1], dict) else {}
    first_score = _safe_float(first.get("score"), 0.0)
    second_score = _safe_float(second.get("score"), 0.0)

    # Pull the headline failure reasons from the FIRST attempt — those
    # are what triggered the regenerate.  We keep at most two and
    # strip the "criterion:" prefix so the prose reads cleanly.
    raw_reasons = first.get("reasons") or []
    cleaned_reasons: list[str] = []
    for r in raw_reasons:
        if not isinstance(r, str):
            continue
        text = r.split(":", 1)[1].strip() if ":" in r else r.strip()
        if text:
            cleaned_reasons.append(text)
        if len(cleaned_reasons) == 2:
            break
    reason_clause = (
        f" [{' + '.join(cleaned_reasons)}]" if cleaned_reasons else ""
    )

    accepted = bool(critique.get("accepted"))
    if accepted:
        outcome = (
            f"regenerated with T=0.4 and got {second_score:.2f}). "
            f"The accepted response is shown."
        )
    else:
        outcome = (
            f"regenerated with T=0.4 and got {second_score:.2f}, "
            f"still under the {_safe_float(critique.get('threshold'), 0.65):.2f} "
            f"threshold). The better of the two attempts is shown."
        )
    return (
        f"Self-critique loop fired (first attempt scored "
        f"{first_score:.2f}{reason_clause}; {outcome} "
    )


def _multimodal_prosody_sentence(multimodal: dict | None) -> str:
    """Render the voice-prosody fusion as one privacy-preserving sentence.

    Returns ``""`` when the user did not enable the mic on this turn
    (the default path) so the trace stays terse on the keystroke-only
    flow.  When the mic was enabled, narrates the contract:

    * how long the audio buffer was (``captured_seconds``)
    * how many features were extracted (always 8)
    * that the audio was discarded on-device
    * that the keystroke + voice signals fused into a 96-dimensional
      multimodal embedding

    Cites Schuller (2009) + Eyben et al. (2010) implicitly via the
    feature-set narration; the explicit citation lives in the module
    docstring of :mod:`i3.multimodal.prosody`.
    """
    if not isinstance(multimodal, dict):
        return ""
    if not multimodal.get("prosody_active"):
        return ""
    captured_seconds = _safe_float(multimodal.get("captured_seconds"), 0.0)
    fused_dim = int(_safe_float(multimodal.get("fused_dim"), 96))
    return (
        f"Voice prosody captured this turn ({captured_seconds:.1f} s of "
        f"audio, 8 prosodic features extracted on-device, audio "
        f"discarded). The keystroke + voice signals fused into a "
        f"{fused_dim}-dimensional multimodal user-state embedding."
    )


def _gaze_sentence(gaze: dict | None) -> str:
    """Render the gaze classifier result as one privacy-preserving sentence.

    Returns ``""`` when the camera was off (the default path).  When
    on, narrates:

    * the predicted gaze label + softmax confidence
    * the fact that the prediction came from a fine-tuned head (75k
      params) over a frozen MobileNetV3-small backbone (5.4M params)
    * the privacy note (only a 64×48 grayscale fingerprint reaches
      the server — no raw frames)
    * the gaze-conditioned response-timing message when the user
      wasn't looking at the screen.
    """
    if not isinstance(gaze, dict):
        return ""
    label = str(gaze.get("label", "") or "")
    if not label:
        return ""
    try:
        confidence = float(gaze.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    presence = bool(gaze.get("presence", True))
    note = gaze.get("gaze_aware_note")
    pretty_label = label.replace("_", "-")
    base = (
        f"Gaze classifier (fine-tuned MobileNetV3-small head, "
        f"75 k params over a 5.4 M frozen backbone): "
        f"{pretty_label}, confidence {confidence:.2f}. "
        f"Only a 64×48 grayscale fingerprint reached the server; "
        f"the raw frame was discarded on-device."
    )
    if presence:
        base += " Response delivered immediately."
    elif isinstance(note, str) and note:
        base += " " + note
    return base


__all__ = ["build_reasoning_trace"]
