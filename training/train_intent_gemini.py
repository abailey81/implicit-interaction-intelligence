"""Fine-tune Gemini 2.5 Flash via the direct Google AI Studio API.

Iter 51 (2026-04-27).  The closed-weight cloud counterpart to
``training/train_intent_lora.py``.  Uses the same dataset
(`data/processed/intent/{train,val,test}.jsonl`) so the comparison
in `docs/huawei/finetune_artefact.md` is apples-to-apples.

WHY THE DIRECT API (and not Vertex AI):
    * Single env var ``GEMINI_API_KEY`` — no service-account JSON.
    * No GCS bucket — dataset uploaded inline via the
      ``tunedModels.create`` REST body.
    * Lightweight dep: ``pip install google-generativeai`` (~5 MB
      vs Vertex's ~50 MB google-cloud-aiplatform).
    * Free tier (as of 2026-04): ~100 fine-tune jobs / month with
      reasonable token budgets — covers a portfolio piece for £0.
    * Tuned model is callable via the SAME ``models.generateContent``
      path that you'd use for un-tuned Gemini, so the inference
      adapter stays trivial.
    * No GCP project setup required beyond visiting
      https://aistudio.google.com/app/apikey to mint a key.

Cost (April 2026 list pricing, AI Studio supervised tuning):
    Tuning   ~ $0.001 / 1 k characters input.  3 k examples × ~250
             chars × 3 epochs ≈ 2.25 M chars → ~$2.25 ≈ £1.80.
    Inference: $0.075 / M input tokens, $0.30 / M output tokens.  Per
             call (~80 in + ~30 out): ~$0.000015.  Negligible.

Run:
    GEMINI_API_KEY=AIzaSy... python training/train_intent_gemini.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed" / "intent"
OUT_DIR = REPO_ROOT / "checkpoints" / "intent_gemini"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv(env_path: Path = REPO_ROOT / ".env") -> None:
    """Auto-load ``.env`` so callers don't need to ``set`` the key by hand.

    Iter 51: GEMINI_API_KEY can live in the project's gitignored
    ``.env`` file.  We read it line-by-line (no python-dotenv dep) and
    only set vars that are not already in the environment.
    """
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


_load_dotenv()


def _to_aistudio_examples(src: Path) -> list[dict]:
    """Translate our SFT JSONL into AI Studio's tuning format.

    AI Studio expects an array of ``{"text_input": ..., "output": ...}``
    dicts (`tunedModels.create` REST body, `tuningTask.trainingData`).
    """
    examples: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append({
                "text_input": row["input"],
                "output": json.dumps(row["output"], separators=(",", ":")),
            })
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-model",
        default="models/gemini-1.5-flash-001",
        help="Base model to tune from (must be a tunable model id; "
             "AI Studio supervised tuning is gated to the 1.5 line as "
             "of 2026 — 2.5-flash is generate_content-only)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="AI Studio tuning batch size (default 4)",
    )
    parser.add_argument(
        "--learning-rate-multiplier", type=float, default=1.0,
    )
    parser.add_argument(
        "--display-name",
        default=f"i3-intent-{time.strftime('%Y%m%d-%H%M%S')}",
        help="Display name for the tuned model",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key (defaults to GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Translate dataset + write a tuning plan but do NOT submit "
             "the (billable) job.  Use this to verify the pipeline "
             "without spending credits.",
    )
    args = parser.parse_args()

    # Step 1: translate our dataset.
    train_examples = _to_aistudio_examples(DATA_DIR / "train.jsonl")
    val_examples = _to_aistudio_examples(DATA_DIR / "val.jsonl")
    aistudio_train = OUT_DIR / "train_aistudio.json"
    aistudio_val = OUT_DIR / "val_aistudio.json"
    with aistudio_train.open("w", encoding="utf-8") as f:
        json.dump(train_examples, f, indent=2)
    with aistudio_val.open("w", encoding="utf-8") as f:
        json.dump(val_examples, f, indent=2)
    print(
        f"[data] translated {len(train_examples)} train + "
        f"{len(val_examples)} val examples", flush=True,
    )
    print(f"  train -> {aistudio_train}", flush=True)
    print(f"  val   -> {aistudio_val}", flush=True)

    # Cost estimate.
    avg_chars_in = sum(len(e["text_input"]) for e in train_examples) / max(
        len(train_examples), 1
    )
    avg_chars_out = sum(len(e["output"]) for e in train_examples) / max(
        len(train_examples), 1
    )
    total_chars = (
        len(train_examples) * (avg_chars_in + avg_chars_out) * args.epochs
    )
    plan = {
        "source_model": args.source_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate_multiplier": args.learning_rate_multiplier,
        "display_name": args.display_name,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "avg_chars_in": round(avg_chars_in, 1),
        "avg_chars_out": round(avg_chars_out, 1),
        "estimated_total_chars": int(total_chars),
        "estimated_cost_usd_low": round(total_chars * 0.001 / 1000, 3),
        "estimated_cost_usd_high": round(total_chars * 0.0015 / 1000, 3),
        "dry_run": args.dry_run,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    plan_path = OUT_DIR / "tuning_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"[plan] {plan_path}", flush=True)
    print(
        f"[cost] estimated ~${plan['estimated_cost_usd_low']:.2f}-"
        f"${plan['estimated_cost_usd_high']:.2f} (~£"
        f"{plan['estimated_cost_usd_low']*0.79:.2f}) for the tune; "
        "covered by AI Studio free tier in most cases.",
        flush=True,
    )

    if args.dry_run:
        print(
            "[dry-run] launch skipped.  Re-run without --dry-run + "
            "GEMINI_API_KEY set to actually tune.",
            flush=True,
        )
        return 0

    if not args.api_key:
        print(
            "[err] GEMINI_API_KEY not set.  Get one at "
            "https://aistudio.google.com/app/apikey",
            flush=True,
        )
        return 1

    # Detect which SDK is on the path.  The new ``google-genai`` package
    # is the supported route as of 2026; the old ``google-generativeai``
    # is deprecated and its ``create_tuned_model`` path is being retired.
    sdk = None
    try:
        from google import genai as new_genai  # google-genai
        sdk = "new"
    except ImportError:
        pass
    if sdk is None:
        try:
            import google.generativeai as old_genai  # deprecated
            sdk = "old"
        except ImportError:
            print(
                "[err] neither google-genai nor google-generativeai installed.\n"
                "  Recommended:  pip install google-genai",
                flush=True,
            )
            return 1

    print(
        f"[tune] using SDK: {'google-genai (new)' if sdk == 'new' else 'google-generativeai (deprecated)'}",
        flush=True,
    )
    print(
        f"[tune] starting AI Studio SFT on {args.source_model} "
        f"(epochs={args.epochs}, batch={args.batch_size}) ...",
        flush=True,
    )
    started = time.time()

    try:
        if sdk == "new":
            client = new_genai.Client(api_key=args.api_key)
            from google.genai import types as gtypes
            tuning_examples = [
                gtypes.TuningExample(
                    text_input=ex["text_input"],
                    output=ex["output"],
                )
                for ex in train_examples
            ]
            tuning_dataset = gtypes.TuningDataset(examples=tuning_examples)
            cfg = gtypes.CreateTuningJobConfig(
                tuned_model_display_name=args.display_name,
                epoch_count=args.epochs,
                batch_size=args.batch_size,
                learning_rate_multiplier=args.learning_rate_multiplier,
            )
            tuning_job = client.tunings.tune(
                base_model=args.source_model,
                training_dataset=tuning_dataset,
                config=cfg,
            )
            print(f"[tune] job submitted: {tuning_job.name}", flush=True)
            print("[poll] waiting for completion (Ctrl-C to detach; the job "
                  "continues running on Google's servers)...", flush=True)
            last_print = 0.0
            while True:
                state = str(getattr(tuning_job, "state", "")).upper()
                if "SUCCEEDED" in state or "FAILED" in state or "STATE_SUCCEEDED" in state:
                    break
                now = time.time()
                if now - last_print >= 30:
                    elapsed = now - started
                    print(
                        f"[poll] state={state} elapsed={elapsed:.0f}s "
                        f"(typical: 600-3600 s)",
                        flush=True,
                    )
                    last_print = now
                time.sleep(15)
                tuning_job = client.tunings.get(name=tuning_job.name)
            final_state = str(getattr(tuning_job, "state", "?"))
            tuned_model_name = getattr(tuning_job, "tuned_model", None) or \
                               getattr(tuning_job, "name", "?")
        else:
            old_genai.configure(api_key=args.api_key)
            operation = old_genai.create_tuned_model(
                source_model=args.source_model,
                training_data=train_examples,
                id=args.display_name,
                display_name=args.display_name,
                epoch_count=args.epochs,
                batch_size=args.batch_size,
                learning_rate_multiplier=args.learning_rate_multiplier,
            )
            print(f"[tune] job submitted: {operation.metadata.name}", flush=True)
            print("[poll] waiting for completion ...", flush=True)
            last_print = 0.0
            while not operation.done():
                now = time.time()
                if now - last_print >= 30:
                    print(f"[poll] elapsed={now-started:.0f}s", flush=True)
                    last_print = now
                time.sleep(15)
                operation = old_genai.get_operation(operation.metadata.name)
            tm = operation.result()
            tuned_model_name = tm.name
            final_state = str(tm.state)
    except Exception as exc:
        print(f"[err] tune failed: {type(exc).__name__}: {exc}", flush=True)
        # Common 2026 failure: AI Studio dropped supervised tuning of
        # 2.5-flash-001 from the public key tier.  Surface a helpful
        # next step instead of just dying.
        msg = str(exc).lower()
        if "not tunable" in msg or "permission" in msg or "403" in msg:
            print(
                "[hint] AI Studio's free-tier supervised tuning has migrated. "
                "Try '--source-model models/gemini-1.5-flash-001' (legacy "
                "tunable) or use Vertex AI for the 2.5 line.",
                flush=True,
            )
        return 2

    print(f"[done] tuned model: {tuned_model_name}", flush=True)
    print(f"[done] state: {final_state}", flush=True)
    print(f"[done] elapsed: {time.time() - started:.0f}s", flush=True)

    record = {
        **plan,
        "tuned_model_name": str(tuned_model_name),
        "state": str(final_state),
        "wall_time_s": round(time.time() - started, 1),
        "sdk": "google-genai" if sdk == "new" else "google-generativeai",
    }
    with (OUT_DIR / "tuning_result.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[save] tuning_result.json -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
