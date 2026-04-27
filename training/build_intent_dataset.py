"""Synthetic HMI command-intent dataset generator (deterministic).

Iter 51 (2026-04-27).  Produces a training corpus for fine-tuning a
small open-weight LLM on smartwatch- / smart-glasses-grade voice
commands.  The task: free-form English utterance → structured JSON
``{"action": str, "params": {...}}`` capturing the user's intent and
slot values.

Why HMI command-intent specifically:
    * Aligns with Huawei R&D UK HMI Lab's "interaction concepts" remit.
    * Doesn't compete with the from-scratch SLM thesis — the SLM is
      a generative chat model; this tuned LLM is a structured-output
      classifier+slot-extractor.  Different layer of the stack.
    * Tiny output space (10 actions, 4 typical slots) keeps fine-tune
      cheap on a 6.4 GB RTX 4050.

Reproducibility: ``random.seed(42)`` → identical 3000-row corpus on
every run.  90/5/5 train/val/test split with stratification on action.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path("data/processed/intent")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


# ---------------------------------------------------------------------------
# Action taxonomy and templates (10 actions covering the most-common HMI
# voice-command surfaces from a 2025 Smart Glasses / smartwatch UX scan).
# ---------------------------------------------------------------------------

ACTIONS: dict[str, dict[str, Any]] = {
    "set_timer": {
        "templates": [
            "set a timer for {duration}",
            "set timer {duration}",
            "start a {duration} timer",
            "wake me in {duration}",
            "remind me in {duration}",
            "{duration} timer please",
            "could you set a {duration} timer for me",
            "ping me in {duration}",
            "give me a {duration} timer",
            "i need a {duration} timer",
        ],
        "slots": ["duration_seconds"],
    },
    "set_alarm": {
        "templates": [
            "set an alarm for {time}",
            "wake me at {time}",
            "alarm at {time}",
            "set the alarm to {time}",
            "wake me up at {time} please",
            "i need to be up by {time}",
            "alarm for {time}",
            "remind me at {time} sharp",
        ],
        "slots": ["time"],
    },
    "send_message": {
        "templates": [
            "text {recipient} {message}",
            "send a message to {recipient} saying {message}",
            "tell {recipient} {message}",
            "message {recipient}: {message}",
            "send {recipient} a message: {message}",
            "ping {recipient} with {message}",
            "let {recipient} know {message}",
            "drop {recipient} a line: {message}",
        ],
        "slots": ["recipient", "message"],
    },
    "play_music": {
        "templates": [
            "play {genre}",
            "play some {genre} music",
            "put on {genre}",
            "i want to listen to {genre}",
            "play {genre} on spotify",
            "queue up some {genre}",
            "throw on some {genre}",
            "play {artist}",
            "play {artist} on spotify",
            "play me some {artist}",
        ],
        "slots": ["genre", "artist"],
    },
    "navigate": {
        "templates": [
            "navigate to {place}",
            "directions to {place}",
            "take me to {place}",
            "how do i get to {place}",
            "route me to {place}",
            "i need to get to {place}",
            "find me a route to {place}",
            "show me the way to {place}",
        ],
        "slots": ["place"],
    },
    "weather_query": {
        "templates": [
            "what's the weather",
            "what's the weather like",
            "is it going to rain today",
            "weather in {place}",
            "what's the weather in {place}",
            "do i need an umbrella",
            "is it warm outside",
            "what's the temperature",
            "will it rain tomorrow",
            "what's tomorrow's weather",
        ],
        "slots": ["place"],
    },
    "call": {
        "templates": [
            "call {recipient}",
            "ring {recipient}",
            "phone {recipient}",
            "dial {recipient}",
            "could you call {recipient}",
            "give {recipient} a call",
            "video call {recipient}",
            "facetime {recipient}",
        ],
        "slots": ["recipient", "video"],
    },
    "set_reminder": {
        "templates": [
            "remind me to {task}",
            "remind me to {task} at {time}",
            "remind me to {task} {when}",
            "set a reminder to {task}",
            "don't let me forget to {task}",
            "i need to {task} {when}",
            "make sure i {task} {when}",
        ],
        "slots": ["task", "time", "when"],
    },
    "control_device": {
        "templates": [
            "turn on the {device}",
            "turn off the {device}",
            "switch on the {device}",
            "switch off the {device}",
            "dim the {device}",
            "brighten the {device}",
            "set the {device} to {value}",
            "{device} on",
            "{device} off",
        ],
        "slots": ["device", "state", "value"],
    },
    "cancel": {
        "templates": [
            "cancel that",
            "stop",
            "nevermind",
            "forget it",
            "scrap that",
            "ignore that last one",
            "cancel the timer",
            "cancel the alarm",
            "stop the music",
            "cancel my reminder",
        ],
        "slots": [],
    },
}


# ---------------------------------------------------------------------------
# Slot-value pools.  Mix of common and edge-case values so the model
# learns to handle informal phrasing.
# ---------------------------------------------------------------------------

POOLS: dict[str, list[Any]] = {
    "duration": [
        ("10 minutes", 600),
        ("15 minutes", 900),
        ("30 minutes", 1800),
        ("an hour", 3600),
        ("two hours", 7200),
        ("five minutes", 300),
        ("twenty minutes", 1200),
        ("90 seconds", 90),
        ("a minute", 60),
        ("three minutes", 180),
        ("45 minutes", 2700),
        ("half an hour", 1800),
        ("a quarter of an hour", 900),
        ("seven minutes", 420),
        ("two minutes thirty", 150),
    ],
    "time": [
        "7am", "7:30am", "8am", "9:15am", "noon", "12:30pm",
        "3pm", "5:30pm", "6pm", "10pm", "midnight",
        "half past seven", "quarter to nine", "twenty past eight",
        "0700", "1830", "2100",
    ],
    "recipient": [
        "Mum", "Dad", "Sarah", "Alex", "Sam", "my brother",
        "my sister", "Tom", "Emily", "Jordan", "Mike",
        "the boss", "James", "Priya", "Wei", "Ahmed",
        "my partner", "my flatmate", "Charlie",
    ],
    "message": [
        "I'll be late", "running 10 minutes behind", "see you soon",
        "pick up milk on the way home", "happy birthday",
        "let's grab coffee tomorrow", "I'm on my way",
        "thanks for today", "can you call me back",
        "got the train, see you at 7",
        "is everything ok", "what time should I arrive",
        "the meeting is moved to 4pm",
    ],
    "genre": [
        "jazz", "lo-fi", "classical", "rock", "indie", "pop",
        "techno", "ambient", "hip hop", "country", "metal",
        "soul", "reggae", "blues", "folk", "drum and bass",
        "edm", "k-pop", "afrobeat",
    ],
    "artist": [
        "Taylor Swift", "Radiohead", "The Beatles", "Drake",
        "Beyoncé", "Stromae", "BTS", "Adele", "Daft Punk",
        "Stevie Wonder", "Bach", "Miles Davis", "Bad Bunny",
        "Kendrick Lamar",
    ],
    "place": [
        "the train station", "King's Cross", "Heathrow",
        "the nearest pharmacy", "Tesco", "Pret",
        "Hyde Park", "the British Museum", "Camden Market",
        "Soho", "Greenwich Observatory", "Cambridge",
        "Manchester", "Edinburgh", "the office", "home",
        "Mum's house", "Sarah's place", "the gym",
        "the closest coffee shop", "St Pancras",
    ],
    "task": [
        "buy milk", "call the dentist", "submit the report",
        "water the plants", "take my medication",
        "check the laundry", "feed the cat", "post the parcel",
        "review Sarah's PR", "book a haircut",
        "renew my passport", "pay the council tax",
    ],
    "when": [
        "tonight", "tomorrow morning", "this evening",
        "in an hour", "at lunchtime", "before bed",
        "after work", "first thing tomorrow",
        "next Monday", "this weekend",
    ],
    "device": [
        "lights", "kitchen light", "bedroom light", "living room lights",
        "thermostat", "tv", "fan", "heating",
        "air conditioning", "garage door", "kettle", "coffee machine",
    ],
    "value": [
        "70 percent", "low", "medium", "high",
        "21 degrees", "18 degrees", "max", "minimum",
    ],
}


def _slot_for(template: str, slot_name: str) -> Any:
    """Pick a random concrete value for *slot_name* in *template*."""
    if slot_name == "duration":
        phrase, seconds = random.choice(POOLS["duration"])
        return phrase, seconds
    if slot_name in POOLS:
        return random.choice(POOLS[slot_name])
    return None


def _render_example(action: str) -> dict[str, Any]:
    spec = ACTIONS[action]
    template = random.choice(spec["templates"])
    params: dict[str, Any] = {}
    surface = template

    if "{duration}" in template:
        phrase, seconds = _slot_for(template, "duration")
        surface = surface.replace("{duration}", phrase)
        params["duration_seconds"] = seconds
    for slot in ("time", "recipient", "message", "genre", "artist",
                 "place", "task", "when", "device", "value"):
        token = "{" + slot + "}"
        if token not in template:
            continue
        val = _slot_for(template, slot)
        surface = surface.replace(token, val)
        # Map surface slot name to params key (small renames).
        params_key = {
            "place": "location",
            "task": "task",
            "when": "when",
            "device": "device",
        }.get(slot, slot)
        params[params_key] = val

    # Special-case for control_device: derive on/off from template.
    if action == "control_device":
        if "turn on" in template or "switch on" in template or " on" in template[-3:]:
            params["state"] = "on"
        elif "turn off" in template or "switch off" in template or " off" in template[-4:]:
            params["state"] = "off"
        elif "dim" in template:
            params["state"] = "dim"
        elif "brighten" in template:
            params["state"] = "brighten"

    if action == "call":
        params["video"] = bool(
            "video" in template or "facetime" in template
        )

    return {
        "input": surface,
        "output": {"action": action, "params": params},
    }


def _build_prompt(input_text: str) -> str:
    """The prompt format the model is fine-tuned and inferenced on.

    Compact JSON-only output convention so the model learns to emit
    parseable JSON without prose.
    """
    return (
        "You are an HMI command parser. Convert the user's voice "
        "utterance into a JSON object {\"action\": ..., \"params\": "
        "{...}}. Reply with ONLY the JSON, no prose.\n\n"
        f"Utterance: {input_text}\nJSON:"
    )


def _adversarial_examples() -> list[dict[str, Any]]:
    """Adversarial / edge-case examples to harden the parser.

    Iter 51 (2026-04-27): patterns the model would otherwise blow on:
        * paraphrases with filler words
        * out-of-vocab actions (cancel route)
        * compound utterances (set timer AND text Mum)
        * negation
        * code-mixing
    """
    return [
        # Filler-word noise.
        {"input": "uh, could you, like, set a timer for 10 minutes please",
         "output": {"action": "set_timer",
                    "params": {"duration_seconds": 600}}},
        {"input": "umm so basically text Mum that I'll be late",
         "output": {"action": "send_message",
                    "params": {"recipient": "Mum",
                               "message": "I'll be late"}}},
        # Out-of-vocab action — model should pick "cancel" or refuse cleanly.
        {"input": "do a backflip",
         "output": {"action": "unsupported", "params": {}}},
        {"input": "make me a sandwich",
         "output": {"action": "unsupported", "params": {}}},
        {"input": "summon an Uber",
         "output": {"action": "unsupported", "params": {}}},
        # Negation / cancellation.
        {"input": "actually no, cancel that",
         "output": {"action": "cancel", "params": {}}},
        {"input": "ignore my last command",
         "output": {"action": "cancel", "params": {}}},
        # Code-mixed (English + numerics).
        {"input": "set timer 7m30s",
         "output": {"action": "set_timer",
                    "params": {"duration_seconds": 450}}},
        # Polite / formal register.
        {"input": "would you be so kind as to ring my mother",
         "output": {"action": "call",
                    "params": {"recipient": "my mother",
                               "video": False}}},
        # Two-step compound (model should pick the dominant action).
        {"input": "set a 10 minute timer and text Sarah I'll be late",
         "output": {"action": "set_timer",
                    "params": {"duration_seconds": 600}}},
    ]


def main(n_examples: int = 5000) -> None:
    actions = list(ACTIONS.keys())
    # Stratified by action: equal counts per action, shuffled.
    per_action = n_examples // len(actions)
    examples: list[dict[str, Any]] = []
    for action in actions:
        for _ in range(per_action):
            examples.append(_render_example(action))
    # Fill remainder with random actions.
    while len(examples) < n_examples:
        examples.append(_render_example(random.choice(actions)))
    # Add adversarial cases (replicated 5× so they appear in train).
    adversarials = _adversarial_examples()
    for _ in range(5):
        examples.extend(adversarials)
    random.shuffle(examples)

    # Shape into the SFT format consumed by both training scripts.
    rows = []
    for ex in examples:
        prompt = _build_prompt(ex["input"])
        completion = json.dumps(ex["output"], separators=(",", ":"))
        rows.append({
            "prompt": prompt,
            "completion": completion,
            "input": ex["input"],
            "output": ex["output"],
        })

    # 90/5/5 split.
    n = len(rows)
    train = rows[: int(n * 0.9)]
    val = rows[int(n * 0.9): int(n * 0.95)]
    test = rows[int(n * 0.95):]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split):>5d} rows -> {path}")

    print(f"\nTotal: {n} rows; per-action ≈ {per_action}; "
          f"actions: {len(actions)}")


if __name__ == "__main__":
    main()
