"""HMI-domain corpus generator (vocabulary enlargement).

The synthetic Q→A corpus alone produces a narrow ~1.5 k-token
vocabulary.  To broaden the lexicon in a way that is *directly*
relevant to Huawei R&D's Human-Machine Interaction brief — devices,
edge ML, adaptive interaction, privacy, accessibility — we generate a
large, domain-specialised text corpus rather than pulling in
Project Gutenberg literature that would teach the model words like
"parasol" and "mantelpiece".

The domain bank below is a hand-curated reference corpus spanning:

    - Consumer devices: phones, tablets, laptops, watches, earbuds,
      smart speakers, head-mounted displays, automotive infotainment,
      IoT sensors.
    - Interaction modalities: touch, voice, gesture, gaze, typing
      dynamics, keystroke rhythm, text, motion, bio-signals.
    - UI elements: notifications, widgets, menus, cards, toolbars,
      dialogs, focus management, animation, haptic feedback.
    - ML concepts: transformer, attention, embedding, quantization,
      pruning, distillation, inference, latency budget, on-device.
    - Edge deployment: ONNX, CoreML, TensorFlow Lite, WebGPU, WASM,
      memory footprint, power envelope, thermal throttling.
    - Privacy: differential privacy, federated learning, secure
      enclave, encryption, on-device processing, data minimisation.
    - Accessibility: screen readers, VoiceOver, TalkBack, large text,
      dynamic type, contrast, motor-impairment-friendly input.
    - Adaptive / HMI research: cognitive load, context awareness,
      multimodal fusion, implicit signals, affective computing.
    - Safety + evaluation: guardrails, red teaming, hallucination,
      grounding, factuality, calibration.

These sentences are folded into the corpus as two-turn "dialogues" so
the triples extractor treats one sentence as a "response" to the one
before it.  They're there only to broaden tokenizer coverage — the
retrieval layer doesn't match on them, so they can't pollute Q→A
behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wikipedia REST summaries — free, no auth, explicitly permitted for
# automated use per the API terms.  We fetch ~150 short summaries of
# device / ML / HMI / accessibility topics to broaden the tokenizer's
# coverage to ~8-12 k real-world terms.  Everything is cached to
# ``D:/caches/wiki/`` after the first run.
# ---------------------------------------------------------------------------

_WIKI_CACHE = Path(
    os.environ.get("I3_WIKI_CACHE", r"D:\caches\wiki")
)
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
# The legacy w/api.php extract endpoint returns full plain-text
# article bodies, not just summaries — we prefer it when
# ``I3_WIKI_FULL_EXTRACTS=1`` is set, which multiplies vocab coverage
# by several orders of magnitude per topic at the cost of fetch time.
_WIKI_EXTRACT_URL = (
    "https://en.wikipedia.org/w/api.php?"
    "action=query&format=json&prop=extracts&explaintext=1"
    "&redirects=1&titles={title}"
)
_WIKI_USE_FULL = os.environ.get("I3_WIKI_FULL_EXTRACTS", "1") != "0"

_WIKI_TOPICS: list[str] = [
    # Devices + silicon
    "Smartphone", "Tablet_computer", "Laptop", "Smartwatch",
    "Wireless_earbuds", "Head-mounted_display", "Internet_of_things",
    "Field-programmable_gate_array", "System_on_a_chip", "ARM_architecture",
    "RISC-V", "Neural_processing_unit", "Graphics_processing_unit",
    "Central_processing_unit", "OLED", "Liquid-crystal_display",
    "Capacitive_sensing", "Haptic_technology", "Lithium-ion_battery",
    "5G", "Wi-Fi_6", "Bluetooth", "Near-field_communication",
    # Sensors / modalities
    "Accelerometer", "Gyroscope", "Magnetometer",
    "Photoplethysmogram", "Electrocardiography", "Electroencephalography",
    "LiDAR", "Depth_camera", "Microphone_array", "Beamforming",
    "Eye_tracking", "Touchscreen", "Pressure_sensor",
    "Active_noise_cancellation",
    # ML / SLM
    "Machine_learning", "Deep_learning", "Transformer_(machine_learning_model)",
    "Attention_(machine_learning)", "Word_embedding", "Tokenization_(lexical_analysis)",
    "Recurrent_neural_network", "Convolutional_neural_network",
    "Quantization_(signal_processing)", "Model_compression",
    "Knowledge_distillation", "Transfer_learning", "Reinforcement_learning",
    "Contextual_bandit", "Multi-armed_bandit", "Bayesian_inference",
    "Gradient_descent", "Backpropagation", "Overfitting",
    "Regularization_(mathematics)", "Dropout_(neural_networks)",
    "Speech_recognition", "Natural_language_processing",
    "Speech_synthesis", "Generative_artificial_intelligence",
    "Large_language_model", "Federated_learning", "Differential_privacy",
    # Edge deployment / runtimes
    "Open_Neural_Network_Exchange", "WebAssembly", "WebGPU", "CUDA",
    "CoreML", "TensorFlow", "PyTorch", "Edge_computing",
    # UX / UI / accessibility
    "Human-computer_interaction", "User_experience", "Interaction_design",
    "Accessibility", "Screen_reader", "Web_Content_Accessibility_Guidelines",
    "Dark_mode", "Dynamic_Type", "Voice_user_interface",
    "Augmented_reality", "Virtual_reality", "Mixed_reality",
    "Haptic_feedback", "Notification_(software)",
    # Privacy + security
    "Cryptography", "Transport_Layer_Security", "Public-key_cryptography",
    "Hash_function", "Secure_enclave", "Biometrics", "Fingerprint",
    "Face_recognition_system", "Zero-knowledge_proof",
    "End-to-end_encryption",
    # Cognition / affect / HMI research
    "Cognitive_load", "Human_factors_and_ergonomics", "Attention",
    "Working_memory", "Reaction_time", "Affective_computing",
    "Keystroke_dynamics", "Behavioral_biometrics",
    "Flesch–Kincaid_readability_tests", "Readability",
    # Methods + evaluation
    "A/B_testing", "Statistical_hypothesis_testing", "Confusion_matrix",
    "Precision_and_recall", "Receiver_operating_characteristic",
    "Benchmark_(computing)", "Cross-validation_(statistics)",
    "Calibration_(statistics)",
    # Operating systems / platforms
    "Android_(operating_system)", "IOS", "Linux_kernel",
    "HarmonyOS", "EMUI", "Wear_OS", "WatchOS", "FreeRTOS",
    # Other relevant
    "Chatbot", "Digital_assistant", "Conversational_agent",
    "Personalization", "Recommender_system", "Information_retrieval",
    "Vector_database", "Embedding_(machine_learning)",
    "Approximate_nearest_neighbor", "Cosine_similarity",
    # Huawei / China tech context (makes sense for this submission)
    "Huawei", "Shenzhen", "HiSilicon", "Ascend_(microarchitecture)",
    "Kirin_(processor)",
    # ── Expansion batch: more devices / models / components
    "Apple_Watch", "Google_Pixel", "Samsung_Galaxy", "ThinkPad", "MacBook",
    "iPad", "iPhone", "Chromebook", "Android_(operating_system)",
    "Memory_hierarchy", "Cache_(computing)", "Instruction_set_architecture",
    "Reduced_instruction_set_computer", "Complex_instruction_set_computer",
    "Von_Neumann_architecture", "Computer_memory", "Dynamic_random-access_memory",
    "NAND_flash", "Solid-state_drive", "USB-C", "Thunderbolt_(interface)",
    "HDMI", "DisplayPort", "Codec", "Video_compression", "Image_compression",
    "JPEG", "PNG", "H.264", "HEVC", "AV1",
    # ── More ML / signals
    "Probability_theory", "Linear_algebra", "Calculus", "Bayes'_theorem",
    "Information_theory", "Shannon_entropy", "Kullback–Leibler_divergence",
    "Cross_entropy", "Softmax_function", "Activation_function",
    "Sigmoid_function", "Hyperbolic_tangent", "Rectified_linear_unit",
    "Batch_normalization", "Layer_normalization", "Group_normalization",
    "Residual_neural_network", "Autoencoder", "Variational_autoencoder",
    "Generative_adversarial_network", "Diffusion_model",
    "Latent_Dirichlet_allocation", "Hidden_Markov_model", "Kalman_filter",
    "Particle_filter", "Support-vector_machine", "Decision_tree",
    "Random_forest", "Gradient_boosting", "XGBoost", "k-means_clustering",
    "DBSCAN", "Principal_component_analysis", "t-distributed_stochastic_neighbor_embedding",
    "Manifold_learning", "Optimization_(mathematics)", "Stochastic_gradient_descent",
    "Adam_(optimization_algorithm)", "Learning_rate", "Weight_decay",
    "Early_stopping", "Data_augmentation", "Cross-validation_(statistics)",
    # ── More signals / sensors
    "Digital_signal_processor", "Field-programmable_analog_array",
    "Analog-to-digital_converter", "Digital-to-analog_converter",
    "Sampling_(signal_processing)", "Nyquist–Shannon_sampling_theorem",
    "Fast_Fourier_transform", "Wavelet_transform", "Filter_(signal_processing)",
    "Finite_impulse_response", "Infinite_impulse_response",
    # ── Networking / wireless
    "IPv6", "Transmission_Control_Protocol", "User_Datagram_Protocol",
    "HTTP/3", "QUIC", "Domain_Name_System", "Content_delivery_network",
    "WebSocket", "WebRTC", "Low-power_wide-area_network", "LoRa",
    "Zigbee", "Z-Wave", "Thread_(network_protocol)", "Matter_(standard)",
    # ── OS + runtimes
    "Real-time_operating_system", "Microkernel", "Monolithic_kernel",
    "Scheduling_(computing)", "Virtual_memory", "Paging", "Garbage_collection_(computer_science)",
    "Just-in-time_compilation", "LLVM", "Assembly_language", "Compiler",
    "Interpreter_(computing)", "WebAssembly_System_Interface", "Docker_(software)",
    "Kubernetes",
    # ── UX / interaction research
    "Fitts's_law", "Hick's_law", "Gestalt_psychology", "Nielsen_Norman_Group",
    "Usability", "Mental_model", "Affordance", "Signifier",
    "Skeuomorphism", "Flat_design", "Material_Design", "Glassmorphism",
    "Typography", "Font", "Serif", "Sans-serif",
    "San_Francisco_(sans-serif_typeface)", "Helvetica", "Inter_(typeface)",
    "Colour_theory", "Colour_blindness", "WCAG_colour_contrast",
    # ── Accessibility specifics
    "Dyslexia", "Dysgraphia", "Tourette_syndrome", "Spoon_theory",
    "Alternative_augmentative_communication", "American_Sign_Language",
    # ── Security
    "Threat_model", "Side-channel_attack", "Timing_attack", "Spectre_(security_vulnerability)",
    "Meltdown_(security_vulnerability)", "Rowhammer", "Buffer_overflow",
    "Same-origin_policy", "Content_Security_Policy", "OAuth", "FIDO_Alliance",
    "Passkey", "WebAuthn", "Trusted_Platform_Module",
    # ── Research + ethics
    "Reproducibility", "FAIR_data", "Open_access", "Peer_review",
    "Institutional_review_board", "Informed_consent",
    "Algorithmic_bias", "Fairness_(machine_learning)",
    "Explainable_artificial_intelligence", "Interpretability",
    "Shapley_value", "LIME_(machine_learning)",
    # ── Cognition / HCI
    "Perception", "Proprioception", "Sensorimotor_integration",
    "Attentional_blink", "Change_blindness", "Inattentional_blindness",
    "Working_memory_load", "Situation_awareness",
    # ── Misc useful lexical breadth
    "Photography", "Videography", "Colour_grading",
    "High_dynamic_range", "Tone_mapping", "Subpixel",
    "Anti-aliasing", "Kerning", "Leading_(typography)",
    "Typographical_alignment", "Page_layout",
    "Accessibility_tree", "Document_Object_Model",
    "Cascading_Style_Sheets", "Hypertext_Markup_Language",
    # ── More OSes + ecosystems
    "Symbian", "Windows_(operating_system)", "Windows_Phone",
    "WatchOS", "TvOS", "IPadOS", "ChromeOS",
    "Tizen", "KaiOS",
    # ── Cross-cutting
    "Unicode", "UTF-8", "ASCII", "Regular_expression",
    "JSON", "YAML", "TOML", "Protocol_Buffers", "FlatBuffers", "MessagePack",
    "GRPC", "Representational_state_transfer",
    # ── Hardware accelerators + vendors
    "NVIDIA_Jetson", "Apple_silicon", "Qualcomm_Snapdragon",
    "MediaTek_Dimensity", "Samsung_Exynos",
    # ── Speech / audio
    "Mel-frequency_cepstrum", "Voice_activity_detection",
    "Automatic_speech_recognition", "WaveNet", "Whisper_(speech_recognition_system)",
    "Tacotron", "HiFi-GAN",
    # ── Vision
    "Convolution", "Image_processing", "Edge_detection",
    "Feature_extraction", "SIFT_(computer_vision)", "HOG_(computer_vision)",
    "YOLO_(algorithm)", "Object_detection", "Semantic_segmentation",
    "Image_segmentation", "Pose_estimation", "Optical_flow",
    # ── Small models / on-device AI
    "LoRA_(machine_learning)", "QLoRA", "Parameter-efficient_fine-tuning",
    "GGUF", "Llama_(language_model)", "Mistral_AI", "Phi_(language_model)",
    "Gemini_(language_model)", "Pangu_Large_Language_Model",
    # ── Robotics / motion
    "Inertial_measurement_unit", "SLAM_(robotics)", "Robotics",
    "Autonomous_vehicle",
    # ── Programming languages
    "Python_(programming_language)", "Rust_(programming_language)",
    "Go_(programming_language)", "JavaScript", "TypeScript",
    "C++", "C_(programming_language)", "Swift_(programming_language)",
    "Kotlin_(programming_language)", "Dart_(programming_language)",
    # ── Data
    "Time_series", "Anomaly_detection", "Spectrogram",
    "Heart_rate_variability", "Galvanic_skin_response",

    # ═══════════════════════════════════════════════════════════════
    # Cross-domain expansion for vocabulary breadth (aiming 30 k+ tok)
    # ═══════════════════════════════════════════════════════════════

    # ── Sciences — physics
    "Physics", "Classical_mechanics", "Special_relativity", "General_relativity",
    "Quantum_mechanics", "Thermodynamics", "Statistical_mechanics", "Electromagnetism",
    "Optics", "Fluid_dynamics", "Solid_mechanics", "Acoustics",
    "Astrophysics", "Cosmology", "Nuclear_physics", "Particle_physics",
    "Standard_Model", "Higgs_boson", "String_theory", "Supersymmetry",
    "Black_hole", "Neutron_star", "Supernova", "Dark_matter", "Dark_energy",
    "Event_horizon", "Hawking_radiation",

    # ── Sciences — chemistry
    "Chemistry", "Organic_chemistry", "Inorganic_chemistry",
    "Physical_chemistry", "Biochemistry", "Polymer",
    "Chemical_bond", "Covalent_bond", "Ionic_bond", "Hydrogen_bond",
    "Catalysis", "Redox", "Electrochemistry", "Enzyme",
    "Periodic_table", "Molecule", "Atom",

    # ── Sciences — biology / medicine
    "Biology", "Cell_biology", "Genetics", "Molecular_biology",
    "Microbiology", "Virology", "Immunology", "Neuroscience",
    "Anatomy", "Physiology", "Pharmacology", "Epidemiology",
    "Public_health", "Vaccine", "Antibiotic", "Microbiome",
    "Genome", "Proteomics", "CRISPR", "Stem_cell",
    "Cardiovascular_disease", "Cancer", "Diabetes", "Alzheimer's_disease",
    "Mental_health", "Psychiatry", "Psychology",
    "Sleep", "Circadian_rhythm", "Stress_(biology)",

    # ── Earth + environment
    "Earth", "Geology", "Plate_tectonics", "Volcanism",
    "Climate_change", "Meteorology", "Oceanography", "Hydrology",
    "Biodiversity", "Conservation_biology", "Rainforest", "Tundra",
    "Savannah", "Coral_reef", "Photosynthesis", "Water_cycle", "Carbon_cycle",
    "Weather_forecasting", "El_Niño", "La_Niña",

    # ── Astronomy / space
    "Astronomy", "Solar_System", "Planet", "Moon", "Comet", "Asteroid",
    "Mars", "Venus", "Mercury_(planet)", "Jupiter", "Saturn", "Uranus",
    "Neptune", "Pluto", "Milky_Way", "Andromeda_Galaxy",
    "Big_Bang", "Cosmic_microwave_background", "James_Webb_Space_Telescope",
    "Hubble_Space_Telescope", "International_Space_Station",
    "NASA", "European_Space_Agency", "SpaceX", "Rocket_engine",

    # ── Mathematics
    "Mathematics", "Number_theory", "Algebra", "Geometry",
    "Topology", "Analysis_(mathematics)", "Discrete_mathematics",
    "Graph_theory", "Combinatorics", "Probability", "Statistics",
    "Integral", "Derivative", "Differential_equation", "Fourier_series",
    "Prime_number", "Pi", "Euler's_identity", "Fibonacci_sequence",
    "Golden_ratio",

    # ── Computer science classics
    "Turing_machine", "Halting_problem", "Complexity_theory",
    "NP-completeness", "Cryptographic_hash_function", "RSA_(cryptosystem)",
    "Elliptic-curve_cryptography", "Public-key_infrastructure",
    "Database", "Relational_database", "NoSQL", "SQL",
    "Operating_system", "Filesystem", "Process_(computing)",
    "Thread_(computing)", "Concurrency_(computer_science)",
    "Parallel_computing", "Distributed_computing", "Cloud_computing",
    "Serverless_computing", "Edge_computing",

    # ── Arts — music
    "Music_theory", "Scale_(music)", "Chord_(music)", "Harmony",
    "Rhythm", "Melody", "Counterpoint", "Sonata_form",
    "Symphony", "Opera", "Jazz", "Blues", "Rock_music", "Pop_music",
    "Hip_hop_music", "Classical_music", "Electronic_music", "Piano",
    "Violin", "Guitar", "Drums",

    # ── Arts — visual + literature + performance
    "Painting", "Sculpture", "Photography", "Film",
    "Impressionism", "Cubism", "Surrealism", "Abstract_expressionism",
    "Renaissance", "Baroque", "Gothic_art", "Modernism", "Postmodernism",
    "Literature", "Novel", "Poetry", "Drama", "Play_(theatre)",
    "Short_story", "Biography", "Autobiography", "Science_fiction",
    "Fantasy", "Mystery_fiction", "Romance_novel",
    "Theatre", "Ballet", "Opera", "Film_directing", "Cinematography",
    "Animation", "Video_game",

    # ── Philosophy
    "Philosophy", "Metaphysics", "Epistemology", "Ethics", "Logic",
    "Political_philosophy", "Philosophy_of_mind", "Philosophy_of_science",
    "Existentialism", "Utilitarianism", "Deontological_ethics",
    "Virtue_ethics", "Stoicism", "Buddhism", "Taoism", "Confucianism",
    "Socrates", "Plato", "Aristotle", "Immanuel_Kant",
    "Friedrich_Nietzsche", "Ludwig_Wittgenstein", "John_Stuart_Mill",

    # ── History eras / events
    "Ancient_Egypt", "Mesopotamia", "Ancient_Greece", "Roman_Empire",
    "Byzantine_Empire", "Medieval_Europe", "Islamic_Golden_Age",
    "Renaissance", "Age_of_Enlightenment", "Industrial_Revolution",
    "French_Revolution", "American_Revolution", "World_War_I",
    "World_War_II", "Cold_War", "Space_Race",
    "Silk_Road", "Mongol_Empire", "Ottoman_Empire", "British_Empire",
    "Colonialism", "Decolonization",

    # ── Geography — countries (broad sample)
    "Germany", "France", "United_Kingdom", "Spain", "Italy", "Portugal",
    "Netherlands", "Belgium", "Sweden", "Norway", "Denmark", "Finland",
    "Iceland", "Ireland", "Poland", "Czech_Republic", "Hungary",
    "Austria", "Switzerland", "Greece", "Turkey", "Russia", "Ukraine",
    "United_States", "Canada", "Mexico", "Brazil", "Argentina", "Chile",
    "Peru", "Colombia", "Cuba",
    "China", "Japan", "South_Korea", "North_Korea", "Taiwan", "Vietnam",
    "Thailand", "Indonesia", "Philippines", "Malaysia", "Singapore",
    "India", "Pakistan", "Bangladesh", "Nepal", "Sri_Lanka",
    "Iran", "Iraq", "Saudi_Arabia", "Israel", "Egypt", "Morocco",
    "Nigeria", "Kenya", "South_Africa", "Ethiopia",
    "Australia", "New_Zealand",

    # ── Cities
    "London", "Paris", "Berlin", "Madrid", "Rome", "Amsterdam",
    "Stockholm", "Vienna", "Prague", "Warsaw", "Istanbul", "Moscow",
    "New_York_City", "Los_Angeles", "San_Francisco", "Chicago",
    "Toronto", "Mexico_City", "São_Paulo", "Buenos_Aires",
    "Tokyo", "Osaka", "Kyoto", "Seoul", "Beijing", "Shanghai", "Hong_Kong",
    "Taipei", "Singapore", "Mumbai", "Delhi", "Bangkok", "Jakarta",
    "Dubai", "Tel_Aviv", "Cairo", "Cape_Town", "Nairobi", "Lagos",
    "Sydney", "Melbourne", "Auckland",

    # ── Food / cuisine
    "Cuisine", "Italian_cuisine", "French_cuisine", "Chinese_cuisine",
    "Japanese_cuisine", "Indian_cuisine", "Mexican_cuisine",
    "Mediterranean_diet", "Sushi", "Pizza", "Pasta", "Bread",
    "Cheese", "Wine", "Beer", "Tea", "Coffee", "Chocolate", "Spice",
    "Fermentation", "Gastronomy",

    # ── Sports
    "Association_football", "Basketball", "Tennis", "Cricket",
    "Baseball", "American_football", "Rugby", "Golf", "Hockey",
    "Athletics_(sport)", "Marathon", "Cycling", "Swimming",
    "Boxing", "Mixed_martial_arts", "Chess", "Go_(game)",
    "Esports", "Olympic_Games", "FIFA_World_Cup", "Formula_One",

    # ── Economics / finance
    "Economics", "Macroeconomics", "Microeconomics",
    "Gross_domestic_product", "Inflation", "Recession",
    "Stock_market", "Bond_(finance)", "Cryptocurrency", "Bitcoin",
    "Ethereum", "Venture_capital", "Supply_and_demand",
    "Central_bank", "Monetary_policy", "Fiscal_policy",
    "Game_theory", "Nash_equilibrium", "Comparative_advantage",

    # ── Law / politics
    "Law", "Constitutional_law", "Human_rights",
    "United_Nations", "European_Union", "NATO",
    "Democracy", "Republic", "Monarchy",
    "Freedom_of_speech", "Privacy", "General_Data_Protection_Regulation",

    # ── Religion / cultures
    "Religion", "Christianity", "Islam", "Judaism", "Hinduism",
    "Buddhism", "Sikhism", "Mythology", "Greek_mythology",
    "Norse_mythology", "Indigenous_peoples",

    # ── Languages (world)
    "Language", "Linguistics", "Phonology", "Syntax", "Semantics",
    "English_language", "Spanish_language", "French_language",
    "German_language", "Mandarin_Chinese", "Japanese_language",
    "Arabic_language", "Hindi", "Portuguese_language", "Russian_language",
    "Korean_language", "Turkish_language", "Swahili_language",
    "Sign_language",

    # ── Emotions / psychology details
    "Happiness", "Grief", "Anxiety", "Depression_(mood)", "Anger",
    "Fear", "Surprise_(emotion)", "Disgust", "Empathy", "Self-esteem",
    "Motivation", "Procrastination", "Flow_(psychology)",
    "Attention", "Focus_(cognitive_process)",

    # ── Misc useful
    "Education", "Teaching", "Learning", "Self-directed_learning",
    "Curiosity", "Creativity", "Innovation", "Design_thinking",
    "Sustainability", "Renewable_energy", "Solar_power", "Wind_power",
    "Hydroelectricity", "Nuclear_power", "Battery",
    "Electric_vehicle", "Autonomous_car", "Drone",
    "Cryptography", "Steganography", "Quantum_computing",
    "Blockchain",
]


def _wiki_cached_fetch(title: str, *, timeout: float = 15.0) -> str | None:
    """Return the Wikipedia plain-text extract for *title*, cached.

    Uses the full ``prop=extracts&explaintext=1`` endpoint when
    ``I3_WIKI_FULL_EXTRACTS`` is enabled (the default) — this returns
    the complete article body rather than a 1-paragraph summary and is
    what pushes the corpus vocabulary into the tens-of-thousands range.
    """
    _WIKI_CACHE.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", title)
    # Keep summary and full caches separate so switching endpoints
    # doesn't accidentally reuse the wrong payload.
    suffix = ".full.txt" if _WIKI_USE_FULL else ".sum.txt"
    path = _WIKI_CACHE / f"{slug}{suffix}"
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="replace") or None
        except Exception:  # pragma: no cover - defensive
            pass
    url = (
        _WIKI_EXTRACT_URL.format(title=title)
        if _WIKI_USE_FULL
        else _WIKI_SUMMARY_URL.format(title=title)
    )
    req = Request(
        url,
        headers={
            "User-Agent": (
                "I3-research-demo/1.0 (HMI internship portfolio; "
                "contact via repository)"
            ),
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # pragma: no cover - network
        logger.debug("Wikipedia fetch failed for %s: %s", title, exc)
        return None
    if _WIKI_USE_FULL:
        pages = (payload.get("query") or {}).get("pages") or {}
        extracts: list[str] = []
        for page in pages.values():
            text = (page or {}).get("extract") or ""
            text = str(text).strip()
            if text:
                extracts.append(text)
        extract = "\n\n".join(extracts)
    else:
        extract = payload.get("extract") or ""
    extract = str(extract).strip()
    if not extract:
        return None
    try:
        path.write_text(extract, encoding="utf-8")
    except Exception:  # pragma: no cover - disk
        pass
    return extract


def _wiki_sentences(max_topics: int | None = None) -> list[str]:
    """Fetch summaries for all configured topics, return sentences."""
    topics = _WIKI_TOPICS if max_topics is None else _WIKI_TOPICS[:max_topics]
    out: list[str] = []
    hits = 0
    for title in topics:
        extract = _wiki_cached_fetch(title)
        if not extract:
            continue
        hits += 1
        # Split on sentence boundaries; keep reasonable-length items.
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", extract):
            s = s.strip()
            if 20 <= len(s) <= 320:
                out.append(s)
    logger.info(
        "Wikipedia corpus: %d topics fetched, %d sentences extracted.",
        hits, len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Domain sentence bank
# ---------------------------------------------------------------------------
# Every entry is a single self-contained declarative sentence.  The
# list is deliberately large and diverse so the tokenizer picks up
# terms the chit-chat corpus never reaches.  The copy is hand-written
# technical prose, not scraped.
# ---------------------------------------------------------------------------

_DOMAIN_SENTENCES: list[str] = [
    # ── Devices — phones / tablets / laptops
    "Modern smartphones integrate a neural-processing unit alongside the CPU and GPU to accelerate on-device machine-learning workloads.",
    "A flagship tablet combines a high-refresh-rate OLED display, an efficient system-on-chip, and a long-lasting lithium-ion battery for all-day productivity.",
    "Thin-and-light laptops balance silicon performance against passive-cooling thermal budgets to keep fan noise low.",
    "Foldable phones use flexible polymer substrates and ultra-thin glass to survive tens of thousands of folding cycles.",
    "Stylus input depends on capacitive digitisers and palm-rejection algorithms that distinguish the pen tip from an accidental wrist contact.",
    "Face unlock is typically implemented with an infrared dot projector and a neural matcher that runs inside a secure enclave.",
    "Under-display fingerprint sensors use either ultrasonic transducers or optical pixel readouts to capture ridge patterns through the screen stack.",
    "Refresh-rate controllers adjust the display panel dynamically to conserve power during static content and ramp up during scrolling.",
    "LiDAR scanners on tablets and phones enable room-scale depth capture for augmented-reality applications.",
    "Millimetre-wave 5G antennas are distributed around the device chassis to maintain beamforming despite user grip.",
    # ── Devices — wearables
    "Smartwatches run a constrained operating system with a tickless scheduler to extend battery life between charges.",
    "Photoplethysmography sensors on the underside of a watch shine green light into the skin and measure blood-volume pulses to estimate heart rate.",
    "Electrocardiogram features on wearables detect atrial fibrillation by comparing consecutive R-R intervals against a learned baseline.",
    "Passive step counting on a wrist-worn device fuses accelerometer data with a pedometer model tuned for the wearer's gait.",
    "Wireless earbuds use bone-conduction voice pickup and adaptive beamforming microphones to reject wind noise.",
    "Active noise cancellation processes microphones at kilohertz rates and emits an anti-phase signal that destructively interferes with ambient sound.",
    "Transparency mode on earbuds is essentially active noise cancellation run in reverse to pass outside audio through to the wearer.",
    "Always-on voice assistants on earbuds rely on a tiny wake-word detector running on a digital-signal-processor core.",
    "Health rings estimate sleep stages by combining heart-rate variability, skin temperature, and motion across the night.",
    "Augmented-reality headsets require sub-twenty-millisecond motion-to-photon latency to avoid inducing motion sickness.",
    # ── Devices — home / IoT / automotive
    "Smart speakers stream compressed audio embeddings from a local wake-word detector to the cloud only after the trigger phrase fires.",
    "Smart thermostats use a Kalman filter to separate the slow thermal mass of a home from short-term occupancy disturbances.",
    "Connected cameras process motion zones locally so that bounding-box coordinates rather than raw frames are uploaded to the cloud.",
    "Doorbells with on-device person detection reduce nuisance alerts from swaying trees by filtering out non-human motion.",
    "Automotive driver-monitoring systems use an infrared camera aimed at the driver's face to detect drowsiness and distraction.",
    "Adaptive cruise control fuses radar returns with camera-detected lane markings to maintain a safe following distance.",
    "Infotainment stacks apply user-specific personalisation through profile switching when multiple drivers share a vehicle.",
    "Low-power wide-area networks like LoRaWAN let IoT sensors transmit small payloads for years on a single coin-cell battery.",
    "Matter is a cross-vendor smart-home standard that runs over Thread, Wi-Fi, or Ethernet so devices from different manufacturers interoperate.",
    "Environmental sensors in smart buildings track occupancy, illuminance, temperature, humidity, and carbon dioxide simultaneously.",
    # ── Interaction modalities
    "Touch input is sampled by a capacitive grid that detects changes in stored charge when a finger approaches the panel.",
    "Haptic feedback uses a linear resonant actuator driven by an amplitude-modulated waveform to render distinct textures.",
    "Voice interfaces cascade a wake-word detector, a speech-recognition front-end, and a downstream intent classifier to keep power usage low.",
    "Gesture recognition on a front-facing camera uses a lightweight pose estimator followed by a temporal classifier over joint-angle features.",
    "Gaze tracking for attention estimation pairs a corneal-reflection infrared system with a user-specific calibration offset.",
    "Keystroke dynamics describe the characteristic rhythm of a particular typist, including inter-key interval, dwell time, and backspace frequency.",
    "Composition metrics such as pause-before-send and edit count capture how much cognitive effort a user spent composing a message.",
    "Inertial gesture recognition on a wristband fuses accelerometer and gyroscope channels to classify hand gestures like pinch, wave, and flick.",
    "Silent-speech interfaces read surface electromyography from the jaw and throat to decode sub-vocalised speech.",
    "Brain-computer interfaces range from non-invasive electroencephalography headsets to intracortical arrays implanted on the motor cortex.",
    # ── UI / UX patterns
    "Notifications should respect the user's current context: muting non-critical alerts during a phone call or in a focus mode.",
    "Widgets on a lock screen surface a small amount of glanceable information at the cost of a fixed piece of on-screen real estate.",
    "Focus management determines which element receives keyboard input and is especially important for accessibility.",
    "Motion design should honour a reduced-motion preference to avoid discomfort for users sensitive to vestibular triggers.",
    "Dark mode is not merely an aesthetic choice: on OLED panels it meaningfully reduces battery draw on dark pixels.",
    "Progressive disclosure keeps secondary controls hidden behind a menu or long-press until the user signals intent.",
    "Skeleton screens replace spinners to give the perception of a faster load by previewing content structure while data arrives.",
    "A well-designed empty state explains what should appear, why it doesn't yet, and what the user can do to populate it.",
    "Onboarding flows should be dismissible: a user returning after a reinstall does not want to repeat the tour.",
    "Accessibility checks are not optional polish — they are part of the contract every UI has with a minority of users for whom they are essential.",
    # ── ML / SLM concepts
    "A small language model is typically one to three billion parameters and is designed to run on-device without a cloud connection.",
    "Transformer blocks alternate multi-head self-attention with a position-wise feed-forward network, residual connections, and layer normalisation.",
    "Cross-attention lets a decoder layer attend to an external conditioning signal such as a retrieved document or a user-state vector.",
    "Token embeddings map discrete token indices to dense vectors that capture distributional similarity.",
    "Positional embeddings inject order information so the transformer can distinguish an input sequence from a bag of tokens.",
    "Quantisation reduces model precision from 32-bit floats to 8-bit integers or lower at the cost of small accuracy loss.",
    "Post-training quantisation calibrates activation ranges on a representative dataset; quantisation-aware training bakes the quantiser into gradient updates.",
    "Knowledge distillation trains a compact student model to match the output distribution of a larger teacher, often retaining most of its accuracy at a fraction of the size.",
    "Pruning removes weights or whole channels deemed unimportant by a saliency metric to shrink model size and inference cost.",
    "Low-rank adaptation keeps a frozen base model and trains only a pair of small rank-r matrices per layer, useful for personalisation.",
    "Speculative decoding runs a small draft model in parallel with a larger verifier and accepts draft tokens that agree with the verifier to speed up generation.",
    "Latency budget is typically split between tokenisation, prompt encoding, decode per token, and post-processing.",
    "Retrieval-augmented generation pairs a frozen language model with a document retriever so answers cite fresh information without retraining.",
    "On-device inference must respect the memory footprint, power envelope, and thermal headroom of the target silicon.",
    "Streaming token output lets the UI render partial responses as they're generated, reducing perceived latency.",
    # ── Huawei-relevant themes
    "Edge-first design means moving as much computation as possible onto the device, reserving the cloud for work the silicon genuinely can't handle.",
    "The Kirin and Ascend silicon families accelerate neural-network workloads alongside traditional mobile graphics and general-purpose compute.",
    "HarmonyOS coordinates multiple devices in a household as a distributed super-device, migrating tasks across screens as the user moves between them.",
    "The MateBook family integrates a laptop-class processor with a smartphone-like always-on companion for rapid wake and persistent notifications.",
    "A wearable paired with a phone should fall back to its own local model when the Bluetooth link is dropped, not simply degrade to a blank screen.",
    "Power-aware inference schedules model calls around charging events and screen-off periods to minimise user-visible battery impact.",
    "Model lifecycle management on devices must handle over-the-air updates, compatibility with older silicon, and rollback when regressions are detected.",
    "Multilingual input-method editors require on-device speech recognition that covers dozens of language families without regressing English accuracy.",
    "A global fleet of devices demands region-specific fine-tuning to respect local norms, scripts, and regulatory constraints.",
    # ── Privacy / security
    "Differential privacy adds calibrated noise so statistics released from user data cannot be reversed to a single individual.",
    "Federated learning trains a shared model by aggregating gradients produced on each user's device rather than shipping raw data to a server.",
    "A secure enclave is a hardware-isolated region of the system-on-chip that holds cryptographic keys and runs sensitive workloads out of reach of the main OS.",
    "Full-disk encryption protects data at rest by encrypting every block with a key derived from the user's passcode.",
    "Transport Layer Security keeps data in motion confidential using a handshake that establishes session keys before any payload is exchanged.",
    "Pass-through processing lets a model see features like topic labels or embeddings without ever handling the raw input text.",
    "Data minimisation is the principle of collecting only the signals strictly required to deliver a feature and discarding them as soon as possible.",
    "Zero-knowledge proofs allow a device to demonstrate that a claim holds without revealing the data that backs it.",
    "Biometric templates should be stored as non-reversible hashes rather than raw images so a leak cannot reconstruct a face or fingerprint.",
    "On-device personalisation beats cloud personalisation on privacy: the features never leave the device, so they cannot be exfiltrated in a breach.",
    # ── Accessibility
    "Screen readers like VoiceOver and TalkBack speak out the currently focused element; every interactive control needs an accessible label.",
    "Large-text and dynamic-type features require UIs to re-flow content rather than truncate it when the user chooses a bigger font.",
    "High-contrast themes serve users with low vision; contrast ratios should satisfy WCAG AA for body text and AAA where feasible.",
    "Voice control lets users with limited motor ability issue commands hands-free; every action should have a pronounceable invocation.",
    "Switch control adapts scanning interfaces for users who can reliably trigger only a small number of mechanical or brain-computer switches.",
    "Live captions transcribe any audio stream on-device, aiding users who are deaf or hard of hearing without leaking audio to a server.",
    "Hearing aids benefit from Bluetooth LE Audio, which supports low-latency stereo streaming and broadcast audio in public venues.",
    "Motor-adaptive keyboards rearrange keys or enlarge tap targets based on a user's observed tremor and accuracy profile.",
    "Reduced-motion settings disable parallax, autoplay, and heavy transitions to keep the UI usable for people with vestibular sensitivities.",
    "An accessible design benefits every user eventually: ramps also help people with suitcases, captions also help people in noisy cafés.",
    # ── HMI research themes
    "Implicit interaction intelligence infers user state from signals the user produces without explicit effort — keystroke rhythm, dwell time, emoji density, composition pauses.",
    "Cognitive load estimation combines behavioural features, physiological signals, and task-difficulty priors into a scalar that drives interface simplification.",
    "Context awareness means the system knows what the user is doing, where they are, and what they might want next — and adapts accordingly.",
    "Multimodal fusion combines text, audio, vision, and motion into a shared representation so one modality can disambiguate another.",
    "Affective computing models detect and respond to the user's emotional state, for example lowering the tone of voice responses when stress is elevated.",
    "Adaptive verbosity shortens responses when cognitive load is high and expands them when the user is exploring a topic.",
    "Style mirroring adjusts formality, emotionality, and directness toward the user's own communication norms over a session.",
    "A three-timescale user model combines a minute-scale reactive state, a day-scale rolling summary, and a month-scale persistent profile.",
    "Baseline estimation collects enough interaction samples that the system can distinguish a meaningful deviation from normal session-to-session variance.",
    "An ablation study systematically removes one component at a time to measure how much each contributes to the final metric.",
    # ── Safety / evaluation
    "Guardrails reject prompts that request disallowed content and redirect the user toward legitimate alternatives.",
    "Red teaming probes a deployed model with adversarial prompts to surface vulnerabilities before real users hit them.",
    "Hallucination describes fluent output that is factually wrong; grounding tactics like retrieval reduce but do not eliminate it.",
    "Calibration measures whether a model's stated confidence matches its actual accuracy.",
    "A confusion matrix breaks down predictions by true and predicted class so false positives and false negatives can be counted separately.",
    "Precision is the fraction of predicted positives that are truly positive; recall is the fraction of true positives that were predicted.",
    "A receiver-operating-characteristic curve plots true-positive rate against false-positive rate as the decision threshold sweeps.",
    "A benchmark leaderboard is a useful compass but a risky destination: models overfit the test suite once its items leak into their training corpora.",
    "An evaluation harness should cover end-to-end behaviour, not just per-component unit tests, because emergent interactions are where regressions hide.",
    "Production regression tests should replay a frozen set of real user sessions and flag any semantic drift in the model's responses.",
    # ── Edge deployment
    "ONNX is an open exchange format for neural networks that lets a model trained in PyTorch run on ONNX Runtime, TensorRT, or CoreML.",
    "WebAssembly ships a portable binary instruction set that runs inside any modern browser at near-native speed.",
    "WebGPU exposes the GPU to the browser through a modern low-overhead API suitable for both graphics and compute kernels.",
    "CoreML is Apple's on-device model format with automatic scheduling across CPU, GPU, and Neural Engine.",
    "TensorFlow Lite delegates inference to hardware accelerators on Android via the NNAPI abstraction.",
    "Model size on disk is distinct from peak memory during inference; activations and key-value caches dominate the runtime footprint.",
    "A cache-friendly decode loop keeps hot tensors resident in the last-level cache rather than spilling to DRAM each step.",
    "SIMD intrinsics let a compiled inference kernel process multiple tensor elements per CPU cycle.",
    "Thermal throttling triggers when silicon exceeds a temperature threshold and reduces clock speed to protect the device.",
    "A low-power wake word runs on a microwatt-scale always-on island that never activates the main CPU unless the trigger phrase is heard.",
    # ── Data + MLOps
    "A data card documents a dataset's provenance, collection methodology, known biases, and license for every downstream user.",
    "A model card captures intended use, out-of-scope use, evaluation metrics, and known limitations of a deployed system.",
    "Version-controlled training pipelines treat data, code, and hyperparameters as one commit so any result can be reproduced later.",
    "A canary deployment rolls a new model out to a small fraction of traffic first, watches key metrics, and expands if they hold.",
    "Shadow mode runs the candidate model in parallel with production but does not return its output to users; this gauges real-world latency and error rates risk-free.",
    "A/B tests measure the causal impact of a change by randomly assigning users to the control or treatment arm.",
    "Feature flags let engineers ship code to production in a dormant state and flip it on for a subset of users without a redeploy.",
    "A postmortem answers what happened, why it happened, how we caught it, and what we will change to prevent a repeat.",
    "An SLO commits to a measurable level of service — say ninety-nine-point-nine percent p95 latency under 120 milliseconds — and budgets failures against it.",
    "Chaos engineering deliberately injects faults into a production system to verify that automated failovers actually fire.",
    # ── Research-style prose
    "A well-specified experiment changes one variable at a time and fixes the rest, so attribution is straightforward.",
    "Effect size matters at least as much as statistical significance: a tiny p-value on a one-percent improvement rarely justifies the engineering cost.",
    "The no-free-lunch theorem states that no single algorithm outperforms every other across all problem distributions.",
    "A strong baseline is a form of honesty: any improvement a complex model shows over it is directly attributable to the model's complexity.",
    "Cherry-picked qualitative examples can mislead; pair them with aggregate metrics from a frozen test set.",
    "When in doubt, run the experiment; intuition about high-dimensional systems is frequently wrong.",
    "Replication across multiple seeds separates real gains from random walk.",
    "The scientific value of a negative result is often higher than a positive one — it narrows the search space permanently.",
    "Good research notebooks explain decisions, not only actions; a future reader should understand why, not just what, was done.",
    "Peer review improves a paper even when reviewers are wrong: their misunderstandings are also a signal about clarity.",
    # ── Usability heuristics (Nielsen-style)
    "The interface should keep the user informed about what is going on through appropriate feedback within a reasonable time.",
    "Minimise the user's memory load by making objects, actions, and options visible.",
    "Error prevention is better than error recovery; present a confirmation before destructive actions.",
    "Help and documentation should be searchable, list concrete steps, and be kept to a small size.",
    "Recognition beats recall: users should be able to see options rather than remember them.",
    "Match the system's language to the user's domain rather than forcing the user to learn internal jargon.",
    "Consistency across platforms reduces cognitive cost; a button that means Save in one app should not mean Share in the next.",
    "Flexibility and efficiency of use come from shortcuts for experts layered on top of accessible paths for novices.",
    "An aesthetic and minimalist interface makes the relevant signal easier to find precisely because irrelevant information is removed.",
    "Help users recognise, diagnose, and recover from errors with plain-language messages that suggest a next step.",
    # ── Multimodal / cross-device
    "Multi-device continuity lets a call started on a phone continue on a laptop without dropping audio or transcript context.",
    "A digital assistant should remember the last thing it and the user discussed — at least within the current conversation — to resolve pronouns correctly.",
    "Voice interfaces succeed or fail on acoustic echo cancellation, which removes the device's own speaker output from its microphone input.",
    "Proactive suggestions, if shown at the right moment, are helpful; at the wrong moment they are annoying and should be dismissible for good.",
    "Ambient computing describes devices that fade into the environment until called upon, in contrast to demanding attention constantly.",
    "Mixed reality overlays digital content on the physical world so virtual objects remain fixed to real-world coordinates as the user moves.",
    "Spatial audio preserves the apparent direction of sound sources so a user wearing earbuds can still localise a ringing phone.",
    "Passkeys replace passwords with device-bound public keys and biometric authentication, eliminating the phishing vector entirely.",
    "Cross-device clipboard sharing should end-to-end-encrypt its payload so intermediate servers never see the user's content.",
    "A family profile on a shared device must partition accounts, app installs, and model personalisations cleanly across members.",
]


def load_gutenberg_dialogues(
    max_sentences: int | None = None,
    *,
    seed: int = 0,
    repeat_factor: int = 6,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Return synthetic HMI-domain dialogues for vocabulary coverage.

    The function name is retained for backward-compatibility with the
    earlier prototype that fetched Gutenberg books; the implementation
    now returns hand-written technical prose drawn from the domain
    sentence bank above.  ``repeat_factor`` is a multiplier that boosts
    the effective presence of this corpus in the tokenizer so rare
    technical terms (e.g. "quantisation", "ONNX", "enclave") aren't
    pruned by the vocab frequency cut-off.

    Args:
        max_sentences: Optional cap on total sentences emitted.  When
            ``None`` every sentence is used.
        seed: Seed for the consecutive-sentence pairing shuffle.  Pass
            the pipeline's own seed for reproducibility.
        repeat_factor: How many times to repeat the sentence bank
            before pairing.  Higher values bias the tokenizer toward
            this corpus.

    Returns:
        A list of ``{"utterances": [s1, s2], "emotions": [0, 0]}``
        dicts suitable to be concatenated with the Q→A dialogues
        before ``extract_triples`` is called.
    """
    # Hand-written HMI-domain bank (curated by the author) + freshly
    # fetched Wikipedia summaries for device / ML / accessibility
    # topics.  The wiki side is what pushes vocab past the ~3 k
    # ceiling of the hand-written corpus alone.
    sentences: list[str] = list(_DOMAIN_SENTENCES)
    try:
        sentences.extend(_wiki_sentences())
    except Exception as exc:  # pragma: no cover - network
        logger.warning("Wikipedia sentence fetch failed: %s", exc)

    # Repeat hand-written entries — the frequency-pruned tokenizer
    # keeps the top-N most frequent tokens, so domain terms need to
    # occur enough times to survive the cut.
    if repeat_factor > 1:
        sentences = _DOMAIN_SENTENCES * repeat_factor + sentences

    rng = random.Random(seed)
    rng.shuffle(sentences)

    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    # Pair consecutive sentences into two-turn dialogues.  These are
    # marked ``kind="filler"`` so the retrieval index can skip them —
    # they're here to broaden tokenizer vocabulary, not to drive Q→A.
    dialogues: list[dict[str, Any]] = []
    for i in range(0, len(sentences) - 1, 2):
        dialogues.append(
            {
                "utterances": [sentences[i], sentences[i + 1]],
                "emotions": [0, 0],
                "kind": "filler",
            }
        )

    logger.info(
        "Generated %d HMI-domain dialogues from %d unique sentences "
        "(repeat_factor=%d).",
        len(dialogues),
        len(_DOMAIN_SENTENCES),
        repeat_factor,
    )
    return dialogues


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    dialogues = load_gutenberg_dialogues()
    print(f"Built {len(dialogues)} HMI-domain dialogues.")
    if dialogues:
        print("First sample:", dialogues[0])
