"""Generate the seven I3 teaching notebooks as valid nbformat v4 JSON.

This script is not part of the runtime; it is a one-shot builder used to
materialise the .ipynb files. Run it once to (re)generate the notebooks.
"""

from __future__ import annotations

import json
import os
from typing import Iterable


def _lines(s: str) -> list[str]:
    """Split a block of text into an nbformat-compliant list of strings.

    nbformat v4 stores cell source as an array of strings where every line
    except the final one ends with \n.
    """
    if not s:
        return []
    parts = s.split("\n")
    out = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + "\n")
        else:
            if p != "":
                out.append(p)
    return out


def md(s: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(s),
    }


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(s),
    }


def notebook(cells: Iterable[dict]) -> dict:
    return {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: str, nb: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Notebook 01 -- Perception: keystroke dynamics
# ---------------------------------------------------------------------------

nb01 = notebook([
    md(
        "# 01 -- Perception: Keystroke Dynamics\n"
        "\n"
        "**Author**: Tamer Atesyakar\n"
        "\n"
        "This notebook walks through the 32-dimensional `InteractionFeatureVector` "
        "produced by the `i3.interaction.features` module. We record a short typing "
        "sample, compute the four feature groups (temporal, behavioural, content, "
        "contextual), and visualise each group with matplotlib.\n"
        "\n"
        "**Citations**\n"
        "- Banerjee & Woodard (2012). *Biometric Authentication and Identification using Keystroke Dynamics.*\n"
        "- Epp, Lippold & Mandryk (2011). *Identifying Emotional States using Keystroke Dynamics.* CHI.\n"
        "- Picard (1997). *Affective Computing.* MIT Press."
    ),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from dataclasses import asdict\n"
        "\n"
        "# Palette\n"
        "C_COLD = '#0f3460'\n"
        "C_HOT  = '#e94560'\n"
        "\n"
        "plt.rcParams.update({\n"
        "    'figure.facecolor': 'white',\n"
        "    'axes.grid': True,\n"
        "    'grid.alpha': 0.25,\n"
        "    'font.size': 10,\n"
        "})"
    ),
    md(
        "## 1.1 A synthetic typing sample\n"
        "\n"
        "We simulate 120 keystrokes with the structure of a real message: quick bursts "
        "of in-word typing, longer pauses at word boundaries, and occasional backspaces."
    ),
    code(
        "rng = np.random.default_rng(42)\n"
        "\n"
        "def synth_sample(n=120):\n"
        "    keys, times, dwell, kinds = [], [], [], []\n"
        "    t = 0.0\n"
        "    for i in range(n):\n"
        "        # Word-internal vs boundary pause\n"
        "        iki = rng.lognormal(mean=4.6, sigma=0.35)\n"
        "        if i > 0 and i % 7 == 0:\n"
        "            iki *= 2.4  # pause at word boundary\n"
        "        t += iki\n"
        "        times.append(t)\n"
        "        dwell.append(max(20.0, rng.normal(70, 12)))\n"
        "        is_bs = rng.random() < 0.07\n"
        "        kinds.append('backspace' if is_bs else 'char')\n"
        "        keys.append('\\x08' if is_bs else chr(ord('a') + rng.integers(0, 26)))\n"
        "    return np.array(times), np.array(dwell), kinds, keys\n"
        "\n"
        "times, dwell, kinds, keys = synth_sample(120)\n"
        "print(f'n = {len(times)}, duration = {times[-1]/1000:.2f} s')"
    ),
    md(
        "## 1.2 The four feature groups\n"
        "\n"
        "The `InteractionFeatureVector` is organised into:\n"
        "1. **Temporal** -- mean/std inter-key interval, pause counts, rhythm consistency.\n"
        "2. **Behavioural** -- backspace rate, edits per second, composition time.\n"
        "3. **Content** -- message length, lexical diversity, punctuation ratios.\n"
        "4. **Contextual** -- time of day, session position, recent engagement.\n"
        "\n"
        "We compute each group from the raw keystroke stream."
    ),
    code(
        "iki = np.diff(times)  # inter-key intervals in ms\n"
        "n_chars = sum(1 for k in kinds if k == 'char')\n"
        "n_bs    = sum(1 for k in kinds if k == 'backspace')\n"
        "\n"
        "# Temporal group (8 dims)\n"
        "temporal = {\n"
        "    'mean_iki':   float(np.mean(iki)),\n"
        "    'std_iki':    float(np.std(iki)),\n"
        "    'median_iki': float(np.median(iki)),\n"
        "    'q90_iki':    float(np.quantile(iki, 0.9)),\n"
        "    'min_iki':    float(np.min(iki)),\n"
        "    'max_iki':    float(np.max(iki)),\n"
        "    'rhythm_cv':  float(np.std(iki) / (np.mean(iki) + 1e-6)),\n"
        "    'pause_ct':   float(np.sum(iki > 500)),\n"
        "}\n"
        "temporal"
    ),
    code(
        "# Behavioural group (8 dims)\n"
        "duration_s = (times[-1] - times[0]) / 1000.0\n"
        "behavioural = {\n"
        "    'backspace_rate':  n_bs / max(n_chars, 1),\n"
        "    'edits_per_sec':   n_bs / max(duration_s, 1e-3),\n"
        "    'composition_s':   duration_s,\n"
        "    'mean_dwell':      float(np.mean(dwell)),\n"
        "    'std_dwell':       float(np.std(dwell)),\n"
        "    'chars_per_sec':   n_chars / max(duration_s, 1e-3),\n"
        "    'keystroke_total': float(len(times)),\n"
        "    'error_bursts':    float(sum(1 for i in range(1, len(kinds))\n"
        "                                 if kinds[i]=='backspace' and kinds[i-1]=='backspace')),\n"
        "}\n"
        "behavioural"
    ),
    code(
        "# Content group (8 dims) -- from the reconstructed text\n"
        "text_chars = [k for k, kd in zip(keys, kinds) if kd == 'char']\n"
        "text = ''.join(text_chars)\n"
        "content = {\n"
        "    'len':          float(len(text)),\n"
        "    'unique_ratio': len(set(text)) / max(len(text), 1),\n"
        "    'vowel_ratio':  sum(c in 'aeiou' for c in text) / max(len(text), 1),\n"
        "    'caps_ratio':   0.0,  # lowercase synth\n"
        "    'punct_ratio':  0.0,\n"
        "    'digit_ratio':  0.0,\n"
        "    'space_ratio':  0.0,\n"
        "    'repeat_ratio': sum(text[i] == text[i-1] for i in range(1, len(text))) / max(len(text)-1, 1),\n"
        "}\n"
        "content"
    ),
    code(
        "# Contextual group (8 dims)\n"
        "import time as _time\n"
        "now = _time.localtime()\n"
        "contextual = {\n"
        "    'hour_norm':      now.tm_hour / 23.0,\n"
        "    'weekday_norm':   now.tm_wday / 6.0,\n"
        "    'session_pos':    0.25,    # 1st message of session\n"
        "    'recent_engage':  0.60,    # placeholder EMA\n"
        "    'msg_since_rest': 0.0,\n"
        "    'device_kind':    0.0,     # 0 = desktop\n"
        "    'input_method':   0.0,     # 0 = physical kb\n"
        "    'lang_signal':    0.0,     # 0 = en\n"
        "}\n"
        "contextual"
    ),
    md(
        "## 1.3 Assemble the 32-dim InteractionFeatureVector\n"
        "\n"
        "In production, `i3.interaction.features.compute_feature_vector` handles this. "
        "Here we concatenate the four groups and verify the resulting shape."
    ),
    code(
        "try:\n"
        "    from i3.interaction import features as _feat  # noqa: F401\n"
        "    print('i3.interaction.features imported ok')\n"
        "except Exception as e:\n"
        "    print(f'(i3.interaction.features not importable in this kernel: {e})')\n"
        "\n"
        "vec = np.array(\n"
        "    list(temporal.values())\n"
        "    + list(behavioural.values())\n"
        "    + list(content.values())\n"
        "    + list(contextual.values()),\n"
        "    dtype=np.float32,\n"
        ")\n"
        "print('shape:', vec.shape)\n"
        "print('dtype:', vec.dtype)"
    ),
    md("## 1.4 Visualising the four groups"),
    code(
        "labels = ['Temporal', 'Behavioural', 'Content', 'Contextual']\n"
        "groups = [temporal, behavioural, content, contextual]\n"
        "fig, axes = plt.subplots(2, 2, figsize=(11, 7))\n"
        "for ax, lab, g in zip(axes.flat, labels, groups):\n"
        "    keys_ = list(g.keys())\n"
        "    vals  = [float(v) for v in g.values()]\n"
        "    # Normalise each group for display only.\n"
        "    m = max(abs(v) for v in vals) or 1.0\n"
        "    norm = [v / m for v in vals]\n"
        "    ax.barh(keys_, norm, color=[C_HOT if v >= 0 else C_COLD for v in norm])\n"
        "    ax.set_title(lab)\n"
        "    ax.axvline(0, color='#333', lw=0.5)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 1.5 Inter-key interval histogram"),
    code(
        "fig, ax = plt.subplots(figsize=(9, 3.5))\n"
        "ax.hist(iki, bins=30, color=C_COLD, edgecolor='white')\n"
        "ax.axvline(np.mean(iki), color=C_HOT, lw=2, label=f'mean={np.mean(iki):.0f} ms')\n"
        "ax.axvline(np.median(iki), color='#333', lw=1.2, ls='--', label=f'median={np.median(iki):.0f} ms')\n"
        "ax.set_xlabel('Inter-key interval (ms)')\n"
        "ax.set_ylabel('count')\n"
        "ax.set_title('IKI distribution -- log-normal shape is typical')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 1.6 Rhythm trace over time"),
    code(
        "fig, ax = plt.subplots(figsize=(9, 3.2))\n"
        "ax.plot(times[1:] / 1000.0, iki, color=C_COLD, lw=1.0, alpha=0.85)\n"
        "ax.scatter(times[1:][iki > 500] / 1000.0, iki[iki > 500],\n"
        "           color=C_HOT, s=20, zorder=3, label='pause > 500 ms')\n"
        "ax.set_xlabel('time (s)')\n"
        "ax.set_ylabel('IKI (ms)')\n"
        "ax.set_title('Rhythm trace -- pauses cluster at word boundaries')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 1.7 Takeaways\n"
        "\n"
        "- The four feature groups are linearly independent: temporal captures *when*, "
        "behavioural captures *how fluently*, content captures *what*, contextual captures "
        "*the surrounding conditions*.\n"
        "- The log-normal IKI distribution is a hallmark of natural typing; deviations toward "
        "heavier tails are a strong signal of cognitive load (Epp et al. 2011).\n"
        "- The 32-dim vector is the input to the TCN encoder covered in notebook 02."
    ),
])

write(os.path.join(HERE, "01_perception_keystroke_dynamics.ipynb"), nb01)


# ---------------------------------------------------------------------------
# Notebook 02 -- TCN encoder from scratch
# ---------------------------------------------------------------------------

nb02 = notebook([
    md(
        "# 02 -- TCN Encoder From Scratch\n"
        "\n"
        "**Author**: Tamer Atesyakar\n"
        "\n"
        "We implement a tiny Temporal Convolutional Network (TCN) using dilated causal "
        "1-D convolutions, attach an NT-Xent contrastive head, and contrast its receptive "
        "field and performance against an LSTM on the same synthetic sequence.\n"
        "\n"
        "**Citations**\n"
        "- Bai, Kolter & Koltun (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.\n"
        "- Chen, Kornblith et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations (NT-Xent).* ICML.\n"
        "- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation."
    ),
    code(
        "import math\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "try:\n"
        "    import torch\n"
        "    import torch.nn as nn\n"
        "    import torch.nn.functional as F\n"
        "    TORCH_OK = True\n"
        "except ImportError as e:\n"
        "    TORCH_OK = False\n"
        "    print(f'torch not available: {e}')\n"
        "\n"
        "C_COLD = '#0f3460'\n"
        "C_HOT  = '#e94560'\n"
        "plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.25})"
    ),
    md(
        "## 2.1 Receptive-field math\n"
        "\n"
        "For a causal TCN stack of `L` layers with kernel size `k` and exponentially "
        "growing dilation `d_l = 2^l`, the total receptive field is\n"
        "\n"
        "$$R = 1 + (k - 1) \\cdot \\sum_{l=0}^{L-1} 2^l = 1 + (k-1)(2^L - 1).$$\n"
        "\n"
        "For `k=3, L=4` this gives `R = 1 + 2 * 15 = 31` past timesteps, enough for the "
        "last 30 keystrokes in a typing sample."
    ),
    code(
        "def receptive_field(k, L):\n"
        "    return 1 + (k - 1) * (2**L - 1)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 3.6))\n"
        "Ls = np.arange(1, 9)\n"
        "for k, color in [(2, C_COLD), (3, C_HOT)]:\n"
        "    ax.plot(Ls, [receptive_field(k, L) for L in Ls],\n"
        "            marker='o', color=color, label=f'kernel={k}')\n"
        "ax.set_xlabel('depth L')\n"
        "ax.set_ylabel('receptive field (timesteps)')\n"
        "ax.set_title('Dilated causal conv stack: receptive field grows 2^L')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 2.2 Dilated causal conv block"),
    code(
        "if TORCH_OK:\n"
        "    class CausalConv1d(nn.Module):\n"
        "        def __init__(self, cin, cout, k, dilation):\n"
        "            super().__init__()\n"
        "            self.pad = (k - 1) * dilation\n"
        "            self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)\n"
        "\n"
        "        def forward(self, x):\n"
        "            # pad only on the left -> strictly causal\n"
        "            x = F.pad(x, (self.pad, 0))\n"
        "            return self.conv(x)\n"
        "\n"
        "    class TCNBlock(nn.Module):\n"
        "        def __init__(self, c, k, dilation, dropout=0.1):\n"
        "            super().__init__()\n"
        "            self.c1 = CausalConv1d(c, c, k, dilation)\n"
        "            self.c2 = CausalConv1d(c, c, k, dilation)\n"
        "            self.drop = nn.Dropout(dropout)\n"
        "\n"
        "        def forward(self, x):\n"
        "            r = x\n"
        "            x = F.relu(self.c1(x)); x = self.drop(x)\n"
        "            x = F.relu(self.c2(x)); x = self.drop(x)\n"
        "            return x + r\n"
        "\n"
        "    class TinyTCN(nn.Module):\n"
        "        def __init__(self, din, c=32, k=3, L=4, proj=16):\n"
        "            super().__init__()\n"
        "            self.stem = nn.Conv1d(din, c, 1)\n"
        "            self.blocks = nn.ModuleList(\n"
        "                [TCNBlock(c, k, dilation=2**l) for l in range(L)]\n"
        "            )\n"
        "            self.head = nn.Linear(c, proj)\n"
        "\n"
        "        def forward(self, x):      # x: (B, T, D)\n"
        "            h = self.stem(x.transpose(1, 2))\n"
        "            for b in self.blocks:\n"
        "                h = b(h)\n"
        "            h = h.mean(dim=-1)      # global temporal avg\n"
        "            z = self.head(h)\n"
        "            return F.normalize(z, dim=-1)\n"
        "\n"
        "    m = TinyTCN(din=32)\n"
        "    n_params = sum(p.numel() for p in m.parameters())\n"
        "    print(f'TinyTCN parameters: {n_params:,}')\n"
        "else:\n"
        "    print('torch unavailable -- model definition skipped')"
    ),
    md(
        "## 2.3 NT-Xent contrastive loss\n"
        "\n"
        "For a batch of `2N` projections (two augmentations per anchor), NT-Xent is\n"
        "\n"
        "$$\\mathcal{L}_{i,j} = -\\log\\frac{\\exp(\\mathrm{sim}(z_i,z_j)/\\tau)}"
        "{\\sum_{k\\neq i}\\exp(\\mathrm{sim}(z_i,z_k)/\\tau)}.$$\n"
        "\n"
        "We implement it below (Chen et al. 2020)."
    ),
    code(
        "if TORCH_OK:\n"
        "    def nt_xent(z1, z2, tau=0.1):\n"
        "        z = torch.cat([z1, z2], dim=0)              # (2N, d)\n"
        "        sim = z @ z.t() / tau                        # cosine because z is l2-normed\n"
        "        N = z1.shape[0]\n"
        "        mask = torch.eye(2 * N, device=z.device).bool()\n"
        "        sim.masked_fill_(mask, -1e9)\n"
        "        labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)\n"
        "        return F.cross_entropy(sim, labels)\n"
        "\n"
        "    # Sanity: random tensors produce a well-conditioned loss.\n"
        "    torch.manual_seed(0)\n"
        "    z1 = F.normalize(torch.randn(8, 16), dim=-1)\n"
        "    z2 = F.normalize(torch.randn(8, 16), dim=-1)\n"
        "    print(f'NT-Xent on random batch (tau=0.1): {nt_xent(z1, z2).item():.3f}')"
    ),
    md("## 2.4 LSTM baseline on the same task"),
    code(
        "if TORCH_OK:\n"
        "    class TinyLSTM(nn.Module):\n"
        "        def __init__(self, din, h=32, proj=16):\n"
        "            super().__init__()\n"
        "            self.lstm = nn.LSTM(din, h, batch_first=True)\n"
        "            self.head = nn.Linear(h, proj)\n"
        "\n"
        "        def forward(self, x):\n"
        "            y, _ = self.lstm(x)\n"
        "            z = self.head(y[:, -1])\n"
        "            return F.normalize(z, dim=-1)\n"
        "\n"
        "    lstm = TinyLSTM(din=32)\n"
        "    print(f'TinyLSTM parameters: {sum(p.numel() for p in lstm.parameters()):,}')"
    ),
    md(
        "## 2.5 Toy training: distinguish two typing rhythms\n"
        "\n"
        "We build two clusters of synthetic 32-dim sequences (relaxed vs stressed) and "
        "show that the TCN separates them after a handful of gradient steps."
    ),
    code(
        "if TORCH_OK:\n"
        "    def synth_seq(kind, T=32):\n"
        "        base = np.zeros((T, 32), dtype=np.float32)\n"
        "        if kind == 'relaxed':\n"
        "            base[:, 0] = np.random.normal(120, 15, T)    # mean IKI\n"
        "            base[:, 1] = np.random.normal(25, 4, T)\n"
        "        else:  # stressed\n"
        "            base[:, 0] = np.random.normal(90, 35, T)\n"
        "            base[:, 1] = np.random.normal(55, 9, T)\n"
        "        base += np.random.normal(0, 0.02, base.shape)\n"
        "        return base\n"
        "\n"
        "    def batch(n=16):\n"
        "        xs = [synth_seq('relaxed') for _ in range(n)] + [synth_seq('stressed') for _ in range(n)]\n"
        "        return torch.tensor(np.stack(xs), dtype=torch.float32)\n"
        "\n"
        "    torch.manual_seed(1)\n"
        "    m = TinyTCN(din=32)\n"
        "    opt = torch.optim.Adam(m.parameters(), lr=3e-3)\n"
        "    losses = []\n"
        "    for step in range(80):\n"
        "        x = batch()\n"
        "        # Simple augmentation: additive noise\n"
        "        x1 = x + 0.01 * torch.randn_like(x)\n"
        "        x2 = x + 0.01 * torch.randn_like(x)\n"
        "        z1, z2 = m(x1), m(x2)\n"
        "        loss = nt_xent(z1, z2, tau=0.2)\n"
        "        opt.zero_grad(); loss.backward(); opt.step()\n"
        "        losses.append(loss.item())\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=(8, 3))\n"
        "    ax.plot(losses, color=C_HOT)\n"
        "    ax.set_xlabel('step'); ax.set_ylabel('NT-Xent loss')\n"
        "    ax.set_title('Contrastive objective decreases on toy typing rhythms')\n"
        "    plt.tight_layout(); plt.show()"
    ),
    md("## 2.6 Embedding geometry"),
    code(
        "if TORCH_OK:\n"
        "    with torch.no_grad():\n"
        "        x = batch(32)\n"
        "        z = m(x).numpy()\n"
        "    # Since proj=16 and l2-normed, project onto the first two PCA components.\n"
        "    zc = z - z.mean(0, keepdims=True)\n"
        "    U, S, Vt = np.linalg.svd(zc, full_matrices=False)\n"
        "    p = zc @ Vt[:2].T\n"
        "    n = x.shape[0] // 2\n"
        "    fig, ax = plt.subplots(figsize=(5.2, 5))\n"
        "    ax.scatter(p[:n, 0], p[:n, 1], color=C_COLD, label='relaxed', s=40, alpha=0.85)\n"
        "    ax.scatter(p[n:, 0], p[n:, 1], color=C_HOT,  label='stressed', s=40, alpha=0.85)\n"
        "    ax.set_title('TCN embedding -- first 2 PCA components')\n"
        "    ax.legend(); ax.set_aspect('equal')\n"
        "    plt.tight_layout(); plt.show()"
    ),
    md(
        "## 2.7 TCN vs LSTM: summary\n"
        "\n"
        "- **Parallelism**: TCN forward pass parallelises across time; LSTM is sequential.\n"
        "- **Receptive field**: TCN grows 2^L; LSTM is effectively limited by gradient flow "
        "over long ranges.\n"
        "- **Stability**: residual connections + weight-norm make TCNs easier to train (Bai et al. 2018).\n"
        "- **Memory**: for T=32 and modest depth the TCN is cheaper on modern accelerators."
    ),
])

write(os.path.join(HERE, "02_tcn_encoder_from_scratch.ipynb"), nb02)


# ---------------------------------------------------------------------------
# Notebook 03 -- Three-timescale user model
# ---------------------------------------------------------------------------

nb03 = notebook([
    md(
        "# 03 -- Three-Timescale User Model\n"
        "\n"
        "**Author**: Tamer Atesyakar\n"
        "\n"
        "The I3 user model maintains *three* running statistics over behavioural features:\n"
        "- **Instant** -- last observation.\n"
        "- **Session** -- short-horizon EMA (~ last few minutes).\n"
        "- **Long-term** -- slow EMA over many sessions.\n"
        "\n"
        "Their sufficient statistics are updated online using Welford's algorithm.\n"
        "\n"
        "**Citations**\n"
        "- Welford (1962). *Note on a method for calculating corrected sums of squares and products.* Technometrics.\n"
        "- Knuth (1998). *The Art of Computer Programming*, vol. 2."
    ),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "C_COLD = '#0f3460'\n"
        "C_HOT  = '#e94560'\n"
        "plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.25})"
    ),
    md(
        "## 3.1 Welford's online variance\n"
        "\n"
        "Naive two-pass variance is catastrophically unstable on streams. Welford's "
        "recurrence keeps the running mean `M` and the sum of squared deviations `S` "
        "and only needs `O(1)` memory per feature:\n"
        "\n"
        "```\n"
        "n   += 1\n"
        "d    = x - M\n"
        "M   += d / n\n"
        "S   += d * (x - M)\n"
        "var  = S / (n - 1)\n"
        "```\n"
        "\n"
        "We verify it matches `np.var(ddof=1)`."
    ),
    code(
        "class Welford:\n"
        "    def __init__(self):\n"
        "        self.n = 0; self.M = 0.0; self.S = 0.0\n"
        "    def push(self, x):\n"
        "        self.n += 1\n"
        "        d = x - self.M\n"
        "        self.M += d / self.n\n"
        "        self.S += d * (x - self.M)\n"
        "    @property\n"
        "    def var(self):\n"
        "        return self.S / max(self.n - 1, 1)\n"
        "    @property\n"
        "    def std(self):\n"
        "        return self.var ** 0.5\n"
        "\n"
        "rng = np.random.default_rng(7)\n"
        "xs = rng.normal(loc=3.0, scale=1.5, size=5000)\n"
        "w = Welford()\n"
        "for x in xs:\n"
        "    w.push(x)\n"
        "print(f'Welford mean={w.M:.6f}   numpy mean={xs.mean():.6f}')\n"
        "print(f'Welford var ={w.var:.6f}  numpy var ={xs.var(ddof=1):.6f}')"
    ),
    md(
        "## 3.2 Exponential moving averages at three timescales\n"
        "\n"
        "An EMA with smoothing `alpha` has effective memory `~ 1/alpha` observations. "
        "We pick three alphas corresponding to instant (alpha=1), session (alpha=0.12 -> ~8 "
        "messages) and long-term (alpha=0.01 -> ~100 messages)."
    ),
    code(
        "class EMA:\n"
        "    def __init__(self, alpha, init=0.0):\n"
        "        self.alpha = alpha\n"
        "        self.v = float(init)\n"
        "        self.started = False\n"
        "    def push(self, x):\n"
        "        if not self.started:\n"
        "            self.v = float(x); self.started = True\n"
        "        else:\n"
        "            self.v = (1 - self.alpha) * self.v + self.alpha * float(x)\n"
        "        return self.v\n"
        "\n"
        "# Simulate 200 messages where the user transitions: relaxed -> stressed -> relaxed\n"
        "T = 200\n"
        "load = np.zeros(T, dtype=np.float32)\n"
        "rng = np.random.default_rng(3)\n"
        "for t in range(T):\n"
        "    if   t <  70:  base = 0.3   # relaxed\n"
        "    elif t < 130:  base = 0.75  # stressed burst\n"
        "    else:          base = 0.4\n"
        "    load[t] = np.clip(base + rng.normal(0, 0.1), 0, 1)\n"
        "\n"
        "inst = EMA(alpha=1.0)\n"
        "sess = EMA(alpha=0.12)\n"
        "long_ = EMA(alpha=0.01)\n"
        "\n"
        "traces = []\n"
        "for x in load:\n"
        "    traces.append([inst.push(x), sess.push(x), long_.push(x)])\n"
        "traces = np.array(traces)\n"
        "traces.shape"
    ),
    md("## 3.3 Visualising the three traces"),
    code(
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.plot(load,          color='#999', lw=1.0, alpha=0.65, label='raw signal')\n"
        "ax.plot(traces[:, 0],  color='#333', lw=0.7, alpha=0.6,  label='instant (alpha=1)')\n"
        "ax.plot(traces[:, 1],  color=C_COLD, lw=2.0, label='session (alpha=0.12)')\n"
        "ax.plot(traces[:, 2],  color=C_HOT,  lw=2.0, label='long-term (alpha=0.01)')\n"
        "ax.axvspan(70, 130, color=C_HOT, alpha=0.07, label='stressed burst')\n"
        "ax.set_xlabel('message index')\n"
        "ax.set_ylabel('cognitive load')\n"
        "ax.set_title('Three-timescale EMA traces over 200 messages')\n"
        "ax.legend(loc='upper right', fontsize=9)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 3.4 Deviation as an anomaly signal\n"
        "\n"
        "A powerful derived signal is the *deviation* between the session and long-term "
        "traces, standardised by the long-run variance. Large positive deviations flag "
        "acute stress above baseline."
    ),
    code(
        "# Running Welford on the raw signal gives the long-run variance.\n"
        "w = Welford()\n"
        "zs = []\n"
        "for t in range(T):\n"
        "    w.push(load[t])\n"
        "    sigma = max(w.std, 1e-3)\n"
        "    zs.append((traces[t, 1] - traces[t, 2]) / sigma)\n"
        "zs = np.array(zs)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 3.2))\n"
        "ax.plot(zs, color=C_HOT, lw=1.4)\n"
        "ax.axhline(0, color='#555', lw=0.5)\n"
        "ax.axhline(1.5, color=C_COLD, lw=0.8, ls='--', label='+1.5 sigma')\n"
        "ax.axhline(-1.5, color=C_COLD, lw=0.8, ls='--')\n"
        "ax.fill_between(np.arange(T), zs, 0, where=(np.abs(zs) > 1.5),\n"
        "                color=C_HOT, alpha=0.25)\n"
        "ax.set_title('Session-minus-baseline deviation (standardised)')\n"
        "ax.set_xlabel('message index'); ax.set_ylabel('z-score')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 3.5 Why three timescales?\n"
        "\n"
        "- **Instant** preserves responsiveness: a single outlier can still trigger a safety "
        "intervention.\n"
        "- **Session** captures the current mood trajectory.\n"
        "- **Long-term** acts as an *anchor* so that the system does not drift with transient "
        "variation.\n"
        "\n"
        "The online Welford update (1962) gives the long-term variance with constant memory "
        "and numerically stable deviation z-scores."
    ),
])

write(os.path.join(HERE, "03_three_timescale_user_model.ipynb"), nb03)


# ---------------------------------------------------------------------------
# Notebook 04 -- Cross-attention conditioning (centrepiece)
# ---------------------------------------------------------------------------

nb04 = notebook([
    md(
        "# 04 -- Cross-Attention Conditioning (Centrepiece)\n"
        "\n"
        "**Author**: Tamer Atesyakar\n"
        "\n"
        "This is the load-bearing notebook. We construct a small `ConditioningProjector` "
        "that maps a 7-dim `AdaptationVector` into conditioning-token space, plug it into "
        "a minimal pre-norm transformer decoder block via *cross-attention*, and run the "
        "**conditioning-sensitivity test**: given the same prompt, we measure that two "
        "different adaptation vectors induce next-token distributions with *measurably* "
        "large Kullback-Leibler divergence. This is the necessary condition for the "
        "adaptation pipeline to actually influence generation.\n"
        "\n"
        "**Citations**\n"
        "- Vaswani et al. (2017). *Attention is All You Need.* NeurIPS.\n"
        "- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture.* ICML.\n"
        "- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* ICLR."
    ),
    code(
        "import math\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "try:\n"
        "    import torch\n"
        "    import torch.nn as nn\n"
        "    import torch.nn.functional as F\n"
        "    TORCH_OK = True\n"
        "except ImportError as e:\n"
        "    TORCH_OK = False\n"
        "    print(f'torch unavailable: {e}')\n"
        "\n"
        "C_COLD = '#0f3460'\n"
        "C_HOT  = '#e94560'\n"
        "plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.25})"
    ),
    md(
        "## 4.1 The AdaptationVector\n"
        "\n"
        "Seven scalar dials in [0, 1], each describing how to shape a response:\n"
        "`verbosity, formality, directness, warmth, cognitive_accessibility, "
        "emotional_tone, technical_depth`."
    ),
    code(
        "ADAPT_DIMS = ['verbosity','formality','directness','warmth',\n"
        "              'cognitive_accessibility','emotional_tone','technical_depth']\n"
        "\n"
        "def adapt(v=0.5, f=0.5, d=0.5, w=0.5, a=0.5, e=0.5, t=0.5):\n"
        "    return torch.tensor([[v, f, d, w, a, e, t]], dtype=torch.float32)\n"
        "\n"
        "if TORCH_OK:\n"
        "    av_a = adapt(v=0.25, d=0.85, a=0.9)   # low cognitive load\n"
        "    av_b = adapt(v=0.9,  w=0.95, e=0.9)   # high warmth, chatty\n"
        "    print(dict(zip(ADAPT_DIMS, av_a[0].tolist())))\n"
        "    print(dict(zip(ADAPT_DIMS, av_b[0].tolist())))"
    ),
    md(
        "## 4.2 ConditioningProjector\n"
        "\n"
        "Takes the 7-dim adaptation vector and projects it to `K` conditioning tokens of "
        "dimension `d_model` using a small MLP that emits `K * d_model` values. The K "
        "tokens are treated as a sequence the decoder can cross-attend over."
    ),
    code(
        "if TORCH_OK:\n"
        "    class ConditioningProjector(nn.Module):\n"
        "        def __init__(self, d_adapt=7, d_model=32, k_tokens=4):\n"
        "            super().__init__()\n"
        "            self.k = k_tokens\n"
        "            self.d = d_model\n"
        "            self.mlp = nn.Sequential(\n"
        "                nn.Linear(d_adapt, 64),\n"
        "                nn.GELU(),\n"
        "                nn.Linear(64, k_tokens * d_model),\n"
        "            )\n"
        "            self.pos = nn.Parameter(torch.randn(1, k_tokens, d_model) * 0.02)\n"
        "\n"
        "        def forward(self, a):\n"
        "            # a: (B, d_adapt) -> (B, k, d_model)\n"
        "            y = self.mlp(a).view(a.shape[0], self.k, self.d)\n"
        "            return y + self.pos\n"
        "\n"
        "    proj = ConditioningProjector()\n"
        "    toks = proj(av_a)\n"
        "    print('conditioning tokens shape:', tuple(toks.shape))"
    ),
    md("## 4.3 Minimal pre-norm transformer block with cross-attention"),
    code(
        "if TORCH_OK:\n"
        "    class TinyXAttnBlock(nn.Module):\n"
        "        def __init__(self, d_model=32, n_heads=4, d_ff=64, dropout=0.0):\n"
        "            super().__init__()\n"
        "            self.ln1 = nn.LayerNorm(d_model)\n"
        "            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)\n"
        "            self.ln2 = nn.LayerNorm(d_model)\n"
        "            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)\n"
        "            self.ln3 = nn.LayerNorm(d_model)\n"
        "            self.ff = nn.Sequential(\n"
        "                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),\n"
        "            )\n"
        "            self.drop = nn.Dropout(dropout)\n"
        "            self.last_attn_weights = None\n"
        "\n"
        "        def forward(self, x, cond):\n"
        "            # Pre-norm (Xiong 2020) -- more stable than post-norm for small models\n"
        "            h = self.ln1(x)\n"
        "            sa, _ = self.self_attn(h, h, h, need_weights=False)\n"
        "            x = x + self.drop(sa)\n"
        "            h = self.ln2(x)\n"
        "            ca, w = self.cross_attn(h, cond, cond, need_weights=True, average_attn_weights=True)\n"
        "            self.last_attn_weights = w.detach()     # (B, T_q, T_k)\n"
        "            x = x + self.drop(ca)\n"
        "            x = x + self.drop(self.ff(self.ln3(x)))\n"
        "            return x\n"
        "\n"
        "    class TinyLM(nn.Module):\n"
        "        def __init__(self, vocab=64, d_model=32, n_blocks=4, k_cond=4):\n"
        "            super().__init__()\n"
        "            self.embed = nn.Embedding(vocab, d_model)\n"
        "            self.pos   = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)\n"
        "            self.proj  = ConditioningProjector(d_adapt=7, d_model=d_model, k_tokens=k_cond)\n"
        "            self.blocks = nn.ModuleList(\n"
        "                [TinyXAttnBlock(d_model=d_model) for _ in range(n_blocks)]\n"
        "            )\n"
        "            self.ln_f  = nn.LayerNorm(d_model)\n"
        "            self.head  = nn.Linear(d_model, vocab, bias=False)\n"
        "\n"
        "        def forward(self, tokens, adapt_vec):\n"
        "            B, T = tokens.shape\n"
        "            x = self.embed(tokens) + self.pos[:, :T]\n"
        "            cond = self.proj(adapt_vec)\n"
        "            for blk in self.blocks:\n"
        "                x = blk(x, cond)\n"
        "            return self.head(self.ln_f(x))\n"
        "\n"
        "    m = TinyLM(vocab=64)\n"
        "    n = sum(p.numel() for p in m.parameters())\n"
        "    print(f'TinyLM parameters: {n:,}')"
    ),
    md(
        "## 4.4 Train on a toy objective that exposes the conditioning\n"
        "\n"
        "Teaching signal: given a prompt `[BOS, 1, 2, 3]`, the target last-token is either "
        "`HIGH` or `LOW` depending on `adapt[0]` (verbosity). If the conditioning channel "
        "is functional, the model learns to use it; if not, it cannot solve the task."
    ),
    code(
        "if TORCH_OK:\n"
        "    VOCAB = 64; BOS = 1; HIGH = 40; LOW = 10\n"
        "    torch.manual_seed(0)\n"
        "    lm  = TinyLM(vocab=VOCAB)\n"
        "    opt = torch.optim.Adam(lm.parameters(), lr=3e-3)\n"
        "\n"
        "    def make_batch(n=32):\n"
        "        toks = torch.tensor([[BOS, 2, 3, 4]] * n)\n"
        "        a = torch.rand(n, 7)\n"
        "        # Label depends on verbosity > 0.5\n"
        "        y = torch.where(a[:, 0] > 0.5, torch.full((n,), HIGH), torch.full((n,), LOW))\n"
        "        return toks, a, y\n"
        "\n"
        "    losses = []\n"
        "    for step in range(300):\n"
        "        toks, a, y = make_batch()\n"
        "        logits = lm(toks, a)                # (B, T, V)\n"
        "        loss = F.cross_entropy(logits[:, -1], y)\n"
        "        opt.zero_grad(); loss.backward(); opt.step()\n"
        "        losses.append(loss.item())\n"
        "    print(f'final CE: {losses[-1]:.3f}')\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=(8, 3))\n"
        "    ax.plot(losses, color=C_HOT); ax.set_title('Training loss')\n"
        "    ax.set_xlabel('step'); ax.set_ylabel('CE')\n"
        "    plt.tight_layout(); plt.show()"
    ),
    md(
        "## 4.5 The conditioning-sensitivity test\n"
        "\n"
        "For the same prompt, we compute `p(next | adapt=A)` and `p(next | adapt=B)` and "
        "measure their KL divergence. A working conditioning pathway must yield a "
        "*materially large* KL -- otherwise the adaptation vector has no effect on the "
        "output distribution."
    ),
    code(
        "if TORCH_OK:\n"
        "    lm.eval()\n"
        "    with torch.no_grad():\n"
        "        toks = torch.tensor([[BOS, 2, 3, 4]])\n"
        "        p_a = F.softmax(lm(toks, av_a)[0, -1], dim=-1)\n"
        "        p_b = F.softmax(lm(toks, av_b)[0, -1], dim=-1)\n"
        "        kl = torch.sum(p_a * (torch.log(p_a + 1e-12) - torch.log(p_b + 1e-12))).item()\n"
        "    print(f'KL(p_A || p_B) = {kl:.4f} nats  ({kl/math.log(2):.4f} bits)')\n"
        "    print('(For an ineffective conditioning pathway this would be ~0.)')\n"
        "    print(f'argmax under A: token {int(p_a.argmax())}   expected {HIGH} (verbosity>0.5? {av_a[0,0].item()>0.5})')\n"
        "    print(f'argmax under B: token {int(p_b.argmax())}   expected {HIGH} (verbosity>0.5? {av_b[0,0].item()>0.5})')"
    ),
    md("## 4.6 Visualising cross-attention weights"),
    code(
        "if TORCH_OK:\n"
        "    with torch.no_grad():\n"
        "        _ = lm(toks, av_a)\n"
        "    rows = []\n"
        "    for blk in lm.blocks:\n"
        "        w = blk.last_attn_weights[0, -1].cpu().numpy()   # last query over K keys\n"
        "        rows.append(w)\n"
        "    mat = np.stack(rows)\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=(5.2, 4))\n"
        "    from matplotlib.colors import LinearSegmentedColormap\n"
        "    cmap = LinearSegmentedColormap.from_list('i3', [C_COLD, C_HOT])\n"
        "    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=mat.max())\n"
        "    ax.set_xlabel('conditioning token')\n"
        "    ax.set_ylabel('transformer block')\n"
        "    ax.set_title('Cross-attention: last generated token -> conditioning tokens')\n"
        "    for r in range(mat.shape[0]):\n"
        "        for c in range(mat.shape[1]):\n"
        "            ax.text(c, r, f'{mat[r,c]:.2f}', ha='center', va='center',\n"
        "                    fontsize=9, color='white' if mat[r,c] > mat.max()/2 else '#333')\n"
        "    fig.colorbar(im, ax=ax, shrink=0.8)\n"
        "    plt.tight_layout(); plt.show()"
    ),
    md(
        "## 4.7 KL sweep across the adaptation simplex\n"
        "\n"
        "We sweep one dimension (verbosity) while holding the rest fixed and plot the KL "
        "of the output distribution versus a neutral baseline."
    ),
    code(
        "if TORCH_OK:\n"
        "    baseline = adapt()              # all 0.5\n"
        "    xs = np.linspace(0.0, 1.0, 21)\n"
        "    kls = []\n"
        "    with torch.no_grad():\n"
        "        p_base = F.softmax(lm(toks, baseline)[0, -1], dim=-1)\n"
        "        for v in xs:\n"
        "            a = adapt(v=float(v))\n"
        "            p = F.softmax(lm(toks, a)[0, -1], dim=-1)\n"
        "            k = torch.sum(p * (torch.log(p + 1e-12) - torch.log(p_base + 1e-12))).item()\n"
        "            kls.append(k)\n"
        "    fig, ax = plt.subplots(figsize=(7, 3.2))\n"
        "    ax.plot(xs, kls, color=C_HOT, lw=2, marker='o')\n"
        "    ax.axvline(0.5, color='#555', ls='--', lw=0.7, label='baseline')\n"
        "    ax.set_xlabel('verbosity'); ax.set_ylabel('KL(p || p_baseline) [nats]')\n"
        "    ax.set_title('Output distribution sensitivity to verbosity')\n"
        "    ax.legend(); plt.tight_layout(); plt.show()"
    ),
    md(
        "## 4.8 Why this matters\n"
        "\n"
        "The conditioning-sensitivity test is the single most diagnostic check for the I3 "
        "architecture: without it, the entire adaptation pipeline could be a *placebo*. By "
        "measuring a non-trivial KL between outputs conditioned on different adaptation "
        "vectors, we empirically verify that cross-attention is actually routing information "
        "from the adaptation pathway into the generation step (Vaswani 2017, Xiong 2020)."
    ),
])

write(os.path.join(HERE, "04_cross_attention_conditioning_centrepiece.ipynb"), nb04)


# ---------------------------------------------------------------------------
# Notebook 05 -- Contextual Thompson Sampling
# ---------------------------------------------------------------------------

nb05 = notebook([
    md(
        "# 05 -- Contextual Thompson Sampling\n"
        "\n"
        "**Author**: Tamer Atesyakar\n"
        "\n"
        "We build a Bayesian logistic-regression contextual bandit from scratch, derive the "
        "Laplace approximation of its posterior via Newton-Raphson MAP fitting, and run a "
        "simulated regret study against Uniform and epsilon-Greedy baselines on a toy "
        "tone-choice problem.\n"
        "\n"
        "**Citations**\n"
        "- Russo, Van Roy, Kazerouni, Osband & Wen (2018). *A Tutorial on Thompson Sampling.* Foundations and Trends in ML.\n"
        "- Chapelle & Li (2011). *An Empirical Evaluation of Thompson Sampling.* NeurIPS.\n"
        "- MacKay (1992). *The Evidence Framework Applied to Classification Networks.* Neural Computation."
    ),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "C_COLD = '#0f3460'\n"
        "C_HOT  = '#e94560'\n"
        "plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.25})"
    ),
    md(
        "## 5.1 Problem setup\n"
        "\n"
        "- Context: `x in R^d` summarises the user state (cognitive load, warmth, etc.).\n"
        "- Arm: one of `K` tone choices (concise / encouraging / technical).\n"
        "- Reward: `r in {0, 1}` (user accepted vs dismissed the adaptation).\n"
        "- Model: each arm k has its own weight vector `w_k`; `p(r=1 | x, k) = sigma(x^T w_k)`.\n"
        "\n"
        "This is one independent Bayesian logistic regression per arm."
    ),
    code(
        "D, K = 6, 3\n"
        "rng = np.random.default_rng(11)\n"
        "# Ground-truth weight per arm (unknown to the learner)\n"
        "W_true = rng.normal(0, 1.0, size=(K, D)) * 0.8\n"
        "\n"
        "def sigmoid(z):\n"
        "    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))\n"
        "\n"
        "def draw_context():\n"
        "    return rng.normal(0, 1.0, size=D)\n"
        "\n"
        "def pull(arm, x):\n"
        "    p = sigmoid(x @ W_true[arm])\n"
        "    return int(rng.random() < p), float(p)"
    ),
    md(
        "## 5.2 Laplace approximation via Newton-Raphson MAP\n"
        "\n"
        "We place a Gaussian prior `w ~ N(0, (1/lambda) I)` on each arm's weight. The MAP "
        "objective is the log posterior\n"
        "\n"
        "$$\\mathcal{L}(w) = \\sum_t \\big[ y_t \\log\\sigma(x_t^\\top w) + (1-y_t)\\log(1-\\sigma(x_t^\\top w)) \\big] - \\tfrac{\\lambda}{2}\\|w\\|^2.$$\n"
        "\n"
        "The Hessian at the MAP `w*` equals `-X^T S X - lambda I` where `S = diag(p(1-p))`. "
        "The Laplace posterior is then `N(w*, (-H)^{-1})` (MacKay 1992)."
    ),
    code(
        "class BayesLogReg:\n"
        "    def __init__(self, d, lam=1.0):\n"
        "        self.d = d; self.lam = lam\n"
        "        self.mu    = np.zeros(d)\n"
        "        self.Sigma = np.eye(d) / lam\n"
        "        self.X = np.zeros((0, d)); self.y = np.zeros(0)\n"
        "\n"
        "    def observe(self, x, y):\n"
        "        self.X = np.vstack([self.X, x[None]])\n"
        "        self.y = np.append(self.y, y)\n"
        "\n"
        "    def fit(self, n_iter=20, tol=1e-6):\n"
        "        # Newton-Raphson on the negative log posterior.\n"
        "        w = self.mu.copy()\n"
        "        for _ in range(n_iter):\n"
        "            if self.X.shape[0] == 0:\n"
        "                break\n"
        "            p = sigmoid(self.X @ w)\n"
        "            S = p * (1 - p)\n"
        "            grad = self.X.T @ (p - self.y) + self.lam * w\n"
        "            H = (self.X.T * S) @ self.X + self.lam * np.eye(self.d)\n"
        "            step = np.linalg.solve(H, grad)\n"
        "            w_new = w - step\n"
        "            if np.linalg.norm(step) < tol:\n"
        "                w = w_new; break\n"
        "            w = w_new\n"
        "        self.mu = w\n"
        "        if self.X.shape[0] > 0:\n"
        "            p = sigmoid(self.X @ w)\n"
        "            S = p * (1 - p)\n"
        "            H = (self.X.T * S) @ self.X + self.lam * np.eye(self.d)\n"
        "            self.Sigma = np.linalg.inv(H)\n"
        "        else:\n"
        "            self.Sigma = np.eye(self.d) / self.lam\n"
        "\n"
        "    def sample_weight(self):\n"
        "        # One posterior sample via Cholesky.\n"
        "        L = np.linalg.cholesky(self.Sigma + 1e-8 * np.eye(self.d))\n"
        "        return self.mu + L @ rng.normal(size=self.d)"
    ),
    md("## 5.3 Thompson-sampling policy and baselines"),
    code(
        "class Thompson:\n"
        "    def __init__(self, k, d, lam=1.0, refit_every=5):\n"
        "        self.models = [BayesLogReg(d, lam) for _ in range(k)]\n"
        "        self.t = 0; self.refit_every = refit_every\n"
        "    def choose(self, x):\n"
        "        scores = [x @ m.sample_weight() for m in self.models]\n"
        "        return int(np.argmax(scores))\n"
        "    def update(self, a, x, y):\n"
        "        self.models[a].observe(x, y)\n"
        "        self.t += 1\n"
        "        if self.t % self.refit_every == 0:\n"
        "            for m in self.models:\n"
        "                m.fit()\n"
        "\n"
        "class EpsGreedy:\n"
        "    def __init__(self, k, d, eps=0.1, lam=1.0, refit_every=5):\n"
        "        self.models = [BayesLogReg(d, lam) for _ in range(k)]\n"
        "        self.eps = eps; self.t = 0; self.refit_every = refit_every\n"
        "    def choose(self, x):\n"
        "        if rng.random() < self.eps:\n"
        "            return int(rng.integers(0, len(self.models)))\n"
        "        scores = [x @ m.mu for m in self.models]\n"
        "        return int(np.argmax(scores))\n"
        "    def update(self, a, x, y):\n"
        "        self.models[a].observe(x, y)\n"
        "        self.t += 1\n"
        "        if self.t % self.refit_every == 0:\n"
        "            for m in self.models:\n"
        "                m.fit()\n"
        "\n"
        "class Uniform:\n"
        "    def __init__(self, k, d): self.k = k\n"
        "    def choose(self, x): return int(rng.integers(0, self.k))\n"
        "    def update(self, a, x, y): pass"
    ),
    md("## 5.4 Simulated regret"),
    code(
        "def run(policy, T=800):\n"
        "    cum = 0.0; trace = []\n"
        "    for _ in range(T):\n"
        "        x = draw_context()\n"
        "        # Best possible expected reward under full knowledge of W_true:\n"
        "        best = max(sigmoid(x @ W_true[k]) for k in range(K))\n"
        "        a = policy.choose(x)\n"
        "        r, p = pull(a, x)\n"
        "        policy.update(a, x, r)\n"
        "        cum += best - p\n"
        "        trace.append(cum)\n"
        "    return np.array(trace)\n"
        "\n"
        "T = 800\n"
        "ts  = run(Thompson(K, D))\n"
        "eg  = run(EpsGreedy(K, D, eps=0.1))\n"
        "un  = run(Uniform(K, D), T=T)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.plot(un, color='#888', lw=1.2, label='Uniform')\n"
        "ax.plot(eg, color=C_COLD, lw=1.8, label='eps-Greedy (0.1)')\n"
        "ax.plot(ts, color=C_HOT,  lw=1.8, label='Thompson sampling')\n"
        "ax.set_xlabel('t'); ax.set_ylabel('cumulative regret')\n"
        "ax.set_title('Contextual bandit regret (K=3 arms, d=6)')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'final regret: TS={ts[-1]:.2f}, eps-G={eg[-1]:.2f}, Uniform={un[-1]:.2f}')"
    ),
    md(
        "## 5.5 Posterior uncertainty shrinks with data\n"
        "\n"
        "The central aesthetic of TS: early on, the posterior on `w` is wide, so samples "
        "cover a broad range of policies (exploration). As data accumulates, the posterior "
        "concentrates, and sampling converges to the MAP (exploitation). Chapelle & Li (2011) "
        "showed this achieves near-optimal empirical regret on many contextual problems."
    ),
    code(
        "ts2 = Thompson(K, D)\n"
        "trace_std = []\n"
        "for t in range(400):\n"
        "    x = draw_context()\n"
        "    a = ts2.choose(x)\n"
        "    r, _ = pull(a, x)\n"
        "    ts2.update(a, x, r)\n"
        "    trace_std.append(np.mean([np.sqrt(np.diag(m.Sigma)).mean() for m in ts2.models]))\n"
        "fig, ax = plt.subplots(figsize=(8, 3))\n"
        "ax.plot(trace_std, color=C_HOT)\n"
        "ax.set_title('Average posterior std across arms -- decreases with t')\n"
        "ax.set_xlabel('t'); ax.set_ylabel('mean std')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 5.6 Why Thompson sampling for I3\n"
        "\n"
        "- It is *context-aware*: different user states activate different arms.\n"
        "- The Laplace approximation is cheap (O(d^3) per refit; d is small) and on-device.\n"
        "- It is *naturally exploratory* early on, a safety plus: we don't lock in a bad "
        "adaptation policy during cold-start.\n"
        "- It produces calibrated uncertainty we can surface on the dashboard."
    ),
])

write(os.path.join(HERE, "05_contextual_thompson_sampling.ipynb"), nb05)


# ---------------------------------------------------------------------------
# Notebook 06 -- Privacy by architecture
# ---------------------------------------------------------------------------

nb06 = notebook([
    md(
        "# 06 -- Privacy by Architecture\n"
        "\n"
        "**Author**: Tamer Atesyakar\n"
        "\n"
        "We examine the three concrete privacy guarantees of I3:\n"
        "1. **No raw text at rest** -- the interaction diary stores only derived summaries.\n"
        "2. **PII sanitisation** -- a regex filter scrubs emails, phone numbers, URLs, and card-like digit runs before *any* content crosses a trust boundary.\n"
        "3. **At-rest encryption** -- Fernet round-trip on all database fields tagged sensitive.\n"
        "\n"
        "**Citations**\n"
        "- Sweeney (2002). *k-Anonymity: A Model for Protecting Privacy.* IJUFKS.\n"
        "- Bernstein (2011). *Extending the Salsa20 nonce.* (underpins Fernet's AES-CBC + HMAC-SHA256).\n"
        "- Dwork & Roth (2014). *The Algorithmic Foundations of Differential Privacy.*"
    ),
    code(
        "import re\n"
        "import json\n"
        "import sqlite3\n"
        "import tempfile\n"
        "import os\n"
        "import pprint"
    ),
    md(
        "## 6.1 The diary SQLite schema\n"
        "\n"
        "The runtime diary stores *summaries and tags*, not the original user text. Below "
        "we construct an in-memory version of the schema and verify that the content column "
        "is always derived, never raw."
    ),
    code(
        "SCHEMA = '''\n"
        "CREATE TABLE diary (\n"
        "    id         INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        "    ts         REAL    NOT NULL,\n"
        "    user_id    TEXT    NOT NULL,\n"
        "    summary    BLOB    NOT NULL,    -- Fernet-encrypted derived summary\n"
        "    emotion    TEXT,\n"
        "    topics     TEXT,                -- JSON array of short tags\n"
        "    -- NOTE: no `raw_text` column exists by design\n"
        "    UNIQUE(ts, user_id)\n"
        ");\n"
        "'''\n"
        "conn = sqlite3.connect(':memory:')\n"
        "conn.executescript(SCHEMA)\n"
        "cols = [r[1] for r in conn.execute('PRAGMA table_info(diary)').fetchall()]\n"
        "print('columns:', cols)\n"
        "assert 'raw_text' not in cols and 'message' not in cols, 'raw-text leakage!'\n"
        "print('invariant holds: no raw-text columns in schema')"
    ),
    md(
        "## 6.2 The PII sanitiser\n"
        "\n"
        "Hand-crafted regex set covering email, phone, URL, and card-like digit sequences. "
        "In production this is `i3.privacy.sanitizer.scrub`."
    ),
    code(
        "RULES = [\n"
        "    ('EMAIL', re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,}')),\n"
        "    ('URL',   re.compile(r'https?://\\\\S+')),\n"
        "    ('PHONE', re.compile(r'(?<!\\\\d)(\\\\+?\\\\d[\\\\d\\\\s().-]{7,}\\\\d)(?!\\\\d)')),\n"
        "    ('CARD',  re.compile(r'(?<!\\\\d)(?:\\\\d[ -]?){13,19}(?!\\\\d)')),\n"
        "    ('SSN',   re.compile(r'(?<!\\\\d)\\\\d{3}-\\\\d{2}-\\\\d{4}(?!\\\\d)')),\n"
        "]\n"
        "\n"
        "def scrub(text: str) -> tuple[str, dict]:\n"
        "    out = text; counts = {}\n"
        "    for tag, pat in RULES:\n"
        "        out, n = pat.subn(f'<{tag}>', out)\n"
        "        if n: counts[tag] = n\n"
        "    return out, counts\n"
        "\n"
        "samples = [\n"
        "    'Contact me at tamer@example.com or 4242 4242 4242 4242',\n"
        "    'My phone is +44 7700 900123, see https://example.org/doc',\n"
        "    'SSN 123-45-6789 is sensitive, obviously.',\n"
        "    'No PII here -- just a reflection on cognitive load.',\n"
        "]\n"
        "for s in samples:\n"
        "    cleaned, counts = scrub(s)\n"
        "    print(counts, '->', cleaned)"
    ),
    md("## 6.3 Verifying the sanitiser on an adversarial test-suite"),
    code(
        "ADVERSARIAL = [\n"
        "    'tamer.a.tes@example.co.uk',              # must match EMAIL\n"
        "    '4532-1488-0343-6467',                    # test card\n"
        "    '(415) 555-0199',                         # US phone\n"
        "    'http://totally.safe.site/path?q=1',      # URL\n"
        "    '987-65-4320',                            # SSN-like\n"
        "]\n"
        "passed = 0\n"
        "for s in ADVERSARIAL:\n"
        "    _, counts = scrub(s)\n"
        "    assert counts, f'LEAK on {s!r}'\n"
        "    passed += 1\n"
        "print(f'{passed}/{len(ADVERSARIAL)} adversarial inputs scrubbed')"
    ),
    md(
        "## 6.4 Fernet round-trip\n"
        "\n"
        "Fernet composes AES-128-CBC with HMAC-SHA256 under a 256-bit key, and prepends a "
        "random 128-bit IV. It is authenticated, which matters: a bit-flipped ciphertext is "
        "rejected loudly rather than silently decrypted to garbage."
    ),
    code(
        "try:\n"
        "    from cryptography.fernet import Fernet, InvalidToken\n"
        "    CRYPTO_OK = True\n"
        "except Exception as e:\n"
        "    CRYPTO_OK = False\n"
        "    print(f'cryptography not available: {e}')\n"
        "\n"
        "if CRYPTO_OK:\n"
        "    key = Fernet.generate_key()\n"
        "    box = Fernet(key)\n"
        "    plain = json.dumps({'summary': 'reflected on pacing', 'emotion': 'calm'}).encode()\n"
        "    ct = box.encrypt(plain)\n"
        "    pt = box.decrypt(ct)\n"
        "    assert pt == plain\n"
        "    print(f'plain  : {plain}')\n"
        "    print(f'cipher : {ct[:48]}...  (len={len(ct)})')\n"
        "    print(f'round-trip ok')\n"
        "\n"
        "    # Tampering test\n"
        "    bad = bytearray(ct); bad[-1] ^= 0x01\n"
        "    try:\n"
        "        box.decrypt(bytes(bad))\n"
        "        print('UNEXPECTED: tampered ciphertext decrypted')\n"
        "    except InvalidToken:\n"
        "        print('tampered ciphertext correctly rejected')"
    ),
    md("## 6.5 End-to-end: write a diary row, read it back"),
    code(
        "if CRYPTO_OK:\n"
        "    import time as _time\n"
        "    summary_plain = 'user paused 2.3s, expressed mild frustration'\n"
        "    cleaned, _ = scrub(summary_plain)\n"
        "    enc = box.encrypt(cleaned.encode())\n"
        "    conn.execute(\n"
        "        'INSERT INTO diary (ts, user_id, summary, emotion, topics) VALUES (?,?,?,?,?)',\n"
        "        (_time.time(), 'tamer', enc, 'frustration', json.dumps(['pacing']))\n"
        "    )\n"
        "    row = conn.execute(\n"
        "        'SELECT summary, emotion, topics FROM diary LIMIT 1'\n"
        "    ).fetchone()\n"
        "    print('stored summary cipher :', row[0][:48], '...')\n"
        "    print('stored emotion        :', row[1])\n"
        "    print('stored topics         :', row[2])\n"
        "    print('decrypted summary     :', box.decrypt(row[0]).decode())"
    ),
    md(
        "## 6.6 Privacy invariants we rely on\n"
        "\n"
        "1. The schema **never** declares a raw-text column (checked via PRAGMA).\n"
        "2. All user-originating strings pass through `scrub` before summarisation.\n"
        "3. Summary payloads are encrypted with Fernet before hitting disk; tampering is detected.\n"
        "4. The bandit and user-model components consume only numeric feature vectors, "
        "so even an exfiltration of their internal state does not reveal content."
    ),
])

write(os.path.join(HERE, "06_privacy_by_architecture.ipynb"), nb06)


# ---------------------------------------------------------------------------
# Notebook 07 -- Edge profiling & quantisation
# ---------------------------------------------------------------------------

nb07 = notebook([
    md(
        "# 07 -- Edge Profiling & Quantisation\n"
        "\n"
        "**Author**: Tamer Atesyakar\n"
        "\n"
        "This notebook sizes the I3 edge model: we tabulate parameter counts, compare FP32 "
        "and INT8 storage footprints, measure inference latency percentiles, and extrapolate "
        "throughput on a Kirin-class mobile SoC.\n"
        "\n"
        "**Citations**\n"
        "- Jacob, Kligys et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR.\n"
        "- Han, Mao & Dally (2016). *Deep Compression.* ICLR.\n"
        "- Ignatov et al. (2019). *AI Benchmark: All About Deep Learning on Smartphones in 2019.* ICCVW."
    ),
    code(
        "import math, time\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "try:\n"
        "    import torch\n"
        "    import torch.nn as nn\n"
        "    TORCH_OK = True\n"
        "except ImportError:\n"
        "    TORCH_OK = False\n"
        "\n"
        "C_COLD = '#0f3460'\n"
        "C_HOT  = '#e94560'\n"
        "plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.25})"
    ),
    md("## 7.1 Parameter-count breakdown for the edge stack"),
    code(
        "COMPONENTS = [\n"
        "    ('Perception TCN',          312_000),\n"
        "    ('User Model (3x Welford)',   8_000),\n"
        "    ('Adaptation head',          90_000),\n"
        "    ('Conditioning Projector',   14_400),\n"
        "    ('Tiny transformer (4 blk)', 820_000),\n"
        "    ('Bandit (3 arms x d=6)',       120),\n"
        "]\n"
        "names = [c[0] for c in COMPONENTS]\n"
        "params = np.array([c[1] for c in COMPONENTS], dtype=np.float64)\n"
        "total = params.sum()\n"
        "print(f'total params: {total:,.0f}  ({total/1e6:.2f} M)')\n"
        "for n, p in zip(names, params):\n"
        "    print(f'  {n:30s} {p:>10,.0f}   {100*p/total:5.1f}%')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 3.6))\n"
        "colors = [C_COLD if i % 2 == 0 else C_HOT for i in range(len(names))]\n"
        "ax.barh(names, params, color=colors)\n"
        "ax.set_xlabel('parameters')\n"
        "ax.set_title('Edge stack parameter budget')\n"
        "ax.invert_yaxis()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 7.2 FP32 vs INT8 storage footprint\n"
        "\n"
        "Per-tensor symmetric INT8 (Jacob et al. 2018) stores 1 byte per weight plus a small "
        "overhead for scale and zero-point; call it ~1.02 bytes/weight amortised."
    ),
    code(
        "fp32_bytes = total * 4\n"
        "int8_bytes = total * 1.02\n"
        "fig, ax = plt.subplots(figsize=(5.2, 3.2))\n"
        "ax.bar(['FP32', 'INT8'], [fp32_bytes/1e6, int8_bytes/1e6], color=[C_COLD, C_HOT])\n"
        "ax.set_ylabel('size (MB)')\n"
        "for i, v in enumerate([fp32_bytes/1e6, int8_bytes/1e6]):\n"
        "    ax.text(i, v, f'{v:.2f} MB', ha='center', va='bottom')\n"
        "ax.set_title(f'Weight storage: {fp32_bytes/int8_bytes:.2f}x reduction')\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'FP32: {fp32_bytes/1e6:.2f} MB, INT8: {int8_bytes/1e6:.2f} MB')"
    ),
    md(
        "## 7.3 Latency percentiles on a stand-in dense layer\n"
        "\n"
        "We run a matmul of comparable size to the transformer block repeatedly and record "
        "latency percentiles. This gives us a host-machine calibration point we can then "
        "extrapolate from."
    ),
    code(
        "if TORCH_OK:\n"
        "    torch.set_num_threads(1)\n"
        "    d_model, seq = 32, 128\n"
        "    x = torch.randn(1, seq, d_model)\n"
        "    lin = nn.Linear(d_model, d_model * 4)\n"
        "    # Warm-up\n"
        "    for _ in range(10):\n"
        "        _ = lin(x)\n"
        "    N = 300\n"
        "    ts = []\n"
        "    for _ in range(N):\n"
        "        t0 = time.perf_counter()\n"
        "        _ = lin(x)\n"
        "        ts.append((time.perf_counter() - t0) * 1e3)\n"
        "    ts = np.array(ts)\n"
        "    p50, p90, p99 = np.percentile(ts, [50, 90, 99])\n"
        "    print(f'p50 = {p50:.3f} ms   p90 = {p90:.3f} ms   p99 = {p99:.3f} ms')\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=(8, 3.2))\n"
        "    ax.hist(ts, bins=40, color=C_COLD, edgecolor='white')\n"
        "    for v, c, lab in [(p50, C_HOT, 'p50'), (p90, '#555', 'p90'), (p99, '#333', 'p99')]:\n"
        "        ax.axvline(v, color=c, lw=1.5, label=f'{lab}={v:.2f} ms')\n"
        "    ax.set_xlabel('latency (ms)')\n"
        "    ax.set_title('Host-side single-layer latency (FP32, 1 thread)')\n"
        "    ax.legend()\n"
        "    plt.tight_layout(); plt.show()\n"
        "else:\n"
        "    print('torch unavailable -- using assumed host latency p50=0.35 ms')\n"
        "    p50 = 0.35"
    ),
    md(
        "## 7.4 Kirin extrapolation\n"
        "\n"
        "Huawei Kirin 9000-class NPUs deliver ~15 INT8 TOPS. Our edge model is dominated by "
        "matmul FLOPs; its single forward pass costs\n"
        "\n"
        "$$\\mathrm{FLOPs} \\approx 2 \\cdot P \\cdot T,$$\n"
        "\n"
        "where `P` is parameter count and `T` is sequence length -- the standard 2x factor "
        "(multiply + add) for dense layers.\n"
        "\n"
        "Converting to time on the NPU gives a lower bound; real devices suffer from memory-"
        "bandwidth limits and kernel launch overhead (Ignatov 2019)."
    ),
    code(
        "T_seq = 128\n"
        "flops_fp32 = 2.0 * total * T_seq          # rough upper estimate\n"
        "# Kirin NPU FP16 throughput ~8 TFLOPs, INT8 ~15 TOPS.\n"
        "kirin_fp16 = 8e12\n"
        "kirin_int8 = 15e12\n"
        "t_fp16 = flops_fp32 / kirin_fp16 * 1e3    # ms\n"
        "t_int8 = (flops_fp32 / 2) / kirin_int8 * 1e3  # halve nominal FLOPs for INT8\n"
        "print(f'Compute-bound lower bound -- Kirin FP16: {t_fp16:.3f} ms')\n"
        "print(f'Compute-bound lower bound -- Kirin INT8: {t_int8:.3f} ms')\n"
        "\n"
        "# With a 3x slowdown factor for overhead/memory-bound layers:\n"
        "overhead = 3.0\n"
        "print(f'Realistic estimate (3x overhead) -- FP16: {t_fp16*overhead:.2f} ms, INT8: {t_int8*overhead:.2f} ms')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7.2, 3.6))\n"
        "labels  = ['host FP32\\n(p50)', 'Kirin FP16\\nideal', 'Kirin FP16\\nrealistic',\n"
        "           'Kirin INT8\\nideal', 'Kirin INT8\\nrealistic']\n"
        "values  = [p50, t_fp16, t_fp16 * overhead, t_int8, t_int8 * overhead]\n"
        "colors  = [C_COLD, C_COLD, C_HOT, C_COLD, C_HOT]\n"
        "bars = ax.bar(labels, values, color=colors)\n"
        "for b, v in zip(bars, values):\n"
        "    ax.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)\n"
        "ax.set_ylabel('latency (ms)')\n"
        "ax.set_title('Edge inference latency: host vs extrapolated Kirin targets')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 7.5 Bandwidth sanity check\n"
        "\n"
        "INT8 weights for the whole edge stack (~1.27 MB) comfortably fit in L2 on Kirin-class "
        "SoCs, meaning weight-streaming bandwidth is not the bottleneck during generation."
    ),
    code(
        "int8_mb = total * 1.02 / 1e6\n"
        "print(f'INT8 weight footprint: {int8_mb:.2f} MB')\n"
        "print('Kirin 9000 L2: ~8 MB per cluster -> weights fit cleanly')"
    ),
    md(
        "## 7.6 Summary\n"
        "\n"
        "- Full edge stack: ~1.24 M parameters, ~5 MB FP32, ~1.27 MB INT8.\n"
        "- Host single-layer p50 gives a calibration point; scaled up the whole pass is "
        "~2-4 ms on desktop CPU.\n"
        "- Kirin extrapolation suggests <10 ms realistic INT8 inference, well inside the "
        "conversational budget of 200 ms.\n"
        "- Memory fits in L2, so we stay compute-bound rather than bandwidth-bound."
    ),
])

write(os.path.join(HERE, "07_edge_profiling_and_quantisation.ipynb"), nb07)


print('wrote 7 notebooks.')
