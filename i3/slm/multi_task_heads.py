"""Auxiliary multi-task heads attached to the TCN user-state embedding.

Three lightweight classifier / regressor MLPs that sit on top of the
64-dimensional :class:`~i3.encoder.tcn.TemporalConvNet` output.  They are
architecturally **independent** of the SLM decoder -- the decoder's
forward pass never touches them, so they add zero inference-time cost
unless the caller explicitly invokes a head.  During training they are
supervised with auxiliary losses that complement the main language-modelling
objective.

Why these three heads (Huawei HMI rationale)
--------------------------------------------
Each head demonstrates a distinct value-add that a from-scratch encoder
brings to a Human-Machine Interface stack:

(a) :class:`TypingBiometricsHead` -- user-identification from keystroke
    dynamics, the core primitive behind password-less continuous
    authentication on a phone / wearable.  By showing that the 64-dim
    TCN embedding carries enough information to classify the active
    user out of a closed set, we substantiate the security story
    without having to train a dedicated authentication model.

(b) :class:`AffectHead` -- seven-way emotion classification (Ekman's
    six primary emotions plus "neutral").  Emotion recognition from
    typing patterns is the cleanest demonstration that the implicit
    behavioural signal captures affective state, which is what drives
    the adaptation controller's ``emotional_tone`` dimension.

(c) :class:`ReadingLevelHead` -- regression of the Flesch-Kincaid grade
    level of the user's input.  This closes the loop with the
    ``cognitive_load`` dimension of the adaptation vector: the encoder
    learns to predict the complexity of the user's own writing, which
    is precisely the signal we use to modulate response complexity.

Implementation notes
--------------------
* Every head is a 2-layer MLP with GELU + dropout.  Keeping them small
  and identical in structure makes the multi-task training setup
  uniform: one optimizer, one loss weighting scheme per head.
* Forward signatures are intentionally minimal -- each takes only the
  embedding and returns raw logits / a scalar so that training code
  composes the loss (cross-entropy or MSE) in one place.
* No external dependencies beyond ``torch``.  No pretrained backbones.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _init_linear(module: nn.Linear) -> None:
    """Xavier-uniform weight init with zero bias -- used by every head."""
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# (a) Typing biometrics
# ---------------------------------------------------------------------------

class TypingBiometricsHead(nn.Module):
    """Closed-set user classifier on top of the TCN embedding.

    A 2-layer MLP that maps the 64-dim user-state embedding to logits
    over ``n_users``.  Trained with standard cross-entropy: the goal is
    to demonstrate that the TCN representation is user-identifying,
    which is the core assumption behind behavioural biometrics.

    Parameters
    ----------
    embedding_dim : int, optional
        Dimensionality of the TCN output (default 64).
    n_users : int, optional
        Number of enrolled users (default 256).  This is a stand-in;
        the real value is set at config time based on the enrolment
        cohort.
    hidden_dim : int, optional
        Hidden width of the MLP (default 128).  Following the task
        spec, the standard layout is ``64 -> 128 -> n_users``.
    dropout : float, optional
        Dropout between the two layers (default 0.1).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_users: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or n_users <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"invalid dimensions: embedding_dim={embedding_dim}, "
                f"n_users={n_users}, hidden_dim={hidden_dim}"
            )

        self.embedding_dim: int = embedding_dim
        self.n_users: int = n_users

        self.fc1: nn.Linear = nn.Linear(embedding_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, n_users)
        self.act: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, 64]`` -> ``[batch, n_users]`` logits."""
        if embedding.dim() != 2 or embedding.size(-1) != self.embedding_dim:
            raise ValueError(
                f"expected embedding shape [batch, {self.embedding_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        h = self.act(self.fc1(embedding))
        h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# (b) Affect / emotion
# ---------------------------------------------------------------------------

class AffectHead(nn.Module):
    """Seven-way emotion classifier (Ekman 6 + neutral).

    Label layout (matches the order used by the adaptation data-prep
    stage in ``training/prepare_dialogue.py``)::

        0 neutral, 1 joy, 2 sadness, 3 anger, 4 fear, 5 surprise, 6 disgust

    Parameters
    ----------
    embedding_dim : int, optional
        TCN embedding dimension (default 64).
    n_emotions : int, optional
        Number of output classes (default 7).
    hidden_dim : int, optional
        Hidden width of the MLP (default 128).
    dropout : float, optional
        Dropout between layers (default 0.1).
    """

    EMOTION_LABELS: tuple[str, ...] = (
        "neutral",
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
    )

    def __init__(
        self,
        embedding_dim: int = 64,
        n_emotions: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or n_emotions <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"invalid dimensions: embedding_dim={embedding_dim}, "
                f"n_emotions={n_emotions}, hidden_dim={hidden_dim}"
            )

        self.embedding_dim: int = embedding_dim
        self.n_emotions: int = n_emotions

        self.fc1: nn.Linear = nn.Linear(embedding_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, n_emotions)
        self.act: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, 64]`` -> ``[batch, n_emotions]`` logits."""
        if embedding.dim() != 2 or embedding.size(-1) != self.embedding_dim:
            raise ValueError(
                f"expected embedding shape [batch, {self.embedding_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        h = self.act(self.fc1(embedding))
        h = self.dropout(h)
        return self.fc2(h)


# ---------------------------------------------------------------------------
# (c) Reading level regression
# ---------------------------------------------------------------------------

class ReadingLevelHead(nn.Module):
    """Flesch-Kincaid grade-level regressor.

    Produces a single scalar approximating the U.S. grade level of the
    user's own writing (typically 1-18).  Trained with MSE against the
    reference grade computed by
    :func:`training.prepare_dialogue.compute_flesch_kincaid_grade`.

    Parameters
    ----------
    embedding_dim : int, optional
        TCN embedding dimension (default 64).
    hidden_dim : int, optional
        Hidden width of the MLP (default 128).
    dropout : float, optional
        Dropout between layers (default 0.1).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"invalid dimensions: embedding_dim={embedding_dim}, "
                f"hidden_dim={hidden_dim}"
            )

        self.embedding_dim: int = embedding_dim

        self.fc1: nn.Linear = nn.Linear(embedding_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, 1)
        self.act: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, 64]`` -> ``[batch]`` scalar grade prediction."""
        if embedding.dim() != 2 or embedding.size(-1) != self.embedding_dim:
            raise ValueError(
                f"expected embedding shape [batch, {self.embedding_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        h = self.act(self.fc1(embedding))
        h = self.dropout(h)
        # Squeeze the trailing dim so the output is [batch], not [batch, 1],
        # matching the convention used by the MSE training target.
        return self.fc2(h).squeeze(-1)
