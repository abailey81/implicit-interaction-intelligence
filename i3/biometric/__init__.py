"""Keystroke-biometric identification and continuous authentication.

This package provides two complementary capabilities that sit on top of
the 64-dim embeddings produced by
:class:`~i3.encoder.tcn.TemporalConvNet`:

- :class:`~.keystroke_id.KeystrokeBiometricID` -- enrolment and
  identification of users by keystroke-dynamics centroid (Monrose &
  Rubin, 1997).
- :class:`~.continuous_auth.ContinuousAuthentication` -- per-session
  drift monitor that flags a *low-confidence-user* event when the
  current embedding drifts more than three standard deviations from
  the registered centroid (Killourhy & Maxwell, 2009).

References
----------
- Monrose, F. & Rubin, A. (1997). *Authentication via keystroke
  dynamics*.  ACM CCS '97.
- Killourhy, K. S. & Maxwell, R. A. (2009). *Comparing anomaly-detection
  algorithms for keystroke dynamics*.  IEEE/IFIP DSN 2009.
- Chen, T. et al. (2020).  *SimCLR* -- basis for the TCN's L2-normalised
  hypersphere embeddings which make cosine similarity the natural
  distance metric here.
"""

from i3.biometric.continuous_auth import (
    AuthenticationEvent,
    ContinuousAuthentication,
)
from i3.biometric.keystroke_id import (
    IdentificationResult,
    KeystrokeBiometricID,
)

__all__ = [
    "AuthenticationEvent",
    "ContinuousAuthentication",
    "IdentificationResult",
    "KeystrokeBiometricID",
]
