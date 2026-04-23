# `i3.biometric` — keystroke-biometric identification and continuous authentication

This package layers user-identification and anomaly-detection capabilities
on top of the 64-dim behavioural embeddings produced by
`i3.encoder.tcn.TemporalConvNet`. It is a **stretch goal** from the
the original specification §9 list ("Keystroke-biometric user identification
for multi-user devices"); see also `the brief analysis` §9 for rationale.

## Contents

| Module | Purpose |
| --- | --- |
| `keystroke_id.py` | `KeystrokeBiometricID` — enrolment + identification |
| `continuous_auth.py` | `ContinuousAuthentication` — session-level drift monitoring |

## Design

### Why cosine similarity

The TCN encoder L2-normalises its output to the unit hypersphere (see
`i3/encoder/tcn.py`). For unit vectors, cosine similarity is monotone
in the Euclidean distance:

```
|a - b|^2 = 2 (1 - a · b)
```

so the same threshold ordering is obtained whether we use `a · b` or
`|a - b|`. Cosine is the natural metric on a hypersphere and matches
the loss under which the encoder is trained (NT-Xent; Chen et al.,
2020, *SimCLR*).

### Running-mean centroid update

When a user is re-enrolled, the stored centroid is updated by the
running mean formulation used in FaceNet (Schroff et al., 2015):

```
new_centroid = normalise((n * old_centroid + new_embedding) / (n + 1))
```

This is stable against bounded noise and does not require knowing the
total sample count in advance.

### Continuous authentication

Continuous authentication tracks the cosine drift
`d_i = 1 - cos_sim(centroid, embedding_i)` over a session, using
Welford's (1962) online mean/variance algorithm for numerical
stability. When `d_i > mean + 3σ`, a structured
`AuthenticationEvent` is emitted — the classical 3-sigma out-of-control
rule from statistical process control (Shewhart, 1931). This is a
drop-in anomaly detector of exactly the style evaluated by
Killourhy & Maxwell (2009) for keystroke biometrics.

### Privacy

- Centroids are encrypted at rest using `i3.privacy.encryption.ModelEncryptor`
  (Fernet / AES-128-CBC + HMAC-SHA256) with the key sourced from the
  `I3_ENCRYPTION_KEY` environment variable.
- No raw text ever reaches the biometric store.
- The SQLite schema stores only `(user_id, n_samples, encrypted, updated_at)`.
- Centroid comparison happens in-memory after decryption; ciphertexts are
  never matched against ciphertexts (no encrypted-search misfeature).

## References

- Monrose, F. & Rubin, A. (1997). *Authentication via keystroke dynamics*.
  ACM CCS '97.
- Killourhy, K. S. & Maxwell, R. A. (2009). *Comparing anomaly-detection
  algorithms for keystroke dynamics*. IEEE/IFIP DSN 2009.
- Schroff, F., Kalenichenko, D. & Philbin, J. (2015). *FaceNet: A Unified
  Embedding for Face Recognition and Clustering*. CVPR 2015.
- Welford, B. P. (1962). *Note on a method for calculating corrected sums
  of squares and products*. Technometrics 4(3), 419-420.
- Shewhart, W. A. (1931). *Economic Control of Quality of Manufactured
  Product*. D. Van Nostrand Company.
- Chen, T. et al. (2020). *A Simple Framework for Contrastive Learning of
  Visual Representations* (SimCLR). ICML 2020. — provides the NT-Xent
  training objective used by the upstream TCN encoder.
