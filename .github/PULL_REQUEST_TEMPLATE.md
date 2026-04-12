<!--
Thanks for contributing to Implicit Interaction Intelligence (I³)!

Please fill out this template to help reviewers evaluate your PR quickly. Keep
descriptions focused and avoid copy-pasting code that is already in the diff.
-->

## Summary

<!-- One or two sentences describing what this PR does and why it matters. -->

## Type of Change

<!-- Check all that apply. -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Refactor (no functional change)
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Test coverage improvement
- [ ] CI / tooling / build change
- [ ] Security fix

## Related Issues

<!--
Link any issues this PR addresses. Use GitHub's "Closes #123" / "Fixes #456"
syntax to automatically close them when this PR is merged.
-->

Closes #

## Changes Made

<!--
High-level list of the concrete changes. Group by component where helpful.
Example:
- `i3/perception/tcn_encoder.py`: added `mask` parameter for variable-length inputs
- `tests/unit/test_tcn_encoder.py`: new parametrized tests for masked pooling
-->

-
-
-

## Testing

<!--
Describe how you validated the change. Include commands, outputs, or screenshots
where useful. If you added tests, call them out explicitly.
-->

- [ ] `poetry run pytest` passes locally
- [ ] New tests added for changed behavior
- [ ] Manually exercised the demo / API / WebSocket flow where relevant
- [ ] Coverage has not regressed

**Test commands run:**

```bash
poetry run pytest -v
poetry run ruff check i3/ server/ training/ tests/
poetry run mypy i3/
```

## Checklist

- [ ] My code follows the project's [style guide](CONTRIBUTING.md#code-style) (Ruff + mypy clean)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code in hard-to-understand areas
- [ ] I have updated the documentation (README, ARCHITECTURE, docstrings) where needed
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally
- [ ] I have checked that no secrets, credentials, or PII are included in the diff
- [ ] I have updated `CHANGELOG.md` under `[Unreleased]` if the change is user-visible
- [ ] I have considered privacy and security implications of this change
- [ ] I understand this project is privacy-by-architecture and my change does not introduce unencrypted PII storage, unbounded telemetry, or third-party trackers

## Screenshots / Demos (if UI change)

<!--
If this PR changes the frontend or visualization, add screenshots or a short
GIF/video. For API changes, consider attaching example request/response JSON.
-->

## Additional Notes

<!--
Anything reviewers should know: caveats, follow-up work, open questions, etc.
-->
