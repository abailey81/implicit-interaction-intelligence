"""Run red-team harness against sanitizer/PDDL/guardrails without torch.

The main ``scripts/security/run_redteam.py`` eagerly imports ``i3``, which imports
``i3.interaction.types`` (which imports torch).  On Windows boxes where
torch fails to load ``c10.dll`` (missing VC++ redistributable), we cannot
exercise even the non-torch target surfaces.

This thin wrapper stubs ``torch`` in ``sys.modules`` before anything else
imports it, so the sanitizer / PDDL / guardrails surfaces stay runnable.
"""
from __future__ import annotations

import sys
import types

# ---- Stub torch so i3.__init__ can import without a real torch install.
_torch_stub = types.ModuleType("torch")

class _Tensor:
    pass

def _tensor(*args, **kwargs):
    return _Tensor()

_torch_stub.Tensor = _Tensor
_torch_stub.tensor = _tensor
_torch_stub.float32 = "float32"
_torch_stub.no_grad = lambda: _Noop()
class _Noop:
    def __enter__(self): return self
    def __exit__(self, *a): return False
sys.modules.setdefault("torch", _torch_stub)

# Now run the real CLI.
import runpy
sys.argv = ["run_redteam"] + sys.argv[1:]
runpy.run_module("scripts.security.run_redteam", run_name="__main__")
