"""Meta-learning package for few-shot user adaptation (Batch G5).

This package implements the meta-learning rebuttal to the panel critique
"the 5-message warmup is too slow". The idea is to meta-train the TCN
encoder on a *distribution* of synthetic users (the
:mod:`i3.eval.simulation.personas` library) so that, at inference time,
the encoder can identify a user's archetype in 1-2 messages via a
handful of gradient updates rather than requiring the 5-message EMA
warmup baked into the baseline pipeline.

Three complementary trainers are provided:

* :class:`MAMLTrainer` -- the second-order Model-Agnostic Meta-Learning
  algorithm of Finn, Abbeel & Levine (2017). Supports ``first_order=True``
  for the cheap FO-MAML approximation.
* :class:`ReptileTrainer` -- the simpler first-order Reptile algorithm of
  Nichol, Achiam & Schulman (2018): for each task, take ``k`` SGD steps,
  then pull the meta-weights in the direction of the adapted weights.
* :class:`FewShotAdapter` -- an inference-time wrapper that, given a
  meta-trained model and a handful of support messages from a new user,
  produces an adapted model with a cached weight delta.

Tasks are drawn from a :class:`PersonaTaskGenerator` that samples fresh
``(support, query)`` splits from each persona's simulator, giving a
reproducible task distribution that covers all eight archetypes in the
canonical HCI persona library.

References
----------
* Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic
  Meta-Learning for Fast Adaptation of Deep Networks.* ICML.
* Nichol, A., Achiam, J., & Schulman, J. (2018). *On First-Order
  Meta-Learning Algorithms.* arXiv:1803.02999 (Reptile).
* Raghu, A., Raghu, M., Bengio, S., & Vinyals, O. (2020). *Rapid
  Learning or Feature Reuse? Towards Understanding the Effectiveness of
  MAML.* ICLR (ANIL).
* Rajeswaran, A., Finn, C., Kakade, S., & Levine, S. (2019).
  *Meta-Learning with Implicit Gradients.* NeurIPS (iMAML).
"""

from __future__ import annotations

from i3.meta_learning.few_shot_adapter import FewShotAdapter
from i3.meta_learning.maml import MAMLTrainer, MetaBatch, MetaTask
from i3.meta_learning.reptile import ReptileTrainer
from i3.meta_learning.task_generator import PersonaTaskGenerator

__all__: list[str] = [
    "FewShotAdapter",
    "MAMLTrainer",
    "MetaBatch",
    "MetaTask",
    "PersonaTaskGenerator",
    "ReptileTrainer",
]
