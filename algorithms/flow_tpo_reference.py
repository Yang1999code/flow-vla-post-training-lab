"""Reference Flow-TPO objective for continuous Flow Matching policies.

This code is framework-agnostic pseudocode expressed in PyTorch-like Python.
Integrate it only after verifying a common robot/action/normalization protocol.
It intentionally does not contain weights, environment access, or private data.
"""

from dataclasses import dataclass
from typing import Callable, Mapping


Observation = Mapping[str, object]
FlowLoss = Callable[[object, Observation, object], object]


@dataclass(frozen=True)
class PreferencePair:
    """A pair must share task and initial-state identifiers."""

    task_id: str
    initial_state_id: str
    chosen_observation: Observation
    chosen_actions: object
    rejected_observation: Observation
    rejected_actions: object


def validate_pair(pair: PreferencePair) -> None:
    if not pair.task_id or not pair.initial_state_id:
        raise ValueError("Preference pairs require task and initial-state IDs")
    if pair.chosen_actions is None or pair.rejected_actions is None:
        raise ValueError("Both chosen and rejected actions are required")


def flow_tpo_loss(
    policy: object,
    frozen_reference: object,
    pair: PreferencePair,
    flow_loss: FlowLoss,
    beta: float = 0.1,
    chosen_weight: float = 0.25,
):
    """Return a reference-relative surrogate preference objective.

    `flow_loss` should use matched stochastic conditions for chosen/rejected
    comparisons, such as a documented timestep/noise protocol. The reference
    must not receive gradients.
    """
    validate_pair(pair)

    chosen = flow_loss(policy, pair.chosen_observation, pair.chosen_actions)
    rejected = flow_loss(policy, pair.rejected_observation, pair.rejected_actions)
    ref_chosen = flow_loss(frozen_reference, pair.chosen_observation, pair.chosen_actions)
    ref_rejected = flow_loss(frozen_reference, pair.rejected_observation, pair.rejected_actions)

    policy_margin = -chosen + rejected
    reference_margin = -ref_chosen + ref_rejected
    relative_margin = policy_margin - reference_margin

    # Replace this pseudocode operation with the chosen framework's log-sigmoid.
    preference_term = -log_sigmoid(beta * relative_margin)
    return chosen_weight * chosen + preference_term


def log_sigmoid(value):
    """Placeholder: use torch.nn.functional.logsigmoid or jax.nn.log_sigmoid."""
    raise NotImplementedError("Use the log-sigmoid implementation of your ML framework")
