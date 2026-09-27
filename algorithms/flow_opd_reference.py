"""Reference Flow-OPD objective for teacher action distillation.

The teacher is a frozen, protocol-compatible action policy. It is not a
trajectory reward model and does not take over during student evaluation.
"""

from dataclasses import dataclass
from typing import Callable, Mapping


Observation = Mapping[str, object]
FlowLoss = Callable[[object, Observation, object], object]


@dataclass(frozen=True)
class StudentVisitedState:
    trajectory_id: str
    initial_state_id: str
    observation: Observation
    prompt: str
    teacher_actions: object
    original_student_actions: object
    action_protocol: str


def validate_protocol(sample: StudentVisitedState, expected_protocol: str) -> None:
    if sample.action_protocol != expected_protocol:
        raise ValueError("Teacher/student action protocol mismatch")
    if sample.teacher_actions is None or sample.original_student_actions is None:
        raise ValueError("Teacher and original student chunks are both required")


def flow_opd_loss(
    student_policy: object,
    sample: StudentVisitedState,
    flow_loss: FlowLoss,
    expected_protocol: str,
    replay_weight: float = 0.1,
):
    """Distill teacher actions while retaining a weak student replay anchor."""
    validate_protocol(sample, expected_protocol)
    teacher_loss = flow_loss(student_policy, sample.observation, sample.teacher_actions)
    replay_loss = flow_loss(student_policy, sample.observation, sample.original_student_actions)
    return teacher_loss + replay_weight * replay_loss
