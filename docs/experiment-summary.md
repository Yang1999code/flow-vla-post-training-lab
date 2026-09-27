# Experiment Summary and Limitations

## Experimental Scope

The experiments used continuous Flow Matching VLA policies in two simulated
settings. They are intended to validate an engineering workflow, not to claim
production readiness or broad generalization.

| Setting | Starting policy | Post-training focus |
|---|---|---|
| RoboTwin click_bell | pi0.5 LoRA SFT policy | efficiency, action smoothness, and recovery under controlled OOD conditions |
| LIBERO-10 task 9 | released `pi0_libero` policy | trajectory preference and student-state teacher distillation |

## Model Roles

### RoboTwin

```text
pi0.5 pretrained model
  -> LoRA SFT-1000
  -> Flow-TPO student
  -> Flow-OPD student

Frozen SFT-1000:
  - TPO reference policy
  - OPD action teacher
```

The OPD teacher outputs protocol-compatible `50 x 14` bimanual action chunks.
It is not a reward model and does not take over during evaluation.

### LIBERO

```text
released pi0_libero
  -> Flow-TPO20 student
  -> Flow-OPD20 student

Frozen pi0_libero:
  - TPO reference policy
  - OPD action teacher
```

The LIBERO teacher outputs a protocol-compatible `50 x 7` action chunk.

## What Worked

1. Corrected the data contract so chosen and rejected trajectories retain
   their own observations rather than sharing an incompatible state.
2. Ran a memory-aware sequential-gradient version of paired Flow-TPO on
   V100-class GPUs.
3. Ran OPD-style action distillation on student-visited states, with student
   evaluation performed without teacher takeover.
4. Observed a strong positive signal on one frozen LIBERO protocol and a
   recovery signal for a degraded RoboTwin TPO student.

## What Did Not Work

| Variant | Observed result | Interpretation |
|---|---|---|
| More TPO updates (TPO30) | Regressed relative to TPO20 | Small paired data can overfit |
| Fine ranking among successful trajectories | Regressed | Small quality differences produced noisy preference supervision |
| Strong OPD replay anchor | Regressed | The anchor can suppress teacher correction |
| Hard LIBERO task without success data | All sampled trajectories timed out | No credible chosen trajectory exists for TPO |

## Evaluation Protocol

For a valid comparison, hold fixed or explicitly report:

```text
task ID
initial-state ID
environment seed
sampling-noise manifest
action horizon / replan interval
observation and normalization protocol
student checkpoint hash
```

Report success rate, mean steps, successful-trajectory steps, wall-clock
latency, action-difference jerk proxy, and failure modes together.

## Interpreting the Result Tables

The results in `data/` are aggregate metrics copied from internal experiment
logs after removing raw trajectories and private infrastructure details.

- A success increase under one fixed sampling noise is a signal, not proof of
  robust generalization.
- Jerk is a proxy computed from adjacent actions. It is not a force, torque,
  collision, or safety certification.
- Flow-TPO uses a surrogate score. A lower Flow Matching loss does not by
  itself prove better environment performance.

## Recommended Next Experiment

Use a non-saturated task with successful demonstrations or a compatible
recovery teacher. Build a fixed development/test split with at least 50
independent initial states, collect multiple sampling-noise rollouts per state,
annotate stage progress and failures, then compare equal-budget SFT, weighted
Flow Matching, TPO, OPD, and TPO+OPD.
