# AETHER integrated adaptation implementation plan

**Goal:** Complete the existing stress-task, plasticity, profiling and workbench contracts as one usable, reproducible exploratory lab.

**Architecture:** Retain Python 3.12, NumPy, SciPy, PyTorch CPU and the existing HTTP/canvas workbench. Thread the selected task through support generation, evaluation, checkpoints and replay. Treat structural edits as validated transactions and use only executed regional activity for live rewiring.

**Scope basis:** The existing `2026-11-07-aether-advance-design.md`, the requested whole-project improvement, and reproducible defects found during source inspection. This is a completion and reliability pass; no claim of a new learning advantage is assumed.

## Decisions

- Complete and integrate the existing features before expanding into neural growth or cross-project adapters. Adding another architecture would leave the demonstrated task and topology inconsistencies unresolved; a cosmetic workbench pass would leave training on the wrong task.
- Retain the negative original pilot unchanged and load checkpoints without a task field as straight navigation.
- Live candidate and clean control continue to compare channel interventions on the same selected model. Structural edits affect that selected model in both branches and are explicitly labeled; rollback provides the previous model.
- Correlation-driven rewiring is an exploratory heuristic. Two measured ticks are the minimum input; never substitute synthetic data.
- Analytical resource estimates and observed inference latency remain separate from formal promotion. Reject missing or invalid comparison measurements.
- Keep all work local on `codex/aether-integrated-adaptation`; no publication or long confirmatory study.

## Tasks

- [x] Runtime: reproduce atomicity and equal-edge-count remapping failures; validate/rebuild derived buffers before committing changes; preserve unchanged parameter identities; validate plasticity inputs and finite activity; make profile comparison fail closed and label estimates honestly.
- [x] Study: add validated `task_family='straight'`, use the shared world/expert factory throughout training and evaluation, persist the task in checkpoints, and replay all conditions for all families while preserving old checkpoints.
- [x] Live training: propagate task family and delay through support/query/retention, records and candidate identity; report exact family evidence boundaries.
- [x] Live execution: record actual regional activity, refuse premature rewiring, keep structural provenance and reports, clear incompatible activity on resets/model changes, and expose current goal/radius/waypoint progress.
- [x] Coordinator/workbench: invalidate stale candidates on task changes, make structural edits reversible, expose lesion/rewire controls and readiness, and render current task details and measured inference latency.
- [x] Verification: run targeted regressions then the full suite; verify original receipt and checkpoint replay; execute a small all-family recorded smoke study, verify receipts/replay, inspect the live workbench, and update research documentation with measured outcomes and remaining limits.

## Review focus

- Rejected topology changes preserve the original model, config, digest and parameter identities.
- Existing checkpoints and receipts remain usable without migration.
- Switching task family invalidates a candidate trained on another family.
- Task telemetry follows executed environment state without entering the policy observation.
- Rewiring requires recorded finite activity and reports actual edited edges; mutation remains cloneable and rollback works.

## Validation commands

`python -m pytest -q`

`python -m aether verify reports/pilot-2026-11-07/receipt.json`

`python -m aether replay reports/pilot-2026-11-07/checkpoints/wave-11.pt`

New reduced smoke studies are exploratory software integration checks, not replications or evidence of superiority.
