# AETHER Advance Implementation Plan

**Goal:** Advance the AETHER research laboratory with Stress Task Ecology, Rigorous Computational Profiling for Resource Comparability, Stage P2 SynapseForge Structural Plasticity, and Live Lab v0.3 Telemetry.

**Spec:** `../specs/2026-11-07-aether-advance-design.md`

## Tasks

### Task 1: Communication Stress Task Ecology
- [x] Implement `DistractionNavigationWorld` with intermediate sensory distractor pulses during the delay.
- [x] Implement `WaypointNavigationWorld` for sequential multi-waypoint navigation.
- [x] Implement unified factory `create_world(task_family, seed, **kwargs)` and expert batch generation.
- [x] Write tests verifying deterministic seed replay, distractor perturbation, expert generation, and observation/action contract conformance (12 inputs, 7 actions).

### Task 2: Rigorous Computational Profiler & Resource Parity Engine
- [x] Implement `ModelProfiler` with explicit analytical FLOP counts for all 7 protocols (sparse MM, mode projection, attention dot-products, linear layers).
- [x] Implement CPU latency benchmarking with microsecond resolution and memory accounting (parameters, buffers, activation peak, AdamW optimizer moments).
- [x] Implement `verify_resource_comparability(candidate_profile, baseline_profiles, tolerance=1.5)` returning verifiable evidence dict.
- [x] Integrate profiler with `TissueModel.resource_metrics()` and wire into `StudyConfig` / `promotion_decision`.
- [x] Write tests verifying FLOP monotonicity, memory accounting accuracy, benchmark repeatability, and comparability gating.

### Task 3: Stage P2 SynapseForge Structural Plasticity Engine
- [x] Implement `StructuralLesion` for selective ablation of regional links and scalar bundles.
- [x] Implement `reconcile_model_topology` to re-derive sparse COO indices, degree normalizers, and orthonormal Laplacian modes via `eigh` with sign alignment.
- [x] Implement `HebbianPlasticityEngine` for activity-dependent link pruning and degree-budgeted synaptogenesis.
- [x] Write tests verifying topology invariance, buffer reconciliation correctness, non-disruption of model parameters, and lesion impact.

### Task 4: Live Workbench & CLI Integration
- [x] Add task family selection and compute profiling to `LiveSession` and `/api/control` / `/api/state`.
- [x] Update browser UI with task family dropdown, live compute readout (GFLOPs, latency us, memory KB), and lesion toggle.
- [x] Add `aether profile` CLI command for instant multi-protocol compute benchmarking.
- [x] Write integration tests covering live state, task switching, and CLI commands.

### Task 5: Verification & Full Conformance
- [x] Run full pytest test suite (target: all tests passing).
- [x] Run receipt verification on `reports/pilot-2026-11-07/receipt.json`.
- [x] Execute live server verification on port 8789.
