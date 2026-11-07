# NeuroWave-X Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver a functioning CPU NeuroWave-X research prototype with trained controls and replayable evidence.

**Architecture:** Independent packet-reference, tensor-runtime, and evidence modules join a deterministic navigation experiment. Root coordinates integration and the observatory; workers own disjoint files.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, SciPy, pytest; standard-library HTML and HTTP serving.

**Spec:** ../specs/2026-11-07-neurowave-design.md

## Global Constraints

- Default topology: 512 neurons, 4096 scalar edges, 16 regions, 96 directed message links, state dimension 32.
- Wave payload: 4 modes × 8 complex dimensions × 2 components × FP16 = 128 bytes/link, 12288 bytes/tick.
- Initial residual gain is exactly 0.0; growth and local-rule research disabled.
- No edits or implicit imports in other projects. Real learning and executed evaluation; preserve negative results.
- Query evaluation freezes parameters and persistent buffers. Reproducibility is scoped to the recorded runtime environment.

## Review Focus

- Invalid packets and authority spoofing fail before queue/RNG state changes.
- Zero-gain wave output equals the scalar path exactly.
- Degenerate spectral bases retain eigenvector validity after alignment.
- Evaluation seeds and goal observations cannot leak into training or policy input.
- A small pilot or resource mismatch must fail closed for promotion.

### Task 1: Packet reference and spectral routing

**Files:** aether/neurowave/{protocol,channel_simulator,graph_modes}.py; tests/test_protocol.py; tests/test_modes.py.

**Interfaces:** `NeuralPacket(sender_id, receiver_id, timestep, mode_count, payload, precision_bits, topology_version)` with payload shaped (modes, complex_dim, 2). `ChannelConfig`, `PacketBus` document their public constructors. `graph_modes(adjacency, mode_count, previous=None)` returns eigenvalues and orthonormal carriers.

- [x] Write and run failing tests for packet validation, budgets, deterministic impairment/latency, and spectral orthogonality/alignment.
- [x] Implement immutable validated packets, transactionally validated delivery, seeded disturbances, normalized Laplacian and stable eigenspace alignment.
- [x] Run focused tests and report exact commands/results.

### Task 2: Trainable tensor runtime

**Files:** aether/config.py; aether/neurofabric/topology.py; aether/kernel/{runtime,reference}.py; tests/test_runtime.py.

**Interfaces:** `ModelConfig` is flat and serializable. `TissueModel(config, protocol, seed)`, `initial_state(batch_size)`, `step(observation, state, intervention='none', disturbance_seed=0)` returns logits/state; `forward(observations, intervention='none', disturbance_seed=0)` handles batch/time. `state_digest()` fingerprints parameters and persistent buffers. Topology exposes scalar_sources/targets and region_sources/targets. Protocol names: scalar, scalar_matched, vector, attention, phase, modes, wave. Observations have 12 features, actions 7.

- [x] Write and run failing tests for exact zero-gain identity, finite gradients, topology counts, replay and frozen query parameters.
- [x] Implement trainable sparse dynamics, graph carriers, phase gates, quantization, interventions and NumPy scalar reference.
- [x] Compare PyTorch scalar step with independent NumPy calculations and run focused tests.

### Task 3: Evidence and receipts

**Files:** aether/evidence/{statistics,promotion,receipts}.py; tests/test_evidence.py.

**Interfaces:** `paired_summary(candidate, baseline, seed=0)` returns JSON-safe n/mean/ci95; `promotion_decision(...)` documents explicit gate arguments; receipts expose write/verify functions and deterministic SHA256 source/file fingerprints.

- [x] Write and run failing tests for seed-level pairing, nonfinite inputs, insufficient samples, rejected resource/integrity gates, and modified-artifact detection.
- [x] Implement bootstrap intervals, fail-closed promotion, append-preserving hashed receipts and safe verification paths.
- [x] Run focused tests and report evidence.

### Task 4: Navigation, training and study CLI

**Files:** aether/environments/navigation.py; aether/experiments/study.py; aether/cli.py; aether/__main__.py; pyproject.toml; configs/*.json; tests/test_navigation.py; tests/test_study.py.

- [x] Test cue disappearance, action effects, split seeds, successful expert rollouts, checkpoint replay and query immutability before implementation.
- [x] Implement executed expert trajectories, AdamW imitation training, learned action evaluation, paired study loop and explicit scientific gates.
- [x] Implement run, replay, verify, and serve commands with valid configuration checking and non-overwriting output directories.
- [x] Run a bounded real pilot with all seven controls and recorded causal interventions.

### Task 5: Observatory, review and documentation

**Files:** aether/observatory.py; README.md; docs/{architecture,research-status,roadmap}.md; reports/pilot/*.

- [x] Build a readable static observatory from actual recorded outcomes and inspect its rendered UI.
- [x] Document commands, architecture, implementation boundaries, source/API references and pilot limitations.
- [x] Run full tests, package/CLI checks, exact replay and receipt verification; request independent code/scientific review and address material findings.

