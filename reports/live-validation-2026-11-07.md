# AETHER Live Lab Validation Record

**Date:** 2026-11-07  
**Version:** 0.2.0  
**Endpoint:** http://127.0.0.1:8789  
**Study Reference:** `reports/pilot-2026-11-07`  
**Receipt Status:** Verified (26 artifacts verified against `reports/pilot-2026-11-07/receipt.json`)

---

## 1. Test Suite Conformance

- **Total Test Cases:** 220
- **Status:** 220 passed in 26.25s
- **Coverage Areas:**
  - `tests/test_conformance.py`: Differentiable clean tensor transport conformance against packet reference for all protocols (`scalar_matched`, `vector`, `attention`, `phase`, `modes`, `wave`).
  - `tests/test_evidence.py`: Seed-level paired bootstrap summaries, promotion gates, SHA256 receipt validation, and traversal protections.
  - `tests/test_live_channel.py`: Deterministic channel impairment operations (loss, noise, latency queue, bandwidth restriction).
  - `tests/test_live_models.py`: Deep copy isolation of model weights, authoritative topology, and buffer immutability.
  - `tests/test_live_server.py`: Single scheduler thread, HTTP/SSE coordination, atomic mutations, client limits, origin/host security, and study access boundaries.
  - `tests/test_live_session.py`: Paired world execution, step atomicity, waiting-period action separation, and causal metrics.
  - `tests/test_live_training.py`: Asynchronous cloned training, disjoint support/query/retention seed sets, retention gate enforcement, and rollback.
  - `tests/test_modes.py`, `tests/test_navigation.py`, `tests/test_protocol.py`, `tests/test_runtime.py`, `tests/test_study.py`, `tests/test_topology.py`: Authority, sparse scalar graph, and 3D navigation environment.

---

## 2. Live Server & Interactive Verification

The live application is running on `http://127.0.0.1:8789` via `python -m aether live --study reports/pilot-2026-11-07 --output-root runs/live --port 8789`.

### Validated Operations:
1. **Paired Execution:**
   - Evaluated active `wave-11` checkpoint (512 neurons, 16 regions, 96 links, 4 modes).
   - Clean control and intervened branch start from identical state, weights, and seed.
   - Waiting-period ticks (goal cue presentation) are segregated from effective navigation movement ticks.
2. **Channel Interventions:**
   - Tested interactive disturbance modes: phase shuffle, packet loss (0.0 to 1.0), additive noise, latency queueing (up to 4 ticks), and bandwidth throttling.
   - Live telemetry correctly captures position separation, action divergences, and return delta.
3. **Signal Observatory:**
   - Real-time RMS regional activity and mode energies calculated directly from inference tensors without duplicate forward passes.
   - Admitted and delivered payload bytes tracked at FP16 precision.
4. **Isolated Candidate Adaptation:**
   - Triggered 5 AdamW updates on a cloned copy of the 512-neuron model.
   - Evaluated before/after query and retention performance on disjoint seed sets.
   - Retention gate verified: candidate applied to live session only when longer-delay retention was preserved.
   - Provenance transitioned from `recorded` to `session_adapted`, and clean rollback restored the original `recorded` checkpoint and weights.
5. **Trace Export:**
   - Downloaded `/api/export` producing valid `aether.live.trace.v1` JSON structure bounded to 512 steps.
6. **Preserved Pilot Study:**
   - Verified that `/study/` serves the historical report and checkpoints without mutation.

---

## 3. Boundaries & Limitations

- **Exploratory Demo Only:** Live observations and candidate training are interactive exploration tools. They do not constitute scientific benchmark evidence or proof of wave communication advantage.
- **Compute Equivalence:** Resource comparability across protocols remains unestablished.
- **Single Session:** The local server coordinates a single shared session across loopback connections with mutation serialization.
