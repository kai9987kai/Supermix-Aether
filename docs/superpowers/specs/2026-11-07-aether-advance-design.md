# AETHER Advance: Stress Ecology, SynapseForge Plasticity & Resource Profiling

This specification advances the SUPERMiX AETHER research laboratory from exploratory NeuroWave-X pilot infrastructure (P1) into communication-stressed task ecology, rigorous computational profiling for resource comparability, and Stage P2 SynapseForge structural plasticity.

## 1. Motivation and Problem Statement

The initial 7 November 2026 pilot established CPU execution, 7 protocol controls, and receipt-backed verification, but concluded with:
1. **Zero wave advantage** on the 6-direction straight-line navigation task because simple scalar persistence easily solves straight paths without inter-regional auxiliary carrier modulation.
2. **Resource comparability withheld (`resources_comparable=False`)** because tensor shape matching does not measure active FLOPs, optimizer state, memory peaks, or CPU step latency.
3. **Stage P2 (EvoTissue / SynapseForge)** remained proposed without concrete implementation for structural rewiring or lesion repair.

## 2. Architecture & Components

```mermaid
flowchart TD
    subgraph Task Ecology
        S[NavigationWorld: Straight 3D]
        D[DistractionWorld: Intermediate noise pulses]
        W[WaypointWorld: Sequential multi-goal routes]
    end

    subgraph Computational Profiler
        F[Theoretical FLOPs counter]
        M[Peak activation & optimizer memory]
        L[Calibrated CPU latency benchmark]
        C[Comparability Certificate]
    end

    subgraph SynapseForge Plasticity
        LE[Structural Lesion Injection]
        PR[Activity-Dependent Pruning]
        SY[Budgeted Synaptogenesis]
        RE[Orthonormal Laplacian Mode Recomputation]
    end

    subgraph Live Neural Lab v0.3
        UI[Interactive Task Family Selector]
        TEL[Real-time Latency & FLOPs Readout]
        LES[Interactive Lesion & Plasticity Workbench]
    end

    Task Ecology --> TissueModel
    Computational Profiler --> Promotion Gate
    SynapseForge Plasticity --> TissueModel
    TissueModel --> Live Neural Lab v0.3
```

### Component A: Communication Stress Task Ecology (`aether/environments/stress.py`)
- **`DistractionNavigationWorld`**:
  - Inherits standard observation/action contracts (12 inputs, 7 actions).
  - During the waiting delay (ticks `1 <= t < delay`), emits structured distractor pulses on sensory channels (e.g. orthogonal false direction cues with decayed amplitude).
  - Demands phase-coherent filtering or attention queries to isolate the true cue emitted at tick 0 from distractor noise.
- **`WaypointNavigationWorld`**:
  - Requires reaching waypoint A before waypoint B within the horizon.
  - Demands multi-channel carrier modulation where primary and secondary waypoints are encoded into orthogonal mode carriers.
- Preserves full deterministic reproducibility, strict seed isolation, and compatibility with the expert trajectory generator.

### Component B: Computational Profiler & Resource Parity (`aether/kernel/profiler.py`)
- Explicit FLOP accounting:
  - Sparse graph message passing: `2 * E_scalar * D_state`
  - Mode projections: `2 * M * R * D_complex`
  - Attention queries/keys: `2 * R * D_state * D_query`
  - Elementwise, nonlinear activations, and layer normalizations.
- Memory profiler:
  - Parameter bytes, buffer bytes, optimizer moments (AdamW fp32 states).
  - Peak activation tensor memory during backpropagation.
- Latency benchmark:
  - Microsecond CPU timing over calibrated iterations.
- Comparability certificate:
  - Evaluates whether candidate compute (FLOPs + latency) is within a defined ratio (default <= 1.5x) of baseline controls.

### Component C: SynapseForge Structural Plasticity (`aether/neurofabric/plasticity.py`)
- **`lesion_region_links(model, source_region, target_region)`**:
  - Causally severs directed communication links between chosen regions.
- **`reconcile_model_topology(model, new_topology)`**:
  - Atomically reconstructs COO index buffers, degree vectors, and recomputes orthonormal spatial eigenvectors via SciPy `eigh`, maintaining sign conventions without disrupting neuron recurrent state.
- **`adapt_topology_hebbian(model, activity_history, pruning_rate, growth_rate)`**:
  - Prunes under-utilized edges and synthesizes new connections along high-correlation pathways while preserving degree bounds.

### Component D: Live Lab v0.3 Integration
- Expose task family switching in `/api/control` (`action: 'configure', task_family: ...`).
- Stream computational profile metrics in `/api/state` (`profile: {flops_per_step, latency_us, peak_memory_kb}`).
- Render task-specific visual indicators (distractor pulses, waypoints) in the 3D projection view.

## 3. Verification & Acceptance Criteria
- 100% preservation of all existing 220 tests and the original pilot receipt `reports/pilot-2026-11-07/receipt.json`.
- Comprehensive unit and integration test coverage for stress environments, computational profiler, plasticity engine, and live server endpoints.
- Verifiable promotion gate evaluation demonstrating how `resources_comparable` is established through empirical profiling.
