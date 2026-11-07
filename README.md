# SUPERMiX AETHER

A CPU-local research lab for **NeuroWave-X**, the first milestone of the supplied AETHER architecture. It trains a sparse, 512-neuron model with seven communication controls, executes learned actions in a delayed-cue 3D navigation task, and preserves replayable experimental evidence.

Version 0.3 advances the lab with:
- **Computational Profiler & Resource Parity**: Microsecond latency benchmarks, analytical step FLOP counters, optimizer memory accounting, and formal multi-protocol compute comparability (`aether profile`).
- **Communication Stress Task Ecology**: Multi-waypoint route navigation and sensory distractor pulse tasks that test phase-coherence filtering against intermediate cue corruption.
- **Stage P2 SynapseForge Structural Plasticity**: Causal regional link lesioning, activity-dependent Hebbian rewiring, and atomic Laplacian carrier re-computation without neuron state disruption.
- **Live Workbench v0.3**: Real-time GFLOPs / latency telemetry and interactive task family switching at [http://127.0.0.1:8789](http://127.0.0.1:8789).

Self-assembling learning, neural growth, grafting, 4D worlds, cross-project adapters, and Sonapse integration remain future roadmap stages. No other project is modified.

## Run locally

Python **3.12+**, NumPy, SciPy, and PyTorch CPU are required. A GPU and external services are unnecessary.

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m aether live --study reports/pilot-2026-11-07 --output-root runs/live --port 8789
```

Open [the live lab](http://127.0.0.1:8789), then Play. Select checkpoints, step the paired worlds, disrupt communication, train a candidate, apply eligible weights or roll back, and export the actual action trace. See [live lab controls and evidence boundaries](docs/live-lab.md). The preserved study is available at [the recorded report](http://127.0.0.1:8789/study/).

To reproduce the offline study instead:

```powershell
python -m aether run --config configs/pilot.json --output runs/pilot
python -m aether verify runs/pilot/receipt.json
python -m aether replay runs/pilot/checkpoints/wave-11.pt
python -m aether serve runs/pilot --port 8789
```

Open [the local observatory](http://127.0.0.1:8789). Run outputs are never overwritten; choose a new output directory for another study. `Ctrl+C` stops the server. The HTML report can also be opened directly from disk. Port 8789 avoids the existing SYNAPSE workbench on 8765 on this machine.

## Completed pilot

The first study is preserved in [reports/pilot-2026-11-07](reports/pilot-2026-11-07/index.html): **21 trained models** (seven protocols × three seeds), 40 updates/model, all evaluation conditions replayed locally, and **26 receipt-covered artifacts**. That milestone passed **169 tests**; the live upgrade adds its own execution, transport, training and server tests.

Wave and scalar each averaged **16.7%** query success. Scalar-matched, vector, attention and phase each averaged **20.8%**; modes averaged **16.7%**. Wave interventions did not improve success. This pilot does **not** demonstrate useful wave communication or a performance advantage. Promotion is withheld; the small sample and unestablished compute equivalence independently prevent promotion.

`configs/pilot.json` requests three independent training seeds, seven protocols, forty AdamW updates per model and twenty-four executed query episodes per seed. `configs/study-30.json` defines a larger, **unexecuted** thirty-seed study. Changing training length, learning rate, seeds or task difficulty after inspecting results is exploration and must be reported as such.

## What works

- An authoritative sparse scalar graph with **512 neurons × 32 state components**, 4,096 scalar edges, 16 regions and 96 directed region links.
- Trainable vector, attention, phase, spatial-mode and combined wave protocols, plus scalar-only and learned scalar-bottleneck controls.
- Four modes × eight complex components encoded as real pairs at FP16: **12,288 payload bytes/tick** on auxiliary routes. Headers, queue memory and scalar connections are excluded.
- Zero-initialized auxiliary gain, explicit clipped residuals, and tests for exact scalar-path identity at zero gain.
- Typed immutable packets, authority/topology/shape/budget validation, independently seeded reference disturbances, and deterministic delivery.
- A differentiable clean transport checked against validated packet delivery and independent NumPy receiver decoding; an independent NumPy scalar-step reference.
- Genuine supervised imitation training on executed expert trajectories; evaluation executes the policy's own actions without changing learned parameters or persistent buffers.
- Packet erasure, zero gain, phase shuffling and phase randomization interventions; longer-delay evaluation; replay of **every recorded evaluation condition**.
- Seed-level paired bootstrap summaries, conservative promotion gates, source snapshots, checkpoints and unsigned SHA256 artifact receipts.

All auxiliary controls instantiate the same parameter shapes and wire tensor sizes. Their active computations differ. Equal examples, update counts or total parameter counts do **not** establish equal compute. Resource comparability stays unestablished, which blocks promotion even in a larger run.

## Reading the evidence

The environment contains six fixed straight-line 3D goal tasks. Initial goal information enters the sensory region only; the action head reads a different region. The cue disappears after the first tick. Fresh query seeds are new draws from those same six tasks, **not new geometries**. The model is trained with backpropagation through time and imitation labels; this is neither local-rule learning nor reinforcement learning.

The packet reference supports loss, Gaussian noise, phase noise, same-source cross-talk, congestion erasure and latency. These mechanisms are tested independently. The training study uses clean transport; it does not establish robustness or learned adaptation to those impairments. Precision in this milestone is FP16/FP32/FP64 in the reference packets and FP16 in the model; low-bit encodings are future work.

A receipt detects accidental artifact changes; it is not a signature or proof of authenticity. Exact replay is verified locally in the recorded software/hardware environment. Cross-platform bit identity is not promised.

See [architecture](docs/architecture.md), [research status](docs/research-status.md), [roadmap](docs/roadmap.md), [design](docs/superpowers/specs/2026-11-07-neurowave-design.md), and [implementation plan](docs/superpowers/plans/2026-11-07-neurowave.md).
