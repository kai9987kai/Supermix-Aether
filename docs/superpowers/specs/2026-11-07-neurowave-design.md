# SUPERMiX AETHER: NeuroWave-X first milestone

The supplied 7 November 2026 proposal is the project brief. The user confirmed “build it”. This milestone implements its final, bounded target: a standalone CPU research runtime and a reproducible 512-neuron communication experiment. The full AETHER architecture remains a documented roadmap.

## Approach

A new Python package uses trainable PyTorch CPU dynamics and a NumPy reference step. Its authoritative immutable topology contains 512 neurons, 4,096 directed scalar edges, 16 regions, and 96 directed message links. Every neuron has 32 state components. Four modes carry eight complex components, encoded as paired real values, at FP16: 12,288 payload bytes per tick. The scalar route is preserved with a strictly zero-initialized, separately trainable residual wave gain.

The alternatives were modifying Sonapse in place or scaffolding every proposed subsystem. A standalone, functioning first experiment follows the brief's explicit sequence and enables causal comparison. This is the builder's implementation choice; other repositories are neither changed nor implicitly imported.

## Runtime and protocol

Protocols are scalar-only, vector, attention, phase-only, mode-only, and combined wave. The five auxiliary protocols use the same dense encoder/decoder parameter shapes and transmitted tensor shape. A learned scalar bottleneck control is added. Report actual parameters, wire payload, and measured time; equal training examples do not establish equal compute.

The packet simulator validates immutable payloads, sender authority, recipient identity, topology versions, precision, shape, finite values, tick order, and per-tick payload budgets. It supports independently seeded loss, noise, cross-talk, latency, phase disruptions and congestion. Training uses a differentiable clean tensor transport with matching quantization; the CLI checks clean transport conformance against the packet reference route before running experiments. Impairments are separately tested reference capabilities. Headers and queue memory are excluded from payload counts and identified as such.

The symmetric normalized graph Laplacian supplies routing carriers while directed links retain direction. Sign canonicalization and alignment inside equal-eigenvalue subspaces preserve orthogonality and eigenvector validity. Topology changes require rebuilding all derived buffers; growth is disabled in this milestone.

## Learning and evidence

A partially observed, delayed-context 3D grid-navigation task supplies an initial hidden-goal cue, a waiting interval, and executed movement actions. Supervised imitation trains on executed teacher trajectories; evaluation rolls out the learned policy's actual actions on disjoint seeds. This is imitation learning, not reinforcement learning. Query evaluation may update transient recurrent state but cannot mutate learned parameters or persistent buffers.

Compare all communication protocols on identical training examples, budgets, topology and paired seeds. Evaluate zero gain, phase shuffling/randomization, packet erasure, and delayed-cue retention. Store checkpoints, config, source fingerprint, environment versions, learning curves, per-seed outcomes, hashes, and promotion reasons. Exact replay is checked on the same software/hardware; cross-platform bit identity is not promised.

Paired confidence intervals use independent training seeds. A small pilot is explicitly exploratory. Promotion requires positive lower paired interval versus every predeclared control, retention within margin, valid integrity/replay, and comparable measured resource budgets; otherwise preserve a rejected or inconclusive result. No broad intelligence, biological, self-assembly or superiority claim follows from implementation.

## Delivery

Provide installable package, CLI, configs, tests, source-backed static HTML observatory, report and roadmap. Run meaningful tests and an actual bounded pilot. A 30-seed configuration is supplied for a later exploratory study but is not represented as executed. Sonapse source preservation is unverified until an explicit source-pinned adapter study exists. P2–P5 are not stubbed as working capabilities.
