# AETHER Live Lab

The user requested innovation across the project and a live feature, after authorizing construction and launch. Extend the existing CPU runtime with a functioning local laboratory; preserve the original pilot and its negative/inconclusive conclusions.

## Selected approach

Use the existing Python/PyTorch process, standard-library HTTP and server-sent events, plus dependency-free browser canvas/SVG. Keep the local application on port 8789 and expose the previous report at `/study/`. A static animation would not run the neural model, and a hosted service cannot run this local CPU checkpoint directly; neither meets the requested experience.

## Live execution

The browser receives real server snapshots at up to 8 Hz. Play, pause, single step, reset, episode seed, protocol/checkpoint selection and speed all control executed navigation worlds. A paired clean control starts from identical weights, hidden state and world seed. The intervention branch applies erasure, phase disruption, bandwidth restriction, seeded packet loss/noise and bounded latency to actual message tensors. Settings changes restart both branches on the same episode seed so a comparison's starting conditions are explicit.

Show a projected 3D navigation world, paired trajectories, action probabilities, a 16-region graph with measured state/payload activity, per-mode magnitude, an action timeline and a current-episode causal comparison. Spectator goals are never fed to the neural policy after the initial cue. Mark locked waiting ticks and count effective action differences separately. Visualization is an instrument, not a decorative claim of intelligence.

## Training sandbox

Train a cloned candidate with bounded supervised imitation updates, stream actual loss/progress, and evaluate it on disjoint fixed query/longer-delay seeds before applying it. Pause live execution while the job runs. Parameters in the active and recorded models remain untouched until an explicit UI apply action; rollback returns to the previous active model. A candidate that worsens longer-delay success by more than 0.05 cannot be applied. Applying a candidate is a demo decision, not scientific promotion. Retain candidate checkpoint, loss/evaluation records and unsigned artifact receipt in a fresh run directory.

## Provenance and lifecycle

Catalog only recognized checkpoints under the configured study directory; use safe weights-only loading and strict existing model validation. UI labels distinguish recorded, fresh and session-adapted weights. Export an honest bounded live trace with seed/settings, model digest, executed actions and diagnostics; state explicitly if history was truncated and do not label it confirmatory evidence or full neural-state replay. Existing recorded-study receipts must remain valid.

Only a single local lab is active, shared across its tabs. Serialize mutations and return immutable snapshots; slow/disconnected streams do not block simulation. Bound histories, update sizes, request bodies and client count. Reject invalid controls without partial changes. Accept mutating requests only from the same loopback origin/host. Static serving has an explicit path allowlist and exposes `/study/` read-only. Server shutdown stops the scheduler and training jobs cleanly.

## Delivery

Add `aether live --study reports/pilot-2026-11-07 --port 8789`, package its static assets, launch it, and open the current tab. Verify controls, live advancement, actual training, candidate apply/rollback, telemetry, connection recovery, desktop/mobile layouts, export and preservation of the original receipt. Report observed results without claiming improved general performance.
