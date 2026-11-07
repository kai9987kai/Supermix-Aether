# AETHER Live Lab v0.2

The live lab executes the neural model on the CPU and streams measured state to the browser. It extends the first milestone with interactive paired interventions and isolated supervised adaptation. It does not change the original pilot or its inconclusive outcome.

## Start

```powershell
python -m aether live --study reports/pilot-2026-11-07 --output-root runs/live --port 8789
```

Open http://127.0.0.1:8789/. The recorded study remains available at `/study/`. The live application starts paused; Play begins a shared session. Each browser tab observes and controls the same lab. Closing a tab does not create or stop a second simulation. `Ctrl+C` stops the server and its workers.

## Explore

Select a recorded checkpoint, then Play or Step. Both branches use identical weights, initial recurrent state and world seed. The control has a clean message channel. The intervention branch uses the chosen phase/gain intervention, packet-loss probability, additive packet noise, available bandwidth and fixed latency. Changing these settings restarts both worlds on the same seed. Each finished episode advances to the next seed while playback runs.

The projected world displays the actual agent and control positions and executed trajectories. Goal coordinates are spectator metadata, excluded from the policy input after the initial cue. The world still contains six fixed direction-goal tasks; fresh seeds are not new task geometries.

The region graph shows RMS activity computed from actual recurrent states. Link and mode energy is mean-square magnitude of the packets presented to the receiver, measured during the same inference step. No display inference is run separately. Scalar-only checkpoints have no regional packet traffic.

Bandwidth admits complete directed-link packets; seeded selection chooses which links are admitted. Loss erases admitted packets; Gaussian noise changes surviving packet values and rounds them to FP16. Latency queues complete message tensors for up to four ticks, initially supplying zeros. Each branch owns a separate deterministic channel generator. Pilot training used clean transport; live disturbances do not establish learned robustness.

Payload accounting distinguishes attempted bytes, admitted bytes before loss, surviving bytes after loss and delivered bytes after latency. These counters exclude scalar connections, headers and queue memory. Wave phase pilots occupy one of four modes within the counted payload. Recorded model weights and graph sizes are shown in the current frame; selecting a different checkpoint updates the model label.

Actions during the waiting period do not move the agent. The causal comparison therefore separates waiting-period action differences from movement-enabled differences. It also reports actual position and reward differences. This is one paired exploratory trajectory, not a statistical effect estimate.

## Train a candidate

The training sandbox clones the active model and pauses playback. It runs 1–100 bounded AdamW imitation updates with batch size eight and reports actual support loss/accuracy. Supervision comes from executed expert trajectories; it is not reinforcement learning or learned local plasticity.

Before and after optimization, the candidate is evaluated on the same twelve query worlds and twelve longer-delay worlds. Support/query/retention seed ranges are disjoint, although all sample the same six fixed goals. Repeated interactive evaluation is exploratory feedback, not independent confirmation.

The active model stays unchanged until Use candidate. This action is allowed only when longer-delay success has not fallen by more than 0.05 and the active weights/task delay still match the candidate's source. Applying is a local demo choice; it never promotes a model in the recorded research study. Rollback restores the preceding active weights and provenance. Switching checkpoints returns to the selected recorded weights.

Each job writes a new directory under the configured output root with a candidate checkpoint when complete, executed evaluation records, support learning curve, source fingerprint and unsigned SHA256 receipt. Cancelled or failed jobs preserve their status where storage permits. The configured output directory cannot be inside the recorded study.

## Export and limits

Export downloads a bounded executed-action trace, current settings, model digest, seed, measured frame and provenance. At most 512 step records are retained; truncation is explicit. This export is neither a full neural-state replay nor confirmatory benchmark evidence. A model replacement begins a new provenance epoch and trace.

The local server validates and serializes controls, limits request and stream resources, and serves only configured assets/study paths. It binds to loopback. The original report and checkpoint files remain read-only to the live application.

Transport references: [HTML server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) and [Python HTTP servers](https://docs.python.org/3.12/library/http.server.html). The implementation uses one scheduler with cached immutable frames; receiving or reconnecting to the stream does not advance the simulation.
