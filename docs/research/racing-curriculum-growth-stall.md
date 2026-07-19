# Racing Curriculum NGE Growth Stall and Dense Visualization

Research artifact for `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` —
Phase 8 — Racing Curriculum v2, Step 09 — Research NGE growth stall and dense
network visualization.

## Question

Why do NGE networks in the racing-curriculum browser demo stop growing near
~120 nodes (live capture shows ~109) instead of reaching the 4k/8k/16k targets,
what thread owns the continuous-evolution boundary and is GPU acceleration
reachable, and what visualization strategy can render 4k–16k node live diagrams
without FPS collapse?

## Evidence

### 1. Static source authority — growth engine

- The NGE core growth engine is wired and supports capacity ceilings of
  **8,000 nodes / 32,000 connections**
  (`src/neat/nge-juvenile/neat.nge-juvenile.constants.ts:125,134`).
- The canonical lifecycle `runNgeLifecycle` seeds, plans morphs, applies deltas,
  and commits hysteresis correctly
  (`src/neat/neat.nge-lifecycle.ts:161-234`).
- The racing runtime adaptation layer `examples/racing_curriculum/controller/runtime.adaptation.ts`
  wraps `runNgeLifecycle` in `adaptOnTick`.
- Default limits already allow up to 8,000 nodes / 32,000 connections
  (`runtime.adaptation.ts:139-145`), so the clamp is not a capacity limit.
- Growth throttle only engages above 1,000 nodes
  (`runtime.adaptation.ts:156, computeGrowthThrottle`), so it is not active at
  the observed stall point.
- `modeIsEvolvable` is set to `true` in the racing coevolution envelope
  (`simulation-worker.coevolution.service.ts:255-263`), so NGE extensions are
  attached and the evolvable flag is not the blocker.

### 2. Static source authority — the size-penalty evaluator is the root cause

- `runtime.adaptation.ts:433-451` defines the default
  `evaluateRollingScoreWindow`:
  ```ts
  const sizePenalty = (network.nodes.length + network.connections.length) * 0.000_1;
  return scoreMean + scoreTrend * 0.5 - sizePenalty;
  ```
- `adaptOnTick` evaluates the **same** rolling evidence window twice:
  1. baseline score on the pre-mutation network (`runtime.adaptation.ts:201`);
  2. candidate score on the post-mutation network (`runtime.adaptation.ts:339`).
- Any growth morph strictly increases `nodes.length + connections.length`, so
  `sizePenalty` increases by at least `0.0001`. Because `scoreMean` and
  `scoreTrend` are computed from the *same* evidence window, they are
  unchanged. Therefore `candidateScore < baselineScore` for every pure growth
  morph, and `improvement >= improvementThreshold (0)` is false.
- Every growth morph is rolled back (`runtime.adaptation.ts:359-360`). Pruning
  can still commit because it shrinks the size penalty, but pruning does not
  help reach 4k–16k nodes.
- The NGE E2E tests explicitly bypass this size penalty with a
  `trendOnlyEvaluator`:
  (`examples/racing_curriculum/controller/nge-e2e-growth.test.ts:9-12,19-32`).
- The racing worker creates adaptation engines with **no custom evaluator**
  (`simulation-worker.evolution.protocol.service.ts:526`), so it uses the
  default size-penalty evaluator.
- The adaptation network and the inference network are the **same object**:
  `createRaceRunnerForState` builds controller handles from
  `genome.activate(inputs)` and adaptation maps from `genome.getNetwork()`
  (`simulation-worker.evolution.protocol.service.ts:607-621`); both close
  over the single `network` instance stored in `createCarGenome`
  (`simulation-worker.coevolution.service.ts:312-345`).

### 3. Runtime/validation authority — repro scripts

- `npx jest examples/racing_curriculum/controller/nge-e2e-growth.test.ts`
  passes: 6/6 tests, including `grows the network beyond its initial seed size`
  and `respects configured maxNodes/maxConnections`.
- A standalone repro with the **default evaluator** and a flat score history
  (`tmp/racing-nge-growth-stall-demo.ts`) shows:
  - initial: 105 nodes / 206 edges;
  - after 200 ticks: **105 nodes / 206 edges** (no growth);
  - 200 growth attempts, **0 commits, 200 rollbacks**.
- The same repro with the **trend-only evaluator**
  (`tmp/racing-nge-growth-with-trend-evaluator-demo.ts`) shows:
  - initial: 105 nodes / 206 edges;
  - after 200 ticks: **705 nodes / 1801 edges**;
  - 200 growth attempts, **200 commits, 0 rollbacks**.
- This demonstrates that the growth engine is healthy and the default evaluator
  is the sole stall trigger under flat or slowly-improving rolling-score
  conditions.

### 4. Live browser authority — node/edge counts and FPS

- A visible Chrome browser was launched against
  `http://localhost:8080/docs/examples/racing_curriculum/index.html`.
- Tier 4 live capture: **109 nodes / 420 edges**, **~25 FPS**, worker
  adaptation reported `STABLE`, no console errors.
- Tier 5 live capture: **109 nodes / 420 edges**, **~20 FPS**, worker
  adaptation reported `STABLE`, no console errors.
- Tier switch succeeded without reload, but the network topology did not grow.
- The live counts confirm the stall at roughly the user-reported ~120-node
  ceiling and show that FPS is already degraded before any thousands-of-nodes
  scale is reached.

### 5. Static source authority — GPU reachability

- A GPU-aware race controller exists
  (`examples/racing_curriculum/workers/simulation-worker/simulation-worker.gpu.ts`).
- It only opts into the GPU path when `agentCount >= RACING_BROWSER_GPU_THRESHOLD = 130`
  (`simulation-worker.gpu.ts:27,52`).
- Tier 4/5 racing uses 4–6 cars, far below the threshold, so the GPU path is
  **never selected** in the live demo.
- The race-pack service itself calls `network.activate(observationInput)`
  **without** `{ useGPU: true }`
  (`simulation-worker.race-pack.service.ts:545`), so even if the threshold were
  lower the GPU hint would not be passed.
- The worker has no WebGPU device binding and no GPU-accelerated growth path;
  NGE morph planning and application run on the CPU.

### 6. Static source authority — visualizer scaling

- The racing network view delegates entirely to the shared Flappy Bird Canvas 2D
  visualizer (`examples/racing_curriculum/browser-entry/network-view/network-view.ts`).
- The shared visualizer resolves a topology plan, positions every node, and
  draws every connection every frame
  (`examples/flappy_bird/browser-entry/visualization/visualization.draw.service.ts`).
- There is no level-of-detail, abstraction, spatial culling, or deferred
  rendering path. Complexity is O(nodes + edges) per frame.
- At 109 nodes / 420 edges the live demo already drops to ~20–25 FPS, so a
  4k–16k node diagram would collapse the frame budget.

## Decision

The NGE growth stall in the racing demo is **not** a missing upstream primitive
or a capacity limit. It is a local policy bug in the default runtime evaluator:
`evaluateRollingScoreWindow` subtracts a size penalty from a score that is
recomputed on the *same* evidence window before and after mutation, which makes
every growth morph fail the `improvement >= 0` rollback gate. The core growth
engine can grow to 8k/32k when that penalty is bypassed, as the existing E2E
repro tests prove.

**Immediate fix options (to be decided by 01-planning / 04-implementing):**

1. **Replace or parametrize the evaluator in the racing worker.** Pass a
   racing-appropriate evaluator such as a trend-only or novelty-aware scorer when
   `createPerCarAdaptationEngines` is created in
   `simulation-worker.evolution.protocol.service.ts`.
2. **Make the size penalty conditional.** Do not apply a size penalty when the
   goal is explicit growth; reserve it for pruning-only phases or add a growth
   bonus that outweighs the penalty.
3. **Change the score signal.** Use a non-stationary score (e.g., lap-time
   improvement, sector-time deltas, or a stability/novelty objective) so that a
   growth morph can produce a positive score trend larger than the `0.0001`
   per-parameter penalty.

The CPU-only path is acceptable for growth because the worker tick loop is
synchronous and the observed network sizes are small. GPU inference is
currently unreachable in the racing demo due to the 130-agent threshold and the
absence of a `{ useGPU: true }` call site.

For visualization, the current Canvas 2D renderer cannot scale to 4k–16k nodes.
The recommended strategy is a **tiered level-of-detail (LOD) renderer**:

- **Far zoom / many nodes:** render aggregated clusters (e.g., input shelf,
  hidden-layer bins, output shelf) instead of every individual node/edge.
- **Medium zoom:** draw nodes as instanced quads and edges as batched line
  segments; drop edge-weight labels and bias values.
- **Near zoom / hovered node:** show full detail for a local neighborhood only
  (e.g., 2-hop ego graph), culling distant topology.
- **Adaptive quality:** monitor FPS and reduce detail (skip every N-th edge,
  lower node resolution) when the frame budget is exceeded.
- **Layout caching:** recompute topology-derived depth only when the network
  changes, not every animation frame.
- **Long-term:** migrate the network diagram to WebGL/WebGPU instanced rendering
  so the GPU can handle 4k–16k nodes at interactive frame rates.

## Risks

| Risk | Owner | Mitigation |
| ---- | ----- | ---------- |
| The default size penalty may have been intentionally designed to curb bloat in a different demo context; removing it could cause unbounded growth elsewhere. | 01-planning / 04-implementing | Scope the change to the racing worker only (pass a custom evaluator) rather than changing the default evaluator. |
| Even with the evaluator fixed, growth beyond a few hundred nodes may make the synchronous CPU inference per tick too slow for the 1/60 s fixed timestep. | 04-implementing / 05-green-testing | Benchmark tick duration as a function of node/edge count before raising growth targets; add a per-tick time budget guard. |
| The live HUD node/edge counts were read from rendered text, not a direct network object, so small read errors are possible. | 05-green-testing / browser-ui-specialist | Add a worker→host telemetry field for exact node/edge counts and assert it matches the HUD in tests. |
| The Canvas 2D visualizer will not scale; any fix to the growth evaluator will make the FPS problem worse, not better. | 04-implementing (renderer) | Implement LOD before or alongside the growth-evaluator fix so the diagram remains usable as networks grow. |
| GPU inference is currently unused. If growth reaches the 130-agent threshold, the existing GPU eligibility check (`no gates, no self-connections, supported activations`) must still pass for racing controller networks. | 02-researching / nge-core-scout | Verify GPU eligibility of evolved racing networks once growth is enabled. |
| The worker's synchronous `network.activate` assumes a plain `number[]` return. The GPU path returns a `Promise<Float32Array>`; enabling it in the worker would require async tick refactoring. | 04-implementing | Keep GPU inference out of scope for the growth-stall fix; treat it as a separate performance slice. |

## Research artifact link

- `docs/research/racing-curriculum-growth-stall.md`
