# Flappy Bird Performance Audit

## Scope

This report summarizes the performance audit of the browser Flappy Bird demo and
its evolution worker using the trace capture in `Trace-20260309T191949.json`.

The audit focused on two questions:

1. Where the current playback and evolution pipeline spends time in practice.
2. Which issues belong to the demo layer versus the core NeatapticTS runtime.

The trace was analyzed with the reusable script in `scripts/analyze-trace.ts`
using:

```bash
npm run trace:analyze -- test/examples/flappy_bird/Trace-20260309T191949.json --top=15
```

## Executive Summary

The attached trace is primarily render-bound, not worker-bound.

- The dominant steady-state cost is the renderer main thread executing the
	Flappy browser bundle during `requestAnimationFrame` work.
- The worker is active and not free, but it is secondary in this capture.
- The GPU is materially involved because the renderer performs many immediate
	mode canvas operations with glow and blur effects.
- There is also a core NeatapticTS performance gap: the Flappy runtime appears
	to be using feed-forward evolution, but it likely does not qualify for the
	network slab fast path because acyclic mode is not explicitly enabled.

In short:

- User-visible stutter comes mostly from canvas rendering pressure.
- Scalability risk comes mostly from missed NeatapticTS inference fast-path use.

## Trace Summary

The analyzer produced these high-level results:

- Trace window: `93291.83ms`
- Event count: `18,636`
- Frames observed: `91`
- Dropped frames: `14`

This means roughly `15.4%` of observed frames missed budget during the captured
window.

### Thread Summary

- `Renderer / CrRendererMain`: `1494.70ms` total, `419.85ms` RunTask, 4 tasks
	above `16.7ms`, 2 tasks above `50ms`, worst task `67.59ms`
- `Renderer / DedicatedWorker thread / flappy-evolution.worker.bundle.js`:
	`856.90ms` total, `172.90ms` RunTask, 5 tasks above `16.7ms`, worst task
	`42.52ms`
- `GPU Process / CrGpuMain`: `1062.26ms` total, `561.07ms` RunTask
- `GPU Process / GpuVSyncThread`: `769.74ms` total
- `Browser / CrBrowserMain`: `392.00ms` total, with one `197.84ms` task

### Longest Events

The most important long events were:

- `197.84ms` on `Browser / CrBrowserMain / RunTask`
- `67.59ms` on `Renderer / CrRendererMain / RunTask`
- `66.99ms` on `Renderer / CrRendererMain / FireAnimationFrame`
- `66.76ms` on `Renderer / CrRendererMain / FunctionCall` in
	`flappy-bird.bundle.js`
- `42.52ms` on `Renderer / DedicatedWorker thread / RunTask`
- `42.51ms` on `Renderer / DedicatedWorker thread / HandlePostMessage`
- `42.48ms` on `Renderer / DedicatedWorker thread / FunctionCall` in
	`flappy-evolution.worker.bundle.js`

### Event Rollups

- `RunTask`: `2395.97ms` total
- `GPUTask`: `501.19ms` total
- `FunctionCall`: `219.77ms` total
- `HandlePostMessage`: `122.61ms` total
- `FireAnimationFrame`: `88.65ms` total
- `ImageUploadTask`: `48.14ms` total

### FunctionCall Attribution

- `flappy-bird.bundle.js`: `106.43ms` total
- `flappy-evolution.worker.bundle.js`: `95.11ms` total

This confirms that both the renderer bundle and worker bundle matter, but the
main-thread render loop is the most user-visible source of missed frames.

## Detailed Findings

### 1. The trace is renderer-bound first, worker-bound second

The hottest recurring stall is not the worker itself. It is the renderer main
thread executing the browser playback bundle during animation frame handling.

Evidence:

- The biggest steady-state `FunctionCall` belongs to `flappy-bird.bundle.js`.
- The worst `FireAnimationFrame` and `FunctionCall` durations align closely,
	which points to the browser frame callback doing too much work in one pass.
- The worker is active but its worst steady-state function call is materially
	smaller than the renderer's worst frame callback.

Interpretation:

- Fixing only worker-side simulation will not remove the visible jank shown by
	this trace.
- Main-thread canvas load needs to come down for the demo to feel smooth.

### 2. Full-scene immediate-mode canvas rendering is the main demo bottleneck

The render loop redraws the entire scene every frame in
`browser-entry/playback/frame-render/playback.frame-render.service.ts`.

That per-frame work includes:

- full background composition,
- pipe rendering,
- bird rendering,
- trail rendering.

This is visually rich, but expensive.

#### 2.1 Background rendering is only partially cached

The sky starfield uses cached tiles and repeated `drawImage`, which is already a
good optimization.

However, the lower ground-grid is still recomputed and drawn as many line
segments every frame. The grid rendering path builds batches and then performs
multiple `moveTo`/`lineTo`/`stroke` passes per batch.

Why this matters:

- The ground grid is purely decorative.
- Decorative geometry should be cheaper than gameplay drawing.
- Right now it still consumes meaningful main-thread canvas budget.

Assessment:

- Current state: moderately optimized.
- Remaining issue: still too much per-frame geometry and stroke work.

#### 2.2 Trail rendering is a high-probability spike source

Trail rendering is especially expensive because each segment is drawn as its own
stroke operation after per-segment opacity calculation.

Key constraints:

- Browser population size is `10`.
- Trail max points per bird is `40`.
- Trail drawing uses stepped segments rather than one simple polyline.

Worst-case implication:

- Each bird can generate many short trail segments.
- Across 10 birds, the number of small path operations becomes large enough to
	create frame spikes even if the median frame looks acceptable.

Assessment:

- Current state: visually effective.
- Cost profile: too expensive for steady 60 FPS on the current renderer path.

#### 2.3 Bird rendering compounds the problem with glow-heavy passes

Each active bird may trigger several passes:

- aura plate,
- champion glow plate,
- filled square body,
- shine pass,
- leader ring.

This is fine for one bird, but the browser intentionally renders the full living
population. Combined with trails and blur, it increases both CPU-side draw work
and GPU-side upload/compositing cost.

Evidence:

- `ImageUploadTask` appears in the trace.
- `GPUTask` totals are high enough to matter.

Assessment:

- Current state: artistically strong.
- Cost profile: too expensive for the current full-population renderer.

### 3. Worker playback cost is real, but it is not the first-order stutter source

The worker still has visible costs:

- simulation stepping,
- activation calls,
- snapshot creation,
- postMessage handoff.

The biggest specific worker-side symptom in the trace is `HandlePostMessage`.

Interpretation:

- The worker is likely spending noticeable time both computing and serializing.
- But in this capture the main-thread renderer still dominates visible frame
	misses.

Assessment:

- Current state: acceptable for current population size.
- Scaling risk: high if population size, substeps, or telemetry payload grow.

### 4. Snapshot transport is more expensive than it needs to be

The worker snapshot builder creates new serializable arrays for birds and pipes
on every playback batch.

Consequences:

- repeated allocation pressure,
- repeated structured clone work,
- avoidable `postMessage` overhead,
- avoidable main-thread deserialization work.

The current protocol is simple and correct, but not transport-efficient.

Assessment:

- Current state: good developer ergonomics.
- Performance state: not ideal for a real-time render loop.

### 5. Worker-channel request pattern adds avoidable per-batch overhead

Each playback step request currently installs transient message and error
listeners and resolves a promise around a one-request/one-response exchange.

This is a clean abstraction, but it means the playback loop pays repeated event
listener setup and teardown costs for the entire episode.

This is not the dominant issue today, but it is a real contributor to the total
overhead of the browser-worker boundary.

Assessment:

- Current state: clean and easy to reason about.
- Performance state: acceptable, but not the lowest-overhead model.

### 6. Feed-forward fast-path eligibility was the main repo-level runtime gap

This was the most important repo-level finding outside the demo renderer.

The worker runtime is configured using:

- `mutation: methods.mutation.FFW`
- `network: Architect.perceptron(...)`

That strongly suggests the Flappy population is intended to stay feed-forward.

However, the network fast slab activation path requires all of the following:

- inference mode,
- acyclic mode enabled,
- clean topology,
- no gates,
- no self connections,
- no dropout,
- no stochastic/weight-noise features.

The critical requirement here is explicit acyclic mode. That gap has now been
addressed at the library level by propagating feed-forward intent into the
acyclic runtime contract required by the fast slab gate.

Implication:

- The demo remains feed-forward in concept,
- and Trace-3 indicates it now reaches the cheaper worker activation path,
- which lowers worker cost even though renderer work still dominates UX.

Why this matters even though the trace is render-bound:

- today: render cost dominates,
- tomorrow: once render cost is reduced, worker activation cost becomes the next
	bottleneck,
- larger populations or more substeps will magnify this immediately.

Assessment:

- Current state: validated at the library/runtime level.
- Performance state: no longer the primary open bottleneck in the demo.

### 7. Control substeps multiply all worker inference costs

The Flappy worker runs with `FLAPPY_CONTROL_SUBSTEPS_PER_FRAME = 4`.

That means every logical frame may execute four control/physics passes. Since
policy activation happens inside the control substep, inference cost is
multiplied by:

- number of alive birds,
- number of substeps,
- number of render iterations.

This is a reasonable gameplay design choice, but it means inference efficiency
matters a lot.

Assessment:

- Current state: sensible for gameplay precision.
- Performance consequence: raises the importance of fast activation paths.

## Root Cause Summary

### Demo-layer root causes

1. Too much per-frame canvas work on the renderer main thread.
2. Decorative background work still costs more than it should.
3. Trail rendering uses many tiny immediate-mode strokes.
4. Full-population bird rendering stacks glow-heavy draw passes.
5. Snapshot transfer between worker and host allocates and clones too much.

### Shared runtime root causes

1. Control substeps still multiply all worker inference costs.
2. Snapshot transport still allocates and clones more than necessary.
3. Playback request orchestration still adds per-batch messaging overhead.

## Prioritized Action Plan

### Priority 0: Measure after every change

Before changing behavior, keep the analyzer-based workflow in place.

Required commands:

```bash
npm run trace:analyze -- test/examples/flappy_bird/Trace-20260309T191949.json --top=15
```

Recommended for follow-up captures:

1. capture a new trace,
2. compare dropped frames,
3. compare renderer main-thread totals,
4. compare worker totals,
5. compare `HandlePostMessage`, `FunctionCall`, and `GPUTask` totals.

### Priority 1: Enable the NeatapticTS fast path for Flappy feed-forward networks [DONE]

Goal:

- make feed-forward intent a first-class library contract so feed-forward
	architectures automatically qualify for acyclic fast-path eligibility when
	other slab predicates hold,
- verify the Flappy demo inherits that behavior from the library with no
	demo-specific compensation code.

Current status:

- Flappy is already configured as a feed-forward workload by intent: both the
	trainer setup and browser worker runtime seed NEAT with
	`Architect.perceptron(...)` and constrain structural mutation with
	`methods.mutation.FFW`.
- Playback and rollout evaluation both execute `network.activate(...)` in their
	hot loops, so the fast-path decision affects both browser playback and offline
	evaluation cost.
- The current NeatapticTS fast slab gate requires inference mode, explicit
	acyclic enforcement, a clean topological order, and no gates,
	self-connections, dropout, or stochastic weight features.
- The missing piece is library-level intent propagation. No current builder or
	mutation preset automatically turns feed-forward intent into the explicit
	acyclic runtime contract required by the fast slab gate. In practice, the
	demo is feed-forward by structure, but not yet feed-forward by guaranteed
	library semantics, so it likely falls back to the slower object-graph
	activation path.

Technical note: what "feed-forward" means here

- A feed-forward network is a directed acyclic graph: signals move from inputs
	through hidden nodes to outputs without recurrent loops or self-feedback.
- That property matters because the runtime can compute one stable topological
	order, pack connections into typed-array slabs, and execute a simpler forward
	pass without the extra bookkeeping needed for recurrent or gated graphs.

Why this change is necessary:

- Flappy already multiplies inference cost aggressively: each logical frame runs
	`FLAPPY_CONTROL_SUBSTEPS_PER_FRAME = 4`, and each substep activates one
	network per alive bird.
- The trace shows renderer cost is the first bottleneck today, but once render
	cost is reduced the worker-side policy loop becomes the next scaling limit.
- World-class DX should not require every demo or downstream project to know
	that feed-forward intent must be re-expressed through an extra runtime flag.
- The library should make the obvious path the fast path: when users choose a
	feed-forward architecture and feed-forward mutation policy, the runtime should
	carry that intent through to inference automatically unless the topology stops
	qualifying.

Implementation steps:

1. Decide the public API trigger for feed-forward intent at the library level:
	`Architect.perceptron(...)`, an explicit constructor option, a `Neat`
	configuration contract, or a combination with clear precedence rules.
2. Define the library contract for feed-forward intent so architecture builders,
	mutation presets, cloning, crossover, serialization, and runtime activation
	agree on what qualifies as acyclic fast-path eligible.
3. Update the library so feed-forward builders and/or feed-forward mutation
	configuration automatically enable the acyclic runtime contract instead of
	requiring demo-local opt-in.
4. Implement the library behavior so feed-forward seed networks carry acyclic
	intent through cloning, crossover, serialization, and worker/runtime
	materialization.
5. Verify that the `FFW` mutation set preserves the no-recurrence contract and
	does not silently disable fast-path eligibility during evolution.
6. Add focused validation that proves feed-forward networks created through the
	public library API use the slab path in both playback-style inference and
	rollout-style evaluation.
7. Add regression tests around the public API rather than the Flappy demo so
	any future feed-forward example inherits the optimized behavior by default.
8. Keep the Flappy demo as the acceptance case: it should require no special
	fast-path bootstrap beyond using the public feed-forward library API.
9. Re-profile after the library fix and compare worker-side `FunctionCall`
	time, activation-heavy tasks, and dropped frames to confirm the improvement is
	real in the demo and reusable for downstream projects.

Expected impact:

- medium improvement immediately,
- high scalability improvement as population size or substeps increase.

Risk:

- moderate, because this touches core network semantics and must not change
	evolutionary correctness.

### Priority 2: Cut main-thread render cost by simplifying trails first [WIP]

Goal:

- reduce the worst RAF spikes without materially harming demo readability.

Actions:

1. Stop rendering trails for non-champion birds so the full population no
	 longer pays per-frame trail stroke cost.
2. Keep only a short champion trail history to preserve readability without the
	 previous full-length cost profile.
3. Re-profile the renderer before attempting a more invasive batched-trail
	 renderer, because champion-only trails may already remove most of the spike.

Expected impact:

- high for frame stability,
- low implementation complexity relative to other render changes.

Risk:

- low, because this is visual-only.

### Priority 3: Cache the repeating ground-grid loop and keep pulses vector-sharp [WIP]

Goal:

- move decorative background cost out of the hot RAF path without switching to
	 a blurred bitmap pre-render.

Actions:

1. Keep the grid fully vector-based, but cache draw-ready geometry for the
	 repeating horizontal bands and wrapped vertical-ray cycle so the renderer
	 reuses the same command structure instead of rebuilding line paths every
	 frame.
2. Treat the lower-band grid as a deterministic loop keyed by viewport size and
	 wrapped scroll offset, so only cache misses pay the full geometry-build cost.
3. Keep the random pulses as live vector overlays, but source them from the same
	 cached pulse-eligible lanes used by the grid cycle so only travel progress is
	 frame-local.
4. Avoid full offscreen pre-rendering unless the vector-path cache still proves
	 too expensive, because bitmap caching risks softer lines and higher startup
	 cost for a background that already repeats cleanly.

Expected impact:

- medium to high main-thread improvement,
- preserves line sharpness and keeps startup cost flat,
- keeps the pulse detail intact while reducing repeated path construction.

Risk:

- low to moderate, because the background is isolated and decorative but the
	 cache keys must stay aligned with viewport size and wrapped scroll.

### Priority 4: Reduce bird rendering complexity for non-champion birds [WIP]

Goal:

- keep population visibility while reducing per-frame draw passes.

Actions:

1. Remove the remaining non-champion glow-heavy body styling so non-champions
	 render as crisp body-only markers.
2. Keep the full aura, glow, shine, and leader-ring stack only for the
	 leader/champion.
3. Re-profile before changing non-champion opacity or size, because the first
	 pass should reduce draw cost without changing the readability contract more
	 than necessary.

Expected impact:

- medium improvement in both renderer and GPU cost.

Risk:

- low, because this is visual-only.

### Priority 5: Make worker snapshot transport cheaper

Goal:

- reduce worker-host boundary overhead during playback.

Actions:

1. Replace array-of-objects snapshots with typed-array snapshots.
2. Use transferable buffers where practical.
3. Consider a stable shared buffer protocol for fixed-layout bird and pipe data.
4. Avoid sending fields that the renderer can derive locally.

Expected impact:

- medium worker-side gain,
- medium main-thread gain,
- lower GC and clone pressure.

Risk:

- moderate, because it changes protocol shape and renderer decoding logic.

### Priority 6: Replace request-per-step listener churn with a persistent playback channel

Goal:

- reduce messaging overhead and simplify the hot loop.

Actions:

1. Keep one persistent `message` listener for playback.
2. Use sequence ids or a playback-session state machine instead of temporary
	 listeners.
3. Avoid creating a fresh promise/listener pair for each worker step.

Expected impact:

- low to medium on its own,
- useful after larger renderer costs are reduced.

Risk:

- moderate, because it changes protocol orchestration semantics.

## Recommended Implementation Order

To maximize return while reducing risk, implement in this order:

1. Enable and verify acyclic fast-path use for Flappy feed-forward networks.
2. Simplify or batch trail rendering.
3. Cache the repeating ground-grid loop.
4. Reduce non-champion bird visual complexity.
5. Re-profile.
6. Only then redesign snapshot transport and playback messaging if still needed.

## Validation Plan

After each optimization pass, collect a fresh browser trace and compare these
metrics against the current baseline:

- dropped frame count,
- renderer main-thread total duration,
- max `FireAnimationFrame`,
- max `FunctionCall` in `flappy-bird.bundle.js`,
- worker `HandlePostMessage` total,
- total `GPUTask`,
- total `FunctionCall` in `flappy-evolution.worker.bundle.js`.

Success criteria for the next optimization round:

1. materially fewer dropped frames,
2. lower renderer-main worst-frame cost,
3. no regression in gameplay correctness,
4. no regression in worker determinism.

## Trace-3 Validation

The newer capture in `Trace-3.json` is consistent with a successful
feed-forward fast-path rollout in the worker, even though it does not solve the
renderer bottleneck.

Trace comparison summary versus `Trace-20260309T191949.json`:

- Worker `FunctionCall` total fell from `95.11ms` to `78.46ms` despite the new
	trace covering a longer window (`93.29s` -> `114.84s`). Normalized by trace
	duration, worker `FunctionCall` cost dropped from about `1.02ms/s` to
	`0.68ms/s`.
- Worker worst `FunctionCall` fell from `42.48ms` to `25.84ms`.
- Worker `HandlePostMessage` total stayed similar in absolute terms
	(`122.61ms` -> `118.80ms`), but improved when normalized by trace duration
	(`1.31ms/s` -> `1.03ms/s`). Its worst event also dropped from `42.51ms` to
	`25.87ms`.
- Worker thread total time also improved when normalized by trace duration
	(`856.90ms / 93.29s` -> `783.67ms / 114.84s`).

What did not improve:

- Dropped frames got worse in this capture (`14 / 91` -> `38 / 130`).
- Renderer main-thread and GPU totals remain the dominant user-visible cost.
- Worst renderer `FireAnimationFrame` and renderer `FunctionCall` are still
	about `66ms`, which keeps the demo visibly janky even when worker compute is
	cheaper.

Interpretation:

- This trace supports the library-level conclusion that feed-forward intent is
	now reaching the cheaper worker activation path in practice.
- This trace does not support any claim that the full demo is "fixed" from a
	UX perspective, because the main-thread render path is still the primary
	source of missed frames.
- Priority 1 should be considered validated enough to keep, while the next
	optimization work should stay focused on Priority 2 through Priority 4.

## Final Conclusions

The demo is visually ambitious and architecturally clean, but the current
performance profile shows that the rendering path is doing more work per frame
than the browser can comfortably sustain.

The most important user-facing problem is renderer-side frame cost. The most
important architectural performance problem is no longer feed-forward fast-path
eligibility. Trace-3 indicates that library-level gap is addressed, while
renderer work and browser-worker transport remain the main open costs.

Therefore the project should proceed on two tracks:

1. reduce main-thread canvas work so the demo becomes smooth,
2. keep trimming renderer and playback transport cost so the cheaper worker
	 path can translate into smoother visible playback.
