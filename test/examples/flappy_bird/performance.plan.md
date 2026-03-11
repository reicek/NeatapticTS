# Flappy Bird Performance Backlog

## Current State

- `Trace-7.json` is the cleanest short steady-state capture so far after the
	pipe rewrite.
- User-visible playback is now much smoother in the sampled window, with only
	`4` dropped frames across `1,321` begin frames.
- Remaining repeated cost is still render-pipeline and GPU-heavy first, but it
	now reads more like pipe-outline headroom work than an acute jank crisis.

## Validated

1. Feed-forward fast-path propagation is validated by `Trace-3.json`.
2. Trails were simplified to champion-only rendering.
3. The ground-grid loop now uses cached vector geometry.
4. Non-champion birds now render as cheaper body-only markers.
5. Playback snapshots now use packed typed arrays plus transfer lists.
6. Playback-step messaging now uses a persistent channel with request-id
   correlation.
7. The champion now renders with a single red body glow only.
8. Background glow is now retained only on the horizon.
9. Pipes now render as outline-only neon geometry with a glowing entrance rim
	and no pipe-body fill.

## What Still Dominates

1. Full-scene canvas rendering in
   `browser-entry/playback/frame-render/playback.frame-render.service.ts`.
2. Remaining GPU/compositing cost from per-pipe neon-outline drawing.
3. Per-RAF playback-step request/response traffic plus snapshot packaging.

## Trace References

- `Trace-20260309T191949.json`: baseline.
- `Trace-3.json`: validates worker fast-path improvement.
- `Trace-4.json`: warmish post-optimization trace, but contaminated by `EvaluateScript` and
  `CpuProfiler::StartProfiling`.
- `Trace-5.json`: pre-simplification trace; useful for repeated totals and event rollups,
	but still contaminated in longest-event output by profiler startup.
- `Trace-6.json`: post-simplification trace; useful steady-state signal once
	generation-turnover and profiler spikes are filtered out.
- `Trace-7.json`: short post-pipe-rewrite steady-state trace with minimal frame
	loss and no large worker stall inside the sampled window.

## Next Trace

Capture the next trace only after the page, bundles, and worker are fully warm.

Compare against the baseline:

1. Dropped frames.
2. `Renderer / CrRendererMain` total.
3. `GPU Process / CrGpuMain` total.
4. Total `GPUTask`.
5. Total and max `HandlePostMessage`.
6. Total `FunctionCall` in `flappy-evolution.worker.bundle.js`.
7. Total and max `FunctionCall` in `flappy-bird.bundle.js`.
8. Max `FireAnimationFrame`.

When generation rollover occurs inside the capture, treat the longest worker
`HandlePostMessage` and `RunTask` as contamination unless the same stall shape
repeats inside the steady-state portion of the trace.

Command:

```bash
npm run trace:analyze -- test/examples/flappy_bird/<new-trace>.json --top=15
```

## Minimal Backlog

### If Renderer Still Dominates

1. Reduce per-pipe render cost in
   `browser-entry/playback/frame-render/playback.frame-render.services.ts`.
2. Consider caching or flattening the pipe-outline draw shape in
	`browser-entry/playback/playback.render.service.ts`.
3. Revisit champion or background rendering only if a later clean trace shows
	them materially again.

### If Worker Transport Still Matters After Renderer Cuts

1. Consider buffer reuse or a shared-buffer layout for playback snapshots.
2. Do not spend another round on listener/protocol reshaping first.

## Decision Rule

1. If `GPUTask`, GPU-thread totals, and render-pipeline draw cost still
	dominate, stay on render simplification work.
2. If `HandlePostMessage` and worker-bundle `FunctionCall` stay materially high
	after render cuts, revisit transport memory reuse.
3. If the trace is contaminated by startup again, capture another warm run
	before changing code.

Success criteria for the next optimization round:

1. materially fewer dropped frames,
2. lower GPU-task total and lower renderer-main total,
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

## Trace-4 Validation

The newer capture in `Trace-4.json` validates the direction of the completed
optimization batch, but it is not a clean steady-state apples-to-apples
comparison against the earlier traces.

Trace-4 summary:

- Trace window: `41.39s`
- Event count: `20,838`
- Frames observed: `99`
- Dropped frames: `34`

Capture-quality caveat:

- This trace includes startup and profiling overhead that pollutes the single
	worst-frame numbers. `EvaluateScript` remains visible (`88.49ms` total), and
	`CpuProfiler::StartProfiling` appears as a repeated long renderer event
	(`126.01ms` total, `75.50ms` max).
- Because of that contamination, Trace-4 should not be used by itself to claim
	that the optimization batch regressed steady-state playback. It is still
	useful for validating which layer remains dominant after the recent changes.

What Trace-4 does validate:

- The demo is still renderer-bound first. `CrRendererMain` remains the primary
	steady-state user-visible bottleneck (`1653.76ms` total), and the worst frame
	still lands inside `FireAnimationFrame` (`75.92ms`) and `FunctionCall` in
	`flappy-bird.bundle.js` (`75.72ms`, `126.29ms` total).
- The completed worker transport and playback-channel work did not become the
	new dominant UX bottleneck. Worker `FunctionCall` (`121.34ms` total) and
	`HandlePostMessage` (`150.74ms` total, `47.67ms` max) are still secondary to
	the main-thread render path.
- GPU-side composition cost remains materially involved. `GPUTask`
	(`769.07ms` total), `Paint` (`169.80ms` total), and `ImageUploadTask`
	(`112.56ms` total) confirm that canvas composition and glow-heavy drawing are
	still expensive even after trail, ground-grid, and non-champion simplification.

Concrete hotspot mapping for the codebase at Trace-4 capture time:

- Main-thread frame cost still funnels through
	`browser-entry/playback/frame-render/playback.frame-render.service.ts`, which
	rebuilds the entire visible scene every iteration.
- The most obvious remaining demo-layer hot paths are in
	`browser-entry/playback/frame-render/playback.frame-render.services.ts` and
	`browser-entry/playback/frame-render/playback.frame-render.utils.ts`, where
	pipes still render with per-pipe fill and neon outline passes and the champion
	still pays multiple glow/shadow passes.
- Background work is materially better isolated than before, but it still runs
	inside the same frame budget through
	`browser-entry/playback/background/playback.background.services.ts`.
- Worker snapshot transport is now cheaper and structurally cleaner. The hot
	worker-side transport seam is concentrated in
	`flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.ts`, while the
	browser decode seam is concentrated in
	`browser-entry/playback/playback.snapshot.utils.ts`.
- Request-per-step listener churn is no longer the main protocol concern after
	`browser-entry/worker-channel/worker-channel.playback.service.ts` moved to
	persistent listeners plus request-id correlation.

Interpretation:

- Priority 2 through Priority 6 should be considered directionally validated:
	the architecture is cleaner, the worker boundary is cheaper, and the renderer
	is still clearly the first-order problem.
- Trace-4 does not justify another worker-channel refactor before more renderer
	work. The next round should stay focused on the remaining expensive draw paths
	and on producing a cleaner steady-state validation capture.

## Trace-5 Validation

The newer capture in `Trace-5.json` changes the bottleneck shape compared with
the earlier traces. The demo is still bottlenecked by the playback/render path,
but the strongest repeated cost is now GPU/compositing plus per-frame playback
transport rather than main-thread `requestAnimationFrame` JavaScript.

Trace-5 summary:

- Trace window: `20.32s`
- Event count: `260,542`
- Frames observed: `2,042`
- Dropped frames: `219`

Capture-quality caveat:

- The single worst renderer and worker events are still polluted by profiler
	startup. `EvaluateScript` (`88.70ms`) and `CpuProfiler::StartProfiling`
	(`88.27ms` on the renderer, `58.09ms` on the worker) are visible in the
	longest-event table, so Trace-5 should not be used to claim a steady-state
	worst-frame regression from max-event values alone.
- Unlike Trace-4, the capture window is long enough that repeated totals and
	event rollups are still useful for prioritization.

What Trace-5 does validate:

- GPU/compositing is now the clearest repeated user-visible cost.
	`GPU Process / CrGpuMain` totals `39,838.63ms`,
	`GPU Process / GpuVSyncThread` totals `20,262.36ms`, and `GPUTask` totals
	`19,669.41ms`.
- Main-thread playback JavaScript is no longer the primary repeated bottleneck
	in this capture. `FireAnimationFrame` totals only `260.52ms` with a
	`0.53ms` max, and `FunctionCall` attributed to
	`flappy-bird.bundle.js?v=20260227-4` totals only `90.92ms` with a `0.19ms`
	max.
- Worker cost is still materially involved, but it is concentrated in the
	playback-step boundary rather than in a core-library fast-path gap.
	`HandlePostMessage` totals `2,653.15ms`, the worker bundle
	`FunctionCall` total is `1,776.53ms`, and the worker thread totals
	`7,543.29ms`.
- Dropped frames remain materially high (`219 / 2,042`), so the user-visible
	problem is still not solved even though the bottleneck composition shifted.

Concrete hotspot mapping for the codebase at Trace-5 capture time:

- Full-scene redraw still funnels through
	`browser-entry/playback/frame-render/playback.frame-render.service.ts`, which
	rebuilds the background, pipes, birds, and trail state every playback
	iteration.
- The most obvious GPU-heavy draw paths remain in
	`browser-entry/playback/frame-render/playback.frame-render.services.ts` and
	`browser-entry/playback/playback.render.service.ts`, where every visible pipe
	still pays fill plus neon-outline passes.
- Champion-only highlight cost remains concentrated in
	`browser-entry/playback/frame-render/playback.frame-render.utils.ts`, where
	the aura, red glow plate, shine, and leader ring still trigger multiple glow
	and shadow passes.
- Worker transport cost is no longer mostly about listener churn. It is now
	concentrated in the playback-step request cadence and snapshot materialization
	across `browser-entry/playback/playback.ts`,
	`browser-entry/worker-channel/worker-channel.playback.service.ts`,
	`flappy-evolution-worker/flappy-evolution-worker.playback.service.ts`,
	`flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.ts`, and
	`browser-entry/playback/playback.snapshot.utils.ts`.

Interpretation:

- Trace-5 does not point back to a core NeatapticTS feed-forward problem. The
	fast-path conclusion from Trace-3 still stands.
- The next optimization round should still start with renderer simplification,
	but the target should now be the GPU-feeding draw passes rather than generic
	main-thread JavaScript work.
- If render cuts reduce `GPUTask` materially but `HandlePostMessage` remains
	high, the next transport step should be snapshot buffer reuse or a shared
	buffer layout, not another protocol-shape rewrite.

## Suggested Next Round

Recommended order for the next optimization batch:

1. Reduce per-pipe draw cost in
	`browser-entry/playback/frame-render/playback.frame-render.services.ts` by
	batching, caching, or otherwise simplifying the repeated neon pipe outline
	work before revisiting more exotic renderer changes.
2. Consider caching or reusing pipe `Path2D` geometry inside
	`browser-entry/playback/playback.render.service.ts` if another round is worth
	the added complexity.
3. Revisit worker transport only if a later, longer clean trace still shows meaningful
	`HandlePostMessage` and worker-bundle `FunctionCall` cost after renderer
	improvements. The next transport step should then be buffer reuse or a shared-
	buffer layout, not another listener or payload-shape rewrite.
4. Do not spend another optimization round on champion or background glow
	unless a later trace shows they are materially back in the frame budget.

## Trace-6 Validation

The newer capture in `Trace-6.json` is the first trace that meaningfully
reflects the recent visual simplification batch. Once recurring
generation-turnover noise is filtered out, it supports the conclusion that the
demo is much smoother and that the remaining render cost is now narrower and
more pipe-focused.

Trace-6 summary:

- Trace window: `33.38s`
- Event count: `469,559`
- Frames observed: `3,223`
- Dropped frames: `38`

Capture-quality caveat:

- This capture spans multiple generation turnovers. The single longest worker
	event is a `1144.88ms` `HandlePostMessage`, paired with a `1144.89ms`
	worker `RunTask`. Per your note, that aligns with all birds dying and a new
	generation starting roughly every `10s`, so it should be treated as
	generation-boundary noise rather than steady-state playback cost.
- Renderer-side `EvaluateScript` (`66.14ms`) and `CpuProfiler::StartProfiling`
	(`65.63ms`) are still visible in the longest-event table, so worst single
	renderer events are again polluted by profiler startup.

What Trace-6 does validate:

- User-visible smoothness is materially improved. Dropped frames fell from
	`219 / 2,042` in `Trace-5.json` to `38 / 3,223` here.
- GPU pressure is materially lower than Trace-5 even before any deeper pipe
	render rewrite. `GPUTask` fell from `19,669.41ms` over `20.32s` to
	`8,419.30ms` over `33.38s`. Normalized by trace window, that is a drop from
	about `968ms/s` to `252ms/s`.
- Main-thread playback JavaScript remains non-primary. `FireAnimationFrame`
	totals `395.58ms` with a `0.41ms` max, and `FunctionCall` attributed to
	`flappy-bird.bundle.js?v=20260227-4` totals only `143.94ms` with a `0.18ms`
	max.
- Useful worker cost is still present, but it now reads more as a playback
	transport/scalability concern than a visible stutter source. Worker-bundle
	`FunctionCall` totals `2,799.10ms`, and `HandlePostMessage` totals
	`5,061.09ms`, but the dramatic `1144ms` max is not representative of the
	steady-state playback loop.

Concrete hotspot mapping for the current codebase:

- Full-scene redraw still funnels through
	`browser-entry/playback/frame-render/playback.frame-render.service.ts`, which
	rebuilds the background, visible pipes, bird bodies, and champion-only trail
	each playback iteration.
- The most obvious remaining render hotspot is still pipe drawing across
	`browser-entry/playback/frame-render/playback.frame-render.services.ts` and
	`browser-entry/playback/playback.render.service.ts`, where each visible pipe
	segment still pays a glow-supported outline pass.
- Champion/body rendering is no longer a broad multi-pass hotspot in the latest
	code. `browser-entry/playback/frame-render/playback.frame-render.utils.ts`
	now renders the champion as a single red-glow body pass, so it should not be
	treated as the default next optimization target unless a later trace points
	back to it.
- Background glow is no longer a broad likely culprit in the latest code.
	`browser-entry/playback/background/playback.background.services.ts` keeps the
	horizon glow, while stars and the ground grid no longer spend glow passes.
- Transport cost remains concentrated in the playback-step boundary across
	`browser-entry/playback/playback.ts`,
	`browser-entry/worker-channel/worker-channel.playback.service.ts`,
	`flappy-evolution-worker/flappy-evolution-worker.playback.service.ts`,
	`flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.ts`, and
	`browser-entry/playback/playback.snapshot.utils.ts`.

Interpretation:

- Trace-6 validates the recent demo-layer simplification batch. The glow cuts
	were worthwhile and materially improved the visible frame profile.
- The next render optimization should now focus much more narrowly on pipes,
	not on champion/background cleanup that has already been done.
- Worker transport remains the main secondary scalability concern, but the
	largest single worker stall in this capture should be treated as
	generation-turnover noise, not as the steady-state playback bottleneck.

## Trace-7 Validation

The newer capture in `Trace-7.json` is the cleanest validation so far of the
recent pipe simplification. It is short, steady-state, and free of the large
generation-turnover worker stalls that complicated Trace-6.

Trace-7 summary:

- Trace window: `13.26s`
- Event count: `162,692`
- Frames observed: `1,321`
- Dropped frames: `4`

Capture-quality caveat:

- Renderer-side `EvaluateScript` (`64.48ms`) and `CpuProfiler::StartProfiling`
	(`64.09ms`) are still visible in the longest-event table, so the single worst
	renderer event remains polluted by profiler startup.
- Unlike Trace-6, there is no large generation-turnover worker stall inside
	the sampled window. Worker `HandlePostMessage` max is only `2.48ms`, which
	makes this trace much more trustworthy for steady-state playback conclusions.

What Trace-7 does validate:

- User-visible smoothness is now strong in this sampled window. Dropped frames
	fell again from `38 / 3,223` in `Trace-6.json` to `4 / 1,321` here.
- Main-thread playback JavaScript remains decisively non-primary.
	`FireAnimationFrame` totals `158.70ms` with a `0.31ms` max, and
	`FunctionCall` attributed to `flappy-bird.bundle.js?v=20260227-4` totals only
	`60.56ms` with a `0.18ms` max.
- Worker transport is present but no longer reads as a visible-stutter issue in
	steady state. `HandlePostMessage` totals `1,381.45ms` with only a `2.48ms`
	max, and worker-bundle `FunctionCall` totals `893.97ms` with a `2.46ms` max.
- GPU/compositing still leads the repeated actionable totals.
	`GPU Process / CrGpuMain` totals `9,080.73ms`, and `GPUTask` totals
	`4,373.58ms`, so the remaining render headroom question is still mostly about
	the pipe outline path and the full-scene redraw cadence.

Concrete hotspot mapping for the current codebase:

- Full-scene redraw still funnels through
	`browser-entry/playback/frame-render/playback.frame-render.service.ts`, which
	rebuilds background, pipes, birds, and trail state every playback iteration.
- The most likely remaining renderer hotspot is still pipe drawing across
	`browser-entry/playback/frame-render/playback.frame-render.services.ts` and
	`browser-entry/playback/playback.render.service.ts`, where every visible pipe
	segment now draws only an outline plus the gap-facing entrance rim, both with
	one glow pass and one crisp pass.
- The latest code no longer supports the older Trace-5 interpretation that pipe
	body fill is part of the cost. That work has been removed from the current
	render path.
- Worker transport remains concentrated in the playback-step boundary across
	`browser-entry/playback/playback.ts`,
	`browser-entry/worker-channel/worker-channel.playback.service.ts`,
	`flappy-evolution-worker/flappy-evolution-worker.playback.service.ts`,
	`flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.ts`, and
	`browser-entry/playback/playback.snapshot.utils.ts`, but this trace suggests
	it is now secondary for UX and more relevant as scalability headroom.

Interpretation:

- Trace-7 validates that the pipe simplification batch was worthwhile.
- The demo is no longer in the same visibly janky state reflected by Trace-5.
- If more optimization is desired, the next renderer work should stay narrowly
	focused on pipe-outline path reuse or other low-risk draw-call reductions,
	not on broad visual simplification that has already paid off.

## Final Conclusions

The demo is visually ambitious and architecturally clean, and the latest trace
shows that playback is now close to smooth in a clean steady-state window.

The most important remaining user-facing cost is the pipe-heavy render path,
with playback-step transport now reading more as scalability headroom than as a
current visible-stutter source. The most important architectural performance
problem is no longer feed-forward fast-path eligibility. Trace-3 indicates that
library-level gap is addressed, while pipe rendering and browser-worker
transport remain the main open costs.

Therefore the project should proceed on two tracks:

1. keep trimming pipe-centric canvas work so the demo preserves smooth playback
	as complexity grows,
2. keep trimming renderer and playback transport cost so the cheaper worker
	 path can translate into smoother visible playback.
