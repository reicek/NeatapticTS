# Flappy Bird Performance Backlog

## Current State

- The demo is still render-pipeline-bound first, but `Trace-5.json` shows the
	repeated cost is now more GPU/compositing-heavy than main-thread
	`requestAnimationFrame` JavaScript-heavy.
- Worker/runtime work is still secondary, and the remaining worker cost is now
	concentrated more in playback-step transport and snapshot packaging than in
	core feed-forward activation.
- `Trace-4.json` and `Trace-5.json` are directionally useful, but both still
	contain profiler startup noise in their worst single-event numbers.

## Validated

1. Feed-forward fast-path propagation is validated by `Trace-3.json`.
2. Trails were simplified to champion-only rendering.
3. The ground-grid loop now uses cached vector geometry.
4. Non-champion birds now render as cheaper body-only markers.
5. Playback snapshots now use packed typed arrays plus transfer lists.
6. Playback-step messaging now uses a persistent channel with request-id
   correlation.

## What Still Dominates

1. Full-scene canvas rendering in
   `browser-entry/playback/frame-render/playback.frame-render.service.ts`.
2. GPU/compositing cost from glow-heavy drawing, especially pipes and the
	champion bird.
3. Per-RAF playback-step request/response traffic plus snapshot packaging.

## Trace References

- `Trace-20260309T191949.json`: baseline.
- `Trace-3.json`: validates worker fast-path improvement.
- `Trace-4.json`: warmish post-optimization trace, but contaminated by `EvaluateScript` and
  `CpuProfiler::StartProfiling`.
- `Trace-5.json`: newest trace; useful for repeated totals and event rollups,
	but still contaminated in longest-event output by profiler startup.

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

Command:

```bash
npm run trace:analyze -- test/examples/flappy_bird/<new-trace>.json --top=15
```

## Minimal Backlog

### If Renderer Still Dominates

1. Reduce per-pipe render cost in
   `browser-entry/playback/frame-render/playback.frame-render.services.ts`.
2. Reduce remaining champion-only glow cost in
   `browser-entry/playback/frame-render/playback.frame-render.utils.ts`.
3. Revisit background cost only if it still shows up materially after pipe and
   champion-render reductions.

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

Concrete hotspot mapping for the current codebase:

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

Concrete hotspot mapping for the current codebase:

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

1. Capture a warm steady-state trace after the page, bundles, and worker are
	fully initialized so the next comparison is not skewed by startup or profiler
	overhead.
2. Reduce per-pipe draw cost in
	`browser-entry/playback/frame-render/playback.frame-render.services.ts` by
	batching, caching, or otherwise simplifying the repeated neon pipe outline
	work before revisiting more exotic renderer changes.
3. Reduce the remaining champion-only glow cost in
	`browser-entry/playback/frame-render/playback.frame-render.utils.ts`, because
	`GPUTask` still suggests blur/compositing pressure even after non-champion
	simplification.
4. Revisit background cost only if a clean steady-state trace still shows the
	background contributing materially after pipe and champion-render reductions.
5. Revisit worker transport only if a cleaner trace still shows meaningful
	`HandlePostMessage` and worker-bundle `FunctionCall` cost after renderer
	improvements. The next transport step should then be buffer reuse or a shared-
	buffer layout, not another listener or payload-shape rewrite.

## Final Conclusions

The demo is visually ambitious and architecturally clean, but the current
performance profile shows that the playback/render path is still doing more
work per frame than the browser can comfortably sustain.

The most important user-facing problem is now GPU-heavy render cost plus the
remaining playback-step transport overhead. The most important architectural
performance problem is no longer feed-forward fast-path eligibility. Trace-3
indicates that library-level gap is addressed, while renderer-side draw work
and browser-worker transport remain the main open costs.

Therefore the project should proceed on two tracks:

1. reduce main-thread canvas work so the demo becomes smooth,
2. keep trimming renderer and playback transport cost so the cheaper worker
	 path can translate into smoother visible playback.
