# browser-entry/playback/worker-channel

Playback-specific worker-channel contracts.

These types sit above the lower-level browser worker protocol and describe the
request budgeting plus summary data flow used by the playback loop.

The key idea is cadence smoothing: the browser renders on animation frames,
while the worker advances simulation in step batches. These contracts describe
how a fractional render-time budget becomes a concrete worker request.

Tiny example:
a budget of `2.4` frames this render usually becomes `2` simulation steps now
plus `0.4` carried forward to the next render tick.

## browser-entry/playback/worker-channel/playback.worker-channel.types.ts

### PlaybackStepPayload

Shared alias for the worker playback-step payload.

This keeps the playback worker-channel modules focused on playback semantics
instead of long imported protocol names.

### PlaybackStepRequest

Request payload for one playback-step worker call.

The browser asks the worker to advance simulation by a small batch of steps
and to package the result for the current viewport dimensions.

### ResolvePlaybackStepRequestInput

Input used to resolve the next playback-step request and budget remainder.

Playback uses a fractional frame budget so browser render cadence and worker
simulation cadence can be smoothed together over time.

### ResolvePlaybackStepRequestResult

Output for the resolved playback-step request and frame-budget remainder.

The resolved request records both the integer step batch to send now and the
leftover fractional budget to carry into the next render tick.

## browser-entry/playback/worker-channel/playback.worker-channel.request.services.ts

Playback batch-request helpers for the browser worker channel.

The playback loop accumulates simulation budget in fractional units, then
converts that budget into integer worker step requests on each render tick.

### resolvePlaybackStepRequest

```ts
resolvePlaybackStepRequest(
  input: ResolvePlaybackStepRequestInput,
): ResolvePlaybackStepRequestResult
```

Resolves step count and request payload for the next worker playback batch.

This is the pacing bridge between browser rendering and worker simulation.
Rather than sending a fixed step count every frame, the loop carries forward
fractional remainder so long-term playback speed stays closer to the intended
emulation rate.

Parameters:
- `input` - Current frame budget and viewport dimensions.

Returns: Request payload plus carried-over fractional frame budget.

## browser-entry/playback/worker-channel/playback.worker-channel.summary.services.ts

Summary and HUD helpers for playback worker-channel results.

Once the worker replies with a playback-step payload, these helpers turn that
raw protocol data into the browser-facing telemetry and end-of-episode summary
values used elsewhere in the playback loop.

### resolvePlaybackCompletionSummary

```ts
resolvePlaybackCompletionSummary(
  playbackStepPayload: { requestId: number; snapshot: EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; winnerNetworkJson?: SerializedNetwork | undefined; },
  latestLeaderPipesPassed: number,
  latestLeaderFramesSurvived: number,
): { averagePipesPassed: number; p90FramesSurvived: number; winnerPipesPassed: number; winnerFramesSurvived: number; winnerNetworkJson?: Record<string, unknown> | undefined; }
```

Resolves final playback summary values when the worker reports completion.

Some end-of-episode aggregates may be omitted from the worker payload, so the
browser falls back to the latest leader values it has already observed during
playback.

Parameters:
- `playbackStepPayload` - Playback payload returned by worker.
- `latestLeaderPipesPassed` - Last observed leader pipes passed fallback.
- `latestLeaderFramesSurvived` - Last observed leader frames fallback.

Returns: Final aggregate playback summary.

### resolvePlaybackFrameStats

```ts
resolvePlaybackFrameStats(
  playbackStepPayload: { requestId: number; snapshot: EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; winnerNetworkJson?: SerializedNetwork | undefined; },
  frameIndex: number,
  activeBirdCount: number,
  leaderPipesPassed: number,
  leaderFramesSurvived: number,
): PlaybackFrameStats
```

Resolves HUD playback frame stats from worker payload and leader metrics.

The frame-stats payload combines browser-derived leader information with any
instrumentation values provided by the worker.

Parameters:
- `playbackStepPayload` - Playback payload returned by worker.
- `frameIndex` - Current render frame index.
- `activeBirdCount` - Number of alive birds in current frame.
- `leaderPipesPassed` - Current frame leader pipes passed.
- `leaderFramesSurvived` - Current frame leader survived frames.

Returns: Normalized per-frame HUD telemetry payload.
