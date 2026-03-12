# browser-entry/playback/worker-channel

## browser-entry/playback/worker-channel/playback.worker-channel.types.ts

### PlaybackStepPayload

Shared alias for the worker playback-step payload.

This keeps the playback worker-channel modules focused on playback semantics
instead of long imported protocol names.

### PlaybackStepRequest

Playback-specific worker-channel contracts.

These types sit above the lower-level browser worker protocol and describe the
request budgeting plus summary data flow used by the playback loop.

### ResolvePlaybackStepRequestInput

Input used to resolve the next playback-step request and budget remainder.

Playback uses a fractional frame budget so browser render cadence and worker
simulation cadence can be smoothed together over time.

### ResolvePlaybackStepRequestResult

Output for the resolved playback-step request and frame-budget remainder.

The resolved request records both the integer step batch to send now and the
leftover fractional budget to carry into the next render tick.

## browser-entry/playback/worker-channel/playback.worker-channel.request.services.ts

### resolvePlaybackStepRequest

`(input: import("test/examples/flappy_bird/browser-entry/playback/worker-channel/playback.worker-channel.types").ResolvePlaybackStepRequestInput) => import("test/examples/flappy_bird/browser-entry/playback/worker-channel/playback.worker-channel.types").ResolvePlaybackStepRequestResult`

Playback batch-request helpers for the browser worker channel.

The playback loop accumulates simulation budget in fractional units, then
converts that budget into integer worker step requests on each render tick.

## browser-entry/playback/worker-channel/playback.worker-channel.summary.services.ts

### resolvePlaybackCompletionSummary

`(playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, latestLeaderPipesPassed: number, latestLeaderFramesSurvived: number) => { averagePipesPassed: number; p90FramesSurvived: number; winnerPipesPassed: number; winnerFramesSurvived: number; }`

Resolves final playback summary values when the worker reports completion.

Some end-of-episode aggregates may be omitted from the worker payload, so the
browser falls back to the latest leader values it has already observed during
playback.

Parameters:
- `playbackStepPayload` - - Playback payload returned by worker.
- `latestLeaderPipesPassed` - - Last observed leader pipes passed fallback.
- `latestLeaderFramesSurvived` - - Last observed leader frames fallback.

Returns: Final aggregate playback summary.

### resolvePlaybackFrameStats

`(playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, frameIndex: number, activeBirdCount: number, leaderPipesPassed: number, leaderFramesSurvived: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats`

Summary and HUD helpers for playback worker-channel results.

Once the worker replies with a playback-step payload, these helpers turn that
raw protocol data into the browser-facing telemetry and end-of-episode summary
values used elsewhere in the playback loop.
