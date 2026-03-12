# browser-entry/playback/worker-channel

## browser-entry/playback/worker-channel/playback.worker-channel.types.ts

### PlaybackStepPayload

Shared alias for the worker playback-step payload.

### PlaybackStepRequest

Request payload for one playback-step worker call.

### ResolvePlaybackStepRequestInput

Input used to resolve the next playback-step request and budget remainder.

### ResolvePlaybackStepRequestResult

Output for the resolved playback-step request and frame-budget remainder.

## browser-entry/playback/worker-channel/playback.worker-channel.request.services.ts

### resolvePlaybackStepRequest

`(input: import("test/examples/flappy_bird/browser-entry/playback/worker-channel/playback.worker-channel.types").ResolvePlaybackStepRequestInput) => import("test/examples/flappy_bird/browser-entry/playback/worker-channel/playback.worker-channel.types").ResolvePlaybackStepRequestResult`

Resolves step count and request payload for the next worker playback batch.

Parameters:
- `input` - - Current frame budget and viewport dimensions.

Returns: Request payload plus carried-over fractional frame budget.

## browser-entry/playback/worker-channel/playback.worker-channel.summary.services.ts

### resolvePlaybackCompletionSummary

`(playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, latestLeaderPipesPassed: number, latestLeaderFramesSurvived: number) => { averagePipesPassed: number; p90FramesSurvived: number; winnerPipesPassed: number; winnerFramesSurvived: number; }`

Resolves final playback summary values when the worker reports completion.

Parameters:
- `playbackStepPayload` - - Playback payload returned by worker.
- `latestLeaderPipesPassed` - - Last observed leader pipes passed fallback.
- `latestLeaderFramesSurvived` - - Last observed leader frames fallback.

Returns: Final aggregate playback summary.

### resolvePlaybackFrameStats

`(playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, frameIndex: number, activeBirdCount: number, leaderPipesPassed: number, leaderFramesSurvived: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats`

Resolves HUD playback frame stats from worker payload and leader metrics.

Parameters:
- `playbackStepPayload` - - Playback payload returned by worker.
- `frameIndex` - - Current render frame index.
- `activeBirdCount` - - Number of alive birds in current frame.
- `leaderPipesPassed` - - Current frame leader pipes passed.
- `leaderFramesSurvived` - - Current frame leader survived frames.

Returns: Normalized per-frame HUD telemetry payload.
