# browser-entry/playback/snapshot

## browser-entry/playback/snapshot/playback.snapshot.services.ts

### applyPlaybackSnapshot

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Applies worker snapshot data to the mutable playback render state.

Parameters:
- `renderState` - - Mutable render state mirror used by the browser.
- `snapshot` - - Worker playback snapshot for the current render tick.

Returns: Nothing.

### syncPlaybackSnapshotBirds

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Synchronizes packed bird snapshot fields into the reusable render-state bird array.

Parameters:
- `renderState` - - Mutable render state mirror used by the browser.
- `snapshot` - - Packed worker playback snapshot for the current render tick.

Returns: Nothing.

### syncPlaybackSnapshotPipes

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Synchronizes packed pipe snapshot fields into the reusable render-state pipe array.

Parameters:
- `renderState` - - Mutable render state mirror used by the browser.
- `snapshot` - - Packed worker playback snapshot for the current render tick.

Returns: Nothing.

## browser-entry/playback/snapshot/playback.snapshot.summary.utils.ts

### resolveLeaderFramesSurvived

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => number`

Resolves the maximum survived-frame count in the current render state.

Parameters:
- `renderState` - - Current render state.

Returns: Maximum frames survived by any bird.
