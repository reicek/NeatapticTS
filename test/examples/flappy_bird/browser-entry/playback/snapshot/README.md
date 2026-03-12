# browser-entry/playback/snapshot

## browser-entry/playback/snapshot/playback.snapshot.services.ts

### applyPlaybackSnapshot

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Snapshot synchronization helpers for playback.

The worker streams packed typed-array snapshots, while the browser renderer
wants stable mutable arrays of pipes and birds. This module performs that
translation in place so playback can stay fast and allocation-light.

### syncPlaybackSnapshotBirds

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Synchronizes packed bird snapshot fields into the reusable render-state bird array.

This mirrors the pipe strategy: keep a stable array shape when possible and
update fields in place from the packed worker buffers.

Parameters:
- `renderState` - - Mutable render state mirror used by the browser.
- `snapshot` - - Packed worker playback snapshot for the current render tick.

Returns: Nothing.

### syncPlaybackSnapshotPipes

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Synchronizes packed pipe snapshot fields into the reusable render-state pipe array.

Instead of recreating pipe objects every frame, the browser grows the array as
needed and then mutates the existing records in place.

Parameters:
- `renderState` - - Mutable render state mirror used by the browser.
- `snapshot` - - Packed worker playback snapshot for the current render tick.

Returns: Nothing.

## browser-entry/playback/snapshot/playback.snapshot.summary.utils.ts

### resolveLeaderFramesSurvived

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => number`

Small summary helpers derived from hydrated playback render state.

Once a packed snapshot has been synchronized into the browser render state,
these helpers compute simple aggregate values needed by the HUD and playback
reporting flow.
