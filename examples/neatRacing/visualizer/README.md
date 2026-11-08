# visualizer

NeatRacing network panel adapter over the shared rich browser network visualizer (AC-034).

The NeatRacing browser page must not own its own network-frame math. Just like
the racing curriculum adapter, this module reuses the shared resolved frame,
padding, node sizing, and connection drawing path while only swapping the
NeatRacing-specific pieces: the most-forward-agent selection law, the NGE
controller output labels, and the NGE input/output sizes.

## Panel law (LD-32 — "most forward agent only", AC-034)

- **L1 — Progress is host-owned.** The packed sim frame (`neatracing-packed-v1`)
  carries kinematics only: no laps, no along-track distance, no networks. The
  adapter therefore NEVER derives race progress from world XY or track
  geometry; the host wiring passes an along-track progress metric per agent
  (for example `laps * trackLength + along-track distance`). The panel stays
  free of track-geometry knowledge by construction.
- **L2 — Sticky incumbent.** The current leader keeps the panel unless another
  candidate's progress is STRICTLY greater, so an equal-progress tick or a
  near-tie never flickers the panel ("the current winner at all times").
- **L3 — Non-finite progress never leads.** Candidates whose progress is
  `NaN` or infinite are invisible to the selection, mirroring the recycled
  finite-number law of the recycled goals machine (`ai/ai-mode.goals.ts`).
- **L4 — Deterministic tie-break.** Without a present incumbent, the maximum
  finite progress wins; an exact progress tie resolves to the LOWEST
  agent id, so repeated ticks resolve identically for the same inputs.
- **L5 — Thin over the shared visualizer.** Frames are resolved through
  `examples/shared/network-visualization` with the shared LOD defaults and
  the shared default input-label path; the NeatRacing observation channels
  stay owner-defined in `ai/observation.assembler.ts` and are not restated
  here.
- **L6 — One sizing authority per canvas.** The racing-parity panel wiring
  (P2-S4-C) composes `installNeatRacingNetworkResize` (the backing-store
  owner) with `resolveNeatRacingNetworkPanelVisualizationFrame` +
  `drawNeatRacingNetworkPanelFromFrame` on the same canvas. The
  all-in-one `drawNeatRacingNetworkPanel` is the standalone self-sizing
  path and must NOT be composed with the resize service on the same
  canvas, because its sync pass would silently override the resize
  service's backing store (review F2).

## visualizer/network-panel.adapter.ts

### buildNeatRacingNetworkPanelSettings

```ts
buildNeatRacingNetworkPanelSettings(): NetworkVisualizationSettings
```

Builds the NeatRacing settings bag that drives the shared visualizer path.

LOD stays force-enabled with the shared LOD defaults so dense evolved
controllers collapse into the shared abstract renderer. Input-label groups
are deliberately NOT overridden (L5): the panel renders the shared default
input-label path because the NeatRacing observation channels stay
owner-defined in `ai/observation.assembler.ts`.

Returns: NeatRacing settings bag for the shared visualizer.

### drawNeatRacingNetworkPanel

```ts
drawNeatRacingNetworkPanel(
  canvas: HTMLCanvasElement,
  network: default | undefined,
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationPositionedScene
```

Draw the NeatRacing network panel for one network using the shared
network visualizer.

Composition law (L6, review F2): this is the STANDALONE self-sizing entry
point — it owns the canvas backing store through its sync pass. The
racing-parity panel wiring must instead compose
`installNeatRacingNetworkResize` with
`resolveNeatRacingNetworkPanelVisualizationFrame` +
`drawNeatRacingNetworkPanelFromFrame`; do not pair this function with the
resize service on the same canvas.

Parameters:
- `canvas` - Canvas element to render onto.
- `network` - Runtime network of the most-forward agent, or `undefined`
while the panel waits for the first network (renders the idle panel).
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Positioned node snapshot reused by host-side hover hit testing.

Example:

```ts
const leader = resolveMostForwardNeatRacingAgent(candidates, incumbentId);
drawNeatRacingNetworkPanel(networkPanelCanvas, leader?.network);
```

### drawNeatRacingNetworkPanelFromFrame

```ts
drawNeatRacingNetworkPanelFromFrame(
  context: CanvasRenderingContext2D,
  resolvedFrame: NetworkVisualizationResolvedFrame,
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationPositionedScene
```

Draws a previously resolved NeatRacing network-panel frame.

The host uses this path for hover-only repaint work because it can reuse
the cached static frame and only vary interactive emphasis.

Parameters:
- `context` - Canvas 2D drawing context.
- `resolvedFrame` - Reusable frame cache.
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Positioned node snapshot reused by host-side hover hit testing.

### drawNeatRacingOutputNodeLabels

```ts
drawNeatRacingOutputNodeLabels(
  context: CanvasRenderingContext2D,
  outputNodes: readonly PositionedNetworkNodeLike[],
): void
```

Draws the NGE controller output tags next to the positioned output nodes.

Parameters:
- `context` - Canvas 2D drawing context.
- `outputNodes` - Output nodes sorted by ascending y.

### installNeatRacingNetworkResize

```ts
installNeatRacingNetworkResize(
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onResize: () => void,
): { uninstall: () => void; resize: () => void; redraw: () => void; }
```

Installs NeatRacing network canvas resize listeners.

Mirrors the `installRacingNetworkResize` pattern exactly: on every window
`resize` event the service measures the host element and updates the canvas
backing store, then invokes `onResize` so the caller can schedule a network
redraw. On install the size is applied once immediately. This service is
the backing-store OWNER of the racing-parity panel wiring (L6, review F2):
pair it with `resolveNeatRacingNetworkPanelVisualizationFrame` +
`drawNeatRacingNetworkPanelFromFrame`, never with the self-sizing
`drawNeatRacingNetworkPanel`.

If the host element has no CSS layout yet (for example in jsdom tests), the
service falls back to `window.innerWidth * 0.25` by
`window.innerHeight * 0.35` so the canvas still has a positive backing size.
The returned `redraw` handle is intentionally a no-op; the host schedules
redraws through `onResize`.

Parameters:
- `networkCanvas` - Network visualization canvas to resize.
- `networkCanvasHost` - Host element whose bounds drive the canvas size.
- `onResize` - Callback invoked after the canvas backing size changes.

Returns: Uninstall, resize, and redraw handles.

Example:

```ts
const { uninstall, resize } = installNeatRacingNetworkResize(
  networkPanelCanvas,
  networkPanelHost,
  () => {
    const context = networkPanelCanvas.getContext('2d', {
      desynchronized: true,
    });
    if (!context) {
      return;
    }
    const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
      context,
      leaderNetwork,
    );
    drawNeatRacingNetworkPanelFromFrame(context, frame);
  },
);
resize();
window.addEventListener('beforeunload', uninstall);
```

### NEATRACING_NETWORK_OUTPUT_LABELS

Ordered NGE controller output labels.

Law: the first two channels are the pinned NGE control axes (index 0 =
THROTTLE, index 1 = STEER) and channels 2..8 stream the degenerate
single-car self-radio seam writes (`R0`..`R6`) on radio-augmented ladder
levels. Output indices past the table fall back to `OUT<index>` so a wider
evolved output layer still renders deterministically.

### NeatRacingNetworkPanelCandidate

One host-supplied fleet candidate for the most-forward network panel.

The host wiring owns both the identity and the race progress of every car;
the adapter only compares the numbers it is handed.

### NetworkVisualizationPositionedScene

Reusable positioned-node snapshot returned by the network-view draw path.

The host reuses this exact layout snapshot for pointer hit testing so hover
logic can stay aligned with the scene that was actually rendered.

### NetworkVisualizationResolvedFrame

Reusable resolved frame cache for hover-only network redraws.

The host can keep this resolved frame between pointer-driven redraws so it
does not recompute topology, layout, legend inputs, or connection lookups
until the network payload or canvas size changes.

### resolveHostContentBoxDimensionPx

```ts
resolveHostContentBoxDimensionPx(
  hostElement: HTMLElement | null,
  hostClientDimensionPx: number,
  startPaddingProperty: "paddingBottom" | "paddingLeft" | "paddingRight" | "paddingTop",
  endPaddingProperty: "paddingBottom" | "paddingLeft" | "paddingRight" | "paddingTop",
): number
```

Resolves the host content-box dimension without double counting padding.

Mirrors the racing adapter: the host element's computed padding is
subtracted from the measured client dimension so the backing store fits the
visible shelf.

Parameters:
- `hostElement` - Host content element, or `null`.
- `hostClientDimensionPx` - Fallback client dimension of the host.
- `startPaddingProperty` - Padding property on the axis start.
- `endPaddingProperty` - Padding property on the axis end.

Returns: Resolved content-box dimension in px.

### resolveMostForwardNeatRacingAgent

```ts
resolveMostForwardNeatRacingAgent(
  candidates: readonly NeatRacingNetworkPanelCandidate[],
  incumbentAgentId: number | undefined,
): NeatRacingNetworkPanelCandidate | undefined
```

Resolves the agent whose network the panel renders this tick.

Law (module header L1-L4): progress is host-owned, the incumbent stays
leader unless a challenger is STRICTLY more forward, non-finite progress
never leads, and the no-incumbent tie-break is lowest agent id.

Parameters:
- `candidates` - Host-supplied fleet candidates for this tick.
- `incumbentAgentId` - Agent id that currently owns the panel, if any.

Returns: The winning candidate, or `undefined` when no finite-progress
candidate exists (the host then parks the panel). Network presence never
affects the selection: when the elected winner carries no network the
panel renders its idle state and the host must NOT fall back to a
networked runner-up (review F3, AC-034).

Example:

```ts
const leader = resolveMostForwardNeatRacingAgent(
  fleet.map((car, agentId) => ({
    agentId,
    progress: car.raceProgressMeters,
    network: car.network,
  })),
  renderedAgentId,
);
```

### resolveMostForwardWithoutIncumbent

```ts
resolveMostForwardWithoutIncumbent(
  finiteCandidates: readonly NeatRacingNetworkPanelCandidate[],
  incumbent: NeatRacingNetworkPanelCandidate | undefined,
): NeatRacingNetworkPanelCandidate
```

Applies the L4 max-progress/lowest-id law against the L2 sticky rule.

Parameters:
- `finiteCandidates` - Candidates with finite progress only.
- `incumbent` - Incumbent candidate kept on ties, when supplied.

Returns: The candidate that owns the panel this tick.

### resolveNeatRacingHoverState

```ts
resolveNeatRacingHoverState(
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationHoverState | undefined
```

Resolves the host-supplied hover indices into the shared hover state.

Parameters:
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Shared hover state, or `undefined` when nothing is hovered.

### resolveNeatRacingNetworkPanelArchitectureLabel

```ts
resolveNeatRacingNetworkPanelArchitectureLabel(
  network: default,
): string
```

Resolves the compact architecture summary for the NeatRacing network legend.

Parameters:
- `network` - Runtime network being visualized.

Returns: Shared architecture label with NeatRacing IO counts; the input
size falls back to the default ladder level width (Tier A, 82 channels)
when the runtime network does not expose `input`.

### resolveNeatRacingNetworkPanelCanvasDimensions

```ts
resolveNeatRacingNetworkPanelCanvasDimensions(
  measuredWidthPx: number,
  measuredHeightPx: number,
): { widthPx: number; heightPx: number; }
```

Resolves responsive network-panel canvas dimensions from the host panel
shelf, mirroring the racing panel dimension law: the measured width is
clamped to a positive minimum, the height falls back to the width-driven
aspect value, and a positive measured panel height wins over the fallback.

Parameters:
- `measuredWidthPx` - Current measured canvas width from layout.
- `measuredHeightPx` - Current measured host-panel height from layout.

Returns: Width and height for the canvas backing store.

Example:

```ts
resolveNeatRacingNetworkPanelCanvasDimensions(0, 0);
// -> { widthPx: 320, heightPx: 240 }
```

### resolveNeatRacingNetworkPanelVisualizationFrame

```ts
resolveNeatRacingNetworkPanelVisualizationFrame(
  context: CanvasRenderingContext2D,
  network: default | undefined,
): NetworkVisualizationResolvedFrame
```

Resolves a reusable network-visualization frame for the NeatRacing panel.

Host-owned caching can reuse this frame between pointer-driven redraws so
the expensive topology/layout work is not repeated for every hover event.

Parameters:
- `context` - Canvas 2D drawing context.
- `network` - Network to visualize, or `undefined` for an idle panel.

Returns: Reusable resolved frame for subsequent draw passes.

Example:

```ts
const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
  context,
  leaderNetwork,
);
drawNeatRacingNetworkPanelFromFrame(context, frame, hoveredNodeIndices);
```

### resolveSortedNeatRacingOutputNodes

```ts
resolveSortedNeatRacingOutputNodes(
  positionedNodes: readonly PositionedNetworkNodeLike[],
): PositionedNetworkNodeLike[]
```

Sorts the positioned output nodes top-to-bottom so labels map onto the
`NEATRACING_NETWORK_OUTPUT_LABELS` table deterministically.

Parameters:
- `positionedNodes` - Positioned nodes of the current frame.

Returns: Output nodes sorted by ascending y.

### syncNeatRacingCanvasToPanel

```ts
syncNeatRacingCanvasToPanel(
  canvas: HTMLCanvasElement,
): void
```

Syncs the network panel canvas backing store to the host panel shelf.

Parameters:
- `canvas` - Canvas element to keep in sync with its host shelf.
