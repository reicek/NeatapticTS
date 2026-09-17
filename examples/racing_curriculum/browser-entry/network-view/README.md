# browser-entry/network-view

Racing curriculum adapter over the shared rich browser network visualizer.

The racing browser demo should not own its own network-frame math. Instead it
reuses the shared resolved frame, padding, node sizing, and connection drawing
path while only swapping the semantic input labels and the short output tags
to match the Tier 1 racing controller.

## browser-entry/network-view/network-view.ts

### buildRacingNetworkViewSettings

```ts
buildRacingNetworkViewSettings(
  inputSize: number,
): NetworkVisualizationSettings
```

Builds the racing settings bag that drives the shared visualizer path.

LOD stays force-enabled with the shared LOD defaults so dense curriculum
networks collapse into the shared abstract renderer, while the semantic
input labels flow through the settings bag the shared orchestrator reads.

Parameters:
- `inputSize` - Effective input-layer size for label group resolution.

Returns: Racing settings bag for the shared visualizer.

### drawRacingNetworkVisualization

```ts
drawRacingNetworkVisualization(
  canvas: HTMLCanvasElement,
  network: default | undefined,
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationPositionedScene
```

Draw the racing curriculum network panel using the shared network visualizer.

Parameters:
- `canvas` - Canvas element to render onto.
- `network` - Runtime network used for architecture metadata and weights.
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Positioned node snapshot reused by host-side hover hit testing.

### drawRacingNetworkVisualizationFromFrame

```ts
drawRacingNetworkVisualizationFromFrame(
  context: CanvasRenderingContext2D,
  resolvedFrame: NetworkVisualizationResolvedFrame,
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationPositionedScene
```

Draws a previously resolved racing network-visualization frame.

The host uses this path for hover-only repaint work because it can reuse the
cached static frame and only vary interactive emphasis.

Parameters:
- `context` - Canvas 2D drawing context.
- `resolvedFrame` - Reusable frame cache.
- `hoveredNodeIndices` - Optional host-owned hovered node ids.

Returns: Positioned node snapshot reused by host-side hover hit testing.

### NetworkVisualizationResolvedFrame

Reusable resolved frame cache for hover-only network redraws.

The host can keep this resolved frame between pointer-driven redraws so it
does not recompute topology, layout, legend inputs, or connection lookups
until the network payload or canvas size changes.

### RacingNetworkRenderResult

Full return value from `drawRacingNetworkVisualization`.

### resolveNetworkVisualizationTooltipScene

```ts
resolveNetworkVisualizationTooltipScene(
  canvasPoint: NetworkVisualizationCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined
```

Resolves the tooltip scene for the current hovered network overlay target.

Hit-test priority, from narrowest to broadest:
1. Hidden-column node hit.
2. Input node hit.
3. Hidden-column label region.
4. Input description region.
5. Input group label band.

Input descriptions and input nodes intentionally share the same tooltip copy,
while semantic group bands resolve a broader group-level teaching tooltip.

Parameters:
- `canvasPoint` - Hover point in network-canvas coordinates.
- `positionedScene` - Rendered positioned scene reused for hover hit testing.

Returns: Tooltip scene model for the hovered overlay target, or undefined.

Example:

```ts
const scene = resolveNetworkVisualizationFrame(context, network, inputSize, outputSize);
const tooltip = resolveNetworkVisualizationTooltipScene({ xPx: 120, yPx: 80 }, scene);
if (tooltip) {
  showTooltip(tooltip.heading, tooltip.bodyParagraphs, tooltip.anchorCenterXPx, tooltip.anchorTopPx);
}
```

### resolveRacingArchitectureLabel

```ts
resolveRacingArchitectureLabel(
  network: default,
): string
```

Resolve the compact architecture summary for the racing network legend.

Parameters:
- `network` - Runtime network being visualized.

Returns: Shared architecture label with racing IO counts.

### resolveRacingInputLabelGroupDefinitions

```ts
resolveRacingInputLabelGroupDefinitions(
  inputSize: number | undefined,
): readonly InputLabelGroupDefinition[]
```

Resolves the shared visualizer input-label definitions for the racing demo.

Parameters:
- `inputSize` - Optional network input size; when 124, the full Tier 6
label set is returned, otherwise the Tier 1 set is returned.

Returns: Racing semantic label groups expressed in the shared visualizer format.

### resolveRacingNetworkCanvasDimensions

```ts
resolveRacingNetworkCanvasDimensions(
  measuredWidthPx: number,
  measuredHeightPx: number,
): { widthPx: number; heightPx: number; }
```

Resolve responsive network-canvas dimensions from the host panel shelf.

Parameters:
- `measuredWidthPx` - Current measured canvas width from layout.
- `measuredHeightPx` - Current measured host-panel height from layout.

Returns: Width and height for the canvas backing store.

### resolveRacingNetworkVisualizationFrame

```ts
resolveRacingNetworkVisualizationFrame(
  context: CanvasRenderingContext2D,
  network: default | undefined,
): NetworkVisualizationResolvedFrame
```

Resolves a reusable network-visualization frame for the racing controller.

Host-owned caching can reuse this frame between pointer-driven redraws so the
expensive topology/layout work is not repeated for every hover event.

Parameters:
- `context` - Canvas 2D drawing context.
- `network` - Network to visualize.

Returns: Reusable resolved frame for subsequent draw passes.

## browser-entry/network-view/network-view.constants.ts

Visual constants and input/output label definitions for the racing curriculum
network-view panel.

The Tier 1 racing controller consumes a 70-element observation vector and
produces a 2-element action vector (throttle and steer). Higher tiers widen
the same base vector in-place:
  Tier 2 — 77 inputs (70 base + 7 self-radio channels).
  Tier 3 — 91 inputs (70 base + 21 teammate-radio channels).
  Tier 4/5 — 103 inputs (91 Tier 3 + 4 tire-health + 8 pit/strategy channels).
  Tier 6 — 124 inputs (103 Tier 4/5 + 21 opponent-perception channels).

These constants describe the semantic input groups, per-node chip labels,
hover tooltip copy, and the short output tags shown in the right-sidebar
network visualizer.

Tier 1 observation vector layout (in order):
  0..19   Car state scalars — position, heading, speed, yaw, slip angle,
          progress, boundary distances, hazard/waypoint distances, optimal-line
          offset, target speed, etc.
  20..59  Five look-ahead track segments, 8 channels each — relX, relY,
          nextRelX, nextRelY, sinTangent, cosTangent, trackWidth, distance.
  60..69  Recurrent memory trace — ten channels of self-feedback state.

Tier 6 tail layout:
  70..76   Self-radio — 7 learned self-communication channels (Tier 2 only).
  77..97   Teammate radio — 3 slots × 7 channels each.
  98..101  Own-car tire health — frontLeft, frontRight, rearLeft, rearRight.
 102..109  Pit/strategy state — 8 channels.
 110..130  Opponent perception — 3 slots × 7 ego-relative channels each.

Action vector layout:
  0  throttle (THR)
  1  steer     (STR)
  2..8  radio-write channels for Tier 3+ controllers.

### buildLookAheadGroupDefinitions

```ts
buildLookAheadGroupDefinitions(
  segmentIndex: number,
): readonly RacingInputGroupDef[]
```

Builds one look-ahead input group definition for the given segment index.

Each of the five look-ahead segments occupies eight consecutive channels and
describes the track geometry ahead of the car in a local frame.

Parameters:
- `segmentIndex` - Zero-based look-ahead segment (0..4).

Returns: One group definition with eight node descriptions.

### buildOpponentPerceptionGroupDefinition

```ts
buildOpponentPerceptionGroupDefinition(
  slotIndex: number,
): readonly RacingInputGroupDef[]
```

Builds one opponent-perception input group definition for the given slot index.

Parameters:
- `slotIndex` - Zero-based opponent slot (0..2).

Returns: One group definition with seven node descriptions.

### buildPitStrategyGroupDefinition

```ts
buildPitStrategyGroupDefinition(): readonly RacingInputGroupDef[]
```

Builds the pit/strategy input group definition.

Returns: One group definition with eight node descriptions.

### buildSelfRadioGroupDefinition

```ts
buildSelfRadioGroupDefinition(): readonly RacingInputGroupDef[]
```

Builds the Tier 2 self-radio input group definition.

The seven self-radio channels are learned latent broadcasts that the
controller emits and reads back on the next tick. They have no fixed
semantic meaning; instead they let the network evolve its own short-term
memory protocol on top of the base observation vector.

Returns: One group definition with seven node descriptions.

### buildTeammateRadioGroupDefinition

```ts
buildTeammateRadioGroupDefinition(
  slotIndex: number,
): readonly RacingInputGroupDef[]
```

Builds one teammate-radio input group definition for the given slot index.

Parameters:
- `slotIndex` - Zero-based teammate slot (0..2).

Returns: One group definition with seven node descriptions.

### buildTireHealthGroupDefinition

```ts
buildTireHealthGroupDefinition(): readonly RacingInputGroupDef[]
```

Builds the own-car tire-health input group definition.

Returns: One group definition with four node descriptions.

### RACING_GROUP_COLORS

Light neon ramp used for group band fills.

Reuses the same shared light neon ramp so the racing panel feels visually
consistent with the other browser demos. Each group picks a deterministic
color from this ramp in index order.

### RACING_INPUT_GROUP_DEFS

Ordered input group definitions for the Tier 1 racing observation vector.

The seven groups correspond to CAR STATE, five LOOK AHEAD segments, and the
recurrent MEMORY trace. Group ordering matches the observation vector so the
visualizer bands align exactly with the network's input shelf.

### RACING_INPUT_SIZE

Number of observation inputs consumed by the Tier 1 racing network.

### RACING_NETWORK_CONNECTION_LAYER_STYLE

Racing-specific connection layer style override.

The shared network visualizer defaults to a low default alpha (0.3) so dense
networks stay legible. Racing's right-sidebar panel uses a much smaller
controller snapshot against the same dark background, so connections are
intentionally brighter and slightly thicker so the topology reads as a vivid
neon graph instead of disappearing into the background.

### RACING_OUTPUT_LABELS

Short labels for the action output nodes, ordered to match the action vector.
Tier 1–2 controllers only use the first two entries (throttle and steer);
Tier 3+ controllers append seven radio-write channels.

### RACING_OUTPUT_SIZE

Number of action outputs produced by the Tier 1 racing network.

### RacingInputGroupDef

Definition for one input semantic group, covering band and chip metadata.

### RacingNodeDescDef

Definition for one per-node chip label, including hover tooltip content.

### resolveRacingInputGroupDefinitions

```ts
resolveRacingInputGroupDefinitions(
  inputSize: number | undefined,
): readonly RacingInputGroupDef[]
```

Resolves the racing input group definitions matching the requested input size.

The default (Tier 1) set covers the first 70 channels and keeps all pre-existing
visualizer tests unchanged. When the network consumes the full Tier 6 vector,
this resolver returns the extended set that labels every teammate-radio,
tire-health, pit/strategy, and opponent-perception channel.

Parameters:
- `inputSize` - Optional network input size; defaults to the Tier 1 width.

Returns: Ordered racing input group definitions for the shared visualizer.
