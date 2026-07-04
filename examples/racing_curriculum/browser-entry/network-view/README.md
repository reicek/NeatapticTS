# browser-entry/network-view

Racing curriculum adapter over the shared rich browser network visualizer.

The racing browser demo should not own its own network-frame math. Instead it
reuses the same resolved frame, padding, node sizing, and connection drawing
path as Flappy Bird and ASCII Maze, while only swapping the semantic input
labels and the short output tags to match the Tier 1 racing controller.

## browser-entry/network-view/network-view.ts

### drawRacingNetworkVisualization

```ts
drawRacingNetworkVisualization(
  canvas: HTMLCanvasElement,
  network: default | undefined,
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationPositionedScene
```

Draw the racing curriculum network panel using the shared Flappy visualizer.

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
resolveRacingInputLabelGroupDefinitions(): readonly InputLabelGroupDefinition[]
```

Resolves the shared visualizer input-label definitions for the racing demo.

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
produces a 2-element action vector (throttle and steer). These constants
describe the semantic input groups, per-node chip labels, hover tooltip copy,
and the short output tags shown in the right-sidebar network visualizer.

Observation vector layout (in order):
0..19 Car state scalars — position, heading, speed, yaw, slip angle,
progress, boundary distances, hazard/waypoint distances, optimal-line
offset, target speed, etc.
20..59 Five look-ahead track segments, 8 channels each — relX, relY,
nextRelX, nextRelY, sinTangent, cosTangent, trackWidth, distance.
60..69 Recurrent memory trace — ten channels of self-feedback state.

Action vector layout:
0 throttle (THR)
1 steer (STR)

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

### RACING_GROUP_COLORS

Light neon ramp used for group band fills.

Reuses the same Flappy Bird light neon ramp so the racing panel feels visually
consistent with the other browser demos. Each group picks a deterministic
color from this ramp in index order.

### RACING_INPUT_GROUP_DEFS

Ordered input group definitions for the Tier 1 racing observation vector.

The seven groups correspond to CAR STATE, five LOOK AHEAD segments, and the
recurrent MEMORY trace. Group ordering matches the observation vector so the
visualizer bands align exactly with the network's input shelf.

### RACING_INPUT_SIZE

Number of observation inputs consumed by the Tier 1 racing network.

### RACING_LABEL_BAND_START_PX

X offset where the group band starts.

### RACING_LABEL_BAND_WIDTH_PX

Width of the colored group band rectangle.

### RACING_LABEL_CHIP_FONT_SIZE_PX

Font size for chip label text.

### RACING_LABEL_CHIP_LEFT_PX

X offset where per-node chip labels start.

### RACING_LABEL_CHIP_RADIUS_PX

Corner radius for chip label rectangles.

### RACING_LABEL_CHIP_RIGHT_GAP_PX

Gap between the right edge of a chip and the node's left edge.

### RACING_LABEL_CHIP_VERTICAL_GAP_PX

Vertical breathing room added above and below a group's node span.

### RACING_LABEL_LEFT_PADDING_PX

Total left padding reserved for the input label panel (band + chip + gap).

### RACING_LEGEND_ITEM_GAP_PX

Vertical gap between legend items.

### RACING_LEGEND_ITEM_HEIGHT_PX

Height of each legend item row.

### RACING_LEGEND_RIGHT_OFFSET_PX

Right offset of the weight legend from the canvas edge.

### RACING_LEGEND_SWATCH_SIZE_PX

Size of the colored swatch square inside each legend row.

### RACING_LEGEND_TEXT_FONT_SIZE_PX

Font size for legend text labels.

### RACING_LEGEND_TOP_OFFSET_PX

Top offset of the weight legend from the canvas edge.

### RACING_MONOSPACE_FONT

Monospace font stack used throughout the label panel.

### RACING_NETWORK_CONNECTION_LAYER_STYLE

Racing-specific connection layer style override.

The shared Flappy visualizer defaults to a low default alpha (0.3) so dense
Flappy networks stay legible. Racing's right-sidebar panel uses a much
smaller controller snapshot against the same dark background, so connections
are intentionally brighter and slightly thicker so the topology reads as a
vivid neon graph instead of disappearing into the background.

### RACING_NETWORK_CONNECTION_UNDERLAY_COLOR

Bright neon underlay color used behind racing network connection strokes.

### RACING_OUTPUT_LABELS

Short labels for the two output nodes, ordered to match the action vector.
Used to annotate the throttle and steer outputs on the right side of the graph.

### RACING_OUTPUT_SIZE

Number of action outputs produced by the Tier 1 racing network.

### RacingInputGroupDef

Definition for one input semantic group, covering band and chip metadata.

### RacingNodeDescDef

Definition for one per-node chip label, including hover tooltip content.
