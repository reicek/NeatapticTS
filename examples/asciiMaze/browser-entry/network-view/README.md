# browser-entry/network-view

ASCII Maze adapter over the shared rich browser network visualizer.

The maze demo should not own its own network-frame math. Instead it reuses
the same resolved frame, padding, node sizing, and connection drawing path
as Flappy Bird, while only swapping the semantic input labels and the short
output tags.

## browser-entry/network-view/network-view.ts

### drawMazeNetworkVisualization

```ts
drawMazeNetworkVisualization(
  canvas: HTMLCanvasElement,
  network: default,
  graph: VisualizationGraphV1,
  hoveredNodeIndices: readonly number[],
): MazeNetworkRenderResult
```

Draw the ASCII Maze network panel using the shared Flappy visualizer owner.

Parameters:
- `canvas` - Canvas element to render onto.
- `network` - Runtime network used for architecture metadata and weights.
- `graph` - Exported graph carrying authoritative input/output counts.
- `hoveredNodeIndices` - Host-owned hovered node ids.

Returns: Shared resolved frame plus maze hover hit areas.

### MazeHitArea

A canvas-space rectangular hit area with associated tooltip content.

### MazeNetworkRenderResult

Full return value from `drawMazeNetworkVisualization`.

### resolveMazeArchitectureLabel

```ts
resolveMazeArchitectureLabel(
  network: default,
  graph: VisualizationGraphV1,
): string
```

Resolve the compact architecture summary for the maze network legend.

Parameters:
- `network` - Runtime network being visualized.
- `graph` - Exported graph carrying authoritative input/output counts.

Returns: Shared architecture label with maze IO counts.

### resolveMazeInputLabelGroupDefinitions

```ts
resolveMazeInputLabelGroupDefinitions(): readonly InputLabelGroupDefinition[]
```

Resolve the shared visualizer input-label definitions for the maze demo.

Returns: Maze semantic label groups expressed in the shared visualizer format.

### resolveMazeNetworkCanvasDimensions

```ts
resolveMazeNetworkCanvasDimensions(
  measuredWidthPx: number,
  measuredHeightPx: number,
): { widthPx: number; heightPx: number; }
```

Resolve responsive network-canvas dimensions from the host panel shelf.

Parameters:
- `measuredWidthPx` - Current measured canvas width from layout.
- `measuredHeightPx` - Current measured host-panel height from layout.

Returns: Width and height for the canvas backing store.

### resolveMazeVisualizationTopologyPlan

```ts
resolveMazeVisualizationTopologyPlan(
  network: default,
  graph: VisualizationGraphV1,
): NetworkVisualizationTopologyPlan
```

Resolve the shared topology plan for the maze network.

Parameters:
- `network` - Runtime network being visualized.
- `graph` - Exported graph carrying authoritative input/output counts.

Returns: Shared topology plan with recurrent annotations when present.

## browser-entry/network-view/network-view.constants.ts

Visual constants and input/output label definitions for the ASCII Maze
network-view panel.

The maze agent consumes a 6-element observation vector and produces 4 action
logits. These constants describe both the visual layout of the label panel
(geometry, palette) and the educational content (headings, tooltip paragraphs)
shown for each input group and individual input node.

Observation vector layout (in order):
  0  compassScalar  — BFS-preferred direction as [0, 0.75] scalar
  1  openN          — North corridor passable (0 / 1)
  2  openE          — East corridor passable (0 / 1)
  3  openS          — South corridor passable (0 / 1)
  4  openW          — West corridor passable (0 / 1)
  5  progressDelta  — Normalised step-progress signal

Action vector layout:
  0  North  1  East  2  South  3  West

### MAZE_GROUP_COLORS

Per-group palette: background fill and accent border/text color.

Index matches the ordering of `MAZE_INPUT_GROUP_DEFS`.
- 0 HEADING  — cyan
- 1 OPENNESS — green
- 2 PROGRESS — amber

### MAZE_INPUT_GROUP_DEFS

Ordered input group definitions for the maze observation vector.

The three groups correspond to the HEADING, OPENNESS, and PROGRESS semantic
families used by `MazeVision.buildInputs6`. Group ordering matches the input
vector: HEADING at index 0, OPENNESS at indices 1–4, PROGRESS at index 5.

### MAZE_INPUT_SIZE

Number of observation inputs consumed by the maze network.

### MAZE_LABEL_BAND_START_PX

X offset where the group band starts.

### MAZE_LABEL_BAND_WIDTH_PX

Width of the colored group band rectangle.

### MAZE_LABEL_CHIP_FONT_SIZE_PX

Font size for chip label text.

### MAZE_LABEL_CHIP_LEFT_PX

X offset where per-node chip labels start.

### MAZE_LABEL_CHIP_RADIUS_PX

Corner radius for chip label rectangles.

### MAZE_LABEL_CHIP_RIGHT_GAP_PX

Gap between the right edge of a chip and the node's left edge.

### MAZE_LABEL_CHIP_VERTICAL_GAP_PX

Vertical breathing room added above and below a group's node span.

### MAZE_LABEL_LEFT_PADDING_PX

Total left padding reserved for the input label panel (band + chip + gap).

### MAZE_LEGEND_ITEM_GAP_PX

Vertical gap between legend items.

### MAZE_LEGEND_ITEM_HEIGHT_PX

Height of each legend item row.

### MAZE_LEGEND_RIGHT_OFFSET_PX

Right offset of the weight legend from the canvas edge.

### MAZE_LEGEND_SWATCH_SIZE_PX

Size of the colored swatch square inside each legend row.

### MAZE_LEGEND_TEXT_FONT_SIZE_PX

Font size for legend text labels.

### MAZE_LEGEND_TOP_OFFSET_PX

Top offset of the weight legend from the canvas edge.

### MAZE_MONOSPACE_FONT

Monospace font stack used throughout the label panel.

### MAZE_OUTPUT_LABELS

Short labels for the four output nodes, ordered to match the action vector.
Used to annotate output nodes on the right side of the graph.

### MAZE_OUTPUT_SIZE

Number of action outputs produced by the maze network.

### MazeInputGroupDef

Definition for one input semantic group, covering band and chip metadata.

### MazeNodeDescDef

Definition for one per-node chip label, including hover tooltip content.
