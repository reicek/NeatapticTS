# browser-entry/network-view

Rich network-view renderer for the ASCII Maze browser demo.

This module wraps the shared `renderNetworkView` canvas renderer and layers
on maze-specific educational overlays: colored input group bands, per-node
chip labels, output direction labels, and a weight/bias legend.

The result includes hit areas that the host can use for hover tooltip testing
without needing to re-render the canvas on every pointer event.

```ts
const result = drawMazeNetworkVisualization(canvas, network, graph);
// result.hitAreas — hover hit test geometry + tooltip content
// result.frame    — positioned nodes (reusable for custom redraws)
```

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

Draws a full educational network visualization for the ASCII Maze demo.

Orchestration:
1. Sync canvas backing-store dimensions to the panel.
2. Render the base graph (nodes + connections + background) via the shared renderer.
3. Draw the input label panel in the reserved left padding area.
4. Draw output direction labels to the right of output nodes.
5. Draw the connection weight + bias legend.

Parameters:
- `canvas` - Canvas element to render onto.
- `network` - Runtime network used for architecture and dynamic color scales.
- `graph` - Exported visualization graph from `exportVisualizationGraph`.

Returns: Resolved frame plus hover hit areas for the host tooltip system.

### drawOutputNodeLabels

```ts
drawOutputNodeLabels(
  context: CanvasRenderingContext2D,
  outputNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): void
```

Draws short direction labels (N / E / S / W) to the right of each output node.

Parameters:
- `context` - Canvas 2D context.
- `outputNodes` - Output nodes sorted top-to-bottom.
- `nodeDimensions` - Node dimensions for offset calculation.

### MazeHitArea

A canvas-space rectangular hit area with associated tooltip content.

Returned in bulk by `drawMazeNetworkVisualization` so the host can test
pointer positions against them during mousemove without re-rendering.

### MazeNetworkRenderResult

Full return value from `drawMazeNetworkVisualization`.

The frame gives access to positioned node geometry (useful for custom
overlay logic) and the hit areas are ready for pointer hit testing.

### resolveAndDrawInputLabelPanel

```ts
resolveAndDrawInputLabelPanel(
  _context: CanvasRenderingContext2D,
  inputNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): { inputDescriptionScenes: NetworkInputDescriptionScene[]; inputGroupLabelBandScenes: NetworkInputGroupLabelBandScene[]; hitAreas: MazeHitArea[]; }
```

Draws colored group bands and per-node chip labels in the left padding area.

Parameters:
- `context` - Canvas 2D drawing context.
- `inputNodes` - Input nodes sorted top-to-bottom.
- `nodeDimensions` - Node width/height from the resolved frame.

Returns: Hit areas for all drawn groups and chips.

### syncCanvasToPanel

```ts
syncCanvasToPanel(
  canvas: HTMLCanvasElement,
): void
```

Aligns canvas backing-store dimensions to the responsive panel width.

Parameters:
- `canvas` - Target canvas.

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
