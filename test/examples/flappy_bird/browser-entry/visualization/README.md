# browser-entry/visualization

Public visualization facade for browser-entry network rendering.

The dedicated visualization folder focuses on turning network structure and
parameter ranges into readable graphics. Layer resolution itself is shared
with the neighboring network-view topology boundary, so this facade exposes
that topology helper while keeping the visualization subsystem's public story
in one place.

The goal of this boundary is not generic charting. It is specifically to make
evolved controllers inspectable: which nodes exist, how layers are grouped,
and how sign and magnitude are encoded visually.

Minimal example:
```ts
const layers = resolveNetworkVisualizationLayers(network, 38, 2);
console.log(layers.length);
```

## browser-entry/visualization/visualization.ts

### resolveNetworkVisualizationLayers

```ts
resolveNetworkVisualizationLayers(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNodeLike[][]
```

Resolves layered node groups for network-view layout and rendering.

Educational note:
Layer grouping is a network-view concern because it drives sizing, node
placement, and architecture presentation. Visualization code can still reuse
the result, but this helper now lives with the module that owns layout.

The resolver prefers explicit layer metadata when it exists, then falls back
to a topology-derived depth estimate so even loosely structured networks can
still be drawn in an intelligible left-to-right order.

Parameters:
- `network` - - Runtime network instance.
- `inputSize` - - Input count fallback.
- `outputSize` - - Output count fallback.

Returns: Layered nodes for rendering.

## browser-entry/visualization/visualization.types.ts

Visualization-specific color-scale contracts for network rendering.

The Flappy Bird demo renders connection weights and node biases with tiered
neon ramps so humans can quickly read sign and magnitude without parsing raw
numbers on every edge and node.

A useful mental model is that these types define the legend contract for the
network panel: what range was observed, how values were bucketed, and which
colors should be shown for each tier.

### DynamicColorScale

Dynamic tiered color scale used by visualization render layers.

The scale records the observed numeric range plus the ordered threshold tiers
used to map values into colors.

### NetworkVisualizationColorScales

Grouped color scales for connection and bias channels.

Keeping the two scales together ensures the legend and drawing code read from
one consistent view of the active network range.

## browser-entry/visualization/visualization.constants.ts

Pixel side length for dotted negative-connection square markers.

### FLAPPY_NETWORK_DOTTED_CONNECTION_SQUARE_SIDE_PX

Pixel side length for dotted negative-connection square markers.

### FLAPPY_NETWORK_ENABLED_CONNECTION_ALPHA

Stroke alpha used for enabled connection lines.

### FLAPPY_NETWORK_DISABLED_CONNECTION_ALPHA

Stroke alpha used for disabled connection lines.

### FLAPPY_NETWORK_DISABLED_CONNECTION_DASH_PATTERN

Dash pattern used for disabled positive connection lines.

### FLAPPY_NETWORK_DOTTED_CONNECTION_STEP_COMPACT_RATIO

Spacing multiplier used to tighten square-dot trail cadence.

### FLAPPY_NETWORK_DOTTED_CONNECTION_WIDTH_SPACING_RATIO

Line-width multiplier used to scale dotted-step spacing.

### FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON

Tiny alignment epsilon to stabilize axis-aligned dot centering.

### FLAPPY_NETWORK_HIDDEN_NODE_VERTICAL_PADDING_PX

Extra vertical padding reserved for hidden-node labels inside rectangles.

### FLAPPY_NETWORK_OUTPUT_NODE_STROKE_WIDTH_PX

Stroke width used for emphasized output nodes.

### FLAPPY_NETWORK_HIDDEN_NODE_STROKE_WIDTH_PX

Stroke width used for hidden and input nodes.

### FLAPPY_NETWORK_OUTPUT_NODE_SHADOW_BLUR_PX

Glow blur radius used for output-node emphasis.

### FLAPPY_NETWORK_MIN_RENDER_NODE_HEIGHT_PX

Minimum render height for any node rectangle.

### FLAPPY_NETWORK_OUTPUT_NODE_HEIGHT_REDUCTION_PX

Height reduction applied to output-node rectangles for tighter framing.

### FLAPPY_NETWORK_HEADER_PADDING_PX

Left and top padding used by the network header block.

### FLAPPY_NETWORK_HEADER_LINE_HEIGHT_PX

Vertical spacing between multiline header rows.

### FLAPPY_NETWORK_LEGEND_MIN_ARCHITECTURE_TOP_PX

Minimum top bound for architecture text above the legend box.

### FLAPPY_NETWORK_LEGEND_ARCHITECTURE_GAP_PX

Gap between architecture text and the legend container.

### FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX

Shared inner padding used by legend labels and swatches.

### FLAPPY_NETWORK_LEGEND_HEADER_TOP_PADDING_PX

Top offset for the legend title inside the legend box.

### FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_END_X_PX

End x-position for connection sample lines in legend rows.

### FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_Y_OFFSET_PX

Vertical offset for connection sample lines inside legend rows.

### FLAPPY_NETWORK_LEGEND_CONNECTION_LABEL_X_PX

Label x-position for connection legend row text.

### FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_X_PX

X-position for bias legend swatches.

### FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_Y_PX

Y-position offset for bias legend swatches.

### FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_SIZE_PX

Side length for bias legend color swatches.

### FLAPPY_NETWORK_LEGEND_BIAS_LABEL_X_PX

Label x-position for bias legend row text.

## browser-entry/visualization/visualization.draw.service.ts

### drawWeightedConnectionsLayer

```ts
drawWeightedConnectionsLayer(
  context: CanvasRenderingContext2D,
  runtimeConnections: VisualNetworkConnectionLike[],
  positionByNodeIndex: Map<number, PositionedNetworkNodeLike>,
  connectionScale: DynamicColorScale,
): void
```

Draws weighted connection lines.

Connection styling carries semantic meaning: color encodes magnitude and sign,
while dash patterns and auxiliary marks help distinguish disabled or negative
edges in a way that still reads quickly on a dense graph.

Parameters:
- `context` - - Render context.
- `runtimeConnections` - - Runtime connection list.
- `positionByNodeIndex` - - Node layout map.
- `connectionScale` - - Dynamic connection color scale.

Returns: Nothing.

### drawBiasNodesLayer

```ts
drawBiasNodesLayer(
  context: CanvasRenderingContext2D,
  positionedNodes: PositionedNetworkNodeLike[],
  nodeDimensions: NetworkNodeDimensionsLike,
  biasScale: DynamicColorScale,
): void
```

Draws all network nodes with bias labels.

The node layer pairs each rectangle with a compact bias label so the panel can
show both topology and a lightweight hint of parameter state.

Parameters:
- `context` - - Render context.
- `positionedNodes` - - Positioned nodes.
- `nodeDimensions` - - Node dimensions.
- `biasScale` - - Dynamic bias color scale.

Returns: Nothing.

### drawNetworkVisualizationHeader

```ts
drawNetworkVisualizationHeader(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
): void
```

Draws network architecture header text.

The header gives viewers a compact architecture summary before they inspect
individual nodes and edges.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Header label.

Returns: Nothing.

### drawNetworkColorLegend

```ts
drawNetworkColorLegend(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
  colorScales: NetworkVisualizationColorScales,
): void
```

Draws the color legend for connections and node bias values.

This legend is what turns the panel from "colorful art" into an interpretable
instrument: it tells the viewer what each weight and bias color actually
means numerically.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Compact architecture description.
- `colorScales` - - Connection and bias color scales.

Returns: Nothing.

### resolveWeightedConnectionScene

```ts
resolveWeightedConnectionScene(
  runtimeConnection: VisualNetworkConnectionLike,
  positionByNodeIndex: Map<number, PositionedNetworkNodeLike>,
  connectionScale: DynamicColorScale,
): WeightedConnectionScene | undefined
```

Resolves a renderable connection scene from runtime data and node positions.

Parameters:
- `runtimeConnection` - - Candidate runtime connection.
- `positionByNodeIndex` - - Node layout map.
- `connectionScale` - - Connection color scale.

Returns: Renderable connection scene, when both endpoint nodes exist.

### drawWeightedConnectionScene

```ts
drawWeightedConnectionScene(
  context: CanvasRenderingContext2D,
  weightedConnectionScene: WeightedConnectionScene,
): void
```

Draws a previously resolved weighted connection scene.

Parameters:
- `context` - - Render context.
- `weightedConnectionScene` - - Render-ready connection scene.

Returns: Nothing.

### resolveBiasNodeScene

```ts
resolveBiasNodeScene(
  context: CanvasRenderingContext2D,
  positionedNode: PositionedNetworkNodeLike,
  nodeDimensions: NetworkNodeDimensionsLike,
  halfNodeWidthPx: number,
  biasScale: DynamicColorScale,
): BiasNodeScene
```

Resolves all paint attributes needed to render a single node.

Parameters:
- `context` - - Render context.
- `positionedNode` - - Positioned node payload.
- `nodeDimensions` - - Shared node dimensions.
- `halfNodeWidthPx` - - Cached half node width.
- `biasScale` - - Bias color scale.

Returns: Paint-ready node scene.

### resolveBiasNodePaintStyle

```ts
resolveBiasNodePaintStyle(
  positionedNode: PositionedNetworkNodeLike,
  biasScale: DynamicColorScale,
): BiasNodePaintStyle
```

Resolves node fill, stroke, and glow styling.

Parameters:
- `positionedNode` - - Positioned node payload.
- `biasScale` - - Bias color scale.

Returns: Node paint style.

### resolveBiasNodeLabelMetrics

```ts
resolveBiasNodeLabelMetrics(
  context: CanvasRenderingContext2D,
  nodeLabel: string,
  nodeDimensions: NetworkNodeDimensionsLike,
): BiasNodeLabelMetrics
```

Measures a bias label and resolves its font declaration.

Parameters:
- `context` - - Render context.
- `nodeLabel` - - Bias label string.
- `nodeDimensions` - - Shared node dimensions.

Returns: Measured label metrics.

### resolveBiasNodeHeightPx

```ts
resolveBiasNodeHeightPx(
  nodeDimensions: NetworkNodeDimensionsLike,
  biasNodeLabelMetrics: BiasNodeLabelMetrics,
  isOutputNode: boolean,
): number
```

Resolves node rectangle height from label metrics and node role.

Parameters:
- `nodeDimensions` - - Shared node dimensions.
- `biasNodeLabelMetrics` - - Measured label metrics.
- `isOutputNode` - - Whether the node is an output node.

Returns: Render height for the node rectangle.

### drawBiasNodeScene

```ts
drawBiasNodeScene(
  context: CanvasRenderingContext2D,
  biasNodeScene: BiasNodeScene,
  nodeWidthPx: number,
): void
```

Draws a resolved node rectangle and optional bias label.

Parameters:
- `context` - - Render context.
- `biasNodeScene` - - Paint-ready node scene.
- `nodeWidthPx` - - Shared node width.

Returns: Nothing.

### shouldHideNetworkColorLegend

```ts
shouldHideNetworkColorLegend(
  context: CanvasRenderingContext2D,
): boolean
```

Determines whether the responsive viewport intentionally hides the overlay legend.

Parameters:
- `context` - - Render context.

Returns: True when the legend should be omitted.

### resolveLegendSceneContext

```ts
resolveLegendSceneContext(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
  colorScales: NetworkVisualizationColorScales,
): LegendSceneContext
```

Resolves the legend rows, layout, and architecture label bounds.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Multiline architecture label.
- `colorScales` - - Connection and bias color scales.

Returns: Legend scene context.

### drawLegendArchitectureLabel

```ts
drawLegendArchitectureLabel(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void
```

Draws the architecture label block above the legend frame.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendFrame

```ts
drawLegendFrame(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void
```

Draws the legend container frame.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendHeader

```ts
drawLegendHeader(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void
```

Draws the legend title row.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendConnectionSection

```ts
drawLegendConnectionSection(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void
```

Draws the connection-weight legend section.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendConnectionRow

```ts
drawLegendConnectionRow(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
  connectionLegendRow: ColorLegendRow,
  connectionRowTopPx: number,
): void
```

Draws a single connection legend row.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.
- `connectionLegendRow` - - Legend row.
- `connectionRowTopPx` - - Row top coordinate.

Returns: Nothing.

### drawLegendBiasSection

```ts
drawLegendBiasSection(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void
```

Draws the bias legend section.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendBiasRow

```ts
drawLegendBiasRow(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
  biasLegendRow: ColorLegendRow,
  biasRowTopPx: number,
): void
```

Draws a single bias legend row.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.
- `biasLegendRow` - - Legend row.
- `biasRowTopPx` - - Row top coordinate.

Returns: Nothing.

### drawLeftAlignedTextRows

```ts
drawLeftAlignedTextRows(
  context: CanvasRenderingContext2D,
  request: { lines: string[]; leftPx: number; topPx: number; lineHeightPx: number; font: string; fillStyle: string; },
): void
```

Draws multiline text rows aligned to a fixed left edge.

Parameters:
- `context` - - Render context.
- `request` - - Multiline text draw request.

Returns: Nothing.

### drawSquareDottedConnection

```ts
drawSquareDottedConnection(
  context: CanvasRenderingContext2D,
  input: { fromXPx: number; fromYPx: number; toXPx: number; toYPx: number; color: string; lineWidthPx: number; },
): void
```

Draws a square-dotted connection stroke for negative weights.

Parameters:
- `context` - - Render context.
- `input` - - Dotted-stroke endpoints and style.

Returns: Nothing.

## browser-entry/visualization/visualization.legend.utils.ts

Legend-layout helpers for network visualization.

The legend explains how colors map back to numeric weights and biases. These
helpers turn color scales into labeled rows and place the legend so it stays
readable across different canvas sizes.

### createColorLegendRows

```ts
createColorLegendRows(
  scale: DynamicColorScale,
  symbol: "w" | "b",
): ColorLegendRow[]
```

Creates legend rows from ordered tiers.

Each row describes one closed numeric interval and the swatch used to paint
it, making the dynamic color scale legible to a human reader.

Parameters:
- `scale` - - Dynamic color scale containing bounds, tiers, and overflow color.
- `symbol` - - Label symbol.

Returns: Legend rows.

### resolveNetworkLegendLayout

```ts
resolveNetworkLegendLayout(
  context: CanvasRenderingContext2D,
  connectionLegendRows: ColorLegendRow[],
  biasLegendRows: ColorLegendRow[],
): NetworkLegendLayout
```

Resolves network legend layout from canvas constraints.

The legend layout adapts between regular and compact modes so the network
panel can stay informative on smaller viewports without swallowing the whole
canvas.

Parameters:
- `context` - - Render context.
- `connectionLegendRows` - - Connection legend rows.
- `biasLegendRows` - - Bias legend rows.

Returns: Computed legend layout.

### resolveDefaultNetworkLegendLayout

```ts
resolveDefaultNetworkLegendLayout(
  context: CanvasRenderingContext2D,
  network: default | undefined,
): NetworkLegendLayout
```

Resolves default legend layout from internal tier definitions.

This convenience helper is used when the caller wants a layout driven by the
currently active network and does not need to assemble the intermediate rows
manually.

Parameters:
- `context` - - Render context.
- `network` - - Active network instance.

Returns: Legend layout.

## browser-entry/visualization/visualization.colors.utils.ts

Color-scale synthesis helpers for network visualization.

These utilities convert raw connection weights and node biases into tiered
neon color scales. The goal is not photorealism; it is interpretability. A
reader should be able to glance at the network panel and see where strong
positive, strong negative, and near-zero values live.

### createLogDivergingColorTiers

```ts
createLogDivergingColorTiers(
  input: { maxAbsValue: number; centerBlueThreshold: number; negativePalette: readonly string[]; centerBluePalette: readonly string[]; positivePalette: readonly string[]; logarithmicSteepness: number; edgeStartAbsValue?: number | undefined; edgeTierCount?: number | undefined; },
): ColorTier[]
```

Builds logarithmic diverging color tiers with a center band and edge extension.

Diverging scales are useful here because network parameters naturally split
around zero. Negative and positive values should feel visually related, but
not identical.

Parameters:
- `input` - - Tier creation options.

Returns: Ordered tier list.

### resolveTierColor

```ts
resolveTierColor(
  value: number,
  tiers: ColorTier[],
  aboveTierColor: string,
): string
```

Resolves a color from ordered tier definitions.

This is the final classification step that maps one numeric weight or bias to
the swatch color the renderer should paint.

Parameters:
- `value` - - Numeric value to classify.
- `tiers` - - Ordered tier list.
- `aboveTierColor` - - Fallback color for values above the last tier.

Returns: Resolved color string.

### resolveConnectionRangeColor

```ts
resolveConnectionRangeColor(
  connectionWeight: number,
): string
```

Resolves connection color for a raw weight.

This small helper is useful when one-off drawing code wants the same color
semantics as the full dynamic scale machinery.

Parameters:
- `connectionWeight` - - Connection weight.

Returns: Tier color.

### resolveBiasRangeColor

```ts
resolveBiasRangeColor(
  nodeBias: number,
): string
```

Resolves bias color for a raw node bias.

Bias colors follow the same diverging logic as connection colors so the legend
remains conceptually consistent across channels.

Parameters:
- `nodeBias` - - Node bias.

Returns: Tier color.

### resolveNetworkVisualizationColorScales

```ts
resolveNetworkVisualizationColorScales(
  network: default | undefined,
): NetworkVisualizationColorScales
```

Resolves dynamic connection/bias color scales from the active network range.

The active network may contain only a narrow slice of the full theoretical
value range, so the legend adapts to what is currently present instead of
always rendering a fixed generic scale.

Parameters:
- `network` - - Active network.

Returns: Dynamic scales used by graph drawing and legend rows.

## browser-entry/visualization/visualization.topology.utils.ts

Shared topology formatting helpers used by network-view and visualization.

The topology boundary resolves node layering, while this helper module adds a
few small presentation-oriented utilities that are reused by the visualization
panel.

### formatNodeBiasLabel

```ts
formatNodeBiasLabel(
  nodeBias: number,
): string
```

Formats node bias labels with fixed sign and precision.

Consistent sign and precision make dense node labels easier to scan quickly in
the rendered network panel.

Parameters:
- `nodeBias` - - Node bias value.

Returns: Label text.

### resolveNetworkVisualizationLayers

```ts
resolveNetworkVisualizationLayers(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNodeLike[][]
```

Resolves layered node groups for network-view layout and rendering.

Educational note:
Layer grouping is a network-view concern because it drives sizing, node
placement, and architecture presentation. Visualization code can still reuse
the result, but this helper now lives with the module that owns layout.

The resolver prefers explicit layer metadata when it exists, then falls back
to a topology-derived depth estimate so even loosely structured networks can
still be drawn in an intelligible left-to-right order.

Parameters:
- `network` - - Runtime network instance.
- `inputSize` - - Input count fallback.
- `outputSize` - - Output count fallback.

Returns: Layered nodes for rendering.

## browser-entry/visualization/visualization.errors.ts

Error text for non-finite legend bounds.

### assertFiniteLegendBound

```ts
assertFiniteLegendBound(
  value: number,
): void
```

Guards legend-bound formatting against non-finite values.

Parameters:
- `value` - - Legend bound candidate.

Returns: Nothing.

### FLAPPY_VISUALIZATION_NON_FINITE_BOUND_ERROR_MESSAGE

Error text for non-finite legend bounds.

### VisualizationNonFiniteBoundError

Thrown when a legend bound cannot be safely formatted.
