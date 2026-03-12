# browser-entry/visualization

## browser-entry/visualization/visualization.types.ts

### DynamicColorScale

Visualization-specific color-scale contracts for network rendering.

The Flappy Bird demo renders connection weights and node biases with tiered
neon ramps so humans can quickly read sign and magnitude without parsing raw
numbers on every edge and node.

### NetworkVisualizationColorScales

Grouped color scales for connection and bias channels.

Keeping the two scales together ensures the legend and drawing code read from
one consistent view of the active network range.

## browser-entry/visualization/visualization.ts

### visualization

Public visualization facade for browser-entry network rendering.

The dedicated visualization folder focuses on turning network structure and
parameter ranges into readable graphics. Layer resolution itself is shared
with the neighboring network-view topology boundary, so this facade exposes
that topology helper while keeping the visualization subsystem's public story
in one place.

### resolveNetworkVisualizationLayers

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Topology resolution helpers for the browser network view.

These helpers answer a key visualization question: how should the current
network be partitioned into ordered layers so layout and architecture labels
stay meaningful even when some metadata is missing?

## browser-entry/visualization/visualization.errors.ts

### visualization.errors

Error text for non-finite legend bounds.

### assertFiniteLegendBound

`(value: number) => void`

Guards legend-bound formatting against non-finite values.

Parameters:
- `value` - - Legend bound candidate.

Returns: Nothing.

### FLAPPY_VISUALIZATION_NON_FINITE_BOUND_ERROR_MESSAGE

### VisualizationNonFiniteBoundError

Thrown when a legend bound cannot be safely formatted.

## browser-entry/visualization/visualization.constants.ts

### visualization.constants

Pixel side length for dotted negative-connection square markers.

### FLAPPY_NETWORK_DISABLED_CONNECTION_ALPHA

### FLAPPY_NETWORK_DISABLED_CONNECTION_DASH_PATTERN

### FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON

### FLAPPY_NETWORK_DOTTED_CONNECTION_SQUARE_SIDE_PX

### FLAPPY_NETWORK_DOTTED_CONNECTION_STEP_COMPACT_RATIO

### FLAPPY_NETWORK_DOTTED_CONNECTION_WIDTH_SPACING_RATIO

### FLAPPY_NETWORK_ENABLED_CONNECTION_ALPHA

### FLAPPY_NETWORK_HEADER_LINE_HEIGHT_PX

### FLAPPY_NETWORK_HEADER_PADDING_PX

### FLAPPY_NETWORK_HIDDEN_NODE_STROKE_WIDTH_PX

### FLAPPY_NETWORK_HIDDEN_NODE_VERTICAL_PADDING_PX

### FLAPPY_NETWORK_LEGEND_ARCHITECTURE_GAP_PX

### FLAPPY_NETWORK_LEGEND_BIAS_LABEL_X_PX

### FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_SIZE_PX

### FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_X_PX

### FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_Y_PX

### FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX

### FLAPPY_NETWORK_LEGEND_CONNECTION_LABEL_X_PX

### FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_END_X_PX

### FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_Y_OFFSET_PX

### FLAPPY_NETWORK_LEGEND_HEADER_TOP_PADDING_PX

### FLAPPY_NETWORK_LEGEND_MIN_ARCHITECTURE_TOP_PX

### FLAPPY_NETWORK_MIN_RENDER_NODE_HEIGHT_PX

### FLAPPY_NETWORK_OUTPUT_NODE_HEIGHT_REDUCTION_PX

### FLAPPY_NETWORK_OUTPUT_NODE_SHADOW_BLUR_PX

### FLAPPY_NETWORK_OUTPUT_NODE_STROKE_WIDTH_PX

## browser-entry/visualization/visualization.draw.service.ts

### drawBiasNodeScene

`(context: CanvasRenderingContext2D, biasNodeScene: BiasNodeScene, nodeWidthPx: number) => void`

Draws a resolved node rectangle and optional bias label.

Parameters:
- `context` - - Render context.
- `biasNodeScene` - - Paint-ready node scene.
- `nodeWidthPx` - - Shared node width.

Returns: Nothing.

### drawBiasNodesLayer

`(context: CanvasRenderingContext2D, positionedNodes: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, biasScale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

Draws all network nodes with bias labels.

The node layer pairs each rectangle with a compact bias label so the panel can
show both topology and a lightweight hint of parameter state.

Parameters:
- `context` - - Render context.
- `positionedNodes` - - Positioned nodes.
- `nodeDimensions` - - Node dimensions.
- `biasScale` - - Dynamic bias color scale.

Returns: Nothing.

### drawLeftAlignedTextRows

`(context: CanvasRenderingContext2D, request: { lines: string[]; leftPx: number; topPx: number; lineHeightPx: number; font: string; fillStyle: string; }) => void`

Draws multiline text rows aligned to a fixed left edge.

Parameters:
- `context` - - Render context.
- `request` - - Multiline text draw request.

Returns: Nothing.

### drawLegendArchitectureLabel

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext) => void`

Draws the architecture label block above the legend frame.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendBiasRow

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext, biasLegendRow: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow, biasRowTopPx: number) => void`

Draws a single bias legend row.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.
- `biasLegendRow` - - Legend row.
- `biasRowTopPx` - - Row top coordinate.

Returns: Nothing.

### drawLegendBiasSection

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext) => void`

Draws the bias legend section.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendConnectionRow

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext, connectionLegendRow: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow, connectionRowTopPx: number) => void`

Draws a single connection legend row.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.
- `connectionLegendRow` - - Legend row.
- `connectionRowTopPx` - - Row top coordinate.

Returns: Nothing.

### drawLegendConnectionSection

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext) => void`

Draws the connection-weight legend section.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendFrame

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext) => void`

Draws the legend container frame.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawLegendHeader

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext) => void`

Draws the legend title row.

Parameters:
- `context` - - Render context.
- `legendSceneContext` - - Legend scene context.

Returns: Nothing.

### drawNetworkColorLegend

`(context: CanvasRenderingContext2D, architectureLabel: string, colorScales: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales) => void`

Draws the color legend for connections and node bias values.

This legend is what turns the panel from "colorful art" into an interpretable
instrument: it tells the viewer what each weight and bias color actually
means numerically.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Compact architecture description.
- `colorScales` - - Connection and bias color scales.

Returns: Nothing.

### drawNetworkVisualizationHeader

`(context: CanvasRenderingContext2D, architectureLabel: string) => void`

Draws network architecture header text.

The header gives viewers a compact architecture summary before they inspect
individual nodes and edges.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Header label.

Returns: Nothing.

### drawSquareDottedConnection

`(context: CanvasRenderingContext2D, input: { fromXPx: number; fromYPx: number; toXPx: number; toYPx: number; color: string; lineWidthPx: number; }) => void`

Draws a square-dotted connection stroke for negative weights.

Parameters:
- `context` - - Render context.
- `input` - - Dotted-stroke endpoints and style.

Returns: Nothing.

### drawWeightedConnectionScene

`(context: CanvasRenderingContext2D, weightedConnectionScene: WeightedConnectionScene) => void`

Draws a previously resolved weighted connection scene.

Parameters:
- `context` - - Render context.
- `weightedConnectionScene` - - Render-ready connection scene.

Returns: Nothing.

### drawWeightedConnectionsLayer

`(context: CanvasRenderingContext2D, runtimeConnections: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike[], positionByNodeIndex: Map<number, import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>, connectionScale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

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

### resolveBiasNodeHeightPx

`(nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, biasNodeLabelMetrics: BiasNodeLabelMetrics, isOutputNode: boolean) => number`

Resolves node rectangle height from label metrics and node role.

Parameters:
- `nodeDimensions` - - Shared node dimensions.
- `biasNodeLabelMetrics` - - Measured label metrics.
- `isOutputNode` - - Whether the node is an output node.

Returns: Render height for the node rectangle.

### resolveBiasNodeLabelMetrics

`(context: CanvasRenderingContext2D, nodeLabel: string, nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => BiasNodeLabelMetrics`

Measures a bias label and resolves its font declaration.

Parameters:
- `context` - - Render context.
- `nodeLabel` - - Bias label string.
- `nodeDimensions` - - Shared node dimensions.

Returns: Measured label metrics.

### resolveBiasNodePaintStyle

`(positionedNode: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike, biasScale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => BiasNodePaintStyle`

Resolves node fill, stroke, and glow styling.

Parameters:
- `positionedNode` - - Positioned node payload.
- `biasScale` - - Bias color scale.

Returns: Node paint style.

### resolveBiasNodeScene

`(context: CanvasRenderingContext2D, positionedNode: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike, nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, halfNodeWidthPx: number, biasScale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => BiasNodeScene`

Resolves all paint attributes needed to render a single node.

Parameters:
- `context` - - Render context.
- `positionedNode` - - Positioned node payload.
- `nodeDimensions` - - Shared node dimensions.
- `halfNodeWidthPx` - - Cached half node width.
- `biasScale` - - Bias color scale.

Returns: Paint-ready node scene.

### resolveLegendSceneContext

`(context: CanvasRenderingContext2D, architectureLabel: string, colorScales: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales) => LegendSceneContext`

Resolves the legend rows, layout, and architecture label bounds.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Multiline architecture label.
- `colorScales` - - Connection and bias color scales.

Returns: Legend scene context.

### resolveWeightedConnectionScene

`(runtimeConnection: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike, positionByNodeIndex: Map<number, import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>, connectionScale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => WeightedConnectionScene | undefined`

Resolves a renderable connection scene from runtime data and node positions.

Parameters:
- `runtimeConnection` - - Candidate runtime connection.
- `positionByNodeIndex` - - Node layout map.
- `connectionScale` - - Connection color scale.

Returns: Renderable connection scene, when both endpoint nodes exist.

### shouldHideNetworkColorLegend

`(context: CanvasRenderingContext2D) => boolean`

Determines whether the responsive viewport intentionally hides the overlay legend.

Parameters:
- `context` - - Render context.

Returns: True when the legend should be omitted.

## browser-entry/visualization/visualization.colors.utils.ts

### createLogDivergingColorTiers

`(input: { maxAbsValue: number; centerBlueThreshold: number; negativePalette: readonly string[]; centerBluePalette: readonly string[]; positivePalette: readonly string[]; logarithmicSteepness: number; edgeStartAbsValue?: number | undefined; edgeTierCount?: number | undefined; }) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[]`

Color-scale synthesis helpers for network visualization.

These utilities convert raw connection weights and node biases into tiered
neon color scales. The goal is not photorealism; it is interpretability. A
reader should be able to glance at the network panel and see where strong
positive, strong negative, and near-zero values live.

### resolveBiasRangeColor

`(nodeBias: number) => string`

Resolves bias color for a raw node bias.

Bias colors follow the same diverging logic as connection colors so the legend
remains conceptually consistent across channels.

Parameters:
- `nodeBias` - - Node bias.

Returns: Tier color.

### resolveConnectionRangeColor

`(connectionWeight: number) => string`

Resolves connection color for a raw weight.

This small helper is useful when one-off drawing code wants the same color
semantics as the full dynamic scale machinery.

Parameters:
- `connectionWeight` - - Connection weight.

Returns: Tier color.

### resolveNetworkVisualizationColorScales

`(network: import("src/architecture/network").default | undefined) => import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales`

Resolves dynamic connection/bias color scales from the active network range.

The active network may contain only a narrow slice of the full theoretical
value range, so the legend adapts to what is currently present instead of
always rendering a fixed generic scale.

Parameters:
- `network` - - Active network.

Returns: Dynamic scales used by graph drawing and legend rows.

### resolveTierColor

`(value: number, tiers: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[], aboveTierColor: string) => string`

Resolves a color from ordered tier definitions.

This is the final classification step that maps one numeric weight or bias to
the swatch color the renderer should paint.

Parameters:
- `value` - - Numeric value to classify.
- `tiers` - - Ordered tier list.
- `aboveTierColor` - - Fallback color for values above the last tier.

Returns: Resolved color string.

## browser-entry/visualization/visualization.legend.utils.ts

### createColorLegendRows

`(scale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale, symbol: "w" | "b") => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]`

Legend-layout helpers for network visualization.

The legend explains how colors map back to numeric weights and biases. These
helpers turn color scales into labeled rows and place the legend so it stays
readable across different canvas sizes.

### resolveDefaultNetworkLegendLayout

`(context: CanvasRenderingContext2D, network: import("src/architecture/network").default | undefined) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves default legend layout from internal tier definitions.

This convenience helper is used when the caller wants a layout driven by the
currently active network and does not need to assemble the intermediate rows
manually.

Parameters:
- `context` - - Render context.
- `network` - - Active network instance.

Returns: Legend layout.

### resolveNetworkLegendLayout

`(context: CanvasRenderingContext2D, connectionLegendRows: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[], biasLegendRows: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves network legend layout from canvas constraints.

The legend layout adapts between regular and compact modes so the network
panel can stay informative on smaller viewports without swallowing the whole
canvas.

Parameters:
- `context` - - Render context.
- `connectionLegendRows` - - Connection legend rows.
- `biasLegendRows` - - Bias legend rows.

Returns: Computed legend layout.

## browser-entry/visualization/visualization.topology.utils.ts

### visualization.topology.utils

Shared topology formatting helpers used by network-view and visualization.

The topology boundary resolves node layering, while this helper module adds a
few small presentation-oriented utilities that are reused by the visualization
panel.

### formatNodeBiasLabel

`(nodeBias: number) => string`

Formats node bias labels with fixed sign and precision.

Consistent sign and precision make dense node labels easier to scan quickly in
the rendered network panel.

Parameters:
- `nodeBias` - - Node bias value.

Returns: Label text.

### resolveNetworkVisualizationLayers

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Topology resolution helpers for the browser network view.

These helpers answer a key visualization question: how should the current
network be partitioned into ordered layers so layout and architecture labels
stay meaningful even when some metadata is missing?
