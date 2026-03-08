# browser-entry/visualization

## browser-entry/visualization/visualization.types.ts

### DynamicColorScale

Dynamic tiered color scale used by visualization render layers.

### NetworkVisualizationColorScales

Grouped color scales for connection and bias channels.

## browser-entry/visualization/visualization.ts

### resolveNetworkVisualizationLayers

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Resolves layered node groups for network-view layout and rendering.

Educational note:
Layer grouping is a network-view concern because it drives sizing, node
placement, and architecture presentation. Visualization code can still reuse
the result, but this helper now lives with the module that owns layout.

Parameters:
- `network` - - Runtime network instance.
- `inputSize` - - Input count fallback.
- `outputSize` - - Output count fallback.

Returns: Layered nodes for rendering.

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

`(context: CanvasRenderingContext2D, positionedNodes: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, biasScale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

Draws all network nodes with bias labels.

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

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext, biasLegendRow: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow, biasRowTopPx: number) => void`

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

`(context: CanvasRenderingContext2D, legendSceneContext: LegendSceneContext, connectionLegendRow: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow, connectionRowTopPx: number) => void`

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

`(context: CanvasRenderingContext2D, architectureLabel: string, colorScales: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales) => void`

Draws the color legend for connections and node bias values.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Compact architecture description.
- `colorScales` - - Connection and bias color scales.

Returns: Nothing.

### drawNetworkVisualizationHeader

`(context: CanvasRenderingContext2D, architectureLabel: string) => void`

Draws network architecture header text.

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

`(context: CanvasRenderingContext2D, runtimeConnections: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike[], positionByNodeIndex: Map<number, import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>, connectionScale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

Draws weighted connection lines.

Parameters:
- `context` - - Render context.
- `runtimeConnections` - - Runtime connection list.
- `positionByNodeIndex` - - Node layout map.
- `connectionScale` - - Dynamic connection color scale.

Returns: Nothing.

### resolveBiasNodeHeightPx

`(nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, biasNodeLabelMetrics: BiasNodeLabelMetrics, isOutputNode: boolean) => number`

Resolves node rectangle height from label metrics and node role.

Parameters:
- `nodeDimensions` - - Shared node dimensions.
- `biasNodeLabelMetrics` - - Measured label metrics.
- `isOutputNode` - - Whether the node is an output node.

Returns: Render height for the node rectangle.

### resolveBiasNodeLabelMetrics

`(context: CanvasRenderingContext2D, nodeLabel: string, nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => BiasNodeLabelMetrics`

Measures a bias label and resolves its font declaration.

Parameters:
- `context` - - Render context.
- `nodeLabel` - - Bias label string.
- `nodeDimensions` - - Shared node dimensions.

Returns: Measured label metrics.

### resolveBiasNodePaintStyle

`(positionedNode: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike, biasScale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => BiasNodePaintStyle`

Resolves node fill, stroke, and glow styling.

Parameters:
- `positionedNode` - - Positioned node payload.
- `biasScale` - - Bias color scale.

Returns: Node paint style.

### resolveBiasNodeScene

`(context: CanvasRenderingContext2D, positionedNode: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike, nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, halfNodeWidthPx: number, biasScale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => BiasNodeScene`

Resolves all paint attributes needed to render a single node.

Parameters:
- `context` - - Render context.
- `positionedNode` - - Positioned node payload.
- `nodeDimensions` - - Shared node dimensions.
- `halfNodeWidthPx` - - Cached half node width.
- `biasScale` - - Bias color scale.

Returns: Paint-ready node scene.

### resolveLegendSceneContext

`(context: CanvasRenderingContext2D, architectureLabel: string, colorScales: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales) => LegendSceneContext`

Resolves the legend rows, layout, and architecture label bounds.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Multiline architecture label.
- `colorScales` - - Connection and bias color scales.

Returns: Legend scene context.

### resolveWeightedConnectionScene

`(runtimeConnection: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike, positionByNodeIndex: Map<number, import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>, connectionScale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => WeightedConnectionScene | undefined`

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

`(input: { maxAbsValue: number; centerBlueThreshold: number; negativePalette: readonly string[]; centerBluePalette: readonly string[]; positivePalette: readonly string[]; logarithmicSteepness: number; edgeStartAbsValue?: number | undefined; edgeTierCount?: number | undefined; }) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[]`

Builds logarithmic diverging color tiers with a center band and edge extension.

Parameters:
- `input` - - Tier creation options.

Returns: Ordered tier list.

### resolveBiasRangeColor

`(nodeBias: number) => string`

Resolves bias color for a raw node bias.

Parameters:
- `nodeBias` - - Node bias.

Returns: Tier color.

### resolveConnectionRangeColor

`(connectionWeight: number) => string`

Resolves connection color for a raw weight.

Parameters:
- `connectionWeight` - - Connection weight.

Returns: Tier color.

### resolveNetworkVisualizationColorScales

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales`

Resolves dynamic connection/bias color scales from the active network range.

Parameters:
- `network` - - Active network.

Returns: Dynamic scales used by graph drawing and legend rows.

### resolveTierColor

`(value: number, tiers: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[], aboveTierColor: string) => string`

Resolves a color from ordered tier definitions.

Parameters:
- `value` - - Numeric value to classify.
- `tiers` - - Ordered tier list.
- `aboveTierColor` - - Fallback color for values above the last tier.

Returns: Resolved color string.

## browser-entry/visualization/visualization.legend.utils.ts

### createColorLegendRows

`(scale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale, symbol: "w" | "b") => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]`

Creates legend rows from ordered tiers.

Parameters:
- `scale` - - Dynamic color scale containing bounds, tiers, and overflow color.
- `symbol` - - Label symbol.

Returns: Legend rows.

### resolveDefaultNetworkLegendLayout

`(context: CanvasRenderingContext2D, network: import("C:/NeatapticTS/src/architecture/network").default | undefined) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves default legend layout from internal tier definitions.

Parameters:
- `context` - - Render context.
- `network` - - Active network instance.

Returns: Legend layout.

### resolveNetworkLegendLayout

`(context: CanvasRenderingContext2D, connectionLegendRows: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[], biasLegendRows: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves network legend layout from canvas constraints.

Parameters:
- `context` - - Render context.
- `connectionLegendRows` - - Connection legend rows.
- `biasLegendRows` - - Bias legend rows.

Returns: Computed legend layout.

## browser-entry/visualization/visualization.topology.utils.ts

### formatNodeBiasLabel

`(nodeBias: number) => string`

Formats node bias labels with fixed sign and precision.

Parameters:
- `nodeBias` - - Node bias value.

Returns: Label text.

### resolveNetworkVisualizationLayers

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Resolves layered node groups for network-view layout and rendering.

Educational note:
Layer grouping is a network-view concern because it drives sizing, node
placement, and architecture presentation. Visualization code can still reuse
the result, but this helper now lives with the module that owns layout.

Parameters:
- `network` - - Runtime network instance.
- `inputSize` - - Input count fallback.
- `outputSize` - - Output count fallback.

Returns: Layered nodes for rendering.
