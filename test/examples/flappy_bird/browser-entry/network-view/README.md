# browser-entry/network-view

## browser-entry/network-view/network-view.types.ts

### network-view.types

Input-group label band geometry and style contract.

### InputGroupLabelBand

Input-group label band geometry and style contract.

## browser-entry/network-view/network-view.ts

### clampRecommendedNetworkHeightPx

`(recommendedHeightPx: number) => number`

Clamps a recommended network height into the configured panel range.

Parameters:

- `recommendedHeightPx` - - Recommended panel height.

Returns: Clamped panel height.

### createPositionByNodeIndex

`(centeredPositionedNodes: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[]) => Map<number, import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>`

Builds a node-index lookup map for resolved positioned nodes.

Parameters:

- `centeredPositionedNodes` - - Positioned nodes after centering.

Returns: Map keyed by node index.

### drawNetworkVisualization

`(context: CanvasRenderingContext2D, network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => void`

Draws a complete, layer-based visualization of the active network.

Parameters:

- `context` - - Canvas 2D drawing context.
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Nothing.

### drawPositionedNetworkGraph

`(context: CanvasRenderingContext2D, networkVisualizationScene: NetworkVisualizationScene, positionedNetworkGraphScene: PositionedNetworkGraphScene) => void`

Draws the positioned graph layers and optional guide overlays.

Parameters:

- `context` - - Canvas 2D drawing context.
- `networkVisualizationScene` - - Frame scene context.
- `positionedNetworkGraphScene` - - Positioned graph scene.

Returns: Nothing.

### formatArchitectureLabel

`(architectureInputSize: number, hiddenLayersLabel: string, architectureOutputSize: number, totalNodeCount: number, totalConnectionCount: number) => string`

Formats the two-line architecture label used by the header and legend.

Parameters:

- `architectureInputSize` - - Input layer size.
- `hiddenLayersLabel` - - Hidden-layer description.
- `architectureOutputSize` - - Output layer size.
- `totalNodeCount` - - Total node count.
- `totalConnectionCount` - - Total connection count.

Returns: Formatted architecture label.

### paintNetworkVisualizationCanvasBase

`(context: CanvasRenderingContext2D, networkVisualizationScene: NetworkVisualizationScene) => void`

Paints the base network visualization canvas background.

Parameters:

- `context` - - Canvas 2D drawing context.
- `networkVisualizationScene` - - Frame scene context.

Returns: Nothing.

### resolveAdjustedGraphPaddingContext

`(context: CanvasRenderingContext2D, network: import("C:/NeatapticTS/src/architecture/network").default | undefined, canvasWidthPx: number, hideNetworkOverlays: boolean, graphPaddingContext: NetworkGraphPaddingContext) => Pick<NetworkGraphPaddingContext, "graphLeftPaddingPx" | "graphRightPaddingPx">`

Adjusts graph-side padding to keep the floating legend from overlapping nodes.

Parameters:

- `context` - - Canvas 2D drawing context.
- `network` - - Network to visualize.
- `canvasWidthPx` - - Canvas width.
- `hideNetworkOverlays` - - Whether overlays are hidden.
- `graphPaddingContext` - - Base graph padding context.

Returns: Adjusted graph padding context.

### resolveBaseGraphPaddingContext

`() => NetworkGraphPaddingContext`

Resolves the base graph padding before legend-aware adjustments are applied.

Returns: Base graph padding context.

### resolveHiddenLayersLabel

`(hiddenLayerSizes: number[], architectureSource: "layer-metadata" | "graph-topology" | "inferred") => string`

Resolves the hidden-layer portion of the compact architecture label.

Parameters:

- `hiddenLayerSizes` - - Hidden-layer sizes.
- `architectureSource` - - Architecture source metadata.

Returns: Hidden-layer label.

### resolveNetworkArchitectureLabel

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => string`

Resolves compact architecture label text for headers and HUD rows.

Parameters:

- `network` - - Network to describe.
- `inputSize` - - Configured input size.
- `outputSize` - - Configured output size.

Returns: Readable architecture label.

### resolveNetworkDrawableArea

`(networkVisualizationScene: NetworkVisualizationScene) => NetworkDrawableArea`

Resolves the drawable graph area after scene padding is applied.

Parameters:

- `networkVisualizationScene` - - Frame scene context.

Returns: Drawable area dimensions.

### resolveNetworkNodeDimensions

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number, drawableWidthPx: number, drawableHeightPx: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike`

Resolves node rectangle dimensions from topology density and drawable bounds.

Parameters:

- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.
- `drawableWidthPx` - - Drawable graph width.
- `drawableHeightPx` - - Drawable graph height.

Returns: Node dimensions.

### resolveNetworkNodeDimensionsFromTopologySummary

`(networkTopologySummary: NetworkTopologySummary, drawableWidthPx: number, drawableHeightPx: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike`

Resolves node rectangle dimensions from topology density and drawable bounds.

Parameters:

- `networkTopologySummary` - - Topology summary.
- `drawableWidthPx` - - Drawable graph width.
- `drawableHeightPx` - - Drawable graph height.

Returns: Node dimensions.

### resolveNetworkTopologySummary

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => NetworkTopologySummary`

Resolves a reusable topology summary for layout and sizing helpers.

Parameters:

- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Topology summary.

### resolveNetworkVisualizationHeightPx

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => number`

Resolves responsive visualization canvas height from network shape.

Parameters:

- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Recommended height in pixels.

### resolveNetworkVisualizationScene

`(context: CanvasRenderingContext2D, network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => NetworkVisualizationScene`

Resolves all non-topology canvas state needed to draw the network view.

Parameters:

- `context` - - Canvas 2D drawing context.
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Scene context for the current frame.

### resolvePositionedNetworkGraphScene

`(networkVisualizationScene: NetworkVisualizationScene, network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => PositionedNetworkGraphScene`

Resolves positioned nodes, connection lookup state, and shared node dimensions.

Parameters:

- `networkVisualizationScene` - - Frame scene context.
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Positioned graph scene.

### resolveRecommendedNetworkHeightPx

`(networkTopologySummary: NetworkTopologySummary, topologyDrivenHeightPx: number) => number`

Resolves the recommended panel height from topology and density adjustments.

Parameters:

- `networkTopologySummary` - - Topology summary.
- `topologyDrivenHeightPx` - - Minimum readable topology height.

Returns: Recommended panel height.

### resolveRuntimeConnections

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike[]`

Resolves the runtime connection array from the active network.

Parameters:

- `network` - - Network to visualize.

Returns: Runtime connection list.

### resolveTopologyDrivenHeightPx

`(networkTopologySummary: NetworkTopologySummary) => number`

Resolves the topology-driven minimum readable height.

Parameters:

- `networkTopologySummary` - - Topology summary.

Returns: Minimum readable height in pixels.

### shouldHideNetworkOverlays

`(context: CanvasRenderingContext2D, fallbackViewportWidthPx: number) => boolean`

Determines whether responsive rules hide auxiliary network overlays.

Parameters:

- `context` - - Canvas 2D drawing context.
- `fallbackViewportWidthPx` - - Fallback viewport width.

Returns: True when overlays should be hidden.

## browser-entry/network-view/network-view.constants.ts

### network-view.constants

Ordered labels for grouped Flappy network input bands.

### FLAPPY_INPUT_GROUP_LABELS

## browser-entry/network-view/network-view.draw.service.ts

### drawInputGroupLabelBands

`(context: CanvasRenderingContext2D, positionedNodes: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => void`

Draws vertical neon bands that label semantic groups in the input layer.

Parameters:

- `context` - - Canvas 2D rendering context.
- `positionedNodes` - - Positioned nodes in graph coordinates.
- `nodeDimensions` - - Resolved node dimensions.

Returns: Nothing.

### drawRoundedRect

`(context: CanvasRenderingContext2D, leftXPx: number, topYPx: number, widthPx: number, heightPx: number, radiusPx: number, fillColor: string) => void`

Draws a filled rounded rectangle path.

## browser-entry/network-view/network-view.labels.utils.ts

### resolveInputGroupLabelBands

`(inputNodeCount: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/network-view/network-view.types").InputGroupLabelBand[]`

Resolves input-layer semantic label bands for Flappy temporal observation channels.

Parameters:

- `inputNodeCount` - - Input-layer node count.

Returns: Group label ranges with band colors.

## browser-entry/network-view/network-view.layout.utils.ts

### centerPositionedNodesInDrawableArea

`(positionedNodes: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], leftPaddingPx: number, topPaddingPx: number, drawableWidthPx: number, drawableHeightPx: number, nodeLayoutPaddingPx: number, nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[]`

Centers positioned nodes within the drawable graph area.

Parameters:

- `positionedNodes` - - Positioned nodes before centering.
- `leftPaddingPx` - - Left graph padding.
- `topPaddingPx` - - Top graph padding.
- `drawableWidthPx` - - Drawable graph width.
- `drawableHeightPx` - - Drawable graph height.
- `nodeLayoutPaddingPx` - - Inner graph padding.
- `nodeDimensions` - - Node dimensions.

Returns: Center-aligned positioned nodes.

### positionNetworkNodes

`(networkLayers: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][], leftPaddingPx: number, topPaddingPx: number, drawableWidthPx: number, drawableHeightPx: number, nodeLayoutPaddingPx: number, nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[]`

Positions network nodes into drawable canvas coordinates.

Parameters:

- `networkLayers` - - Resolved network layers.
- `leftPaddingPx` - - Left graph padding.
- `topPaddingPx` - - Top graph padding.
- `drawableWidthPx` - - Drawable graph width.
- `drawableHeightPx` - - Drawable graph height.
- `nodeLayoutPaddingPx` - - Inner graph padding.
- `nodeDimensions` - - Node dimensions.

Returns: Positioned nodes.

## browser-entry/network-view/network-view.topology.utils.ts

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
