# browser-entry/network-view

## browser-entry/network-view/network-view.types.ts

### network-view.types

Shared type contracts for network-view overlays.

The most notable overlay is the input-group label band system, which annotates
stacked temporal observation channels so the input layer reads as grouped
semantics instead of a flat strip of anonymous nodes.

### InputGroupLabelBand

Shared type contracts for network-view overlays.

The most notable overlay is the input-group label band system, which annotates
stacked temporal observation channels so the input layer reads as grouped
semantics instead of a flat strip of anonymous nodes.

## browser-entry/network-view/network-view.ts

### clampRecommendedNetworkHeightPx

`(recommendedHeightPx: number) => number`

Clamps a recommended network height into the configured panel range.

Parameters:
- `recommendedHeightPx` - - Recommended panel height.

Returns: Clamped panel height.

### createPositionByNodeIndex

`(centeredPositionedNodes: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[]) => Map<number, import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>`

Builds a node-index lookup map for resolved positioned nodes.

Parameters:
- `centeredPositionedNodes` - - Positioned nodes after centering.

Returns: Map keyed by node index.

### drawNetworkVisualization

`(context: CanvasRenderingContext2D, network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => void`

Draws a complete, layer-based visualization of the active network.

Conceptually, this is the main fold from network object to finished panel:
resolve scene state, compute layout, paint the graph, then paint overlays.

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

### NetworkTopologySummary

Network-view orchestration for the browser-side architecture panel.

This subsystem sits between raw network data and the lower-level visualization
drawing helpers. It resolves topology summaries, chooses panel size, lays out
nodes inside the drawable area, and coordinates overlays such as legends and
input-group bands.

### paintNetworkVisualizationCanvasBase

`(context: CanvasRenderingContext2D, networkVisualizationScene: NetworkVisualizationScene) => void`

Paints the base network visualization canvas background.

Parameters:
- `context` - - Canvas 2D drawing context.
- `networkVisualizationScene` - - Frame scene context.

Returns: Nothing.

### resolveAdjustedGraphPaddingContext

`(context: CanvasRenderingContext2D, network: import("src/architecture/network").default | undefined, canvasWidthPx: number, hideNetworkOverlays: boolean, graphPaddingContext: NetworkGraphPaddingContext) => Pick<NetworkGraphPaddingContext, "graphLeftPaddingPx" | "graphRightPaddingPx">`

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

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => string`

Resolves compact architecture label text for headers and HUD rows.

The label compresses the active network into a short human-readable summary:
input size, hidden-layer structure, output size, and graph size metadata.

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

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number, drawableWidthPx: number, drawableHeightPx: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike`

Resolves node rectangle dimensions from topology density and drawable bounds.

Parameters:
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.
- `drawableWidthPx` - - Drawable graph width.
- `drawableHeightPx` - - Drawable graph height.

Returns: Node dimensions.

### resolveNetworkNodeDimensionsFromTopologySummary

`(networkTopologySummary: NetworkTopologySummary, drawableWidthPx: number, drawableHeightPx: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike`

Resolves node rectangle dimensions from topology density and drawable bounds.

Parameters:
- `networkTopologySummary` - - Topology summary.
- `drawableWidthPx` - - Drawable graph width.
- `drawableHeightPx` - - Drawable graph height.

Returns: Node dimensions.

### resolveNetworkTopologySummary

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => NetworkTopologySummary`

Resolves a reusable topology summary for layout and sizing helpers.

Parameters:
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Topology summary.

### resolveNetworkVisualizationHeightPx

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => number`

Resolves responsive visualization canvas height from network shape.

Dense or deeper networks need more vertical room to stay readable, so panel
height is driven by topology rather than fixed to a single constant.

Parameters:
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Recommended height in pixels.

### resolveNetworkVisualizationScene

`(context: CanvasRenderingContext2D, network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => NetworkVisualizationScene`

Resolves all non-topology canvas state needed to draw the network view.

This separates frame-scene concerns such as canvas size, overlays, and color
scales from the later graph-topology layout step.

Parameters:
- `context` - - Canvas 2D drawing context.
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Scene context for the current frame.

### resolvePositionedNetworkGraphScene

`(networkVisualizationScene: NetworkVisualizationScene, network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => PositionedNetworkGraphScene`

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

`(network: import("src/architecture/network").default | undefined) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike[]`

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

`(context: CanvasRenderingContext2D, positionedNodes: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => void`

Overlay drawing helpers specific to the network-view panel.

These helpers render semantic guides that sit on top of the raw graph, most
notably the colored input-group bands that explain how temporal observation
channels are organized.

### drawRoundedRect

`(context: CanvasRenderingContext2D, leftXPx: number, topYPx: number, widthPx: number, heightPx: number, radiusPx: number, fillColor: string) => void`

Draws a filled rounded rectangle path.

This is the small geometry primitive used by the input-group band renderer.

## browser-entry/network-view/network-view.labels.utils.ts

### resolveInputGroupLabelBands

`(inputNodeCount: number) => import("test/examples/flappy_bird/browser-entry/network-view/network-view.types").InputGroupLabelBand[]`

Semantic input-label helpers for the network-view panel.

The Flappy controller input layer is not just a list of anonymous scalars; it
is organized into stacked observation frames plus action-history channels.
These helpers recover that grouping for visual annotation.

## browser-entry/network-view/network-view.layout.utils.ts

### centerPositionedNodesInDrawableArea

`(positionedNodes: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], leftPaddingPx: number, topPaddingPx: number, drawableWidthPx: number, drawableHeightPx: number, nodeLayoutPaddingPx: number, nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[]`

Centers positioned nodes within the drawable graph area.

Positioning establishes relative structure first; centering then shifts the
whole graph as a block so it sits comfortably within the padded draw region.

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

`(networkLayers: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][], leftPaddingPx: number, topPaddingPx: number, drawableWidthPx: number, drawableHeightPx: number, nodeLayoutPaddingPx: number, nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[]`

Node-positioning helpers for the browser network view.

Once topology has been resolved into layers, these helpers place nodes inside
the drawable panel and then center the final graph so it feels balanced inside
the available canvas space.

## browser-entry/network-view/network-view.topology.utils.ts

### resolveNetworkVisualizationLayers

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Topology resolution helpers for the browser network view.

These helpers answer a key visualization question: how should the current
network be partitioned into ordered layers so layout and architecture labels
stay meaningful even when some metadata is missing?
