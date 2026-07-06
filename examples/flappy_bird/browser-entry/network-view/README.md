# browser-entry/network-view

Network-view orchestration for the browser-side architecture panel.

This module is the browser-facing fold from a live evolved controller to a
readable inspection panel. It does not own the low-level drawing primitives,
and it does not invent topology semantics from scratch. Instead it composes
both into one higher-level question: how should this network be laid out so a
human can actually learn from it?

The boundary exists because "draw the network" hides several distinct jobs:
summarize topology, size the panel, place nodes, choose overlay policy, and
then delegate the final painting work. Keeping those steps together here makes
the generated README read like an inspection chapter instead of a pile of
canvas helpers.

## browser-entry/network-view/network-view.ts

### clampRecommendedNetworkHeightPx

```ts
clampRecommendedNetworkHeightPx(
  recommendedHeightPx: number,
): number
```

Clamps a recommended network height into the configured panel range.

Parameters:
- `recommendedHeightPx` - Recommended panel height.

Returns: Clamped panel height.

### createPositionByNodeIndex

```ts
createPositionByNodeIndex(
  centeredPositionedNodes: PositionedNetworkNodeLike[],
): Map<number, PositionedNetworkNodeLike>
```

Builds a node-index lookup map for resolved positioned nodes.

Parameters:
- `centeredPositionedNodes` - Positioned nodes after centering.

Returns: Map keyed by node index.

### drawNetworkVisualization

```ts
drawNetworkVisualization(
  context: CanvasRenderingContext2D,
  network: default | undefined,
  inputSize: number,
  outputSize: number,
  hoverState: NetworkVisualizationHoverState | undefined,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
  connectionLayerStyle: Partial<WeightedConnectionLayerStyle> | undefined,
): NetworkVisualizationPositionedScene
```

Draws a complete, layer-based visualization of the active network.

Conceptually, this is the main fold from network object to finished panel:
resolve scene state, compute layout, paint the graph, then paint overlays.

Parameters:
- `context` - Canvas 2D drawing context.
- `network` - Network to visualize.
- `inputSize` - Input-layer size.
- `outputSize` - Output-layer size.
- `hoverState` - Optional host-owned hover state for interactive emphasis.
- `inputLabelGroupDefinitions` - Optional semantic input-label group definitions.
- `connectionLayerStyle` - Optional override for connection stroke visibility.

Returns: Positioned node snapshot reused by host-side hover hit testing.

Example:

```ts
drawNetworkVisualization(networkContext, bestNetwork, 12, 2);
```

### drawPositionedNetworkGraph

```ts
drawPositionedNetworkGraph(
  context: CanvasRenderingContext2D,
  resolvedNetworkVisualizationFrame: NetworkVisualizationResolvedFrame,
  hoverState: NetworkVisualizationHoverState | undefined,
  connectionLayerStyle: Partial<WeightedConnectionLayerStyle> | undefined,
): void
```

Draws the positioned graph layers and optional guide overlays.

Parameters:
- `context` - Canvas 2D drawing context.
- `resolvedNetworkVisualizationFrame` - Resolved network visualization frame containing positioned scene, connections, and color scales.
- `hoverState` - Optional host-owned hover state for interactive emphasis.
- `connectionLayerStyle` - Optional override for connection stroke visibility.

Returns: Nothing.

### drawResolvedNetworkVisualization

```ts
drawResolvedNetworkVisualization(
  context: CanvasRenderingContext2D,
  resolvedNetworkVisualizationFrame: NetworkVisualizationResolvedFrame,
  hoverState: NetworkVisualizationHoverState | undefined,
  connectionLayerStyle: Partial<WeightedConnectionLayerStyle> | undefined,
): NetworkVisualizationPositionedScene
```

Draws a previously resolved network-visualization frame.

The host uses this path for hover-only repaint work because it can reuse the
cached static frame and only vary interactive emphasis.

Parameters:
- `context` - Canvas 2D drawing context.
- `resolvedNetworkVisualizationFrame` - Reusable frame cache.
- `hoverState` - Optional host-owned hover state for interactive emphasis.
- `connectionLayerStyle` - Optional override for connection stroke visibility.

Returns: Positioned node snapshot reused by host-side hover hit testing.

### formatArchitectureLabel

```ts
formatArchitectureLabel(
  architectureInputSize: number,
  hiddenLayersLabel: string,
  architectureOutputSize: number,
  totalNodeCount: number,
  totalConnectionCount: number,
  schedulingStatusLine: string | null | undefined,
): string
```

Formats the two-line architecture label used by the header and legend.

Parameters:
- `architectureInputSize` - Input layer size.
- `hiddenLayersLabel` - Hidden-layer description.
- `architectureOutputSize` - Output layer size.
- `totalNodeCount` - Total node count.
- `totalConnectionCount` - Total connection count.

Returns: Formatted architecture label.

### NetworkVisualizationResolvedFrame

Reusable resolved frame cache for hover-only network redraws.

The host can keep this resolved frame between pointer-driven redraws so it
does not recompute topology, layout, legend inputs, or connection lookups
until the network payload or canvas size changes.

### paintNetworkVisualizationCanvasBase

```ts
paintNetworkVisualizationCanvasBase(
  context: CanvasRenderingContext2D,
  networkVisualizationScene: Pick<NetworkVisualizationResolvedFrame, "canvasWidthPx" | "canvasHeightPx">,
): void
```

Paints the static background fill for the network visualization canvas.

Parameters:
- `context` - Canvas 2D drawing context.
- `networkVisualizationScene` - Frame scene context.

Returns: Nothing.

### resolveAdjustedGraphPaddingContext

```ts
resolveAdjustedGraphPaddingContext(
  context: CanvasRenderingContext2D,
  network: default | undefined,
  canvasWidthPx: number,
  hideNetworkOverlays: boolean,
  graphPaddingContext: NetworkGraphPaddingContext,
): Pick<NetworkGraphPaddingContext, "graphLeftPaddingPx" | "graphRightPaddingPx">
```

Adjusts graph-side padding to keep the floating legend from overlapping nodes.

Parameters:
- `context` - Canvas 2D drawing context.
- `network` - Network to visualize.
- `canvasWidthPx` - Canvas width.
- `hideNetworkOverlays` - Whether overlays are hidden.
- `graphPaddingContext` - Base graph padding context.

Returns: Adjusted graph padding context.

### resolveBaseGraphPaddingContext

```ts
resolveBaseGraphPaddingContext(
  hideNetworkOverlays: boolean,
  inputNodeCount: number,
  network: default | undefined,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): NetworkGraphPaddingContext
```

Resolves the base graph padding before legend-aware adjustments are applied.

Returns: Base graph padding context.

### resolveHiddenLayersLabel

```ts
resolveHiddenLayersLabel(
  hiddenLayerSizes: number[],
  architectureSource: "layer-metadata" | "graph-topology" | "inferred",
): string
```

Resolves the hidden-layer portion of the compact architecture label.

Parameters:
- `hiddenLayerSizes` - Hidden-layer sizes.
- `architectureSource` - Architecture source metadata.

Returns: Hidden-layer label.

### resolveNetworkArchitectureLabel

```ts
resolveNetworkArchitectureLabel(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): string
```

Resolves compact architecture label text for headers and HUD rows.

The label compresses the active network into a short human-readable summary:
input size, hidden-layer structure, output size, and graph size metadata.
When a runtime network is present, explicit input/output role metadata is
treated as the authoritative boundary size instead of the caller's fallback
hints so the browser panel reflects the network's current public contract.
The label can also append a compact scheduling line when the runtime exposes
a non-standard activation contract such as recurrent execution or cycle
fallback behavior.

Parameters:
- `network` - Network to describe.
- `inputSize` - Configured input size.
- `outputSize` - Configured output size.

Returns: Readable architecture label.

### resolveNetworkDrawableArea

```ts
resolveNetworkDrawableArea(
  networkVisualizationScene: NetworkVisualizationScene,
): NetworkDrawableArea
```

Resolves the drawable graph area after scene padding is applied.

Parameters:
- `networkVisualizationScene` - Frame scene context.

Returns: Drawable area dimensions.

### resolveNetworkNodeDimensionsFromTopologySummary

```ts
resolveNetworkNodeDimensionsFromTopologySummary(
  networkTopologySummary: NetworkTopologySummary,
  drawableWidthPx: number,
  drawableHeightPx: number,
): NetworkNodeDimensionsLike
```

Resolves node rectangle dimensions from topology density and drawable bounds.

Parameters:
- `networkTopologySummary` - Topology summary.
- `drawableWidthPx` - Drawable graph width.
- `drawableHeightPx` - Drawable graph height.

Returns: Node dimensions.

### resolveNetworkTopologySummary

```ts
resolveNetworkTopologySummary(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): NetworkTopologySummary
```

Resolves a reusable topology summary for layout and sizing helpers.

Parameters:
- `network` - Network to visualize.
- `inputSize` - Input-layer size.
- `outputSize` - Output-layer size.

Returns: Topology summary.

### resolveNetworkVisualizationFrame

```ts
resolveNetworkVisualizationFrame(
  context: CanvasRenderingContext2D,
  network: default | undefined,
  inputSize: number,
  outputSize: number,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): NetworkVisualizationResolvedFrame
```

Resolves a reusable network-visualization frame from the active payload.

This fold captures the expensive static work for the panel in one object so
hover-only redraws can repaint from cached layout and legend data.

Parameters:
- `context` - Canvas 2D drawing context.
- `network` - Network to visualize.
- `inputSize` - Input-layer size.
- `outputSize` - Output-layer size.

Returns: Reusable resolved frame for subsequent draw passes.

### resolveNetworkVisualizationHeightPx

```ts
resolveNetworkVisualizationHeightPx(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): number
```

Resolves responsive visualization canvas height from network shape.

Dense or deeper networks need more vertical room to stay readable, so panel
height is driven by topology rather than fixed to a single constant.

Parameters:
- `network` - Network to visualize.
- `inputSize` - Input-layer size.
- `outputSize` - Output-layer size.

Returns: Recommended height in pixels.

Example:

```ts
const recommendedHeightPx = resolveNetworkVisualizationHeightPx(network, 12, 2);
```

### resolveNetworkVisualizationScene

```ts
resolveNetworkVisualizationScene(
  context: CanvasRenderingContext2D,
  network: default | undefined,
  inputSize: number,
  outputSize: number,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): NetworkVisualizationScene
```

Resolves all non-topology canvas state needed to draw the network view.

This separates frame-scene concerns such as canvas size, overlays, and color
scales from the later graph-topology layout step.

Parameters:
- `context` - Canvas 2D drawing context.
- `network` - Network to visualize.
- `inputSize` - Input-layer size.
- `outputSize` - Output-layer size.

Returns: Scene context for the current frame.

### resolvePositionedNetworkGraphScene

```ts
resolvePositionedNetworkGraphScene(
  networkVisualizationScene: NetworkVisualizationScene,
  network: default | undefined,
  inputSize: number,
  outputSize: number,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): PositionedNetworkGraphScene
```

Resolves positioned nodes, connection lookup state, and shared node dimensions.

Parameters:
- `networkVisualizationScene` - Frame scene context.
- `network` - Network to visualize.
- `inputSize` - Input-layer size.
- `outputSize` - Output-layer size.

Returns: Positioned graph scene.

### resolveRecommendedNetworkHeightPx

```ts
resolveRecommendedNetworkHeightPx(
  networkTopologySummary: NetworkTopologySummary,
  topologyDrivenHeightPx: number,
): number
```

Resolves the recommended panel height from topology and density adjustments.

Parameters:
- `networkTopologySummary` - Topology summary.
- `topologyDrivenHeightPx` - Minimum readable topology height.

Returns: Recommended panel height.

### resolveRuntimeConnections

```ts
resolveRuntimeConnections(
  network: default | undefined,
): VisualNetworkConnectionLike[]
```

Resolves the runtime connection array from the active network.

Parameters:
- `network` - Network to visualize.

Returns: Runtime connection list.

### resolveSchedulingExecutionLabel

```ts
resolveSchedulingExecutionLabel(
  executionPath: ActivationSchedulingExecutionPath,
): string
```

Resolve a short human-readable execution label for browser architecture text.

Parameters:
- `executionPath` - Scheduling execution path reported by the runtime.

Returns: Compact browser-facing label.

### resolveSchedulingStatusLine

```ts
resolveSchedulingStatusLine(
  network: default,
): string | null
```

Resolve a compact scheduling status line for the architecture label.

The browser panel should stay quiet for the standard feed-forward contract,
but it should surface a small extra line when a network is recurrent or when
acyclic scheduling fell back because of a detected cycle.

Parameters:
- `network` - Network being visualized.

Returns: Scheduling status line or null for the normal feed-forward path.

### resolveTopologyDrivenHeightPx

```ts
resolveTopologyDrivenHeightPx(
  networkTopologySummary: NetworkTopologySummary,
): number
```

Resolves the topology-driven minimum readable height.

Parameters:
- `networkTopologySummary` - Topology summary.

Returns: Minimum readable height in pixels.

### shouldHideNetworkOverlays

```ts
shouldHideNetworkOverlays(
  context: CanvasRenderingContext2D,
): boolean
```

Determines whether responsive rules hide auxiliary network overlays.

Parameters:
- `context` - Canvas 2D drawing context.

Returns: True when overlays should be hidden.

## browser-entry/network-view/network-view.types.ts

Shared type contracts for network-view overlays.

The most notable overlays are the input-group label bands and the per-input
row descriptions. Together they turn the Flappy input shelf back into a
readable teaching surface instead of a flat strip of anonymous nodes.

### InputGroupLabelBand

Input-group label band geometry and style contract.

Each band identifies a contiguous span of input nodes and the visual style
used to render that group marker.

### InputLabelGroupDefinition

One reusable semantic input group definition for the shared network visualizer.

### InputLabelNodeDefinition

One reusable description definition before it is bound to a concrete node row.

### InputNodeDescriptionLabel

One horizontal description aligned to a specific Flappy input node.

The label sits between the semantic group band and the network itself so the
viewer can understand each observation channel without inspecting source.

## browser-entry/network-view/network-view.constants.ts

Ordered labels for grouped Flappy network input bands.

### FLAPPY_INPUT_GROUP_LABELS

Ordered labels for grouped Flappy network input bands.

## browser-entry/network-view/network-view.layout.utils.ts

Node-positioning helpers for the browser network view.

Once topology has been resolved into layers, these helpers place nodes inside
the drawable panel and then center the final graph so it feels balanced inside
the available canvas space.

### centerPositionedNodesInDrawableArea

```ts
centerPositionedNodesInDrawableArea(
  positionedNodes: PositionedNetworkNodeLike[],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensionsLike,
): PositionedNetworkNodeLike[]
```

Centers positioned nodes within the drawable graph area.

Positioning establishes relative structure first; centering then shifts the
whole graph as a block so it sits comfortably within the padded draw region.

Parameters:
- `positionedNodes` - Positioned nodes before centering.
- `leftPaddingPx` - Left graph padding.
- `topPaddingPx` - Top graph padding.
- `drawableWidthPx` - Drawable graph width.
- `drawableHeightPx` - Drawable graph height.
- `nodeLayoutPaddingPx` - Inner graph padding.
- `nodeDimensions` - Node dimensions.

Returns: Center-aligned positioned nodes.

### positionNetworkNodes

```ts
positionNetworkNodes(
  networkLayers: VisualNetworkNodeLike[][],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensionsLike,
): PositionedNetworkNodeLike[]
```

Positions network nodes into drawable canvas coordinates.

The layout keeps layer ordering stable while adapting inter-node spacing to
the amount of available vertical space.

Parameters:
- `networkLayers` - Resolved network layers.
- `leftPaddingPx` - Left graph padding.
- `topPaddingPx` - Top graph padding.
- `drawableWidthPx` - Drawable graph width.
- `drawableHeightPx` - Drawable graph height.
- `nodeLayoutPaddingPx` - Inner graph padding.
- `nodeDimensions` - Node dimensions.

Returns: Positioned nodes.

## browser-entry/network-view/network-view.topology.utils.ts

Topology resolution helpers for the browser network view.

These helpers answer a key visualization question: how should the current
network be partitioned into ordered layers so layout and architecture labels
stay meaningful even when some metadata is missing?

### NetworkHiddenColumnAnnotation

Semantic annotation for one hidden column in the browser network view.

These annotations power recurrent-role guide chips such as “input gate” or
“IN t-1”, making recurrent presets readable without flattening them into one
anonymous hidden shelf.

### NetworkVisualizationTopologyPlan

Full topology plan for the browser network view.

The plan preserves the original layer-array input used by layout helpers and
adds semantic hidden-column annotations for recurrent-aware overlays.

### resolveNetworkVisualizationLayers

```ts
resolveNetworkVisualizationLayers(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNodeLike[][]
```

Resolves layered node groups for network-view layout and rendering.

Parameters:
- `network` - Runtime network instance.
- `inputSize` - Input count fallback.
- `outputSize` - Output count fallback.

Returns: Layered nodes for rendering.

### resolveNetworkVisualizationTopologyPlan

```ts
resolveNetworkVisualizationTopologyPlan(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): NetworkVisualizationTopologyPlan
```

Resolves the full topology plan for browser layout and recurrent guides.

Parameters:
- `network` - Runtime network instance.
- `inputSize` - Input count fallback.
- `outputSize` - Output count fallback.

Returns: Layered nodes plus semantic hidden-column annotations.

## browser-entry/network-view/network-view.draw.service.ts

Overlay drawing helpers specific to the network-view panel.

These helpers render semantic guides that sit on top of the raw graph, most
notably the colored input-group bands and per-input labels that explain how
the simplified observation shelf is organized.

### alignInputNodesToDescriptionScenes

```ts
alignInputNodesToDescriptionScenes(
  positionedNodes: PositionedNetworkNodeLike[],
  inputDescriptionScenes: readonly NetworkInputDescriptionScene[],
): PositionedNetworkNodeLike[]
```

Aligns input-node centers with the resolved description chip centers.

Parameters:
- `positionedNodes` - Positioned nodes in graph coordinates.
- `inputDescriptionScenes` - Positioned input-description scenes.

Returns: Positioned nodes with input-node rows aligned to their description chips.

### drawHiddenColumnLabelScenes

```ts
drawHiddenColumnLabelScenes(
  context: CanvasRenderingContext2D,
  hiddenColumnLabelScenes: readonly NetworkHiddenColumnLabelScene[],
  hoveredNodeIndices: readonly number[] | undefined,
): void
```

Draws hidden-column guide chips for recurrent-aware layouts.

Parameters:
- `context` - Canvas 2D rendering context.
- `hiddenColumnLabelScenes` - Positioned hidden-column label scenes.
- `hoveredNodeIndices` - Optional hovered-node indices used to focus the matching column.

Returns: Nothing.

### drawInputGroupLabelBands

```ts
drawInputGroupLabelBands(
  context: CanvasRenderingContext2D,
  inputGroupLabelBandScenes: NetworkInputGroupLabelBandScene[],
  hoveredNodeIndices: readonly number[] | undefined,
): void
```

Draws vertical neon bands that label semantic groups in the input layer.

Parameters:
- `context` - Canvas 2D rendering context.
- `inputGroupLabelBandScenes` - Positioned label-band scenes.

Returns: Nothing.

### drawInputNodeDescriptions

```ts
drawInputNodeDescriptions(
  context: CanvasRenderingContext2D,
  inputDescriptionScenes: NetworkInputDescriptionScene[],
  hoveredNodeIndices: readonly number[] | undefined,
): void
```

Draws the horizontal per-input description rows.

Parameters:
- `context` - Canvas 2D rendering context.
- `inputDescriptionScenes` - Positioned input-description scenes.

Returns: Nothing.

### drawRoundedRect

```ts
drawRoundedRect(
  context: CanvasRenderingContext2D,
  leftXPx: number,
  topYPx: number,
  widthPx: number,
  heightPx: number,
  radiusPx: number,
  fillColor: string,
): void
```

Draws a filled rounded rectangle path.

This is the small geometry primitive used by the input-group band renderer.

### resolveHiddenColumnLabelScenes

```ts
resolveHiddenColumnLabelScenes(
  positionedNodes: PositionedNetworkNodeLike[],
  nodeDimensions: NetworkNodeDimensionsLike,
  hiddenColumnAnnotations: readonly NetworkHiddenColumnAnnotation[],
): NetworkHiddenColumnLabelScene[]
```

Resolves hidden-column guide scenes for recurrent-aware layouts.

Parameters:
- `positionedNodes` - Positioned nodes in graph coordinates.
- `nodeDimensions` - Resolved node dimensions.
- `hiddenColumnAnnotations` - Semantic hidden-column annotations.

Returns: Positioned hidden-column label scenes.

### resolveInputDescriptionScenes

```ts
resolveInputDescriptionScenes(
  positionedNodes: PositionedNetworkNodeLike[],
  nodeDimensions: NetworkNodeDimensionsLike,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): NetworkInputDescriptionScene[]
```

Resolves one horizontal description scene for each input node.

Parameters:
- `positionedNodes` - Positioned nodes in graph coordinates.
- `nodeDimensions` - Resolved node dimensions.

Returns: Positioned input-description scenes.

### resolveInputGroupLabelBandScenes

```ts
resolveInputGroupLabelBandScenes(
  positionedNodes: PositionedNetworkNodeLike[],
  nodeDimensions: NetworkNodeDimensionsLike,
  inputDescriptionScenes: readonly NetworkInputDescriptionScene[] | undefined,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): NetworkInputGroupLabelBandScene[]
```

Resolves vertical neon band scenes that label semantic groups in the input layer.

Resolving the bands up front lets drawing and hover hit testing reuse the
same geometry instead of maintaining duplicate layout logic.

Parameters:
- `positionedNodes` - Positioned nodes in graph coordinates.
- `nodeDimensions` - Resolved node dimensions.

Returns: Positioned label-band scenes.

## browser-entry/network-view/network-view.labels.utils.ts

Semantic input-label helpers for the network-view panel.

The Flappy controller now uses a compact current-frame shelf, but the panel
still needs to teach what each row means. These helpers recover both the
broader semantic families and the per-input descriptions.

### resolveInputDescriptionChipWidthPx

```ts
resolveInputDescriptionChipWidthPx(
  labelLines: readonly string[],
): number
```

Resolves the content-driven width of one input-description chip.

Parameters:
- `labelLines` - Human-readable label lines shown inside the chip.

Returns: Pixel width needed to render the chip without clipping.

### resolveInputDescriptionColumnWidthPx

```ts
resolveInputDescriptionColumnWidthPx(
  inputNodeCount: number,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): number
```

Resolves the maximum width required by the current input-description column.

The layout shelf should reserve enough space for the widest chip so the
semantic group bands never get pushed off the left edge of the canvas.

Parameters:
- `inputNodeCount` - Input-layer node count.

Returns: Maximum chip width needed by the current input-description column.

### resolveInputGroupLabelBands

```ts
resolveInputGroupLabelBands(
  inputNodeCount: number,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): InputGroupLabelBand[]
```

Resolves input-layer semantic label bands for the simplified Flappy inputs.

When the input size matches the current-frame layout, the view can annotate
the full input band directly beside the input layer.

Parameters:
- `inputNodeCount` - Input-layer node count.

Returns: Group label ranges with band colors.

### resolveInputNodeDescriptionLabels

```ts
resolveInputNodeDescriptionLabels(
  inputNodeCount: number,
  inputLabelGroupDefinitions: readonly InputLabelGroupDefinition[] | undefined,
): InputNodeDescriptionLabel[]
```

Resolves one short horizontal description for each simplified Flappy input.

These descriptions sit between the group bands and the network so each input
row can be read directly from the browser visualizer.

Parameters:
- `inputNodeCount` - Input-layer node count.

Returns: Ordered node descriptions for the input shelf.
