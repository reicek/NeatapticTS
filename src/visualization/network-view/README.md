# visualization/network-view

Shared canvas renderer for browser network visualization.

This module provides the main public API: `renderNetworkView(canvas, graph, options)`.
It orchestrates the layout, topology inference, and drawing steps, allowing demos
to inject custom overlays via optional hooks while keeping the core renderer
demo-agnostic.

## visualization/network-view/network-view.types.ts

Shared type contracts for the browser-based network canvas renderer.

These types define the generic layout, positioning, and scene contracts
that any demo or external user can implement. Demo-specific overlays
(Flappy input bands, ASCII Maze labels, etc.) are injected as optional
callback hooks rather than baked into this core layer.

### EdgePadding

Padding on all four edges.

### NetworkNodeDimensions

Dimensions shared across all nodes in a rendering pass.

### NetworkVisualizationColorScales

Color scale for weight and activation visualization.

### NetworkVisualizationResolvedFrame

Complete resolved frame for hover-driven incremental redraws.

The host can cache this between pointer events so it only recomputes
topology, layout, and legend when the network payload changes.

### OverlayFactoryHooks

Optional hook functions that demos can use to inject custom overlays.

Flappy Bird injects input-group label bands and per-input descriptions.
ASCII Maze could inject custom layer labels, or leave hooks undefined.

### PositionedNetworkNode

A node with its position and dimensions resolved in canvas coordinates.

### RenderNetworkViewOptions

Options passed to the shared renderer.

### VisualNetworkConnection

A visual connection between two positioned nodes.

## visualization/network-view/network-view.ts

### drawConnections

```ts
drawConnections(
  context: CanvasRenderingContext2D,
  frame: NetworkVisualizationResolvedFrame,
): void
```

Draws weighted connections between nodes.

Parameters:
- `context` - Canvas 2D context.
- `frame` - Resolved frame.

### drawNetworkVisualization

```ts
drawNetworkVisualization(
  context: CanvasRenderingContext2D,
  frame: NetworkVisualizationResolvedFrame,
): void
```

Draws the network visualization on a canvas context.

Parameters:
- `context` - Canvas 2D context.
- `frame` - Resolved visualization frame.

### drawNodes

```ts
drawNodes(
  context: CanvasRenderingContext2D,
  frame: NetworkVisualizationResolvedFrame,
): void
```

Draws nodes with type-specific shapes and styling.

Parameters:
- `context` - Canvas 2D context.
- `frame` - Resolved frame.

### mapGraphToNetworkLayers

```ts
mapGraphToNetworkLayers(
  graph: VisualizationGraphV1,
): VisualNetworkNode[][]
```

Converts a VisualizationGraphV1 into network layers for layout.

Parameters:
- `graph` - Visualization graph.

Returns: Layered nodes (input, hidden, output).

### renderNetworkView

```ts
renderNetworkView(
  canvas: HTMLCanvasElement,
  graph: VisualizationGraphV1,
  options: RenderNetworkViewOptions | undefined,
): NetworkVisualizationResolvedFrame
```

Renders a network visualization onto a canvas.

This is the main public entry point. It accepts a `VisualizationGraphV1` (from
`exportVisualizationGraph`), lays out the nodes, and draws them with optional
demo-specific overlays.

**Typical usage:**
```ts
const graph = exportVisualizationGraph(network);
const canvas = document.getElementById('network-canvas') as HTMLCanvasElement;
const frame = renderNetworkView(canvas, graph, {
  nodeDimensions: { widthPx: 32, heightPx: 32 },
  overlayFactory: { createDemoOverlayScenes: myCustomOverlays },
});
// frame contains positioned nodes for hover hit testing
```

Parameters:
- `canvas` - Canvas element to render onto.
- `graph` - Visualization graph (from `exportVisualizationGraph`).
- `options` - Optional render settings (dimensions, padding, colors, overlays).

Returns: Resolved frame with positioned nodes and scene state (reusable for hover).

## visualization/network-view/network-view.layout.utils.ts

Generic layout helpers for browser network visualization.

Once topology has been resolved into layers, these helpers place nodes
in canvas coordinates and center the final graph within available space.

### centerPositionedNodesInDrawableArea

```ts
centerPositionedNodesInDrawableArea(
  positionedNodes: PositionedNetworkNode[],
  drawableWidthPx: number,
  drawableLeftPx: number,
): PositionedNetworkNode[]
```

Centers positioned nodes horizontally within the drawable area.

Shifts all node x-coordinates so the leftmost and rightmost nodes
are balanced around the center of available space.

Parameters:
- `positionedNodes` - Positioned nodes.
- `drawableWidthPx` - Drawable width.

Returns: Centered positioned nodes.

### positionNetworkNodes

```ts
positionNetworkNodes(
  networkLayers: VisualNetworkNode[][],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensions,
  inputLayerTargetGapPx: number,
): PositionedNetworkNode[]
```

Positions network nodes into drawable canvas coordinates.

The layout preserves layer ordering while adapting inter-node spacing to
available vertical space. Nodes in earlier layers are placed left; nodes
in later layers are placed right.

Parameters:
- `networkLayers` - Resolved network layers (each layer is a list of nodes).
- `leftPaddingPx` - Left graph padding.
- `topPaddingPx` - Top graph padding.
- `drawableWidthPx` - Drawable graph width (canvas width minus horizontal padding).
- `drawableHeightPx` - Drawable graph height (canvas height minus vertical padding).
- `nodeLayoutPaddingPx` - Inner graph padding around nodes.
- `nodeDimensions` - Node dimensions (width × height px).
- `inputLayerTargetGapPx` - Optional target gap between input nodes (default: 0).

Returns: Positioned nodes.

### VisualNetworkNode

A simple node representation for layout input.

## visualization/network-view/network-view.topology.utils.ts

Generic topology inference helpers for browser network visualization.

These helpers answer: how should the network be partitioned into ordered
layers so layout and semantic labels stay meaningful even when metadata is missing?

### NetworkLayerAnnotation

Semantic annotation for one layer of nodes.

For example, recurrent networks can label hidden columns as "input gate",
"hidden state t-1", etc. Feed-forward networks might have generic labels.

### NetworkVisualizationTopologyPlan

Full topology plan for layout and rendering.

Preserves the layer-array input used by layout helpers and adds optional
semantic annotations for overlays.

### resolveNetworkVisualizationLayers

```ts
resolveNetworkVisualizationLayers(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNode[][]
```

Resolves layered node groups for network layout.

Parameters:
- `network` - Runtime network instance (or undefined for fallback).
- `inputSize` - Input count (used if network is undefined).
- `outputSize` - Output count (used if network is undefined).

Returns: Layered nodes for rendering.

### resolveNetworkVisualizationTopologyPlan

```ts
resolveNetworkVisualizationTopologyPlan(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): NetworkVisualizationTopologyPlan
```

Resolves the full topology plan including optional layer annotations.

For recurrent networks, this detects temporal modules and creates annotations.
For feed-forward networks, this creates a simple acyclic plan.

Parameters:
- `network` - Runtime network instance (or undefined for fallback).
- `inputSize` - Input count (used if network is undefined).
- `outputSize` - Output count (used if network is undefined).

Returns: Layered nodes plus semantic layer annotations.
