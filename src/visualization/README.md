# visualization

Browser-based network visualization renderer.

This module provides the shared canvas renderer for demos and external users
who want to visualize neural networks in the browser. It accepts a
`VisualizationGraphV1` (from `exportVisualizationGraph`) and lays out/draws
the network with optional demo-specific overlay hooks.
*Main entry point:**
- `renderNetworkView(canvas, graph, options)` — render a network on a canvas
*Shared infrastructure:**
- `positionNetworkNodes()` — generic layer-based node positioning
- `centerPositionedNodesInDrawableArea()` — centering logic
- `resolveNetworkVisualizationTopologyPlan()` — topology inference

## Usage: canvas renderer (browser)

Drop-in minimal usage — no overlay config required:

```ts
import { Network, exportVisualizationGraph, renderNetworkView } from 'neataptic';

const network = Network.createMLP(2, [4], 1);
const graph = exportVisualizationGraph(network);
const canvas = document.getElementById('viz-canvas') as HTMLCanvasElement;
const frame = renderNetworkView(canvas, graph);
// frame.positionedNodes can be used for hover hit-testing
```

To inject demo-specific overlays (e.g. sensor-band labels), pass an
`overlayFactory` hook — the shared renderer stays unmodified:

```ts
const frame = renderNetworkView(canvas, graph, {
  nodeDimensions: { widthPx: 28, heightPx: 28 },
  colorScales: { weightPositive: '#00ff88', weightNegative: '#ff0088',
                 activationHot: '#ffcc00', activationCold: '#0088ff', bias: '#8800ff' },
  overlayFactory: {
    createDemoOverlayScenes: (positionedNodes, nodeDimensions) => {
      // return your custom overlay scene objects here
      return [];
    },
  },
});
```

## Usage: schema export + Graphviz DOT (external tooling)

Export a network as a portable JSON schema and convert it to a DOT diagram:

```ts
import { Network, exportVisualizationGraph, toDot } from 'neataptic';

const network = Network.createMLP(3, [5], 2);
// Full export (weights + biases included by default).
const graph = exportVisualizationGraph(network);

// Lightweight export — omit weights and biases for a large population.
const compact = exportVisualizationGraph(network, {
  includeWeights: false,
  includeBiases: false,
});

// Convert to Graphviz DOT and paste into https://dreampuf.github.io/GraphvizOnline/
const dot = toDot(graph);
console.log(dot);
```

The two paths share one data contract (`VisualizationGraphV1`) so a single
`exportVisualizationGraph` call can feed both a canvas renderer and an
external tool in the same session.

## visualization/visualization.ts

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

### EdgePadding

Per-edge pixel padding that defines the inset drawable area inside the canvas, providing margins for graph layout and label overflow.

### NetworkLayerAnnotation

Semantic annotation for one layer of nodes.

For example, recurrent networks can label hidden columns as "input gate",
"hidden state t-1", etc. Feed-forward networks might have generic labels.

### NetworkNodeDimensions

Pixel dimensions shared by every node in a single rendering pass, controlling both the visual node size and the hit-test bounding box for hover interactions.

### NetworkVisualizationColorScales

Color palette strings used to encode positive and negative weights, hot and cold activations, and bias magnitudes during canvas rendering passes.

### NetworkVisualizationResolvedFrame

Complete resolved frame for hover-driven incremental redraws.

The host can cache this between pointer events so it only recomputes
topology, layout, and legend when the network payload changes.

### NetworkVisualizationTopologyPlan

Full topology plan for layout and rendering.

Preserves the layer-array input used by layout helpers and adds optional
semantic annotations for overlays.

### OverlayFactoryHooks

Optional hook functions that demos can use to inject custom overlays.

Flappy Bird injects input-group label bands and per-input descriptions.
ASCII Maze could inject custom layer labels, or leave hooks undefined.

### PositionedNetworkNode

A network node with its center position and pixel dimensions resolved in canvas space, ready for hit-testing and rendering passes.

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

### RenderNetworkViewOptions

Configuration bag passed to the shared canvas renderer to override default node dimensions, padding, color scales, and optional demo overlay hooks.

### resolveNetworkVisualizationLayers

```ts
resolveNetworkVisualizationLayers(
  network: default | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNode[][]
```

Resolve ordered layered node groups from the topology plan, used by canvas layout and topology-aware rendering helpers.

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

### VisualNetworkConnection

A visual edge between two positioned network nodes, carrying the synapse weight and enabled state for color-coded rendering.

### VisualNetworkNode

Minimal node representation used as input to the layout engine, carrying only the index, type role, and bias value needed for positioning.
