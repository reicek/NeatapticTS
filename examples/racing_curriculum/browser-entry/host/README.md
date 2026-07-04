# browser-entry/host

Stable DOM handles returned by the racing browser host.

The Tier 0 host is intentionally narrow: a canvas region on the left and a
dedicated network-visualizer sidebar on the right. Runtime controls live below
the track inside the canvas region. The returned `resizeRedrawController`
keeps the right-sidebar network canvas in sync with the viewport.

## browser-entry/host/host.types.ts

### RacingHostHandle

### RacingNetworkHudNodes

Live text nodes owned by the right-sidebar network HUD strip.

The browser shell keeps these nodes up to date each frame so the panel header
shows generation/run context, the active network size, the last adaptation
change, and a short status line without repainting the canvas.

## browser-entry/host/host.ts

### createHostNetworkVisualizationController

```ts
createHostNetworkVisualizationController(
  networkCanvasHost: HTMLDivElement,
  networkCanvas: HTMLCanvasElement,
  networkContext: CanvasRenderingContext2D,
  resizeRedrawController: { resize: () => void; },
): HostNetworkVisualizationController
```

Creates the host-owned network visualization controller.

This controller keeps hover state, a resolved-frame cache, and a real redraw
function for the right-sidebar network panel. It mirrors the Flappy Bird
controller shape while staying narrow enough for the racing host.

Parameters:

- `networkCanvasHost` - Host element that sizes the network canvas.
- `networkCanvas` - Network visualization canvas.
- `networkContext` - 2D rendering context for the network canvas.
- `resizeRedrawController` - Host resize/redraw controller used for sizing.

Returns: Render and redraw callbacks for the network panel.

### createHudCell

```ts
createHudCell(
  label: string,
  valueNode: Text,
  className: string | undefined,
): HTMLDivElement
```

Creates one labeled HUD cell with a live-updating value.

Parameters:

- `label` - Small uppercase label for the cell.
- `valueNode` - Live text node whose content is mutated each frame.
- `className` - Optional extra CSS class for the value span.

Returns: Completed HUD cell element.

### createRacingHost

```ts
createRacingHost(
  containerElement: HTMLElement,
): RacingHostHandle
```

Creates the ultra-thin racing browser shell.

The host owns only stable DOM regions and viewport class toggling. Simulation
truth stays off the main thread; this file is responsible for presenting the
regions the worker-owned simulation and controller inference will later drive.

Tier 0 layout mirrors the Flappy Bird parity shape: an outer frame containing a
left canvas region and a right sidebar network-visualizer region. Runtime
controls live below the track inside the canvas region; there is no
visualizer-bottom region.

Parameters:

- `containerElement` - Root element that receives the host tree.

Returns: Stable handles for the Tier 0 browser shell.

Example:

```ts
const host = createRacingHost(document.getElementById('racing-output')!);
host.applyViewportLayout(window.innerWidth);
```

### createRacingNetworkHelpStrip

```ts
createRacingNetworkHelpStrip(): HTMLDivElement
```

Creates the static help/instruction chip strip beneath the network canvas.

These chips give first-time viewers concise cues about how to read the panel:
how to inspect nodes, what connection thickness means, and how the neon color
ramp encodes weight sign and magnitude.

Returns: Help strip element.

### createRacingNetworkHud

```ts
createRacingNetworkHud(): { hudElement: HTMLDivElement; nodes: RacingNetworkHudNodes; }
```

Creates the neon top HUD strip for the right-sidebar network panel.

The strip shows run context, network size, the last adaptation reason, and a
short status line. It lives above the network canvas and is updated through
the live text nodes returned here.

Returns: HUD element and mutable live text nodes.

### HostNetworkVisualizationController

Controller handle returned by {@link createHostNetworkVisualizationController}.

### HostNetworkVisualizationState

Internal hover/cache state kept by the host-side network visualization controller.

### HostNetworkVisualizationTooltipElements

Tooltip DOM elements owned by the host network visualization controller.

### RACING_NARROW_VIEWPORT_THRESHOLD_PX

Narrow viewport threshold used by the Tier 0 host layout seam.

## browser-entry/host/host.resize.service.ts

Racing curriculum network canvas resize service.

Mirrors the Flappy Bird responsive sizing surface so the network canvas
backing store stays in sync with the host element and viewport.

### installRacingNetworkResize

```ts
installRacingNetworkResize(
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onResize: () => void,
): { uninstall: () => void; resize: () => void; redraw: () => void; }
```

Installs racing network canvas resize listeners.

On every window `resize` event the service measures the host element and
updates the canvas backing store, then invokes `onResize` so the caller can
schedule a network redraw.

If the host element has no CSS layout yet (for example in jsdom tests), the
service falls back to `window.innerWidth * 0.25` by `window.innerHeight * 0.35`
so the canvas still has a positive backing size. The returned `redraw` handle
is intentionally a no-op; the host schedules redraws through `onResize`.

Parameters:

- `networkCanvas` - Network visualization canvas to resize.
- `networkCanvasHost` - Host element whose bounds drive the canvas size.
- `onResize` - Callback invoked after the canvas backing size changes.

Returns: Uninstall, resize, and redraw handles.

Example:

```ts
const { uninstall, resize, redraw } = installRacingNetworkResize(
  networkCanvas,
  networkCanvasHost,
  () => renderNetworkView(networkCanvas, graph, options),
);
resize();
window.addEventListener('beforeunload', uninstall);
```

## browser-entry/host/host.network-tooltip.service.ts

Racing curriculum network tooltip resolver.

Mirrors the Flappy Bird tooltip resolver so the right-sidebar network
visualizer can show educational hover annotations for input nodes, hidden
columns, and input group bands.

### HostCanvasPointLike

Canvas-space point used when resolving network tooltip targets.

### NetworkHiddenColumnLabelScene

Scene describing one hidden-column label overlay region.

### NetworkInputDescriptionScene

Scene describing one input-description overlay region.

### NetworkInputGroupLabelBandScene

Scene describing one input group label band overlay region.

### NetworkNodeLike

Minimal node shape referenced by tooltip scenes.

### NetworkVisualizationPositionedScene

Positioned scene produced by the network visualizer for hover hit testing.

### NetworkVisualizationTooltipScene

Tooltip scene model resolved from the hovered network overlay target.

### PositionedNetworkNode

Positioned node used for hit testing.

### resolveRacingNetworkTooltipScene

```ts
resolveRacingNetworkTooltipScene(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined
```

Resolves the tooltip scene for the current hovered network overlay target.

Hit-test priority, from narrowest to broadest:

1. Hidden-column node hit.
2. Input node hit.
3. Hidden-column label region.
4. Input description region.
5. Input group label band.

Input descriptions and input nodes intentionally share the same tooltip copy,
while semantic group bands resolve a broader group-level teaching tooltip.

Parameters:

- `canvasPoint` - Hover point in network-canvas coordinates.
- `positionedScene` - Rendered positioned scene reused for hover hit testing.

Returns: Tooltip scene model for the hovered overlay target, or undefined.

Example:

```ts
const scene = renderNetworkView(networkCanvas, graph, options);
const tooltip = resolveRacingNetworkTooltipScene({ xPx: 120, yPx: 80 }, scene);
if (tooltip) {
  showTooltip(
    tooltip.heading,
    tooltip.bodyParagraphs,
    tooltip.anchorCenterXPx,
    tooltip.anchorTopPx,
  );
}
```
