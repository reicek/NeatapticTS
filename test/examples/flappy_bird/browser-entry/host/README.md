# browser-entry/host

## browser-entry/host/host.types.ts

### CanvasHostResult

Result payload returned after constructing the browser host UI tree.

### HostStatsPartialValues

Partial stats update map keyed by stats-table keys.

## browser-entry/host/host.ts

### createCanvasHost

`(containerElement: HTMLElement) => import("test/examples/flappy_bird/browser-entry/host/host.types").CanvasHostResult`

Builds the browser demo host tree and returns rendering handles.

Parameters:

- `containerElement` - - Root host container.

Returns: Canvas handles, stats cells and network render callback.

### createCanvasHostInternal

`(containerElement: HTMLElement) => import("test/examples/flappy_bird/browser-entry/host/host.types").CanvasHostResult`

Builds the browser demo host tree and returns rendering handles.

Parameters:

- `containerElement` - - Root host container.

Returns: Canvas handles, stats cells and network render callback.

### createHeaderFrameRenderer

`(headerCanvas: HTMLCanvasElement, headerContext: CanvasRenderingContext2D) => () => void`

Creates the reusable title-frame renderer for the header canvas.

Parameters:

- `headerCanvas` - - Header canvas element.
- `headerContext` - - Header canvas 2D context.

Returns: Callback that redraws the framed title.

### createHostCanvasElements

`(hostVisualPrimitives: HostVisualPrimitives) => HostCanvasElements`

Creates the canvases and 2D contexts used by the host UI.

Parameters:

- `hostVisualPrimitives` - - Shared visual primitives for border and shadow styling.

Returns: Simulation, header, and network canvases with required contexts.

### createHostLayoutElements

`(hostVisualPrimitives: HostVisualPrimitives) => HostLayoutElements`

Creates the host layout elements used to assemble the browser UI tree.

Parameters:

- `hostVisualPrimitives` - - Shared visual primitives for border and shadow styling.

Returns: Layout elements grouped by host responsibility.

### createHostNetworkVisualizationController

`(networkCanvasHost: HTMLDivElement, networkCanvas: HTMLCanvasElement, networkContext: CanvasRenderingContext2D) => HostNetworkVisualizationController`

Creates the network visualization renderer and redraw controller.

Parameters:

- `networkCanvasHost` - - Host element wrapping the network canvas.
- `networkCanvas` - - Network visualization canvas.
- `networkContext` - - Network visualization 2D context.

Returns: Renderer and redraw callbacks for the network panel.

### installCanvasHostResizeHooks

`(canvas: HTMLCanvasElement, hostLayoutElements: HostLayoutElements, networkCanvas: HTMLCanvasElement, drawHeaderFrame: () => void, hostNetworkVisualizationController: HostNetworkVisualizationController) => void`

Installs responsive resize hooks for the simulation canvas and side panel.

Parameters:

- `canvas` - - Simulation canvas.
- `hostLayoutElements` - - Prepared layout containers.
- `networkCanvas` - - Network visualization canvas.
- `drawHeaderFrame` - - Callback that redraws the header title.
- `hostNetworkVisualizationController` - - Network panel resize/redraw controller.

Returns: Nothing.

### mountCanvasHostTree

`(containerElement: HTMLElement, hostLayoutElements: HostLayoutElements, headerCanvas: HTMLCanvasElement, canvas: HTMLCanvasElement, networkCanvas: HTMLCanvasElement) => void`

Mounts the completed host DOM tree into the container in final order.

Parameters:

- `containerElement` - - Root host container.
- `hostLayoutElements` - - Prepared layout containers.
- `headerCanvas` - - Header title canvas.
- `canvas` - - Main simulation canvas.
- `networkCanvas` - - Network visualization canvas.

Returns: Nothing.

### renderInitialCanvasHostState

`(drawHeaderFrame: () => void, renderNetworkArchitecture: (network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => void) => void`

Renders the initial header and placeholder network visualization state.

Parameters:

- `drawHeaderFrame` - - Callback that redraws the header title.
- `renderNetworkArchitecture` - - Network visualization renderer.

Returns: Nothing.

### resetHostContainer

`(containerElement: HTMLElement) => void`

Clears any previous runtime DOM before rebuilding the browser host tree.

Parameters:

- `containerElement` - - Root host container.

Returns: Nothing.

### resolveHostVisualPrimitives

`() => HostVisualPrimitives`

Resolves shared border, shadow, and padding values for host assembly.

Returns: Shared visual primitives reused across host sections.

### updateStatsTableValues

`(statsValueByKey: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>, partialValues: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, string>>) => void`

Applies partial stat updates to the rendered stats table.

Parameters:

- `statsValueByKey` - - Lookup of stat keys to value cells.
- `partialValues` - - Subset of values to write this tick.

Returns: Nothing.

## browser-entry/host/host.constants.ts

### host.constants

Shared panel padding used by the host stats container.

### FLAPPY_HOST_PANEL_PADDING

### FLAPPY_HOST_PANEL_TRANSITION

### FLAPPY_HOST_STATS_SPLIT_GAP

### FLAPPY_HOST_TABLE_FONT_SIZE

### FLAPPY_HOST_TABLE_HOST_PADDING

## browser-entry/host/host.dom.service.ts

### host.dom.service

Resolves a required 2D context from a canvas element.

@param canvas - Target canvas element.
@param errorMessage - Error message when 2D context is unavailable.
@returns Canvas 2D rendering context.

### resolveRequiredCanvas2dContext

`(canvas: HTMLCanvasElement, errorMessage: string) => CanvasRenderingContext2D`

Resolves a required 2D context from a canvas element.

Parameters:

- `canvas` - - Target canvas element.
- `errorMessage` - - Error message when 2D context is unavailable.

Returns: Canvas 2D rendering context.

## browser-entry/host/host.stats.service.ts

### createAndAttachHostStatsTable

`(statsTableHost: HTMLElement) => Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>`

Creates the host stats table, appends it into the provided host element, and
initializes all HUD values to their baseline placeholders.

Parameters:

- `statsTableHost` - - DOM host that receives the table.

Returns: Lookup map for future incremental stat updates.

### updateStatsTableValues

`(statsValueByKey: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>, partialValues: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, string>>) => void`

Applies partial stat updates to the rendered stats table.

Parameters:

- `statsValueByKey` - - Lookup of stat keys to value cells.
- `partialValues` - - Subset of values to write this tick.

Returns: Nothing.

## browser-entry/host/host.canvas.service.ts

### host.canvas.service

Applies a canvas backing store size and CSS width/height.

@param canvas - Target canvas element.
@param widthPx - Desired backing-store width in pixels.
@param heightPx - Desired backing-store height in pixels.
@returns True when canvas dimensions changed.

### applyCanvasBackingSize

`(canvas: HTMLCanvasElement, widthPx: number, heightPx: number) => boolean`

Applies a canvas backing store size and CSS width/height.

Parameters:

- `canvas` - - Target canvas element.
- `widthPx` - - Desired backing-store width in pixels.
- `heightPx` - - Desired backing-store height in pixels.

Returns: True when canvas dimensions changed.

### applySimulationCanvasBounds

`(canvas: HTMLCanvasElement, widthPx: number, heightPx: number) => boolean`

Applies fixed simulation-canvas bounds so layout does not stretch unexpectedly.

Parameters:

- `canvas` - - Simulation canvas element.
- `widthPx` - - Desired width in pixels.
- `heightPx` - - Desired height in pixels.

Returns: True when backing-store dimensions changed.

### resolveNetworkCanvasSizePx

`(networkCanvasHost: HTMLElement, hostInsetPx: number) => { widthPx: number; heightPx: number; }`

Computes the drawable network canvas size from host element dimensions.

Parameters:

- `networkCanvasHost` - - Host element wrapping the network canvas.
- `hostInsetPx` - - Total inset to subtract from both dimensions.

Returns: Width/height pair in pixels.

## browser-entry/host/host.resize.service.ts

### installResponsiveViewportSizing

`(canvas: HTMLCanvasElement, containerElement: HTMLElement, mainSplitContainer: HTMLElement, statsContainer: HTMLElement, statsSplitContainer: HTMLElement, statsTableHost: HTMLElement, networkCanvas: HTMLCanvasElement, networkCanvasHost: HTMLElement, onNetworkResize: () => void) => void`

Installs responsive viewport sizing for simulation and network canvases.

Parameters:

- `canvas` - - Simulation canvas to resize.
- `containerElement` - - Width/height source.
- `mainSplitContainer` - - Main split panel host.
- `statsContainer` - - Stats host element.
- `statsSplitContainer` - - Stats split panel containing stats and network panes.
- `statsTableHost` - - Stats table host element.
- `networkCanvas` - - Network canvas.
- `networkCanvasHost` - - Network host element.
- `onNetworkResize` - - Callback after network resize.

Returns: Nothing.
