# browser-entry/host

## browser-entry/host/host.types.ts

### CanvasHostResult

Public type contracts for the browser-entry host boundary.

These types describe what the host builder returns to the runtime and how HUD
value updates are represented once the UI tree exists.

### HostStatsPartialValues

Partial stats update map keyed by stats-table keys.

Using a partial map lets the runtime update only the HUD fields that changed
on a given tick.

## browser-entry/host/host.ts

### createCanvasHost

`(containerElement: HTMLElement) => import("test/examples/flappy_bird/browser-entry/host/host.types").CanvasHostResult`

Builds the browser demo host tree and returns rendering handles.

This is the public host entrypoint used by the runtime startup path.

Parameters:
- `containerElement` - - Root host container.

Returns: Canvas handles, stats cells and network render callback.

### createCanvasHostInternal

`(containerElement: HTMLElement) => import("test/examples/flappy_bird/browser-entry/host/host.types").CanvasHostResult`

Builds the browser demo host tree and returns rendering handles.

The orchestration is deliberately step-shaped: clear old DOM, build layout,
create canvases, wire resize behavior, render placeholders, then return the
handles the runtime will mutate during execution.

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

The host manages three canvas surfaces with different jobs: a title/header
frame, the main simulation view, and the side-panel network visualization.

Parameters:
- `hostVisualPrimitives` - - Shared visual primitives for border and shadow styling.

Returns: Simulation, header, and network canvases with required contexts.

### createHostLayoutElements

`(hostVisualPrimitives: HostVisualPrimitives) => HostLayoutElements`

Creates the host layout elements used to assemble the browser UI tree.

This creates the structural DOM only. Canvases, stats content, and
visualization wiring are layered on afterward.

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

### HostVisualPrimitives

Browser host assembly for the Flappy Bird demo UI.

The host boundary is responsible for building the browser-side shell around
the simulation: framed title, main canvas, stats panel, and network
visualization panel. It does not run evolution itself; it prepares the stage
on which the runtime loop renders.

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

The demo rebuilds the host from scratch on each startup so repeated runs begin
from a known clean DOM state.

Parameters:
- `containerElement` - - Root host container.

Returns: Nothing.

### resolveHostVisualPrimitives

`() => HostVisualPrimitives`

Resolves shared border, shadow, and padding values for host assembly.

Centralizing these primitives keeps the DOM-building code focused on layout
structure instead of duplicating presentation constants everywhere.

Returns: Shared visual primitives reused across host sections.

### updateStatsTableValues

`(statsValueByKey: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>, partialValues: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, string>>) => void`

Applies partial stat updates to the rendered stats table.

The runtime writes HUD values incrementally, so the host exposes a narrow
partial-update helper rather than requiring full table redraws.

Parameters:
- `statsValueByKey` - - Lookup of stat keys to value cells.
- `partialValues` - - Subset of values to write this tick.

Returns: Nothing.

## browser-entry/host/host.constants.ts

### host.constants

Shared presentation constants for browser host assembly.

These values tune the spacing and responsiveness of the host HUD and side
panels without burying layout numbers inside DOM-building code.

### FLAPPY_HOST_PANEL_PADDING

### FLAPPY_HOST_PANEL_TRANSITION

### FLAPPY_HOST_STATS_SPLIT_GAP

### FLAPPY_HOST_TABLE_FONT_SIZE

### FLAPPY_HOST_TABLE_HOST_PADDING

## browser-entry/host/host.dom.service.ts

### host.dom.service

Low-level DOM safety helpers for the browser host boundary.

Host assembly depends on several canvases. This helper turns the browser's
nullable `getContext` API into a strict contract before higher-level host
assembly begins.

### resolveRequiredCanvas2dContext

`(canvas: HTMLCanvasElement, errorMessage: string) => CanvasRenderingContext2D`

Low-level DOM safety helpers for the browser host boundary.

Host assembly depends on several canvases. This helper turns the browser's
nullable `getContext` API into a strict contract before higher-level host
assembly begins.

## browser-entry/host/host.stats.service.ts

### createAndAttachHostStatsTable

`(statsTableHost: HTMLElement) => Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>`

Stats-table creation and update helpers for the browser host HUD.

The host treats the stats table as a small indexed dashboard: build it once,
keep direct references to value cells, then apply partial text updates during
the runtime loop.

### updateStatsTableValues

`(statsValueByKey: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>, partialValues: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, string>>) => void`

Applies partial stat updates to the rendered stats table.

This keeps HUD writes cheap and explicit: only supplied keys are rewritten,
and architecture values receive their display formatting in one place.

Parameters:
- `statsValueByKey` - - Lookup of stat keys to value cells.
- `partialValues` - - Subset of values to write this tick.

Returns: Nothing.

## browser-entry/host/host.canvas.service.ts

### host.canvas.service

Canvas sizing helpers for the browser host boundary.

These utilities keep host layout and backing-store sizing aligned so the
simulation and network canvases render crisply without stretching artifacts.

### applyCanvasBackingSize

`(canvas: HTMLCanvasElement, widthPx: number, heightPx: number) => boolean`

Canvas sizing helpers for the browser host boundary.

These utilities keep host layout and backing-store sizing aligned so the
simulation and network canvases render crisply without stretching artifacts.

### applySimulationCanvasBounds

`(canvas: HTMLCanvasElement, widthPx: number, heightPx: number) => boolean`

Applies fixed simulation-canvas bounds so layout does not stretch unexpectedly.

The main simulation canvas uses fixed bounds because the world renderer is
tuned for a controlled viewport rather than fluid DOM stretching.

Parameters:
- `canvas` - - Simulation canvas element.
- `widthPx` - - Desired width in pixels.
- `heightPx` - - Desired height in pixels.

Returns: True when backing-store dimensions changed.

### resolveNetworkCanvasSizePx

`(networkCanvasHost: HTMLElement, hostInsetPx: number) => { widthPx: number; heightPx: number; }`

Computes the drawable network canvas size from host element dimensions.

The side-panel network view needs the drawable size after panel insets are
accounted for, not just the raw host client box.

Parameters:
- `networkCanvasHost` - - Host element wrapping the network canvas.
- `hostInsetPx` - - Total inset to subtract from both dimensions.

Returns: Width/height pair in pixels.

## browser-entry/host/host.resize.service.ts

### installResponsiveViewportSizing

`(canvas: HTMLCanvasElement, containerElement: HTMLElement, mainSplitContainer: HTMLElement, statsContainer: HTMLElement, statsSplitContainer: HTMLElement, statsTableHost: HTMLElement, networkCanvas: HTMLCanvasElement, networkCanvasHost: HTMLElement, onNetworkResize: () => void) => void`

Top-level responsive sizing orchestration for the browser host.

This module wires the host resize lifecycle together: gather the relevant DOM
elements, run the initial layout pass, and keep canvas sizing synchronized with
viewport changes over time.
