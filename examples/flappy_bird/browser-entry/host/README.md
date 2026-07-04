# browser-entry/host

Browser host assembly for the Flappy Bird demo UI.

The host boundary builds the stage that the rest of the browser runtime plays
on: title frame, simulation canvas, HUD table, and the network-inspection
panel. It is deliberately separate from runtime orchestration so DOM setup and
responsive layout stay understandable without also reading worker or playback
code.

Read this module as a browser chapter about presentation ownership:
resolve the shell, mount the panels, keep the canvases sized correctly, then
hand the runtime narrow handles for drawing and HUD updates.

## browser-entry/host/host.ts

### createCanvasHost

```ts
createCanvasHost(
  containerElement: HTMLElement,
  options: CanvasHostOptions,
): CanvasHostResult
```

Builds the browser demo host tree and returns rendering handles.

This is the public host entrypoint used by the runtime startup path.

Parameters:
- `containerElement` - Root host container.

Returns: Canvas handles, stats cells and network render callback.

### createCanvasHostInternal

```ts
createCanvasHostInternal(
  containerElement: HTMLElement,
  options: CanvasHostOptions,
): CanvasHostResult
```

Builds the browser demo host tree and returns rendering handles.

The orchestration is deliberately step-shaped: clear old DOM, build layout,
create canvases, wire resize behavior, render placeholders, then return the
handles the runtime will mutate during execution.

Parameters:
- `containerElement` - Root host container.

Returns: Canvas handles, stats cells and network render callback.

### createHeaderFrameRenderer

```ts
createHeaderFrameRenderer(
  headerCanvas: HTMLCanvasElement,
  headerContext: CanvasRenderingContext2D,
): () => void
```

Creates the reusable title-frame renderer for the header canvas.

Parameters:
- `headerCanvas` - Header canvas element.
- `headerContext` - Header canvas 2D context.

Returns: Callback that redraws the framed title.

### createHostCanvasElements

```ts
createHostCanvasElements(
  hostVisualPrimitives: HostVisualPrimitives,
): HostCanvasElements
```

Creates the canvases and 2D contexts used by the host UI.

The host manages three canvas surfaces with different jobs: a title/header
frame, the main simulation view, and the side-panel network visualization.

Parameters:
- `hostVisualPrimitives` - Shared visual primitives for border and shadow styling.

Returns: Simulation, header, and network canvases with required contexts.

### createHostLayoutElements

```ts
createHostLayoutElements(
  hostVisualPrimitives: HostVisualPrimitives,
): HostLayoutElements
```

Creates the host layout elements used to assemble the browser UI tree.

This creates the structural DOM only. Canvases, stats content, and
visualization wiring are layered on afterward.

Parameters:
- `hostVisualPrimitives` - Shared visual primitives for border and shadow styling.

Returns: Layout elements grouped by host responsibility.

### createHostNetworkVisualizationController

```ts
createHostNetworkVisualizationController(
  networkCanvasHost: HTMLDivElement,
  networkCanvas: HTMLCanvasElement,
  networkContext: CanvasRenderingContext2D,
): HostNetworkVisualizationController
```

Creates the network visualization renderer and redraw controller.

Parameters:
- `networkCanvasHost` - Host element wrapping the network canvas.
- `networkCanvas` - Network visualization canvas.
- `networkContext` - Network visualization 2D context.

Returns: Renderer and redraw callbacks for the network panel.

### installCanvasHostResizeHooks

```ts
installCanvasHostResizeHooks(
  canvas: HTMLCanvasElement,
  hostLayoutElements: HostLayoutElements,
  networkCanvas: HTMLCanvasElement,
  drawHeaderFrame: () => void,
  hostNetworkVisualizationController: HostNetworkVisualizationController,
): void
```

Installs responsive resize hooks for the simulation canvas and side panel.

Parameters:
- `canvas` - Simulation canvas.
- `hostLayoutElements` - Prepared layout containers.
- `networkCanvas` - Network visualization canvas.
- `drawHeaderFrame` - Callback that redraws the header title.
- `hostNetworkVisualizationController` - Network panel resize/redraw controller.

Returns: Nothing.

### mountCanvasHostTree

```ts
mountCanvasHostTree(
  containerElement: HTMLElement,
  hostLayoutElements: HostLayoutElements,
  headerCanvas: HTMLCanvasElement,
  canvas: HTMLCanvasElement,
  networkCanvas: HTMLCanvasElement,
  architectureSelectorElement: HTMLDivElement,
): void
```

Mounts the completed host DOM tree into the container in final order.

Parameters:
- `containerElement` - Root host container.
- `hostLayoutElements` - Prepared layout containers.
- `headerCanvas` - Header title canvas.
- `canvas` - Main simulation canvas.
- `networkCanvas` - Network visualization canvas.

Returns: Nothing.

### renderInitialCanvasHostState

```ts
renderInitialCanvasHostState(
  drawHeaderFrame: () => void,
  renderNetworkArchitecture: (network: default | undefined, inputSize: number, outputSize: number) => void,
): void
```

Renders the initial header and placeholder network visualization state.

Parameters:
- `drawHeaderFrame` - Callback that redraws the header title.
- `renderNetworkArchitecture` - Network visualization renderer.

Returns: Nothing.

### resetHostContainer

```ts
resetHostContainer(
  containerElement: HTMLElement,
): void
```

Clears any previous runtime DOM before rebuilding the browser host tree.

The demo rebuilds the host from scratch on each startup so repeated runs begin
from a known clean DOM state.

Parameters:
- `containerElement` - Root host container.

Returns: Nothing.

### resolveHostVisualPrimitives

```ts
resolveHostVisualPrimitives(): HostVisualPrimitives
```

Resolves shared border, shadow, and padding values for host assembly.

Centralizing these primitives keeps the DOM-building code focused on layout
structure instead of duplicating presentation constants everywhere.

Returns: Shared visual primitives reused across host sections.

### updateStatsTableValues

```ts
updateStatsTableValues(
  statsValueByKey: Partial<Record<FlappyStatsKey, HTMLTableCellElement>>,
  partialValues: Partial<Record<FlappyStatsKey, string>>,
): void
```

Applies partial stat updates to the rendered stats table.

The runtime writes HUD values incrementally, so the host exposes a narrow
partial-update helper rather than requiring full table redraws.

Parameters:
- `statsValueByKey` - Lookup of stat keys to value cells.
- `partialValues` - Subset of values to write this tick.

Returns: Nothing.

## browser-entry/host/host.types.ts

Public type contracts for the browser-entry host boundary.

These types describe what the host builder returns to the runtime and how HUD
value updates are represented once the UI tree exists.

### CanvasHostOptions

Browser host callback bundle used by the architecture selector control group.

### CanvasHostResult

Result payload returned after constructing the browser host UI tree.

This is the runtime's handle into the rendered browser shell: the main canvas,
its 2D context, the stats-cell lookup, and the network-panel draw callback.

### HostArchitectureSelectorController

Imperative controller returned by the host architecture selector service.

### HostArchitectureSelectorItem

Render-ready button state for one Flappy architecture profile selector item.

### HostStatsPartialValues

Partial stats update map keyed by stats-table keys.

Using a partial map lets the runtime update only the HUD fields that changed
on a given tick.

## browser-entry/host/host.constants.ts

Shared presentation constants for browser host assembly.

These values tune the spacing and responsiveness of the host HUD and side
panels without burying layout numbers inside DOM-building code.

### FLAPPY_HOST_PANEL_PADDING

Shared panel padding used by the host stats container.

This controls the interior breathing room of the main stats panel.

### FLAPPY_HOST_PANEL_TRANSITION

Shared panel max-height transition.

This supports the subtle host-panel resize behavior during responsive layout
changes.

### FLAPPY_HOST_STATS_SPLIT_GAP

Shared stats split gap.

This controls the gutter between the table column and the network panel.

### FLAPPY_HOST_TABLE_FONT_SIZE

Shared stats table font size.

The table uses a compact monospace size so many HUD rows fit comfortably in
the host panel.

### FLAPPY_HOST_TABLE_HOST_PADDING

Shared table host padding used by the stats value section.

The table host gets slightly different padding so dense stat rows remain
readable without wasting horizontal space.

## browser-entry/host/host.canvas.service.ts

Canvas sizing helpers for the browser host boundary.

These utilities keep host layout and backing-store sizing aligned so the
simulation and network canvases render crisply without stretching artifacts.

### applyCanvasBackingSize

```ts
applyCanvasBackingSize(
  canvas: HTMLCanvasElement,
  widthPx: number,
  heightPx: number,
): boolean
```

Applies a canvas backing store size and CSS width/height.

Browser canvases have both backing-store dimensions and CSS box dimensions;
this helper updates both together.

Parameters:
- `canvas` - Target canvas element.
- `widthPx` - Desired backing-store width in pixels.
- `heightPx` - Desired backing-store height in pixels.

Returns: True when canvas dimensions changed.

### applySimulationCanvasBounds

```ts
applySimulationCanvasBounds(
  canvas: HTMLCanvasElement,
  widthPx: number,
  heightPx: number,
): boolean
```

Applies fixed simulation-canvas bounds so layout does not stretch unexpectedly.

The main simulation canvas uses fixed bounds because the world renderer is
tuned for a controlled viewport rather than fluid DOM stretching.

Parameters:
- `canvas` - Simulation canvas element.
- `widthPx` - Desired width in pixels.
- `heightPx` - Desired height in pixels.

Returns: True when backing-store dimensions changed.

### resolveNetworkCanvasSizePx

```ts
resolveNetworkCanvasSizePx(
  networkCanvasHost: HTMLElement,
  hostInsetPx: number,
): { widthPx: number; heightPx: number; }
```

Computes the drawable network canvas size from host element dimensions.

The side-panel network view needs the drawable size after panel insets are
accounted for, not just the raw host client box.

Parameters:
- `networkCanvasHost` - Host element wrapping the network canvas.
- `hostInsetPx` - Total inset to subtract from both dimensions.

Returns: Width/height pair in pixels.

## browser-entry/host/host.dom.service.ts

Low-level DOM safety helpers for the browser host boundary.

Host assembly depends on several canvases. This helper turns the browser's
nullable `getContext` API into a strict contract before higher-level host
assembly begins.

### resolveRequiredCanvas2dContext

```ts
resolveRequiredCanvas2dContext(
  canvas: HTMLCanvasElement,
  errorMessage: string,
): CanvasRenderingContext2D
```

Resolves a required 2D context from a canvas element.

Failing early here keeps later rendering code free from repeated null checks.

Parameters:
- `canvas` - Target canvas element.
- `errorMessage` - Error message when 2D context is unavailable.

Returns: Canvas 2D rendering context.

## browser-entry/host/host.stats.service.ts

Stats-table creation and update helpers for the browser host HUD.

The host treats the stats table as a small indexed dashboard: build it once,
keep direct references to value cells, then apply partial text updates during
the runtime loop.

### createAndAttachHostStatsTable

```ts
createAndAttachHostStatsTable(
  statsTableHost: HTMLElement,
): Partial<Record<FlappyStatsKey, HTMLTableCellElement>>
```

Creates the host stats table, appends it into the provided host element, and
initializes all HUD values to their baseline placeholders.

The initial placeholders make the HUD legible before the first generation or
playback frame has been processed: live counters start at zero, generation
summary metrics wait on worker playback completion, and instrumentation rows
stay explicit about whether runtime telemetry is enabled.

Parameters:
- `statsTableHost` - DOM host that receives the table.

Returns: Lookup map for future incremental stat updates.

### updateStatsTableValues

```ts
updateStatsTableValues(
  statsValueByKey: Partial<Record<FlappyStatsKey, HTMLTableCellElement>>,
  partialValues: Partial<Record<FlappyStatsKey, string>>,
): void
```

Applies partial stat updates to the rendered stats table.

This keeps HUD writes cheap and explicit: only supplied keys are rewritten,
and architecture values receive their display formatting in one place.

Parameters:
- `statsValueByKey` - Lookup of stat keys to value cells.
- `partialValues` - Subset of values to write this tick.

Returns: Nothing.

## browser-entry/host/host.resize.service.ts

### installResponsiveViewportSizing

```ts
installResponsiveViewportSizing(
  canvas: HTMLCanvasElement,
  containerElement: HTMLElement,
  mainSplitContainer: HTMLElement,
  statsContainer: HTMLElement,
  statsSplitContainer: HTMLElement,
  statsTableHost: HTMLElement,
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onNetworkResize: () => void,
): void
```

Installs responsive viewport sizing for simulation and network canvases.

This is the public entrypoint used by host assembly. It captures the elements,
creates the deferred redraw policy, runs the first layout pass, and installs
ongoing resize listeners.

Parameters:
- `canvas` - Simulation canvas to resize.
- `containerElement` - Width/height source.
- `mainSplitContainer` - Main split panel host.
- `statsContainer` - Stats host element.
- `statsSplitContainer` - Stats split panel containing stats and network panes.
- `statsTableHost` - Stats table host element.
- `networkCanvas` - Network canvas.
- `networkCanvasHost` - Network host element.
- `onNetworkResize` - Callback after network resize.

Returns: Nothing.

## browser-entry/host/host.network-tooltip.service.ts

### HostCanvasPointLike

Canvas-space point used when resolving network tooltip targets.

### NetworkVisualizationTooltipScene

Tooltip scene model resolved from the hovered network overlay target.

### resolveHoveredNetworkVisualizationTooltipScene

```ts
resolveHoveredNetworkVisualizationTooltipScene(
  canvasPoint: HostCanvasPointLike,
  positionedScene: NetworkVisualizationPositionedScene,
): NetworkVisualizationTooltipScene | undefined
```

Resolves the educational tooltip scene for the current hovered network overlay target.

Input descriptions and input nodes intentionally share the same tooltip copy,
while semantic group bands resolve a broader group-level teaching tooltip.

Parameters:
- `canvasPoint` - Hover point in network-canvas coordinates.
- `positionedScene` - Rendered positioned scene reused for hover hit testing.

Returns: Tooltip scene model for the hovered overlay target.

## browser-entry/host/host.architecture-selector.service.ts

### applyArchitectureSelectorButtonPresentation

```ts
applyArchitectureSelectorButtonPresentation(
  buttonElement: HTMLButtonElement,
  selectorItem: HostArchitectureSelectorItem,
  disabled: boolean,
  hovered: boolean,
): void
```

Applies the neon-outline presentation for one architecture selector button.

Parameters:
- `buttonElement` - Target button element.
- `selectorItem` - Render-ready selector state.
- `disabled` - Whether the selector is disabled.
- `hovered` - Whether the pointer is currently hovering the button.

Returns: Nothing.

### createArchitectureSelectorItemElement

```ts
createArchitectureSelectorItemElement(
  selectorItem: HostArchitectureSelectorItem,
  disabled: boolean,
  onClick: () => void,
): HTMLDivElement
```

Creates one selector item element with hover glow and optional best-score caption.

Parameters:
- `selectorItem` - Render-ready button state.
- `disabled` - Whether the selector is temporarily disabled.
- `onClick` - Click callback invoked for fresh-run selection.

Returns: Rendered selector item wrapper.

### createArchitectureSelectorTooltipElement

```ts
createArchitectureSelectorTooltipElement(
  selectorItem: HostArchitectureSelectorItem,
): HTMLDivElement
```

Creates the neon tooltip shown above one architecture selector button.

Parameters:
- `selectorItem` - Render-ready selector item with heading and teaching copy.

Returns: Tooltip element positioned above the button.

### createHostArchitectureSelector

```ts
createHostArchitectureSelector(
  options: CanvasHostOptions,
): HostArchitectureSelectorController
```

Creates the Flappy architecture selector control group used in the browser HUD.

The selector is intentionally presentation-only. It renders the current shared
architecture profiles, exposes a narrow click callback, and lets the runtime
update local-record captions without rebuilding the whole host tree.

Parameters:
- `options` - Initial selector items and restart callback.

Returns: Imperative controller for selector item and disabled-state updates.

### createResetScoresButtonElement

```ts
createResetScoresButtonElement(
  onResetScores: () => void,
): HTMLButtonElement
```

Creates a "Reset Scores" button that clears all architecture best-score captions.

The button sits below the architecture grid and matches the selector's visual
language while using a muted amber accent to signal that it is a clearing
action rather than a selection.

Parameters:
- `onResetScores` - Callback invoked when the button is clicked.

Returns: Styled reset button element.

### resolveArchitectureSelectorBoxShadow

```ts
resolveArchitectureSelectorBoxShadow(
  selected: boolean,
  hovered: boolean,
  hoverGlowColor: string,
): string
```

Resolves the neon glow stack for the architecture selector button state.

Parameters:
- `selected` - Whether the button is the active profile.
- `hovered` - Whether the button is hovered.
- `hoverGlowColor` - Primary glow color for the current button state.

Returns: CSS box-shadow string.
