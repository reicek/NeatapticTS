# browser-entry/host/resize

## browser-entry/host/resize/host.resize.service.types.ts

### host.resize.service.types

DOM elements that participate in responsive host sizing.

### DeferredNetworkRedrawController

Deferred redraw controller used after layout-affecting changes.

### ResponsiveViewportLayoutContext

Full responsive layout context shared across sizing helpers.

### ResponsiveViewportLayoutFlags

Layout-mode flags resolved from the current viewport.

### ResponsiveViewportMeasurements

Measured viewport and panel budgets for responsive layout.

### ResponsiveViewportSizingElements

DOM elements that participate in responsive host sizing.

### SimulationCanvasBounds

Width and height bounds for the simulation canvas.

### StatsPanelDimensions

Resolved height and width budgets for the stats panel.

## browser-entry/host/resize/host.resize.service.ts

### applyResponsiveViewportSizing

`(responsiveViewportSizingElements: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportSizingElements, deferredNetworkRedrawController: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").DeferredNetworkRedrawController, onNetworkResize: () => void) => void`

Applies responsive sizing to the simulation and network canvases.

Parameters:
- `responsiveViewportSizingElements` - - Host elements participating in layout.
- `deferredNetworkRedrawController` - - Deferred redraw controller.
- `onNetworkResize` - - Immediate network resize callback.

Returns: Nothing.

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

## browser-entry/host/resize/host.resize.service.services.ts

### applyMinimalMobileViewportLayout

`(responsiveViewportSizingElements: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportSizingElements, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext) => void`

Applies the minimal mobile layout that hides the auxiliary panes.

Parameters:
- `responsiveViewportSizingElements` - - Host elements participating in layout.
- `responsiveViewportLayoutContext` - - Responsive layout context.

Returns: Nothing.

### applyNetworkCanvasHostHeight

`(networkCanvasHost: HTMLElement, deferredNetworkRedrawController: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").DeferredNetworkRedrawController) => void`

Applies the fixed network host height and queues redraw when it changes.

Parameters:
- `networkCanvasHost` - - Network canvas host element.
- `deferredNetworkRedrawController` - - Deferred redraw controller.

Returns: Nothing.

### applyResponsiveCanvasBounds

`(responsiveViewportSizingElements: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportSizingElements, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext, statsPanelDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").StatsPanelDimensions, onNetworkResize: () => void) => void`

Applies simulation and network canvas backing sizes for the active layout.

Parameters:
- `responsiveViewportSizingElements` - - Host elements participating in layout.
- `responsiveViewportLayoutContext` - - Responsive layout context.
- `statsPanelDimensions` - - Resolved stats panel dimensions.
- `onNetworkResize` - - Immediate network resize callback.

Returns: Nothing.

### applySplitContainerLayoutStyles

`(responsiveViewportSizingElements: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportSizingElements, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext) => void`

Applies the split-container styles for the standard layout modes.

Parameters:
- `responsiveViewportSizingElements` - - Host elements participating in layout.
- `responsiveViewportLayoutContext` - - Responsive layout context.

Returns: Nothing.

### applyStandardViewportLayout

`(responsiveViewportSizingElements: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportSizingElements, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext, deferredNetworkRedrawController: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").DeferredNetworkRedrawController) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").StatsPanelDimensions`

Applies the standard tablet and desktop layout and returns panel dimensions.

Parameters:
- `responsiveViewportSizingElements` - - Host elements participating in layout.
- `responsiveViewportLayoutContext` - - Responsive layout context.
- `deferredNetworkRedrawController` - - Deferred redraw controller.

Returns: Resolved stats panel dimensions.

### applyStatsContainerDimensions

`(statsContainer: HTMLElement, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext, statsPanelDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").StatsPanelDimensions) => void`

Applies the resolved dimensions and scrolling rules to the stats container.

Parameters:
- `statsContainer` - - Stats host element.
- `responsiveViewportLayoutContext` - - Responsive layout context.
- `statsPanelDimensions` - - Resolved stats panel dimensions.

Returns: Nothing.

### applyStatsPaneOrdering

`(statsTableHost: HTMLElement, networkCanvasHost: HTMLElement, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext) => void`

Applies ordering and flex styles for stats and network panes.

Parameters:
- `statsTableHost` - - Stats table host element.
- `networkCanvasHost` - - Network canvas host element.
- `responsiveViewportLayoutContext` - - Responsive layout context.

Returns: Nothing.

### createDeferredNetworkRedrawController

`(onNetworkResize: () => void) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").DeferredNetworkRedrawController`

Creates a deferred redraw controller that waits for layout to settle.

Parameters:
- `onNetworkResize` - - Callback after network resize.

Returns: Deferred redraw controller.

### installResponsiveViewportSizingListeners

`(containerElement: HTMLElement, applyCanvasSize: () => void, deferredNetworkRedrawController: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").DeferredNetworkRedrawController) => void`

Installs window and container listeners for responsive host sizing.

Parameters:
- `containerElement` - - Width and height source.
- `applyCanvasSize` - - Shared sizing callback.
- `deferredNetworkRedrawController` - - Deferred redraw controller.

Returns: Nothing.

## browser-entry/host/resize/host.resize.service.constants.ts

### host.resize.service.constants

Shared CSS tokens used by the host resize layout appliers.

### FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX

### FLAPPY_HOST_RESIZE_SPLIT_RATIO

### FLAPPY_HOST_RESIZE_STYLE_TOKENS

## browser-entry/host/resize/host.resize.service.utils.ts

### resolveHeaderHeightPx

`(containerElement: HTMLElement) => number`

Resolves the header canvas height, if present.

Parameters:
- `containerElement` - - Width and height source.

Returns: Header height in pixels.

### resolveMinimalMobileCanvasBounds

`(containerElement: HTMLElement, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").SimulationCanvasBounds`

Resolves the simulation canvas bounds for the minimal mobile layout.

Parameters:
- `containerElement` - - Width and height source.
- `responsiveViewportLayoutContext` - - Responsive layout context.

Returns: Simulation canvas bounds.

### resolveResponsiveViewportLayoutContext

`(containerElement: HTMLElement, statsContainer: HTMLElement, networkCanvasHost: HTMLElement) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext`

Resolves responsive layout measurements and mode flags from the host DOM.

Parameters:
- `containerElement` - - Width and height source.
- `statsContainer` - - Stats host element.
- `networkCanvasHost` - - Network host element.

Returns: Responsive layout context.

### resolveSimulationCanvasBounds

`(containerElement: HTMLElement, responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext, statsPanelDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").StatsPanelDimensions) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").SimulationCanvasBounds`

Resolves simulation canvas bounds from the current viewport layout.

Parameters:
- `containerElement` - - Width and height source.
- `responsiveViewportLayoutContext` - - Responsive layout context.
- `statsPanelDimensions` - - Resolved stats panel dimensions.

Returns: Simulation canvas bounds.

### resolveStatsPanelDimensions

`(responsiveViewportLayoutContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").ResponsiveViewportLayoutContext) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/resize/host.resize.service.types").StatsPanelDimensions`

Resolves the stats panel height and width budgets.

Parameters:
- `responsiveViewportLayoutContext` - - Responsive layout context.

Returns: Stats panel dimensions.
