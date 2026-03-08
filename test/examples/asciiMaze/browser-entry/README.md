# browser-entry

## browser-entry/browser-entry.types.ts

### AsciiMazeRunHandle

Public lifecycle handle returned by the browser demo entrypoint.

### BrowserEntryCurriculumContext

State and callbacks used by the curriculum runtime service.

### BrowserEntryEvolutionSettings

Evolution settings used for a single procedural maze phase.

### BrowserEntryHostElements

Resolved host elements used by logger, dashboard, and resize services.

### BrowserEntryHostServices

Browser host services assembled for one running demo instance.

### BrowserEntryStartFunction

`(container: string | HTMLElement | undefined, opts: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryStartOptions | undefined) => Promise<import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").AsciiMazeRunHandle>`

Stable callable shape used by globals compatibility wiring.

### BrowserEntryStartOptions

Options accepted by the browser-hosted ASCII Maze entrypoint.

### BrowserEntryTelemetryHub

Lightweight telemetry hub contract shared between host and public handle.

### RuntimeAbortSignal

Runtime AbortSignal shape used for older or polyfilled environments.

### RuntimeAbortSignalConstructor

AbortSignal constructor shape with optional static composition helpers.

### RuntimeDashboard

Runtime dashboard surface used by the browser entry host adapter.

### RuntimeEvolutionResult

Runtime evolution result shape used by the browser curriculum adapter.

### RuntimeWindow

Global namespace exposed for direct browser-script loading compatibility.

## browser-entry/browser-entry.ts

### browser-entry

Public browser entry facade for the ASCII Maze demo module boundary.

The folder now owns host bootstrap, runtime orchestration, globals
compatibility, and resize handling behind focused helpers. This facade keeps
the public API stable while presenting a small orchestration-first surface.

### AsciiMazeRunHandle

Public lifecycle handle returned by the browser demo entrypoint.

### BrowserEntryStartFunction

`(container: string | HTMLElement | undefined, opts: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryStartOptions | undefined) => Promise<import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").AsciiMazeRunHandle>`

Stable callable shape used by globals compatibility wiring.

### BrowserEntryStartOptions

Options accepted by the browser-hosted ASCII Maze entrypoint.

### start

`(container: string | HTMLElement, opts: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryStartOptions) => Promise<import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").AsciiMazeRunHandle>`

## browser-entry/browser-entry.services.ts

### browser-entry.services

Compatibility facade for the dedicated browser-entry service modules.

The concrete implementations now live in focused files so callers can keep
importing from this stable boundary while internals evolve independently.

### composeBrowserEntryAbortSignal

`(internalController: AbortController, externalSignal: AbortSignal | undefined) => AbortSignal`

### createBrowserEntryEvolutionHostAdapter

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionHostAdapter`

### createBrowserEntryHostServices

`(hostElements: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryHostElements) => import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryHostServices`

### installBrowserEntryGlobals

`(start: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryStartFunction) => void`

### runBrowserEntryCurriculum

`(context: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryCurriculumContext) => void`

## browser-entry/browser-entry.constants.ts

### browser-entry.constants

Shared constants for the browser-hosted ASCII Maze demo lifecycle.

These values keep the browser entry facade declarative while the host,
resize, and curriculum services consume a single named configuration table.

### BROWSER_ENTRY_CONSTANTS

## browser-entry/browser-entry.host.services.ts

### createBrowserEntryHostServices

`(hostElements: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryHostElements) => import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryHostServices`

### createTelemetryHub

`() => import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryTelemetryHub<TTelemetry>`

Create a minimal telemetry hub backed by a Set of listeners.

Returns: A small hub optimized for browser demo listener counts.

### installResizeRedraw

`(observeTarget: HTMLElement | null, runtimeDashboard: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").RuntimeDashboard) => () => void`

Attach dashboard redraw behavior to host resizes and return a cleanup function.

Parameters:
- `observeTarget` - - Element whose width should trigger redraw checks.
- `runtimeDashboard` - - Runtime dashboard adapter with redraw support.

Returns: Cleanup function that removes active observers or listeners.

### safelyRedrawDashboard

`(runtimeDashboard: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").RuntimeDashboard) => void`

Safely request a dashboard redraw without letting host issues break the run.

Parameters:
- `runtimeDashboard` - - Runtime dashboard adapter with optional redraw support.

## browser-entry/browser-entry.abort.services.ts

### composeBrowserEntryAbortSignal

`(internalController: AbortController, externalSignal: AbortSignal | undefined) => AbortSignal`

## browser-entry/browser-entry.globals.services.ts

### createBrowserEntryEvolutionHostAdapter

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionHostAdapter`

### installBrowserEntryGlobals

`(start: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryStartFunction) => void`

## browser-entry/browser-entry.curriculum.services.ts

### refineBrowserEntryBestNetwork

`(bestNetwork: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork | undefined, previousBestNetwork: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork | undefined) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork | undefined`

Refine the winning network before seeding the next curriculum phase.

Parameters:
- `bestNetwork` - - Network returned by the latest evolution phase.
- `previousBestNetwork` - - Previously carried curriculum seed.

Returns: Refined winner or the best available carry-over network.

### runBrowserEntryCurriculum

`(context: import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryCurriculumContext) => void`

## browser-entry/browser-entry.utils.ts

### createBrowserEvolutionSettings

`(dimension: number) => import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryEvolutionSettings`

### didSolveBrowserMaze

`(progress: unknown) => boolean`

### getNextBrowserMazeDimension

`(currentDimension: number) => number`

### resolveBrowserEntryHostElements

`(container: string | HTMLElement) => import("C:/NeatapticTS/test/examples/asciiMaze/browser-entry/browser-entry.types").BrowserEntryHostElements`

### scheduleBrowserEntryFrame

`(callback: () => void) => void`
