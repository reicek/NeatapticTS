/**
 * Browser teaching surface and lifecycle facade for the Flappy Bird example.
 *
 * This folder is where worker-owned evolution meets human-friendly inspection.
 * The browser side owns DOM setup, HUD updates, playback rendering, and network
 * visualization. The worker side owns the hot-path simulation and packed frame
 * production. `browser-entry/` exists so those responsibilities stay honest
 * instead of drifting into one blurry runtime.
 *
 * That split is the main reason this file stays intentionally small. The public
 * `start(...)` surface should feel simple to call even though the surrounding
 * system is not simple at all. A caller gets one run handle, while the folder
 * behind it fans out into runtime bootstrap, host layout, worker messaging,
 * playback rendering, and inspection views.
 *
 * Read this boundary as the browser-side answer to one practical question:
 * how do you make an evolved controller visible and interactive without moving
 * simulation authority back onto the main thread? The answer is a stable entry
 * facade plus a strict authority split between browser presentation and worker
 * execution.
 *
 * `index.html` is only the local shell that loads the published bundle and then
 * reaches this same start boundary through globals. If you want the real
 * host/runtime seam, start here rather than with the static shell.
 *
 * Read the folder in three passes. Start with this file for the public
 * lifecycle contract. Continue into `runtime/` and `worker-channel/` for the
 * bootstrap and protocol story. Finish with `host/`, `playback/`,
 * `network-view/`, and `visualization/` for the teaching surface the browser
 * renders around the worker-owned simulation.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Start["start()"]:::accent --> Runtime["runtime/\nbootstrap orchestration"]:::base
 *   Runtime --> Host["host/\nDOM and canvas shell"]:::base
 *   Runtime --> Channel["worker-channel/\nworker protocol"]:::base
 *   Channel --> Playback["playback/\npopulation rendering"]:::base
 *   Runtime --> Network["network-view + visualization/\nnetwork inspection"]:::base
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Browser["Main thread browser host"]:::accent --> Ui["canvas HUD and network view"]:::base
 *   Browser --> Handle["FlappyBirdRunHandle\nstop isRunning done"]:::base
 *   Browser --> Protocol["worker-channel\nmessage transport"]:::base
 *   Protocol --> Worker["worker-owned evolution\nand packed playback frames"]:::base
 * ```
 *
 * For background on why the boundary keeps simulation authority off the main
 * thread, see MDN,
 * [Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers),
 * which captures the browser execution model this example leans on.
 *
 * Example: start the demo and stop it from embedding code later.
 *
 * ```ts
 * import { start } from './browser-entry/browser-entry';
 *
 * const handle = await start('flappy-bird-output');
 * setTimeout(() => handle.stop(), 10_000);
 * await handle.done;
 * ```
 *
 * Example: watch the lifecycle handle while the browser host is running.
 *
 * ```ts
 * const handle = await start('flappy-bird-output');
 *
 * console.log(handle.isRunning());
 * await handle.done;
 * ```
 */
export { start } from './runtime/runtime';
export type { FlappyBirdRunHandle } from './browser-entry.types';
