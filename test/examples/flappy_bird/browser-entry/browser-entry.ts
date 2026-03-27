/**
 * Browser demo public entry for the Flappy Bird example.
 *
 * This boundary is the hand-off point between library consumers and the demo's
 * browser runtime. Calling `start` boots the canvas UI, worker channel, HUD,
 * and playback loop without exposing all of that internal orchestration as part
 * of the public API.
 *
 * In other words, this file is intentionally tiny because it acts like a clean
 * facade over a much larger interactive system.
 *
 * Educational note:
 * `index.html` is only the local browser shell that loads the published bundle
 * and then exposes this same start boundary through browser globals. Read this
 * file when you want the real host/runtime seam instead of the static shell.
 *
 * Minimal browser start example:
 * ```ts
 * import { start } from './browser-entry/browser-entry';
 *
 * const handle = await start('flappy-bird-output');
 * // Later: handle.stop();
 * await handle.done;
 * ```
 *
 * Startup path:
 * ```mermaid
 * flowchart LR
 *     Start["start()"] --> Runtime["runtime/\nbootstrap orchestration"]
 *     Runtime --> Host["host/\nDOM and canvas shell"]
 *     Runtime --> Channel["worker-channel/\nworker protocol"]
 *     Channel --> Playback["playback/\npopulation rendering"]
 *     Runtime --> Network["network-view + visualization/\nnetwork inspection"]
 * ```
 */
export { start } from './runtime/runtime';
export type { FlappyBirdRunHandle } from './browser-entry.types';
