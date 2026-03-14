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
 * Minimal browser start example:
 * ```ts
 * import { start } from './browser-entry/browser-entry';
 *
 * const handle = start();
 * // Later: handle.stop();
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
