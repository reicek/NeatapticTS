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
 */
export { start } from './runtime/runtime';
export type { FlappyBirdRunHandle } from './browser-entry.types';
