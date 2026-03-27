/**
 * Compatibility entrypoint for the dedicated browser-entry module.
 *
 * The real implementation now lives under `browser-entry/browser-entry.ts`.
 * This file remains so existing imports such as `./browser-entry` continue to
 * resolve without changes.
 *
 * Educational note:
 * Treat this file as a stable import shelf, not as the best place to learn the
 * browser host. Readers who want lifecycle wiring, host services, and
 * curriculum orchestration should start with `browser-entry/browser-entry.ts`.
 */

export { start } from './browser-entry/browser-entry';
export type {
  AsciiMazeRunHandle,
  BrowserEntryStartFunction,
  BrowserEntryStartOptions,
} from './browser-entry/browser-entry.types';
