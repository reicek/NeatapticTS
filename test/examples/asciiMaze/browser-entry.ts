/**
 * Compatibility entrypoint for the dedicated browser-entry module.
 *
 * The real implementation now lives under `browser-entry/browser-entry.ts`.
 * This file remains so existing imports such as `./browser-entry` continue to
 * resolve without changes.
 */

export { start } from './browser-entry/browser-entry';
export type {
  AsciiMazeRunHandle,
  BrowserEntryStartFunction,
  BrowserEntryStartOptions,
} from './browser-entry/browser-entry.types';
