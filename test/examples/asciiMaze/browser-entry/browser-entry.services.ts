/**
 * Compatibility facade for the dedicated browser-entry service modules.
 *
 * The concrete implementations now live in focused files so callers can keep
 * importing from this stable boundary while internals evolve independently.
 */

export { composeBrowserEntryAbortSignal } from './browser-entry.abort.services';
export { runBrowserEntryCurriculum } from './browser-entry.curriculum.services';
export { installBrowserEntryGlobals } from './browser-entry.globals.services';
export { createBrowserEntryHostServices } from './browser-entry.host.services';
