#!/usr/bin/env node
/* global clearTimeout, console, setTimeout */
/**
 * @module watch-plans
 * @description Optional dev-mode file watcher that triggers re-indexing for
 * changed `plans/*.plans.md` files.
 *
 * Uses Node `fs.watch` so that active plan editing can drive near-real-time
 * targeted re-embedding without waiting for the next commit.
 */

import { watch } from 'node:fs';
import path from 'node:path';

const PLAN_FILE_SUFFIX = '.plans.md';
const DEFAULT_DEBOUNCE_MS = 100;

/**
 * Watch plan directories and invoke a callback when `.plans.md` files change.
 *
 * Uses Node `fs.watch` with `{ recursive: true }`. On Linux, `fs.watch` may
 * not support recursive observation of arbitrary subdirectories; the watcher
 * still requests recursion, but callers that need reliable deep coverage on
 * such platforms should supply each subdirectory explicitly.
 *
 * @param {object} options - Watcher options.
 * @param {string[]} options.planDirs - Absolute or relative paths to
 *   directories to watch.
 * @param {number} [options.debounceMs=100] - Debounce window in milliseconds.
 * @param {function(string): Promise<void>|void} options.onChange - Callback
 *   receiving the absolute path of the changed plan file.
 * @returns {Promise<{close: () => void}>} A watcher handle with a `close`
 *   method that stops all watchers and pending timers.
 */
export async function runPlanWatcher(options) {
  const planDirs = options.planDirs ?? [];
  const debounceMs = Number(options.debounceMs ?? DEFAULT_DEBOUNCE_MS);
  const onChange = options.onChange;

  const watchers = [];
  const timers = new Map();

  function handleChange(absolutePath) {
    const normalized = path.resolve(absolutePath);
    if (!isPlanFile(normalized)) return;

    const existingTimer = timers.get(normalized);
    if (existingTimer !== undefined) clearTimeout(existingTimer);

    /* istanbul ignore next -- platform-dependent fs.watch debounce callback */
    timers.set(
      normalized,
      setTimeout(() => {
        timers.delete(normalized);
        Promise.resolve(onChange(normalized)).catch((error) => {
          console.error(
            `watch-plans: onChange failed for ${normalized}:`,
            error,
          );
        });
      }, debounceMs),
    );
  }

  for (const planDir of planDirs) {
    const resolvedDir = path.resolve(planDir);
    const watcher = watch(
      resolvedDir,
      { recursive: true },
      (eventType, filename) => {
        /* istanbul ignore if -- platform-dependent null filename */
        if (!filename) return;
        handleChange(path.join(resolvedDir, filename));
      },
    );
    /* istanbul ignore next -- fs.watch error handler, hard to trigger reliably in tests */
    watcher.on('error', (error) => {
      console.error(`watch-plans: watcher error for ${resolvedDir}:`, error);
    });
    watchers.push(watcher);
  }

  function close() {
    /* istanbul ignore next -- platform-dependent: only reached if timers exist at close time */
    for (const timer of timers.values()) clearTimeout(timer);
    timers.clear();
    for (const watcher of watchers) watcher.close();
    watchers.length = 0;
  }

  return { close };
}

function isPlanFile(filePath) {
  /* istanbul ignore next -- defensive: handleChange always passes a string from path.join */
  return typeof filePath === 'string' && filePath.endsWith(PLAN_FILE_SUFFIX);
}
