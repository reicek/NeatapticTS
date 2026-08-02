/**
 * @module file-lock-tracker
 * @description In-memory (process-scoped) file-lock tracker for parallel slice
 *   orchestration.
 *
 * Serializes parallel slices that declare overlapping `files_to_change` sets
 * while allowing disjoint slices to run concurrently. The tracker is
 * intentionally process-scoped: it lives in the orchestrator's address space
 * and does not persist across process restarts or cross process boundaries.
 *
 * Exports:
 * - `acquire(files)`  → `lockId | null` — claim a lock over a file set. Returns
 *   `null` (rather than throwing) when the file set conflicts with an existing
 *   lock, so callers can poll or queue.
 * - `release(lockId)`  → `boolean` — release a previously acquired lock.
 * - `isConflict(files)` → `boolean` — check whether a file set would conflict
 *   with any currently held lock without acquiring one.
 * - `reset()`          → `void` — clear all locks (used by tests and on
 *   orchestrator startup).
 *
 * @example
 * ```js
 * import { acquire, release, isConflict } from './file-lock-tracker.mjs';
 * const lockId = acquire(['src/neat.ts', 'testing/neat.test.ts']);
 * if (lockId === null) {
 *   // Wait for the conflicting slice to finish, then retry.
 * }
 * release(lockId);
 * ```
 */

/**
 * Monotonically increasing lock identifier counter.
 * @type {number}
 */
let lockCounter = 0;

/**
 * Active locks keyed by lock id. Each entry stores the normalized file set so
 * `release` and conflict checks can operate without external state.
 * @type {Map<number, string[]>}
 */
const activeLocks = new Map();

/**
 * Normalize a file list into a deduplicated, sorted array of repo-relative
 * POSIX-style paths. Empty/blank entries are dropped. This canonical form
 * makes conflict detection order-independent.
 *
 * @param {string[]} files - Raw file paths (may contain duplicates or blanks).
 * @returns {string[]} Canonical, sorted, deduplicated file list.
 */
function normalizeFiles(files) {
  if (!Array.isArray(files)) return [];
  const seen = new Set();
  for (const file of files) {
    if (typeof file !== 'string') continue;
    const trimmed = file.trim();
    if (trimmed === '') continue;
    seen.add(trimmed.replaceAll('\\', '/'));
  }
  return Array.from(seen).sort();
}

/**
 * Determine whether a candidate file set intersects any currently held lock.
 *
 * @param {string[]} files - Raw file paths to check.
 * @returns {boolean} `true` when at least one file in `files` is already locked
 *   by an active lock; `false` when the set is disjoint from every active lock.
 *
 * @example
 * ```js
 * import { isConflict } from './file-lock-tracker.mjs';
 * if (isConflict(['src/neat.ts'])) {
 *   console.log('file is locked by another slice');
 * }
 * ```
 */
export function isConflict(files) {
  const candidate = normalizeFiles(files);
  if (candidate.length === 0) return false;
  const candidateSet = new Set(candidate);
  for (const lockedFiles of activeLocks.values()) {
    for (const locked of lockedFiles) {
      if (candidateSet.has(locked)) return true;
    }
  }
  return false;
}

/**
 * Acquire a lock over a file set. Returns a unique `lockId` on success or
 * `null` when the file set conflicts with an existing active lock (so callers
 * can poll without catching exceptions).
 *
 * @param {string[]} files - File paths to lock. Empty arrays always succeed
 *   (a no-op lock that never conflicts) so callers can pass slice file lists
 *   uniformly without special-casing.
 * @returns {number|null} A numeric `lockId` on success, or `null` on conflict.
 *
 * @example
 * ```js
 * import { acquire } from './file-lock-tracker.mjs';
 * const lockId = acquire(['src/neat.ts']);
 * if (lockId === null) { / * queue or retry * / }
 * ```
 */
export function acquire(files) {
  const normalized = normalizeFiles(files);
  if (isConflict(normalized)) return null;
  lockCounter += 1;
  activeLocks.set(lockCounter, normalized);
  return lockCounter;
}

/**
 * Release a previously acquired lock. Safe to call with an unknown or already
 * released `lockId` (returns `false` without throwing).
 *
 * @param {number} lockId - The `lockId` returned by `acquire`.
 * @returns {boolean} `true` when a lock was actually released, `false` when
 *   the `lockId` was unknown or already released.
 *
 * @example
 * ```js
 * import { acquire, release } from './file-lock-tracker.mjs';
 * const lockId = acquire(['src/neat.ts']);
 * release(lockId); // true
 * release(lockId); // false — already released
 * ```
 */
export function release(lockId) {
  if (typeof lockId !== 'number' || !activeLocks.has(lockId)) return false;
  return activeLocks.delete(lockId);
}

/**
 * Clear every active lock. Intended for orchestrator startup and tests; not
 * part of the normal orchestration flow.
 */
export function reset() {
  activeLocks.clear();
}

/**
 * Return a snapshot of the active locks for diagnostics. The returned map is a
 * shallow copy so callers cannot mutate internal state.
 *
 * @returns {Map<number, string[]>} Copy of the active-lock map.
 */
export function snapshot() {
  return new Map(
    Array.from(activeLocks.entries()).map(([id, files]) => [id, files.slice()]),
  );
}
