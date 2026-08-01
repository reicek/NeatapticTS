/**
 * @module file-lock-tracker.test
 * @description Jest unit tests for the in-memory file-lock tracker.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly. Covers acquire/release/conflict/release-then-
 * reacquire and disjoint-vs-overlapping slice serialization semantics.
 */
import assert from 'node:assert/strict';

import {
  acquire,
  isConflict,
  release,
  reset,
  snapshot,
} from './file-lock-tracker.mjs';

describe('file-lock-tracker', () => {
  beforeEach(() => {
    reset();
  });

  it('acquires a lock for a fresh file set and returns a numeric lockId', () => {
    const lockId = acquire(['src/neat.ts']);
    assert.equal(typeof lockId, 'number');
    assert.ok(lockId > 0);
  });

  it('detects a conflict for an overlapping file set', () => {
    acquire(['src/neat.ts', 'testing/neat.test.ts']);
    assert.strictEqual(isConflict(['src/neat.ts']), true);
    assert.strictEqual(isConflict(['testing/neat.test.ts']), true);
  });

  it('reports no conflict for a disjoint file set', () => {
    acquire(['src/neat.ts']);
    assert.strictEqual(isConflict(['src/architecture/network.ts']), false);
  });

  it('returns null from acquire when the file set conflicts', () => {
    acquire(['src/neat.ts']);
    const second = acquire(['src/neat.ts']);
    assert.strictEqual(second, null);
  });

  it('releases a held lock and returns true', () => {
    const lockId = acquire(['src/neat.ts']);
    assert.strictEqual(release(lockId), true);
  });

  it('returns false when releasing an unknown or already-released lockId', () => {
    assert.strictEqual(release(9999), false);
    const lockId = acquire(['src/neat.ts']);
    release(lockId);
    assert.strictEqual(release(lockId), false);
  });

  it('allows reacquiring the same file set after release', () => {
    const lockId = acquire(['src/neat.ts']);
    release(lockId);
    assert.strictEqual(isConflict(['src/neat.ts']), false);
    const second = acquire(['src/neat.ts']);
    assert.ok(second !== null);
  });

  it('serializes overlapping slices while parallelizing disjoint slices', () => {
    const overlapping = acquire(['src/neat.ts', 'testing/neat.test.ts']);
    const disjoint = acquire(['src/architecture/network.ts']);
    assert.ok(overlapping !== null);
    assert.ok(disjoint !== null);
    assert.strictEqual(acquire(['src/neat.ts']), null);
    assert.strictEqual(acquire(['testing/neat.test.ts']), null);
    release(overlapping);
    assert.ok(acquire(['src/neat.ts']) !== null);
    release(disjoint);
  });

  it('treats empty file lists as never conflicting', () => {
    acquire([]);
    assert.strictEqual(isConflict(['src/neat.ts']), false);
    assert.ok(acquire([]) !== null);
  });

  it('normalizes backslashes and deduplicates paths before conflict checks', () => {
    acquire(['src\\neat.ts', 'src/neat.ts']);
    assert.strictEqual(isConflict(['src/neat.ts']), true);
    assert.strictEqual(snapshot().size, 1);
  });

  it('snapshot returns a copy that cannot mutate internal state', () => {
    acquire(['src/neat.ts']);
    const snap = snapshot();
    snap.clear();
    assert.strictEqual(snapshot().size, 1);
  });
});