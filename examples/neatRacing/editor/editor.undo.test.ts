/**
 * Red-first contract tests for the NeatRacing editor undo stack (slice P1-S6-B;
 * spec section 8.3 / AC-020 and AC-036).
 *
 * Locks the undo contract: the editor owns a count-bounded stack of at most
 * three dense Uint8Array snapshots of the working volume. Pushing stores a
 * snapshot; undo restores the most recent snapshot into the live volume and
 * returns true, or returns false when the stack is empty. The oldest snapshot
 * is evicted once the cap is reached, and the total bytes tracked by the stack
 * stay within the named editor undo budget.
 */

import { describe, expect, it } from '@jest/globals';
import { createEditorState } from './editor.state';
import {
  createEditorUndo,
  NEATRACING_EDITOR_UNDO_BUDGET_BYTES,
} from './editor.undo';

/**
 * Builds an undo manager and pushes one checkpoint BEFORE writing each value
 * at index 0 of a dense test volume, so each retained snapshot holds the
 * previous value while the live volume holds the last one written. The volume
 * length equals the number of values so `bytesInUse()` is easy to predict.
 */
function makeUndoStackWithValues(values: number[]) {
  const volume = new Uint8Array(values.length);
  const undo = createEditorUndo(volume);
  for (let i = 0; i < values.length; i += 1) {
    undo.push();
    volume[0] = values[i];
  }
  return { undo, volume };
}

describe('editor.undo (P1-S6-B; spec AC-020 / AC-036)', () => {
  describe('undo byte budget', () => {
    it('exports a budget no larger than 8 MiB', () => {
      expect(NEATRACING_EDITOR_UNDO_BUDGET_BYTES).toBeLessThanOrEqual(
        8 * 1024 * 1024,
      );
    });

    it('fits three dense editor volume snapshots within the budget', () => {
      const volume = createEditorState().volume;
      expect(volume.length).toBe(128 * 128 * 128);
      expect(volume.length * 3).toBeLessThanOrEqual(
        NEATRACING_EDITOR_UNDO_BUDGET_BYTES,
      );
    });
  });

  describe('empty stack', () => {
    it('starts with no snapshots and no undo available', () => {
      const undo = createEditorUndo(new Uint8Array(8));
      expect(undo.size()).toBe(0);
      expect(undo.canUndo()).toBe(false);
      expect(undo.bytesInUse()).toBe(0);
    });

    it('returns false when undo is called on an empty stack', () => {
      const undo = createEditorUndo(new Uint8Array(8));
      expect(undo.undo()).toBe(false);
    });
  });

  describe('push and single undo', () => {
    it('records a snapshot and enables undo', () => {
      const undo = createEditorUndo(new Uint8Array(8));
      undo.push();
      expect(undo.size()).toBe(1);
      expect(undo.canUndo()).toBe(true);
      expect(undo.bytesInUse()).toBe(8);
    });

    it('restores the latest snapshot and returns true', () => {
      const volume = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const undo = createEditorUndo(volume);
      undo.push();
      volume[0] = 99;
      expect(undo.undo()).toBe(true);
      expect(volume[0]).toBe(1);
    });
  });

  describe('dense snapshots', () => {
    it('does not alias the live volume after push', () => {
      const volume = new Uint8Array([10, 20, 30]);
      const undo = createEditorUndo(volume);
      undo.push();
      volume[0] = 99;
      volume[1] = 88;
      expect(undo.undo()).toBe(true);
      expect(volume[0]).toBe(10);
      expect(volume[1]).toBe(20);
    });
  });

  describe('3-step window and eviction', () => {
    it('evicts the oldest snapshot when a fourth push exceeds the cap', () => {
      const { undo } = makeUndoStackWithValues([1, 2, 3, 4]);
      expect(undo.size()).toBe(3);
      expect(undo.canUndo()).toBe(true);
      expect(undo.bytesInUse()).toBe(12);
    });

    it('restores the newest retained snapshot after a capped push', () => {
      const { undo, volume } = makeUndoStackWithValues([1, 2, 3, 4]);
      expect(undo.undo()).toBe(true);
      expect(volume[0]).toBe(3);
    });

    it('restores the second-newest retained snapshot on the second undo', () => {
      const { undo, volume } = makeUndoStackWithValues([1, 2, 3, 4]);
      undo.undo();
      expect(undo.undo()).toBe(true);
      expect(volume[0]).toBe(2);
    });

    it('restores the oldest retained snapshot on the third undo', () => {
      const { undo, volume } = makeUndoStackWithValues([1, 2, 3, 4]);
      undo.undo();
      undo.undo();
      expect(undo.undo()).toBe(true);
      expect(volume[0]).toBe(1);
    });

    it('empties the stack after the retained snapshots are exhausted', () => {
      const { undo } = makeUndoStackWithValues([1, 2, 3]);
      undo.undo();
      undo.undo();
      undo.undo();
      expect(undo.canUndo()).toBe(false);
      expect(undo.undo()).toBe(false);
      expect(undo.size()).toBe(0);
    });
  });
});
