/**
 * Editor undo stack for NeatRacing (slice P1-S6-B; spec section 8.3 /
 * AC-020 and AC-036 of `plans/neatracing-voxel-racer.spec.md`): the
 * count-bounded 3-step undo the editor shell drives while painting the
 * dense working volume (spec section 8.3, LD-28: "3-step undo/redo;
 * oldest evicted").
 *
 * Snapshots are dense `Uint8Array` copies of the working volume taken at
 * {@linkcode EditorUndo.push} time, so a snapshot never aliases the live
 * volume and later paints never corrupt one. {@linkcode EditorUndo.undo}
 * restores the newest retained snapshot into the live volume in place —
 * the volume reference stays stable so renderers holding it keep seeing
 * the restored state — and reports `false` on an empty stack. The stack
 * retains at most {@linkcode NEATRACING_EDITOR_UNDO_MAX_STEPS} dense
 * snapshots and evicts the oldest one beyond that cap, keeping the total
 * snapshot bytes tracked by the stack within
 * {@linkcode NEATRACING_EDITOR_UNDO_BUDGET_BYTES} for any volume up to
 * the editor working volume's size.
 *
 * The named budget derives structurally from the editor volume length
 * (3 × 2,097,152 = 6,291,456 bytes ≈ 6.3 MB), so a full cap of
 * editor-volume snapshots can never exceed the ≤ 8 MB editor undo
 * budget the spec names (appendix P7's dense-storage exception, separate
 * from the game runtime's sparse rule, AC-036).
 *
 * The module stays pure data — no DOM, no WebGL context, and no RNG —
 * so the undo stack is testable in the plain Node Jest environment and
 * the editor shell binds it straight onto the painting workflow.
 */

import { NEATRACING_EDITOR_VOLUME_LENGTH } from './editor.state';

/**
 * Maximum number of dense snapshots the editor undo stack retains
 * (spec section 8.3, LD-28: a count-bounded 3-step undo; the oldest
 * snapshot is evicted once the cap is reached).
 */
export const NEATRACING_EDITOR_UNDO_MAX_STEPS = 3;

/**
 * Named editor undo byte budget (spec section 8.3; AC-020, AC-036
 * partial): three dense snapshots of the full editor working volume —
 * 3 × 2,097,152 = 6,291,456 bytes ≈ 6.3 MB — within the ≤ 8 MB editor
 * undo budget the spec names (appendix P7). Deriving the budget from
 * {@linkcode NEATRACING_EDITOR_VOLUME_LENGTH} keeps the guarantee
 * structural: a full {@linkcode NEATRACING_EDITOR_UNDO_MAX_STEPS}-step
 * stack of editor-volume snapshots can never exceed the named budget.
 */
export const NEATRACING_EDITOR_UNDO_BUDGET_BYTES =
  NEATRACING_EDITOR_UNDO_MAX_STEPS * NEATRACING_EDITOR_VOLUME_LENGTH;

/** Public editor undo surface the editor shell drives. */
export interface EditorUndo {
  /**
   * Captures a dense snapshot of the working volume's current contents.
   *
   * The snapshot is a detached copy, so later paints of the live volume
   * never mutate a retained snapshot. When the stack already holds the
   * 3-step cap, the oldest snapshot is evicted so the byte budget stays
   * within {@linkcode NEATRACING_EDITOR_UNDO_BUDGET_BYTES}.
   *
   * @throws RangeError when the volume is larger than the editor volume,
   *   because a cap-3 stack of larger volumes would break the named
   *   byte budget.
   * @example
   * ```ts
   * const undo = createEditorUndo(volume);
   * undo.push(); // snapshot BEFORE the next paint lands
   * volume[0] = 7;
   * ```
   */
  push(): void;

  /**
   * Restores the newest retained snapshot into the live volume in place
   * (newest-first; the volume reference stays stable) and pops it, so a
   * second undo restores the second-newest snapshot next.
   *
   * @returns true when a snapshot was restored, false when the stack is
   *   empty and nothing was restored.
   * @example
   * ```ts
   * const undo = createEditorUndo(volume);
   * undo.push();
   * volume[0] = 99;
   * if (undo.undo()) {
   *   // volume[0] is back to the pushed state (1).
   * }
   * ```
   */
  undo(): boolean;

  /**
   * Counts the dense snapshots currently retained by the stack.
   *
   * @returns Snapshot count in `0..{@linkcode NEATRACING_EDITOR_UNDO_MAX_STEPS}`.
   * @example
   * ```ts
   * const undo = createEditorUndo(volume);
   * undo.push();
   * console.log(undo.size()); // 1
   * ```
   */
  size(): number;

  /**
   * Reports whether an undo restore is available right now.
   *
   * @returns true when at least one dense snapshot is retained.
   * @example
   * ```ts
   * const undo = createEditorUndo(volume);
   * console.log(undo.canUndo()); // false until the first push
   * ```
   */
  canUndo(): boolean;

  /**
   * Sums the bytes of the dense snapshots the stack currently tracks
   * (AC-036 partial: dense snapshot accounting is computed from live
   * retained snapshots, never assumed). For editor volumes the total
   * never exceeds {@linkcode NEATRACING_EDITOR_UNDO_BUDGET_BYTES}
   * because eviction keeps the snapshot count at the 3-step cap.
   *
   * @returns Total snapshot bytes currently tracked by the stack.
   * @example
   * ```ts
   * const undo = createEditorUndo(volume); // length 8 volume
   * undo.push();
   * console.log(undo.bytesInUse()); // 8
   * ```
   */
  bytesInUse(): number;
}

/**
 * Rejects volumes too large for the undo budget: a volume longer than
 * the editor working volume would let a full 3-step snapshot stack
 * exceed {@linkcode NEATRACING_EDITOR_UNDO_BUDGET_BYTES}.
 *
 * @param volume - Volume the undo stack would snapshot and restore into.
 * @throws RangeError when `volume` is longer than the editor volume.
 */
function assertUndoVolumeWithinBudget(volume: Uint8Array): void {
  if (volume.length > NEATRACING_EDITOR_VOLUME_LENGTH) {
    throw new RangeError(
      `undo volume of ${volume.length} bytes exceeds the editor volume length ${NEATRACING_EDITOR_VOLUME_LENGTH} and would break NEATRACING_EDITOR_UNDO_BUDGET_BYTES`,
    );
  }
}

/**
 * Evicts the oldest dense snapshot whenever the stack grows past the
 * 3-step undo cap (spec section 8.3: oldest evicted), keeping the
 * retained byte total within the named budget.
 *
 * @param snapshots - Dense snapshots ordered oldest (index 0) first.
 */
function evictOldestSnapshotsOverCap(snapshots: Uint8Array[]): void {
  while (snapshots.length > NEATRACING_EDITOR_UNDO_MAX_STEPS) {
    snapshots.shift();
  }
}

/**
 * Sums the byte lengths of the retained dense snapshots (AC-036 partial:
 * dense snapshot accounting asserted from live state, not assumed).
 *
 * @param snapshots - Dense snapshots ordered oldest (index 0) first.
 * @returns Total snapshot bytes currently tracked by the stack.
 */
function sumSnapshotBytes(snapshots: readonly Uint8Array[]): number {
  let totalBytes = 0;
  for (let position = 0; position < snapshots.length; position += 1) {
    totalBytes += snapshots[position].byteLength;
  }
  return totalBytes;
}

/**
 * Creates the editor undo stack bound to the working volume it snapshots
 * and restores into.
 *
 * The volume must be the editor's dense palette-index working volume (or
 * a smaller plain Node `Uint8Array` for tests): the stack keeps dense
 * copies of it, restores into it in place on
 * {@linkcode EditorUndo.undo}, and never reorders or rebuilds the
 * caller's array reference.
 *
 * @param volume - Dense working volume to snapshot on push and restore
 *   into on undo.
 * @returns Ready-to-use undo manager implementing {@linkcode EditorUndo}.
 * @example
 * ```ts
 * import { createEditorState } from './editor.state';
 * import { createEditorUndo } from './editor.undo';
 * const state = createEditorState();
 * const undo = createEditorUndo(state.volume);
 * undo.push();
 * state.paint(0, 0, 0, [255, 0, 0, 255, 0]);
 * undo.undo(); // the paint is rolled back in place
 * ```
 */
export function createEditorUndo(volume: Uint8Array): EditorUndo {
  const snapshots: Uint8Array[] = [];

  return {
    push(): void {
      // Step 1: Guard the named byte budget before allocating a snapshot.
      assertUndoVolumeWithinBudget(volume);

      // Step 2: Snapshot the current contents densely; `new Uint8Array`
      // copies, so the retained bytes never alias the live volume.
      snapshots.push(new Uint8Array(volume));

      // Step 3: Enforce the count-bounded 3-step window (oldest evicted).
      evictOldestSnapshotsOverCap(snapshots);
    },

    undo(): boolean {
      // Step 1: Pop the newest retained snapshot; an empty stack undoes
      // nothing (the contract returns false rather than throwing).
      const snapshot = snapshots.pop();
      if (snapshot === undefined) {
        return false;
      }

      // Step 2: Restore the dense snapshot into the live volume in place
      // so the caller's array reference stays stable (newest-first).
      volume.set(snapshot);
      return true;
    },

    size(): number {
      return snapshots.length;
    },

    canUndo(): boolean {
      return snapshots.length > 0;
    },

    bytesInUse(): number {
      return sumSnapshotBytes(snapshots);
    },
  };
}
