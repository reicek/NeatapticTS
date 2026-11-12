/**
 * Sparse voxel memory accounting for the Neat Racing voxel racer (spec
 * section 13.3; slice P3-S1-B, AC-036, risk R-07, appendix P7).
 *
 * AC-036 asks for two proofs about the 128³ voxel racer:
 *
 * 1. **Sparse retention.** The *game runtime* never holds a dense 128³ byte
 *    grid. `decodeVoxelModel` (model.decode.utils.ts) emits one row-major
 *    `Uint8Array` per slice — 128 slices of 16,384 B for the car — and the
 *    boot path releases those slices as soon as the greedy mesher has
 *    produced the expanded vertex data the render passes upload to GL. The
 *    retained footprint is therefore the GL-side vertex data plus the
 *    palette, never one contiguous dense grid.
 * 2. **A named budget.** The peak model memory of the boot pipeline stays
 *    under one named constant,
 *    {@linkcode NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES}.
 *
 * The editor is the *bounded exception* of appendix P7 (the editor tool is a
 * separate page, never the game runtime): it may hold one dense working
 * volume plus count-bounded dense undo snapshots. This module owns that
 * reconciliation because AC-036 names the editor allowance, and it imports
 * the AC-020 constants (`NEATRACING_EDITOR_VOLUME_LENGTH`,
 * `NEATRACING_EDITOR_UNDO_MAX_STEPS`, `NEATRACING_EDITOR_UNDO_BUDGET_BYTES`)
 * instead of restating them, so the editor figures can never drift.
 *
 * Scope boundaries:
 *
 * - **LOD tiers are out of scope.** `buildLodModel` has no runtime call
 *   site today (only its own test), so no LOD tier footprint is accounted
 *   here; a future runtime LOD path must extend this module rather than
 *   assume the budget already covers it.
 * - **Pure data.** No DOM, no WebGL, no RNG: the module runs under the
 *   plain Node Jest lane (`jest:base`) and is safe to import from the
 *   editor, the browser entry, or a build script.
 * - **Dependency direction.** The engine imports the voxel folder, never the
 *   reverse, so this module imports nothing from `engine/` or
 *   `browser-entry/`.
 *
 * Measurement note (peak semantics): the shipped measurements quoted by the
 * slice brief (car 2,146,628 B, wheel 76,088 B) are the *decode + mesh*
 * subtotal — the dense-grid reference plus the quad records plus the
 * expanded vertex data — and therefore exclude the palette bytes and the
 * mesher plane-mask transients that `pipelinePeakBytes` also carries. Both
 * readings agree on the boot aggregate: the subtotals sum to 2,222,716 B and
 * the mask transients add 52,224 B, so the boot peak is 2,274,940 B plus the
 * two 25-byte palettes, i.e. 2,274,990 B either way.
 *
 * @module
 */

import {
  VOXEL_BIN_PALETTE_ENTRY_BYTES,
  VOXEL_BIN_QUAD_BYTES,
} from './bin.codec';
import { VOXEL_GRID_DIMS } from './model.schema';
import { NEATRACING_EDITOR_VOLUME_LENGTH } from '../editor/editor.state';
import {
  NEATRACING_EDITOR_UNDO_BUDGET_BYTES,
  NEATRACING_EDITOR_UNDO_MAX_STEPS,
} from '../editor/editor.undo';

/**
 * Bytes of one greedy-mesh quad record in the `.bin` pipeline
 * (`position` u16 × 3, `normalAxis` u8, `width`/`height` u16 × 2,
 * `paletteIndex` u8).
 *
 * Imported from `bin.codec` instead of restated so the runtime accounting
 * cannot drift from the shipped binary layout (AC-010).
 */
export const NEATRACING_VOXEL_QUAD_RECORD_BYTES = VOXEL_BIN_QUAD_BYTES;

/** Vertices emitted per greedy quad: two counter-clockwise triangles. */
export const NEATRACING_VOXEL_QUAD_VERTICES = 6;

/**
 * Floats per expanded vertex in the widest runtime layout (the car-lit pass
 * uses 9; the glass pass uses 7, so the car is the worst-case rate).
 */
export const NEATRACING_VOXEL_VERTEX_FLOATS = 9;

/** Bytes per Float32 element of the expanded vertex buffers. */
export const NEATRACING_VOXEL_FLOAT32_BYTES = 4;

/**
 * Bytes of expanded vertex data per greedy quad:
 * 6 vertices × 9 floats × 4 B = 216 B.
 */
export const NEATRACING_VOXEL_VERTEX_EXPANSION_BYTES_PER_QUAD =
  NEATRACING_VOXEL_QUAD_VERTICES *
  NEATRACING_VOXEL_VERTEX_FLOATS *
  NEATRACING_VOXEL_FLOAT32_BYTES;

/**
 * Bytes per mesher plane cell: the `Int16Array` face mask (2 B) plus the
 * `Uint8Array` consumed map (1 B). The greedy mesher keeps one plane in
 * flight at a time, so the mask is a transient, not a retained buffer.
 */
export const NEATRACING_VOXEL_MESHER_MASK_CELL_BYTES = 3;

/**
 * Dense 128³ reference footprint (2,097,152 B): the single contiguous byte
 * grid the *game runtime* must never retain. The editor working volume has
 * exactly this size, which is why the two lanes reconcile against one
 * shared dense reference.
 */
export const NEATRACING_VOXEL_DENSE_REFERENCE_BYTES = VOXEL_GRID_DIMS ** 3;

/**
 * Axis count of a voxel grid (`[width, height, depth]`).
 */
export const NEATRACING_VOXEL_DIM_COUNT = 3;

/**
 * Named AC-036 budget for the peak model memory of the game runtime:
 * 2.5 MiB = 2,621,440 B.
 *
 * Derivation from the shipped assets (measured with `decodeVoxelModel` +
 * `meshVoxelModel` on `assets/models/car-f1.model.json` and
 * `assets/models/wheel.model.json`):
 *
 * | model  | dense grid | quad records | vertex data | mask transients |
 * | ------ | ---------- | ------------ | ----------- | --------------- |
 * | car-f1 | 2,097,152  | 2,604        | 46,872      | 49,152          |
 * | wheel  | 32,768     | 2,280        | 41,040      | 3,072           |
 *
 * The decode + mesh subtotals are 2,146,628 B (car) and 76,088 B (wheel),
 * i.e. 2,222,716 B together; the plane-mask transients add 52,224 B and the
 * two 25-byte palettes add 50 B, so the boot instant that decodes both
 * models peaks at 2,274,990 B. 2.5 MiB keeps 346,450 B (≈13 % of the
 * budget) of headroom over that aggregate while staying far below the
 * 8 MiB editor allowance, which covers the separate editor tool lane (P7)
 * and is reconciled by {@linkcode accountEditorMemory}.
 */
export const NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES = 2_621_440;

/**
 * Appendix P7 allowance for the editor exception: 8 MiB = 8,388,608 B.
 *
 * The editor may hold one dense working volume plus count-bounded dense
 * undo snapshots; the envelope of the two (2,097,152 B + 6,291,456 B) lands
 * exactly on this allowance, which
 * {@linkcode accountEditorMemory} asserts.
 */
export const NEATRACING_EDITOR_MEMORY_EXCEPTION_ALLOWANCE_BYTES =
  8 * 1024 * 1024;

/** Rejection code: the named runtime memory budget was exceeded. */
const CODE_MEMORY_BUDGET_EXCEEDED = 'VOXEL_MEMORY_BUDGET_EXCEEDED';

/** Rejection code: a dense volume grid was retained at game runtime. */
const CODE_DENSE_GRID_RETAINED = 'VOXEL_DENSE_GRID_RETAINED';

/** Rejection code: the editor undo snapshots exceed their named budget. */
const CODE_EDITOR_UNDO_BUDGET_EXCEEDED = 'VOXEL_EDITOR_UNDO_BUDGET_EXCEEDED';

/** Rejection code: a memory account input is malformed. */
const CODE_MEMORY_ACCOUNT_INVALID = 'VOXEL_MEMORY_ACCOUNT_INVALID';

/**
 * Named rejection for every memory-budget violation (AC-036).
 *
 * Carries a stable machine-readable `code` so callers can branch on
 * rejection kinds without parsing human-readable messages.
 */
export class VoxelMemoryBudgetError extends Error {
  /** Stable rejection code, for example `VOXEL_MEMORY_BUDGET_EXCEEDED`. */
  readonly code: string;

  /**
   * @param code - Stable machine-readable rejection code.
   * @param message - Human-readable rejection reason.
   */
  constructor(code: string, message: string) {
    super(message);
    this.name = new.target.name;
    this.code = code;
  }
}

/**
 * Typed-buffer classes a voxel model can hold.
 *
 * - `slice-grid` — one decoded per-slice palette-index grid.
 * - `vertex-data` — the expanded Float32 vertex buffer uploaded to GL.
 * - `palette` — the palette entries (r, g, b, a, glow).
 */
export type VoxelBufferKind = 'slice-grid' | 'vertex-data' | 'palette';

/**
 * Pure-data summary of one model's boot-pipeline footprint.
 *
 * Callers derive `quadCount` from `meshVoxelModel` output and
 * `paletteEntryCount` from the decoded palette.
 */
export interface VoxelModelMemoryInput {
  /** Human-readable model label used in error messages. */
  readonly label: string;
  /** Grid dims `[width, height, depth]` of the model. */
  readonly dims: readonly [number, number, number];
  /** Greedy-meshed quad count of the model. */
  readonly quadCount: number;
  /** Number of palette entries (index 0 is the EMPTY sentinel). */
  readonly paletteEntryCount: number;
}

/**
 * One typed-buffer class in a retention manifest.
 */
export interface VoxelBufferManifestEntry {
  /** Which typed-buffer class this entry describes. */
  readonly kind: VoxelBufferKind;
  /** Bytes of ONE buffer of this class (not the class total). */
  readonly byteLength: number;
  /** How many buffers of this class the pipeline allocates. */
  readonly count: number;
  /**
   * True only when this entry is a single contiguous dense volume grid
   * (`byteLength === dims[0] * dims[1] * dims[2]`), which the game runtime
   * must never retain.
   */
  readonly isDenseVolumeGrid: boolean;
}

/**
 * Memory account of a single voxel model.
 *
 * `pipelinePeakBytes` is the worst instant of the boot pipeline for this
 * model: dense-grid reference + palette + quad records + expanded vertex
 * data + the mesher plane-mask transient.
 * `retainedAfterBootBytes` is what survives the boot: the expanded vertex
 * data plus the palette (the slices and the quad records are released).
 */
export interface VoxelModelMemoryAccount {
  /** Model label echoed from the input. */
  readonly label: string;
  /** Grid dims `[width, height, depth]` echoed from the input. */
  readonly dims: readonly [number, number, number];
  /** Dense-equivalent footprint of the grid (`dims[0] * dims[1] * dims[2]`). */
  readonly denseGridBytes: number;
  /** Bytes actually decoded into sparse per-slice grids. */
  readonly decodedSliceBytes: number;
  /** Number of decoded per-slice grids. */
  readonly sliceCount: number;
  /** Bytes of the largest single decoded slice buffer. */
  readonly maxSingleSliceBufferBytes: number;
  /** Palette bytes (`paletteEntryCount × 5 B`). */
  readonly paletteBytes: number;
  /** Bytes of the greedy-meshed quad records. */
  readonly quadRecordBytes: number;
  /** Bytes of the expanded Float32 vertex data. */
  readonly vertexDataBytes: number;
  /** Peak bytes of the mesher plane-mask transient. */
  readonly mesherMaskPeakBytes: number;
  /** Worst-instant footprint of this model's boot pipeline. */
  readonly pipelinePeakBytes: number;
  /** Footprint that survives the boot (vertex data + palette). */
  readonly retainedAfterBootBytes: number;
  /** Typed-buffer classes the boot pipeline allocates for this model. */
  readonly retainedManifest: readonly VoxelBufferManifestEntry[];
  /**
   * Count of retained manifest entries that are dense volume grids. Always
   * `0` for a sparse account; the sparse-retention assertion fails when it
   * is not.
   */
  readonly denseGridBuffersRetained: number;
}

/**
 * Aggregated memory account of the models the boot path decodes at once.
 */
export interface VoxelBootMemoryAccount {
  /** How many models were accounted. */
  readonly modelCount: number;
  /** Per-model accounts, in input order. */
  readonly accounts: readonly VoxelModelMemoryAccount[];
  /** Sum of the per-model pipeline peaks (the boot worst instant). */
  readonly aggregatePipelinePeakBytes: number;
  /** Sum of the per-model retained-after-boot footprints. */
  readonly aggregateRetainedAfterBootBytes: number;
  /** Sum of the per-model dense-grid references. */
  readonly aggregateDenseGridBytes: number;
  /** Sum of the per-model dense-grid buffer counts (0 when sparse). */
  readonly denseGridBuffersRetained: number;
  /** The named runtime budget the aggregate is evaluated against. */
  readonly budgetBytes: number;
  /** Budget left after the aggregate peak (`budgetBytes - peak`). */
  readonly headroomBytes: number;
  /** True when the aggregate peak fits inside the named budget. */
  readonly withinBudget: boolean;
}

/**
 * Memory account of the editor lane (appendix P7 exception).
 *
 * The editor is allowed one dense working volume plus count-bounded dense
 * undo snapshots; every byte figure here comes from the imported AC-020
 * constants, never from a restated literal.
 */
export interface EditorMemoryAccount {
  /** Dense working volume the editor paints into (AC-020). */
  readonly workingVolumeBytes: number;
  /** Named byte budget of the undo snapshots (AC-020). */
  readonly undoBudgetBytes: number;
  /** Bytes the undo stack currently retains (measured, never assumed). */
  readonly editorUndoSnapshotBytes: number;
  /** Dense snapshots the undo stack currently retains. */
  readonly retainedUndoSteps: number;
  /** The P7 allowance the editor exception must fit inside. */
  readonly undoAllowanceBytes: number;
  /** Working volume + undo budget (`2,097,152 + 6,291,456 = 8,388,608`). */
  readonly editorExceptionEnvelopeBytes: number;
  /** True when the retained snapshot bytes fit inside the undo budget. */
  readonly withinUndoBudget: boolean;
  /** True when the editor exception envelope fits inside the allowance. */
  readonly withinAllowance: boolean;
}

/**
 * Accounts the memory footprint of one voxel model through the boot
 * pipeline (AC-036).
 *
 * The account separates three byte groups:
 *
 * - the **decoded sparse footprint** (`decodedSliceBytes`, one
 *   `dims[0] * dims[1]` buffer per slice) versus the **dense reference**
 *   (`denseGridBytes`) the runtime must never hold as one buffer;
 * - the **mesh transients** (quad records, expanded vertex data, the mesher
 *   plane mask), which exist only while the model is being meshed and
 *   uploaded;
 * - the **retained footprint** (`retainedAfterBootBytes` = vertex data +
 *   palette), which is what the GL passes keep alive after the boot.
 *
 * @param model - Pure-data summary of the model's boot-pipeline footprint.
 * @returns The memory account of the model.
 * @throws VoxelMemoryBudgetError with code `VOXEL_MEMORY_ACCOUNT_INVALID`
 *   when the label, dims, quad count, or palette count are malformed.
 *
 * @example
 * ```ts
 * const account = accountVoxelModelMemory({
 *   label: 'car-f1',
 *   dims: [128, 128, 128],
 *   quadCount: 217,
 *   paletteEntryCount: 5,
 * });
 * account.decodedSliceBytes; // 2,097,152 across 128 slices of 16,384 B
 * account.maxSingleSliceBufferBytes; // 16,384 — never the dense 2,097,152
 * account.retainedAfterBootBytes; // 46,897 — the vertex data plus the palette
 * ```
 */
export function accountVoxelModelMemory(
  model: VoxelModelMemoryInput,
): VoxelModelMemoryAccount {
  assertValidModelMemoryInput(model);

  const [width, height, depth] = model.dims;
  const denseGridBytes = width * height * depth;
  const sliceBufferBytes = width * height;
  const decodedSliceBytes = sliceBufferBytes * depth;
  const paletteBytes = model.paletteEntryCount * VOXEL_BIN_PALETTE_ENTRY_BYTES;
  const quadRecordBytes = model.quadCount * NEATRACING_VOXEL_QUAD_RECORD_BYTES;
  const vertexDataBytes =
    model.quadCount * NEATRACING_VOXEL_VERTEX_EXPANSION_BYTES_PER_QUAD;
  const mesherMaskPeakBytes =
    largestPlaneCellCount(model.dims) * NEATRACING_VOXEL_MESHER_MASK_CELL_BYTES;
  const pipelinePeakBytes =
    denseGridBytes +
    paletteBytes +
    quadRecordBytes +
    vertexDataBytes +
    mesherMaskPeakBytes;
  const retainedAfterBootBytes = vertexDataBytes + paletteBytes;
  const retainedManifest = buildRetainedManifest(
    model,
    sliceBufferBytes,
    vertexDataBytes,
  );
  const denseGridBuffersRetained = countDenseGridBuffers(
    retainedManifest,
    denseGridBytes,
  );

  return {
    label: model.label,
    dims: model.dims,
    denseGridBytes,
    decodedSliceBytes,
    sliceCount: depth,
    maxSingleSliceBufferBytes: sliceBufferBytes,
    paletteBytes,
    quadRecordBytes,
    vertexDataBytes,
    mesherMaskPeakBytes,
    pipelinePeakBytes,
    retainedAfterBootBytes,
    retainedManifest,
    denseGridBuffersRetained,
  };
}

/**
 * Aggregates the memory accounts of the models the boot path decodes
 * simultaneously and evaluates the aggregate against the named runtime
 * budget (AC-036).
 *
 * @param models - Models decoded together during boot; must not be empty.
 * @returns The aggregated boot account with the budget evaluation.
 * @throws VoxelMemoryBudgetError with code `VOXEL_MEMORY_ACCOUNT_INVALID`
 *   when the model list is empty or a model input is malformed.
 *
 * @example
 * ```ts
 * const boot = accountVoxelBootMemory([carInput, wheelInput]);
 * boot.aggregatePipelinePeakBytes; // 2,274,990 — the boot worst instant
 * boot.withinBudget; // true — the budget is 2,621,440 B
 * ```
 */
export function accountVoxelBootMemory(
  models: readonly VoxelModelMemoryInput[],
): VoxelBootMemoryAccount {
  if (models.length === 0) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_ACCOUNT_INVALID,
      'boot memory accounting needs at least one model; an empty model list cannot produce a budget evaluation',
    );
  }

  const accounts = models.map((model) => accountVoxelModelMemory(model));
  const aggregatePipelinePeakBytes = sumAccounts(
    accounts,
    (account) => account.pipelinePeakBytes,
  );
  const aggregateRetainedAfterBootBytes = sumAccounts(
    accounts,
    (account) => account.retainedAfterBootBytes,
  );
  const aggregateDenseGridBytes = sumAccounts(
    accounts,
    (account) => account.denseGridBytes,
  );
  const denseGridBuffersRetained = sumAccounts(
    accounts,
    (account) => account.denseGridBuffersRetained,
  );
  const budgetBytes = NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES;

  return {
    modelCount: accounts.length,
    accounts,
    aggregatePipelinePeakBytes,
    aggregateRetainedAfterBootBytes,
    aggregateDenseGridBytes,
    denseGridBuffersRetained,
    budgetBytes,
    headroomBytes: budgetBytes - aggregatePipelinePeakBytes,
    withinBudget: aggregatePipelinePeakBytes <= budgetBytes,
  };
}

/**
 * Accounts the editor lane against the appendix P7 exception and
 * reconciles it with the AC-020 constants (AC-036 scope note).
 *
 * The reconciliation is arithmetic, not restated: the undo budget is
 * `NEATRACING_EDITOR_UNDO_MAX_STEPS × NEATRACING_EDITOR_VOLUME_LENGTH`, and
 * the envelope (working volume + undo budget) is exactly the 8 MiB
 * allowance (2,097,152 + 6,291,456 = 8,388,608).
 *
 * @param editorUndoBytesInUse - Bytes the undo stack currently retains
 *   (measured with `EditorUndo.bytesInUse()`, never assumed).
 * @param retainedUndoSteps - Dense snapshots the undo stack currently
 *   retains (measured with `EditorUndo.size()`).
 * @returns The editor memory account.
 * @throws VoxelMemoryBudgetError with code `VOXEL_MEMORY_ACCOUNT_INVALID`
 *   when the byte figure is not a non-negative integer or the step count is
 *   not an integer inside `0..NEATRACING_EDITOR_UNDO_MAX_STEPS`.
 *
 * @example
 * ```ts
 * const undo = createEditorUndo(volume);
 * undo.push();
 * const account = accountEditorMemory(undo.bytesInUse(), undo.size());
 * account.withinUndoBudget; // true — 2,097,152 B of 6,291,456 B
 * account.editorExceptionEnvelopeBytes; // 8,388,608 — exactly the allowance
 * ```
 */
export function accountEditorMemory(
  editorUndoBytesInUse: number,
  retainedUndoSteps: number,
): EditorMemoryAccount {
  assertValidEditorUndoUsage(editorUndoBytesInUse, retainedUndoSteps);

  const workingVolumeBytes = NEATRACING_EDITOR_VOLUME_LENGTH;
  const undoBudgetBytes = NEATRACING_EDITOR_UNDO_BUDGET_BYTES;
  const undoAllowanceBytes = NEATRACING_EDITOR_MEMORY_EXCEPTION_ALLOWANCE_BYTES;
  const editorExceptionEnvelopeBytes = workingVolumeBytes + undoBudgetBytes;

  return {
    workingVolumeBytes,
    undoBudgetBytes,
    editorUndoSnapshotBytes: editorUndoBytesInUse,
    retainedUndoSteps,
    undoAllowanceBytes,
    editorExceptionEnvelopeBytes,
    withinUndoBudget: editorUndoBytesInUse <= undoBudgetBytes,
    withinAllowance: editorExceptionEnvelopeBytes <= undoAllowanceBytes,
  };
}

/**
 * Asserts that the runtime memory account is sparse and within the named
 * AC-036 budget (the `assertVoxelRuntimeMemoryWithinBudget` contract).
 *
 * @param account - A per-model account or an aggregated boot account.
 * @returns `true` when the account is sparse and within budget.
 * @throws VoxelMemoryBudgetError with code `VOXEL_DENSE_GRID_RETAINED` when
 *   a dense volume grid is retained.
 * @throws VoxelMemoryBudgetError with code `VOXEL_MEMORY_BUDGET_EXCEEDED`
 *   when the peak footprint exceeds the named budget.
 *
 * @example
 * ```ts
 * const boot = accountVoxelBootMemory([carInput, wheelInput]);
 * assertVoxelRuntimeMemoryWithinBudget(boot); // true
 * ```
 */
export function assertVoxelRuntimeMemoryWithinBudget(
  account: VoxelModelMemoryAccount | VoxelBootMemoryAccount,
): true {
  if (isBootMemoryAccount(account)) {
    for (const modelAccount of account.accounts) {
      assertSparseVoxelRetention(modelAccount);
    }
    if (account.denseGridBuffersRetained !== 0) {
      throw new VoxelMemoryBudgetError(
        CODE_DENSE_GRID_RETAINED,
        `the boot account retains ${account.denseGridBuffersRetained} dense volume grids; the game runtime must stay sparse`,
      );
    }
    if (!account.withinBudget) {
      throw new VoxelMemoryBudgetError(
        CODE_MEMORY_BUDGET_EXCEEDED,
        `boot peak model memory of ${account.aggregatePipelinePeakBytes} bytes exceeds the named budget of ${account.budgetBytes} bytes`,
      );
    }
    return true;
  }

  assertSparseVoxelRetention(account);
  if (
    account.pipelinePeakBytes >
    NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES
  ) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_BUDGET_EXCEEDED,
      `model "${account.label}" peaks at ${account.pipelinePeakBytes} bytes, above the named budget of ${NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES} bytes`,
    );
  }
  return true;
}

/**
 * Asserts that a model account retains no dense volume grid (the sparse
 * half of AC-036).
 *
 * The check is per buffer, never per class total: the car's 128 slices add
 * up to the dense 2,097,152 B, but no single buffer is that large, so the
 * class total is allowed while a dense single buffer is not.
 *
 * @param account - The model account to inspect.
 * @returns `true` when the manifest holds no dense volume grid.
 * @throws VoxelMemoryBudgetError with code `VOXEL_DENSE_GRID_RETAINED` when
 *   a manifest entry is a dense volume grid or the dense-buffer count is
 *   not zero.
 *
 * @example
 * ```ts
 * assertSparseVoxelRetention(carAccount); // true — 128 slices of 16,384 B
 * ```
 */
export function assertSparseVoxelRetention(
  account: VoxelModelMemoryAccount,
): true {
  const denseGridBytes = account.dims[0] * account.dims[1] * account.dims[2];
  for (const entry of account.retainedManifest) {
    // The explicit flag is authoritative for every buffer class; the
    // byte-length inference skips the palette, whose fixed 5-byte entry
    // length can coincide with a small dense footprint (a 1x1x5 grid)
    // without ever being a volume grid.
    const isDenseBuffer =
      entry.isDenseVolumeGrid ||
      (entry.kind !== 'palette' && entry.byteLength === denseGridBytes);
    if (isDenseBuffer) {
      throw new VoxelMemoryBudgetError(
        CODE_DENSE_GRID_RETAINED,
        `model "${account.label}" retains a dense ${denseGridBytes}-byte volume grid as a ${entry.kind} buffer; the game runtime must stay sparse`,
      );
    }
  }
  if (account.denseGridBuffersRetained !== 0) {
    throw new VoxelMemoryBudgetError(
      CODE_DENSE_GRID_RETAINED,
      `model "${account.label}" reports ${account.denseGridBuffersRetained} dense volume grids; the game runtime must stay sparse`,
    );
  }
  return true;
}

/**
 * Asserts that the editor lane stays inside the appendix P7 exception
 * (the named undo budget and the 8 MiB allowance).
 *
 * @param account - The editor memory account to inspect.
 * @returns `true` when the account is within both limits.
 * @throws VoxelMemoryBudgetError with code
 *   `VOXEL_EDITOR_UNDO_BUDGET_EXCEEDED` when the retained snapshot bytes
 *   exceed the named undo budget or the exception envelope exceeds the
 *   allowance.
 *
 * @example
 * ```ts
 * const account = accountEditorMemory(undo.bytesInUse(), undo.size());
 * assertEditorMemoryWithinBudget(account); // true
 * ```
 */
export function assertEditorMemoryWithinBudget(
  account: EditorMemoryAccount,
): true {
  if (!account.withinUndoBudget) {
    throw new VoxelMemoryBudgetError(
      CODE_EDITOR_UNDO_BUDGET_EXCEEDED,
      `editor undo snapshots retain ${account.editorUndoSnapshotBytes} bytes, above the named undo budget of ${account.undoBudgetBytes} bytes`,
    );
  }
  if (!account.withinAllowance) {
    throw new VoxelMemoryBudgetError(
      CODE_EDITOR_UNDO_BUDGET_EXCEEDED,
      `the editor exception envelope of ${account.editorExceptionEnvelopeBytes} bytes exceeds the P7 allowance of ${account.undoAllowanceBytes} bytes`,
    );
  }
  return true;
}

/**
 * Rejects malformed model inputs before any accounting runs.
 *
 * @param model - The model input to validate.
 * @throws VoxelMemoryBudgetError with code `VOXEL_MEMORY_ACCOUNT_INVALID`
 *   when the label is empty, the dims are not three positive integers, or
 *   the quad / palette counts are not non-negative integers.
 */
function assertValidModelMemoryInput(model: VoxelModelMemoryInput): void {
  if (typeof model.label !== 'string' || model.label.length === 0) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_ACCOUNT_INVALID,
      'a memory account needs a non-empty model label',
    );
  }
  if (
    !Array.isArray(model.dims) ||
    model.dims.length !== NEATRACING_VOXEL_DIM_COUNT
  ) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_ACCOUNT_INVALID,
      `model "${model.label}" must declare ${NEATRACING_VOXEL_DIM_COUNT} grid dims; received ${String(model.dims)}`,
    );
  }
  for (const axis of model.dims) {
    if (!Number.isInteger(axis) || axis <= 0) {
      throw new VoxelMemoryBudgetError(
        CODE_MEMORY_ACCOUNT_INVALID,
        `model "${model.label}" has an invalid grid dim ${axis}; every dim must be a positive integer`,
      );
    }
  }
  if (!Number.isInteger(model.quadCount) || model.quadCount < 0) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_ACCOUNT_INVALID,
      `model "${model.label}" has an invalid quad count ${model.quadCount}; it must be a non-negative integer`,
    );
  }
  if (
    !Number.isInteger(model.paletteEntryCount) ||
    model.paletteEntryCount < 0
  ) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_ACCOUNT_INVALID,
      `model "${model.label}" has an invalid palette entry count ${model.paletteEntryCount}; it must be a non-negative integer`,
    );
  }
}

/**
 * Rejects malformed editor undo usage before any accounting runs.
 *
 * @param editorUndoBytesInUse - Bytes the undo stack retains.
 * @param retainedUndoSteps - Snapshots the undo stack retains.
 * @throws VoxelMemoryBudgetError with code `VOXEL_MEMORY_ACCOUNT_INVALID`
 *   when the byte figure is not a non-negative integer or the step count is
 *   not an integer inside `0..NEATRACING_EDITOR_UNDO_MAX_STEPS`.
 */
function assertValidEditorUndoUsage(
  editorUndoBytesInUse: number,
  retainedUndoSteps: number,
): void {
  if (!Number.isInteger(editorUndoBytesInUse) || editorUndoBytesInUse < 0) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_ACCOUNT_INVALID,
      `editor undo bytes in use must be a non-negative integer; received ${editorUndoBytesInUse}`,
    );
  }
  if (
    !Number.isInteger(retainedUndoSteps) ||
    retainedUndoSteps < 0 ||
    retainedUndoSteps > NEATRACING_EDITOR_UNDO_MAX_STEPS
  ) {
    throw new VoxelMemoryBudgetError(
      CODE_MEMORY_ACCOUNT_INVALID,
      `editor undo step count must be an integer in 0..${NEATRACING_EDITOR_UNDO_MAX_STEPS}; received ${retainedUndoSteps}`,
    );
  }
}

/**
 * Builds the typed-buffer manifest of one model: the sparse per-slice
 * grids, the expanded vertex data, and the palette.
 *
 * @param model - The validated model input.
 * @param sliceBufferBytes - Bytes of one decoded slice grid.
 * @param vertexDataBytes - Bytes of the expanded vertex buffer.
 * @returns The manifest entries in allocation order.
 */
function buildRetainedManifest(
  model: VoxelModelMemoryInput,
  sliceBufferBytes: number,
  vertexDataBytes: number,
): VoxelBufferManifestEntry[] {
  const denseGridBytes = model.dims[0] * model.dims[1] * model.dims[2];
  return [
    {
      kind: 'slice-grid',
      byteLength: sliceBufferBytes,
      count: model.dims[2],
      isDenseVolumeGrid: sliceBufferBytes === denseGridBytes,
    },
    {
      kind: 'vertex-data',
      byteLength: vertexDataBytes,
      count: 1,
      isDenseVolumeGrid: vertexDataBytes === denseGridBytes,
    },
    {
      kind: 'palette',
      byteLength: VOXEL_BIN_PALETTE_ENTRY_BYTES,
      count: model.paletteEntryCount,
      isDenseVolumeGrid: false,
    },
  ];
}

/**
 * Counts the manifest entries that are dense volume grids.
 *
 * @param manifest - The retention manifest to scan.
 * @param denseGridBytes - Dense-equivalent footprint of the grid.
 * @returns How many entries are dense volume grids (0 when sparse).
 */
function countDenseGridBuffers(
  manifest: readonly VoxelBufferManifestEntry[],
  denseGridBytes: number,
): number {
  let denseBuffers = 0;
  for (const entry of manifest) {
    const isDense =
      entry.isDenseVolumeGrid ||
      (entry.kind !== 'palette' && entry.byteLength === denseGridBytes);
    if (isDense) {
      denseBuffers += 1;
    }
  }
  return denseBuffers;
}

/**
 * Computes the largest face-plane cell count of a grid: the greedy mesher
 * allocates one plane mask at a time, and the biggest plane sets the peak
 * mask footprint.
 *
 * @param dims - Grid dims `[width, height, depth]`.
 * @returns Cells in the largest face plane.
 */
function largestPlaneCellCount(
  dims: readonly [number, number, number],
): number {
  const [width, height, depth] = dims;
  return Math.max(width * height, width * depth, height * depth);
}

/**
 * Sums one byte figure across the per-model accounts.
 *
 * @param accounts - The per-model accounts to sum.
 * @param select - Selects the byte figure to sum.
 * @returns The sum of the selected byte figures.
 */
function sumAccounts(
  accounts: readonly VoxelModelMemoryAccount[],
  select: (account: VoxelModelMemoryAccount) => number,
): number {
  let total = 0;
  for (const account of accounts) {
    total += select(account);
  }
  return total;
}

/**
 * Narrows a runtime memory account to the aggregated boot account.
 *
 * @param account - A per-model account or an aggregated boot account.
 * @returns True when the account is an aggregated boot account.
 */
function isBootMemoryAccount(
  account: VoxelModelMemoryAccount | VoxelBootMemoryAccount,
): account is VoxelBootMemoryAccount {
  return 'accounts' in account;
}
