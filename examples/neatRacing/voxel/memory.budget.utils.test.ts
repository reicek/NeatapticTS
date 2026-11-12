/**
 * Contract tests for the Neat Racing sparse voxel memory budget accounting
 * (slice P3-S1-B; spec section 13.3, AC-036, risk R-07, appendix P7).
 *
 * AC-036 asks for two proofs about the 128³ voxel racer:
 *
 * 1. **Sparse retention.** The *game runtime* never holds a dense 128³ byte
 *    grid. The decoded model is a set of per-slice grids, the boot path
 *    releases them once the greedy mesher has produced the expanded vertex
 *    data the render passes upload to GL, and the retention manifest proves
 *    that no single contiguous volume buffer survives.
 * 2. **A named budget.** The peak model memory of the boot pipeline stays
 *    under {@linkcode NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES},
 *    and the editor lane stays inside the appendix P7 exception allowance.
 *
 * The tests read the SHIPPED model documents (`car-f1.model.json` and
 * `wheel.model.json`), decode them with `decodeVoxelModel`, mesh them with
 * `meshVoxelModel`, and derive the account inputs from the real pipeline —
 * never from restated literals. The measured ground truth of that pipeline:
 *
 * | model  | dense grid | quad records | vertex data | mask transient | palette |
 * | ------ | ---------- | ------------ | ----------- | -------------- | ------- |
 * | car-f1 | 2,097,152  | 2,604        | 46,872      | 49,152         | 25      |
 * | wheel  | 32,768     | 2,280        | 41,040      | 3,072          | 25      |
 *
 * The decode + mesh subtotal of a model (dense grid + quad records +
 * expanded vertex data) is 2,146,628 B for the car and 76,088 B for the
 * wheel; the module's `pipelinePeakBytes` additionally carries the retained
 * palette bytes and the mesher plane-mask transient, so the boot peak is
 * 2,274,990 B against the named 2,621,440 B budget.
 *
 * The editor side reconciles the imported AC-020 constants instead of
 * restating them: one dense 2,097,152 B working volume plus a 3-step
 * 6,291,456 B undo budget lands exactly on the 8,388,608 B (8 MiB)
 * appendix P7 allowance.
 */

import { describe, expect, it } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  NEATRACING_EDITOR_MEMORY_EXCEPTION_ALLOWANCE_BYTES,
  NEATRACING_VOXEL_DENSE_REFERENCE_BYTES,
  NEATRACING_VOXEL_FLOAT32_BYTES,
  NEATRACING_VOXEL_MESHER_MASK_CELL_BYTES,
  NEATRACING_VOXEL_QUAD_RECORD_BYTES,
  NEATRACING_VOXEL_QUAD_VERTICES,
  NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES,
  NEATRACING_VOXEL_VERTEX_EXPANSION_BYTES_PER_QUAD,
  NEATRACING_VOXEL_VERTEX_FLOATS,
  VoxelMemoryBudgetError,
  accountEditorMemory,
  accountVoxelBootMemory,
  accountVoxelModelMemory,
  assertEditorMemoryWithinBudget,
  assertSparseVoxelRetention,
  assertVoxelRuntimeMemoryWithinBudget,
  type VoxelBootMemoryAccount,
  type VoxelBufferKind,
  type VoxelBufferManifestEntry,
  type VoxelModelMemoryAccount,
  type VoxelModelMemoryInput,
} from './memory.budget.utils';
import {
  VOXEL_BIN_PALETTE_ENTRY_BYTES,
  VOXEL_BIN_QUAD_BYTES,
} from './bin.codec';
import {
  VOXEL_GRID_DIMS,
  validateVoxelModelDocument,
  type VoxelModelDocument,
} from './model.schema';
import { decodeVoxelModel } from './model.decode.utils';
import { meshVoxelModel } from './greedy-mesher';
import { NEATRACING_EDITOR_VOLUME_LENGTH } from '../editor/editor.state';
import {
  NEATRACING_EDITOR_UNDO_BUDGET_BYTES,
  NEATRACING_EDITOR_UNDO_MAX_STEPS,
  createEditorUndo,
} from '../editor/editor.undo';

/** Repository root assumed to be Jest's working directory. */
const REPO_ROOT = process.cwd();

/** Directory that holds the committed model JSON assets. */
const COMMITTED_MODELS_DIR = join(
  REPO_ROOT,
  'examples',
  'neatRacing',
  'assets',
  'models',
);

/** File name of the shipped 128³ car body model. */
const CAR_MODEL_NAME = 'car-f1';

/** File name of the shipped 32³ wheel model. */
const WHEEL_MODEL_NAME = 'wheel';

/** Measured greedy-quad count of the shipped car body. */
const CAR_QUAD_COUNT = 217;

/** Measured greedy-quad count of the shipped wheel. */
const WHEEL_QUAD_COUNT = 190;

/** Palette entries of both shipped models (index 0 is the EMPTY sentinel). */
const MODEL_PALETTE_ENTRY_COUNT = 5;

/** Dense 128³ reference footprint — the buffer the runtime must never hold. */
const CAR_DENSE_GRID_BYTES = 2_097_152;

/** Dense 32³ reference footprint of the shipped wheel. */
const WHEEL_DENSE_GRID_BYTES = 32_768;

/** Bytes of one decoded car slice grid (128 cols × 128 rows). */
const CAR_SLICE_BUFFER_BYTES = 16_384;

/** Number of decoded car slice grids (one per depth index). */
const CAR_SLICE_COUNT = 128;

/** Bytes of one decoded wheel slice grid (32 cols × 32 rows). */
const WHEEL_SLICE_BUFFER_BYTES = 1_024;

/** Number of decoded wheel slice grids (one per depth index). */
const WHEEL_SLICE_COUNT = 32;

/** Measured decode + mesh subtotal of the shipped car body. */
const CAR_DECODE_MESH_SUBTOTAL_BYTES = 2_146_628;

/** Measured decode + mesh subtotal of the shipped wheel. */
const WHEEL_DECODE_MESH_SUBTOTAL_BYTES = 76_088;

/** Measured decode + mesh subtotal of the boot aggregate (car + wheel). */
const BOOT_DECODE_MESH_SUBTOTAL_BYTES = 2_222_716;

/** Measured plane-mask transient peak of the boot aggregate. */
const BOOT_MASK_TRANSIENT_BYTES = 52_224;

/** Measured palette bytes of the boot aggregate (two 5-entry palettes). */
const BOOT_PALETTE_BYTES = 50;

/** Measured boot peak: the subtotal plus the masks and the palettes. */
const BOOT_PIPELINE_PEAK_BYTES =
  BOOT_DECODE_MESH_SUBTOTAL_BYTES +
  BOOT_MASK_TRANSIENT_BYTES +
  BOOT_PALETTE_BYTES;

/** Editor working-volume length (AC-020) in bytes. */
const EDITOR_VOLUME_BYTES = 2_097_152;

/** Editor undo byte budget (AC-020) in bytes: 3 steps × one volume. */
const EDITOR_UNDO_BUDGET_BYTES = 6_291_456;

/** Appendix P7 allowance for the editor exception in bytes (8 MiB). */
const EDITOR_ALLOWANCE_BYTES = 8_388_608;

/** Tiny hand-computable fixture grid dims. */
const TINY_DIMS: readonly [number, number, number] = [4, 4, 4];

/** Tiny fixture greedy-quad count. */
const TINY_QUAD_COUNT = 6;

/** Tiny fixture palette entry count. */
const TINY_PALETTE_ENTRY_COUNT = 3;

/**
 * Synthetic grid whose single slice IS the whole dense volume: the one
 * shape where a per-slice buffer legitimately equals the dense reference.
 */
const DENSE_SINGLE_SLICE_DIMS: readonly [number, number, number] = [4, 4, 1];

/** Synthetic quad count that pushes a 128³ model past the named budget. */
const OVER_BUDGET_QUAD_COUNT = 12_000;

/** Rejection code: the named runtime memory budget was exceeded. */
const CODE_MEMORY_BUDGET_EXCEEDED = 'VOXEL_MEMORY_BUDGET_EXCEEDED';

/** Rejection code: a dense volume grid was retained at game runtime. */
const CODE_DENSE_GRID_RETAINED = 'VOXEL_DENSE_GRID_RETAINED';

/** Rejection code: the editor undo snapshots exceed their named budget. */
const CODE_EDITOR_UNDO_BUDGET_EXCEEDED = 'VOXEL_EDITOR_UNDO_BUDGET_EXCEEDED';

/** Rejection code: a memory account input is malformed. */
const CODE_MEMORY_ACCOUNT_INVALID = 'VOXEL_MEMORY_ACCOUNT_INVALID';

/**
 * Cache of the derived memory inputs, so the 128³ car is decoded and
 * meshed once for the whole suite.
 */
const MODEL_INPUT_CACHE = new Map<string, VoxelModelMemoryInput>();

/**
 * Reads one shipped model document from the committed assets directory.
 *
 * @param name - Model file name without the `.model.json` suffix.
 * @returns The validated on-disk document.
 */
function loadShippedModelDocument(name: string): VoxelModelDocument {
  const raw = readFileSync(
    join(COMMITTED_MODELS_DIR, `${name}.model.json`),
    'utf8',
  );
  return validateVoxelModelDocument(JSON.parse(raw));
}

/**
 * Derives the pure-data memory input of one shipped model by running the
 * real decode + mesh pipeline over it (AC-036: the accounting is measured,
 * never assumed).
 *
 * @param name - Model file name without the `.model.json` suffix.
 * @returns The derived memory input.
 */
function deriveModelMemoryInput(name: string): VoxelModelMemoryInput {
  const cached = MODEL_INPUT_CACHE.get(name);
  if (cached !== undefined) {
    return cached;
  }
  const decoded = decodeVoxelModel(loadShippedModelDocument(name));
  const mesh = meshVoxelModel(decoded);
  const input: VoxelModelMemoryInput = {
    label: decoded.name,
    dims: decoded.dims,
    quadCount: mesh.quads.length,
    paletteEntryCount: decoded.palette.length,
  };
  MODEL_INPUT_CACHE.set(name, input);
  return input;
}

/**
 * Sums the decode + mesh subtotal of one model (or of every model of a
 * boot aggregate): the dense-grid reference plus the quad records plus the
 * expanded vertex data.
 *
 * @param account - A per-model account or an aggregated boot account.
 * @returns The decode + mesh subtotal in bytes.
 */
function decodeMeshSubtotalBytes(
  account: VoxelModelMemoryAccount | VoxelBootMemoryAccount,
): number {
  if ('accounts' in account) {
    return account.accounts.reduce(
      (total, modelAccount) => total + decodeMeshSubtotalBytes(modelAccount),
      0,
    );
  }
  return (
    account.denseGridBytes + account.quadRecordBytes + account.vertexDataBytes
  );
}

/**
 * Collects the manifest entries that hold a dense volume grid — the
 * retention proof AC-036 demands, checked per buffer and never per class
 * total.
 *
 * @param account - The model account to scan.
 * @returns The offending entries (empty when the retention is sparse).
 */
function denseManifestEntries(
  account: VoxelModelMemoryAccount,
): VoxelBufferManifestEntry[] {
  const denseGridBytes = account.dims[0] * account.dims[1] * account.dims[2];
  return account.retainedManifest.filter(
    (entry) => entry.isDenseVolumeGrid || entry.byteLength === denseGridBytes,
  );
}

/**
 * Reads one typed-buffer class out of a retention manifest.
 *
 * @param account - The model account to read from.
 * @param kind - The typed-buffer class to read.
 * @returns The manifest entry of that class.
 * @throws Error when the manifest carries no entry of that class.
 */
function manifestEntry(
  account: VoxelModelMemoryAccount,
  kind: VoxelBufferKind,
): VoxelBufferManifestEntry {
  const entry = account.retainedManifest.find(
    (candidate) => candidate.kind === kind,
  );
  if (entry === undefined) {
    throw new Error(
      `the retention manifest of "${account.label}" has no ${kind} entry`,
    );
  }
  return entry;
}

/**
 * Runs an action expected to throw a {@link VoxelMemoryBudgetError} and
 * returns its stable rejection code.
 *
 * @param run - The action to run.
 * @returns The stable rejection code of the thrown error.
 * @throws Error when the action throws nothing or a foreign error.
 */
function captureBudgetErrorCode(run: () => unknown): string {
  try {
    run();
  } catch (error) {
    if (error instanceof VoxelMemoryBudgetError) {
      return error.code;
    }
    throw error;
  }
  throw new Error('expected a VoxelMemoryBudgetError to be thrown');
}

describe('accountVoxelModelMemory (spec section 13.3, AC-036)', () => {
  it('derives the sparse decoded footprint from the declared dims', () => {
    const account = accountVoxelModelMemory({
      label: 'tiny',
      dims: TINY_DIMS,
      quadCount: TINY_QUAD_COUNT,
      paletteEntryCount: TINY_PALETTE_ENTRY_COUNT,
    });

    expect(account.decodedSliceBytes).toBe(64);
    expect(account.sliceCount).toBe(4);
    expect(account.maxSingleSliceBufferBytes).toBe(16);
  });

  it('accounts the mesh byte groups of the fixture', () => {
    const account = accountVoxelModelMemory({
      label: 'tiny',
      dims: TINY_DIMS,
      quadCount: TINY_QUAD_COUNT,
      paletteEntryCount: TINY_PALETTE_ENTRY_COUNT,
    });

    expect(account.quadRecordBytes).toBe(72);
    expect(account.vertexDataBytes).toBe(1_296);
    expect(account.mesherMaskPeakBytes).toBe(48);
  });

  it('sums the pipeline peak and the retained footprint of the fixture', () => {
    const account = accountVoxelModelMemory({
      label: 'tiny',
      dims: TINY_DIMS,
      quadCount: TINY_QUAD_COUNT,
      paletteEntryCount: TINY_PALETTE_ENTRY_COUNT,
    });

    expect(account.paletteBytes).toBe(15);
    expect(account.pipelinePeakBytes).toBe(1_495);
    expect(account.retainedAfterBootBytes).toBe(1_311);
  });

  it('derives the vertex expansion rate from the published layout constants', () => {
    expect(NEATRACING_VOXEL_QUAD_VERTICES).toBe(6);
    expect(NEATRACING_VOXEL_VERTEX_FLOATS).toBe(9);
    expect(NEATRACING_VOXEL_FLOAT32_BYTES).toBe(4);
  });

  it('reuses the binary codec byte layout instead of restating it', () => {
    expect(NEATRACING_VOXEL_QUAD_RECORD_BYTES).toBe(VOXEL_BIN_QUAD_BYTES);
    expect(NEATRACING_VOXEL_MESHER_MASK_CELL_BYTES).toBe(3);
    expect(VOXEL_BIN_PALETTE_ENTRY_BYTES).toBe(5);
  });

  it('shares one dense reference between the runtime and the editor lane', () => {
    expect(NEATRACING_VOXEL_DENSE_REFERENCE_BYTES).toBe(VOXEL_GRID_DIMS ** 3);
    expect(NEATRACING_VOXEL_DENSE_REFERENCE_BYTES).toBe(CAR_DENSE_GRID_BYTES);
  });
});

describe('accountVoxelModelMemory on the shipped model documents (AC-036)', () => {
  it('meshes the shipped car body into 217 greedy quads', () => {
    const carInput = deriveModelMemoryInput(CAR_MODEL_NAME);

    expect(carInput.quadCount).toBe(CAR_QUAD_COUNT);
    expect(carInput.paletteEntryCount).toBe(MODEL_PALETTE_ENTRY_COUNT);
    expect(accountVoxelModelMemory(carInput).quadRecordBytes).toBe(2_604);
  });

  it('reports the car body decode and mesh subtotal of 2,146,628 bytes', () => {
    const carAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(CAR_MODEL_NAME),
    );

    expect(carAccount.denseGridBytes).toBe(CAR_DENSE_GRID_BYTES);
    expect(carAccount.vertexDataBytes).toBe(46_872);
    expect(decodeMeshSubtotalBytes(carAccount)).toBe(
      CAR_DECODE_MESH_SUBTOTAL_BYTES,
    );
  });

  it('reports the car body as 128 decoded slice grids of 16,384 bytes', () => {
    const carAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(CAR_MODEL_NAME),
    );

    expect(carAccount.sliceCount).toBe(CAR_SLICE_COUNT);
    expect(carAccount.maxSingleSliceBufferBytes).toBe(CAR_SLICE_BUFFER_BYTES);
    expect(carAccount.decodedSliceBytes).toBe(CAR_DENSE_GRID_BYTES);
  });

  it('meshes the shipped wheel into 190 greedy quads', () => {
    const wheelInput = deriveModelMemoryInput(WHEEL_MODEL_NAME);

    expect(wheelInput.quadCount).toBe(WHEEL_QUAD_COUNT);
    expect(wheelInput.paletteEntryCount).toBe(MODEL_PALETTE_ENTRY_COUNT);
    expect(accountVoxelModelMemory(wheelInput).quadRecordBytes).toBe(2_280);
  });

  it('reports the wheel decode and mesh subtotal of 76,088 bytes', () => {
    const wheelAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    );

    expect(wheelAccount.denseGridBytes).toBe(WHEEL_DENSE_GRID_BYTES);
    expect(wheelAccount.vertexDataBytes).toBe(41_040);
    expect(decodeMeshSubtotalBytes(wheelAccount)).toBe(
      WHEEL_DECODE_MESH_SUBTOTAL_BYTES,
    );
  });

  it('reports the wheel as 32 decoded slice grids of 1,024 bytes', () => {
    const wheelAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    );

    expect(wheelAccount.sliceCount).toBe(WHEEL_SLICE_COUNT);
    expect(wheelAccount.maxSingleSliceBufferBytes).toBe(
      WHEEL_SLICE_BUFFER_BYTES,
    );
    expect(wheelAccount.decodedSliceBytes).toBe(WHEEL_DENSE_GRID_BYTES);
  });

  it('reports the car body pipeline peak of 2,195,805 bytes', () => {
    const carAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(CAR_MODEL_NAME),
    );

    expect(carAccount.pipelinePeakBytes).toBe(2_195_805);
    expect(carAccount.retainedAfterBootBytes).toBe(46_897);
    expect(carAccount.mesherMaskPeakBytes).toBe(49_152);
  });

  it('reports the wheel pipeline peak of 79,185 bytes', () => {
    const wheelAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    );

    expect(wheelAccount.pipelinePeakBytes).toBe(79_185);
    expect(wheelAccount.retainedAfterBootBytes).toBe(41_065);
    expect(wheelAccount.mesherMaskPeakBytes).toBe(3_072);
  });
});

describe('sparse-only retention manifest (AC-036, risk R-07)', () => {
  it('retains no dense volume grid in either model manifest', () => {
    const carAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(CAR_MODEL_NAME),
    );
    const wheelAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    );

    expect(denseManifestEntries(carAccount)).toEqual([]);
    expect(denseManifestEntries(wheelAccount)).toEqual([]);
    expect(
      carAccount.denseGridBuffersRetained +
        wheelAccount.denseGridBuffersRetained,
    ).toBe(0);
  });

  it('reports the per-slice transient grids instead of one dense buffer', () => {
    const carAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(CAR_MODEL_NAME),
    );
    const sliceEntry = manifestEntry(carAccount, 'slice-grid');

    expect(sliceEntry.byteLength).toBe(CAR_SLICE_BUFFER_BYTES);
    expect(sliceEntry.count).toBe(CAR_SLICE_COUNT);
    expect(
      carAccount.retainedManifest.some(
        (entry) => entry.byteLength === CAR_DENSE_GRID_BYTES,
      ),
    ).toBe(false);
  });

  it('reports the expanded vertex data as quads times 216 bytes', () => {
    const carAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(CAR_MODEL_NAME),
    );
    const wheelAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    );

    expect(manifestEntry(carAccount, 'vertex-data').byteLength).toBe(
      CAR_QUAD_COUNT * NEATRACING_VOXEL_VERTEX_EXPANSION_BYTES_PER_QUAD,
    );
    expect(manifestEntry(wheelAccount, 'vertex-data').byteLength).toBe(
      WHEEL_QUAD_COUNT * NEATRACING_VOXEL_VERTEX_EXPANSION_BYTES_PER_QUAD,
    );
    expect(NEATRACING_VOXEL_VERTEX_EXPANSION_BYTES_PER_QUAD).toBe(216);
  });

  it('asserts the sparse retention of both shipped models', () => {
    const carAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(CAR_MODEL_NAME),
    );
    const wheelAccount = accountVoxelModelMemory(
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    );

    expect(assertSparseVoxelRetention(carAccount)).toBe(true);
    expect(assertSparseVoxelRetention(wheelAccount)).toBe(true);
    expect(assertVoxelRuntimeMemoryWithinBudget(carAccount)).toBe(true);
  });
});

describe('accountVoxelBootMemory (spec section 13.3, AC-036)', () => {
  it('aggregates the boot peak and the retained footprint of both models', () => {
    const boot = accountVoxelBootMemory([
      deriveModelMemoryInput(CAR_MODEL_NAME),
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    ]);

    expect(boot.modelCount).toBe(2);
    expect(decodeMeshSubtotalBytes(boot)).toBe(BOOT_DECODE_MESH_SUBTOTAL_BYTES);
    expect(boot.aggregatePipelinePeakBytes).toBe(BOOT_PIPELINE_PEAK_BYTES);
  });

  it('keeps the boot aggregate inside the named runtime budget', () => {
    const boot = accountVoxelBootMemory([
      deriveModelMemoryInput(CAR_MODEL_NAME),
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    ]);

    expect(boot.budgetBytes).toBe(
      NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES,
    );
    expect(boot.withinBudget).toBe(true);
    expect(boot.headroomBytes).toBe(346_450);
  });

  it('rejects an empty model list', () => {
    expect(captureBudgetErrorCode(() => accountVoxelBootMemory([]))).toBe(
      CODE_MEMORY_ACCOUNT_INVALID,
    );
  });

  it('rejects a model with non-positive dims', () => {
    expect(
      captureBudgetErrorCode(() =>
        accountVoxelModelMemory({
          label: 'zero-dim',
          dims: [0, 4, 4],
          quadCount: 0,
          paletteEntryCount: 1,
        }),
      ),
    ).toBe(CODE_MEMORY_ACCOUNT_INVALID);
    expect(
      captureBudgetErrorCode(() =>
        accountVoxelModelMemory({
          label: 'negative-dim',
          dims: [4, -4, 4],
          quadCount: 0,
          paletteEntryCount: 1,
        }),
      ),
    ).toBe(CODE_MEMORY_ACCOUNT_INVALID);
  });

  it('rejects a model with a malformed count or label', () => {
    expect(
      captureBudgetErrorCode(() =>
        accountVoxelModelMemory({
          label: 'negative-quads',
          dims: TINY_DIMS,
          quadCount: -1,
          paletteEntryCount: 1,
        }),
      ),
    ).toBe(CODE_MEMORY_ACCOUNT_INVALID);
    expect(
      captureBudgetErrorCode(() =>
        accountVoxelModelMemory({
          label: 'fractional-palette',
          dims: TINY_DIMS,
          quadCount: 0,
          paletteEntryCount: 1.5,
        }),
      ),
    ).toBe(CODE_MEMORY_ACCOUNT_INVALID);
    expect(
      captureBudgetErrorCode(() =>
        accountVoxelModelMemory({
          label: '',
          dims: TINY_DIMS,
          quadCount: 0,
          paletteEntryCount: 1,
        }),
      ),
    ).toBe(CODE_MEMORY_ACCOUNT_INVALID);
  });
});

describe('assertVoxelRuntimeMemoryWithinBudget (AC-036)', () => {
  it('accepts the boot account of the shipped models', () => {
    const boot = accountVoxelBootMemory([
      deriveModelMemoryInput(CAR_MODEL_NAME),
      deriveModelMemoryInput(WHEEL_MODEL_NAME),
    ]);

    expect(assertVoxelRuntimeMemoryWithinBudget(boot)).toBe(true);
    expect(boot.denseGridBuffersRetained).toBe(0);
    expect(boot.headroomBytes).toBe(
      NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES -
        boot.aggregatePipelinePeakBytes,
    );
  });

  it('rejects a model that peaks above the named budget', () => {
    const overBudget = accountVoxelModelMemory({
      label: 'over-budget',
      dims: [VOXEL_GRID_DIMS, VOXEL_GRID_DIMS, VOXEL_GRID_DIMS],
      quadCount: OVER_BUDGET_QUAD_COUNT,
      paletteEntryCount: MODEL_PALETTE_ENTRY_COUNT,
    });

    expect(
      captureBudgetErrorCode(() =>
        assertVoxelRuntimeMemoryWithinBudget(overBudget),
      ),
    ).toBe(CODE_MEMORY_BUDGET_EXCEEDED);
    expect(overBudget.pipelinePeakBytes).toBeGreaterThan(
      NEATRACING_VOXEL_RUNTIME_MODEL_MEMORY_BUDGET_BYTES,
    );
  });

  it('rejects a manifest that retains a dense volume grid', () => {
    const denseAccount = accountVoxelModelMemory({
      label: 'dense-single-slice',
      dims: DENSE_SINGLE_SLICE_DIMS,
      quadCount: 0,
      paletteEntryCount: MODEL_PALETTE_ENTRY_COUNT,
    });

    expect(
      captureBudgetErrorCode(() => assertSparseVoxelRetention(denseAccount)),
    ).toBe(CODE_DENSE_GRID_RETAINED);
    expect(denseAccount.denseGridBuffersRetained).toBe(1);
  });

  it('rejects a boot account that retains a dense volume grid', () => {
    const boot = accountVoxelBootMemory([
      {
        label: 'dense-single-slice',
        dims: DENSE_SINGLE_SLICE_DIMS,
        quadCount: 0,
        paletteEntryCount: MODEL_PALETTE_ENTRY_COUNT,
      },
    ]);

    expect(
      captureBudgetErrorCode(() => assertVoxelRuntimeMemoryWithinBudget(boot)),
    ).toBe(CODE_DENSE_GRID_RETAINED);
    expect(boot.denseGridBuffersRetained).toBe(1);
  });
});

describe('accountEditorMemory (AC-036, appendix P7, AC-020)', () => {
  it('imports the AC-020 editor constants without restating them', () => {
    expect(NEATRACING_EDITOR_VOLUME_LENGTH).toBe(EDITOR_VOLUME_BYTES);
    expect(NEATRACING_EDITOR_VOLUME_LENGTH).toBe(VOXEL_GRID_DIMS ** 3);
    expect(NEATRACING_EDITOR_UNDO_BUDGET_BYTES).toBe(EDITOR_UNDO_BUDGET_BYTES);
  });

  it('lands the editor exception envelope exactly on the 8 MiB allowance', () => {
    const account = accountEditorMemory(0, 0);

    expect(account.editorExceptionEnvelopeBytes).toBe(EDITOR_ALLOWANCE_BYTES);
    expect(account.undoBudgetBytes).toBe(
      NEATRACING_EDITOR_UNDO_MAX_STEPS * NEATRACING_EDITOR_VOLUME_LENGTH,
    );
    expect(account.withinAllowance).toBe(true);
    expect(NEATRACING_EDITOR_MEMORY_EXCEPTION_ALLOWANCE_BYTES).toBe(
      EDITOR_ALLOWANCE_BYTES,
    );
  });

  it('accounts a real editor undo stack inside the named budget', () => {
    const undo = createEditorUndo(
      new Uint8Array(NEATRACING_EDITOR_VOLUME_LENGTH),
    );
    undo.push();
    const account = accountEditorMemory(undo.bytesInUse(), undo.size());

    expect(account.editorUndoSnapshotBytes).toBe(EDITOR_VOLUME_BYTES);
    expect(account.retainedUndoSteps).toBe(1);
    expect(account.withinUndoBudget).toBe(true);
  });

  it('asserts the real editor undo stack inside the named budget', () => {
    const undo = createEditorUndo(
      new Uint8Array(NEATRACING_EDITOR_VOLUME_LENGTH),
    );
    undo.push();
    const account = accountEditorMemory(undo.bytesInUse(), undo.size());

    expect(assertEditorMemoryWithinBudget(account)).toBe(true);
    expect(account.undoBudgetBytes).toBe(EDITOR_UNDO_BUDGET_BYTES);
    expect(account.workingVolumeBytes).toBe(EDITOR_VOLUME_BYTES);
  });

  it('reports an over-budget undo stack without throwing', () => {
    const account = accountEditorMemory(
      EDITOR_UNDO_BUDGET_BYTES + 1,
      NEATRACING_EDITOR_UNDO_MAX_STEPS,
    );

    expect(account.withinUndoBudget).toBe(false);
    expect(
      captureBudgetErrorCode(() => assertEditorMemoryWithinBudget(account)),
    ).toBe(CODE_EDITOR_UNDO_BUDGET_EXCEEDED);
  });

  it('rejects malformed editor undo usage', () => {
    expect(captureBudgetErrorCode(() => accountEditorMemory(-1, 0))).toBe(
      CODE_MEMORY_ACCOUNT_INVALID,
    );
    expect(
      captureBudgetErrorCode(() =>
        accountEditorMemory(0, NEATRACING_EDITOR_UNDO_MAX_STEPS + 1),
      ),
    ).toBe(CODE_MEMORY_ACCOUNT_INVALID);
    expect(captureBudgetErrorCode(() => accountEditorMemory(1.5, 0))).toBe(
      CODE_MEMORY_ACCOUNT_INVALID,
    );
  });
});
