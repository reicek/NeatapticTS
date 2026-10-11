/**
 * Red-contract tests for the P1-S6-D deterministic car generator and the
 * committed voxel model assets it produces (spec §10.2 / AC-022).
 *
 * These tests are intentionally RED today: the generator script
 * `examples/neatRacing/scripts/generate-car-voxel.mjs` and the committed
 * assets `assets/models/car-f1.model.json` and `wheel.model.json` do not
 * yet exist. The failure reason must be the missing implementation, not a
 * test or import error.
 */

import { describe, expect, it } from '@jest/globals';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { validateVoxelModelDocument, VOXEL_GRID_DIMS } from './model.schema';

/** Repository root assumed to be Jest's working directory. */
const REPO_ROOT = process.cwd();

/** Directory that should hold the committed model JSON assets. */
const COMMITTED_MODELS_DIR = join(
  REPO_ROOT,
  'examples',
  'neatRacing',
  'assets',
  'models',
);

/** The generator CLI script under test. */
const GENERATOR_SCRIPT_PATH = join(
  REPO_ROOT,
  'examples',
  'neatRacing',
  'scripts',
  'generate-car-voxel.mjs',
);

/** Names of the committed / generated model documents. */
const CAR_MODEL_NAME = 'car-f1';
const WHEEL_MODEL_NAME = 'wheel';

describe('committed voxel model assets (P1-S6-D)', () => {
  it('car-f1.model.json is a valid 128³ non-empty voxel document', () => {
    const raw = readFileSync(
      join(COMMITTED_MODELS_DIR, `${CAR_MODEL_NAME}.model.json`),
      'utf8',
    );
    const document = validateVoxelModelDocument(JSON.parse(raw));

    expect(document.meta.dims).toEqual([
      VOXEL_GRID_DIMS,
      VOXEL_GRID_DIMS,
      VOXEL_GRID_DIMS,
    ]);
    expect(modelHasSolidRun(document.slices)).toBe(true);
  });

  it('wheel.model.json is a valid 32³ non-empty voxel document', () => {
    const raw = readFileSync(
      join(COMMITTED_MODELS_DIR, `${WHEEL_MODEL_NAME}.model.json`),
      'utf8',
    );
    const document = validateVoxelModelDocument(JSON.parse(raw));

    expect(document.meta.dims).toEqual([32, 32, 32]);
    expect(modelHasSolidRun(document.slices)).toBe(true);
  });
});

describe('deterministic car generator (P1-S6-D)', () => {
  const tempDirectories: string[] = [];

  afterEach(() => {
    for (const directory of tempDirectories) {
      rmSync(directory, { recursive: true, force: true });
    }
    tempDirectories.length = 0;
  });

  it('emits byte-identical model files across two independent runs', async () => {
    const runA = await runGeneratorIntoTempDirectory(tempDirectories);
    const runB = await runGeneratorIntoTempDirectory(tempDirectories);

    expect(runA.exitCode).toBe(0);
    expect(runB.exitCode).toBe(0);

    const carA = readFileSync(
      join(runA.outputDirectory, `${CAR_MODEL_NAME}.model.json`),
    );
    const carB = readFileSync(
      join(runB.outputDirectory, `${CAR_MODEL_NAME}.model.json`),
    );
    const wheelA = readFileSync(
      join(runA.outputDirectory, `${WHEEL_MODEL_NAME}.model.json`),
    );
    const wheelB = readFileSync(
      join(runB.outputDirectory, `${WHEEL_MODEL_NAME}.model.json`),
    );

    expect(sha256(carA)).toBe(sha256(carB));
    expect(sha256(wheelA)).toBe(sha256(wheelB));
  });
});

/**
 * Scans RLE row strings for at least one solid voxel run
 * (`paletteIdx !== 0`). This is the cheap proxy the red contract uses to
 * assert non-empty meshes without invoking the greedy mesher.
 *
 * @param slices - The validated document's RLE row matrix.
 * @returns True when at least one run references a non-EMPTY palette index.
 */
function modelHasSolidRun(slices: string[][]): boolean {
  for (const slice of slices) {
    for (const row of slice) {
      for (let i = 0; i < row.length; i += 4) {
        const paletteIndex = Number.parseInt(row.slice(i + 2, i + 4), 16);
        if (paletteIndex !== 0) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Computes a SHA-256 hex digest of a byte buffer.
 *
 * @param bytes - The buffer to hash.
 * @returns The hex digest.
 */
function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

/**
 * Result of one generator invocation in a temporary directory.
 */
interface GeneratorRunResult {
  exitCode: number | null;
  outputDirectory: string;
}

/**
 * Spawns the generator CLI once into a fresh temporary directory using
 * the `--out <dir>` seam. The temporary directory is registered for
 * afterEach cleanup. This never writes the committed assets directory.
 *
 * @param tempDirectories - Mutable array of directories to clean up.
 * @returns The generator result including the output directory for file reads.
 */
async function runGeneratorIntoTempDirectory(
  tempDirectories: string[],
): Promise<GeneratorRunResult> {
  const outputDirectory = mkdtempSync(join(tmpdir(), 'p1-s6-d-car-gen-'));
  tempDirectories.push(outputDirectory);

  const exitCode = await new Promise<number | null>((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [GENERATOR_SCRIPT_PATH, '--out', outputDirectory],
      { shell: false },
    );
    child.on('error', reject);
    child.on('exit', (code) => resolve(code));
  });

  return { exitCode, outputDirectory };
}
