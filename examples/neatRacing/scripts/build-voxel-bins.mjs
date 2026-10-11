/* global console:readonly, process:readonly */

/**
 * Voxel bin build script for Neat Racing (spec sections 7.3.2 + 12.4;
 * slice P1-S3-C, AC-010).
 *
 * Scans `examples/neatRacing/assets/models/*.model.json`, decodes each
 * document with the shared section 7.2 decoder
 * (`voxel/model.decode.utils.ts`), meshes it with the shared greedy mesher
 * (`voxel/greedy-mesher.ts`), and writes one little-endian `NXVB` binary
 * per model to `examples/neatRacing/assets/bins/<name>-lod128.bin` via
 * `voxel/bin.codec.ts`.
 *
 * Node executes the imported `.ts` modules through native type stripping;
 * the `module.register` hook below retries extensionless relative
 * specifiers with a `.ts` suffix so the landed section 7.2 modules keep
 * their extensionless `./model.schema` import (single-backslash `\.`
 * pattern per the confirmed probe vector). The decoder, mesher, and codec
 * imports are late-bound inside the per-model loop: with zero model
 * documents present the script exits 0 with an explicit "nothing to build"
 * message (P1-S3-green runs `npm run voxel:build` before any model assets
 * exist).
 *
 * The mesher seam expects a `meshVoxelModel(decoded)` export (spec AC-009
 * quad records); when that named export is missing it falls back to the
 * module's single function export, and otherwise fails with a diagnostic
 * listing the actual exports — union reconciliation with the greedy-mesher
 * slice is an orchestrator follow-up.
 *
 * LOD sequencing: only the `-lod128.bin` output is emitted in this slice.
 * The lod64 tier arrives with `voxel/lod.builder.ts` (P1-S3-D) and the real
 * car/wheel model assets with P1-S6-D — a capability-sequencing gap
 * recorded here, not deferred cleanup.
 *
 * @module
 */

import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { register } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/** Directory holding this script (`examples/neatRacing/scripts`). */
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));

/** Neat Racing example root (`examples/neatRacing`, spec 10.2). */
const RACING_ROOT = path.resolve(SCRIPT_DIR, '..');

/** Model JSON directory (spec 10.2: `assets/models`). */
const MODELS_DIR = path.join(RACING_ROOT, 'assets', 'models');

/** Generated `.bin` output directory (spec 10.2: `assets/bins`). */
const BINS_DIR = path.join(RACING_ROOT, 'assets', 'bins');

/** Model document file suffix scanned inside the models directory. */
const MODEL_FILE_SUFFIX = '.model.json';

/** LOD suffix of the emitted bins (spec 7.3.2 naming). */
const LOD128_SUFFIX = '-lod128.bin';

/** Prefix of the "nothing to build" short-circuit message. */
const NOTHING_TO_BUILD_MESSAGE = 'voxel:build: nothing to build';

/** Prefix of every per-model and summary progress line. */
const BUILD_LOG_PREFIX = 'voxel:build:';

/** Prefix of the failure message when a model cannot be built. */
const FAILURE_MESSAGE_PREFIX = 'voxel:build failed:';

/**
 * Resolve-hook source registered via `module.register`. Retries
 * `ERR_MODULE_NOT_FOUND` for extensionless relative specifiers by appending
 * `.ts` (the regex must keep its single-backslash `\.` form — a
 * double-escaped form silently matches extensionless specifiers and
 * rethrows instead of retrying).
 */
const TYPESCRIPT_RESOLVE_HOOK_SOURCE = String.raw`
const TYPESCRIPT_EXTENSION_PATTERN = /\.[a-z]+$/;
export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    const retryable =
      error !== null &&
      typeof error === 'object' &&
      error.code === 'ERR_MODULE_NOT_FOUND' &&
      context !== null &&
      typeof context === 'object' &&
      typeof context.parentURL === 'string' &&
      !TYPESCRIPT_EXTENSION_PATTERN.test(specifier);
    if (!retryable) {
      throw error;
    }
    return nextResolve(specifier + '.ts', context);
  }
}
`;

register(
  `data:text/javascript,${encodeURIComponent(TYPESCRIPT_RESOLVE_HOOK_SOURCE)}`,
  pathToFileURL(import.meta.url).href,
);

/**
 * Builds one `-lod128.bin` per model document found in the models
 * directory. With zero model documents the build short-circuits with exit
 * code 0 and an explicit "nothing to build" message.
 *
 * @returns {Promise<void>} Resolves after every bin is written.
 */
async function buildVoxelBins() {
  const modelFiles = await listModelFiles();
  if (modelFiles.length === 0) {
    console.log(
      `${NOTHING_TO_BUILD_MESSAGE} - no *${MODEL_FILE_SUFFIX} documents found in ${displayPath(MODELS_DIR)}`,
    );
    return;
  }
  await mkdir(BINS_DIR, { recursive: true });
  for (const modelFile of modelFiles) {
    await buildModelBin(modelFile);
  }
  console.log(
    `${BUILD_LOG_PREFIX} wrote ${modelFiles.length} lod128 bin(s) to ${displayPath(BINS_DIR)}`,
  );
}

/**
 * Lists the model document file names inside the models directory in
 * deterministic order; a missing models directory means nothing to build.
 *
 * @returns {Promise<string[]>} Sorted model document file names.
 */
async function listModelFiles() {
  let entries;
  try {
    entries = await readdir(MODELS_DIR, { withFileTypes: true });
  } catch {
    return [];
  }
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(MODEL_FILE_SUFFIX))
    .map((entry) => entry.name)
    .toSorted();
}

/**
 * Decodes, meshes, encodes, and writes one model's lod128 bin.
 *
 * @param {string} modelFileName - Model document file name inside the
 *   models directory.
 * @returns {Promise<void>} Resolves after the bin is written.
 */
async function buildModelBin(modelFileName) {
  const modelPath = path.join(MODELS_DIR, modelFileName);
  const document = JSON.parse(await readFile(modelPath, 'utf8'));

  // Late-bound imports: the shared TS modules load only when a model exists.
  const { decodeVoxelModel } = await import('../voxel/model.decode.utils.ts');
  const decoded = decodeVoxelModel(document);

  const meshVoxelModel = await resolveMesherExport();
  const quads = resolveMeshedQuads(await meshVoxelModel(decoded));

  const { writeVoxelBin } = await import('../voxel/bin.codec.ts');
  const bytes = writeVoxelBin({
    dims: decoded.dims,
    palette: decoded.palette,
    quads,
  });

  const binPath = path.join(BINS_DIR, `${decoded.name}${LOD128_SUFFIX}`);
  await writeFile(binPath, bytes);
  console.log(
    `${BUILD_LOG_PREFIX} ${displayPath(modelPath)} -> ${displayPath(binPath)} (${quads.length} quads, ${bytes.length} bytes)`,
  );
}

/**
 * Resolves the mesher callable from `voxel/greedy-mesher.ts`: the named
 * `meshVoxelModel` export when present, otherwise the module's single
 * function export, otherwise a diagnostic naming the actual exports.
 *
 * @returns {Promise<Function>} The mesher callable taking a decoded model.
 * @throws Error when the mesher module exports no unambiguous mesh entry.
 */
async function resolveMesherExport() {
  const mesherModule = await import('../voxel/greedy-mesher.ts');
  if (typeof mesherModule.meshVoxelModel === 'function') {
    return mesherModule.meshVoxelModel;
  }
  const functionExports = Object.entries(mesherModule).filter(
    ([, value]) => typeof value === 'function',
  );
  if (functionExports.length === 1) {
    const [exportName, exportValue] = functionExports[0];
    console.warn(
      `${BUILD_LOG_PREFIX} greedy-mesher.ts has no meshVoxelModel export; falling back to its single function export "${exportName}"`,
    );
    return exportValue;
  }
  const exportNames = Object.keys(mesherModule).join(', ') || '(none)';
  throw new Error(
    `${BUILD_LOG_PREFIX} greedy-mesher.ts must export meshVoxelModel(decoded); found exports: ${exportNames}`,
  );
}

/**
 * Narrows a mesher result to its quad array: either a bare quad array or an
 * object holding a `quads` array (spec AC-009 quad records).
 *
 * @param {unknown} meshed - The mesher result.
 * @returns {unknown[]} The meshed quad array.
 * @throws Error when the mesher result carries no quad array.
 */
function resolveMeshedQuads(meshed) {
  const quads = Array.isArray(meshed) ? meshed : meshed?.quads;
  if (!Array.isArray(quads)) {
    throw new Error(
      `${BUILD_LOG_PREFIX} the greedy mesher must return a quad array or an object with a quads array`,
    );
  }
  return quads;
}

/**
 * Formats an absolute path relative to the current working directory for
 * readable build logs.
 *
 * @param {string} absolutePath - Absolute filesystem path.
 * @returns {string} The path relative to the process working directory.
 */
function displayPath(absolutePath) {
  return path.relative(process.cwd(), absolutePath);
}

try {
  await buildVoxelBins();
} catch (error) {
  console.error(
    `${FAILURE_MESSAGE_PREFIX} ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
}