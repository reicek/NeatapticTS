/**
 * @module trace-compress
 * @description Streaming gzip compression and decompression for Chrome DevTools
 *   performance trace files.
 *
 * Chrome DevTools MCP traces are frequently 10 MB or larger. Compressing them
 * with `zlib.createGzip()` through a `node:stream/promises` `pipeline()` keeps
 * memory usage flat regardless of file size, so trace storage in `tmp/traces/`
 * stays practical.
 *
 * Usage (CLI):
 *   node scripts/trace-compress.mjs <input.json> [output.json.gz]
 */

import { access, constants } from 'node:fs/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import { createGzip, createGunzip } from 'node:zlib';
import { pathToFileURL } from 'node:url';

/**
 * Verifies that a file exists and is readable before streaming.
 *
 * Used to surface a clear, actionable error when the input path is missing
 * instead of letting the stream emit an opaque `ENOENT` event.
 *
 * @param filePath - Absolute or relative path to check.
 * @throws {Error} When the file is missing or unreadable, with the original
 *   access error attached as `cause`.
 * @returns Resolves when the path is readable.
 */
async function assertReadable(filePath) {
  try {
    await access(filePath, constants.R_OK);
  } catch (cause) {
    throw new Error(`Input file not found or unreadable: ${filePath}`, {
      cause,
    });
  }
}

/**
 * Compresses a JSON trace file to gzip format using memory-safe streaming.
 *
 * The input is never buffered in memory in full; bytes flow through a
 * `createReadStream` → `createGzip` → `createWriteStream` pipeline.
 *
 * @param inputPath - Path to the input JSON file.
 * @param outputPath - Path for the output gzip file.
 * @returns Resolves when compression completes.
 * @throws {Error} When the input file is missing or the pipeline fails.
 */
export async function compressTrace(inputPath, outputPath) {
  await assertReadable(inputPath);
  await pipeline(
    createReadStream(inputPath),
    createGzip(),
    createWriteStream(outputPath),
  );
}

/**
 * Decompresses a gzip trace file back to its original JSON using streaming.
 *
 * Mirrors {@link compressTrace} with `createGunzip()` so round-trip
 * verification stays memory-safe for large traces.
 *
 * @param inputPath - Path to the input gzip file.
 * @param outputPath - Path for the restored JSON file.
 * @returns Resolves when decompression completes.
 * @throws {Error} When the input file is missing or the pipeline fails.
 */
export async function decompressTrace(inputPath, outputPath) {
  await assertReadable(inputPath);
  await pipeline(
    createReadStream(inputPath),
    createGunzip(),
    createWriteStream(outputPath),
  );
}

/**
 * Parses CLI arguments for the compress CLI.
 *
 * @param argv - CLI arguments after the script path.
 * @returns Parsed input and output paths. When no output is supplied, the
 *   input path gains a `.gz` suffix.
 * @throws {Error} When no input path is provided.
 */
export function parseCliArgs(argv) {
  const [input, output] = argv;
  if (!input) {
    throw new Error(
      'Missing input path. Example: node scripts/trace-compress.mjs <input.json> [output.json.gz]',
    );
  }
  return { input, output: output ?? `${input}.gz` };
}

/**
 * Determines whether the current process is the CLI entry point.
 *
 * @param argv1 - The `process.argv[1]` value for the current process.
 * @param entryUrl - The `import.meta.url` of the module.
 * @returns True when the module is invoked directly as a script.
 */
export function isCliEntryPoint(argv1, entryUrl) {
  return Boolean(argv1) && entryUrl === pathToFileURL(argv1).href;
}

/**
 * Runs the compress CLI when the module is invoked as the entry point.
 *
 * Exposed for unit testing so the CLI dispatch path can be exercised without
 * spawning a child process.
 *
 * @param argv - CLI arguments after the script path.
 * @param context - Invocation context carrying the entry-point signals.
 * @returns Resolves when the CLI action completes or no-ops when not main.
 */
export async function runCli(argv, context) {
  if (!isCliEntryPoint(context.argv1, context.entryUrl)) {
    return;
  }
  const { input, output } = parseCliArgs(argv);
  await compressTrace(input, output);
  console.log(`Compressed ${input} -> ${output}`);
}

await runCli(process.argv.slice(2), {
  argv1: process.argv[1],
  entryUrl: import.meta.url,
});
