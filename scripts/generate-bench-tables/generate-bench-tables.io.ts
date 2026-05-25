/**
 * generate-bench-tables.io.ts
 *
 * Disk I/O for the benchmark table generator.
 * Keeps all `fs` usage behind one boundary so the rest of the pipeline
 * can remain pure and testable without touching the filesystem.
 */

import fs from 'node:fs';
import { BENCHMARK_ARTIFACT_PATH } from './generate-bench-tables.constants.js';
import type { BenchmarkArtifact } from './generate-bench-tables.types.js';

/**
 * Loads the benchmark artifact from disk.
 *
 * Emits a human-readable error to stderr and returns `null` when the artifact
 * is absent or cannot be parsed, so the caller can decide whether to exit.
 *
 * @returns Parsed artifact when present and readable, otherwise `null`.
 */
export function loadArtifact(): BenchmarkArtifact | null {
  if (!fs.existsSync(BENCHMARK_ARTIFACT_PATH)) {
    console.error(
      '[bench:tables] Artifact not found:',
      BENCHMARK_ARTIFACT_PATH,
    );
    return null;
  }

  try {
    return JSON.parse(
      fs.readFileSync(BENCHMARK_ARTIFACT_PATH, 'utf8'),
    ) as BenchmarkArtifact;
  } catch (error) {
    console.error('[bench:tables] Parse error', error);
    return null;
  }
}
