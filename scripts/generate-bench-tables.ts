/**
 * generate-bench-tables.ts
 *
 * Reads the unified benchmark artifact (benchmark.results.json) and emits two markdown table
 * snippets to STDOUT for easy inclusion in Memory_Optimization.md:
 *  1. Variant Delta Table (timings + bytes/conn)
 *  2. Node Heap Metrics Table (heapUsed/rss)
 *
 * Usage:
 *   npm run bench:tables > bench_tables.md
 *
 * Design notes:
 *  - No external dependencies (keep execution lightweight & CI friendly).
 *  - Gracefully degrades when fields missing (older artifact schema).
 *  - Extend later for variance/regression annotation summaries.
 */

import { loadArtifact } from './generate-bench-tables/generate-bench-tables.io.js';
import {
  buildHeapTable,
  buildVariantDeltaTable,
} from './generate-bench-tables/generate-bench-tables.tables.js';

/**
 * Runs the artifact load and emits both markdown tables to standard output.
 *
 * Exits with code 1 when the artifact cannot be read or parsed, so the
 * calling npm script surfaces the failure clearly in CI logs.
 *
 * @returns Nothing.
 */
function main(): void {
  const artifact = loadArtifact();
  if (!artifact) process.exit(1);
  const out = [
    '```',
    buildVariantDeltaTable(artifact),
    '```',
    '',
    '```',
    buildHeapTable(artifact),
    '```',
  ].join('\n');
  console.log(out);
}

main();
