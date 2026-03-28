/*
 * Generates per-folder README.md aggregating exported symbols' JSDoc.
 * - Copies root README.md into docs/ (manual content retained)
 * - Skips generating README for src root (leave top-level README manual)
 * Usage: npm run docs:folders
 *
 * The folder split keeps this file orchestration-only: target resolution,
 * source loading, symbol normalization, and markdown emission each live in a
 * dedicated chapter file below.
 */
import {
  emitDirectoryDocs,
  emitFolderIndex,
} from './output/generate-docs.output.js';
import { createGenerateDocsState } from './generate-docs.state.js';
import {
  collectDirectorySymbols,
  dedupeDirectorySymbols,
} from './symbols/generate-docs.symbols.js';
import {
  initializeDocsTarget,
  loadTargetSourceFiles,
  resolveDocsTarget,
} from './generate-docs.targets.js';

/**
 * Generates docs for the requested target.
 *
 * The script runs in four passes:
 * 1. Resolve and prepare the target output tree.
 * 2. Load source files into the shared ts-morph project.
 * 3. Collect and normalize rendered symbols.
 * 4. Emit directory README files and the optional folder index.
 *
 * @returns Nothing.
 */
async function main(): Promise<void> {
  const state = createGenerateDocsState();

  // Step 1: Resolve and prepare the requested target.
  const target = resolveDocsTarget(process.argv.slice(2));
  await initializeDocsTarget(target);

  // Step 2: Load the target source files.
  const sourceFiles = await loadTargetSourceFiles(state, target);

  // Step 3: Collect and normalize rendered symbols.
  const directorySymbolMap = collectDirectorySymbols(sourceFiles);
  dedupeDirectorySymbols(directorySymbolMap);

  // Step 4: Emit README files and the optional folder index.
  await emitDirectoryDocs(state, directorySymbolMap, target);
  if (target.includeFolderIndex) {
    await emitFolderIndex(state, directorySymbolMap, target);
  }

  console.log(`[docs:${target.name}] Per-folder README generation complete.`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
