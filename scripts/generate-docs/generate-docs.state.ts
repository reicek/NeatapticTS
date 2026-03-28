/*
 * Creates the shared runtime state for one docs-generation process.
 *
 * The generator intentionally reuses one ts-morph project and one docs-order
 * cache set so each chapter can stay pure about its responsibility while the
 * process still shares expensive state safely.
 */

import { Project } from 'ts-morph';

import type { GenerateDocsState } from './generate-docs.types.js';

/**
 * Creates the mutable runtime state shared across one docs-generation run.
 *
 * The generator intentionally keeps one ts-morph project and one pair of
 * docs-order caches for the full process so each target run can reuse the same
 * lookup behavior without relying on module-level globals.
 *
 * @returns Shared docs-generator state.
 */
export function createGenerateDocsState(): GenerateDocsState {
  return {
    project: new Project({
      tsConfigFilePath: 'tsconfig.json',
      skipAddingFilesFromTsConfig: true,
    }),
    resolvedDirectoryDocsOrderConfigCache: new Map(),
    directoryDocsOrderConfigCache: new Map(),
  };
}
