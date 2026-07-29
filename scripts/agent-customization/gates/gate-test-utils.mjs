/**
 * Shared test-file derivation helpers used by validation gates.
 *
 * Centralises the mapping from changed source files to owner-local test files
 * so that gates such as pre-specialist-smoke and shared-validation do not
 * duplicate the same heuristic.
 */

import path from 'node:path';

/**
 * File extensions that are eligible for co-located test mapping.
 */
export const TESTABLE_EXTENSIONS = new Set([
  '.ts',
  '.tsx',
  '.js',
  '.mjs',
  '.cjs',
]);

/**
 * Test/spec suffixes used to identify co-located test files.
 */
export const TEST_SUFFIXES = ['.test', '.spec'];

/**
 * Derives the narrowest set of owner-local test files for a list of changed
 * source files. A changed test file is returned as-is; a changed source file
 * is mapped to its co-located test file (same directory, basename + `.test.ts` or
 * the matching extension). Test files that cannot be found on disk are still
 * retained so Jest can report the missing-selection failure clearly.
 *
 * @param {string[]} changedFiles - Repo-relative changed file paths.
 * @returns {string[]} Deduplicated repo-relative test file paths.
 */
export function deriveTestFiles(changedFiles) {
  const testFiles = new Set();

  for (const changedFile of changedFiles) {
    const normalized = changedFile.replace(/\\/g, '/');
    const extension = path.extname(normalized);

    if (isTestFile(normalized)) {
      testFiles.add(normalized);
      continue;
    }

    if (!TESTABLE_EXTENSIONS.has(extension)) {
      continue;
    }

    const dirname = path.dirname(normalized);
    const basename = path.basename(normalized, extension);

    for (const suffix of TEST_SUFFIXES) {
      if (extension === '.mjs') {
        testFiles.add(path.posix.join(dirname, `${basename}${suffix}.mjs`));
        // Co-located tests for ESM sources may be authored in TypeScript.
        testFiles.add(path.posix.join(dirname, `${basename}${suffix}.ts`));
      } else if (extension === '.cjs') {
        testFiles.add(path.posix.join(dirname, `${basename}${suffix}.cjs`));
        testFiles.add(path.posix.join(dirname, `${basename}${suffix}.ts`));
      } else {
        // Default TypeScript/JavaScript family uses .test.ts.
        testFiles.add(path.posix.join(dirname, `${basename}${suffix}.ts`));
      }
    }
  }

  return Array.from(testFiles);
}

/**
 * Determines whether a repo-relative path is already a test file.
 *
 * @param {string} filePath - Repo-relative path.
 * @returns {boolean} True when the path matches a test/spec suffix.
 */
export function isTestFile(filePath) {
  const base = path.basename(filePath);
  return TEST_SUFFIXES.some((suffix) =>
    new RegExp(`${suffix}\\.(ts|tsx|js|mjs|cjs)$`, 'u').test(base),
  );
}
