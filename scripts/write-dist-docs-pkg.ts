/*
 * Writes a minimal package.json into dist-docs designating ESM for generated
 * scripts.
 *
 * Keeping this as a tiny compiled TypeScript utility aligns the scripts folder
 * on one file type while preserving the same post-compile behavior.
 */

import fs from 'node:fs';
import path from 'node:path';

const DIST_DOCS_DIRECTORY_PATH = path.resolve('dist-docs');
const DIST_DOCS_PACKAGE_PATH = path.join(
  DIST_DOCS_DIRECTORY_PATH,
  'package.json',
);
const DIST_DOCS_PACKAGE_JSON = {
  type: 'module',
};

/**
 * Writes the `dist-docs/package.json` manifest.
 *
 * @returns Nothing.
 */
function main(): void {
  try {
    fs.mkdirSync(DIST_DOCS_DIRECTORY_PATH, { recursive: true });
    fs.writeFileSync(
      DIST_DOCS_PACKAGE_PATH,
      `${JSON.stringify(DIST_DOCS_PACKAGE_JSON, null, 2)}\n`,
    );
    console.log('[docs] Wrote dist-docs/package.json (type=module)');
  } catch (error) {
    console.error('[docs] Failed to write dist-docs/package.json', error);
    process.exit(1);
  }
}

main();
