/**
 * Custom Jest transformer that transpiles .mjs sources to CommonJS so tests in
 * the agent-customization-scripts project can import them and use jest.mock.
 *
 * This transformer intentionally targets only the small facade modules under
 * scripts/agent-customization/mcp; it does not affect src/ or other projects.
 *
 * Sucrase converts ESM imports to CommonJS but leaves `import.meta.url` as-is,
 * which is a syntax error in Jest's CommonJS runtime.  We pre-replace it with a
 * literal file:// URL derived from the source path so that downstream helpers
 * such as `fileURLToPath(import.meta.url)` keep working.
 */

const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { transform } = require('sucrase');

/** @type {import('@jest/transform').SyncTransformer} */
module.exports = {
  process(sourceText, sourcePath) {
    // When Node is running with --experimental-vm-modules, .mjs files are
    // executed as native ESM by Jest. Transforming them to CommonJS in that
    // mode produces a runtime ReferenceError because `exports` is not defined.
    // In non-ESM mode we still need the CommonJS conversion so the default
    // project can import .mjs sources without the Node ESM loader.
    const usesVmModules =
      process.execArgv.includes('--experimental-vm-modules') ||
      (process.env.NODE_OPTIONS ?? '').includes('--experimental-vm-modules');
    if (usesVmModules) {
      return { code: sourceText, map: null };
    }

    const fileUrl = pathToFileURL(sourcePath).href;
    // Strip a leading UTF-8 BOM so Sucrase receives clean source. Some
    // Node.js .mjs entry points (e.g. shebang modules) are saved with a BOM,
    // which would otherwise produce an opaque "Unexpected character" parse error.
    const sourceWithoutBom = sourceText.replace(/^\uFEFF/u, '');
    const normalized = sourceWithoutBom.replace(
      /import\.meta\.url/g,
      JSON.stringify(fileUrl),
    );
    const result = transform(normalized, {
      transforms: ['imports'],
      filePath: sourcePath,
      sourceMapOptions: { compiledFilename: sourcePath },
    });
    return { code: result.code, map: result.sourceMap };
  },
};
