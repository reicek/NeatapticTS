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
    const fileUrl = pathToFileURL(sourcePath).href;
    const normalized = sourceText.replace(
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
