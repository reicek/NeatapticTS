// Writes a minimal package.json into dist-docs forcing CommonJS so Node can require compiled scripts.
// Using a Node script avoids shell quoting differences between Windows (PowerShell/cmd) and bash.
const fs = require('fs');
const path = require('path');
const outDir = path.resolve('dist-docs');
try {
  fs.mkdirSync(outDir, { recursive: true });
  const pkgFile = path.join(outDir, 'package.json');
  const json = { type: 'commonjs' }; // ensure CJS loader for generated scripts
  fs.writeFileSync(pkgFile, JSON.stringify(json, null, 2));
  console.log('[docs] Wrote dist-docs/package.json');
} catch (e) {
  console.error('[docs] Failed to write dist-docs/package.json', e);
  process.exit(1);
}
