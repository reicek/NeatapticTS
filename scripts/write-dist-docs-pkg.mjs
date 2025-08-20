// Writes a minimal package.json into dist-docs designating ESM for generated scripts.
// Using a Node script avoids shell quoting differences between Windows (PowerShell/cmd) and bash.
import fs from 'fs';
import path from 'path';

const outDir = path.resolve('dist-docs');
try {
  fs.mkdirSync(outDir, { recursive: true });
  const pkgFile = path.join(outDir, 'package.json');
  const json = { type: 'module' }; // ESM environment for generated scripts
  fs.writeFileSync(pkgFile, JSON.stringify(json, null, 2));
  console.log('[docs] Wrote dist-docs/package.json (type=module)');
} catch (e) {
  console.error('[docs] Failed to write dist-docs/package.json', e);
  process.exit(1);
}
