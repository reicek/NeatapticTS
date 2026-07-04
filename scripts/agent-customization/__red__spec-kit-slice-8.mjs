import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const readmePath = join(
  __dirname,
  '..',
  '..',
  '.github',
  'extensions',
  'README.md',
);

let failed = false;

function check(condition, message) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    failed = true;
  } else {
    console.log(`PASS: ${message}`);
  }
}

if (!existsSync(readmePath)) {
  check(false, '.github/extensions/README.md exists');
  process.exit(1);
}

const content = readFileSync(readmePath, 'utf-8');

check(content.includes('extension.yml'), 'README mentions extension.yml');
check(content.includes('presets'), 'README mentions presets');
check(content.includes('bundles'), 'README mentions bundles');
check(
  /```ya?ml[\s\S]*?extension:/i.test(content),
  'README contains a YAML code-block extension.yml manifest',
);

if (failed) {
  process.exit(1);
}

console.log('All RED assertions passed.');
