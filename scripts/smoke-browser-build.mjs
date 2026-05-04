import { access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const browserBundlePath = path.resolve(
  projectRoot,
  'dist',
  'neataptic.browser.esm.js',
);
const smokeInputValues = [0.5, 0.5];
const expectedOutputLength = 1;

await access(browserBundlePath);

const browserModule = await import(pathToFileURL(browserBundlePath).href);
const network = new browserModule.Network(2, 1);
const outputValues = network.activate(smokeInputValues);
const firstOutputValue = outputValues.at(0);

const failureReasons = [
  typeof browserModule.Neat === 'function'
    ? null
    : 'Expected Neat export to be a function.',
  typeof browserModule.Network === 'function'
    ? null
    : 'Expected Network export to be a function.',
  Array.isArray(outputValues)
    ? null
    : 'Expected Network.activate() to return an array.',
  outputValues.length === expectedOutputLength
    ? null
    : `Expected one activation output, received ${outputValues.length}.`,
  Number.isFinite(firstOutputValue)
    ? null
    : `Expected the first activation output to be finite, received ${String(firstOutputValue)}.`,
].filter(Boolean);

if (failureReasons.length > 0) {
  throw new Error(
    `Browser ESM smoke check failed. ${failureReasons.join(' ')}`,
  );
}

console.log(
  '[browser-smoke] Browser ESM bundle import and activation succeeded.',
);