/**
 * Minimal CLI to export a JSON ONNX model from a serialized Network state.
 * Usage (Node >=14):
 *   node scripts/export-onnx.cjs --in network.json --out model.onnx.json [--metadata] [--batch]
 *
 * This is a lightweight helper; for production pipelines integrate directly with the API.
 */

/* eslint-disable @typescript-eslint/no-var-requires */
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const arg = (name, def) => {
  const idx = args.indexOf(name);
  return idx >= 0 ? args[idx + 1] : def;
};
const flag = (name) => args.includes(name);

if (flag('--help') || flag('-h')) {
  console.log(`Usage: node scripts/export-onnx.cjs --in network.json --out model.onnx.json [--metadata] [--batch] [--legacy] [--partial] [--mixed]\n`);
  process.exit(0);
}

const inputFile = arg('--in');
const outputFile = arg('--out');
if (!inputFile || !outputFile) {
  console.error('Error: --in and --out are required.');
  process.exit(1);
}

const includeMetadata = flag('--metadata');
const batchDimension = flag('--batch');
const legacyNodeOrdering = flag('--legacy');
const allowPartialConnectivity = flag('--partial');
const allowMixedActivations = flag('--mixed');

// Lazy import after parsing to avoid ESM interop complications.
const { importFromONNX, exportToONNX } = require('../dist/architecture/onnx');
const Network = require('../dist/architecture/network').default;

try {
  // Read serialized network JSON (as produced by Network#toJSON or similar custom snapshot)
  const raw = JSON.parse(fs.readFileSync(path.resolve(inputFile), 'utf8'));
  // Rehydrate network: we rely on Network.fromJSON existing in built dist (educative note: adapt if shape differs).
  const net = Network.fromJSON(raw);
  const onnx = exportToONNX(net, { includeMetadata, batchDimension, legacyNodeOrdering, allowPartialConnectivity, allowMixedActivations });
  fs.writeFileSync(path.resolve(outputFile), JSON.stringify(onnx, null, 2), 'utf8');
  console.log(`ONNX JSON written to ${outputFile}`);
} catch (err) {
  console.error('Export failed:', err.message || err);
  process.exit(1);
}
