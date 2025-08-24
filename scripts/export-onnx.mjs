/**
 * Minimal CLI to export a JSON ONNX model from a serialized Network state.
 * Usage (Node >=20):
 *   node scripts/export-onnx.mjs --in network.json --out model.onnx.json [--metadata] [--batch] [--legacy] [--partial] [--mixed]
 *
 * This is a lightweight helper; for production pipelines integrate directly with the API.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = process.argv.slice(2);
const arg = (name, def) => {
  const idx = args.indexOf(name);
  return idx >= 0 ? args[idx + 1] : def;
};
const flag = (name) => args.includes(name);

if (flag('--help') || flag('-h')) {
  console.log(
    `Usage: node scripts/export-onnx.mjs --in network.json --out model.onnx.json [--metadata] [--batch] [--legacy] [--partial] [--mixed]\n`
  );
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

// Dynamic import of built library (dist) in ESM context
async function loadDist() {
  // Use pathToFileURL to avoid Windows path issues
  const onnxMod = await import(
    pathToFileURL(path.resolve('dist', 'architecture', 'onnx.js')).href
  ).catch(async () => {
    // Fallback to network/network.onnx if direct path changed
    return import(
      pathToFileURL(
        path.resolve('dist', 'architecture', 'network', 'network.onnx.js')
      ).href
    );
  });
  // network default export
  const netMod = await import(
    pathToFileURL(path.resolve('dist', 'architecture', 'network.js')).href
  );
  const Network = netMod.default;
  const { exportToONNX, importFromONNX } = onnxMod; // importFromONNX unused but re-exported for completeness
  return { Network, exportToONNX, importFromONNX };
}

(async () => {
  try {
    const { Network, exportToONNX } = await loadDist();
    const raw = JSON.parse(fs.readFileSync(path.resolve(inputFile), 'utf8'));
    const net = Network.fromJSON(raw);
    const onnx = exportToONNX(net, {
      includeMetadata,
      batchDimension,
      legacyNodeOrdering,
      allowPartialConnectivity,
      allowMixedActivations,
    });
    fs.writeFileSync(
      path.resolve(outputFile),
      JSON.stringify(onnx, null, 2),
      'utf8'
    );
    console.log(`ONNX JSON written to ${outputFile}`);
  } catch (err) {
    console.error('Export failed:', err.message || err);
    process.exit(1);
  }
})();
