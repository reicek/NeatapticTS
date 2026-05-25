import { scanCodeQuality } from './code-quality-scanner.mjs';

const report = await scanCodeQuality();
const weak = report.evidence.filter((entry) => entry.issue === 'weak JSDoc');
const normalize = (filePath) => String(filePath).replace(/\\/g, '/');
const targets = new Set([
  'src/architecture/network/onnx/parity/network.onnx.parity.ts',
  'src/architecture/network/slab/network.slab.utils.ts',
  'src/architecture/network/prune/network.prune.utils.ts',
]);
const targetWeak = weak.filter((entry) => targets.has(normalize(entry.file)));
console.log(JSON.stringify({ weakGlobal: weak.length, weakInTargets: targetWeak.length, targetWeak }, null, 2));
process.exit(report.pass ? 0 : 1);
