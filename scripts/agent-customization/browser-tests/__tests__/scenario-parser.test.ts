import { readFile } from 'node:fs/promises';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');
const SCENARIO_HTML_PATH = path.join(
  REPO_ROOT,
  'docs',
  'browser-tests',
  'webgpu-inference-smoke.html',
);

describe('WebGPU smoke scenario page', () => {
  it('loads the Neataptic browser IIFE bundle', async () => {
    const html = await readFile(SCENARIO_HTML_PATH, 'utf8');

    expect(html).toContain('../../dist/neataptic.browser.iife.js');
  });

  it('builds a 2-3-1 MLP via Neataptic.Network.createMLP', async () => {
    const html = await readFile(SCENARIO_HTML_PATH, 'utf8');

    expect(html).toContain('Neataptic.Network.createMLP(2, [3], 1)');
  });

  it('emits window.webgpuSmokeResult with the required schema fields', async () => {
    const html = await readFile(SCENARIO_HTML_PATH, 'utf8');

        // The page assembles the canonical result object via runWebGPUSmoke and
    // assigns it to the window global for the harness to read.
    expect(html).toContain('window.webgpuSmokeResult = result;');
    expect(html).toContain('runWebGPUSmoke');
    expect(html).toMatch(/cpuOutput/);
    expect(html).toMatch(/gpuOutput/);
    expect(html).toMatch(/gpuDeviceBound/);
  });
});
