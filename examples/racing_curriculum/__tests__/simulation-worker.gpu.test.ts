/**
 * Red tests for the racing simulation worker GPU-helper cleanup (P8S3-08).
 *
 * The worker GPU helper currently imports from `src/architecture/network/gpu/*`,
 * hardcodes a demo-local `RACING_BROWSER_GPU_THRESHOLD`, and duplicates structural
 * GPU eligibility checks. This suite asserts the post-cleanup source contract:
 * the helper delegates to the generic `src/acceleration/*` layer and keeps no
 * demo-specific GPU decision logic.
 */

import fs from 'fs';
import path from 'path';

import { Network } from '../../../src/browser-entry.ts';

/**
 * Read a racing-curriculum source file as UTF-8 text.
 *
 * @param relativePath - Path relative to `examples/racing_curriculum`.
 * @returns The source file text.
 */
function readRacingSourceFile(relativePath: string): string {
  const sourcePath = path.resolve(
    process.cwd(),
    'examples/racing_curriculum',
    relativePath,
  );
  return fs.readFileSync(sourcePath, 'utf-8');
}

/**
 * Check whether a source file contains an ES module `import ... from` statement
 * that references a given path segment.
 *
 * @param sourceText - Source text to inspect.
 * @param segment - Path segment that must appear inside the import specifier.
 * @returns True when a matching import statement is found.
 */
function hasImportFrom(sourceText: string, segment: string): boolean {
  const escaped = segment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(
    `import\\s+.*?\\s+from\\s+['"][^'"]*${escaped}[^'"]*['"]`,
    's',
  );
  return pattern.test(sourceText);
}

describe('simulation-worker.gpu.ts cleanup contract', () => {
  const workerSource = () =>
    readRacingSourceFile('workers/simulation-worker/simulation-worker.gpu.ts');

  it('does not import from the architecture/network/gpu kernel path', () => {
    expect(hasImportFrom(workerSource(), 'src/architecture/network/gpu')).toBe(
      false,
    );
  });

  it('imports GPU eligibility helpers from the generic acceleration layer', () => {
    expect(hasImportFrom(workerSource(), 'src/acceleration/')).toBe(true);
  });

  it('does not hardcode a demo-local GPU threshold', () => {
    expect(workerSource()).not.toContain('RACING_BROWSER_GPU_THRESHOLD');
  });

  it('does not duplicate structural GPU eligibility checks', () => {
    expect(workerSource()).not.toContain('isNetworkStructurallyGPUEligible');
  });
});

describe('legacy NGE acceleration adapter files', () => {
  /**
   * Paths of the NGE acceleration adapter files that were removed in earlier
   * cleanup slices and must remain deleted.
   */
  const legacyFiles: readonly string[] = [
    'src/performance/nge/nge.acceleration.ts',
    'src/performance/nge/nge.acceleration.types.ts',
    'src/performance/nge/nge.acceleration.variants.ts',
    'src/performance/nge/nge.acceleration.test.ts',
    'src/performance/nge/nge.acceleration.variants.test.ts',
    'src/performance/nge/nge.acceleration.adapter.test.ts',
  ];

  it.each(legacyFiles)('remains deleted: %s', (filePath) => {
    expect(fs.existsSync(path.resolve(process.cwd(), filePath))).toBe(false);
  });
});

describe('racing demo exercises the generic acceleration status API', () => {
  it('Network.getAccelerationStatus returns a stable mode string', () => {
    const network = Network.createMLP(2, [3], 1);
    const status = network.getAccelerationStatus();

    expect(status).toHaveProperty('mode');
    expect(typeof status.mode).toBe('string');
    expect(['cpu', 'gpu', 'worker', 'auto']).toContain(status.mode);
  });
});
