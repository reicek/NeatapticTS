/**
 * Red tests for the racing runtime adaptation cleanup (P8S3-08).
 *
 * The runtime adaptation engine and browser entry must stay free of legacy
 * `src/performance/nge/*` acceleration imports and hardcoded `disableGPU` /
 * `disableWorkers` flags. This suite asserts that cleanup contract and that
 * `AccelerationConfig` is exposed and forwarded through the engine options.
 */

import fs from 'fs';
import path from 'path';

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

describe('runtime.adaptation.ts cleanup contract', () => {
  const runtimeSource = () =>
    readRacingSourceFile('controller/runtime.adaptation.ts');

  it('does not import from the legacy performance/nge acceleration path', () => {
    expect(hasImportFrom(runtimeSource(), 'src/performance/nge')).toBe(false);
  });

  it('does not reference disableGPU or disableWorkers flags', () => {
    expect(runtimeSource()).not.toMatch(/disableGPU|disableWorkers/);
  });

  it('does not pass hardcoded disable flags to the grow-stabilize cycle', () => {
    expect(runtimeSource()).not.toMatch(
      /disableGPU\s*:\s*true|disableWorkers\s*:\s*true/,
    );
  });
});

describe('browser-entry.ts cleanup contract', () => {
  const browserSource = () =>
    readRacingSourceFile('browser-entry/browser-entry.ts');

  it('does not import from the legacy performance/nge acceleration path', () => {
    expect(hasImportFrom(browserSource(), 'src/performance/nge')).toBe(false);
  });

  it('imports acceleration status helpers from the generic acceleration layer', () => {
    expect(hasImportFrom(browserSource(), 'src/acceleration/')).toBe(true);
  });

  it('does not reference the legacy resolveNgeAccelerationMode symbol', () => {
    expect(browserSource()).not.toContain('resolveNgeAccelerationMode');
  });

  it('does not reference disableGPU or disableWorkers flags', () => {
    expect(browserSource()).not.toMatch(/disableGPU|disableWorkers/);
  });
});

describe('RuntimeAdaptationEngineOptions accelerationConfig integration', () => {
  const runtimeSource = () =>
    readRacingSourceFile('controller/runtime.adaptation.ts');

  it('exposes accelerationConfig in RuntimeAdaptationEngineOptions', () => {
    expect(runtimeSource()).toMatch(
      /interface\s+RuntimeAdaptationEngineOptions\s*\{[\s\S]*?accelerationConfig\?:\s*AccelerationConfig/,
    );
  });

  it('forwards accelerationConfig from engine options into the adaptation lifecycle', () => {
    expect(runtimeSource()).toMatch(/options\.accelerationConfig/);
  });
});
