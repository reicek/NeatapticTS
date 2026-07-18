/**
 * Red tests for the racing-curriculum browser-entry acceleration status chip
 * (Acceleration_UI_Parallelism slice 04-demo-chip).
 *
 * These tests assert the source contract of `browser-entry.ts` without
 * importing the browser-only module.  They read the file as text and inspect
 * for the persistent, color-coded acceleration chip and its per-frame update
 * path.
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

describe('browser-entry.ts acceleration status chip contract', () => {
  const browserSource = () =>
    readRacingSourceFile('browser-entry/browser-entry.ts');

  it('creates a persistent acceleration status chip in the stage-card metadata row', () => {
    expect(browserSource()).toMatch(
      /createStatusChip\(\s*['"]Acceleration['"]/,
    );
  });

  it('color-codes the acceleration chip red for CPU mode', () => {
    const source = browserSource();
    expect(
      source.includes('racing-status-chip--cpu') ||
        /cpu[^\n]*#ff[0-9a-f]{2,6}/i.test(source) ||
        /cpu[^\n]*rgb\(\s*255\s*,/i.test(source),
    ).toBe(true);
  });

  it('color-codes the acceleration chip green for GPU mode', () => {
    const source = browserSource();
    expect(
      source.includes('racing-status-chip--gpu') ||
        /gpu[^\n]*#00ff/i.test(source) ||
        /gpu[^\n]*rgb\(\s*0\s*,\s*255\s*,/i.test(source),
    ).toBe(true);
  });

  it('color-codes the acceleration chip yellow for WORKER mode', () => {
    const source = browserSource();
    expect(
      source.includes('racing-status-chip--worker') ||
        /worker[^\n]*#ffff/i.test(source) ||
        /worker[^\n]*rgb\(\s*255\s*,\s*255\s*,/i.test(source),
    ).toBe(true);
  });

  it('updates the acceleration chip text by mutating textContent on a persistent element reference', () => {
    const source = browserSource();
    expect(
      /acceleration(?:Chip|Status|Mode)(?:Element|Node|Value)?\.textContent\s*=/.test(
        source,
      ),
    ).toBe(true);
  });

  it('updates the acceleration chip color by mutating className on a persistent element reference', () => {
    const source = browserSource();
    expect(
      /acceleration(?:Chip|Status|Mode)(?:Element|Node|Value)?\.className\s*=/.test(
        source,
      ),
    ).toBe(true);
  });
});

describe('browser-entry.ts 256x parallelism demo default', () => {
  const browserSource = () =>
    readRacingSourceFile('browser-entry/browser-entry.ts');

  it('resolves a 256x parallelVariantCount demo default', () => {
    expect(browserSource()).toMatch(
      /resolveAccelerationConfig\s*\(\s*\{\s*parallelVariantCount:\s*256\s*,?\s*\}\s*\)/,
    );
  });

  it('passes the resolved acceleration config into per-car runtime adaptation engines', () => {
    const source = browserSource();
    const passesToPerCar =
      /createPerCarAdaptationEngines\s*\([\s\S]*?accelerationConfig/.test(
        source,
      );
    const passesDirectly =
      /createRuntimeAdaptationEngine\s*\([\s\S]*?accelerationConfig/.test(
        source,
      );
    expect(passesToPerCar || passesDirectly).toBe(true);
  });

  it('renders the configured parallelism count in the acceleration chip label', () => {
    const source = browserSource();
    const countInterpolated =
      /\$\{\s*[^}]*parallelVariantCount[^}]*\}\s*x/.test(source);
    const countConcatenated = /parallelVariantCount\s*\+\s*['"]x['"]/.test(
      source,
    );
    expect(countInterpolated || countConcatenated).toBe(true);
  });
});
