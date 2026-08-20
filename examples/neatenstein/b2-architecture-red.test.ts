/**
 * B2 RED-phase tests: Code Quality — Architecture and Debt.
 *
 * These tests encode the six solution items from the Neatenstein Ultimate
 * Quality Upgrade plan (Step B2) as failing contracts.  Each describe block
 * maps to one solution item.  Tests are designed to fail for the right
 * reason: the refactoring has not been performed yet.
 *
 * Structural / architectural tests inspect the filesystem and source content.
 * Behavioral tests exercise runtime contracts (immutability, return types).
 */

import { describe, expect, it } from '@jest/globals';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

const neatensteinRoot = join(process.cwd(), 'examples', 'neatenstein');
const browserEntryRoot = join(neatensteinRoot, 'browser-entry');
const scriptsRoot = join(neatensteinRoot, 'scripts');
const workerRoot = join(browserEntryRoot, 'worker');
const harnessRoot = join(browserEntryRoot, 'harness');

/**
 * Recursively collect all `.ts` files under `dir`, excluding `node_modules`
 * and optionally excluding `.test.ts` / `.d.ts` files.
 */
function collectTsFiles(dir: string, includeTests = false): string[] {
  const results: string[] = [];
  if (!existsSync(dir)) return results;
  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules' || entry === 'dist') continue;
    const fullPath = join(dir, entry);
    const stat = statSync(fullPath);
    if (stat.isDirectory()) {
      results.push(...collectTsFiles(fullPath, includeTests));
    } else if (entry.endsWith('.ts') && !entry.endsWith('.d.ts')) {
      if (!includeTests && entry.endsWith('.test.ts')) continue;
      results.push(fullPath);
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// S1 — browser-entry/shared/ layer
// ---------------------------------------------------------------------------

describe('B2-S1: browser-entry/shared/ layer eliminates bidirectional boundary', () => {
  it('browser-entry/shared/ directory exists', () => {
    expect(existsSync(join(browserEntryRoot, 'shared'))).toBe(true);
  });

  it('shared/math-guards.utils.ts exists and exports consolidated helpers', async () => {
    // Use a variable so ts-jest does not resolve the module at compile time.
    const modulePath = './browser-entry/shared/math-guards.utils';
    let mod: Record<string, unknown> | undefined;
    try {
      mod = (await import(modulePath)) as Record<string, unknown>;
    } catch {
      // Module does not exist yet — expected red failure.
    }
    expect(mod).toBeDefined();
    expect(mod?.clamp).toBeDefined();
    expect(mod?.clamp01).toBeDefined();
    expect(mod?.clampInt).toBeDefined();
    expect(mod?.clampByte).toBeDefined();
    expect(mod?.isFiniteNumber).toBeDefined();
    expect(mod?.isPositiveFinite).toBeDefined();
    expect(mod?.isPositiveFiniteDimension).toBeDefined();
  });

  it('scripts/ does not import from browser-entry/ (no upward dependency)', () => {
    const scriptsFiles = collectTsFiles(scriptsRoot);
    const upwardImports: string[] = [];
    for (const file of scriptsFiles) {
      const content = readFileSync(file, 'utf-8');
      const lines = content.split('\n');
      for (const line of lines) {
        if (/from\s+['"]\.\.\/browser-entry/.test(line)) {
          upwardImports.push(relative(neatensteinRoot, file));
          break;
        }
      }
    }
    expect(upwardImports).toEqual([]);
  });

  it('browser-entry/ does not import from scripts/ (no reverse dependency)', () => {
    const browserEntryFiles = collectTsFiles(browserEntryRoot);
    const reverseImports: string[] = [];
    for (const file of browserEntryFiles) {
      const content = readFileSync(file, 'utf-8');
      const lines = content.split('\n');
      for (const line of lines) {
        // Match imports referencing the scripts/ directory by path
        if (
          /from\s+['"][^'"]*scripts\/(enemy-|snapshot-renderer|voxel-)/.test(
            line,
          )
        ) {
          reverseImports.push(relative(neatensteinRoot, file));
          break;
        }
      }
    }
    expect(reverseImports).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// S2 — display.worker.ts test hooks extraction
// ---------------------------------------------------------------------------

describe('B2-S2: display.worker.ts test hooks extraction', () => {
  it('display.worker.test-hooks.ts file exists in worker/', () => {
    expect(existsSync(join(workerRoot, 'display.worker.test-hooks.ts'))).toBe(
      true,
    );
  });

  it('display.worker.message-handler.utils.ts file exists in worker/', () => {
    expect(
      existsSync(join(workerRoot, 'display.worker.message-handler.utils.ts')),
    ).toBe(true);
  });

  it('display.worker.eval-delegation.utils.ts file exists in worker/', () => {
    expect(
      existsSync(join(workerRoot, 'display.worker.eval-delegation.utils.ts')),
    ).toBe(true);
  });

  it('display.worker.ts no longer exports __testOnly* symbols', () => {
    const content = readFileSync(
      join(workerRoot, 'display.worker.ts'),
      'utf-8',
    );
    expect(content).not.toMatch(/export\s+(?:const|function)\s+__testOnly/);
  });

  it('__testOnlyInjectTestEnemies returns a new array instead of mutating state', async () => {
    let mod: Record<string, unknown> | undefined;
    try {
      // Use a variable so ts-jest does not resolve the module at compile time.
      const modulePath = './browser-entry/worker/display.worker.test-hooks';
      mod = (await import(modulePath)) as Record<string, unknown>;
    } catch {
      // Module does not exist yet — expected red failure.
    }
    expect(mod).toBeDefined();
    const injectFn = mod?.__testOnlyInjectTestEnemies as
      ((positions: { x: number; y: number }[]) => unknown) | undefined;
    expect(injectFn).toBeDefined();
    const result = injectFn?.([{ x: 1, y: 2 }]);
    expect(Array.isArray(result)).toBe(true);
  });

  it('display.worker.message-handler.ts orchestrator file exists in worker/', () => {
    // SOLID pattern: orchestrator (no .utils suffix) calling executor (.utils.ts)
    expect(
      existsSync(join(workerRoot, 'display.worker.message-handler.ts')),
    ).toBe(true);
  });

  it('extracted render module preserves compositing order comment as invariant', () => {
    // The render module must contain a comment block declaring the compositing
    // order as a non-negotiable invariant: floor -> ceiling -> walls ->
    // sprites -> pulses/sparks -> bolts.  Check the message-handler orchestrator
    // (which delegates render calls) for this declaration.
    const orchestratorPath = join(
      workerRoot,
      'display.worker.message-handler.ts',
    );
    if (!existsSync(orchestratorPath)) {
      // Orchestrator does not exist yet — fail.
      expect(existsSync(orchestratorPath)).toBe(true);
      return;
    }
    const content = readFileSync(orchestratorPath, 'utf-8');
    expect(content).toMatch(/floor.*ceiling.*walls.*sprites.*pulses?.*bolts/s);
  });
});

// ---------------------------------------------------------------------------
// S3 — re-export policy: strip all @deprecated markers
// ---------------------------------------------------------------------------

describe('B2-S3: re-export policy — strip all @deprecated markers', () => {
  it('no @deprecated markers remain in browser-entry/ source files', () => {
    const files = collectTsFiles(browserEntryRoot);
    const deprecatedFiles: string[] = [];
    for (const file of files) {
      // C3-4 explicitly requires @deprecated JSDoc on voxel-enemy.ts
      // backward-compatibility re-exports. Exclude it from this scan.
      if (file.endsWith(join('shared', 'voxel-enemy.ts'))) continue;
      const content = readFileSync(file, 'utf-8');
      if (content.includes('@deprecated')) {
        deprecatedFiles.push(relative(neatensteinRoot, file));
      }
    }
    expect(deprecatedFiles).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// S4 — consolidate duplicated helpers
// ---------------------------------------------------------------------------

describe('B2-S4: consolidated math guards in shared/math-guards.utils.ts', () => {
  it('no local clamp/clamp01/clampInt/clampByte definitions outside shared/', () => {
    const allFiles = collectTsFiles(neatensteinRoot);
    const localClampFiles: string[] = [];
    for (const file of allFiles) {
      // Skip the consolidation target itself
      if (file.includes('math-guards.utils')) continue;
      const content = readFileSync(file, 'utf-8');
      if (
        /function\s+clamp\s*\(/.test(content) ||
        /function\s+clamp01\s*\(/.test(content) ||
        /function\s+clampInt\s*\(/.test(content) ||
        /function\s+clampByte\s*\(/.test(content)
      ) {
        localClampFiles.push(relative(neatensteinRoot, file));
      }
    }
    expect(localClampFiles).toEqual([]);
  });

  it('no local isFiniteNumber definitions outside shared/', () => {
    const allFiles = collectTsFiles(neatensteinRoot);
    const localIsFiniteFiles: string[] = [];
    for (const file of allFiles) {
      if (file.includes('math-guards.utils')) continue;
      const content = readFileSync(file, 'utf-8');
      if (/function\s+isFiniteNumber\s*\(/.test(content)) {
        localIsFiniteFiles.push(relative(neatensteinRoot, file));
      }
    }
    expect(localIsFiniteFiles).toEqual([]);
  });

  it('no local isPositiveFiniteDimension definitions outside shared/', () => {
    const allFiles = collectTsFiles(neatensteinRoot);
    const localDimFiles: string[] = [];
    for (const file of allFiles) {
      if (file.includes('math-guards.utils')) continue;
      const content = readFileSync(file, 'utf-8');
      if (/function\s+isPositiveFiniteDimension\s*\(/.test(content)) {
        localDimFiles.push(relative(neatensteinRoot, file));
      }
    }
    expect(localDimFiles).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// S5 — split display.worker.test.ts into focused test files
// ---------------------------------------------------------------------------

describe('B2-S5: split display.worker.test.ts into focused test files', () => {
  it('display.worker.init.test.ts exists', () => {
    expect(existsSync(join(workerRoot, 'display.worker.init.test.ts'))).toBe(
      true,
    );
  });

  it('display.worker.sim.test.ts exists', () => {
    expect(existsSync(join(workerRoot, 'display.worker.sim.test.ts'))).toBe(
      true,
    );
  });

  it('display.worker.render.test.ts exists', () => {
    expect(existsSync(join(workerRoot, 'display.worker.render.test.ts'))).toBe(
      true,
    );
  });

  it('display.worker.eval-delegation.test.ts exists', () => {
    expect(
      existsSync(join(workerRoot, 'display.worker.eval-delegation.test.ts')),
    ).toBe(true);
  });

  it('display.worker.auto-ai.test.ts exists', () => {
    expect(existsSync(join(workerRoot, 'display.worker.auto-ai.test.ts'))).toBe(
      true,
    );
  });
});

// ---------------------------------------------------------------------------
// S6 — quick wins
// ---------------------------------------------------------------------------

describe('B2-S6: quick wins', () => {
  it('createDisplayWorkerState JSDoc summary matches the corrected wording', () => {
    const content = readFileSync(
      join(workerRoot, 'display.worker.sim.utils.ts'),
      'utf-8',
    );
    // Extract the JSDoc block immediately preceding createDisplayWorkerState
    const match = content.match(
      /\/\*\*[\s\S]*?\*\/\s*export function createDisplayWorkerState/,
    );
    expect(match).not.toBeNull();
    const jsdoc = match![0];
    expect(jsdoc).toContain(
      'Factory for the encapsulated display-worker mutable state object.',
    );
  });

  it('harness/constants.ts uses NEATENSTEIN_FIXED_TIMESTEP_MS instead of literal 16', () => {
    const content = readFileSync(join(harnessRoot, 'constants.ts'), 'utf-8');
    // The literal "16" should not appear in a division expression
    expect(content).not.toMatch(/\/\s*16\b/);
    // The constant should be used instead
    expect(content).toMatch(
      /NEATENSTEIN_FITNESS_EPISODE_DURATION_MS\s*\/\s*NEATENSTEIN_FIXED_TIMESTEP_MS/,
    );
  });

  it('applyFireGate does not mutate the input state object', async () => {
    const mod = (await import(
      /* @vite-ignore */ './browser-entry/harness/neat-io-config'
    )) as {
      createFireGateState: () => { fireActive: boolean };
      applyFireGate: (
        state: { fireActive: boolean },
        enemyVisible: number,
        rawFireOutput: number,
      ) => unknown;
    };
    const state = mod.createFireGateState();
    const originalFireActive = state.fireActive;

    // Call with high enemyVisible to trigger gate-open transition
    mod.applyFireGate(state, 0.5, 1);

    // The input state object must NOT be mutated — the function should
    // return new state instead of mutating in place.
    expect(state.fireActive).toBe(originalFireActive);
  });
});
