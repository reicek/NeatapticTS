/**
 * C3 RED-phase tests: Constants and Types Cleanup.
 *
 * These tests encode the four solution items from the Neatenstein Ultimate
 * Quality Upgrade plan (Step C3) as failing contracts. Each describe block
 * maps to one solution item. Tests are designed to fail for the right
 * reason: the cleanup refactoring has not been performed yet.
 *
 * Structural tests inspect the filesystem and source content directly,
 * mirroring the pattern established by `b2-architecture-red.test.ts`.
 *
 * Coverage mapping:
 * - C3-1: magic number `16` timestep references the canonical constant.
 * - C3-2: voxel anatomy literals promoted to named constants.
 * - C3-3: tombstone stub files removed; notes moved to replacement JSDoc.
 * - C3-4: deprecated re-export JSDoc carries `@example` migration code.
 */

import { describe, expect, it } from '@jest/globals';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

const neatensteinRoot = join(process.cwd(), 'examples', 'neatenstein');
const browserEntryRoot = join(neatensteinRoot, 'browser-entry');
const scriptsRoot = join(neatensteinRoot, 'scripts');
const rendererRoot = join(browserEntryRoot, 'renderer');
const sharedRoot = join(browserEntryRoot, 'shared');

// ---------------------------------------------------------------------------
// C3-1 — Fix magic number 16
// ---------------------------------------------------------------------------

describe('C3-1: magic number 16 timestep references canonical constant', () => {
  it('REFERENCE_TIMESTEP_MS is defined in terms of NEATENSTEIN_FIXED_TIMESTEP_MS, not a bare 16 literal', () => {
    const content = readFileSync(join(browserEntryRoot, 'constants.ts'), 'utf-8');
    const defLine = content
      .split('\n')
      .find((line) => /REFERENCE_TIMESTEP_MS\s*=/.test(line));
    expect(defLine).toBeDefined();
    // The definition must reference the canonical fixed-timestep constant
    // instead of redefining the magic number 16.
    expect(defLine).toMatch(/NEATENSTEIN_FIXED_TIMESTEP_MS/);
    expect(defLine).not.toMatch(/=\s*16\s*;?\s*$/);
  });

  it('REFERENCE_TIMESTEP_MS equals the canonical NEATENSTEIN_FIXED_TIMESTEP_MS value', async () => {
    const { REFERENCE_TIMESTEP_MS } = await import(
      join(browserEntryRoot, 'constants.ts')
    );
    const { NEATENSTEIN_FIXED_TIMESTEP_MS } = await import(
      join(neatensteinRoot, 'browser-entry', 'host', 'game', 'constants.ts')
    );
    expect(REFERENCE_TIMESTEP_MS).toBe(NEATENSTEIN_FIXED_TIMESTEP_MS);
  });
});

// ---------------------------------------------------------------------------
// C3-2 — Promote voxel anatomy literals
// ---------------------------------------------------------------------------

describe('C3-2: voxel anatomy literals promoted to named constants', () => {
  it('voxel-enemy.ts does not inline anatomy magic-number local consts', () => {
    const content = readFileSync(join(sharedRoot, 'voxel-enemy.ts'), 'utf-8');
    // These inline numeric local consts currently live inside the build*
    // functions. After promotion they must be imported from the constants
    // module instead of being redefined as bare literals.
    expect(content).not.toMatch(
      /const\s+(legWidth|legHeight|legDepth|torsoYMin|armYMin|cannonYMin|headYMin|diskRadius)\s*=\s*\d+/,
    );
  });

  it('voxel-enemy.constants.ts exports named anatomy constants by body region', () => {
    const content = readFileSync(
      join(sharedRoot, 'voxel-enemy.constants.ts'),
      'utf-8',
    );
    // After promotion, anatomy constants named by body region should be
    // exported alongside the existing palette/grid constants.
    expect(content).toMatch(
      /export\s+const\s+(LEG|TORSO|ARM|HEAD|CANNON|DISK|KNEE)_/,
    );
  });
});

// ---------------------------------------------------------------------------
// C3-3 — Remove tombstone files
// ---------------------------------------------------------------------------

describe('C3-3: tombstone stub files removed', () => {
  it('scripts/voxel-gun.ts is removed', () => {
    expect(existsSync(join(scriptsRoot, 'voxel-gun.ts'))).toBe(false);
  });

  it('renderer/gun-sprite.ts is removed', () => {
    expect(existsSync(join(rendererRoot, 'gun-sprite.ts'))).toBe(false);
  });

  it('gun-sprite-decode.ts JSDoc carries the tombstone note for the removed voxel gun projector', () => {
    const content = readFileSync(
      join(rendererRoot, 'gun-sprite-decode.ts'),
      'utf-8',
    );
    // The tombstone notes from the removed stub files must be relocated into
    // the replacement module's JSDoc, naming the retired APIs.
    expect(content).toMatch(/projectVoxelGunSprite|ProjectedGunVoxel|buildVoxelGun/);
    expect(content).toMatch(/@deprecated|removed|tombstone/i);
  });
});

// ---------------------------------------------------------------------------
// C3-4 — Clean up @deprecated re-export JSDoc
// ---------------------------------------------------------------------------

describe('C3-4: deprecated re-export JSDoc includes migration example', () => {
  it('voxel-enemy.ts compatibility re-exports carry @deprecated and @example JSDoc', () => {
    const content = readFileSync(join(sharedRoot, 'voxel-enemy.ts'), 'utf-8');
    // The backward-compat re-export block (export ... from './voxel-enemy.types'
    // and export ... from './voxel-enemy.constants') exists so legacy imports
    // keep working. It must be marked @deprecated and include an @example
    // migration snippet pointing consumers at the extracted modules.
    expect(content).toMatch(
      /\/\*{2,}[\s\S]*?@deprecated[\s\S]*?@example[\s\S]*?\*\/\s*export\s+(?:type\s+)?\{[^}]*\}\s+from\s+['"]\.\/voxel-enemy\.(?:types|constants)['"]/,
    );
  });
});