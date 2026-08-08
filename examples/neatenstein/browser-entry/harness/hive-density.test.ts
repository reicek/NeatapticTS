import { describe, expect, it } from '@jest/globals';

import type { SwarmSnapshot } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/hive-density.ts
 * (Phase 5 Step 02).
 *
 * Covers AC-501-S02-003 and AC-501-S02-004: deterministic normalized HIVE
 * DENSITY computation with 0.25/0.50/0.75/1.0 thresholds, plus a coordinate-
 * shuffle ablation signal that proves the metric is not hardcoded.
 */

interface HiveDensityModule {
  computeHiveDensity: (snapshot: SwarmSnapshot) => number;
  HIVE_DENSITY_THRESHOLDS: number[] | Record<string, number>;
  HIVE_DENSITY_BEHAVIOR_LABELS: unknown[];
}

const hiveDensityPath = './hive-density.ts';

function makeSwarmSnapshot(): SwarmSnapshot {
  return {
    kind: 'swarm',
    dna: 'swarm:42:abcdefgh',
    coordinates: [
      { x: 0.1, y: 0.2 },
      { x: -0.3, y: 0.4 },
      { x: 0.5, y: -0.6 },
      { x: -0.7, y: -0.8 },
    ],
  };
}

describe('Neatenstein harness HIVE DENSITY', () => {
  it('exports computeHiveDensity as a function', async () => {
    const mod = (await import(hiveDensityPath)) as unknown as HiveDensityModule;
    expect(typeof mod.computeHiveDensity).toBe('function');
  });

  it('returns a normalized density in the [0, 1] range for a SWARM snapshot', async () => {
    const mod = (await import(hiveDensityPath)) as unknown as HiveDensityModule;
    const density = mod.computeHiveDensity(makeSwarmSnapshot());

    expect(density).toBeGreaterThanOrEqual(0);
    expect(density).toBeLessThanOrEqual(1);
  });

  it('returns deterministic density for the same snapshot', async () => {
    const mod = (await import(hiveDensityPath)) as unknown as HiveDensityModule;
    const snapshot = makeSwarmSnapshot();

    expect(mod.computeHiveDensity(snapshot)).toBe(
      mod.computeHiveDensity(snapshot),
    );
  });

  it('changes density under coordinate-shuffle ablation', async () => {
    const mod = (await import(hiveDensityPath)) as unknown as HiveDensityModule;
    const original = makeSwarmSnapshot();
    const shuffled: SwarmSnapshot = {
      ...original,
      coordinates: [...original.coordinates].reverse(),
    };

    expect(mod.computeHiveDensity(original)).not.toBe(
      mod.computeHiveDensity(shuffled),
    );
  });

  it('exposes density thresholds at 0.25, 0.50, 0.75, and 1.0', async () => {
    const mod = (await import(hiveDensityPath)) as unknown as HiveDensityModule;
    const thresholdValues = Array.isArray(mod.HIVE_DENSITY_THRESHOLDS)
      ? mod.HIVE_DENSITY_THRESHOLDS
      : Object.values(mod.HIVE_DENSITY_THRESHOLDS);

    expect(thresholdValues).toEqual(
      expect.arrayContaining([0.25, 0.5, 0.75, 1.0]),
    );
  });

  it('exposes a behavior label for each density threshold', async () => {
    const mod = (await import(hiveDensityPath)) as unknown as HiveDensityModule;
    const thresholdValues = Array.isArray(mod.HIVE_DENSITY_THRESHOLDS)
      ? mod.HIVE_DENSITY_THRESHOLDS
      : Object.values(mod.HIVE_DENSITY_THRESHOLDS);

    expect(Array.isArray(mod.HIVE_DENSITY_BEHAVIOR_LABELS)).toBe(true);
    expect(mod.HIVE_DENSITY_BEHAVIOR_LABELS.length).toBe(
      thresholdValues.length,
    );
  });
});
