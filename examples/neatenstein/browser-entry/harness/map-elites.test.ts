/**
 * Red-phase contract tests for Step B4 item 1 + item 10:
 * MAP-Elites Quality-Diversity archive for enemies + novelty search integration.
 *
 * Defines the contracts for a 2D behavior-descriptor grid (10x10) keyed by
 * (aggression, positioning), dominated-cell replacement, archive admission
 * via fitness OR novelty score, and archive persistence across waves.
 *
 * All tests in this file MUST fail (RED) until the B4 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import * as MapElitesModule from './map-elites';

/** Access unknown (future) exports on the map-elites module. */
const mapElites = MapElitesModule as Record<string, unknown>;

/** Fixed MLP weight count for the 6→6→4→4 topology (90 parameters). */
const MLP_WEIGHT_COUNT = 6 * 6 + 6 * 4 + 4 * 4 + 6 + 4 + 4;

/** Build a deterministic weight vector for fixture use. */
function makeWeights(seed: number): Float32Array {
  const weights = new Float32Array(MLP_WEIGHT_COUNT);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (seed + i) * 0.01;
  }
  return weights;
}

/** Minimal behavior metrics fixture in [0, 1]. */
function makeMetrics(aggression: number, positioning: number) {
  return { aggression, movementPattern: 0.5, positioning };
}

describe('B4: MAP-Elites Quality-Diversity archive', () => {
  describe('B4.1: archive creation and grid structure', () => {
    it('exports createMapElitesArchive as a function', () => {
      expect(typeof mapElites.createMapElitesArchive).toBe('function');
    });

    it('creates a 10x10 grid initialized as empty cells', () => {
      const archive = (mapElites.createMapElitesArchive as () => unknown)();
      expect(archive).toBeDefined();
      const grid = (archive as Record<string, unknown>).grid;
      expect(Array.isArray(grid)).toBe(true);
      expect((grid as unknown[][]).length).toBe(10);
      expect((grid as unknown[][])[0].length).toBe(10);
    });
  });

  describe('B4.1: archive admission via fitness', () => {
    it('exports addToMapElitesArchive as a function', () => {
      expect(typeof mapElites.addToMapElitesArchive).toBe('function');
    });

    it('admits a candidate into an empty cell by fitness', () => {
      const archive = (
        mapElites.createMapElitesArchive as () => Record<string, unknown>
      )();
      const candidate = {
        weights: makeWeights(1),
        fitness: 10,
        behaviorMetrics: makeMetrics(0.3, 0.7),
      };
      (mapElites.addToMapElitesArchive as (...args: unknown[]) => unknown)(
        archive,
        candidate,
      );
      const grid = archive.grid as unknown[][];
      const cell = grid[3][7];
      expect(cell).not.toBeNull();
      expect(cell).toBeDefined();
    });

    it('replaces a dominated cell when the new candidate has higher fitness', () => {
      const archive = (
        mapElites.createMapElitesArchive as () => Record<string, unknown>
      )();
      const weak = {
        weights: makeWeights(1),
        fitness: 5,
        behaviorMetrics: makeMetrics(0.3, 0.7),
      };
      const strong = {
        weights: makeWeights(2),
        fitness: 20,
        behaviorMetrics: makeMetrics(0.3, 0.7),
      };
      (mapElites.addToMapElitesArchive as (...args: unknown[]) => unknown)(
        archive,
        weak,
      );
      (mapElites.addToMapElitesArchive as (...args: unknown[]) => unknown)(
        archive,
        strong,
      );
      const grid = archive.grid as Record<string, unknown>[][];
      const cell = grid[3][7];
      expect(cell.fitness).toBe(20);
    });

    it('does not replace a cell when the new candidate has lower fitness', () => {
      const archive = (
        mapElites.createMapElitesArchive as () => Record<string, unknown>
      )();
      const strong = {
        weights: makeWeights(2),
        fitness: 20,
        behaviorMetrics: makeMetrics(0.3, 0.7),
      };
      const weak = {
        weights: makeWeights(1),
        fitness: 5,
        behaviorMetrics: makeMetrics(0.3, 0.7),
      };
      (mapElites.addToMapElitesArchive as (...args: unknown[]) => unknown)(
        archive,
        strong,
      );
      (mapElites.addToMapElitesArchive as (...args: unknown[]) => unknown)(
        archive,
        weak,
      );
      const grid = archive.grid as Record<string, unknown>[][];
      const cell = grid[3][7];
      expect(cell.fitness).toBe(20);
    });
  });

  describe('B4.10: novelty-based archive admission', () => {
    it('exports computeNoveltyForArchive as a function', () => {
      expect(typeof mapElites.computeNoveltyForArchive).toBe('function');
    });

    it('admits a high-novelty candidate into an empty neighboring cell', () => {
      const archive = (
        mapElites.createMapElitesArchive as () => Record<string, unknown>
      )();
      const candidate = {
        weights: makeWeights(3),
        fitness: 1,
        behaviorMetrics: makeMetrics(0.35, 0.72),
        noveltyScore: 0.9,
      };
      (mapElites.addToMapElitesArchive as (...args: unknown[]) => unknown)(
        archive,
        candidate,
      );
      const grid = archive.grid as unknown[][];
      let anyOccupied = false;
      for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
          if (grid[i][j] != null) anyOccupied = true;
        }
      }
      expect(anyOccupied).toBe(true);
    });
  });

  describe('B4.1: archive introspection', () => {
    it('exports getArchiveOccupiedCount as a function', () => {
      expect(typeof mapElites.getArchiveOccupiedCount).toBe('function');
    });

    it('returns zero for a fresh archive', () => {
      const archive = (
        mapElites.createMapElitesArchive as () => Record<string, unknown>
      )();
      const count = (
        mapElites.getArchiveOccupiedCount as (...a: unknown[]) => number
      )(archive);
      expect(count).toBe(0);
    });

    it('returns occupied count after admitting candidates', () => {
      const archive = (
        mapElites.createMapElitesArchive as () => Record<string, unknown>
      )();
      (mapElites.addToMapElitesArchive as (...a: unknown[]) => unknown)(
        archive,
        {
          weights: makeWeights(1),
          fitness: 10,
          behaviorMetrics: makeMetrics(0.2, 0.5),
        },
      );
      const count = (
        mapElites.getArchiveOccupiedCount as (...a: unknown[]) => number
      )(archive);
      expect(count).toBe(1);
    });
  });

  describe('B4.1: archive sampling', () => {
    it('exports sampleFromArchive as a function', () => {
      expect(typeof mapElites.sampleFromArchive).toBe('function');
    });

    it('returns diverse strategy samples from occupied cells', () => {
      const archive = (
        mapElites.createMapElitesArchive as () => Record<string, unknown>
      )();
      (mapElites.addToMapElitesArchive as (...a: unknown[]) => unknown)(
        archive,
        {
          weights: makeWeights(1),
          fitness: 10,
          behaviorMetrics: makeMetrics(0.2, 0.3),
        },
      );
      (mapElites.addToMapElitesArchive as (...a: unknown[]) => unknown)(
        archive,
        {
          weights: makeWeights(2),
          fitness: 15,
          behaviorMetrics: makeMetrics(0.8, 0.9),
        },
      );
      const samples = (
        mapElites.sampleFromArchive as (...a: unknown[]) => unknown[]
      )(archive, 2);
      expect(Array.isArray(samples)).toBe(true);
      expect(samples.length).toBe(2);
    });
  });
});
