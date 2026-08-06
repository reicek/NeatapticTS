import { describe, expect, it } from '@jest/globals';
import type {
  BarrierState,
  CombatQualitySignal,
  EnemyEpisodeTelemetry,
  EnemyPopulation,
  EnemyVariant,
  FitnessScore,
  GenerationResult,
  HarnessConfig,
  Individual,
  MainVariant,
  PopulationConfig,
  SeedPack,
  Snapshot,
} from './types';

/**
 * Red-phase type-shape tests for examples/neatenstein/browser-entry/harness/types.ts.
 *
 * These tests import the types that the harness implementation must expose.
 * Because the source module does not exist yet, the TypeScript compiler fails
 * with module-not-found errors until the 03-harness-scaffold slice creates it.
 *
 * Coverage mapping:
 * - AC-305 (CombatQualitySignal fields) is exercised by the signal shape test.
 * - AC-304 (SeedPack) is exercised by the seed-pack shape test.
 * - AC-306 (EnemyPopulation interface) is exercised by the population shape test.
 * - AC-307 (main-agent inputs/outputs) is exercised by the variant shapes.
 * - AC-301/AC-302 (BarrierState + rolling Snapshot) is exercised by the barrier shapes.
 * - AC-308 is a lint-hygiene criterion; it is not validated by runtime tests.
 */

describe('Neatenstein harness types', () => {
  describe('AC-305: CombatQualitySignal shape', () => {
    it('accepts a CombatQualitySignal with all required fields', () => {
      const signal: CombatQualitySignal = {
        survivalTicks: 120,
        damageDealt: 45,
        kills: 2,
        damageTaken: 10,
        aimMissRate: 0.1,
        complexityBonus: 5,
        parsimonyDensityPenalty: 1,
      };
      expect(signal).toBeDefined();
    });
  });

  describe('AC-304: SeedPack shape', () => {
    it('accepts a SeedPack tied to a generation', () => {
      const pack: SeedPack = { generation: 7, seeds: [1, 2, 3] };
      expect(pack.generation).toBe(7);
    });
  });

  describe('AC-306: EnemyPopulation interface', () => {
    it('accepts an object implementing EnemyPopulation', () => {
      const population: EnemyPopulation = {
        kind: 'mlp',
        size: 32,
        sample: () => ({ weights: new Float32Array(8) }),
        snapshot: () => ({ kind: 'mlp', weights: new Float32Array(8) }),
      };
      expect(population.kind).toBe('mlp');
    });
  });

  describe('AC-307: MainVariant and EnemyVariant shapes', () => {
    it('accepts a MainVariant with id and genome', () => {
      const variant: MainVariant = {
        id: 0,
        genome: { nodes: [], connections: [] },
      };
      expect(variant.id).toBe(0);
    });

    it('accepts an EnemyVariant with id and weights', () => {
      const variant: EnemyVariant = {
        id: 1,
        weights: new Float32Array(8),
      };
      expect(variant.id).toBe(1);
    });
  });

  describe('AC-301 / AC-302: BarrierState and Snapshot shapes', () => {
    it('accepts a BarrierState with mainSnapshot and enemySnapshot', () => {
      const barrier: BarrierState = {
        generation: 1,
        mainSnapshot: { id: 0, genome: { nodes: [], connections: [] } },
        enemySnapshot: { kind: 'mlp', weights: new Float32Array(8) },
        seed: 42,
      };
      expect(barrier.seed).toBe(42);
    });

    it('accepts a Snapshot union for MLP and SWARM backends', () => {
      const mlpSnapshot: Snapshot = {
        kind: 'mlp',
        weights: new Float32Array(8),
      };
      const swarmSnapshot: Snapshot = {
        kind: 'swarm',
        dna: 'abc',
        coordinates: [],
      };
      expect({
        mlpKind: mlpSnapshot.kind,
        swarmKind: swarmSnapshot.kind,
      }).toEqual({
        mlpKind: 'mlp',
        swarmKind: 'swarm',
      });
    });
  });

  describe('AC-306: EnemyPopulation SWARM backend', () => {
    it('accepts an EnemyPopulation with kind swarm', () => {
      const population: EnemyPopulation = {
        kind: 'swarm',
        size: 8,
        sample: () => ({ dna: 'abc', coordinates: [] }),
        snapshot: () => ({ kind: 'swarm', dna: 'abc', coordinates: [] }),
      };
      expect(population.kind).toBe('swarm');
    });
  });

  describe('AC-307: Individual shape', () => {
    it('accepts an Individual carrying a MainVariant', () => {
      const individual: Individual<MainVariant> = {
        id: 0,
        variant: { id: 0, genome: { nodes: [], connections: [] } },
      };
      expect(individual.id).toBe(0);
    });
  });

  describe('AC-305: FitnessScore shape', () => {
    it('accepts a FitnessScore as a numeric value', () => {
      const score: FitnessScore = 42;
      expect(score).toBe(42);
    });
  });

  describe('AC-10.5e-002: EnemyEpisodeTelemetry shape', () => {
    it('accepts an EnemyEpisodeTelemetry with all required fields', () => {
      const telemetry: EnemyEpisodeTelemetry = {
        position: { x: 60, y: 60 },
        bfsDistances: [20, 18, 16, 14],
        damageDealt: 50,
        enemiesSurvived: 1,
        cellsVisited: 5,
        stagnationTicks: 3,
        finalDistance: 12,
      };
      expect(telemetry.position.x).toBe(60);
      expect(telemetry.bfsDistances).toHaveLength(4);
      expect(telemetry.damageDealt).toBe(50);
    });
  });

  describe('AC-306: PopulationConfig shape', () => {
    it('accepts a PopulationConfig with size and kind', () => {
      const config: PopulationConfig = { size: 32, kind: 'mlp' };
      expect(config.kind).toBe('mlp');
    });
  });

  describe('AC-301: HarnessConfig shape', () => {
    it('accepts a HarnessConfig with maxGenerations and enemy config', () => {
      const config: HarnessConfig = {
        maxGenerations: 100,
        enemy: { size: 8, kind: 'swarm' },
      };
      expect(config.maxGenerations).toBe(100);
    });
  });

  describe('AC-301: GenerationResult shape', () => {
    it('accepts a GenerationResult with generation, champion, and quality', () => {
      const result: GenerationResult = {
        generation: 5,
        champion: { id: 0, genome: { nodes: [], connections: [] } },
        quality: {
          survivalTicks: 120,
          damageDealt: 45,
          kills: 2,
          damageTaken: 10,
          aimMissRate: 0.1,
          complexityBonus: 5,
          parsimonyDensityPenalty: 1,
        },
      };
      expect(result.generation).toBe(5);
    });
  });
});
