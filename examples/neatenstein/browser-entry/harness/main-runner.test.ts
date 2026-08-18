import { describe, expect, it, jest } from '@jest/globals';

import type * as MainRunner from './main-runner';
import type { MlpSnapshot, SwarmSnapshot } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/main-runner.ts.
 *
 * Covers AC-307: a single main agent lifecycle runner advances a deterministic
 * generation against a frozen enemy snapshot and emits a CombatQualitySignal.
 */

interface MainRunnerModule {
  runMainGeneration: typeof MainRunner.runMainGeneration;
}

describe('Neatenstein harness main-runner', () => {
  describe('AC-307: single main agent lifecycle runner', () => {
    it('exports runMainGeneration as a function', async () => {
      const mod = (await import('./main-runner.ts')) as Record<string, unknown>;
      expect(typeof mod.runMainGeneration).toBe('function');
    });

    it('returns a CombatQualitySignal for a valid config', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const enemySnapshot: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array(8),
      };
      const result = runMainGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot,
      });
      expect({
        hasSurvivalTicks: typeof result.survivalTicks === 'number',
        hasDamageDealt: typeof result.damageDealt === 'number',
        hasKills: typeof result.kills === 'number',
        hasDamageTaken: typeof result.damageTaken === 'number',
        hasAimMissRate: typeof result.aimMissRate === 'number',
        hasComplexityBonus: typeof result.complexityBonus === 'number',
        hasParsimonyPenalty: typeof result.parsimonyDensityPenalty === 'number',
      }).toEqual({
        hasSurvivalTicks: true,
        hasDamageDealt: true,
        hasKills: true,
        hasDamageTaken: true,
        hasAimMissRate: true,
        hasComplexityBonus: true,
        hasParsimonyPenalty: true,
      });
    });

    it('produces deterministic output for the same config', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const config = {
        seed: 7,
        generation: 2,
        enemySnapshot: {
          kind: 'mlp',
          weights: new Float32Array(8),
        } as const satisfies MlpSnapshot,
      };
      const first = runMainGeneration(config);
      const second = runMainGeneration(config);
      expect(first).toEqual(second);
    });

    it('falls back to a generated MLP enemy snapshot when none is supplied', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 3, generation: 1 });
      expect(typeof result.survivalTicks).toBe('number');
    });

    it('falls back to a generated swarm enemy snapshot when enemy.kind is swarm', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({
        seed: 3,
        generation: 1,
        enemy: { kind: 'swarm' },
      });
      expect(typeof result.survivalTicks).toBe('number');
    });

    it('refreshes the fallback MLP enemy snapshot on refresh generations', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 3, generation: 5 });
      expect(typeof result.survivalTicks).toBe('number');
    });

    it('produces deterministic output for a swarm enemy snapshot', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const enemySnapshot: SwarmSnapshot = {
        kind: 'swarm',
        dna: 'swarm-dna',
        coordinates: [{ x: 0.1, y: 0.2 }],
      };
      const config = {
        seed: 5,
        generation: 1,
        enemySnapshot,
      };
      const first = runMainGeneration(config);
      const second = runMainGeneration(config);
      expect(first).toEqual(second);
    });
  });

  describe('AC-401-S02-001: real NGE main-agent genomes', () => {
    it('returns a champion genome instead of a placeholder genome', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 42, generation: 0 });
      expect(
        (result as unknown as Record<string, unknown>).championGenome,
      ).toBeDefined();
    });

    it('materializes NGE archetypes in the champion genome', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 42, generation: 0 });
      const extendedResult = result as unknown as Record<string, unknown>;
      const genome = extendedResult.championGenome as
        { archetypes?: unknown[] } | undefined;
      expect(genome?.archetypes?.length).toBeGreaterThan(0);
    });
  });

  describe('AC-401-S02-004: fitness against MLP snapshot, not live MLP', () => {
    it('exposes the evaluated enemy snapshot in the generation result', async () => {
      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const enemySnapshot = {
        kind: 'mlp',
        weights: new Float32Array(8),
      } as const;
      const result = runMainGeneration({
        seed: 42,
        generation: 1,
        enemySnapshot,
      });
      expect(
        (result as unknown as Record<string, unknown>).evaluatedEnemySnapshot,
      ).toEqual(enemySnapshot);
    });
  });

  describe('P5S1: fire gate wired through fitness evaluation', () => {
    beforeEach(() => {
      jest.resetModules();
    });

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('suppresses fire when no enemy is visible', async () => {
      const networkModule =
        (await import('../../../../src/architecture/network/network')) as unknown as {
          default: { prototype: { activate: jest.Mock } };
        };
      jest
        .spyOn(networkModule.default.prototype, 'activate')
        .mockReturnValue([0, 0, 0, 1, 0]);

      const enemyNavModule =
        (await import('../shared/enemy-navigation')) as unknown as {
          extractSensors: jest.Mock;
        };
      jest
        .spyOn(enemyNavModule, 'extractSensors')
        .mockReturnValue(new Array(15).fill(0));

      const { runEpisode } = (await import('./main-runner.ts')) as {
        runEpisode: typeof MainRunner.runEpisode;
      };
      const signal = runEpisode(
        { id: 0, genome: { nodes: [], connections: [] } },
        { kind: 'mlp', weights: new Float32Array(8) },
        99,
      );
      expect(signal.shotsFired).toBe(0);
    });

    it('allows fire when an enemy is visible', async () => {
      const networkModule =
        (await import('../../../../src/architecture/network/network')) as unknown as {
          default: { prototype: { activate: jest.Mock } };
        };
      jest
        .spyOn(networkModule.default.prototype, 'activate')
        .mockReturnValue([0, 0, 0, 1, 0]);

      const enemyNavModule =
        (await import('../shared/enemy-navigation')) as unknown as {
          extractSensors: jest.Mock;
        };
      const sensors = new Array(15).fill(0);
      sensors[12] = 1;
      jest.spyOn(enemyNavModule, 'extractSensors').mockReturnValue(sensors);

      const { runEpisode } = (await import('./main-runner.ts')) as {
        runEpisode: typeof MainRunner.runEpisode;
      };
      const signal = runEpisode(
        { id: 0, genome: { nodes: [], connections: [] } },
        { kind: 'mlp', weights: new Float32Array(8) },
        99,
      );
      expect(signal.shotsFired ?? 0).toBeGreaterThan(0);
    });

    it('falls back to zero when the enemy-visible sensor is undefined', async () => {
      const networkModule =
        (await import('../../../../src/architecture/network/network')) as unknown as {
          default: { prototype: { activate: jest.Mock } };
        };
      jest
        .spyOn(networkModule.default.prototype, 'activate')
        .mockReturnValue([0, 0, 0, 1, 0]);

      const enemyNavModule =
        (await import('../shared/enemy-navigation')) as unknown as {
          extractSensors: jest.Mock;
        };
      const sensors = new Array(15).fill(0);
      sensors[12] = undefined;
      jest.spyOn(enemyNavModule, 'extractSensors').mockReturnValue(sensors);

      const { runEpisode } = (await import('./main-runner.ts')) as {
        runEpisode: typeof MainRunner.runEpisode;
      };
      const signal = runEpisode(
        { id: 0, genome: { nodes: [], connections: [] } },
        { kind: 'mlp', weights: new Float32Array(8) },
        99,
      );
      expect(signal.shotsFired).toBe(0);
    });
  });

  describe('P8S1-coverage-closure: main-runner edge branches', () => {
    beforeEach(() => {
      jest.resetModules();
    });

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('pads network outputs shorter than the expected output count', async () => {
      const networkModule =
        (await import('../../../../src/architecture/network/network')) as unknown as {
          default: { prototype: { activate: jest.Mock } };
        };
      const spy = jest
        .spyOn(networkModule.default.prototype, 'activate')
        .mockReturnValue([0.1, 0.2]);

      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 20, generation: 1 });
      expect(typeof result.survivalTicks).toBe('number');
      spy.mockRestore();
    });

    it('falls back to zero outputs when network activation returns a non-array', async () => {
      const networkModule =
        (await import('../../../../src/architecture/network/network')) as unknown as {
          default: { prototype: { activate: jest.Mock } };
        };
      const spy = jest
        .spyOn(networkModule.default.prototype, 'activate')
        .mockReturnValue(undefined as unknown as number[]);

      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 21, generation: 1 });
      expect(typeof result.survivalTicks).toBe('number');
      spy.mockRestore();
    });

    it('sanitises non-finite network outputs to zero', async () => {
      const networkModule =
        (await import('../../../../src/architecture/network/network')) as unknown as {
          default: { prototype: { activate: jest.Mock } };
        };
      const spy = jest
        .spyOn(networkModule.default.prototype, 'activate')
        .mockReturnValue([NaN, Infinity, -Infinity, 0.5, 0.5]);

      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 22, generation: 1 });
      expect(typeof result.survivalTicks).toBe('number');
      spy.mockRestore();
    });

    it('uses default telemetry when the episode ends without telemetry', async () => {
      const episodeModule = await import('../host/game/episode');
      const spy = jest.spyOn(episodeModule, 'endEpisode').mockImplementation(
        (state) =>
          ({
            ...(state as unknown as Record<string, unknown>),
            telemetry: undefined,
          }) as ReturnType<typeof episodeModule.endEpisode>,
      );

      const { runMainGeneration } =
        (await import('./main-runner.ts')) as MainRunnerModule;
      const result = runMainGeneration({ seed: 23, generation: 1 });
      expect(typeof result.survivalTicks).toBe('number');
      expect(typeof result.damageDealt).toBe('number');
      spy.mockRestore();
    });
  });
});
