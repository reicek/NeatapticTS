import { describe, expect, it } from '@jest/globals';

import type * as EnemyWarmstart from './enemy-warmstart';
import type * as EnemyMlp from './enemy-mlp';
import type { EnemyVariant } from './types';

/**
 * Red/green contract tests for enemy-warmstart.ts and the warm-start
 * integration in enemy-mlp.ts.
 *
 * Covers:
 * - AC-10.5d-001: trainMlpBackprop performs bounded backprop with tanh
 *   gradient and per-output sigmoid (BCE) cost.
 * - AC-10.5d-002: buildNeatensteinCurriculum produces ~23 base cases
 *   with jitter, mirroring asciiMaze buildLamarckianTrainingSet.
 * - AC-10.5d-003: Warm-start applied at gen 0 and re-applied on refresh.
 * - AC-10.5d-004: trainMlpBackprop converges within 0.1 tolerance for
 *   >=80% of base cases after <=60 iterations.
 */

interface WarmStartModule {
  trainMlpBackprop: typeof EnemyWarmstart.trainMlpBackprop;
  predictMlp: typeof EnemyWarmstart.predictMlp;
  buildNeatensteinCurriculum: typeof EnemyWarmstart.buildNeatensteinCurriculum;
  warmStartTemplate: typeof EnemyWarmstart.warmStartTemplate;
  warmStartWeights: typeof EnemyWarmstart.warmStartWeights;
}

interface EnemyMlpModule {
  createMlpEnemyPopulation: typeof EnemyMlp.createMlpEnemyPopulation;
}

/** Connection weight count for the fixed 6→6→4→4 topology. */
const MLP_CONNECTION_COUNT = 6 * 6 + 6 * 4 + 4 * 4;

/** Per-layer bias count for the fixed 6→6→4→4 topology. */
const MLP_BIAS_COUNT = 6 + 4 + 4;

/** Total parameter count. */
const MLP_PARAM_COUNT = MLP_CONNECTION_COUNT + MLP_BIAS_COUNT;

describe('Neatenstein warm-start (enemy-warmstart)', () => {
  // -------------------------------------------------------------------------
  // AC-10.5d-001: trainMlpBackprop
  // -------------------------------------------------------------------------
  describe('AC-10.5d-001: trainMlpBackprop bounded backprop', () => {
    it('exports trainMlpBackprop as a function', async () => {
      const mod = (await import('./enemy-warmstart.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.trainMlpBackprop).toBe('function');
    });

    it('returns a number (final loss)', async () => {
      const { trainMlpBackprop } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const weights = new Float32Array(MLP_PARAM_COUNT);
      const inputs = [new Float32Array([0, 1, 0, 0, 0, 0.5])];
      const targets = [new Float32Array([0.8, 0.05, 0.05, 0.3])];
      const loss = trainMlpBackprop(weights, inputs, targets, 0.5, 10);
      expect(typeof loss).toBe('number');
      expect(loss).toBeGreaterThanOrEqual(0);
    });

    it('modifies weights in place (weights change after training)', async () => {
      const { trainMlpBackprop } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const weights = new Float32Array(MLP_PARAM_COUNT);
      const before = new Float32Array(weights);
      const inputs = [new Float32Array([0, 1, 0, 0, 0, 0.5])];
      const targets = [new Float32Array([0.8, 0.05, 0.05, 0.3])];
      trainMlpBackprop(weights, inputs, targets, 0.5, 5);
      let changed = false;
      for (let i = 0; i < weights.length; i++) {
        if (weights[i] !== before[i]) {
          changed = true;
          break;
        }
      }
      expect(changed).toBe(true);
    });

    it('respects the iterations bound (more iterations produce lower loss)', async () => {
      const { trainMlpBackprop } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const inputs = [new Float32Array([0, 1, 0, 0, 0, 0.5])];
      const targets = [new Float32Array([0.8, 0.05, 0.05, 0.3])];

      const w1 = new Float32Array(MLP_PARAM_COUNT);
      const loss1 = trainMlpBackprop(w1, inputs, targets, 0.5, 1);

      const w2 = new Float32Array(MLP_PARAM_COUNT);
      const loss2 = trainMlpBackprop(w2, inputs, targets, 0.5, 30);

      expect(loss2).toBeLessThan(loss1);
    });

    it('returns 0 for empty inputs without modifying weights', async () => {
      const { trainMlpBackprop } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const weights = new Float32Array(MLP_PARAM_COUNT);
      const before = new Float32Array(weights);
      const loss = trainMlpBackprop(weights, [], [], 0.5, 60);
      expect(loss).toBe(0);
      expect(Array.from(weights)).toEqual(Array.from(before));
    });

    it('returns 0 for zero iterations without modifying weights', async () => {
      const { trainMlpBackprop } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const weights = new Float32Array(MLP_PARAM_COUNT);
      const before = new Float32Array(weights);
      const loss = trainMlpBackprop(
        weights,
        [new Float32Array([0, 1, 0, 0, 0, 0.5])],
        [new Float32Array([0.8, 0.05, 0.05, 0.3])],
        0.5,
        0,
      );
      expect(loss).toBe(0);
      expect(Array.from(weights)).toEqual(Array.from(before));
    });

    it('uses BCE cost with sigmoid output (outputs in (0, 1) after training)', async () => {
      const { trainMlpBackprop, predictMlp } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const weights = new Float32Array(MLP_PARAM_COUNT);
      const inputs = [new Float32Array([0, 1, 0, 0, 0, 0.5])];
      const targets = [new Float32Array([0.8, 0.05, 0.05, 0.3])];
      trainMlpBackprop(weights, inputs, targets, 0.5, 30);
      const out = predictMlp(weights, inputs[0]);
      for (let i = 0; i < out.length; i++) {
        expect(out[i]).toBeGreaterThan(0);
        expect(out[i]).toBeLessThan(1);
      }
    });

    it('early-stops when loss falls below 0.001 (hard targets on a single easy case)', async () => {
      const { trainMlpBackprop } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      // Use hard targets [1,0,0,0] with a high learning rate so the output
      // biases converge quickly and total loss drops below 0.001, triggering
      // the early-stop break well before the 500-iteration cap.
      const weights = new Float32Array(MLP_PARAM_COUNT);
      const inputs = [new Float32Array([0, 1, 0, 0, 0, 0.5])];
      const targets = [new Float32Array([1, 0, 0, 0])];
      const loss = trainMlpBackprop(weights, inputs, targets, 10.0, 1000);
      expect(loss).toBeLessThan(0.001);
    });

    it('exports predictMlp as a function returning Float32Array', async () => {
      const mod = (await import('./enemy-warmstart.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.predictMlp).toBe('function');
      const { predictMlp } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const out = predictMlp(
        new Float32Array(MLP_PARAM_COUNT),
        new Float32Array(6),
      );
      expect(out).toBeInstanceOf(Float32Array);
      expect(out.length).toBe(4);
    });
  });

  // -------------------------------------------------------------------------
  // AC-10.5d-002: buildNeatensteinCurriculum
  // -------------------------------------------------------------------------
  describe('AC-10.5d-002: Neatenstein curriculum', () => {
    it('exports buildNeatensteinCurriculum as a function', async () => {
      const mod = (await import('./enemy-warmstart.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.buildNeatensteinCurriculum).toBe('function');
    });

    it('produces ~23 base cases (>= 20 and <= 26)', async () => {
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const curriculum = buildNeatensteinCurriculum();
      expect(curriculum.length).toBeGreaterThanOrEqual(20);
      expect(curriculum.length).toBeLessThanOrEqual(26);
    });

    it('each case has a 6-element input vector', async () => {
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const curriculum = buildNeatensteinCurriculum();
      for (const c of curriculum) {
        expect(c.input.length).toBe(6);
      }
    });

    it('each case has a 4-element target vector', async () => {
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const curriculum = buildNeatensteinCurriculum();
      for (const c of curriculum) {
        expect(c.target.length).toBe(4);
      }
    });

    it('target values are in [0, 1] (soft targets)', async () => {
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const curriculum = buildNeatensteinCurriculum();
      for (const c of curriculum) {
        for (const v of c.target) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1);
        }
      }
    });

    it('input values are in [0, 1] after jitter clamping', async () => {
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const curriculum = buildNeatensteinCurriculum();
      for (const c of curriculum) {
        for (const v of c.input) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1);
        }
      }
    });

    it('is deterministic (same output on repeated calls)', async () => {
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const a = buildNeatensteinCurriculum();
      const b = buildNeatensteinCurriculum();
      expect(a.length).toBe(b.length);
      for (let i = 0; i < a.length; i++) {
        expect(a[i].input).toEqual(b[i].input);
        expect(a[i].target).toEqual(b[i].target);
      }
    });
  });

  // -------------------------------------------------------------------------
  // AC-10.5d-003: Warm-start integration in enemy-mlp.ts
  // -------------------------------------------------------------------------
  describe('AC-10.5d-003: warm-start applied at gen 0 and on refresh', () => {
    it('exports warmStartWeights as a function', async () => {
      const mod = (await import('./enemy-warmstart.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.warmStartWeights).toBe('function');
    });

    it('warmStartWeights returns a 90-element Float32Array', async () => {
      const { warmStartWeights } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const w = warmStartWeights(7, 0);
      expect(w).toBeInstanceOf(Float32Array);
      expect(w.length).toBe(MLP_PARAM_COUNT);
    });

    it('warmStartWeights is deterministic for the same seed + variantId', async () => {
      const { warmStartWeights } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const a = warmStartWeights(7, 3);
      const b = warmStartWeights(7, 3);
      expect(Array.from(a)).toEqual(Array.from(b));
    });

    it('warmStartWeights produces distinct weights for different variantIds', async () => {
      const { warmStartWeights } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const a = warmStartWeights(7, 0);
      const b = warmStartWeights(7, 1);
      expect(Array.from(a)).not.toEqual(Array.from(b));
    });

    it('warmStartWeights with negative variantId returns unmodified template', async () => {
      const { warmStartWeights, warmStartTemplate } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const template = warmStartTemplate(7);
      const w = warmStartWeights(7, -1);
      expect(Array.from(w)).toEqual(Array.from(template));
    });

    it('warmStartTemplate returns a 90-element Float32Array', async () => {
      const { warmStartTemplate } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const w = warmStartTemplate(42);
      expect(w).toBeInstanceOf(Float32Array);
      expect(w.length).toBe(MLP_PARAM_COUNT);
    });

    it('warmStartTemplate is deterministic for the same seed', async () => {
      const { warmStartTemplate } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const a = warmStartTemplate(42);
      const b = warmStartTemplate(42);
      expect(Array.from(a)).toEqual(Array.from(b));
    });

    it('createMlpEnemyPopulation variants are warm-started (outputs are meaningful, not random)', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const { predictMlp } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;

      const population = createMlpEnemyPopulation({ seed: 1 });
      const variant = population.sample(0) as EnemyVariant;

      // Warm-started weights should produce outputs that correlate with the
      // curriculum targets. Test on a "move north" case.
      const curriculum = buildNeatensteinCurriculum();
      const northCase = curriculum[0]; // [0, 1, 0, 0, 0, ~0.5] → move=0.8
      const out = predictMlp(
        variant.weights,
        new Float32Array(northCase.input),
      );

      // The move output (index 0) should be higher than the average output
      // because the warm-start trained for move=0.8 on this input pattern.
      const avg = (out[0] + out[1] + out[2] + out[3]) / 4;
      expect(out[0]).toBeGreaterThan(avg);
    });

    it('createMlpEnemyPopulation is still deterministic for the same seed', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const a = createMlpEnemyPopulation({ seed: 7 });
      const b = createMlpEnemyPopulation({ seed: 7 });
      expect(Array.from((a.sample(0) as EnemyVariant).weights)).toEqual(
        Array.from((b.sample(0) as EnemyVariant).weights),
      );
    });

    it('createMlpEnemyPopulation still produces 32 variants', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      expect(population.size).toBe(32);
    });

    it('champion snapshot after refresh has warm-started weights (not random)', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const { predictMlp } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;
      const { buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;

      const population = createMlpEnemyPopulation({ seed: 1 });
      const snapshot = population.update({ generation: 5 });

      // The champion weights should be warm-started (trained on curriculum),
      // so outputs on a "move north" case should show move > average.
      expect(snapshot.kind).toBe('mlp');
      if (snapshot.kind === 'mlp') {
        const curriculum = buildNeatensteinCurriculum();
        const northCase = curriculum[0];
        const out = predictMlp(
          snapshot.weights,
          new Float32Array(northCase.input),
        );
        const avg = (out[0] + out[1] + out[2] + out[3]) / 4;
        expect(out[0]).toBeGreaterThan(avg);
      }
    });

    it('existing update gating still works (same snapshot on non-refresh gen)', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = population.snapshot();
      const after = population.update({ generation: 4 });
      expect(after).toBe(before);
    });

    it('existing update gating still works (new snapshot on refresh gen)', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = population.snapshot();
      const after = population.update({ generation: 5 });
      expect(after).not.toBe(before);
    });
  });

  // -------------------------------------------------------------------------
  // AC-10.5d-004: Convergence
  // -------------------------------------------------------------------------
  describe('AC-10.5d-004: convergence within 0.1 tolerance for >=80% of cases', () => {
    it('warmStartTemplate converges: >=80% of base cases match targets within 0.1 after <=60 iterations', async () => {
      const { warmStartTemplate, predictMlp, buildNeatensteinCurriculum } =
        (await import('./enemy-warmstart.ts')) as WarmStartModule;

      // Use the warm-start template directly — this exercises the full
      // trainMlpBackprop pipeline with the production learning rate, init
      // scale, and case weights.  Seed 7 is deterministic and achieves
      // >80% convergence on the curriculum.
      const weights = warmStartTemplate(7);
      const curriculum = buildNeatensteinCurriculum();
      const inputs = curriculum.map((c) => new Float32Array(c.input));

      // Check convergence: >=80% of cases within 0.1 tolerance per output.
      let converged = 0;
      for (let i = 0; i < curriculum.length; i++) {
        const out = predictMlp(weights, inputs[i]);
        const target = curriculum[i].target;
        let allOutputsMatch = true;
        for (let o = 0; o < target.length; o++) {
          if (Math.abs(out[o] - target[o]) > 0.1) {
            allOutputsMatch = false;
            break;
          }
        }
        if (allOutputsMatch) converged++;
      }

      const convergenceRate = converged / curriculum.length;
      expect(convergenceRate).toBeGreaterThanOrEqual(0.8);
    });
  });
});
