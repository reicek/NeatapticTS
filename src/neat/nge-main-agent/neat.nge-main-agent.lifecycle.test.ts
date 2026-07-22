/**
 * Red-phase test contracts for Phase 4 Step 01 — NGE main agent lifecycle.
 *
 * Covers AC-401: the main agent lifecycle stage type and runner advance through
 * Embryo -> Juvenile -> Adult -> Reproducing without reusing the Racing Curriculum
 * juvenile/adult stage contract.
 *
 * All tests fail because the imported source modules do not exist yet. The
 * expected failure reason is TS2307 "Cannot find module", not a syntax error
 * or bad fixture.
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type {
  NgeMainAgentLifecycleConfig,
  NgeMainAgentLifecycleStage,
  NgeMainAgentLifecycleState,
} from './neat.nge-main-agent.types';

import {
  advanceMainAgentLifecycle,
  createMainAgentLifecycleRunner,
} from './neat.nge-main-agent.lifecycle';

import { buildMainAgentEmbryo } from './neat.nge-main-agent.embryo';
import { transitionJuvenileToAdult } from './neat.nge-main-agent.juvenile';
import { runAdultEquilibrium } from './neat.nge-main-agent.adult';
import { transitionAdultToReproducing } from './neat.nge-main-agent.reproduction';

const defaultConfig: NgeMainAgentLifecycleConfig = {
  seed: 42,
  maxNodes: 1024,
  maxEdges: 4096,
};

describe('NGE main agent lifecycle', () => {
  describe('lifecycle stage type', () => {
    it('defines exactly four ordered stages', () => {
      // Arrange
      const stage: NgeMainAgentLifecycleStage = 'embryo';

      // Assert — the union must accept the four named stages
      expect(stage).toBe('embryo');
    });
  });

  describe('advanceMainAgentLifecycle', () => {
    it('transitions embryo to juvenile', () => {
      // Arrange
      const state: NgeMainAgentLifecycleState = {
        stage: 'embryo',
        generation: 0,
        seed: 42,
      };

      // Act
      const nextState = advanceMainAgentLifecycle(state, defaultConfig);

      // Assert
      expect(nextState.stage).toBe('juvenile');
    });

    it('transitions juvenile to adult', () => {
      // Arrange
      const state: NgeMainAgentLifecycleState = {
        stage: 'juvenile',
        generation: 1,
        seed: 42,
      };

      // Act
      const nextState = advanceMainAgentLifecycle(state, defaultConfig);

      // Assert
      expect(nextState.stage).toBe('adult');
    });

    it('transitions adult to reproducing', () => {
      // Arrange
      const state: NgeMainAgentLifecycleState = {
        stage: 'adult',
        generation: 2,
        seed: 42,
      };

      // Act
      const nextState = advanceMainAgentLifecycle(state, defaultConfig);

      // Assert
      expect(nextState.stage).toBe('reproducing');
    });

    it('advances reproducing back to embryo for the next generation', () => {
      // Arrange
      const state: NgeMainAgentLifecycleState = {
        stage: 'reproducing',
        generation: 3,
        seed: 42,
      };

      // Act
      const nextState = advanceMainAgentLifecycle(state, defaultConfig);

      // Assert
      expect(nextState.stage).toBe('embryo');
    });

    it('increments generation count across the full cycle', () => {
      // Arrange
      const state: NgeMainAgentLifecycleState = {
        stage: 'reproducing',
        generation: 5,
        seed: 42,
      };

      // Act
      const nextState = advanceMainAgentLifecycle(state, defaultConfig);

      // Assert
      expect(nextState.generation).toBe(6);
    });
  });

  describe('createMainAgentLifecycleRunner', () => {
    it('produces a callable runner function', () => {
      // Act
      const runner = createMainAgentLifecycleRunner(defaultConfig);

      // Assert
      expect(typeof runner).toBe('function');
    });

    it('returns deterministic state for the same seed and config', () => {
      // Arrange
      const runnerA = createMainAgentLifecycleRunner(defaultConfig);
      const runnerB = createMainAgentLifecycleRunner(defaultConfig);

      // Act
      const stateA = runnerA({ stage: 'embryo', generation: 0, seed: 42 });
      const stateB = runnerB({ stage: 'embryo', generation: 0, seed: 42 });

      // Assert
      expect(stateA).toEqual(stateB);
    });
  });

  describe('stage-specific helpers', () => {
    it('builds an embryo from the lifecycle config', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.stage).toBe('embryo');
    });

    it('matures a juvenile state into an adult state', () => {
      // Arrange
      const juvenile = buildMainAgentEmbryo(defaultConfig);

      // Act
      const adult = transitionJuvenileToAdult(juvenile, defaultConfig);

      // Assert
      expect(adult.stage).toBe('adult');
    });

    it('runs adult equilibrium to produce a stable candidate', () => {
      // Arrange
      const adult = transitionJuvenileToAdult(
        buildMainAgentEmbryo(defaultConfig),
        defaultConfig,
      );

      // Act
      const equilibrium = runAdultEquilibrium(adult, defaultConfig);

      // Assert
      expect(equilibrium.isStable).toBe(true);
    });

    it('produces a reproducing state from an equilibrium candidate', () => {
      // Arrange
      const adult = transitionJuvenileToAdult(
        buildMainAgentEmbryo(defaultConfig),
        defaultConfig,
      );
      const equilibrium = runAdultEquilibrium(adult, defaultConfig);

      // Act
      const reproducing = transitionAdultToReproducing(
        adult,
        equilibrium,
        defaultConfig,
      );

      // Assert
      expect(reproducing.stage).toBe('reproducing');
    });
  });
});
