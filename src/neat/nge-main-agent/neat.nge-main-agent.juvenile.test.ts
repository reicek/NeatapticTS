/**
 * Green-phase coverage tests for slice-fix `04-main-juvenile`.
 *
 * Closes coverage gaps on `evaluateJuvenileGrowGate` and
 * `growJuvenileTopologyWithInternalAssimilation` while keeping the juvenile
 * topology budget and weak internal-assimilation contracts intact.
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type { EquilibriumCandidate } from '../nge-adult/neat.nge-adult.types';
import type { NgeAssimilationCandidate } from '../nge-assimilation/neat.nge-assimilation.types';
import { DEFAULT_ASSIMILATION_WRITE_BACK_RATE } from '../nge-assimilation/neat.nge-assimilation.constants';

import type {
  NgeMainAgentEmbryo,
  NgeMainAgentLifecycleConfig,
} from './neat.nge-main-agent.types';

import {
  evaluateJuvenileGrowGate,
  growJuvenileTopologyWithInternalAssimilation,
} from './neat.nge-main-agent.juvenile';

const defaultConfig: NgeMainAgentLifecycleConfig = {
  seed: 42,
  maxNodes: 1024,
  maxEdges: 4096,
};

const JUVENILE_ASSIMILATION_DECAY = 0.9;

function createBaseEmbryo(
  overrides?: Partial<NgeMainAgentEmbryo>,
): NgeMainAgentEmbryo {
  return {
    stage: 'embryo',
    generation: 0,
    seed: defaultConfig.seed,
    nodeCount: 10,
    edgeCount: 20,
    archetypes: [
      {
        archetypeId: 'arch:juvenile-test',
        computationType: 'AttentionHead',
        coordinate: [0.1, 0.2, 0.3],
        zoneId: 'z:0:0:0',
      },
    ],
    schemaVersion: 'A.1.0',
    reproductionMode: 'parthenogenesis',
    modeIsEvolvable: true,
    ...overrides,
  };
}

function createEquilibriumCandidate(
  overrides?: Partial<EquilibriumCandidate>,
): EquilibriumCandidate {
  return {
    zoneId: 'zone:main-1',
    isGainStable: true,
    isPlateau: true,
    ...overrides,
  };
}

function createAssimilationCandidate(
  overrides?: Partial<NgeAssimilationCandidate>,
): NgeAssimilationCandidate {
  return {
    equilibriumCandidate: createEquilibriumCandidate(),
    sourceDnaFingerprint: 'fp:main-agent-equilibrium',
    sourceSchemaVersion: 'A.1.0',
    moduleDelta: {
      moduleId: 'module:main-1',
      zoneId: 'zone:main-1',
      ruleParameters: {
        connectionDensity: {
          currentValue: 0.2,
          targetValue: 0.8,
        },
      },
    },
    ...overrides,
  };
}

describe('evaluateJuvenileGrowGate', () => {
  describe('gate open conditions', () => {
    it('opens at generation 0 for any seed', () => {
      // Arrange
      const embryo = createBaseEmbryo({ generation: 0 });

      // Act
      const canGrow = evaluateJuvenileGrowGate(embryo, defaultConfig);

      // Assert
      expect(canGrow).toBe(true);
    });

    it('opens when the generation is a multiple of the seed-driven cooldown', () => {
      // Arrange
      // seed 42 -> 42 % 3 = 0 -> cooldown = 2, so generation 2 opens.
      const embryo = createBaseEmbryo({ generation: 2 });

      // Act
      const canGrow = evaluateJuvenileGrowGate(embryo, defaultConfig);

      // Assert
      expect(canGrow).toBe(true);
    });
  });

  describe('gate closed conditions', () => {
    it('closes when generation is not a multiple of the cooldown', () => {
      // Arrange
      // seed 42 -> cooldown = 2, so generation 1 is closed.
      const embryo = createBaseEmbryo({ generation: 1 });

      // Act
      const canGrow = evaluateJuvenileGrowGate(embryo, defaultConfig);

      // Assert
      expect(canGrow).toBe(false);
    });
  });

  describe('determinism and seed sensitivity', () => {
    it('is deterministic for the same seed and generation', () => {
      // Arrange
      const embryoA = createBaseEmbryo({ generation: 4 });
      const embryoB = createBaseEmbryo({ generation: 4 });

      // Act
      const resultA = evaluateJuvenileGrowGate(embryoA, defaultConfig);
      const resultB = evaluateJuvenileGrowGate(embryoB, defaultConfig);

      // Assert
      expect(resultA).toBe(resultB);
    });

    it('produces different cooldown cadences for different seeds', () => {
      // Arrange
      const embryo = createBaseEmbryo({ generation: 2 });
      const configA: NgeMainAgentLifecycleConfig = {
        ...defaultConfig,
        seed: 42,
      };
      const configB: NgeMainAgentLifecycleConfig = {
        ...defaultConfig,
        seed: 43,
      };

      // Act
      const canGrowA = evaluateJuvenileGrowGate(embryo, configA);
      const canGrowB = evaluateJuvenileGrowGate(embryo, configB);

      // Assert
      expect(canGrowA).not.toBe(canGrowB);
    });
  });
});

describe('growJuvenileTopologyWithInternalAssimilation', () => {
  describe('grow gate integration', () => {
    it('grows topology when the gate is open', () => {
      // Arrange
      const embryo = createBaseEmbryo({
        generation: 0,
        nodeCount: 10,
        edgeCount: 20,
        seed: 600_000,
      });
      const growthConfig: NgeMainAgentLifecycleConfig = {
        ...defaultConfig,
        seed: 600_000,
      };

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        growthConfig,
      );

      // Assert
      expect(pass.juvenile.nodeCount).toBeGreaterThan(embryo.nodeCount);
    });

    it('preserves topology when the gate is closed', () => {
      // Arrange
      const embryo = createBaseEmbryo({
        generation: 1,
        nodeCount: 100,
        edgeCount: 200,
      });

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
      );

      // Assert
      expect(pass).toMatchObject({
        didGrow: false,
        juvenile: {
          nodeCount: embryo.nodeCount,
          edgeCount: embryo.edgeCount,
        },
      });
    });

    it('returns a juvenile-stage snapshot', () => {
      // Arrange
      const embryo = createBaseEmbryo();

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
      );

      // Assert
      expect(pass.juvenile.stage).toBe('juvenile');
    });
  });

  describe('internal assimilation integration', () => {
    it('returns null assimilation when no candidate is provided', () => {
      // Arrange
      const embryo = createBaseEmbryo();

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
      );

      // Assert
      expect(pass.assimilation).toBeNull();
    });

    it('applies weak internal assimilation priors when a candidate is provided', () => {
      // Arrange
      const embryo = createBaseEmbryo();
      const candidate = createAssimilationCandidate();

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
        candidate,
      );

      // Assert
      expect(pass.assimilation).toMatchObject({
        sourceDnaFingerprint: 'fp:main-agent-equilibrium',
        appliedDecay: JUVENILE_ASSIMILATION_DECAY,
        updatedModuleDelta: {
          moduleId: 'module:main-1',
          ruleParameters: {
            connectionDensity: {
              currentValue: 0.254,
              targetValue: 0.8,
            },
          },
        },
      });
    });

    it('does not incorporate enemy-derived weights', () => {
      // Arrange
      const embryo = createBaseEmbryo();
      const candidate = createAssimilationCandidate();

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
        candidate,
      );

      // Assert
      expect(pass.assimilation?.enemyWeightsIncorporated).toBe(false);
    });

    it('does not incorporate enemy-derived structure', () => {
      // Arrange
      const embryo = createBaseEmbryo();
      const candidate = createAssimilationCandidate();

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
        candidate,
      );

      // Assert
      expect(pass.assimilation?.enemyStructureIncorporated).toBe(false);
    });

    it('uses the default juvenile decay factor to weaken the prior update', () => {
      // Arrange
      const embryo = createBaseEmbryo();
      const candidate = createAssimilationCandidate();
      const expectedDelta =
        candidate.moduleDelta.ruleParameters!.connectionDensity.currentValue +
        DEFAULT_ASSIMILATION_WRITE_BACK_RATE *
          JUVENILE_ASSIMILATION_DECAY *
          (candidate.moduleDelta.ruleParameters!.connectionDensity.targetValue -
            candidate.moduleDelta.ruleParameters!.connectionDensity
              .currentValue);

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
        candidate,
      );

      // Assert
      expect(
        pass.assimilation?.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(Number(expectedDelta.toFixed(12)));
    });
  });

  describe('full result contract', () => {
    it('carries the embryo archetypes and schema version into the juvenile', () => {
      // Arrange
      const embryo = createBaseEmbryo();

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
      );

      // Assert
      expect(pass.juvenile).toMatchObject({
        archetypes: embryo.archetypes,
        schemaVersion: embryo.schemaVersion,
      });
    });

    it('reports didGrow true when the gate opens and topology changes', () => {
      // Arrange
      const embryo = createBaseEmbryo({
        generation: 0,
        nodeCount: 10,
        edgeCount: 20,
        seed: 600_000,
      });
      const growthConfig: NgeMainAgentLifecycleConfig = {
        ...defaultConfig,
        seed: 600_000,
      };

      // Act
      const pass = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        growthConfig,
      );

      // Assert
      expect(pass.didGrow).toBe(true);
    });

    it('produces a deterministic pass for the same embryo, config, and candidate', () => {
      // Arrange
      const embryo = createBaseEmbryo();
      const candidate = createAssimilationCandidate();

      // Act
      const passA = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
        candidate,
      );
      const passB = growJuvenileTopologyWithInternalAssimilation(
        embryo,
        defaultConfig,
        candidate,
      );

      // Assert
      expect(passA).toEqual(passB);
    });
  });
});
