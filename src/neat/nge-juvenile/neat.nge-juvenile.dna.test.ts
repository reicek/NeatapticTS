/**
 * Red-phase test contracts for the NGE grow-stabilize DNA governance translator.
 *
 * These tests define the expected contract for
 * `src/neat/nge-juvenile/neat.nge-juvenile.dna.ts` before the implementation
 * exists. The target function is `translateDnaToGrowStabilizeConfig`, which reads
 * DNA `governance.growStabilize` overrides and falls back to caller-supplied
 * defaults for missing fields.
 *
 * All tests fail because `./neat.nge-juvenile.dna` does not exist yet.
 * The failure reason is "missing implementation" (import error), not syntax
 * error or bad fixture.
 *
 * Single-expect rule enforced throughout. AAA structure in every test.
 */
import { translateDnaToGrowStabilizeConfig } from './neat.nge-juvenile.dna';
import {
  NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
  NGE_GROW_STABILIZE_DEFAULT_MODULE_ID,
  NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
  NGE_MAX_EDGE_CAPACITY,
  NGE_MAX_NODE_CAPACITY,
} from './neat.nge-juvenile.constants';
import type { NgeGrowStabilizeConfig } from './neat.nge-juvenile.types';
import type { NgeDnaCanonicalEnvelope } from '../nge-dna/neat.nge-dna.types';

const makeEnvelope = (
  growStabilizeGovernance?: Partial<NgeGrowStabilizeConfig>,
): NgeDnaCanonicalEnvelope => {
  const envelope = {
    schemaVersion:
      'v1' as unknown as import('../nge-dna/neat.nge-dna.types').NgeSchemaVersion,
    compatibilityVersion: '1.0.0',
    encodingMode: 'lossless' as const,
    fingerprint: 'test-fingerprint',
    substrate: {
      dimensions: 3 as const,
      normalization: 'unit-cube' as const,
      zonePartition: {
        x: { count: 2 },
        y: { count: 2 },
        z: { count: 2 },
      },
    },
    reproductionPolicy: {
      mode: 'parthenogenesis' as const,
      parthenogenesisMutationRate: 0.0,
      polyandricDroneCount: 0,
      polyandricDroneContributionFraction: 0,
      queenBias: 1,
      assignedRegionStrategy: 'roundRobin' as const,
      modeIsEvolvable: false,
      seedPolicy: {
        siblingsDifferBySeed: false,
        twinsAllowed: false,
      },
    },
    rulePasses: [],
    cppnPrograms: [],
    moduleArchetypes: [],
  } as NgeDnaCanonicalEnvelope;

  if (growStabilizeGovernance !== undefined) {
    (
      envelope as NgeDnaCanonicalEnvelope & {
        governance: { growStabilize: Partial<NgeGrowStabilizeConfig> };
      }
    ).governance = { growStabilize: growStabilizeGovernance };
  }

  return envelope;
};

describe('NGE DNA governance translator', () => {
  describe('translateDnaToGrowStabilizeConfig', () => {
    it('is exported as a function', () => {
      // Arrange — imported at top of file

      // Act
      const fnType = typeof translateDnaToGrowStabilizeConfig;

      // Assert
      expect(fnType).toBe('function');
    });

    describe('default fallback scenarios', () => {
      it('returns supplied defaults when DNA has no governance', () => {
        // Arrange — minimal envelope without governance and complete defaults
        const dna = makeEnvelope();
        const defaults: Partial<NgeGrowStabilizeConfig> = {
          maxStructuralEditsPerStep: 7,
          maxNodes: 500,
          maxConnections: 2_000,
          maxEpisodicSlots: 8,
          moduleId: 'test:module',
        };

        // Act
        const result = translateDnaToGrowStabilizeConfig(dna, defaults);

        // Assert
        expect(result).toEqual(defaults);
      });
    });

    describe('DNA governance override scenarios', () => {
      it('overrides defaults from DNA governance.growStabilize', () => {
        // Arrange — envelope with a complete governance override
        const dna = makeEnvelope({
          maxStructuralEditsPerStep: 3,
          maxNodes: 1_000,
          maxConnections: 4_000,
          maxEpisodicSlots: 10,
          moduleId: 'dna:override',
        });
        const defaults: Partial<NgeGrowStabilizeConfig> = {
          maxStructuralEditsPerStep: 7,
          maxNodes: 500,
          maxConnections: 2_000,
          maxEpisodicSlots: 8,
          moduleId: 'test:module',
        };

        // Act
        const result = translateDnaToGrowStabilizeConfig(dna, defaults);

        // Assert
        expect(result).toEqual({
          maxStructuralEditsPerStep: 3,
          maxNodes: 1_000,
          maxConnections: 4_000,
          maxEpisodicSlots: 10,
          moduleId: 'dna:override',
        });
      });

      it('uses DNA values for only the fields present and falls back to defaults for missing ones', () => {
        // Arrange — envelope with a partial governance override
        const dna = makeEnvelope({
          maxStructuralEditsPerStep: 9,
          moduleId: 'dna:partial',
        });
        const defaults: Partial<NgeGrowStabilizeConfig> = {
          maxStructuralEditsPerStep:
            NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
          maxNodes: NGE_MAX_NODE_CAPACITY,
          maxConnections: NGE_MAX_EDGE_CAPACITY,
          maxEpisodicSlots: NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
          moduleId: NGE_GROW_STABILIZE_DEFAULT_MODULE_ID,
        };

        // Act
        const result = translateDnaToGrowStabilizeConfig(dna, defaults);

        // Assert
        expect(result).toEqual({
          maxStructuralEditsPerStep: 9,
          maxNodes: NGE_MAX_NODE_CAPACITY,
          maxConnections: NGE_MAX_EDGE_CAPACITY,
          maxEpisodicSlots: NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
          moduleId: 'dna:partial',
        });
      });
    });
  });
});
