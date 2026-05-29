import type Network from '../../architecture/network/network';
import Neat from '../../neat';

import {
  NGE_DNA_CPPN_INPUT_COUNT,
  NGE_DNA_CPPN_OUTPUT_COUNT,
  NGE_DNA_DEFAULT_BUDGET_MAX_EDGES,
  NGE_DNA_DEFAULT_BUDGET_MAX_NODES,
  NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD,
  NGE_DNA_DEFAULT_RULE_PRIORITY,
  NGE_DNA_DEFAULT_ZONE_PARTITION_COUNT,
  NGE_DNA_SCHEMA_VERSION,
} from './neat.nge-dna.constants';
import {
  NGE_DNA_BudgetError,
  NGE_DNA_CppnError,
  NGE_DNA_SchemaError,
  NGE_DNA_SubstrateError,
} from './neat.nge-dna.errors';
import { evaluateCppnProgram } from './neat.nge-dna.cppn';
import { realizePhenotypeFromPlan } from './neat.nge-dna.realize';
import { executeRulePasses } from './neat.nge-dna.rules';
import {
  assignZone,
  buildSubstrateFingerprint,
  buildZoneMap,
  normalizeCoordinate,
} from './neat.nge-dna.substrate';
import { NGE_DNA } from './neat.nge-dna';
import { canonicalSerialize, validateIdentity } from './neat.nge-dna.utils';
import type {
  NgeCppnProgram,
  NgeDnaModuleArchetype,
  NgeRealizedPhenotypeDescriptor,
  NgeRulePass,
  NgeSubstrateConfig,
  NgeVirtualModulePlan,
} from './neat.nge-dna.types';

describe('NGE_DNA', () => {
  describe('canonical serialization helpers', () => {
    describe('given a fully populated canonical envelope', () => {
      it('serializes the envelope with recursively sorted keys', () => {
        // Arrange
        const canonicalEnvelope = {
          compatibilityVersion: '2.0.0',
          encodingMode: 'lossy' as const,
          fingerprint: 'fingerprint',
          reproductionPolicy: {
            assignedRegionStrategy: 'byFitness' as const,
            mode: 'polyandric' as const,
            modeIsEvolvable: true,
            parthenogenesisMutationRate: 0.25,
            polyandricDroneContributionFraction: 0.4,
            polyandricDroneCount: 4,
            queenBias: 0.75,
            seedPolicy: {
              siblingsDifferBySeed: true,
              twinsAllowed: false,
            },
          },
          rulePasses: [
            {
              archetypeId: 'archetype:alpha',
              kind: 'replicate' as const,
              placements: [
                {
                  computationType: 'DenseFeedForward' as const,
                  coordinate: [0.25, 0.5, 0.75] as const,
                },
              ],
              priority: 0,
            },
          ],
          schemaVersion: NGE_DNA_SCHEMA_VERSION,
          substrate: {
            budgetOverride: {
              maxEdges: undefined,
              maxNodes: 64,
            },
            dimensions: 3 as const,
            normalization: 'unit-cube' as const,
            zonePartition: {
              x: { count: 2 },
              y: { count: 2 },
              z: { count: 2 },
            },
          },
        };

        // Act
        const canonicalJson = canonicalSerialize(canonicalEnvelope);

        // Assert
        expect(canonicalJson).toBe(
          '{"compatibilityVersion":"2.0.0","encodingMode":"lossy","fingerprint":"fingerprint","reproductionPolicy":{"assignedRegionStrategy":"byFitness","mode":"polyandric","modeIsEvolvable":true,"parthenogenesisMutationRate":0.25,"polyandricDroneContributionFraction":0.4,"polyandricDroneCount":4,"queenBias":0.75,"seedPolicy":{"siblingsDifferBySeed":true,"twinsAllowed":false}},"rulePasses":[{"archetypeId":"archetype:alpha","kind":"replicate","placements":[{"computationType":"DenseFeedForward","coordinate":[0.25,0.5,0.75]}],"priority":0}],"schemaVersion":"A.1.0","substrate":{"budgetOverride":{"maxNodes":64},"dimensions":3,"normalization":"unit-cube","zonePartition":{"x":{"count":2},"y":{"count":2},"z":{"count":2}}}}',
        );
      });
    });

    describe('given valid identity fields', () => {
      it('accepts the identity shelf without throwing', () => {
        // Arrange
        const dna = new NGE_DNA();

        // Assert
        expect(() => validateIdentity(dna.toCanonical())).not.toThrow();
      });
    });

    describe('given a missing fingerprint', () => {
      it('rejects the incomplete identity shelf', () => {
        // Arrange
        const dna = new NGE_DNA();
        const invalidIdentity = {
          ...dna.toCanonical(),
          fingerprint: '',
        };

        // Assert
        expect(() => validateIdentity(invalidIdentity)).toThrow(
          NGE_DNA_SchemaError,
        );
      });
    });
  });

  describe('default resolution', () => {
    describe('given an empty constructor input', () => {
      it('resolves the required identity defaults and substrate sentinels deterministically', () => {
        // Arrange
        const dna = new NGE_DNA();

        // Act
        const resolvedEnvelope = dna.toCanonical();

        // Assert
        expect({
          compatibilityVersion: dna.compatibilityVersion,
          edgeBudget: resolvedEnvelope.substrate.budgetOverride?.maxEdges,
          encodingMode: dna.encodingMode,
          fingerprintLength: dna.fingerprint.length,
          mode: resolvedEnvelope.reproductionPolicy.mode,
          nodeBudget: resolvedEnvelope.substrate.budgetOverride?.maxNodes,
          rulePassCount: resolvedEnvelope.rulePasses.length,
          schemaVersion: dna.schemaVersion,
          zonePartition: resolvedEnvelope.substrate.zonePartition,
        }).toEqual({
          compatibilityVersion: '1.0.0',
          edgeBudget: NGE_DNA_DEFAULT_BUDGET_MAX_EDGES,
          encodingMode: 'lossless',
          fingerprintLength: 64,
          mode: 'sexual',
          nodeBudget: NGE_DNA_DEFAULT_BUDGET_MAX_NODES,
          rulePassCount: 0,
          schemaVersion: NGE_DNA_SCHEMA_VERSION,
          zonePartition: createZonePartition(),
        });
      });
    });

    describe('given explicit constructor overrides', () => {
      it('preserves the provided canonical values while filling the remaining defaults', () => {
        // Arrange
        const dna = new NGE_DNA({
          compatibilityVersion: '2.1.0',
          encodingMode: 'lossy',
          reproductionPolicy: {
            assignedRegionStrategy: 'byFitness',
            mode: 'parthenogenesis',
            modeIsEvolvable: true,
            parthenogenesisMutationRate: 0,
            polyandricDroneContributionFraction: 0.25,
            polyandricDroneCount: 3,
            queenBias: 0.8,
            seedPolicy: {
              twinsAllowed: true,
            },
          },
          substrate: {
            budgetOverride: {
              maxEdges: 128,
              maxNodes: 64,
            },
            zonePartition: {
              x: { count: 2 },
              y: { count: 4 },
              z: { count: 5 },
            },
          },
          rulePasses: [
            {
              archetypeId: 'archetype:default-priority',
              kind: 'replicate',
              placements: [
                {
                  computationType: 'DenseFeedForward',
                  coordinate: [0.25, 0.5, 0.75],
                },
              ],
            },
          ],
        });

        // Act
        const resolvedEnvelope = dna.toCanonical();

        // Assert
        expect(resolvedEnvelope).toEqual({
          compatibilityVersion: '2.1.0',
          cppnPrograms: [],
          encodingMode: 'lossy',
          fingerprint: dna.fingerprint,
          moduleArchetypes: [],
          reproductionPolicy: {
            assignedRegionStrategy: 'byFitness',
            mode: 'parthenogenesis',
            modeIsEvolvable: true,
            parthenogenesisMutationRate: 0,
            polyandricDroneContributionFraction: 0.25,
            polyandricDroneCount: 3,
            queenBias: 0.8,
            seedPolicy: {
              siblingsDifferBySeed: true,
              twinsAllowed: true,
            },
          },
          schemaVersion: NGE_DNA_SCHEMA_VERSION,
          substrate: {
            budgetOverride: {
              maxEdges: 128,
              maxNodes: 64,
            },
            dimensions: 3,
            normalization: 'unit-cube',
            zonePartition: {
              x: { count: 2 },
              y: { count: 4 },
              z: { count: 5 },
            },
          },
          rulePasses: [
            {
              archetypeId: 'archetype:default-priority',
              kind: 'replicate',
              placements: [
                {
                  computationType: 'DenseFeedForward',
                  coordinate: [0.25, 0.5, 0.75],
                },
              ],
              priority: NGE_DNA_DEFAULT_RULE_PRIORITY,
            },
          ],
        });
      });
    });
  });

  describe('fingerprints', () => {
    describe('given two identical constructor inputs', () => {
      it('produces the same deterministic fingerprint', () => {
        // Arrange
        const firstDna = new NGE_DNA({ compatibilityVersion: '3.0.0' });
        const secondDna = new NGE_DNA({ compatibilityVersion: '3.0.0' });

        // Assert
        expect(firstDna.fingerprint).toBe(secondDna.fingerprint);
      });
    });

    describe('given different canonical content', () => {
      it('recomputes a different hash for the changed envelope', () => {
        // Arrange
        const firstDna = new NGE_DNA({ compatibilityVersion: '4.0.0' });
        const secondDna = new NGE_DNA({ compatibilityVersion: '4.1.0' });

        // Act
        const firstFingerprint = firstDna.recomputeFingerprint();
        const secondFingerprint = secondDna.recomputeFingerprint();

        // Assert
        expect(firstFingerprint).not.toBe(secondFingerprint);
      });
    });
  });

  describe('serialization roundtrips', () => {
    describe('given a serialized canonical payload', () => {
      it('deserializes back to an identical canonical envelope', () => {
        // Arrange
        const originalDna = new NGE_DNA({
          compatibilityVersion: '5.0.0',
          reproductionPolicy: {
            mode: 'polyandric',
            polyandricDroneCount: 5,
          },
          rulePasses: [
            {
              archetypeId: 'archetype:roundtrip',
              kind: 'hierarchy',
              placements: [
                {
                  computationType: 'AttentionHead',
                  coordinate: [0.1, 0.2, 0.3],
                },
              ],
              priority: 1,
            },
          ],
          substrate: {
            zonePartition: {
              x: { count: 2 },
              y: { count: 3 },
              z: { count: 4 },
            },
          },
        });

        // Act
        const restoredDna = NGE_DNA.deserialize(originalDna.serialize());

        // Assert
        expect(restoredDna.toCanonical()).toEqual(originalDna.toCanonical());
      });

      it('accepts one legacy payload that omits the new empty step-04 shelves', () => {
        // Arrange
        const dna = new NGE_DNA();
        const legacyEnvelope = JSON.parse(dna.serialize()) as Record<
          string,
          unknown
        >;

        delete legacyEnvelope.cppnPrograms;
        delete legacyEnvelope.moduleArchetypes;

        // Act
        const restoredDna = NGE_DNA.deserialize(JSON.stringify(legacyEnvelope));

        // Assert
        expect({
          cppnPrograms: restoredDna.cppnPrograms,
          moduleArchetypes: restoredDna.moduleArchetypes,
        }).toEqual({
          cppnPrograms: [],
          moduleArchetypes: [],
        });
      });
    });

    describe('given a stale fingerprint', () => {
      it('rejects the canonical envelope', () => {
        // Arrange
        const dna = new NGE_DNA();
        const staleEnvelope = {
          ...dna.toCanonical(),
          fingerprint: 'stale',
        };

        // Assert
        expect(() => NGE_DNA.fromCanonical(staleEnvelope)).toThrow(
          NGE_DNA_SchemaError,
        );
      });
    });

    describe('given an incompatible schema version', () => {
      it('throws the schema validation error', () => {
        // Arrange
        const dna = new NGE_DNA();
        const incompatibleJson = dna
          .serialize()
          .replace('"schemaVersion":"A.1.0"', '"schemaVersion":"A.2.0"');

        // Assert
        expect(() => NGE_DNA.deserialize(incompatibleJson)).toThrow(
          NGE_DNA_SchemaError,
        );
      });
    });
  });

  describe('budget guards', () => {
    describe('given an oversized node budget override', () => {
      it('throws the budget validation error', () => {
        // Assert
        expect(
          () =>
            new NGE_DNA({
              substrate: {
                budgetOverride: {
                  maxNodes: NGE_DNA_DEFAULT_BUDGET_MAX_NODES + 1,
                },
              },
            }),
        ).toThrow(NGE_DNA_BudgetError);
      });
    });

    describe('given an invalid substrate partition override', () => {
      it('throws the substrate validation error', () => {
        // Assert
        expect(
          () =>
            new NGE_DNA({
              substrate: {
                zonePartition: {
                  x: { count: 0 },
                },
              },
            }),
        ).toThrow(NGE_DNA_SubstrateError);
      });
    });
  });

  describe('substrate coordinate system', () => {
    it('keeps one already normalized coordinate unchanged', () => {
      // Arrange
      const normalizedCoordinate = [0.5, 0.5, 0.5] as const;

      // Act
      const result = normalizeCoordinate([...normalizedCoordinate]);

      // Assert
      expect(result).toEqual([0.5, 0.5, 0.5]);
    });

    it('clamps one out-of-range coordinate into the unit cube', () => {
      // Arrange
      const rawCoordinate = [1.5, -0.1, 0.5] as const;

      // Act
      const result = normalizeCoordinate([...rawCoordinate]);

      // Assert
      expect(result).toEqual([1, 0, 0.5]);
    });

    it('assigns one lower-corner coordinate to the first zone', () => {
      // Arrange
      const zonePartition = createZonePartition();

      // Act
      const zoneId = assignZone([0.1, 0.1, 0.1], zonePartition);

      // Assert
      expect(zoneId).toBe('z:0:0:0');
    });

    it('assigns one upper-corner coordinate to the last zone', () => {
      // Arrange
      const zonePartition = createZonePartition();

      // Act
      const zoneId = assignZone([0.9, 0.9, 0.9], zonePartition);

      // Assert
      expect(zoneId).toBe('z:2:2:2');
    });

    it('builds the expected number of zones for a two-partition grid', () => {
      // Arrange
      const zonePartition = createZonePartition(2);

      // Act
      const zoneMap = buildZoneMap(zonePartition);

      // Assert
      expect(zoneMap.size).toBe(8);
    });

    it('produces the same substrate fingerprint for the same config', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();

      // Act
      const firstFingerprint = buildSubstrateFingerprint(substrateConfig);
      const secondFingerprint = buildSubstrateFingerprint(substrateConfig);

      // Assert
      expect(firstFingerprint).toBe(secondFingerprint);
    });

    it('rejects one non-finite coordinate axis', () => {
      // Assert
      expect(() => normalizeCoordinate([Number.NaN, 0, 0])).toThrow(
        NGE_DNA_SubstrateError,
      );
    });

    it('rejects one invalid zone partition count', () => {
      // Assert
      expect(() =>
        buildZoneMap({
          x: { count: 0 },
          y: { count: 1 },
          z: { count: 1 },
        }),
      ).toThrow(NGE_DNA_SubstrateError);
    });
  });

  describe('deterministic rule passes', () => {
    it('produces the same plan fingerprint for the same passes and seed', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();
      const rulePasses = createDeterministicRulePasses();

      // Act
      const firstPlan = executeRulePasses(rulePasses, substrateConfig, 17);
      const secondPlan = executeRulePasses(rulePasses, substrateConfig, 17);

      // Assert
      expect(firstPlan.planFingerprint).toBe(secondPlan.planFingerprint);
    });

    it('changes the plan fingerprint when the seed changes', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();
      const rulePasses = createDeterministicRulePasses();

      // Act
      const firstPlan = executeRulePasses(rulePasses, substrateConfig, 17);
      const secondPlan = executeRulePasses(rulePasses, substrateConfig, 23);

      // Assert
      expect(firstPlan.planFingerprint).not.toBe(secondPlan.planFingerprint);
    });

    it('keeps all generated module ids unique', () => {
      // Arrange
      const plan = executeRulePasses(
        createDeterministicRulePasses(),
        createSubstrateConfig(),
        17,
      );

      // Assert
      expect({
        moduleCount: plan.modules.length,
        uniqueModuleCount: new Set(
          plan.modules.map((moduleDescriptor) => moduleDescriptor.moduleId),
        ).size,
      }).toEqual({
        moduleCount: 3,
        uniqueModuleCount: 3,
      });
    });

    it('orders lower-priority rule passes before higher-priority ones', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();
      const rulePasses = [
        {
          archetypeId: 'archetype:late',
          kind: 'symmetry',
          placements: [
            {
              computationType: 'AttentionHead',
              coordinate: [0.75, 0.75, 0.75],
            },
          ],
          priority: 2,
        },
        {
          archetypeId: 'archetype:early',
          kind: 'replicate',
          placements: [
            {
              computationType: 'DenseFeedForward',
              coordinate: [0.25, 0.25, 0.25],
            },
          ],
          priority: 1,
        },
      ] satisfies NgeRulePass[];

      // Act
      const plan = executeRulePasses(rulePasses, substrateConfig, 17);

      // Assert
      expect(
        plan.modules.map((moduleDescriptor) => moduleDescriptor.archetypeId),
      ).toEqual(['archetype:early', 'archetype:late']);
    });

    it('breaks equal-priority ties alphabetically by kind', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();
      const rulePasses = [
        {
          archetypeId: 'archetype:symmetry',
          kind: 'symmetry',
          placements: [
            {
              computationType: 'AttentionHead',
              coordinate: [0.75, 0.75, 0.75],
            },
          ],
          priority: 1,
        },
        {
          archetypeId: 'archetype:hierarchy',
          kind: 'hierarchy',
          placements: [
            {
              computationType: 'DenseFeedForward',
              coordinate: [0.25, 0.25, 0.25],
            },
          ],
          priority: 1,
        },
      ] satisfies NgeRulePass[];

      // Act
      const plan = executeRulePasses(rulePasses, substrateConfig, 17);

      // Assert
      expect(
        plan.modules.map((moduleDescriptor) => moduleDescriptor.moduleId),
      ).toEqual([
        '0:hierarchy:archetype:hierarchy:0',
        '1:symmetry:archetype:symmetry:0',
      ]);
    });

    it('returns one empty but fully fingerprinted plan when no passes exist', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();

      // Act
      const plan = executeRulePasses([], substrateConfig, 0);

      // Assert
      expect({
        moduleCount: plan.modules.length,
        planFingerprintLength: plan.planFingerprint.length,
        substrateFingerprintLength: plan.substrateFingerprint.length,
      }).toEqual({
        moduleCount: 0,
        planFingerprintLength: 64,
        substrateFingerprintLength: 64,
      });
    });

    it('clamps rule-placement coordinates before recording the virtual module', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();
      const rulePasses = [
        {
          archetypeId: 'archetype:clamped',
          kind: 'replicate',
          placements: [
            {
              computationType: 'DenseFeedForward',
              coordinate: [2, 0.5, 0.5],
            },
          ],
          priority: 1,
        },
      ] satisfies NgeRulePass[];

      // Act
      const plan = executeRulePasses(rulePasses, substrateConfig, 17);

      // Assert
      expect(plan.modules[0]?.coordinate[0]).toBe(1);
    });

    it('rejects duplicate rule-pass identity tuples before execution', () => {
      // Arrange
      const substrateConfig = createSubstrateConfig();
      const duplicateRulePasses = [
        {
          archetypeId: 'archetype:duplicate',
          kind: 'replicate',
          placements: [
            {
              computationType: 'DenseFeedForward',
              coordinate: [0.25, 0.25, 0.25],
            },
          ],
          priority: 1,
        },
        {
          archetypeId: 'archetype:duplicate',
          kind: 'replicate',
          placements: [
            {
              computationType: 'AttentionHead',
              coordinate: [0.75, 0.75, 0.75],
            },
          ],
          priority: 1,
        },
      ] satisfies NgeRulePass[];

      // Assert
      expect(() =>
        executeRulePasses(duplicateRulePasses, substrateConfig, 17),
      ).toThrow(NGE_DNA_SubstrateError);
    });
  });

  describe('CPPN program evaluator', () => {
    it('evaluates one empty cppn to two zero outputs', () => {
      // Arrange
      const program = createCppnProgram({ hiddenNodes: [] });

      // Act
      const outputVector = evaluateCppnProgram(
        program,
        createCppnInputVector(),
      );

      // Assert
      expect(outputVector).toEqual(
        Array.from({ length: NGE_DNA_CPPN_OUTPUT_COUNT }, () => 0),
      );
    });

    it('evaluates one linear hidden-node path with the expected weighted sum', () => {
      // Arrange
      const program = createCppnProgram({
        edges: [
          {
            sourceNodeId: 'x1',
            targetNodeId: 'hidden:linear',
            weight: 1,
          },
          {
            sourceNodeId: 'hidden:linear',
            targetNodeId: 'weight',
            weight: 1,
          },
        ],
        hiddenNodes: [
          {
            activationKind: 'linear',
            bias: 1,
            nodeId: 'hidden:linear',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'weight',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'enableBias',
          },
        ],
      });

      // Act
      const outputVector = evaluateCppnProgram(
        program,
        createCppnInputVector([1]),
      );

      // Assert
      expect(outputVector[0]).toBe(2);
    });

    it('accumulates multiple incoming edges into one output node deterministically', () => {
      // Arrange
      const program = createCppnProgram({
        edges: [
          {
            sourceNodeId: 'x1',
            targetNodeId: 'weight',
            weight: 1,
          },
          {
            sourceNodeId: 'x2',
            targetNodeId: 'weight',
            weight: 1,
          },
        ],
      });

      // Act
      const outputVector = evaluateCppnProgram(
        program,
        createCppnInputVector([1, 0, 0, 1]),
      );

      // Assert
      expect(outputVector[0]).toBe(2);
    });

    it('evaluates one tanh output node bias deterministically', () => {
      // Arrange
      const program = createCppnProgram({
        hiddenNodes: [
          {
            activationKind: 'tanh',
            bias: 1,
            nodeId: 'weight',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'enableBias',
          },
        ],
      });

      // Act
      const outputVector = evaluateCppnProgram(
        program,
        createCppnInputVector(),
      );

      // Assert
      expect(outputVector[0]).toBeCloseTo(Math.tanh(1), 4);
    });

    it('evaluates one sigmoid output node bias deterministically', () => {
      // Arrange
      const program = createCppnProgram({
        hiddenNodes: [
          {
            activationKind: 'sigmoid',
            bias: 0,
            nodeId: 'weight',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'enableBias',
          },
        ],
      });

      // Act
      const outputVector = evaluateCppnProgram(
        program,
        createCppnInputVector(),
      );

      // Assert
      expect(outputVector[0]).toBe(0.5);
    });

    it('evaluates one gaussian output node bias deterministically', () => {
      // Arrange
      const program = createCppnProgram({
        hiddenNodes: [
          {
            activationKind: 'gaussian',
            bias: 0,
            nodeId: 'weight',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'enableBias',
          },
        ],
      });

      // Act
      const outputVector = evaluateCppnProgram(
        program,
        createCppnInputVector(),
      );

      // Assert
      expect(outputVector[0]).toBe(1);
    });

    it('evaluates one sine output node bias deterministically', () => {
      // Arrange
      const program = createCppnProgram({
        hiddenNodes: [
          {
            activationKind: 'sine',
            bias: Math.PI / 2,
            nodeId: 'weight',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'enableBias',
          },
        ],
      });

      // Act
      const outputVector = evaluateCppnProgram(
        program,
        createCppnInputVector(),
      );

      // Assert
      expect(outputVector[0]).toBeCloseTo(1, 4);
    });

    it('rejects one cyclic cppn topology', () => {
      // Arrange
      const cyclicProgram = createCppnProgram({
        edges: [
          {
            sourceNodeId: 'hidden:first',
            targetNodeId: 'hidden:second',
            weight: 1,
          },
          {
            sourceNodeId: 'hidden:second',
            targetNodeId: 'hidden:first',
            weight: 1,
          },
        ],
        hiddenNodes: [
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'hidden:first',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'hidden:second',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'weight',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'enableBias',
          },
        ],
      });

      // Assert
      expect(() =>
        evaluateCppnProgram(cyclicProgram, createCppnInputVector()),
      ).toThrow(NGE_DNA_CppnError);
    });

    it('rejects one input vector whose length is too short', () => {
      // Arrange
      const program = createCppnProgram();

      // Assert
      expect(() =>
        evaluateCppnProgram(program, createCppnInputVector().slice(0, 6)),
      ).toThrow(NGE_DNA_CppnError);
    });

    it('rejects one cppn program whose input-node count is invalid', () => {
      // Arrange
      const invalidProgram = createCppnProgram({
        inputNodeIds: createDefaultCppnInputNodeIds().slice(0, 6),
      });

      // Assert
      expect(() =>
        evaluateCppnProgram(invalidProgram, createCppnInputVector()),
      ).toThrow(NGE_DNA_CppnError);
    });

    it('rejects one cppn program whose output-node count is invalid', () => {
      // Arrange
      const invalidProgram = createCppnProgram({
        outputNodeIds: ['weight'],
      });

      // Assert
      expect(() =>
        evaluateCppnProgram(invalidProgram, createCppnInputVector()),
      ).toThrow(NGE_DNA_CppnError);
    });

    it('rejects one cppn edge that references an unknown node id', () => {
      // Arrange
      const invalidProgram = createCppnProgram({
        edges: [
          {
            sourceNodeId: 'missing',
            targetNodeId: 'weight',
            weight: 1,
          },
        ],
      });

      // Assert
      expect(() =>
        evaluateCppnProgram(invalidProgram, createCppnInputVector()),
      ).toThrow(NGE_DNA_CppnError);
    });
  });

  describe('CPPN edge realization', () => {
    it('returns no edges when the dna carries no cppn programs', () => {
      // Arrange
      const plan = createVirtualModulePlan();
      const envelope = createCanonicalEnvelope();

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(descriptor.edges).toEqual([]);
    });

    it('filters one below-threshold cppn edge weight out of the realized descriptor', () => {
      // Arrange
      const plan = createVirtualModulePlan();
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [
          createConstantWeightCppnProgram(
            NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD - 0.1,
          ),
        ],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(descriptor.edges).toHaveLength(0);
    });

    it('realizes directed edges when one cppn output exceeds the threshold', () => {
      // Arrange
      const plan = createVirtualModulePlan();
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [
          createConstantWeightCppnProgram(
            NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD + 0.2,
          ),
        ],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(descriptor.edges).toHaveLength(2);
    });

    it('assigns zero wiring cost to one residual-tap edge', () => {
      // Arrange
      const plan = createVirtualModulePlan([
        {
          archetypeId: 'archetype:residual',
          computationType: 'DenseFeedForward',
          coordinate: [0, 0, 0],
          moduleId: 'module:residual',
        },
        {
          archetypeId: 'archetype:plain',
          computationType: 'DenseFeedForward',
          coordinate: [1, 0, 0],
          moduleId: 'module:plain',
        },
      ]);
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
        moduleArchetypes: [
          {
            archetypeId: 'archetype:residual',
            computationType: 'DenseFeedForward',
            residualStreamId: 'stream:main',
          },
        ],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(
        descriptor.edges.find(
          (edgeDescriptor) =>
            edgeDescriptor.sourceModuleId === 'module:residual' &&
            edgeDescriptor.targetModuleId === 'module:plain',
        ),
      ).toEqual({
        isModulatorBroadcast: false,
        isResidualTap: true,
        sourceModuleId: 'module:residual',
        targetModuleId: 'module:plain',
        weight: 0.5,
        wiringCost: 0,
      });
    });

    it('assigns zero wiring cost to one in-radius modulator broadcast edge', () => {
      // Arrange
      const plan = createVirtualModulePlan([
        {
          archetypeId: 'archetype:broadcaster',
          computationType: 'ModulatorBroadcaster',
          coordinate: [0, 0, 0],
          moduleId: 'module:broadcaster',
        },
        {
          archetypeId: 'archetype:plain',
          computationType: 'DenseFeedForward',
          coordinate: [0.5, 0, 0],
          moduleId: 'module:plain',
        },
      ]);
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
        moduleArchetypes: [
          {
            archetypeId: 'archetype:broadcaster',
            computationType: 'ModulatorBroadcaster',
            parameterSchema: {
              broadcastRadius: 1,
            },
          },
        ],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(
        descriptor.edges.find(
          (edgeDescriptor) =>
            edgeDescriptor.sourceModuleId === 'module:broadcaster' &&
            edgeDescriptor.targetModuleId === 'module:plain',
        ),
      ).toEqual({
        isModulatorBroadcast: true,
        isResidualTap: false,
        sourceModuleId: 'module:broadcaster',
        targetModuleId: 'module:plain',
        weight: 0.5,
        wiringCost: 0,
      });
    });

    it('retains euclidean wiring cost for one non-exempt edge', () => {
      // Arrange
      const plan = createVirtualModulePlan();
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(
        descriptor.edges.map((edgeDescriptor) => edgeDescriptor.wiringCost),
      ).toEqual([1, 1]);
    });

    it('skips self loops during directed edge realization', () => {
      // Arrange
      const plan = createVirtualModulePlan([
        {
          archetypeId: 'archetype:solo',
          computationType: 'DenseFeedForward',
          coordinate: [0.5, 0.5, 0.5],
          moduleId: 'module:solo',
        },
      ]);
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(descriptor.edges).toHaveLength(0);
    });

    it('produces the same phenotype fingerprint for identical plan, dna, and seed inputs', () => {
      // Arrange
      const plan = createVirtualModulePlan();
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
      });

      // Act
      const firstDescriptor = realizePhenotypeFromPlan(plan, envelope, 17);
      const secondDescriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(firstDescriptor.phenotypeFingerprint).toBe(
        secondDescriptor.phenotypeFingerprint,
      );
    });

    it('changes the phenotype fingerprint when the seed changes', () => {
      // Arrange
      const plan = createVirtualModulePlan();
      const envelope = createCanonicalEnvelope({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
      });

      // Act
      const firstDescriptor = realizePhenotypeFromPlan(plan, envelope, 17);
      const secondDescriptor = realizePhenotypeFromPlan(plan, envelope, 23);

      // Assert
      expect(firstDescriptor.phenotypeFingerprint).not.toBe(
        secondDescriptor.phenotypeFingerprint,
      );
    });

    it('rejects one virtual module whose computationType is not in the public catalogue', () => {
      // Arrange
      const invalidPlan = createVirtualModulePlan([
        {
          archetypeId: 'archetype:invalid',
          computationType: 'UnknownType',
          coordinate: [0, 0, 0],
          moduleId: 'module:invalid',
        },
      ]);
      const envelope = createCanonicalEnvelope();

      // Assert
      expect(() => realizePhenotypeFromPlan(invalidPlan, envelope, 17)).toThrow(
        NGE_DNA_CppnError,
      );
    });
  });

  describe('phenotype materialization', () => {
    it('decorates realized modules with archetype governance metadata', () => {
      // Arrange
      const plan = createVirtualModulePlan([
        {
          archetypeId: 'archetype:governed',
          computationType: 'DenseFeedForward',
          coordinate: [0, 0, 0],
          moduleId: 'module:governed',
        },
      ]);
      const envelope = createCanonicalEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:governed',
            computationType: 'DenseFeedForward',
            parameterSchema: {
              width: 16,
            },
            receivesCoordinates: true,
            residualStreamId: 'stream:main',
            weightSharedCohortId: 'cohort:main',
          },
        ],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(descriptor.modules[0]).toEqual({
        archetypeId: 'archetype:governed',
        archetypeParams: {
          width: 16,
        },
        computationType: 'DenseFeedForward',
        coordinate: [0, 0, 0],
        moduleId: 'module:governed',
        receivesCoordinates: true,
        residualStreamId: 'stream:main',
        weightSharedCohortId: 'cohort:main',
        zoneId: 'z:0:0:0',
      });
    });

    it('groups residual stream assignments by stream id in stable module order', () => {
      // Arrange
      const plan = createVirtualModulePlan([
        {
          archetypeId: 'archetype:streamed',
          computationType: 'DenseFeedForward',
          coordinate: [0, 0, 0],
          moduleId: 'module:first',
        },
        {
          archetypeId: 'archetype:streamed',
          computationType: 'AttentionHead',
          coordinate: [1, 0, 0],
          moduleId: 'module:second',
        },
      ]);
      const envelope = createCanonicalEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:streamed',
            computationType: 'DenseFeedForward',
            residualStreamId: 'stream:main',
          },
        ],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(descriptor.residualStreamAssignments).toEqual({
        'stream:main': ['module:first', 'module:second'],
      });
    });

    it('groups weight-shared cohort assignments by cohort id in stable module order', () => {
      // Arrange
      const plan = createVirtualModulePlan([
        {
          archetypeId: 'archetype:cohort',
          computationType: 'DenseFeedForward',
          coordinate: [0, 0, 0],
          moduleId: 'module:first',
        },
        {
          archetypeId: 'archetype:cohort',
          computationType: 'AttentionHead',
          coordinate: [1, 0, 0],
          moduleId: 'module:second',
        },
      ]);
      const envelope = createCanonicalEnvelope({
        moduleArchetypes: [
          {
            archetypeId: 'archetype:cohort',
            computationType: 'DenseFeedForward',
            weightSharedCohortId: 'cohort:main',
          },
        ],
      });

      // Act
      const descriptor = realizePhenotypeFromPlan(plan, envelope, 17);

      // Assert
      expect(descriptor.weightSharedCohortAssignments).toEqual({
        'cohort:main': ['module:first', 'module:second'],
      });
    });

    it('forwards the source dna fingerprint into the realized descriptor', () => {
      // Arrange
      const dna = new NGE_DNA({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
        rulePasses: createDeterministicRulePasses(),
      });
      const plan = dna.buildVirtualPlan(17);

      // Act
      const descriptor = dna.realizePhenotype(plan, 17);

      // Assert
      expect(descriptor.dnaFingerprint).toBe(dna.fingerprint);
    });

    it('roundtrips one realized descriptor through JSON serialization without loss', () => {
      // Arrange
      const descriptor = realizePhenotypeFromPlan(
        createVirtualModulePlan(),
        createCanonicalEnvelope({
          cppnPrograms: [createConstantWeightCppnProgram(0.5)],
        }),
        17,
      );

      // Act
      const roundTrippedDescriptor = JSON.parse(
        JSON.stringify(descriptor),
      ) as NgeRealizedPhenotypeDescriptor;

      // Assert
      expect(roundTrippedDescriptor).toEqual(descriptor);
    });
  });

  describe('class additions', () => {
    it('returns the resolved substrate config through the substrate accessor', () => {
      // Arrange
      const dna = new NGE_DNA();

      // Assert
      expect(dna.substrate).toEqual(createSubstrateConfig());
    });

    it('resolves omitted rule-pass placements to an empty array', () => {
      // Arrange
      const dna = new NGE_DNA({
        rulePasses: [
          {
            archetypeId: 'archetype:no-placements',
            kind: 'differentiate',
          },
        ],
      });

      // Assert
      expect(dna.toCanonical().rulePasses).toEqual([
        {
          archetypeId: 'archetype:no-placements',
          kind: 'differentiate',
          placements: [],
          priority: NGE_DNA_DEFAULT_RULE_PRIORITY,
        },
      ]);
    });

    it('builds the same virtual plan fingerprint for identical instances and seeds', () => {
      // Arrange
      const constructorInput = {
        rulePasses: createDeterministicRulePasses(),
        substrate: {
          zonePartition: createZonePartition(),
        },
      };
      const firstDna = new NGE_DNA(constructorInput);
      const secondDna = new NGE_DNA(constructorInput);

      // Act
      const firstPlan = firstDna.buildVirtualPlan(17);
      const secondPlan = secondDna.buildVirtualPlan(17);

      // Assert
      expect(firstPlan.planFingerprint).toBe(secondPlan.planFingerprint);
    });

    it('serializes the default rule-pass shelf as an empty array', () => {
      // Arrange
      const dna = new NGE_DNA({});

      // Act
      const canonicalEnvelope = JSON.parse(dna.serialize()) as {
        rulePasses: unknown;
      };

      // Assert
      expect(canonicalEnvelope.rulePasses).toEqual([]);
    });

    it('returns empty step-04 shelves by default through the public accessors', () => {
      // Arrange
      const dna = new NGE_DNA();

      // Assert
      expect({
        cppnPrograms: dna.cppnPrograms,
        moduleArchetypes: dna.moduleArchetypes,
      }).toEqual({
        cppnPrograms: [],
        moduleArchetypes: [],
      });
    });

    it('fills omitted cppn program structure with canonical defaults', () => {
      // Arrange
      const dna = new NGE_DNA({
        cppnPrograms: [
          {
            programId: 'cppn:defaults',
          },
        ],
      });

      // Assert
      expect(dna.cppnPrograms[0]).toEqual({
        edges: [],
        hiddenNodes: [],
        inputNodeIds: createDefaultCppnInputNodeIds(),
        outputNodeIds: ['weight', 'enableBias'],
        programId: 'cppn:defaults',
      });
    });

    it('fills omitted cppn node and edge fields with conservative defaults', () => {
      // Arrange
      const dna = new NGE_DNA({
        cppnPrograms: [
          {
            edges: [
              {
                sourceNodeId: 'x1',
                targetNodeId: 'weight',
              },
            ],
            hiddenNodes: [
              {
                nodeId: 'weight',
              },
              {
                nodeId: 'enableBias',
              },
            ],
            programId: 'cppn:resolved-defaults',
          },
        ],
      });

      // Assert
      expect(dna.cppnPrograms[0]).toEqual({
        edges: [
          {
            sourceNodeId: 'x1',
            targetNodeId: 'weight',
            weight: 0,
          },
        ],
        hiddenNodes: [
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'weight',
          },
          {
            activationKind: 'linear',
            bias: 0,
            nodeId: 'enableBias',
          },
        ],
        inputNodeIds: createDefaultCppnInputNodeIds(),
        outputNodeIds: ['weight', 'enableBias'],
        programId: 'cppn:resolved-defaults',
      });
    });

    it('realizes one descriptor with no edges when cppn programs are absent', () => {
      // Arrange
      const dna = new NGE_DNA({
        rulePasses: createDeterministicRulePasses(),
      });
      const plan = dna.buildVirtualPlan(17);

      // Act
      const descriptor = dna.realizePhenotype(plan, 17);

      // Assert
      expect(descriptor.edges).toEqual([]);
    });

    it('produces the same realized phenotype fingerprint for identical inputs', () => {
      // Arrange
      const dna = new NGE_DNA({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
        rulePasses: createDeterministicRulePasses(),
      });
      const plan = dna.buildVirtualPlan(17);

      // Act
      const firstDescriptor = dna.realizePhenotype(plan, 17);
      const secondDescriptor = dna.realizePhenotype(plan, 17);

      // Assert
      expect(firstDescriptor.phenotypeFingerprint).toBe(
        secondDescriptor.phenotypeFingerprint,
      );
    });

    it('roundtrips cppn programs and module archetypes through serialize and deserialize', () => {
      // Arrange
      const originalDna = new NGE_DNA({
        cppnPrograms: [createConstantWeightCppnProgram(0.5)],
        moduleArchetypes: [
          {
            archetypeId: 'archetype:roundtrip',
            computationType: 'DenseFeedForward',
            parameterSchema: {
              width: 8,
            },
            receivesCoordinates: true,
            residualStreamId: 'stream:main',
            weightSharedCohortId: 'cohort:main',
          },
        ],
      });

      // Act
      const restoredDna = NGE_DNA.deserialize(originalDna.serialize());

      // Assert
      expect({
        cppnPrograms: restoredDna.cppnPrograms,
        moduleArchetypes: restoredDna.moduleArchetypes,
      }).toEqual({
        cppnPrograms: originalDna.cppnPrograms,
        moduleArchetypes: originalDna.moduleArchetypes,
      });
    });
  });

  describe('classic NEAT opt-in behavior', () => {
    describe('given a standard NEAT constructor call with no NGE usage', () => {
      it('leaves classic population construction unchanged', () => {
        // Arrange
        const scoreByNodeCount = (network: Network) => network.nodes.length;

        // Act
        const neat = new Neat(1, 1, scoreByNodeCount, {
          popsize: 3,
          seed: 17,
        });

        // Assert
        expect(neat.population.length).toBe(3);
      });
    });
  });
});

function createZonePartition(count = NGE_DNA_DEFAULT_ZONE_PARTITION_COUNT) {
  return {
    x: { count },
    y: { count },
    z: { count },
  };
}

function createSubstrateConfig(
  overrides: Partial<NgeSubstrateConfig> = {},
): NgeSubstrateConfig {
  return {
    budgetOverride: {
      maxEdges: NGE_DNA_DEFAULT_BUDGET_MAX_EDGES,
      maxNodes: NGE_DNA_DEFAULT_BUDGET_MAX_NODES,
      ...(overrides.budgetOverride ?? {}),
    },
    dimensions: 3,
    normalization: 'unit-cube',
    zonePartition: overrides.zonePartition ?? createZonePartition(),
  };
}

function createDeterministicRulePasses(): NgeRulePass[] {
  return [
    {
      archetypeId: 'archetype:alpha',
      kind: 'hierarchy',
      placements: [
        {
          computationType: 'AttentionHead',
          coordinate: [0.75, 0.75, 0.75],
        },
      ],
      priority: 1,
    },
    {
      archetypeId: 'archetype:beta',
      kind: 'symmetry',
      placements: [
        {
          computationType: 'DenseFeedForward',
          coordinate: [0.25, 0.25, 0.25],
        },
        {
          computationType: 'GatedRecurrentCell',
          coordinate: [0.5, 0.5, 0.5],
        },
      ],
      priority: 2,
    },
  ];
}

function createDefaultCppnInputNodeIds(): string[] {
  return ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'dist'];
}

function createCppnInputVector(values: readonly number[] = []): number[] {
  return Array.from(
    { length: NGE_DNA_CPPN_INPUT_COUNT },
    (_, index) => values[index] ?? 0,
  );
}

function createCppnProgram(
  overrides: Partial<NgeCppnProgram> = {},
): NgeCppnProgram {
  return {
    edges: overrides.edges ? [...overrides.edges] : [],
    hiddenNodes:
      overrides.hiddenNodes === undefined
        ? [
            {
              activationKind: 'linear',
              bias: 0,
              nodeId: 'weight',
            },
            {
              activationKind: 'linear',
              bias: 0,
              nodeId: 'enableBias',
            },
          ]
        : [...overrides.hiddenNodes],
    inputNodeIds:
      overrides.inputNodeIds === undefined
        ? createDefaultCppnInputNodeIds()
        : [...overrides.inputNodeIds],
    outputNodeIds:
      overrides.outputNodeIds === undefined
        ? ['weight', 'enableBias']
        : [...overrides.outputNodeIds],
    programId: overrides.programId ?? 'cppn:test',
  };
}

function createConstantWeightCppnProgram(weight: number): NgeCppnProgram {
  return createCppnProgram({
    hiddenNodes: [
      {
        activationKind: 'linear',
        bias: weight,
        nodeId: 'weight',
      },
      {
        activationKind: 'linear',
        bias: 0,
        nodeId: 'enableBias',
      },
    ],
    programId: `cppn:constant:${weight}`,
  });
}

function createCanonicalEnvelope(
  overrides: {
    cppnPrograms?: readonly NgeCppnProgram[];
    moduleArchetypes?: readonly NgeDnaModuleArchetype[];
  } = {},
) {
  return new NGE_DNA({
    cppnPrograms: overrides.cppnPrograms,
    moduleArchetypes: overrides.moduleArchetypes,
  }).toCanonical();
}

function createVirtualModulePlan(
  moduleInputs: readonly {
    archetypeId: string;
    computationType: string;
    coordinate: [number, number, number];
    moduleId: string;
    zoneId?: string;
  }[] = [
    {
      archetypeId: 'archetype:alpha',
      computationType: 'DenseFeedForward',
      coordinate: [0, 0, 0],
      moduleId: 'module:alpha',
    },
    {
      archetypeId: 'archetype:beta',
      computationType: 'AttentionHead',
      coordinate: [1, 0, 0],
      moduleId: 'module:beta',
    },
  ],
): NgeVirtualModulePlan {
  return {
    modules: moduleInputs.map((moduleInput, moduleIndex) => ({
      archetypeId: moduleInput.archetypeId,
      computationType:
        moduleInput.computationType as NgeVirtualModulePlan['modules'][number]['computationType'],
      coordinate: [...moduleInput.coordinate] as [number, number, number],
      moduleId: moduleInput.moduleId,
      placementOrdinal: 0,
      rulePassIndex: moduleIndex,
      zoneId: moduleInput.zoneId ?? `z:${moduleIndex}:0:0`,
    })),
    planFingerprint: 'plan:test',
    substrateFingerprint: 'substrate:test',
  };
}
