import Network from '../../architecture/network/network';
import type Node from '../../architecture/node/node';
import type { NetworkJSON } from '../../architecture/network/network.types';
import {
  materializeNetworkFromPhenotype,
  extractCanonicalEnvelopeFromNetwork,
} from './neat.nge-dna.bridge';
import { NGE_DNA } from './neat.nge-dna';
import type {
  NgeCppnProgram,
  NgeDnaCanonicalEnvelope,
  NgeRealizedPhenotypeDescriptor,
  NgeRulePass,
  NgeVirtualModulePlan,
} from './neat.nge-dna.types';

// ─────────────────────────────────────────────────────────────────────────────
// Open decisions documented for red tests (assumptions the bridge must honor):
//
// 1. Input/output designation: z=0 → input node, z=1 → output node,
//    0 < z < 1 → hidden node. Requires no descriptor schema widening.
// 2. Deterministic innovation assignment: stable hash of
//    (sourceModuleId, targetModuleId) — pure function, stable across round-trips.
// 3. computationType → squash map: DenseFeedForward→'relu',
//    AttentionHead→'sigmoid', GatedRecurrentCell→'tanh',
//    EpisodicSlot→'identity', ModulatorBroadcaster→'identity',
//    GatingRouter→'sigmoid'. Default 'identity' when NGE disabled.
// 4. Extension carrier schema: { version: 1, ngeDescriptor, ngeEnvelope }
//    inside NetworkJSONExtensions.values.
// 5. Residual-stream/weight-sharing shelves: carried as descriptor-only
//    extension metadata — test they survive round-trip.
// 6. Seed semantics: seed NEVER affects topology — only salts
//    phenotypeFingerprint. Same DNA → identical topology regardless of seed.
// ─────────────────────────────────────────────────────────────────────────────

describe('NGE phenotype → Network bridge', () => {
  describe('materializeNetworkFromPhenotype', () => {
    it('produces a runtime Network with node count matching descriptor module count', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();

      // Act
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );

      // Assert
      expect(network.nodes.length).toBe(descriptor.modules.length);
    });

    it('produces a runtime Network with connection count matching descriptor edge count', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();

      // Act
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );

      // Assert
      expect(network.connections.length).toBe(descriptor.edges.length);
    });

    it('throws when descriptor has zero modules', () => {
      // Arrange
      const envelope = new NGE_DNA().toCanonical();
      const emptyPlan = createEmptyPlan();
      const emptyDescriptor = createEmptyDescriptor();

      // Assert
      expect(() =>
        materializeNetworkFromPhenotype(envelope, emptyPlan, emptyDescriptor),
      ).toThrow();
    });

    it('designates z=0 modules as input nodes', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const inputModuleCount = descriptor.modules.filter(
        (module) => module.coordinate[2] === 0,
      ).length;

      // Act
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );

      // Assert
      expect(
        network.nodes.filter((node: Node) => node.type === 'input').length,
      ).toBe(inputModuleCount);
    });

    it('designates z=1 modules as output nodes', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const outputModuleCount = descriptor.modules.filter(
        (module) => module.coordinate[2] === 1,
      ).length;

      // Act
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );

      // Assert
      expect(
        network.nodes.filter((node: Node) => node.type === 'output').length,
      ).toBe(outputModuleCount);
    });

    it('designates 0 < z < 1 modules as hidden nodes', () => {
      // Arrange
      const envelope = new NGE_DNA().toCanonical();
      const plan = createEmptyPlan();
      const descriptor: NgeRealizedPhenotypeDescriptor = {
        modules: [
          {
            moduleId: 'm:in',
            archetypeId: 'archetype:input',
            computationType: 'DenseFeedForward',
            coordinate: [0.5, 0.5, 0],
            zoneId: 'z:0:0:0',
            receivesCoordinates: false,
          },
          {
            moduleId: 'm:hid',
            archetypeId: 'archetype:hidden',
            computationType: 'GatedRecurrentCell',
            coordinate: [0.5, 0.5, 0.5],
            zoneId: 'z:0:0:0',
            receivesCoordinates: false,
          },
          {
            moduleId: 'm:out',
            archetypeId: 'archetype:output',
            computationType: 'DenseFeedForward',
            coordinate: [0.5, 0.5, 1],
            zoneId: 'z:0:0:1',
            receivesCoordinates: false,
          },
        ],
        edges: [],
        residualStreamAssignments: {},
        weightSharedCohortAssignments: {},
        phenotypeFingerprint: 'fingerprint:hidden',
        dnaFingerprint: 'dna:hidden',
        seed: 42,
      };

      // Act
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );

      // Assert
      expect(
        network.nodes.filter((node: Node) => node.type === 'hidden').length,
      ).toBe(1);
    });

    it('carries NGE descriptor and envelope in NetworkJSONExtensions.values', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();

      // Act
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );
      const json = network.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(json.extensions?.values).toMatchObject({
        ngeDescriptor: expect.any(Object),
        ngeEnvelope: expect.any(Object),
        version: 1,
      });
    });
  });

  describe('extractCanonicalEnvelopeFromNetwork', () => {
    it('extracts an NgeDnaCanonicalEnvelope with matching schemaVersion', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );

      // Act
      const extractedEnvelope = extractCanonicalEnvelopeFromNetwork(network);

      // Assert
      expect(extractedEnvelope.schemaVersion).toBe(envelope.schemaVersion);
    });

    it('extracts an NgeDnaCanonicalEnvelope with matching fingerprint', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        {
          ngeEnabled: true,
        },
      );

      // Act
      const extractedEnvelope = extractCanonicalEnvelopeFromNetwork(network);

      // Assert
      expect(extractedEnvelope.fingerprint).toBe(envelope.fingerprint);
    });

    it('throws when the Network carries no NGE extension bag', () => {
      // Arrange
      const classicNetwork = new Network(1, 1);

      // Assert
      expect(() =>
        extractCanonicalEnvelopeFromNetwork(classicNetwork),
      ).toThrow();
    });
  });

  describe('round-trip fidelity', () => {
    it('preserves node count across Network → envelope → Network round-trip', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const originalNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Act
      const extractedEnvelope =
        extractCanonicalEnvelopeFromNetwork(originalNetwork);
      const roundTrippedNetwork = materializeNetworkFromPhenotype(
        extractedEnvelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Assert
      expect(roundTrippedNetwork.nodes.length).toBe(
        originalNetwork.nodes.length,
      );
    });

    it('preserves connection count across round-trip', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const originalNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Act
      const extractedEnvelope =
        extractCanonicalEnvelopeFromNetwork(originalNetwork);
      const roundTrippedNetwork = materializeNetworkFromPhenotype(
        extractedEnvelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Assert
      expect(roundTrippedNetwork.connections.length).toBe(
        originalNetwork.connections.length,
      );
    });

    it('preserves phenotypeFingerprint across round-trip', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const originalNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Act
      const extractedEnvelope =
        extractCanonicalEnvelopeFromNetwork(originalNetwork);
      const roundTrippedNetwork = materializeNetworkFromPhenotype(
        extractedEnvelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );
      const roundTrippedJson =
        roundTrippedNetwork.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(
        (
          roundTrippedJson.extensions
            ?.values as unknown as BridgeExtensionValues
        ).ngeDescriptor.phenotypeFingerprint,
      ).toBe(descriptor.phenotypeFingerprint);
    });

    it('preserves dnaFingerprint across round-trip', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();
      const originalNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Act
      const extractedEnvelope =
        extractCanonicalEnvelopeFromNetwork(originalNetwork);
      const roundTrippedNetwork = materializeNetworkFromPhenotype(
        extractedEnvelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );
      const roundTrippedJson =
        roundTrippedNetwork.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(
        (
          roundTrippedJson.extensions
            ?.values as unknown as BridgeExtensionValues
        ).ngeDescriptor.dnaFingerprint,
      ).toBe(descriptor.dnaFingerprint);
    });

    it('preserves residualStreamAssignments across round-trip', () => {
      // Arrange
      const { envelope, plan, descriptor } = createShelfFixture();
      const originalNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Act
      const extractedEnvelope =
        extractCanonicalEnvelopeFromNetwork(originalNetwork);
      const roundTrippedNetwork = materializeNetworkFromPhenotype(
        extractedEnvelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );
      const roundTrippedJson =
        roundTrippedNetwork.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(
        (
          roundTrippedJson.extensions
            ?.values as unknown as BridgeExtensionValues
        ).ngeDescriptor.residualStreamAssignments,
      ).toEqual(descriptor.residualStreamAssignments);
    });

    it('preserves weightSharedCohortAssignments across round-trip', () => {
      // Arrange
      const { envelope, plan, descriptor } = createShelfFixture();
      const originalNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Act
      const extractedEnvelope =
        extractCanonicalEnvelopeFromNetwork(originalNetwork);
      const roundTrippedNetwork = materializeNetworkFromPhenotype(
        extractedEnvelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );
      const roundTrippedJson =
        roundTrippedNetwork.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(
        (
          roundTrippedJson.extensions
            ?.values as unknown as BridgeExtensionValues
        ).ngeDescriptor.weightSharedCohortAssignments,
      ).toEqual(descriptor.weightSharedCohortAssignments);
    });
  });

  describe('determinism', () => {
    it('produces identical Network topology for same DNA and same seed', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();

      // Act
      const firstNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );
      const secondNetwork = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
        { ngeEnabled: true },
      );

      // Assert
      expect(firstNetwork.toJSON() as unknown as NetworkJSON).toEqual(
        secondNetwork.toJSON() as unknown as NetworkJSON,
      );
    });

    it('produces identical topology for same DNA with different seeds', () => {
      // Arrange
      const seedOne = 42;
      const seedTwo = 99;
      const dna = createBridgeDna();
      const envelope = dna.toCanonical();
      const planOne = dna.buildVirtualPlan(seedOne);
      const descriptorOne = dna.realizePhenotype(planOne, seedOne);
      const planTwo = dna.buildVirtualPlan(seedTwo);
      const descriptorTwo = dna.realizePhenotype(planTwo, seedTwo);

      // Act
      const firstNetwork = materializeNetworkFromPhenotype(
        envelope,
        planOne,
        descriptorOne,
        { ngeEnabled: true },
      );
      const secondNetwork = materializeNetworkFromPhenotype(
        envelope,
        planTwo,
        descriptorTwo,
        { ngeEnabled: true },
      );

      // Assert
      expect(firstNetwork.nodes.length).toBe(secondNetwork.nodes.length);
    });
  });

  describe('opt-in isolation', () => {
    it('does not attach NGE extension properties when ngeEnabled is not true', () => {
      // Arrange
      const { envelope, plan, descriptor } = createBridgeFixture();

      // Act
      const network = materializeNetworkFromPhenotype(
        envelope,
        plan,
        descriptor,
      );
      const json = network.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(json.extensions?.values?.ngeDescriptor).toBeUndefined();
    });

    it('leaves classic NEAT Network unchanged when bridge is not invoked', () => {
      // Arrange
      const classicNetwork = new Network(1, 1);

      // Act
      const json = classicNetwork.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(json.extensions?.values?.ngeDescriptor).toBeUndefined();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Fixture helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Shape of the NGE extension carrier inside NetworkJSONExtensions.values. */
interface BridgeExtensionValues {
  version: number;
  ngeDescriptor: NgeRealizedPhenotypeDescriptor;
  ngeEnvelope: NgeDnaCanonicalEnvelope;
}

/** Rule passes placing one module at z=0 (input) and one at z=1 (output). */
function createBridgeRulePasses(): NgeRulePass[] {
  return [
    {
      archetypeId: 'archetype:input',
      kind: 'replicate',
      placements: [
        {
          computationType: 'DenseFeedForward',
          coordinate: [0.5, 0.5, 0],
        },
      ],
      priority: 0,
    },
    {
      archetypeId: 'archetype:output',
      kind: 'replicate',
      placements: [
        {
          computationType: 'DenseFeedForward',
          coordinate: [0.5, 0.5, 1],
        },
      ],
      priority: 1,
    },
  ];
}

/** CPPN program that always outputs a constant weight from the 'weight' node. */
function createConstantWeightCppnProgram(weight: number): NgeCppnProgram {
  return {
    edges: [],
    hiddenNodes: [
      { activationKind: 'linear', bias: weight, nodeId: 'weight' },
      { activationKind: 'linear', bias: 0, nodeId: 'enableBias' },
    ],
    inputNodeIds: ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'dist'],
    outputNodeIds: ['weight', 'enableBias'],
    programId: `cppn:constant:${weight}`,
  };
}

/** NGE_DNA instance with z=0 input and z=1 output modules plus constant-weight CPPN. */
function createBridgeDna(): NGE_DNA {
  return new NGE_DNA({
    cppnPrograms: [createConstantWeightCppnProgram(0.5)],
    moduleArchetypes: [
      { archetypeId: 'archetype:input', computationType: 'DenseFeedForward' },
      { archetypeId: 'archetype:output', computationType: 'DenseFeedForward' },
    ],
    rulePasses: createBridgeRulePasses(),
  });
}

/** Full bridge fixture using the real NGE_DNA pipeline (buildVirtualPlan + realizePhenotype). */
function createBridgeFixture(): {
  envelope: NgeDnaCanonicalEnvelope;
  plan: NgeVirtualModulePlan;
  descriptor: NgeRealizedPhenotypeDescriptor;
} {
  const seed = 42;
  const dna = createBridgeDna();
  const plan = dna.buildVirtualPlan(seed);
  const descriptor = dna.realizePhenotype(plan, seed);
  return { envelope: dna.toCanonical(), plan, descriptor };
}

/** Empty virtual module plan for the zero-module throw test. */
function createEmptyPlan(): NgeVirtualModulePlan {
  return {
    modules: [],
    planFingerprint: 'plan:empty',
    substrateFingerprint: 'substrate:empty',
  };
}

/** Descriptor with zero modules and zero edges. */
function createEmptyDescriptor(): NgeRealizedPhenotypeDescriptor {
  return {
    modules: [],
    edges: [],
    residualStreamAssignments: {},
    weightSharedCohortAssignments: {},
    phenotypeFingerprint: 'fingerprint:empty',
    dnaFingerprint: 'dna:empty',
    seed: 0,
  };
}

/** Synthetic fixture with residual-stream and weight-shared-cohort shelves populated. */
function createShelfFixture(): {
  envelope: NgeDnaCanonicalEnvelope;
  plan: NgeVirtualModulePlan;
  descriptor: NgeRealizedPhenotypeDescriptor;
} {
  const envelope = new NGE_DNA().toCanonical();
  const plan: NgeVirtualModulePlan = {
    modules: [],
    planFingerprint: 'plan:shelf',
    substrateFingerprint: 'substrate:shelf',
  };
  const descriptor: NgeRealizedPhenotypeDescriptor = {
    modules: [
      {
        moduleId: 'module:shelf-a',
        archetypeId: 'archetype:shelf',
        computationType: 'DenseFeedForward',
        coordinate: [0.5, 0.5, 0],
        zoneId: 'z:0:0:0',
        receivesCoordinates: false,
        residualStreamId: 'stream:main',
        weightSharedCohortId: 'cohort:main',
      },
      {
        moduleId: 'module:shelf-b',
        archetypeId: 'archetype:shelf',
        computationType: 'DenseFeedForward',
        coordinate: [0.5, 0.5, 1],
        zoneId: 'z:0:0:1',
        receivesCoordinates: false,
        residualStreamId: 'stream:main',
        weightSharedCohortId: 'cohort:main',
      },
    ],
    edges: [],
    residualStreamAssignments: {
      'stream:main': ['module:shelf-a', 'module:shelf-b'],
    },
    weightSharedCohortAssignments: {
      'cohort:main': ['module:shelf-a', 'module:shelf-b'],
    },
    phenotypeFingerprint: 'fingerprint:shelf',
    dnaFingerprint: 'dna:shelf',
    seed: 42,
  };
  return { envelope, plan, descriptor };
}
