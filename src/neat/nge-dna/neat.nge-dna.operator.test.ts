import type { NetworkJSON } from '../../architecture/network/network.types';
import { NGE_DNA } from './neat.nge-dna';
import type { NgeCppnProgram, NgeRulePass } from './neat.nge-dna.types';

describe('NGE envelope → Network operator', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('activateNgeNetworkFromEnvelope', () => {
    it('exports a callable function named activateNgeNetworkFromEnvelope', async () => {
      // Act
      const { activateNgeNetworkFromEnvelope } = await import(
        './neat.nge-dna.operator' as string
      );

      // Assert
      expect(typeof activateNgeNetworkFromEnvelope).toBe('function');
    });

    it('attaches the NGE extension carrier when modeIsEvolvable is true', async () => {
      // Arrange
      const dna = createOperatorDna({ modeIsEvolvable: true });
      const envelope = dna.toCanonical();
      const seed = 42;
      const { activateNgeNetworkFromEnvelope } = await import(
        './neat.nge-dna.operator' as string
      );

      // Act
      const network = activateNgeNetworkFromEnvelope(envelope, seed);
      const json = network.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(json.extensions?.values?.ngeEnvelope).toBeDefined();
    });

    it('calls NGE_DNA.buildVirtualPlan with the provided seed when modeIsEvolvable is true', async () => {
      // Arrange
      const buildPlanSpy = jest.spyOn(NGE_DNA.prototype, 'buildVirtualPlan');
      const dna = createOperatorDna({ modeIsEvolvable: true });
      const envelope = dna.toCanonical();
      const seed = 42;
      const { activateNgeNetworkFromEnvelope } = await import(
        './neat.nge-dna.operator' as string
      );

      // Act
      activateNgeNetworkFromEnvelope(envelope, seed);

      // Assert
      expect(buildPlanSpy).toHaveBeenCalledWith(seed);
    });

    it('calls NGE_DNA.realizePhenotype with the plan and seed when modeIsEvolvable is true', async () => {
      // Arrange
      const dna = createOperatorDna({ modeIsEvolvable: true });
      const envelope = dna.toCanonical();
      const seed = 42;
      const expectedPlan = dna.buildVirtualPlan(seed);
      const realizePhenotypeSpy = jest.spyOn(
        NGE_DNA.prototype,
        'realizePhenotype',
      );
      const { activateNgeNetworkFromEnvelope } = await import(
        './neat.nge-dna.operator' as string
      );

      // Act
      activateNgeNetworkFromEnvelope(envelope, seed);

      // Assert
      expect(realizePhenotypeSpy).toHaveBeenCalledWith(expectedPlan, seed);
    });

    it('does not attach NGE extensions when modeIsEvolvable is false', async () => {
      // Arrange
      const dna = createOperatorDna({ modeIsEvolvable: false });
      const envelope = dna.toCanonical();
      const seed = 42;
      const { activateNgeNetworkFromEnvelope } = await import(
        './neat.nge-dna.operator' as string
      );

      // Act
      const network = activateNgeNetworkFromEnvelope(envelope, seed);
      const json = network.toJSON() as unknown as NetworkJSON;

      // Assert
      expect(json.extensions).toBeUndefined();
    });

    it('produces identical toJSON output for the same envelope and seed', async () => {
      // Arrange
      const dna = createOperatorDna({ modeIsEvolvable: true });
      const envelope = dna.toCanonical();
      const seed = 42;
      const { activateNgeNetworkFromEnvelope } = await import(
        './neat.nge-dna.operator' as string
      );

      // Act
      const firstNetwork = activateNgeNetworkFromEnvelope(envelope, seed);
      const secondNetwork = activateNgeNetworkFromEnvelope(envelope, seed);

      // Assert
      expect(firstNetwork.toJSON() as unknown as NetworkJSON).toEqual(
        secondNetwork.toJSON() as unknown as NetworkJSON,
      );
    });

    it('throws when the envelope yields a phenotype descriptor with zero modules', async () => {
      // Arrange
      const envelope = new NGE_DNA().toCanonical();
      const seed = 42;
      const { activateNgeNetworkFromEnvelope } = await import(
        './neat.nge-dna.operator' as string
      );

      // Assert
      expect(() => activateNgeNetworkFromEnvelope(envelope, seed)).toThrow();
    });
  });
});

function createOperatorDna(options: { modeIsEvolvable: boolean }): NGE_DNA {
  return new NGE_DNA({
    cppnPrograms: [createConstantWeightCppnProgram(0.5)],
    moduleArchetypes: [
      { archetypeId: 'archetype:input', computationType: 'DenseFeedForward' },
      { archetypeId: 'archetype:output', computationType: 'DenseFeedForward' },
    ],
    reproductionPolicy: { modeIsEvolvable: options.modeIsEvolvable },
    rulePasses: createBridgeRulePasses(),
  });
}

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

function createBridgeRulePasses(): NgeRulePass[] {
  return [
    {
      archetypeId: 'archetype:input',
      kind: 'replicate',
      placements: [
        { computationType: 'DenseFeedForward', coordinate: [0.5, 0.5, 0] },
      ],
      priority: 0,
    },
    {
      archetypeId: 'archetype:output',
      kind: 'replicate',
      placements: [
        { computationType: 'DenseFeedForward', coordinate: [0.5, 0.5, 1] },
      ],
      priority: 1,
    },
  ];
}
