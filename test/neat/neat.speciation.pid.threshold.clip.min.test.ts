import { _speciate } from '../../src/neat/neat.speciation';
import type {
  ConnectionLike,
  GenomeDetailed,
  SpeciationOptions,
  SpeciesLike,
} from '../../src/neat/neat.types';

type SpeciationPidContext = {
  population: GenomeDetailed[];
  _species: SpeciesLike[];
  _nextSpeciesId: number;
  generation: number;
  options: SpeciationOptions & {
    compatibilityThreshold: number;
    compatAdjust: Required<NonNullable<SpeciationOptions['compatAdjust']>>;
  };
  _speciesCreated: Map<number, number>;
  _prevSpeciesMembers: Map<number, Set<number>>;
  _speciesLastStats: Map<
    number,
    { meanNodes: number; meanConns: number; best: number }
  >;
  _speciesHistory: Array<Record<string, unknown>>;
  _compatIntegral: number;
  _getRNG: () => () => number;
  _compatibilityDistance: (
    genomeA: GenomeDetailed,
    genomeB: GenomeDetailed,
  ) => number;
  _fallbackInnov: (connection: ConnectionLike) => number;
  _structuralEntropy: (genome: GenomeDetailed) => number;
};

const buildPidContext = (
  overrides: Partial<SpeciationPidContext['options']>,
): SpeciationPidContext => {
  const genome: GenomeDetailed = {
    nodes: [],
    connections: [],
    _id: 1,
    score: 1,
  };
  const options: SpeciationPidContext['options'] = {
    speciation: true,
    targetSpecies: 5,
    compatibilityThreshold: 0.6,
    compatAdjust: {
      kp: 1,
      ki: 0.5,
      smoothingWindow: 1,
      minThreshold: 0.5,
      maxThreshold: 10,
      decay: 1,
    },
  };
  return {
    population: [genome],
    _species: [],
    _nextSpeciesId: 1,
    generation: 0,
    options: { ...options, ...overrides },
    _speciesCreated: new Map<number, number>(),
    _prevSpeciesMembers: new Map<number, Set<number>>(),
    _speciesLastStats: new Map<
      number,
      { meanNodes: number; meanConns: number; best: number }
    >(),
    _speciesHistory: [],
    _compatIntegral: 100, // ensure reset when clipping
    _getRNG: () => () => 0.5,
    _compatibilityDistance: (
      genomeA: GenomeDetailed,
      genomeB: GenomeDetailed,
    ) => {
      void genomeA;
      void genomeB;
      return 0;
    },
    _fallbackInnov: (_connection: ConnectionLike) => 1,
    _structuralEntropy: (genome: GenomeDetailed) => {
      void genome;
      return 0;
    },
  };
};

describe('speciation - pid controller', () => {
  test('clips compatibility threshold at minimum and resets integral', () => {
    // Arrange
    const ctx = buildPidContext({});

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx._compatIntegral).toBe(0);
  });
});
