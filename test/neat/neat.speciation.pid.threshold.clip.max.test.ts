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
  population: GenomeDetailed[],
  overrides: Partial<SpeciationPidContext['options']>,
  compatIntegral: number,
): SpeciationPidContext => {
  const defaultOptions: SpeciationPidContext['options'] = {
    speciation: true,
    targetSpecies: 1,
    compatibilityThreshold: 9.9,
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
    population,
    _species: [],
    _nextSpeciesId: 1,
    generation: 0,
    options: { ...defaultOptions, ...overrides },
    _speciesCreated: new Map<number, number>(),
    _prevSpeciesMembers: new Map<number, Set<number>>(),
    _speciesLastStats: new Map<
      number,
      { meanNodes: number; meanConns: number; best: number }
    >(),
    _speciesHistory: [],
    _compatIntegral: compatIntegral,
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
  test('clips compatibility threshold at maximum and resets integral', () => {
    // Arrange
    const genomes: GenomeDetailed[] = [
      { nodes: [], connections: [], _id: 1, score: 1 },
      { nodes: [], connections: [], _id: 2, score: 1 },
    ];
    const ctx = buildPidContext(genomes, {}, -100); // ensure reset when clipping top

    // Act
    _speciate.call(ctx);

    // Assert
    expect(ctx._compatIntegral).toBe(0);
  });
});
