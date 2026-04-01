import Network from '../../../architecture/network';
import Neat from '../../../neat';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
  SpeciesWithMetadata,
} from '../evolve.types';
import {
  addSpeciatedOffspring,
  applyProvenance,
} from './evolve.population.utils';

type PopulationAllocationSnapshot = Array<{ id: number; alloc: number }>;
type PopulationAllocationHost = Neat & {
  _lastOffspringAlloc: PopulationAllocationSnapshot;
};
type PopulationMetadataNetwork = Network & GenomeWithMetadata;

function createPopulationMember(input: {
  genomeId: number;
  score: number;
  depth: number;
}): PopulationMetadataNetwork {
  const populationMember = new Network(3, 1) as PopulationMetadataNetwork;

  populationMember._id = input.genomeId;
  populationMember.score = input.score;
  populationMember._depth = input.depth;

  return populationMember;
}

function countHiddenNodes(genome: Network): number {
  return genome.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden').length;
}

function createSpeciesSnapshot(input: {
  speciesId: number;
  members: PopulationMetadataNetwork[];
  lastImproved?: number;
}): SpeciesWithMetadata {
  return {
    id: input.speciesId,
    members: input.members,
    generation: 0,
    lastImproved: input.lastImproved ?? 0,
  };
}

function createDeterministicRandom(input: { values: number[] }): () => number {
  let randomIndex = 0;

  return () => {
    const nextValue = input.values[randomIndex] ?? input.values.at(-1) ?? 0;

    randomIndex += 1;
    return nextValue;
  };
}

function createEvolutionController(input: {
  species: SpeciesWithMetadata[];
  randomValues: number[];
  generation?: number;
  crossSpeciesMatingProb?: number;
  popsize?: number;
  provenance?: number;
  minHidden?: number;
  speciesAgeBonus?: NeatControllerForEvolution['options']['speciesAgeBonus'];
}): NeatControllerForEvolution {
  const randomSource = createDeterministicRandom({
    values: input.randomValues,
  });
  const population = input.species.flatMap((species) => species.members);

  return {
    input: 3,
    output: 1,
    population,
    generation: input.generation ?? 0,
    options: {
      survivalThreshold: 1,
      crossSpeciesMatingProb: input.crossSpeciesMatingProb ?? 1,
      equal: false,
      mutation: [],
      popsize: input.popsize ?? population.length,
      provenance: input.provenance ?? 0,
      minHidden: input.minHidden,
      reenableProb: 0.25,
      speciesAllocation: {
        minOffspring: 0,
      },
      speciesAgeBonus: input.speciesAgeBonus ?? {
        youngThreshold: 0,
        youngMultiplier: 1,
        oldThreshold: 100,
        oldMultiplier: 1,
      },
    },
    _bestGlobalScore: 0,
    _getRNG: () => randomSource,
    _species: input.species,
    _paretoArchive: [],
    _paretoObjectivesArchive: [],
    _lastEpsilonAdjustGen: 0,
    _objectiveStale: new Map(),
    _pendingObjectiveAdds: [],
    _pendingObjectiveRemoves: [],
    _objectiveAges: new Map(),
    _lastOffspringAlloc: [],
    _prevInbreedingCount: 0,
    _lastInbreedingCount: 0,
    _nextGenomeId: 900,
    _lineageEnabled: true,
    _sortSpeciesMembers: (species) => {
      species.members.sort(
        (leftMember, rightMember) =>
          (rightMember.score ?? 0) - (leftMember.score ?? 0),
      );
    },
    _updateSpeciesStagnation: () => {},
    _lastEvolveDuration: 0,
    evaluate: async () => {},
    sort: () => {},
    mutate: async () => {},
    getOffspring: async () => population[0] as GenomeWithMetadata,
    selectParent: () => population[0] as GenomeWithMetadata,
    registerObjective: () => {},
    ensureMinHiddenNodes: async () => {},
    ensureNoDeadEnds: () => {},
  };
}

describe('neat evolve population chapter', () => {
  describe('species-aware offspring allocation', () => {
    const scoreByNodeCount = (network: Network) => network.nodes.length;

    describe('given speciated reproduction with a one-child minimum per surviving species', () => {
      describe('when the latest allocation snapshot is inspected after several generations', () => {
        it('keeps every recorded species allocation at or above that minimum', async () => {
          // Arrange
          const neat = new Neat(3, 2, scoreByNodeCount, {
            popsize: 50,
            seed: 500,
            targetSpecies: 6,
            speciation: true,
            speciesAllocation: { minOffspring: 1, extendedHistory: false },
          }) as PopulationAllocationHost;

          await neat.evaluate();
          for (
            let generationIndex = 0;
            generationIndex < 3;
            generationIndex++
          ) {
            await neat.evolve();
          }

          // Act
          const lastOffspringAllocation = neat._lastOffspringAlloc;

          // Assert
          expect(
            lastOffspringAllocation.every(
              (allocationEntry) => allocationEntry.alloc >= 1,
            ),
          ).toBe(true);
        });
      });
    });

    describe('given speciated evolve with elitism already reserving part of the population budget', () => {
      describe('when the next generation finishes rebuilding', () => {
        it('still restores the configured population size exactly', async () => {
          // Arrange
          const configuredPopulationSize = 100;
          const neat = new Neat(3, 2, scoreByNodeCount, {
            popsize: configuredPopulationSize,
            elitism: 10,
            targetSpecies: 34,
            seed: 501,
            speciation: true,
            speciesAllocation: { minOffspring: 1, extendedHistory: false },
          });

          await neat.evaluate();

          // Act
          await neat.evolve();

          // Assert
          expect(neat.population.length).toBe(configuredPopulationSize);
        });
      });
    });

    describe('given equally fit young and mature species with a young-species bonus configured', () => {
      describe('when the latest offspring allocation snapshot is recorded', () => {
        it('assigns more offspring slots to the younger species', async () => {
          // Arrange
          const youngSpeciesLeader = createPopulationMember({
            genomeId: 301,
            score: 10,
            depth: 1,
          });
          const matureSpeciesLeader = createPopulationMember({
            genomeId: 401,
            score: 10,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 1,
                members: [youngSpeciesLeader],
                lastImproved: 6,
              }),
              createSpeciesSnapshot({
                speciesId: 2,
                members: [matureSpeciesLeader],
                lastImproved: 2,
              }),
            ],
            randomValues: [0],
            generation: 6,
            crossSpeciesMatingProb: 0,
            speciesAgeBonus: {
              youngThreshold: 1,
              youngMultiplier: 2,
              oldThreshold: 100,
              oldMultiplier: 1,
            },
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(evolutionController, nextPopulation, 5, {
            minOffspringDefault: 0,
            survivalThresholdDefault: 1,
            youngThresholdDefault: 0,
            youngMultiplierDefault: 1,
            oldThresholdDefault: 100,
            oldMultiplierDefault: 1,
            crossSpeciesGuardLimit: 4,
          });

          // Assert
          expect(evolutionController._lastOffspringAlloc).toEqual([
            { id: 1, alloc: 3 },
            { id: 2, alloc: 2 },
          ]);
        });
      });
    });

    describe('given species-aware reproduction with cross-species mating forced on', () => {
      describe('when one offspring is bred from the leading species budget', () => {
        it('records parent ids from two different species on the created child', async () => {
          // Arrange
          const firstSpeciesParent = createPopulationMember({
            genomeId: 101,
            score: 5,
            depth: 1,
          });
          const secondSpeciesParent = createPopulationMember({
            genomeId: 202,
            score: 4,
            depth: 3,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 1,
                members: [firstSpeciesParent],
              }),
              createSpeciesSnapshot({
                speciesId: 2,
                members: [secondSpeciesParent],
              }),
            ],
            randomValues: [0, 0, 0.75, 0],
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(evolutionController, nextPopulation, 1, {
            minOffspringDefault: 0,
            survivalThresholdDefault: 1,
            youngThresholdDefault: 0,
            youngMultiplierDefault: 1,
            oldThresholdDefault: 100,
            oldMultiplierDefault: 1,
            crossSpeciesGuardLimit: 4,
          });

          // Assert
          expect(
            (nextPopulation[0] as PopulationMetadataNetwork)._parents,
          ).toEqual([101, 202]);
        });
      });
    });
  });

  describe('provenance seeding', () => {
    describe('given fresh provenance genomes are built without a seed network', () => {
      describe('when a minimum hidden size is configured', () => {
        it('creates provenance genomes that already satisfy that hidden floor', () => {
          // Arrange
          const minimumHiddenCount = 4;
          const evolutionController = createEvolutionController({
            species: [],
            randomValues: [0],
            popsize: 2,
            provenance: 2,
            minHidden: minimumHiddenCount,
          });
          const nextPopulation: Network[] = [];

          // Act
          applyProvenance(evolutionController, nextPopulation);
          const hiddenNodeCounts = nextPopulation.map((genome) =>
            countHiddenNodes(genome),
          );

          // Assert
          expect(
            hiddenNodeCounts.every(
              (hiddenNodeCount) => hiddenNodeCount >= minimumHiddenCount,
            ),
          ).toBe(true);
        });
      });
    });
  });
});
