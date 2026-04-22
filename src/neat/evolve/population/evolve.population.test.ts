import Network from '../../../architecture/network';
import Neat from '../../../neat';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
  SpeciesWithMetadata,
} from '../evolve.types';
import {
  addOffspring,
  addSpeciatedOffspring,
  addUnspeciatedOffspring,
  applyElitism,
  applyProvenance,
  enforcePopulationConstraints,
} from './evolve.population.utils';

type PopulationAllocationSnapshot = Array<{ id: number; alloc: number }>;
type PopulationAllocationHost = Neat & {
  _lastOffspringAlloc: PopulationAllocationSnapshot;
};
type PopulationMetadataNetwork = Network & GenomeWithMetadata;
type PopulationOffspringConfig = Parameters<typeof addSpeciatedOffspring>[3];

type PopulationRandomCarrier = {
  _rand?: () => number;
};

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

function setPopulationRandomSource(
  network: Network,
  randomGenerator: () => number,
): void {
  (network as unknown as PopulationRandomCarrier)._rand = randomGenerator;
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
  network?: Network;
  speciation?: { enabled?: boolean };
  elitism?: number;
  survivalThreshold?: number;
  minOffspring?: number;
  lineageEnabled?: boolean;
  speciesAgeBonus?: NeatControllerForEvolution['options']['speciesAgeBonus'];
  ensureMinHiddenNodes?: NeatControllerForEvolution['ensureMinHiddenNodes'];
  ensureNoDeadEnds?: NeatControllerForEvolution['ensureNoDeadEnds'];
  getOffspring?: NeatControllerForEvolution['getOffspring'];
  speciate?: NeatControllerForEvolution['_speciate'];
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
      survivalThreshold: input.survivalThreshold ?? 1,
      crossSpeciesMatingProb: input.crossSpeciesMatingProb ?? 1,
      equal: false,
      mutation: [],
      popsize: input.popsize ?? population.length,
      provenance: input.provenance ?? 0,
      minHidden: input.minHidden,
      network: input.network,
      speciation: input.speciation,
      elitism: input.elitism,
      reenableProb: 0.25,
      speciesAllocation: {
        minOffspring: input.minOffspring ?? 0,
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
    _lineageEnabled: input.lineageEnabled ?? true,
    _sortSpeciesMembers: (species) => {
      species.members.sort(
        (leftMember, rightMember) =>
          (rightMember.score ?? 0) - (leftMember.score ?? 0),
      );
    },
    _speciate: input.speciate,
    _updateSpeciesStagnation: () => {},
    _lastEvolveDuration: 0,
    evaluate: async () => {},
    sort: () => {},
    mutate: async () => {},
    getOffspring:
      input.getOffspring ??
      (async () => population[0] as GenomeWithMetadata),
    selectParent: () => population[0] as GenomeWithMetadata,
    registerObjective: () => {},
    ensureMinHiddenNodes: input.ensureMinHiddenNodes ?? (async () => {}),
    ensureNoDeadEnds: input.ensureNoDeadEnds ?? (() => {}),
  };
}

function createPopulationOffspringConfig(
  input?: Partial<PopulationOffspringConfig>,
): PopulationOffspringConfig {
  return {
    minOffspringDefault: 0,
    survivalThresholdDefault: 1,
    youngThresholdDefault: 0,
    youngMultiplierDefault: 1,
    oldThresholdDefault: 100,
    oldMultiplierDefault: 1,
    crossSpeciesGuardLimit: 4,
    ...input,
  };
}

describe('neat evolve population chapter', () => {
  describe('population assembly helpers', () => {
    describe('enforcePopulationConstraints', () => {
      describe('given the next population includes one empty slot and one genome', () => {
        it('repairs only the defined genome once', async () => {
          // Arrange
          const genome = createPopulationMember({
            genomeId: 11,
            score: 1,
            depth: 1,
          });
          const ensureMinHiddenNodes = jest.fn(async () => undefined);
          const ensureNoDeadEnds = jest.fn(() => undefined);
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 1,
                members: [genome],
              }),
            ],
            randomValues: [0],
            ensureMinHiddenNodes,
            ensureNoDeadEnds,
          });
          const nextPopulation = [undefined as never as Network, genome];

          // Act
          await enforcePopulationConstraints(evolutionController, nextPopulation);

          // Assert
          expect({
            minHiddenCalls: ensureMinHiddenNodes.mock.calls,
            noDeadEndsCalls: ensureNoDeadEnds.mock.calls,
          }).toEqual({
            minHiddenCalls: [[genome]],
            noDeadEndsCalls: [[genome]],
          });
        });
      });
    });

    describe('applyElitism', () => {
      describe('given the elite window contains an empty population slot', () => {
        it('skips the empty slot while preserving the live elite', () => {
          // Arrange
          const eliteGenome = createPopulationMember({
            genomeId: 21,
            score: 5,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 1,
                members: [eliteGenome],
              }),
            ],
            randomValues: [0],
            elitism: 2,
          });
          evolutionController.population = [
            undefined as never as GenomeWithMetadata,
            eliteGenome,
          ] as never;
          const nextPopulation: Network[] = [];

          // Act
          applyElitism(evolutionController, nextPopulation);

          // Assert
          expect(nextPopulation).toEqual([eliteGenome]);
        });
      });
    });

    describe('applyProvenance', () => {
      describe('given no population budget is configured for provenance', () => {
        it('leaves the next population unchanged', () => {
          // Arrange
          const evolutionController = createEvolutionController({
            species: [],
            randomValues: [0],
          });
          const nextPopulation: Network[] = [];

          // Act
          applyProvenance(evolutionController, nextPopulation);

          // Assert
          expect(nextPopulation).toEqual([]);
        });
      });

      describe('given a seed network is configured for provenance cloning', () => {
        it('clones the seed network into the remaining provenance slots', () => {
          // Arrange
          const seedNetwork = new Network(3, 1, { minHidden: 2 });
          const expectedSeedJson = seedNetwork.toJSON();
          const evolutionController = createEvolutionController({
            species: [],
            randomValues: [0],
            popsize: 1,
            provenance: 1,
            network: seedNetwork,
          });
          const nextPopulation: Network[] = [];

          // Act
          applyProvenance(evolutionController, nextPopulation);

          // Assert
          expect({
            populationLength: nextPopulation.length,
            reusedOriginalSeed: nextPopulation[0] === seedNetwork,
            clonedSeedJson: nextPopulation[0]?.toJSON(),
          }).toEqual({
            populationLength: 1,
            reusedOriginalSeed: false,
            clonedSeedJson: expectedSeedJson,
          });
        });
      });
    });

    describe('addOffspring', () => {
      describe('given the next population already fills the configured budget', () => {
        it('returns without invoking either offspring helper', async () => {
          // Arrange
          const existingGenome = createPopulationMember({
            genomeId: 30,
            score: 4,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 30,
                members: [existingGenome],
              }),
            ],
            randomValues: [0],
          });
          const addSpeciatedOffspring = jest.fn(async () => undefined);
          const addUnspeciatedOffspring = jest.fn(async () => undefined);
          const nextPopulation: Network[] = [existingGenome];
          evolutionController.options.popsize = undefined;

          // Act
          await addOffspring(evolutionController, nextPopulation, {
            addSpeciatedOffspring,
            addUnspeciatedOffspring,
          });

          // Assert
          expect({
            speciatedCalls: addSpeciatedOffspring.mock.calls.length,
            unspeciatedCalls: addUnspeciatedOffspring.mock.calls.length,
          }).toEqual({
            speciatedCalls: 0,
            unspeciatedCalls: 0,
          });
        });
      });

      describe('given speciation is enabled and a stale registry can be rebuilt', () => {
        it('refreshes the registry and uses the speciated offspring helper', async () => {
          // Arrange
          const speciesLeader = createPopulationMember({
            genomeId: 31,
            score: 6,
            depth: 1,
          });
          const refreshedSpecies = createSpeciesSnapshot({
            speciesId: 1,
            members: [speciesLeader],
          });
          const evolutionController = createEvolutionController({
            species: [refreshedSpecies],
            randomValues: [0],
            popsize: 1,
            speciation: { enabled: true },
          });
          const speciate = jest.fn(() => {
            evolutionController._species = [refreshedSpecies];
          });
          evolutionController._species = [];
          evolutionController._speciate = speciate;
          const addSpeciatedOffspring = jest.fn(async () => undefined);
          const addUnspeciatedOffspring = jest.fn(async () => undefined);
          const nextPopulation: Network[] = [];

          // Act
          await addOffspring(evolutionController, nextPopulation, {
            addSpeciatedOffspring,
            addUnspeciatedOffspring,
          });

          // Assert
          expect({
            speciateCalls: speciate.mock.calls.length,
            speciatedCalls: addSpeciatedOffspring.mock.calls.length,
            unspeciatedCalls: addUnspeciatedOffspring.mock.calls.length,
            suppressTournamentError: evolutionController._suppressTournamentError,
          }).toEqual({
            speciateCalls: 1,
            speciatedCalls: 1,
            unspeciatedCalls: 0,
            suppressTournamentError: false,
          });
        });
      });

      describe('given speciation is enabled but no usable registry exists', () => {
        it('falls back to the unspeciated offspring helper', async () => {
          // Arrange
          const speciesLeader = createPopulationMember({
            genomeId: 41,
            score: 6,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 1,
                members: [speciesLeader],
              }),
            ],
            randomValues: [0],
            popsize: 1,
            speciation: { enabled: true },
          });
          evolutionController._species = undefined;
          evolutionController._speciate = undefined;
          const addSpeciatedOffspring = jest.fn(async () => undefined);
          const addUnspeciatedOffspring = jest.fn(async () => undefined);
          const nextPopulation: Network[] = [];

          // Act
          await addOffspring(evolutionController, nextPopulation, {
            addSpeciatedOffspring,
            addUnspeciatedOffspring,
          });

          // Assert
          expect({
            speciatedCalls: addSpeciatedOffspring.mock.calls.length,
            unspeciatedCalls: addUnspeciatedOffspring.mock.calls.length,
            suppressTournamentError: evolutionController._suppressTournamentError,
          }).toEqual({
            speciatedCalls: 0,
            unspeciatedCalls: 1,
            suppressTournamentError: false,
          });
        });
      });

      describe('given no species registry is active', () => {
        it('fills every remaining slot through global offspring selection', async () => {
          // Arrange
          const firstOffspring = createPopulationMember({
            genomeId: 42,
            score: 2,
            depth: 1,
          });
          const secondOffspring = createPopulationMember({
            genomeId: 43,
            score: 3,
            depth: 2,
          });
          const offspringQueue = [firstOffspring, secondOffspring];
          const evolutionController = createEvolutionController({
            species: [],
            randomValues: [0],
            getOffspring: async () => offspringQueue.shift() as GenomeWithMetadata,
          });
          const nextPopulation: Network[] = [];

          // Act
          await addUnspeciatedOffspring(evolutionController, nextPopulation, 2);

          // Assert
          expect(nextPopulation).toEqual([firstOffspring, secondOffspring]);
        });
      });
    });
  });

  describe('species-aware offspring allocation', () => {
    describe('given the species registry disappears before allocation is recorded', () => {
      describe('when species-aware offspring filling is requested', () => {
        it('records no allocation snapshot and adds no offspring', async () => {
          // Arrange
          const evolutionController = createEvolutionController({
            species: [],
            randomValues: [0],
            popsize: 1,
          });
          evolutionController._species = undefined;
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect({
            populationLength: nextPopulation.length,
            lastAllocation: evolutionController._lastOffspringAlloc,
          }).toEqual({
            populationLength: 0,
            lastAllocation: [],
          });
        });
      });
    });

    describe('given the species registry changes after allocations are recorded', () => {
      describe('when breeding begins for the recorded budget', () => {
        it('skips breeding for species entries that disappeared', async () => {
          // Arrange
          const mutableSpeciesLeader = createPopulationMember({
            genomeId: 44,
            score: 7,
            depth: 1,
          });
          const mutableSpecies = createSpeciesSnapshot({
            speciesId: 24,
            members: [mutableSpeciesLeader],
          });
          const evolutionController = createEvolutionController({
            species: [mutableSpecies],
            randomValues: [0],
            popsize: 1,
            crossSpeciesMatingProb: 0,
          });
          Object.defineProperty(mutableSpecies, 'id', {
            configurable: true,
            get: () => {
              evolutionController._species = [];
              return 24;
            },
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect({
            populationLength: nextPopulation.length,
            lastAllocation: evolutionController._lastOffspringAlloc,
          }).toEqual({
            populationLength: 0,
            lastAllocation: [{ id: 24, alloc: 1 }],
          });
        });
      });
    });

    describe('given all recorded species are empty', () => {
      describe('when species-aware offspring filling is requested', () => {
        it('records zero allocation for every species and adds no offspring', async () => {
          // Arrange
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({ speciesId: 11, members: [] }),
              createSpeciesSnapshot({ speciesId: 12, members: [] }),
            ],
            randomValues: [0],
            popsize: 3,
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            3,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect({
            populationLength: nextPopulation.length,
            lastAllocation: evolutionController._lastOffspringAlloc,
          }).toEqual({
            populationLength: 0,
            lastAllocation: [
              { id: 11, alloc: 0 },
              { id: 12, alloc: 0 },
            ],
          });
        });
      });
    });

    describe('given one active species has zero adjusted fitness and another species is inactive', () => {
      describe('when species-aware offspring are allocated', () => {
        it('uses the fallback total-adjusted floor and leaves the inactive species at zero', async () => {
          // Arrange
          const activeSpeciesLeader = createPopulationMember({
            genomeId: 51,
            score: 0,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 13,
                members: [activeSpeciesLeader],
              }),
              createSpeciesSnapshot({ speciesId: 14, members: [] }),
            ],
            randomValues: [0, 0, 0],
            popsize: 2,
            crossSpeciesMatingProb: 0,
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            2,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect({
            populationLength: nextPopulation.length,
            lastAllocation: evolutionController._lastOffspringAlloc,
          }).toEqual({
            populationLength: 2,
            lastAllocation: [
              { id: 13, alloc: 2 },
              { id: 14, alloc: 0 },
            ],
          });
        });
      });
    });

    describe('given the minimum-offspring floor is affordable but oversubscribes the raw share result', () => {
      describe('when species-aware offspring are allocated', () => {
        it('trims the largest species allocation back to the remaining-slot budget', async () => {
          // Arrange
          const dominantSpeciesLeader = createPopulationMember({
            genomeId: 61,
            score: 10,
            depth: 1,
          });
          const trailingSpeciesLeader = createPopulationMember({
            genomeId: 62,
            score: 0,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 15,
                members: [dominantSpeciesLeader],
              }),
              createSpeciesSnapshot({
                speciesId: 16,
                members: [trailingSpeciesLeader],
              }),
            ],
            randomValues: [0, 0, 0, 0, 0, 0],
            popsize: 3,
            crossSpeciesMatingProb: 0,
            minOffspring: 1,
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            3,
            createPopulationOffspringConfig({ minOffspringDefault: 1 }),
          );

          // Assert
          expect(evolutionController._lastOffspringAlloc).toEqual([
            { id: 15, alloc: 2 },
            { id: 16, alloc: 1 },
          ]);
        });
      });
    });

    describe('given minimum-offspring defaults are used while several species stay at the floor', () => {
      describe('when oversubscription trimming makes multiple passes', () => {
        it('leaves floor-sized species unchanged while trimming only the larger allocation', async () => {
          // Arrange
          const dominantSpeciesLeader = createPopulationMember({
            genomeId: 63,
            score: 10,
            depth: 1,
          });
          const firstTrailingLeader = createPopulationMember({
            genomeId: 64,
            score: 0,
            depth: 1,
          });
          const secondTrailingLeader = createPopulationMember({
            genomeId: 65,
            score: 0,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 25,
                members: [dominantSpeciesLeader],
              }),
              createSpeciesSnapshot({
                speciesId: 26,
                members: [firstTrailingLeader],
              }),
              createSpeciesSnapshot({
                speciesId: 27,
                members: [secondTrailingLeader],
              }),
            ],
            randomValues: [0, 0, 0, 0, 0, 0],
            popsize: 3,
            crossSpeciesMatingProb: 0,
          });
          evolutionController.options.speciesAllocation = undefined;
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            3,
            createPopulationOffspringConfig({ minOffspringDefault: 1 }),
          );

          // Assert
          expect(evolutionController._lastOffspringAlloc).toEqual([
            { id: 25, alloc: 1 },
            { id: 26, alloc: 1 },
            { id: 27, alloc: 1 },
          ]);
        });
      });
    });

    describe('given the minimum-offspring floor is not affordable', () => {
      describe('when species-aware offspring are allocated', () => {
        it('keeps the raw floored allocation unchanged', async () => {
          // Arrange
          const leadingSpeciesMember = createPopulationMember({
            genomeId: 71,
            score: 8,
            depth: 1,
          });
          const trailingSpeciesMember = createPopulationMember({
            genomeId: 72,
            score: 0,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 17,
                members: [leadingSpeciesMember],
              }),
              createSpeciesSnapshot({
                speciesId: 18,
                members: [trailingSpeciesMember],
              }),
            ],
            randomValues: [0, 0, 0],
            popsize: 1,
            crossSpeciesMatingProb: 0,
            minOffspring: 1,
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig({ minOffspringDefault: 1 }),
          );

          // Assert
          expect(evolutionController._lastOffspringAlloc).toEqual([
            { id: 17, alloc: 1 },
            { id: 18, alloc: 0 },
          ]);
        });
      });
    });

    describe('given equally fit old and fresh species with an old-age penalty configured', () => {
      describe('when the latest offspring allocation snapshot is recorded', () => {
        it('assigns fewer offspring slots to the older species', async () => {
          // Arrange
          const olderSpeciesLeader = createPopulationMember({
            genomeId: 81,
            score: 10,
            depth: 1,
          });
          const fresherSpeciesLeader = createPopulationMember({
            genomeId: 82,
            score: 10,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 19,
                members: [olderSpeciesLeader],
                lastImproved: 0,
              }),
              createSpeciesSnapshot({
                speciesId: 20,
                members: [fresherSpeciesLeader],
                lastImproved: 9,
              }),
            ],
            randomValues: [0],
            generation: 10,
            crossSpeciesMatingProb: 0,
            speciesAgeBonus: {
              youngThreshold: 0,
              youngMultiplier: 1,
              oldThreshold: 2,
              oldMultiplier: 0.5,
            },
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            3,
            createPopulationOffspringConfig({
              youngThresholdDefault: 0,
              youngMultiplierDefault: 1,
              oldThresholdDefault: 2,
              oldMultiplierDefault: 0.5,
            }),
          );

          // Assert
          expect(evolutionController._lastOffspringAlloc).toEqual([
            { id: 19, alloc: 1 },
            { id: 20, alloc: 2 },
          ]);
        });
      });
    });

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
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect(
            (nextPopulation[0] as PopulationMetadataNetwork)._parents,
          ).toEqual([101, 202]);
        });
      });
    });

    describe('given species-aware reproduction with a single survivor in one species', () => {
      describe('when one offspring is bred from that survivor pool', () => {
        it('records duplicate parent ids and increments the inbreeding counter', async () => {
          // Arrange
          const loneSurvivor = createPopulationMember({
            genomeId: 91,
            score: 12,
            depth: 3,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 21,
                members: [loneSurvivor],
              }),
            ],
            randomValues: [0, 0],
            crossSpeciesMatingProb: 0,
            popsize: 1,
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect({
            parents: (nextPopulation[0] as PopulationMetadataNetwork)._parents,
            depth: (nextPopulation[0] as PopulationMetadataNetwork)._depth,
            inbreedingCount: evolutionController._lastInbreedingCount,
          }).toEqual({
            parents: [91, 91],
            depth: 4,
            inbreedingCount: 1,
          });
        });
      });
    });

    describe('given the species registry is cleared before second-parent selection', () => {
      describe('when cross-species mating is checked for that offspring', () => {
        it('uses the current-species survivor because no registry remains to cross against', async () => {
          // Arrange
          const loneSurvivor = createPopulationMember({
            genomeId: 93,
            score: 6,
            depth: 2,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 31,
                members: [loneSurvivor],
              }),
            ],
            randomValues: [0, 0],
            crossSpeciesMatingProb: 1,
            popsize: 1,
          });
          evolutionController._sortSpeciesMembers = () => {
            evolutionController._species = undefined;
          };
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect(
            (nextPopulation[0] as PopulationMetadataNetwork)._parents,
          ).toEqual([93, 93]);
        });
      });
    });

    describe('given lineage metadata is missing on the chosen parents', () => {
      describe('when one offspring is bred from that survivor pool', () => {
        it('falls back to zero-valued parent ids and depths', async () => {
          // Arrange
          const survivorWithoutLineage = createPopulationMember({
            genomeId: 92,
            score: 5,
            depth: 2,
          });
          survivorWithoutLineage._id = undefined;
          survivorWithoutLineage._depth = undefined;
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 28,
                members: [survivorWithoutLineage],
              }),
            ],
            randomValues: [0, 0],
            crossSpeciesMatingProb: 0,
            popsize: 1,
          });
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect({
            parents: (nextPopulation[0] as PopulationMetadataNetwork)._parents,
            depth: (nextPopulation[0] as PopulationMetadataNetwork)._depth,
          }).toEqual({
            parents: [0, 0],
            depth: 1,
          });
        });
      });
    });

    describe('given cross-species mating uses the default survival threshold fallback', () => {
      describe('when the second parent is sampled from another species', () => {
        it('uses the default survivor window to select that cross-species parent', async () => {
          // Arrange
          const firstSpeciesParent = createPopulationMember({
            genomeId: 1011,
            score: 20,
            depth: 1,
          });
          const strongerOtherParent = createPopulationMember({
            genomeId: 1012,
            score: 9,
            depth: 2,
          });
          const weakerOtherParent = createPopulationMember({
            genomeId: 1013,
            score: 1,
            depth: 1,
          });
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 22,
                members: [firstSpeciesParent],
              }),
              createSpeciesSnapshot({
                speciesId: 23,
                members: [strongerOtherParent, weakerOtherParent],
              }),
            ],
            randomValues: [0, 0, 0.75, 0],
            crossSpeciesMatingProb: 1,
            popsize: 1,
          });
          evolutionController.options.survivalThreshold = undefined;
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig({ survivalThresholdDefault: 0.5 }),
          );

          // Assert
          expect(
            (nextPopulation[0] as PopulationMetadataNetwork)._parents,
          ).toEqual([1011, 1012]);
        });
      });
    });

    describe('given cross-species mating selects a sparse missing species slot', () => {
      describe('when the other-species lookup resolves to no species object', () => {
        it('falls back to another survivor from the current species', async () => {
          // Arrange
          const currentSpeciesParent = createPopulationMember({
            genomeId: 1014,
            score: 20,
            depth: 1,
          });
          const currentSpecies = createSpeciesSnapshot({
            speciesId: 29,
            members: [currentSpeciesParent],
          });
          const evolutionController = createEvolutionController({
            species: [currentSpecies],
            randomValues: [0, 0, 0.75, 0],
            crossSpeciesMatingProb: 1,
            popsize: 1,
          });
          evolutionController._species = [currentSpecies];
          evolutionController._species.length = 2;
          const nextPopulation: Network[] = [];

          // Act
          await addSpeciatedOffspring(
            evolutionController,
            nextPopulation,
            1,
            createPopulationOffspringConfig(),
          );

          // Assert
          expect(
            (nextPopulation[0] as PopulationMetadataNetwork)._parents,
          ).toEqual([1014, 1014]);
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

        describe('given the species registry disappears after cross-species mating is chosen', () => {
          describe('when another species index is sampled under the retry guard', () => {
            it('uses the one-species fallback length and returns a current-species survivor', async () => {
              // Arrange
              const currentSpeciesParent = createPopulationMember({
                genomeId: 1015,
                score: 20,
                depth: 1,
              });
              const otherSpeciesParent = createPopulationMember({
                genomeId: 1016,
                score: 5,
                depth: 1,
              });
              const currentSpecies = createSpeciesSnapshot({
                speciesId: 32,
                members: [currentSpeciesParent],
              });
              const evolutionController = createEvolutionController({
                species: [
                  currentSpecies,
                  createSpeciesSnapshot({
                    speciesId: 33,
                    members: [otherSpeciesParent],
                  }),
                ],
                randomValues: [0],
                crossSpeciesMatingProb: 1,
                popsize: 1,
              });
              let randomCallCount = 0;
              evolutionController._getRNG = () => () => {
                randomCallCount += 1;
                if (randomCallCount === 2) {
                  evolutionController._species = undefined;
                }

                return 0;
              };
              const nextPopulation: Network[] = [];

              // Act
              await addSpeciatedOffspring(
                evolutionController,
                nextPopulation,
                1,
                createPopulationOffspringConfig(),
              );

              // Assert
              expect(
                (nextPopulation[0] as PopulationMetadataNetwork)._parents,
              ).toEqual([1015, 1015]);
            });
          });
        });

      });
    });

    describe('given speciated crossover receives an explicit controller rng', () => {
      describe('when parent-owned runtime rng disagrees with the controller rng', () => {
        it('still uses the controller rng for disabled-gene re-enable decisions', async () => {
          // Arrange
          const firstParent = createPopulationMember({
            genomeId: 501,
            score: 10,
            depth: 1,
          });
          const secondParent = firstParent.clone() as PopulationMetadataNetwork;
          secondParent._id = 502;
          secondParent.score = 5;
          secondParent._depth = 2;
          firstParent.connections[0].enabled = false;
          secondParent.connections[0].enabled = false;
          firstParent._reenableProb = 0.75;
          secondParent._reenableProb = 0.75;
          setPopulationRandomSource(firstParent, () => 0.99);
          setPopulationRandomSource(secondParent, () => 0.99);
          const evolutionController = createEvolutionController({
            species: [
              createSpeciesSnapshot({
                speciesId: 1,
                members: [firstParent, secondParent],
                lastImproved: 0,
              }),
            ],
            randomValues: [0, 0.9, 0.8, 0.8, 0.5],
            crossSpeciesMatingProb: 0,
            popsize: 1,
          });
          const nextPopulation: Network[] = [];
          const mathRandomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.99);

          try {
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
            expect(nextPopulation[0]?.connections[0]?.enabled).toBe(true);
          } finally {
            mathRandomSpy.mockRestore();
          }
        });
      });
    });
  });
});
