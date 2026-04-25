import Network from '../../architecture/network';
import Connection from '../../architecture/connection';
import Node from '../../architecture/node';
import Neat from '../../neat';
import { Architect, methods } from '../../neataptic';
import {
  createInnovationTracker,
  recordNodeSplitRecord,
} from '../innovation-tracker/innovation-tracker';
import type { InnovationTracker } from '../innovation-tracker/innovation-tracker.types';
import type {
  SpeciesHistoryEntry,
  SpeciesLastStats,
} from '../shared/neat.shared.types';
import {
  NeatExportPopulationValidationError,
  NeatExportStateControllerRestoreError,
  NeatExportStateBundleValidationError,
} from './neat.export.errors';
import { validateNativeGenome } from '../validate/neat.validate';
import {
  importStateImpl,
  type NeatMetaJSON,
  type NeatStateJSON,
} from './neat.export';
import type { NetworkJSON } from '../../architecture/network/network.types';

type ExportGenome = Network & {
  score?: number;
  _id?: number;
  _parents?: number[];
  _depth?: number;
  _reenableProb?: number;
  _compatInnovationMode?: 'require-explicit' | 'allow-fallback';
};

type ExportSpecies = {
  id: number;
  members: ExportGenome[];
  representative?: ExportGenome;
  bestScore?: number;
  lastImproved?: number;
  sharedFitness?: number;
  avgSharedFitness?: number;
  offspring?: number;
};

type ExportControllerState = {
  _nextGenomeId?: number;
  _lineageEnabled?: boolean;
  _lastGlobalImproveGeneration?: number;
  _speciesHistory?: SpeciesHistoryEntry[];
  _species?: ExportSpecies[];
  _nextSpeciesId?: number;
  _speciesCreated?: Map<number, number>;
  _prevSpeciesMembers?: Map<number, Set<number>>;
  _speciesLastStats?: Map<number, SpeciesLastStats>;
  _compatIntegral?: number;
  _compatSpeciesEMA?: number;
};

type ArchitectureCounterSnapshot = {
  nextConnectionInnovation: number;
  nextNodeGeneId: number;
  nextNodeIndex: number;
};

function readArchitectureCounterSnapshot(): ArchitectureCounterSnapshot {
  return {
    nextConnectionInnovation: (
      Connection as unknown as { _nextInnovation: number }
    )._nextInnovation,
    nextNodeGeneId: (Node as unknown as { _nextGeneId: number })._nextGeneId,
    nextNodeIndex: (Node as unknown as { _globalNodeIndex: number })
      ._globalNodeIndex,
  };
}

function restoreArchitectureCounterSnapshot(
  counterSnapshot: ArchitectureCounterSnapshot,
): void {
  (
    Connection as unknown as { _nextInnovation: number }
  )._nextInnovation = counterSnapshot.nextConnectionInnovation;
  (Node as unknown as { _nextGeneId: number })._nextGeneId =
    counterSnapshot.nextNodeGeneId;
  (Node as unknown as { _globalNodeIndex: number })._globalNodeIndex =
    counterSnapshot.nextNodeIndex;
}

function createTemporalModuleExtensions(
  networkJson: NetworkJSON,
): NonNullable<NetworkJSON['extensions']> {
  const hiddenNodeGeneIds = networkJson.nodes
    .filter((node) => node.type !== 'input' && node.type !== 'output')
    .map((node) => node.geneId)
    .filter((geneId): geneId is number => typeof geneId === 'number');
  const gatedConnections = networkJson.connections.filter(
    (
      connection,
    ): connection is NetworkJSON['connections'][number] & {
      innovation: number;
      gaterGeneId: number;
    } =>
      typeof connection.innovation === 'number' &&
      typeof connection.gaterGeneId === 'number',
  );

  if (hiddenNodeGeneIds.length === 0 || gatedConnections.length === 0) {
    throw new Error('Expected recurrent module fixtures with hidden nodes and gated connections.');
  }

  return {
    version: 1,
    values: {
      recurrentModules: [
        {
          moduleId: 'module:lstm:0',
          kind: 'lstm',
          nodeGeneIdsByRole: {
            recurrentCore: hiddenNodeGeneIds,
          },
          connectionInnovations: gatedConnections.map(
            (connection) => connection.innovation,
          ),
        },
      ],
      gatedBlocks: [
        {
          blockId: 'gated:block:0',
          gaterGeneIds: [...new Set(gatedConnections.map((connection) => connection.gaterGeneId))],
          connectionInnovations: gatedConnections.map(
            (connection) => connection.innovation,
          ),
        },
      ],
    },
  };
}

function readFirstTemporalConnectionInnovation(
  extensions: NonNullable<NetworkJSON['extensions']>,
): number {
  const extensionValues = extensions.values as {
    gatedBlocks?: Array<{ connectionInnovations: number[] }>;
  };
  const connectionInnovation =
    extensionValues.gatedBlocks?.[0]?.connectionInnovations?.[0];

  if (typeof connectionInnovation !== 'number') {
    throw new Error('Expected one temporal gated-block connection innovation.');
  }

  return connectionInnovation;
}

describe('neat export chapter', () => {
  describe('meta-only restore', () => {
    describe('given a run that has already advanced generation state', () => {
      const scoreByNodeCount = (network: Network) => network.nodes.length;

      let exportedMeta: NeatMetaJSON;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, {
          popsize: 5,
          seed: 802,
        });

        await neat.evaluate();
        await neat.evolve();

        // Act
        exportedMeta = neat.toJSON();
      });

      describe('when the controller is rebuilt from meta only', () => {
        let restoredGeneration: number;
        let restoredNextInnovationId: number;

        beforeAll(() => {
          // Arrange
          const restored = Neat.fromJSON(exportedMeta, scoreByNodeCount);

          // Act
          restoredGeneration = restored.generation;
          restoredNextInnovationId =
            restored.toJSON().innovationTracker.nextInnovationId;
        });

        it('restores the exported generation counter', () => {
          // Assert
          expect(restoredGeneration).toBe(exportedMeta.generation);
        });

        it('restores the exported innovation-tracker cursor', () => {
          // Assert
          expect(restoredNextInnovationId).toBe(
            exportedMeta.innovationTracker.nextInnovationId,
          );
        });

        it('marks exported meta with the versioned checkpoint contract', () => {
          // Assert
          expect(exportedMeta.formatVersion).toBe(1);
        });

        it('restores exported node-split innovation records', () => {
          // Arrange
          const neat = new Neat(2, 1, scoreByNodeCount, {
            popsize: 1,
            seed: 803,
          });
          const mutationState = neat as unknown as {
            _innovationTracker: InnovationTracker;
          };
          mutationState._innovationTracker = createInnovationTracker();
          recordNodeSplitRecord(
            mutationState._innovationTracker,
            'splitConnectionInnovation:11',
            {
            newNodeGeneId: 7,
            inInnov: 11,
            outInnov: 12,
            },
          );

          // Act
          const restored = Neat.fromJSON(neat.toJSON(), scoreByNodeCount);

          // Assert
          expect(restored.toJSON().innovationTracker.nodeSplitRecords).toEqual(
            neat.toJSON().innovationTracker.nodeSplitRecords,
          );
        });

        it('keeps the generation-zero innovation cursor above the starter graph', () => {
          // Arrange
          const neat = new Neat(2, 1, scoreByNodeCount, {
            popsize: 3,
            seed: 804,
          });
          const maxObservedInnovation = neat.population
            .flatMap((genome) =>
              genome.connections.map((connection) => connection.innovation),
            )
            .reduce(
              (currentMaxInnovation, innovation) =>
                Math.max(currentMaxInnovation, innovation),
              -1,
            );

          // Act
          const exportedMeta = neat.toJSON();

          // Assert
          expect(exportedMeta.innovationTracker.nextInnovationId).toBe(
            maxObservedInnovation + 1,
          );
        });

        it('restores the exported controller rng replay state', () => {
          // Arrange
          const neat = new Neat(2, 1, scoreByNodeCount, {
            popsize: 2,
            seed: 805,
          });
          neat.sampleRandom(7);
          const exportedMeta = neat.toJSON();
          const expectedFutureSequence = neat.sampleRandom(4);

          // Act
          const restored = Neat.fromJSON(exportedMeta, scoreByNodeCount);
          const restoredCheckpointRngState = restored.exportRNGState();
          const restoredFutureSequence = restored.sampleRandom(4);

          // Assert
          expect({
            rngState: restoredCheckpointRngState,
            futureSequence: restoredFutureSequence,
            architectureCounters: readArchitectureCounterSnapshot(),
          }).toEqual({
            rngState: exportedMeta.runtime?.rngState,
            futureSequence: expectedFutureSequence,
            architectureCounters: {
              nextConnectionInnovation:
                exportedMeta.runtime?.nextConnectionInnovation,
              nextNodeGeneId: exportedMeta.runtime?.nextNodeGeneId,
              nextNodeIndex: exportedMeta.runtime?.nextNodeIndex,
            },
          });
        });
      });
    });

    describe('given invalid serialized meta state', () => {
      const scoreByNodeCount = (network: Network) => network.nodes.length;

      it('rejects unsupported future meta format versions', () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 1, seed: 806 });
        const invalidMeta = {
          ...neat.toJSON(),
          formatVersion: 99,
        } as NeatMetaJSON;

        // Assert
        expect(() => Neat.fromJSON(invalidMeta, scoreByNodeCount)).toThrow(
          NeatExportStateControllerRestoreError,
        );
      });

      it('rejects meta payloads that omit innovation tracker state', () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 1, seed: 807 });
        const invalidMeta = {
          ...neat.toJSON(),
          innovationTracker: undefined,
        } as unknown as NeatMetaJSON;

        // Assert
        expect(() => Neat.fromJSON(invalidMeta, scoreByNodeCount)).toThrow(
          NeatExportStateControllerRestoreError,
        );
      });

      it('rebuilds meta-only checkpoints when serialized options are missing', () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 1, seed: 808 });
        const exportedMeta = neat.toJSON();
        const optionsFreeMeta = {
          ...exportedMeta,
          options: undefined,
        } as unknown as NeatMetaJSON;

        // Act
        const restored = Neat.fromJSON(optionsFreeMeta, scoreByNodeCount);

        // Assert
        expect({
          generation: restored.generation,
          nextInnovationId: restored.toJSON().innovationTracker.nextInnovationId,
        }).toEqual({
          generation: exportedMeta.generation,
          nextInnovationId: exportedMeta.innovationTracker.nextInnovationId,
        });
      });

      it('ignores non-object runtime payloads during meta restore', () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 1, seed: 809 });
        const exportedMeta = neat.toJSON();
        const metaWithPrimitiveRuntime = {
          ...exportedMeta,
          runtime: 7,
        } as unknown as NeatMetaJSON;

        // Assert
        expect(() =>
          Neat.fromJSON(metaWithPrimitiveRuntime, scoreByNodeCount)
        ).not.toThrow();
      });
    });
  });

  describe('population-only snapshots', () => {
    const scoreByNodeCount = (network: Network) => network.nodes.length;

    describe('given the controller currently has no genomes', () => {
      it('exports an empty population array', () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 0 });
        neat.population = [];

        // Act
        const exportedPopulation = neat.export();

        // Assert
        expect(exportedPopulation).toEqual([]);
      });
    });

    describe('given a saved population is imported into another controller', () => {
      it('replaces the destination population with the exported genome count', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 2,
          seed: 21,
        });
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 22,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);

        // Assert
        expect(destinationController.population.length).toBe(
          exportedPopulation.length,
        );
      });

      it('preserves opt-in connection gain extensions across export and import', async () => {
        // Arrange
        const sourceController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 211,
          genomeExtensions: {
            connectionGain: true,
          },
        });
        sourceController.population[0].connections[0].gain = 1.5;
        const destinationController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 212,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);

        // Assert
        expect(destinationController.population[0].connections[0].gain).toBe(
          1.5,
        );
      });

      it('preserves opt-in response and re-enable extensions even without controller-meta fallback', async () => {
        // Arrange
        const sourceController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 213,
          genomeExtensions: {
            nodeResponse: true,
            disabledConnectionReenableProbability: true,
          },
        });
        const sourceGenome = sourceController.population[0] as ExportGenome;
        const outputNode = sourceGenome.nodes.at(-1);
        if (!outputNode) {
          throw new Error('Expected an output node for extension export.');
        }

        outputNode.response = 1.5;
        sourceGenome._reenableProb = 0.6;
        const destinationController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 214,
        });
        const exportedPopulation = sourceController.export() as Array<{
          controllerMeta?: { reenableProb?: number };
        }>;
        if (exportedPopulation[0].controllerMeta) {
          delete exportedPopulation[0].controllerMeta.reenableProb;
        }

        // Act
        await destinationController.import(exportedPopulation);
        const importedGenome = destinationController.population[0] as ExportGenome;

        // Assert
        expect({
          response: importedGenome.nodes.at(-1)?.response ?? null,
          reenableProb: importedGenome._reenableProb,
        }).toEqual({
          response: 1.5,
          reenableProb: 0.6,
        });
      });

      it('preserves canonical activation mutation without using genome extensions', async () => {
        // Arrange
        const sourceController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 215,
        });
        const sourceGenome = sourceController.population[0] as ExportGenome;
        const outputNode = sourceGenome.nodes.at(-1);
        if (!outputNode) {
          throw new Error('Expected an output node for activation export.');
        }

        outputNode.squash = methods.Activation.tanh;
        const destinationController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 216,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);
        const importedGenome = destinationController.population[0] as ExportGenome;

        // Assert
        expect({
          importedSquash: importedGenome.nodes.at(-1)?.squash ?? null,
          usesExtensions: 'extensions' in (exportedPopulation[0] as Record<string, unknown>),
        }).toEqual({
          importedSquash: methods.Activation.tanh,
          usesExtensions: false,
        });
      });

      it('preserves explicit temporal module descriptors across export and import', async () => {
        // Arrange
        const sourceController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 217,
        });
        const taggedPayload = Architect.lstm(1, 2, 1)
          .toJSON() as unknown as NetworkJSON;
        taggedPayload.extensions = createTemporalModuleExtensions(taggedPayload);
        sourceController.population = [
          Network.fromJSON(taggedPayload as unknown as Record<string, unknown>),
        ] as unknown as ExportGenome[];
        const destinationController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 218,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);

        // Assert
        expect(
          (
            destinationController.population[0].toJSON() as unknown as NetworkJSON
          ).extensions,
        ).toEqual(taggedPayload.extensions);
      });

      it('preserves disabled temporal module genes across export and import', async () => {
        // Arrange
        const sourceController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 219,
        });
        const taggedPayload = Architect.lstm(1, 2, 1)
          .toJSON() as unknown as NetworkJSON;
        taggedPayload.extensions = createTemporalModuleExtensions(taggedPayload);
        const moduleConnectionInnovation = readFirstTemporalConnectionInnovation(
          taggedPayload.extensions,
        );
        const taggedConnection = taggedPayload.connections.find(
          (connection) => connection.innovation === moduleConnectionInnovation,
        );

        if (!taggedConnection) {
          throw new Error('Expected one module-owned connection for dormant-state export coverage.');
        }

        taggedConnection.enabled = false;
        sourceController.population = [
          Network.fromJSON(taggedPayload as unknown as Record<string, unknown>),
        ] as unknown as ExportGenome[];
        const destinationController = new Neat(1, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 220,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);
        const importedPayload = destinationController.population[0]
          .toJSON() as unknown as NetworkJSON;
        const importedConnection = importedPayload.connections.find(
          (connection) => connection.innovation === moduleConnectionInnovation,
        );

        // Assert
        expect({
          extensions: importedPayload.extensions,
          enabled: importedConnection?.enabled ?? null,
        }).toEqual({
          extensions: taggedPayload.extensions,
          enabled: false,
        });
      });

      it('preserves exported connection innovations after population import', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 23,
        });
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 24,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);
        const importedPopulationSnapshot = destinationController.export() as
          Array<{ connections: Array<{ innovation?: number }> }>;
        const sourcePopulationSnapshot = sourceController.export() as Array<{
          connections: Array<{ innovation?: number }>;
        }>;

        // Assert
        expect(
          importedPopulationSnapshot[0].connections.map(
            (connection: { innovation?: number }) => connection.innovation,
          ),
        ).toEqual(
          sourcePopulationSnapshot[0].connections.map(
            (connection: { innovation?: number }) => connection.innovation,
          ),
        );
      });

      it('exports controller-owned genome metadata next to the network payload', () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 27,
        });
        const genome = sourceController.population[0] as ExportGenome & {
          getRNGState?: () => number | undefined;
        };
        genome.score = 9;
        genome._id = 91;
        genome._parents = [7, 8];
        genome._depth = 3;
        genome._reenableProb = 0.25;
        genome._compatInnovationMode = 'allow-fallback';
        const networkRngState = genome.getRNGState?.();

        // Act
        const exportedPopulation = sourceController.export() as Array<{
          controllerMeta?: {
            score?: number;
            genomeId?: number;
            networkRngState?: number;
            parents?: number[];
            depth?: number;
            reenableProb?: number;
            compatInnovationMode?: 'require-explicit' | 'allow-fallback';
          };
        }>;

        // Assert
        expect(exportedPopulation[0].controllerMeta).toEqual({
          score: 9,
          genomeId: 91,
          networkRngState,
          parents: [7, 8],
          depth: 3,
          reenableProb: 0.25,
          compatInnovationMode: 'allow-fallback',
        });
      });

      it('restores genome metadata and advances the next genome id floor', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 28,
        });
        const sourceGenome = sourceController.population[0] as ExportGenome;
        sourceGenome.score = 11;
        sourceGenome._id = 91;
        sourceGenome._parents = [4, 5];
        sourceGenome._depth = 6;
        sourceGenome._reenableProb = 0.4;
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 29,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);
        const destinationState =
          destinationController as unknown as ExportControllerState;
        const importedGenome = destinationController.population[0] as ExportGenome;

        // Assert
        expect({
          score: importedGenome.score,
          genomeId: importedGenome._id,
          parents: importedGenome._parents,
          depth: importedGenome._depth,
          reenableProb: importedGenome._reenableProb,
          nextGenomeId: destinationState._nextGenomeId,
        }).toEqual({
          score: 11,
          genomeId: 91,
          parents: [4, 5],
          depth: 6,
          reenableProb: 0.4,
          nextGenomeId: 92,
        });
      });

      it('seeds the next genome id floor from one when the destination cursor is missing', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 223,
        });
        const sourceGenome = sourceController.population[0] as ExportGenome;
        sourceGenome._id = 41;
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 224,
        });
        const destinationState =
          destinationController as unknown as ExportControllerState;
        destinationState._nextGenomeId = undefined;
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);

        // Assert
        expect(destinationState._nextGenomeId).toBe(42);
      });

      it('imports snapshots even when dropout metadata is not numeric', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 225,
        });
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 226,
        });
        const exportedPopulation = sourceController.export() as unknown as Array<
          NetworkJSON & { controllerMeta?: Record<string, unknown> }
        >;
        exportedPopulation[0].dropout = 'invalid' as unknown as number;

        // Act
        await destinationController.import(
          exportedPopulation as unknown as NeatStateJSON['population'],
        );

        // Assert
        expect({
          populationSize: destinationController.population.length,
          isValid: validateNativeGenome(destinationController.population[0]).isValid,
        }).toEqual({ populationSize: 1, isValid: true });
      });

      it('rejects imported genomes that no longer carry explicit connection innovations', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 30,
        });
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 31,
        });
        const exportedPopulation = sourceController.export() as Array<{
          connections: Array<{ innovation?: number }>;
        }>;
        delete exportedPopulation[0].connections[0].innovation;

        // Act
        const invalidImport = destinationController.import(
          exportedPopulation as unknown as NeatStateJSON['population'],
        );

        // Assert
        await expect(invalidImport).rejects.toThrow(
          NeatExportPopulationValidationError,
        );
      });

      it('keeps imported genomes valid for native NEAT flows', async () => {
        // Arrange
        const sourceController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 2,
          seed: 25,
        });
        const destinationController = new Neat(2, 1, scoreByNodeCount, {
          popsize: 1,
          seed: 26,
        });
        const exportedPopulation = sourceController.export();

        // Act
        await destinationController.import(exportedPopulation);
        const allGenomesValidate = destinationController.population.every(
          (genome: Network) => validateNativeGenome(genome).isValid,
        );

        // Assert
        expect(allGenomesValidate).toBe(true);
      });
    });

    describe('given an empty array is imported', () => {
      it('sets both runtime population and configured popsize to zero', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2 });

        // Act
        await neat.import([]);

        // Assert
        expect({
          populationSize: neat.population.length,
          popsize: neat.options.popsize,
        }).toEqual({ populationSize: 0, popsize: 0 });
      });
    });

    describe('given the import payload is not an array', () => {
      it('rejects the population snapshot', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2, seed: 221 });
        const invalidImport = neat.import(
          undefined as unknown as NeatStateJSON['population'],
        );

        // Assert
        await expect(invalidImport).rejects.toThrow(
          'Population snapshots must be arrays of serialized genomes.',
        );
      });
    });

    describe('given the import payload contains a non-object entry', () => {
      it('rejects the malformed snapshot entry', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2, seed: 222 });
        const invalidImport = neat.import([
          undefined as unknown as NetworkJSON,
        ] as unknown as NeatStateJSON['population']);

        // Assert
        await expect(invalidImport).rejects.toThrow(
          'Population snapshot entry 0 must be a serialized genome object.',
        );
      });
    });
  });

  describe('full-state restore', () => {
    describe('given an invalid state bundle', () => {
      const scoreByNodeCount = (network: Network) => network.nodes.length;

      it('throws the export bundle validation error', async () => {
        // Arrange
        const importInvalidState = async () =>
          Neat.importState(
            undefined as unknown as NeatStateJSON,
            scoreByNodeCount,
          );

        // Act
        const invalidImport = importInvalidState();

        // Assert
        await expect(invalidImport).rejects.toThrow(
          NeatExportStateBundleValidationError,
        );
      });

      it('throws when a full checkpoint omits the population array', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2, seed: 32 });
        const incompleteState = {
          ...neat.exportState(),
          population: undefined,
        } as unknown as NeatStateJSON;

        // Act
        const invalidImport = Neat.importState(incompleteState, scoreByNodeCount);

        // Assert
        await expect(invalidImport).rejects.toThrow(
          NeatExportStateBundleValidationError,
        );
      });

      it('throws when a versioned full checkpoint omits the full checkpoint marker', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2, seed: 35 });
        const invalidState = {
          ...neat.exportState(),
          checkpointMode: 'meta-only',
        } as unknown as NeatStateJSON;

        // Act
        const invalidImport = Neat.importState(invalidState, scoreByNodeCount);

        // Assert
        await expect(invalidImport).rejects.toThrow(
          NeatExportStateBundleValidationError,
        );
      });

      it('throws when a full checkpoint advertises an unsupported state format version', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2, seed: 38 });
        const invalidState = {
          ...neat.exportState(),
          formatVersion: 99,
        } as unknown as NeatStateJSON;

        // Act
        const invalidImport = Neat.importState(invalidState, scoreByNodeCount);

        // Assert
        await expect(invalidImport).rejects.toThrow(
          'Unsupported NEAT checkpoint format version: 99.',
        );
      });

      it('throws when a full checkpoint omits serialized neat meta state', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2, seed: 39 });
        const invalidState = {
          ...neat.exportState(),
          neat: undefined,
        } as unknown as NeatStateJSON;

        // Act
        const invalidImport = Neat.importState(invalidState, scoreByNodeCount);

        // Assert
        await expect(invalidImport).rejects.toThrow(
          'Full checkpoint bundles must include serialized NEAT meta state.',
        );
      });

      it('restores legacy full checkpoints even when speciation resume state is absent', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, { popsize: 2, seed: 40 });
        const legacyState = {
          ...neat.exportState(),
          formatVersion: 0,
          speciation: undefined,
        } as unknown as NeatStateJSON;

        // Act
        const restored = await Neat.importState(legacyState, scoreByNodeCount);

        // Assert
        expect({
          generation: restored.generation,
          populationSize: restored.population.length,
        }).toEqual({ generation: legacyState.neat.generation, populationSize: 2 });
      });

      it('throws when a versioned full checkpoint omits speciation resume state', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, {
          popsize: 2,
          seed: 36,
          speciation: true,
        });
        const invalidState = {
          ...neat.exportState(),
          speciation: undefined,
        } as unknown as NeatStateJSON;

        // Act
        const invalidImport = Neat.importState(invalidState, scoreByNodeCount);

        // Assert
        await expect(invalidImport).rejects.toThrow(
          NeatExportStateBundleValidationError,
        );
      });

      it('throws when the controller cannot be rebuilt from serialized meta state', async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNodeCount, {
          popsize: 2,
          seed: 37,
          speciation: true,
        });
        const stateBundle = neat.exportState();
        const invalidImport = importStateImpl.call(
          {
            fromJSON: () => undefined,
          } as unknown as ThisParameterType<typeof importStateImpl>,
          stateBundle,
          (network) => scoreByNodeCount(network as Network),
        );

        // Assert
        await expect(invalidImport).rejects.toThrow(
          NeatExportStateControllerRestoreError,
        );
      });
    });

    describe('given a saved checkpoint from an evolved run', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let exportedState: NeatStateJSON;
      let restoredGeneration: number;
      let restoredPopulationSize: number;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(3, 1, scoreByConnectionCount, {
          popsize: 6,
          seed: 11,
        });

        await neat.evolve();
        await neat.evolve();
        exportedState = neat.exportState();

        // Act
        const restored = await Neat.importState(
          exportedState,
          scoreByConnectionCount,
        );

        restoredGeneration = restored.generation;
        restoredPopulationSize = restored.population.length;
      });

      describe('when the saved generation is compared after restore', () => {
        it('keeps the exported generation value', () => {
          // Assert
          expect(restoredGeneration).toBe(exportedState.neat.generation);
        });
      });

      describe('when the restored pool is compared to the saved checkpoint', () => {
        it('keeps the exported population size', () => {
          // Assert
          expect(restoredPopulationSize).toBe(exportedState.population.length);
        });
      });

      describe('when restored genomes are checked before later NEAT work', () => {
        it('keeps the restored pool validator-clean', async () => {
          // Arrange
          const neat = new Neat(3, 1, scoreByConnectionCount, {
            popsize: 4,
            seed: 12,
          });
          await neat.evolve();
          const restored = await Neat.importState(
            neat.exportState(),
            scoreByConnectionCount,
          );

          // Act
          const allGenomesValidate = restored.population.every((genome) =>
            validateNativeGenome(genome).isValid,
          );

          // Assert
          expect(allGenomesValidate).toBe(true);
        });
      });
    });

    describe('given a versioned checkpoint with species and controller runtime state', () => {
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;

      let exportedState: NeatStateJSON;
      let restoredNeat: Neat;
      let restoredState: ExportControllerState;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByConnectionCount, {
          popsize: 2,
          seed: 33,
          speciation: true,
        });
        const controller = neat as unknown as ExportControllerState;
        const firstGenome = neat.population[0] as ExportGenome;
        const secondGenome = neat.population[1] as ExportGenome;

        firstGenome._id = 101;
        firstGenome.score = 5;
        firstGenome._parents = [1];
        firstGenome._depth = 2;
        secondGenome._id = 102;
        secondGenome.score = 3;
        secondGenome._parents = [2];
        secondGenome._depth = 4;

        controller._nextGenomeId = 205;
        controller._lineageEnabled = true;
        controller._lastGlobalImproveGeneration = 6;
        controller._speciesHistory = [
          {
            generation: 6,
            stats: [
              { id: 7, size: 1, bestScore: 5, lastImproved: 0 },
              { id: 8, size: 1, bestScore: 3, lastImproved: 1 },
            ],
          },
        ];
        controller._species = [
          {
            id: 7,
            members: [firstGenome],
            representative: firstGenome,
            bestScore: 5,
            lastImproved: 6,
            sharedFitness: 5,
            avgSharedFitness: 5,
            offspring: 1,
          },
          {
            id: 8,
            members: [secondGenome],
            representative: secondGenome,
            bestScore: 3,
            lastImproved: 5,
            sharedFitness: 3,
            avgSharedFitness: 3,
            offspring: 1,
          },
        ];
        controller._nextSpeciesId = 9;
        controller._speciesCreated = new Map([
          [7, 4],
          [8, 5],
        ]);
        controller._prevSpeciesMembers = new Map([
          [7, new Set([101])],
          [8, new Set([102])],
        ]);
        controller._speciesLastStats = new Map([
          [7, { meanNodes: firstGenome.nodes.length, meanConns: firstGenome.connections.length, best: 5 }],
          [8, { meanNodes: secondGenome.nodes.length, meanConns: secondGenome.connections.length, best: 3 }],
        ]);
        controller._compatIntegral = 0.75;
        controller._compatSpeciesEMA = 2;
        exportedState = neat.exportState();

        // Act
        restoredNeat = await Neat.importState(
          exportedState,
          scoreByConnectionCount,
        );
        restoredState = restoredNeat as unknown as ExportControllerState;
      });

      it('marks exported full-state snapshots as versioned full checkpoints', () => {
        // Assert
        expect({
          formatVersion: exportedState.formatVersion,
          checkpointMode: exportedState.checkpointMode,
          hasSpeciation: typeof exportedState.speciation === 'object',
        }).toEqual({
          formatVersion: 1,
          checkpointMode: 'full',
          hasSpeciation: true,
        });
      });

      it('restores controller runtime counters and species history', () => {
        // Assert
        expect({
          nextGenomeId: restoredState._nextGenomeId,
          architectureCounters: readArchitectureCounterSnapshot(),
          lineageEnabled: restoredState._lineageEnabled,
          lastGlobalImproveGeneration: restoredState._lastGlobalImproveGeneration,
          speciesHistory: restoredNeat.getSpeciesHistory(),
        }).toEqual({
          nextGenomeId: 205,
          architectureCounters: {
            nextConnectionInnovation:
              exportedState.neat.runtime?.nextConnectionInnovation,
            nextNodeGeneId: exportedState.neat.runtime?.nextNodeGeneId,
            nextNodeIndex: exportedState.neat.runtime?.nextNodeIndex,
          },
          lineageEnabled: true,
          lastGlobalImproveGeneration: 6,
          speciesHistory: [
            {
              generation: 6,
              stats: [
                { id: 7, size: 1, bestScore: 5, lastImproved: 0 },
                { id: 8, size: 1, bestScore: 3, lastImproved: 1 },
              ],
            },
          ],
        });
      });

      it('restores live species membership and speciation bookkeeping by genome id', () => {
        // Arrange
        const liveSpecies: ExportSpecies[] = restoredState._species ?? [];

        // Assert
        expect({
          species: liveSpecies.map((species: ExportSpecies) => ({
            id: species.id,
            members: species.members.map((member: ExportGenome) => member._id),
            representative: species.representative?._id,
            bestScore: species.bestScore,
            lastImproved: species.lastImproved,
          })),
          nextSpeciesId: restoredState._nextSpeciesId,
          speciesCreated: Array.from(restoredState._speciesCreated ?? []),
          prevSpeciesMembers: Array.from(
            restoredState._prevSpeciesMembers ?? [],
            ([speciesId, memberIds]) => [speciesId, Array.from(memberIds)],
          ),
          compatIntegral: restoredState._compatIntegral,
          compatSpeciesEMA: restoredState._compatSpeciesEMA,
        }).toEqual({
          species: [
            {
              id: 7,
              members: [101],
              representative: 101,
              bestScore: 5,
              lastImproved: 6,
            },
            {
              id: 8,
              members: [102],
              representative: 102,
              bestScore: 3,
              lastImproved: 5,
            },
          ],
          nextSpeciesId: 9,
          speciesCreated: [
            [7, 4],
            [8, 5],
          ],
          prevSpeciesMembers: [
            [7, [101]],
            [8, [102]],
          ],
          compatIntegral: 0.75,
          compatSpeciesEMA: 2,
        });
      });
    });

    describe('given the same full checkpoint resumes with the same code and seed', () => {
      it('continues with the same future innovation assignments and species outcomes', async () => {
        // Arrange
        const originalCounters = readArchitectureCounterSnapshot();
        const scoreByStructure = (network: Network) =>
          network.connections.length * 100 + network.nodes.length;
        const sourceNeat = new Neat(2, 1, scoreByStructure, {
          popsize: 6,
          seed: 34,
          speciation: true,
          mutation: [
            methods.mutation.ADD_NODE,
            methods.mutation.ADD_CONN,
          ],
          mutationRate: 1,
          mutationAmount: 1,
        });
        await sourceNeat.evolve();
        await sourceNeat.evolve();
        const checkpoint = structuredClone(sourceNeat.exportState());

        let firstReplayFutureState: NeatStateJSON;
        let secondReplayFutureState: NeatStateJSON;

        try {
          // Step 1: Restore the same checkpoint twice.
          const firstReplay = await Neat.importState(
            structuredClone(checkpoint),
            scoreByStructure,
          );
          const replayCounters = readArchitectureCounterSnapshot();
          const secondReplay = await Neat.importState(
            structuredClone(checkpoint),
            scoreByStructure,
          );

          expect(secondReplay.exportState()).toEqual(firstReplay.exportState());

          // Step 2: Advance both replays from the same restored
          // architecture-counter baseline.
          restoreArchitectureCounterSnapshot(replayCounters);
          await firstReplay.evolve();
          firstReplayFutureState = firstReplay.exportState();

          restoreArchitectureCounterSnapshot(replayCounters);
          await secondReplay.evolve();
          secondReplayFutureState = secondReplay.exportState();
        } finally {
          restoreArchitectureCounterSnapshot(originalCounters);
        }

        // Assert
        expect(secondReplayFutureState).toEqual(firstReplayFutureState);
      });
    });
  });
});
