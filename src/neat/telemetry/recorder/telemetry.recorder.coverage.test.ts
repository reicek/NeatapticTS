import {
  buildTelemetryEntry,
  computeDiversityStats,
  createTelemetryEntryBase,
  recordTelemetryEntry,
  structuralEntropy,
} from './telemetry.recorder';

type TelemetryGenomeStub = {
  _depth?: number;
  connections: Array<{
    enabled: boolean;
    from: { geneId: number };
    to: { geneId: number };
  }>;
  nodes: Array<{ geneId: number }>;
};

function createPairRngFactory(
  firstValue: number,
  secondValue: number,
): () => () => number {
  return () => {
    const randomValues = [firstValue, secondValue];
    let randomValueIndex = 0;

    return () => {
      const nextValue =
        randomValues[randomValueIndex] ?? randomValues.at(-1) ?? 0;
      randomValueIndex += 1;
      return nextValue;
    };
  };
}

function createCyclingRandom(values: number[]): () => number {
  let randomValueIndex = 0;

  return () => {
    const nextValue = values[randomValueIndex % values.length] ?? 0;
    randomValueIndex += 1;
    return nextValue;
  };
}

function createGenome(input: {
  connectionCount: number;
  depth?: number;
  nodeCount: number;
}): TelemetryGenomeStub {
  const nodes = Array.from({ length: input.nodeCount }, (_, nodeIndex) => ({
    geneId: nodeIndex + 1,
  }));

  return {
    _depth: input.depth,
    connections: Array.from({ length: input.connectionCount }, (_, connectionIndex) => ({
      enabled: true,
      from: nodes[0],
      to: nodes[Math.min(connectionIndex + 1, nodes.length - 1)],
    })),
    nodes,
  };
}

describe('neat telemetry recorder chapter', () => {
  describe('createTelemetryEntryBase', () => {
    describe('given a generation summary', () => {
      describe('when building the strict base telemetry entry', () => {
        it('returns the required baseline fields and defaults', () => {
          // Act
          const telemetryEntry = createTelemetryEntryBase(7, 1.25, 3);

          // Assert
          expect(telemetryEntry).toEqual({
            best: 1.25,
            gen: 7,
            hyper: 0,
            objImportance: {},
            ops: [],
            species: 3,
          });
        });
      });
    });
  });

  describe('computeDiversityStats', () => {
    afterEach(() => {
      jest.restoreAllMocks();
    });

    describe('given diversity telemetry is disabled', () => {
      describe('when computing diversity stats', () => {
        it('leaves the cached diversity block untouched', () => {
          // Arrange
          const telemetryContext = {
            _diversityStats: { stale: true },
            options: {},
          };

          // Act
          computeDiversityStats.call(telemetryContext);

          // Assert
          expect(telemetryContext._diversityStats).toEqual({ stale: true });
        });
      });
    });

    describe('given explicit diversity helpers are configured', () => {
      describe('when computing diversity stats', () => {
        it('stores the derived compatibility, entropy, graphlet, and lineage summary', () => {
          // Arrange
          const population = [
            createGenome({ connectionCount: 1, depth: 0, nodeCount: 2 }),
            createGenome({ connectionCount: 3, depth: 2, nodeCount: 2 }),
          ];
          const telemetryContext: {
            _compatibilityDistance: (
              firstGenome: TelemetryGenomeStub,
              secondGenome: TelemetryGenomeStub,
            ) => number;
            _diversityStats?: unknown;
            _getRNG: () => () => number;
            _lineageEnabled: boolean;
            _structuralEntropy: (genome: TelemetryGenomeStub) => number;
            options: {
              diversityMetrics: {
                enabled: boolean;
                graphletSample: number;
                pairSample: number;
              };
            };
            population: TelemetryGenomeStub[];
          } = {
            _compatibilityDistance: (
              firstGenome: TelemetryGenomeStub,
              secondGenome: TelemetryGenomeStub,
            ) =>
              Math.abs(
                firstGenome.connections.length - secondGenome.connections.length,
              ),
            _getRNG: createPairRngFactory(0, 0.99),
            _lineageEnabled: true,
            _structuralEntropy: (genome: TelemetryGenomeStub) =>
              genome.connections.length,
            options: {
              diversityMetrics: {
                enabled: true,
                graphletSample: 2,
                pairSample: 3,
              },
            },
            population,
          };

          // Act
          computeDiversityStats.call(telemetryContext);

          // Assert
          expect(telemetryContext._diversityStats).toEqual({
            graphletEntropy: 0,
            lineageMeanDepth: 1,
            lineageMeanPairDist: 2,
            meanCompat: 2,
            meanEntropy: 2,
            varCompat: 0,
            varEntropy: 1,
          });
        });
      });
    });

    describe('given fallback diversity helpers must be used', () => {
      describe('when computing diversity stats from a single sparse genome', () => {
        it('uses the recorder fallbacks and stores zeroed aggregate stats', () => {
          // Arrange
          const mathRandomSpy = jest
            .spyOn(Math, 'random')
            .mockImplementation(createCyclingRandom([0, 0.4, 0.8]));
          const telemetryContext: {
            _diversityStats?: unknown;
            _lineageEnabled: boolean;
            options: {
              diversityMetrics: {
                enabled: boolean;
              };
            };
            population: TelemetryGenomeStub[];
          } = {
            _lineageEnabled: false,
            options: {
              diversityMetrics: {
                enabled: true,
              },
            },
            population: [createGenome({ connectionCount: 0, nodeCount: 3 })],
          };

          // Act
          computeDiversityStats.call(telemetryContext);

          mathRandomSpy.mockRestore();

          // Assert
          expect(telemetryContext._diversityStats).toEqual(
            expect.objectContaining({
              graphletEntropy: 0,
              lineageMeanDepth: 0,
              lineageMeanPairDist: 0,
              meanCompat: 0,
              meanEntropy: expect.closeTo(0, 6),
              varCompat: 0,
              varEntropy: 0,
            }),
          );
        });
      });
    });

    describe('given no population snapshot is available', () => {
      describe('when computing diversity stats', () => {
        it('falls back to zeroed diversity aggregates', () => {
          // Arrange
          const telemetryContext = {
            options: {
              diversityMetrics: {
                enabled: true,
              },
            },
          };

          // Act
          computeDiversityStats.call(telemetryContext);

          // Assert
          expect(
            (telemetryContext as { _diversityStats?: unknown })._diversityStats,
          ).toEqual({
            graphletEntropy: 0,
            lineageMeanDepth: 0,
            lineageMeanPairDist: 0,
            meanCompat: 0,
            meanEntropy: 0,
            varCompat: 0,
            varEntropy: 0,
          });
        });
      });
    });

    describe('given a custom structural entropy helper returns no value', () => {
      describe('when computing diversity stats', () => {
        it('falls back to zero for the entropy aggregates', () => {
          // Arrange
          const telemetryContext = {
            _structuralEntropy: () => undefined,
            options: {
              diversityMetrics: {
                enabled: true,
                graphletSample: 1,
                pairSample: 1,
              },
            },
            population: [createGenome({ connectionCount: 0, nodeCount: 2 })],
          };

          // Act
          computeDiversityStats.call(telemetryContext);

          // Assert
          expect(
            (telemetryContext as { _diversityStats?: unknown })._diversityStats,
          ).toEqual({
            graphletEntropy: 0,
            lineageMeanDepth: 0,
            lineageMeanPairDist: 0,
            meanCompat: 0,
            meanEntropy: 0,
            varCompat: 0,
            varEntropy: 0,
          });
        });
      });
    });
  });

  describe('recordTelemetryEntry', () => {
    describe('given the recorder has no telemetry buffer yet', () => {
      describe('when recording a telemetry entry', () => {
        it('initializes the in-memory history and stores the entry', () => {
          // Arrange
          const telemetryContext = {};
          const telemetryEntry = {
            best: 4,
            gen: 2,
            hyper: 0,
            objImportance: {},
            ops: [],
            species: 1,
          };

          // Act
          recordTelemetryEntry.call(telemetryContext, telemetryEntry);

          // Assert
          expect((telemetryContext as { _telemetry?: unknown[] })._telemetry).toEqual([
            telemetryEntry,
          ]);
        });
      });
    });

    describe('given the recorder already has a full telemetry buffer and a stream callback', () => {
      describe('when recording a selected telemetry entry', () => {
        it('keeps the filtered entry, streams it, and trims the bounded history', () => {
          // Arrange
          const streamedEntries: unknown[] = [];
          const telemetryContext = {
            _telemetry: Array.from({ length: 500 }, (_, entryIndex) => ({
              best: entryIndex,
              gen: entryIndex,
              species: 1,
            })),
            _telemetrySelect: new Set(['detail']),
            options: {
              telemetryStream: {
                enabled: true,
                onEntry: (entry: unknown) => {
                  streamedEntries.push(entry);
                },
              },
            },
          };
          const telemetryEntry = {
            best: 10,
            detail: 'keep',
            drop: 'remove',
            gen: 501,
            hyper: 0,
            objImportance: {},
            ops: [],
            species: 2,
          };

          // Act
          recordTelemetryEntry.call(telemetryContext, telemetryEntry);

          // Assert
          expect({
            bufferedEntry: telemetryContext._telemetry.at(-1),
            bufferLength: telemetryContext._telemetry.length,
            streamedEntry: streamedEntries[0],
          }).toEqual({
            bufferedEntry: {
              best: 10,
              detail: 'keep',
              gen: 501,
              species: 2,
            },
            bufferLength: 500,
            streamedEntry: {
              best: 10,
              detail: 'keep',
              gen: 501,
              species: 2,
            },
          });
        });
      });
    });
  });

  describe('buildTelemetryEntry', () => {
    describe('given mono-objective telemetry has no explicit options or generation', () => {
      describe('when building the entry from a minimal context', () => {
        it('falls back to the mono-objective default base values', () => {
          // Arrange
          const telemetryContext = {};

          // Act
          const telemetryEntry = buildTelemetryEntry.call(telemetryContext, {});

          // Assert
          expect(telemetryEntry).toMatchObject({
            best: 0,
            gen: 0,
            hyper: 0,
            objImportance: {},
            ops: [],
            species: 0,
          });
        });
      });
    });

    describe('given multi-objective telemetry has no population or species snapshot', () => {
      describe('when building the entry from a minimal multi-objective context', () => {
        it('falls back to zeroed multi-objective summary values', () => {
          // Arrange
          const telemetryContext = {
            options: {
              multiObjective: { enabled: true },
            },
          };

          // Act
          const telemetryEntry = buildTelemetryEntry.call(telemetryContext, {});

          // Assert
          expect(telemetryEntry).toMatchObject({
            best: 0,
            fronts: [],
            gen: 0,
            hyper: 0,
            objImportance: {},
            ops: [],
            species: 0,
          });
        });
      });
    });
  });

  describe('structuralEntropy', () => {
    describe('given a graph snapshot has no nodes', () => {
      describe('when computing structural entropy', () => {
        it('falls back to a safe zero entropy value', () => {
          // Arrange
          const telemetryContext = { generation: 1 };
          const graph = {
            connections: [],
            nodes: [],
          };

          // Act
          const entropy = structuralEntropy.call(telemetryContext, graph);

          // Assert
          expect(entropy).toBe(0);
        });
      });
    });
  });
});