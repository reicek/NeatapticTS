import {
  exportSpeciesHistoryCSV,
  exportTelemetryCSV,
  exportTelemetryJSONL,
} from './telemetry.exports';
import type {
  SpeciesHistoryEntry,
  TelemetryEntry,
} from '../../shared/neat.shared.types';

type TelemetryExportHost = {
  _telemetry: TelemetryEntry[];
};

type SpeciesHistoryExportHost = {
  _speciesHistory?: SpeciesHistoryEntry[];
  _species?: never;
  generation?: number;
};

function createTelemetryEntry(input: {
  ops?: TelemetryEntry['ops'];
  fronts?: TelemetryEntry['fronts'];
  objectives?: string[];
  objAges?: TelemetryEntry['objAges'];
  objEvents?: TelemetryEntry['objEvents'];
  objImportance?: TelemetryEntry['objImportance'];
  speciesAlloc?: TelemetryEntry['speciesAlloc'];
  diversity?: TelemetryEntry['diversity'];
  complexity?: TelemetryEntry['complexity'];
  perf?: TelemetryEntry['perf'];
  lineage?: TelemetryEntry['lineage'];
  rng?: TelemetryEntry['rng'];
}): TelemetryEntry {
  return {
    gen: 3,
    best: 7,
    species: 2,
    hyper: 0,
    ops: input.ops ?? [],
    fronts: input.fronts,
    objImportance: input.objImportance ?? {},
    objectives: input.objectives,
    objAges: input.objAges,
    objEvents: input.objEvents,
    speciesAlloc: input.speciesAlloc,
    diversity: input.diversity,
    complexity: input.complexity,
    perf: input.perf,
    lineage: input.lineage,
    rng: input.rng,
  };
}

function createSparseTelemetryEntry(): TelemetryEntry {
  return {
    gen: 3,
    best: 7,
    species: 2,
    hyper: 0,
  } as TelemetryEntry;
}

describe('neat telemetry exports chapter', () => {
  describe('exportTelemetryJSONL', () => {
    describe('given a telemetry entry includes nested runtime metric families', () => {
      it('preserves the full-fidelity telemetry object in the JSONL line', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [
            createTelemetryEntry({
              ops: [{ op: 'ADD_NODE', succ: 3, att: 5 }],
              complexity: {
                meanNodes: 4,
                meanConns: 6,
                maxNodes: 5,
                maxConns: 8,
                meanEnabledRatio: 0.75,
                growthNodes: 1,
                growthConns: 2,
                budgetMaxNodes: 10,
                budgetMaxConns: 20,
              },
              perf: { evalMs: 12, evolveMs: 7 },
              lineage: {
                parents: [1, 2],
                depthBest: 2,
                meanDepth: 1.5,
                inbreeding: 0,
                ancestorUniq: 0.6,
              },
            }),
          ],
        };

        // Act
        const [firstLine] = exportTelemetryJSONL
          .call(telemetryExportHost)
          .split(/\r?\n/);
        const parsedTelemetryEntry = JSON.parse(firstLine);

        // Assert
        expect(parsedTelemetryEntry).toMatchObject({
          gen: 3,
          best: 7,
          species: 2,
          hyper: 0,
          ops: [{ op: 'ADD_NODE', succ: 3, att: 5 }],
          complexity: { meanNodes: 4 },
          perf: { evalMs: 12 },
          lineage: { depthBest: 2 },
        });
      });
    });
  });

  describe('exportTelemetryCSV', () => {
    describe('given the sampled telemetry window contains operator snapshots', () => {
      it('includes an ops column in the exported header row', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [
            createTelemetryEntry({
              ops: [{ op: 'ADD_NODE', succ: 3, att: 5 }],
            }),
          ],
        };

        // Act
        const csv = exportTelemetryCSV.call(telemetryExportHost, 50);
        const headers = csv.split(/\r?\n/)[0].split(',');

        // Assert
        expect(headers.includes('ops')).toBe(true);
      });
    });

    describe('given the sampled telemetry window records the active objective keys', () => {
      it('serializes the objectives array into the objectives column', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [createTelemetryEntry({ objectives: ['fitness'] })],
        };

        // Act
        const csv = exportTelemetryCSV.call(telemetryExportHost, 50);
        const [headerRow, firstRow] = csv.split(/\r?\n/);
        const objectivesColumnIndex = headerRow
          .split(',')
          .indexOf('objectives');
        const objectivesCell = firstRow.split(',')[objectivesColumnIndex];

        // Assert
        expect(objectivesCell).toBe('["fitness"]');
      });
    });
    describe('given the sampled telemetry window records objective ages', () => {
      it('includes an objAges column in the exported header row', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [
            createTelemetryEntry({ objAges: { fitness: 7, entropy: 1 } }),
          ],
        };

        // Act
        const headers = exportTelemetryCSV
          .call(telemetryExportHost, 50)
          .split(/\r?\n/)[0]
          .split(',');

        // Assert
        expect(headers.includes('objAges')).toBe(true);
      });
    });

    describe('given the sampled telemetry window mixes rich and sparse entries', () => {
      it('serializes the rich row while leaving missing optional cells empty on the sparse row', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [
            createTelemetryEntry({
              objectives: ['fitness'],
              objAges: { fitness: 7 },
              objEvents: [{ gen: 3, key: 'fitness', type: 'add' }],
              objImportance: { fitness: { range: 1, var: 0.25 } },
              fronts: [1, 2],
              ops: [{ op: 'ADD_NODE', succ: 3, att: 5 }],
              speciesAlloc: [{ id: 1, alloc: 2 }],
              complexity: {
                meanNodes: 4,
                meanConns: 6,
                maxNodes: 5,
                maxConns: 8,
                meanEnabledRatio: 0.75,
                growthNodes: 1,
                growthConns: 2,
                budgetMaxNodes: 10,
                budgetMaxConns: 20,
              },
              perf: { evalMs: 12, evolveMs: 7 },
              lineage: {
                parents: [1, 2],
                depthBest: 2,
                meanDepth: 1.5,
                inbreeding: 0,
                ancestorUniq: 0.6,
              },
              diversity: {
                meanCompat: 0.5,
                varCompat: 0.1,
                meanEntropy: 0.2,
                varEntropy: 0.05,
                graphletEntropy: 0.3,
                lineageMeanDepth: 1.2,
                lineageMeanPairDist: 0.4,
              },
            }),
            createSparseTelemetryEntry(),
          ],
        };

        // Act
        const csv = exportTelemetryCSV.call(telemetryExportHost, 50);

        // Assert
        expect([
          csv.includes('complexity.meanNodes'),
          csv.includes('perf.evalMs'),
          csv.includes('lineage.depthBest'),
          csv.includes('diversity.lineageMeanDepth'),
          csv.includes('fronts'),
          csv.includes('ops'),
          csv.includes('objectives'),
          csv.includes('objAges'),
          csv.includes('objEvents'),
          csv.includes('speciesAlloc'),
          csv.includes('objImportance'),
        ]).toEqual([
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
        ]);
      });
    });

    describe('given the sampled telemetry window records species allocation snapshots', () => {
      it('includes a speciesAlloc column in the exported header row', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [
            createTelemetryEntry({
              speciesAlloc: [{ id: 1, alloc: 2 }],
            }),
          ],
        };

        // Act
        const headers = exportTelemetryCSV
          .call(telemetryExportHost, 50)
          .split(/\r?\n/)[0]
          .split(',');

        // Assert
        expect(headers.includes('speciesAlloc')).toBe(true);
      });
    });

    describe('given a telemetry entry includes complexity, performance, and lineage blocks', () => {
      it('adds the corresponding flattened headers to the CSV export', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [
            createTelemetryEntry({
              diversity: {
                meanCompat: 0.5,
                varCompat: 0.1,
                meanEntropy: 0.2,
                varEntropy: 0.05,
                graphletEntropy: 0.3,
                lineageMeanDepth: 1.2,
                lineageMeanPairDist: 0.4,
              },
              complexity: {
                meanNodes: 4,
                meanConns: 6,
                maxNodes: 5,
                maxConns: 8,
                meanEnabledRatio: 0.75,
                growthNodes: 1,
                growthConns: 2,
                budgetMaxNodes: 10,
                budgetMaxConns: 20,
              },
              perf: { evalMs: 12, evolveMs: 7 },
              lineage: {
                parents: [1, 2],
                depthBest: 2,
                meanDepth: 1.5,
                inbreeding: 0,
                ancestorUniq: 0.6,
              },
            }),
          ],
        };

        // Act
        const headers = exportTelemetryCSV
          .call(telemetryExportHost, 50)
          .split(/\r?\n/)[0]
          .split(',');

        // Assert
        expect(headers).toEqual(
          expect.arrayContaining([
            'complexity.meanNodes',
            'diversity.lineageMeanDepth',
            'perf.evalMs',
            'lineage.depthBest',
          ]),
        );
      });
    });

    describe('given the telemetry buffer is absent', () => {
      it('returns an empty CSV string without throwing', () => {
        // Arrange
        const telemetryExportHost = {} as TelemetryExportHost;

        // Act
        const csv = exportTelemetryCSV.call(telemetryExportHost);

        // Assert
        expect(csv).toBe('');
      });
    });

    describe('given the sampled telemetry window omits objImportance entirely', () => {
      it('leaves the objImportance header out of the exported header row', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [createSparseTelemetryEntry()],
        };

        // Act
        const headers = exportTelemetryCSV.call(telemetryExportHost).split(
          /\r?\n/,
        )[0].split(',');

        // Assert
        expect(headers.includes('objImportance')).toBe(false);
      });
    });

    describe('given the sampled telemetry window records RNG state snapshots', () => {
      it('includes an rng column in the exported header row', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [createTelemetryEntry({ rng: 123 })],
        };

        // Act
        const headers = exportTelemetryCSV
          .call(telemetryExportHost, 50)
          .split(/\r?\n/)[0]
          .split(',');

        // Assert
        expect(headers.includes('rng')).toBe(true);
      });

      it('serializes the RNG snapshot into the rng column cell', () => {
        // Arrange
        const telemetryExportHost: TelemetryExportHost = {
          _telemetry: [createTelemetryEntry({ rng: 123 })],
        };

        // Act
        const [headerRow, firstRow] = exportTelemetryCSV
          .call(telemetryExportHost, 50)
          .split(/\r?\n/);
        const rngColumnIndex = headerRow.split(',').indexOf('rng');
        const rngCell = firstRow.split(',')[rngColumnIndex];

        // Assert
        expect(rngCell).toBe('123');
      });
    });
  });

  describe('exportSpeciesHistoryCSV', () => {
    describe('given the host has no recorded species history yet', () => {
      it('returns the fallback species history header row', () => {
        // Arrange
        const speciesHistoryExportHost = {} as SpeciesHistoryExportHost;

        // Act
        const csv = exportSpeciesHistoryCSV.call(speciesHistoryExportHost);

        // Assert
        expect(csv).toBe('generation,id,size,best,lastImproved');
      });
    });

    describe('given the host already has species history entries', () => {
      it('serializes the existing history into a CSV payload', () => {
        // Arrange
        const speciesHistoryExportHost: SpeciesHistoryExportHost = {
          _speciesHistory: [
            {
              generation: 4,
              stats: [{ id: 1, size: 2, bestScore: 3, lastImproved: 4 }],
            },
          ],
        };

        // Act
        const csv = exportSpeciesHistoryCSV.call(speciesHistoryExportHost);

        // Assert
        expect(csv.includes('4,1,2,3,4')).toBe(true);
      });
    });
  });
});
