import Network from '../../../architecture/network/network';
import {
  DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
  DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
  buildMultiObjectiveMetrics,
  exportParetoArchiveJsonl,
  reconstructParetoFronts,
  sliceParetoArchive,
} from './multiobjective.metrics';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('multiobjective metrics chapter', () => {
  describe('buildMultiObjectiveMetrics()', () => {
    describe('given a genome with explicit rank, crowding, and positive score', () => {
      it('returns the annotated rank value', () => {
        const genome = new Network(1, 1);
        (genome as Record<string, unknown>)._moRank = 2;
        (genome as Record<string, unknown>)._moCrowd = 0.5;
        genome.score = 10;
        const result = buildMultiObjectiveMetrics([genome]);
        expect(result[0].rank).toBe(2);
      });
    });

    describe('given a genome with no rank annotation', () => {
      it('defaults rank to zero', () => {
        const genome = new Network(1, 1);
        const result = buildMultiObjectiveMetrics([genome]);
        expect(result[0].rank).toBe(0);
      });
    });

    describe('given a genome with no crowding annotation', () => {
      it('defaults crowding to zero', () => {
        const genome = new Network(1, 1);
        const result = buildMultiObjectiveMetrics([genome]);
        expect(result[0].crowding).toBe(0);
      });
    });

    describe('given a genome with score of zero', () => {
      it('returns score as zero via the || 0 fallback', () => {
        const genome = new Network(1, 1);
        genome.score = 0;
        const result = buildMultiObjectiveMetrics([genome]);
        expect(result[0].score).toBe(0);
      });
    });

    describe('given a genome with a positive score', () => {
      it('returns the genome score directly', () => {
        const genome = new Network(1, 1);
        genome.score = 7;
        const result = buildMultiObjectiveMetrics([genome]);
        expect(result[0].score).toBe(7);
      });
    });
  });

  describe('reconstructParetoFronts()', () => {
    describe('given multi-objective mode is disabled', () => {
      it('returns a single front containing the whole population', () => {
        const population = [new Network(1, 1), new Network(1, 1)];
        const result = reconstructParetoFronts(population, 3, false);
        expect(result.length).toBe(1);
      });
    });

    describe('given two genomes in rank 0 and multi-objective enabled', () => {
      it('returns one front with both genomes', () => {
        const population = [new Network(1, 1), new Network(1, 1)];
        (population[0] as Record<string, unknown>)._moRank = 0;
        (population[1] as Record<string, unknown>)._moRank = 0;
        const result = reconstructParetoFronts(population, 3, true);
        expect(result.length).toBe(1);
      });
    });

    describe('given a genome with no rank annotation when multi-objective is enabled', () => {
      it('places it in front 0 via the ?? 0 fallback', () => {
        const genome = new Network(1, 1);
        const result = reconstructParetoFronts([genome], 3, true);
        expect(result[0]).toContain(genome);
      });
    });

    describe('given maxFronts is larger than the number of populated fronts', () => {
      it('stops early when a front has no members (break path)', () => {
        const genome = new Network(1, 1);
        (genome as Record<string, unknown>)._moRank = 0;
        const result = reconstructParetoFronts([genome], 5, true);
        expect(result.length).toBe(1);
      });
    });

    describe('given maxFronts is omitted', () => {
      it('uses the default limit and returns the populated fronts', () => {
        const genome = new Network(1, 1);
        (genome as Record<string, unknown>)._moRank = 0;
        const result = reconstructParetoFronts(
          [genome],
          undefined as unknown as number,
          true,
        );
        expect(result.length).toBe(1);
      });
    });
  });

  describe('sliceParetoArchive()', () => {
    describe('given an archive longer than the default limit', () => {
      it('returns only the last DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES items', () => {
        const archive = Array.from({ length: 100 }, (_, i) => i);
        const result = sliceParetoArchive(archive);
        expect(result.length).toBe(DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES);
      });
    });

    describe('given an explicit maxEntries argument', () => {
      it('returns at most that many trailing items', () => {
        const archive = [1, 2, 3, 4, 5];
        const result = sliceParetoArchive(archive, 3);
        expect(result).toEqual([3, 4, 5]);
      });
    });
  });

  describe('exportParetoArchiveJsonl()', () => {
    describe('given an archive with three entries', () => {
      it('returns newline-delimited JSON with three lines', () => {
        const archive = [{ a: 1 }, { a: 2 }, { a: 3 }];
        const result = exportParetoArchiveJsonl(archive);
        expect(result.split('\n').length).toBe(3);
      });
    });

    describe('given an archive longer than the default jsonl limit', () => {
      it('truncates to DEFAULT_PARETO_ARCHIVE_JSONL_MAX lines', () => {
        const archive = Array.from({ length: 200 }, (_, i) => ({ i }));
        const result = exportParetoArchiveJsonl(archive);
        expect(result.split('\n').length).toBe(
          DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
        );
      });
    });

    describe('given an explicit maxEntries argument', () => {
      it('returns only that many entries', () => {
        const archive = [{ x: 1 }, { x: 2 }, { x: 3 }, { x: 4 }];
        const result = exportParetoArchiveJsonl(archive, 2);
        expect(result.split('\n').length).toBe(2);
      });
    });
  });
});
