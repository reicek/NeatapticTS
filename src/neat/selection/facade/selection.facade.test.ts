import {
  getAverage as getAverageScore,
  getFittest as getFittestGenome,
  sort as sortPopulationSummary,
  type NeatPopulationSummaryFacadeHost,
} from './selection.facade';

type SummaryGenome = {
  score?: number;
};

type SummaryHostHarness = NeatPopulationSummaryFacadeHost & {
  evaluate: jest.Mock;
};

function createSummaryHost(input: {
  population: SummaryGenome[];
  evaluatedScores?: number[];
}): SummaryHostHarness {
  const summaryHost = {
    population: input.population,
    options: {},
    _getRNG: () => () => 0.5,
    sort: () =>
      sortPopulationSummary(
        summaryHost as unknown as NeatPopulationSummaryFacadeHost,
      ),
    evaluate: jest.fn(() => {
      input.evaluatedScores?.forEach((score, genomeIndex) => {
        const genome = summaryHost.population[genomeIndex];

        if (genome) genome.score = score;
      });
    }),
  } as unknown as SummaryHostHarness;

  return summaryHost;
}

describe('neat selection facade chapter', () => {
  describe('sort', () => {
    describe('given the population starts out of descending score order', () => {
      it('reorders the population best-first', () => {
        // Arrange
        const summaryHost = createSummaryHost({
          population: [{ score: 1 }, { score: 2 }],
        });

        // Act
        sortPopulationSummary(summaryHost);

        // Assert
        expect(summaryHost.population.map((genome) => genome.score)).toEqual([
          2, 1,
        ]);
      });
    });
  });

  describe('getFittest', () => {
    describe('given the population still needs evaluation and sorting', () => {
      it('returns the champion score after restoring evaluation state', () => {
        // Arrange
        const summaryHost = createSummaryHost({
          population: [{ score: undefined }, { score: undefined }],
          evaluatedScores: [1, 3],
        });

        // Act
        const champion = getFittestGenome(summaryHost);

        // Assert
        expect(champion.score).toBe(3);
      });
    });
  });

  describe('getAverage', () => {
    describe('given the population needs evaluation before averaging', () => {
      it('returns the arithmetic mean after evaluation fills the scores', () => {
        // Arrange
        const summaryHost = createSummaryHost({
          population: [{ score: undefined }, { score: undefined }],
          evaluatedScores: [1, 3],
        });

        // Act
        const averageScore = getAverageScore(summaryHost);

        // Assert
        expect(averageScore).toBe(2);
      });
    });
  });
});
