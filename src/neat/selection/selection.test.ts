import Network from '../../architecture/network';
import Neat from '../../neat';
import { SelectionTournamentOverflowError } from './core/selection.core.errors';

type SelectionHost = {
  evaluate: () => Promise<void>;
  _getRNG: () => () => number;
  population: Array<{ score?: number }>;
  getParent: () => { score?: number };
};

describe('neat selection chapter', () => {
  describe('power parent selection', () => {
    describe('given a population whose leading scores are out of order', () => {
      const scoreByNegativeNodeCount = (network: Network) =>
        -network.nodes.length;

      let leadingScore: number | undefined;
      let secondScore: number | undefined;

      beforeAll(async () => {
        // Arrange
        const neat = new Neat(2, 1, scoreByNegativeNodeCount, {
          popsize: 6,
          seed: 555,
          selection: { name: 'POWER', power: 1 },
        });

        await neat.evaluate();
        neat.population[0].score = 1;
        neat.population[1].score = 5;

        // Act
        neat.getParent();
        leadingScore = neat.population[0].score;
        secondScore = neat.population[1].score;
      });

      describe('when parent selection reads the unsorted population', () => {
        it('restores descending score order before choosing a parent', () => {
          // Assert
          expect(leadingScore).toBeGreaterThanOrEqual(secondScore ?? 0);
        });
      });
    });
  });

  describe('tournament parent selection', () => {
    describe('given a tournament bracket larger than the population', () => {
      const scoreWithoutSideEffects = (network: Network) => {
        void network;
        return 1;
      };

      it('throws the tournament overflow error when overflow is not suppressed', () => {
        // Arrange
        const neat = new Neat(2, 1, scoreWithoutSideEffects, {
          popsize: 3,
          seed: 556,
          selection: { name: 'TOURNAMENT', size: 10, probability: 0.5 },
        });
        const selectParent = () => neat.getParent();

        // Act
        const tournamentSelection = selectParent;

        // Assert
        expect(tournamentSelection).toThrow(SelectionTournamentOverflowError);
      });
    });
  });

  describe('fitness-proportionate parent selection', () => {
    describe('given mixed negative and positive scores with a fixed roulette draw', () => {
      it('shifts the score space and still selects the genome that owns the sampled threshold', async () => {
        // Arrange
        const scoreWithoutSideEffects = (network: Network) => {
          void network;
          return 1;
        };
        const neat = new Neat(2, 1, scoreWithoutSideEffects, {
          popsize: 4,
          seed: 557,
          selection: { name: 'FITNESS_PROPORTIONATE' },
        });
        const selectionHost = neat as unknown as SelectionHost;

        await selectionHost.evaluate();
        selectionHost._getRNG = () => () => 0.95;
        selectionHost.population[0].score = -5;
        selectionHost.population[1].score = -1;
        selectionHost.population[2].score = 2;
        selectionHost.population[3].score = 3;

        // Act
        const chosenParent = selectionHost.getParent();

        // Assert
        expect(chosenParent.score).toBe(3);
      });
    });

    describe('given a roulette threshold that equals total shifted fitness', () => {
      it('falls back to selecting a random population member when threshold scan misses', async () => {
        // Arrange
        const scoreWithoutSideEffects = (network: Network) => {
          void network;
          return 1;
        };
        const neat = new Neat(2, 1, scoreWithoutSideEffects, {
          popsize: 4,
          seed: 558,
          selection: { name: 'FITNESS_PROPORTIONATE' },
        });
        const selectionHost = neat as unknown as SelectionHost;
        const randomDraws = [1, 0.8];

        await selectionHost.evaluate();
        selectionHost._getRNG = () => () => randomDraws.shift() ?? 0;
        selectionHost.population[0].score = 1;
        selectionHost.population[1].score = 2;
        selectionHost.population[2].score = 3;
        selectionHost.population[3].score = 4;

        // Act
        const chosenParent = selectionHost.getParent();

        // Assert
        expect(chosenParent.score).toBe(4);
      });
    });
  });

  describe('tournament parent fallback path', () => {
    describe('given a zero-sized tournament configuration', () => {
      it('returns undefined when no tournament participants are sampled', () => {
        // Arrange
        const scoreWithoutSideEffects = (network: Network) => {
          void network;
          return 1;
        };
        const neat = new Neat(2, 1, scoreWithoutSideEffects, {
          popsize: 3,
          seed: 559,
          selection: { name: 'TOURNAMENT', size: 0, probability: 0.5 },
        });
        const selectionHost = neat as unknown as {
          getParent: () => { score?: number } | undefined;
        };

        // Act
        const chosenParent = selectionHost.getParent();

        // Assert
        expect(chosenParent).toBeUndefined();
      });
    });
  });
});
