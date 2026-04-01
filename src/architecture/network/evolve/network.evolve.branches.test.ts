import Network from '../network';

type TrainingSet = Parameters<Network['evolve']>[0];

jest.mock('../../../neat', () => {
  type FitnessFunction = (network: Network) => number;
  type OptionsRecord = Record<string, unknown>;

  return {
    __esModule: true,
    default: class MockNeat {
      input: number;
      output: number;
      fitnessFn: FitnessFunction;
      options: OptionsRecord;
      generation = 0;

      constructor(
        input: number,
        output: number,
        fitnessFn: FitnessFunction,
        options: OptionsRecord,
      ) {
        this.input = input;
        this.output = output;
        this.fitnessFn = fitnessFn;
        this.options = options;
      }

      async evolve(): Promise<Network> {
        // Step 1: Increment generation counter to emulate progress.
        this.generation += 1;

        // Step 2: Return a genome with NaN score to drive the infinite-error break path.
        const genome = new Network(this.input, this.output);
        genome.score = NaN;
        return genome;
      }

      _warnIfNoBestGenome() {
        console.warn(
          'Evolution completed without finding a valid best genome (mock)',
        );
      }
    },
  };
});

describe('network evolve branch coverage', () => {
  describe('Network.evolve()', () => {
    describe('given zero iterations are requested', () => {
      describe('when evolve() is called', () => {
        it('emits the no-best-genome warning path', async () => {
          // Arrange
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});
          const network = new Network(1, 1, { seed: 473 });
          const trainingSet: TrainingSet = [{ input: [0.1], output: [0.2] }];

          try {
            // Act
            await network.evolve(trainingSet, { iterations: 0 });
            const warningWasEmitted = warnSpy.mock.calls.some(([message]) =>
              /valid best genome/.test(String(message)),
            );

            // Assert
            expect(warningWasEmitted).toBe(true);
          } finally {
            warnSpy.mockRestore();
          }
        });
      });
    });

    describe('given repeated invalid scores occur during evolution', () => {
      describe('when evolve() is called with a high iteration cap', () => {
        it('terminates before the configured upper bound', async () => {
          // Arrange
          const network = new Network(1, 1, { seed: 474 });
          const trainingSet: TrainingSet = [{ input: [0.9], output: [0.1] }];

          // Act
          const evolutionSummary = await network.evolve(trainingSet, {
            iterations: 50,
          });

          // Assert
          expect(evolutionSummary.iterations < 50).toBe(true);
        });
      });
    });
  });
});
