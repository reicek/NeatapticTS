import * as pruningModule from '../pruning';
import { applyAdaptivePruning, applyEvolutionPruning } from './pruning.facade';

jest.mock('../pruning', () => ({
  applyAdaptivePruning: jest.fn(),
  applyEvolutionPruning: jest.fn(),
}));

type PruningFacadeHost = {
  options: Record<string, unknown>;
  generation: number;
  population: unknown[];
};

function createPruningFacadeHost(): PruningFacadeHost {
  return {
    options: {},
    generation: 0,
    population: [],
  };
}

describe('pruning facade chapter', () => {
  const mockedEvolutionPruning = jest.mocked(
    pruningModule.applyEvolutionPruning,
  );
  const mockedAdaptivePruning = jest.mocked(pruningModule.applyAdaptivePruning);

  beforeEach(() => {
    mockedEvolutionPruning.mockReset();
    mockedAdaptivePruning.mockReset();
  });

  describe('applyEvolutionPruning', () => {
    describe('given the pruning module loads successfully', () => {
      it('forwards the host into the scheduled pruning controller', async () => {
        // Arrange
        const pruningFacadeHost = createPruningFacadeHost();

        // Act
        await applyEvolutionPruning(pruningFacadeHost as never);

        // Assert
        expect(mockedEvolutionPruning.mock.contexts[0]).toBe(pruningFacadeHost);
      });
    });

    describe('given the scheduled pruning controller throws', () => {
      it('swallows the optional pruning failure', async () => {
        // Arrange
        const pruningFacadeHost = createPruningFacadeHost();
        mockedEvolutionPruning.mockImplementation(() => {
          throw new Error('optional pruning failure');
        });

        // Assert
        await expect(
          applyEvolutionPruning(pruningFacadeHost as never),
        ).resolves.toBeUndefined();
      });
    });
  });

  describe('applyAdaptivePruning', () => {
    describe('given the pruning module loads successfully', () => {
      it('forwards the host into the adaptive pruning controller', async () => {
        // Arrange
        const pruningFacadeHost = createPruningFacadeHost();

        // Act
        await applyAdaptivePruning(pruningFacadeHost as never);

        // Assert
        expect(mockedAdaptivePruning.mock.contexts[0]).toBe(pruningFacadeHost);
      });
    });

    describe('given the adaptive pruning controller throws', () => {
      it('swallows the optional adaptive pruning failure', async () => {
        // Arrange
        const pruningFacadeHost = createPruningFacadeHost();
        mockedAdaptivePruning.mockImplementation(() => {
          throw new Error('optional adaptive pruning failure');
        });

        // Assert
        await expect(
          applyAdaptivePruning(pruningFacadeHost as never),
        ).resolves.toBeUndefined();
      });
    });
  });
});
