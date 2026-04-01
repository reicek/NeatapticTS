import Activation from '../activation/activation';
import mutation from './mutation';

describe('mutation', () => {
  describe('given the exported mutation shelf', () => {
    describe('when reading the available mutation entries', () => {
      it('exposes the expected keys', () => {
        // Arrange
        const expectedKeys = [
          'ADD_BACK_CONN',
          'ADD_CONN',
          'ADD_GATE',
          'ADD_GRU_NODE',
          'ADD_LSTM_NODE',
          'ADD_NODE',
          'ADD_SELF_CONN',
          'ALL',
          'BATCH_NORM',
          'FFW',
          'MOD_ACTIVATION',
          'MOD_BIAS',
          'MOD_WEIGHT',
          'REINIT_WEIGHT',
          'SUB_BACK_CONN',
          'SUB_CONN',
          'SUB_GATE',
          'SUB_NODE',
          'SUB_SELF_CONN',
          'SWAP_NODES',
        ];

        // Act
        const actualKeys = Object.keys(mutation).toSorted();

        // Assert
        expect(actualKeys).toStrictEqual(expectedKeys);
      });
    });
  });

  describe('SUB_NODE', () => {
    describe('when reading its removal policy', () => {
      it('preserves gates by default', () => {
        // Arrange
        const expectedKeepGates = true;

        // Act
        const actualKeepGates = mutation.SUB_NODE.keep_gates;

        // Assert
        expect(actualKeepGates).toBe(expectedKeepGates);
      });
    });
  });

  describe('MOD_WEIGHT', () => {
    describe('when reading its numeric bounds', () => {
      it('uses the expected range', () => {
        // Arrange
        const expectedRange = { min: -1, max: 1 };

        // Act
        const actualRange = {
          min: mutation.MOD_WEIGHT.min,
          max: mutation.MOD_WEIGHT.max,
        };

        // Assert
        expect(actualRange).toStrictEqual(expectedRange);
      });
    });
  });

  describe('MOD_BIAS', () => {
    describe('when reading its numeric bounds', () => {
      it('uses the expected range', () => {
        // Arrange
        const expectedRange = { min: -1, max: 1 };

        // Act
        const actualRange = {
          min: mutation.MOD_BIAS.min,
          max: mutation.MOD_BIAS.max,
        };

        // Assert
        expect(actualRange).toStrictEqual(expectedRange);
      });
    });
  });

  describe('MOD_ACTIVATION', () => {
    describe('when reading the activation policy', () => {
      it('allows output-node mutation', () => {
        // Arrange
        const expectedMutateOutput = true;

        // Act
        const actualMutateOutput = mutation.MOD_ACTIVATION.mutateOutput;

        // Assert
        expect(actualMutateOutput).toBe(expectedMutateOutput);
      });
    });

    describe('when reading the allowed activation shelf', () => {
      it('uses the expected activation list', () => {
        // Arrange
        const expectedActivations = [
          Activation.logistic,
          Activation.tanh,
          Activation.relu,
          Activation.identity,
          Activation.step,
          Activation.softsign,
          Activation.sinusoid,
          Activation.gaussian,
          Activation.bentIdentity,
          Activation.bipolar,
          Activation.bipolarSigmoid,
          Activation.hardTanh,
          Activation.absolute,
          Activation.inverse,
          Activation.selu,
          Activation.softplus,
          Activation.swish,
          Activation.gelu,
          Activation.mish,
        ];

        // Act
        const actualActivations = mutation.MOD_ACTIVATION.allowed;

        // Assert
        expect(actualActivations).toStrictEqual(expectedActivations);
      });
    });
  });

  describe('SWAP_NODES', () => {
    describe('when reading the swap policy', () => {
      it('allows output-node swapping', () => {
        // Arrange
        const expectedMutateOutput = true;

        // Act
        const actualMutateOutput = mutation.SWAP_NODES.mutateOutput;

        // Assert
        expect(actualMutateOutput).toBe(expectedMutateOutput);
      });
    });
  });

  describe('ALL', () => {
    describe('when reading the broad mutation shelf', () => {
      it('contains every supported mutation operator in order', () => {
        // Arrange
        const expectedMutationShelf = [
          mutation.ADD_NODE,
          mutation.SUB_NODE,
          mutation.ADD_CONN,
          mutation.SUB_CONN,
          mutation.MOD_WEIGHT,
          mutation.MOD_BIAS,
          mutation.MOD_ACTIVATION,
          mutation.ADD_GATE,
          mutation.SUB_GATE,
          mutation.ADD_SELF_CONN,
          mutation.SUB_SELF_CONN,
          mutation.ADD_BACK_CONN,
          mutation.SUB_BACK_CONN,
          mutation.SWAP_NODES,
          mutation.REINIT_WEIGHT,
          mutation.BATCH_NORM,
          mutation.ADD_LSTM_NODE,
          mutation.ADD_GRU_NODE,
        ];

        // Act
        const actualMutationShelf = mutation.ALL;

        // Assert
        expect(actualMutationShelf).toStrictEqual(expectedMutationShelf);
      });
    });
  });

  describe('FFW', () => {
    describe('when reading the feedforward-safe mutation shelf', () => {
      it('contains only the non-recurrent subset', () => {
        // Arrange
        const expectedMutationShelf = [
          mutation.ADD_NODE,
          mutation.SUB_NODE,
          mutation.ADD_CONN,
          mutation.SUB_CONN,
          mutation.MOD_WEIGHT,
          mutation.MOD_BIAS,
          mutation.MOD_ACTIVATION,
          mutation.SWAP_NODES,
          mutation.REINIT_WEIGHT,
          mutation.BATCH_NORM,
        ];

        // Act
        const actualMutationShelf = mutation.FFW;

        // Assert
        expect(actualMutationShelf).toStrictEqual(expectedMutationShelf);
      });
    });
  });
});
