import { mutation } from '../../src/methods/mutation';
import Activation from '../../src/methods/activation';

describe('Mutation Methods', () => {
  describe('ADD_NODE', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.ADD_NODE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_NODE.name).toBe('ADD_NODE');
      });
    });
  });

  describe('SUB_NODE', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.SUB_NODE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_NODE.name).toBe('SUB_NODE');
      });
    });
    describe('Scenario: keep_gates property', () => {
      test('should have keep_gates property', () => {
        // Assert
        expect(mutation.SUB_NODE.keep_gates).toBe(true);
      });
    });
  });

  describe('ADD_CONN', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.ADD_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_CONN.name).toBe('ADD_CONN');
      });
    });
  });

  describe('SUB_CONN', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.SUB_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_CONN.name).toBe('SUB_CONN');
      });
    });
  });

  describe('MOD_WEIGHT', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.MOD_WEIGHT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.MOD_WEIGHT.name).toBe('MOD_WEIGHT');
      });
    });
    describe('Scenario: min property', () => {
      test('should have min property', () => {
        // Assert
        expect(mutation.MOD_WEIGHT.min).toBe(-1);
      });
    });
    describe('Scenario: max property', () => {
      test('should have max property', () => {
        // Assert
        expect(mutation.MOD_WEIGHT.max).toBe(1);
      });
    });
  });

  describe('MOD_BIAS', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.MOD_BIAS).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.MOD_BIAS.name).toBe('MOD_BIAS');
      });
    });
    describe('Scenario: min property', () => {
      test('should have min property', () => {
        // Assert
        expect(mutation.MOD_BIAS.min).toBe(-1);
      });
    });
    describe('Scenario: max property', () => {
      test('should have max property', () => {
        // Assert
        expect(mutation.MOD_BIAS.max).toBe(1);
      });
    });
  });

  describe('MOD_ACTIVATION', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.MOD_ACTIVATION).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.MOD_ACTIVATION.name).toBe('MOD_ACTIVATION');
      });
    });
    describe('Scenario: mutateOutput property', () => {
      test('should have mutateOutput property', () => {
        // Assert
        expect(mutation.MOD_ACTIVATION.mutateOutput).toBe(true);
      });
    });
    describe('Scenario: allowed property', () => {
      test('should have allowed property as an array', () => {
        // Assert
        expect(Array.isArray(mutation.MOD_ACTIVATION.allowed)).toBe(true);
      });
      test('should contain expected activation functions in allowed', () => {
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
        // Assert
        expect(mutation.MOD_ACTIVATION.allowed).toEqual(
          expect.arrayContaining(expectedActivations)
        );
      });
      test('should have allowed property with correct length', () => {
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
        // Assert
        expect(mutation.MOD_ACTIVATION.allowed.length).toBe(
          expectedActivations.length
        );
      });
    });
  });

  describe('ADD_SELF_CONN', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.ADD_SELF_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_SELF_CONN.name).toBe('ADD_SELF_CONN');
      });
    });
  });

  describe('SUB_SELF_CONN', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.SUB_SELF_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_SELF_CONN.name).toBe('SUB_SELF_CONN');
      });
    });
  });

  describe('ADD_GATE', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.ADD_GATE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_GATE.name).toBe('ADD_GATE');
      });
    });
  });

  describe('SUB_GATE', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.SUB_GATE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_GATE.name).toBe('SUB_GATE');
      });
    });
  });

  describe('ADD_BACK_CONN', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.ADD_BACK_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_BACK_CONN.name).toBe('ADD_BACK_CONN');
      });
    });
  });

  describe('SUB_BACK_CONN', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.SUB_BACK_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_BACK_CONN.name).toBe('SUB_BACK_CONN');
      });
    });
  });

  describe('SWAP_NODES', () => {
    describe('Scenario: Existence', () => {
      test('should exist', () => {
        // Assert
        expect(mutation.SWAP_NODES).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      test('should have correct name property', () => {
        // Assert
        expect(mutation.SWAP_NODES.name).toBe('SWAP_NODES');
      });
    });
    describe('Scenario: mutateOutput property', () => {
      test('should have mutateOutput property', () => {
        // Assert
        expect(mutation.SWAP_NODES.mutateOutput).toBe(true);
      });
    });
  });

  describe('ALL', () => {
    describe('Scenario: Existence', () => {
      test('should exist and be an array', () => {
        // Assert
        expect(mutation.ALL).toBeDefined();
      });
      test('should be an array', () => {
        // Assert
        expect(Array.isArray(mutation.ALL)).toBe(true);
      });
    });
    describe('Scenario: Contains all mutation methods', () => {
      test('should contain all individual mutation methods', () => {
        // Arrange
        const expectedMethods = [
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
        ];
        // Assert
        expect(mutation.ALL).toEqual(expect.arrayContaining(expectedMethods));
      });
      test('should have correct length', () => {
        // Arrange
        const expectedMethods = [
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
        ];
        // Assert
        expect(mutation.ALL.length).toBe(expectedMethods.length);
      });
    });
  });

  describe('FFW', () => {
    describe('Scenario: Existence', () => {
      test('should exist and be an array', () => {
        // Assert
        expect(mutation.FFW).toBeDefined();
      });
      test('should be an array', () => {
        // Assert
        expect(Array.isArray(mutation.FFW)).toBe(true);
      });
    });
    describe('Scenario: Contains only feedforward-compatible mutation methods', () => {
      test('should contain only feedforward-compatible mutation methods', () => {
        // Arrange
        const expectedMethods = [
          mutation.ADD_NODE,
          mutation.SUB_NODE,
          mutation.ADD_CONN,
          mutation.SUB_CONN,
          mutation.MOD_WEIGHT,
          mutation.MOD_BIAS,
          mutation.MOD_ACTIVATION,
          mutation.SWAP_NODES,
        ];
        // Assert
        expect(mutation.FFW).toEqual(expect.arrayContaining(expectedMethods));
      });
      test('should have correct length', () => {
        // Arrange
        const expectedMethods = [
          mutation.ADD_NODE,
          mutation.SUB_NODE,
          mutation.ADD_CONN,
          mutation.SUB_CONN,
          mutation.MOD_WEIGHT,
          mutation.MOD_BIAS,
          mutation.MOD_ACTIVATION,
          mutation.SWAP_NODES,
        ];
        // Assert
        expect(mutation.FFW.length).toBe(expectedMethods.length);
      });
    });
    describe('Scenario: Does not contain recurrent mutation methods', () => {
      test('should not contain recurrent mutation methods', () => {
        // Arrange
        const recurrentMethods = [
          mutation.ADD_GATE,
          mutation.SUB_GATE,
          mutation.ADD_SELF_CONN,
          mutation.SUB_SELF_CONN,
          mutation.ADD_BACK_CONN,
          mutation.SUB_BACK_CONN,
        ];
        // Assert
        recurrentMethods.forEach((method) => {
          expect(mutation.FFW).not.toContain(method);
        });
      });
    });
  });
});
