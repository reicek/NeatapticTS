import { mutation } from '../../src/methods/mutation';
import Activation from '../../src/methods/activation';

describe('Mutation Methods', () => {
  // Helper to check basic properties of a mutation object
  const checkMutationObject = (method: any, expectedName: string) => {
    test('should exist', () => {
      expect(method).toBeDefined();
    });
    test('should have correct name property', () => {
      expect(method.name).toBe(expectedName);
    });
  };

  describe('ADD_NODE', () => {
    checkMutationObject(mutation.ADD_NODE, 'ADD_NODE');
  });

  describe('SUB_NODE', () => {
    checkMutationObject(mutation.SUB_NODE, 'SUB_NODE');
    test('should have keep_gates property', () => {
      expect(mutation.SUB_NODE.keep_gates).toBe(true);
    });
  });

  describe('ADD_CONN', () => {
    checkMutationObject(mutation.ADD_CONN, 'ADD_CONN');
  });

  describe('SUB_CONN', () => {
    checkMutationObject(mutation.SUB_CONN, 'SUB_CONN');
  });

  describe('MOD_WEIGHT', () => {
    checkMutationObject(mutation.MOD_WEIGHT, 'MOD_WEIGHT');
    test('should have min property', () => {
      expect(mutation.MOD_WEIGHT.min).toBe(-1);
    });
    test('should have max property', () => {
      expect(mutation.MOD_WEIGHT.max).toBe(1);
    });
  });

  describe('MOD_BIAS', () => {
    checkMutationObject(mutation.MOD_BIAS, 'MOD_BIAS');
    test('should have min property', () => {
      expect(mutation.MOD_BIAS.min).toBe(-1);
    });
    test('should have max property', () => {
      expect(mutation.MOD_BIAS.max).toBe(1);
    });
  });

  describe('MOD_ACTIVATION', () => {
    checkMutationObject(mutation.MOD_ACTIVATION, 'MOD_ACTIVATION');
    test('should have mutateOutput property', () => {
      expect(mutation.MOD_ACTIVATION.mutateOutput).toBe(true);
    });
    test('should have allowed property as an array', () => {
      expect(Array.isArray(mutation.MOD_ACTIVATION.allowed)).toBe(true);
    });
    test('should contain expected activation functions in allowed', () => {
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
      expect(mutation.MOD_ACTIVATION.allowed).toEqual(
        expect.arrayContaining(expectedActivations)
      );
      expect(mutation.MOD_ACTIVATION.allowed.length).toBe(
        expectedActivations.length
      );
    });
  });

  describe('ADD_SELF_CONN', () => {
    checkMutationObject(mutation.ADD_SELF_CONN, 'ADD_SELF_CONN');
  });

  describe('SUB_SELF_CONN', () => {
    checkMutationObject(mutation.SUB_SELF_CONN, 'SUB_SELF_CONN');
  });

  describe('ADD_GATE', () => {
    checkMutationObject(mutation.ADD_GATE, 'ADD_GATE');
  });

  describe('SUB_GATE', () => {
    checkMutationObject(mutation.SUB_GATE, 'SUB_GATE');
  });

  describe('ADD_BACK_CONN', () => {
    checkMutationObject(mutation.ADD_BACK_CONN, 'ADD_BACK_CONN');
  });

  describe('SUB_BACK_CONN', () => {
    checkMutationObject(mutation.SUB_BACK_CONN, 'SUB_BACK_CONN');
  });

  describe('SWAP_NODES', () => {
    checkMutationObject(mutation.SWAP_NODES, 'SWAP_NODES');
    test('should have mutateOutput property', () => {
      expect(mutation.SWAP_NODES.mutateOutput).toBe(true);
    });
  });

  describe('ALL', () => {
    test('should exist and be an array', () => {
      expect(mutation.ALL).toBeDefined();
      expect(Array.isArray(mutation.ALL)).toBe(true);
    });

    test('should contain all individual mutation methods', () => {
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
      expect(mutation.ALL).toEqual(expect.arrayContaining(expectedMethods));
      expect(mutation.ALL.length).toBe(expectedMethods.length);
    });
  });

  describe('FFW', () => {
    test('should exist and be an array', () => {
      expect(mutation.FFW).toBeDefined();
      expect(Array.isArray(mutation.FFW)).toBe(true);
    });

    test('should contain only feedforward-compatible mutation methods', () => {
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
      expect(mutation.FFW).toEqual(expect.arrayContaining(expectedMethods));
      expect(mutation.FFW.length).toBe(expectedMethods.length);
    });

    test('should not contain recurrent mutation methods', () => {
      const recurrentMethods = [
        mutation.ADD_GATE,
        mutation.SUB_GATE,
        mutation.ADD_SELF_CONN,
        mutation.SUB_SELF_CONN,
        mutation.ADD_BACK_CONN,
        mutation.SUB_BACK_CONN,
      ];
      recurrentMethods.forEach((method) => {
        expect(mutation.FFW).not.toContain(method);
      });
    });
  });
});
