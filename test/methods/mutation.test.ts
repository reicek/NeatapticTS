import { mutation } from '../../src/methods/mutation';
import Activation from '../../src/methods/activation';

describe('Mutation Methods', () => {
  describe('ADD_NODE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.ADD_NODE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_NODE.name).toBe('ADD_NODE');
      });
    });
  });

  describe('SUB_NODE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.SUB_NODE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_NODE.name).toBe('SUB_NODE');
      });
    });
    describe('Scenario: keep_gates property', () => {
      it('should have keep_gates property', () => {
        // Assert
        expect(mutation.SUB_NODE.keep_gates).toBe(true);
      });
    });
  });

  describe('ADD_CONN', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.ADD_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_CONN.name).toBe('ADD_CONN');
      });
    });
  });

  describe('SUB_CONN', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.SUB_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_CONN.name).toBe('SUB_CONN');
      });
    });
  });

  describe('MOD_WEIGHT', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.MOD_WEIGHT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.MOD_WEIGHT.name).toBe('MOD_WEIGHT');
      });
    });
    describe('Scenario: min property', () => {
      it('should have min property', () => {
        // Assert
        expect(mutation.MOD_WEIGHT.min).toBe(-1);
      });
    });
    describe('Scenario: max property', () => {
      it('should have max property', () => {
        // Assert
        expect(mutation.MOD_WEIGHT.max).toBe(1);
      });
    });
  });

  describe('MOD_BIAS', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.MOD_BIAS).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.MOD_BIAS.name).toBe('MOD_BIAS');
      });
    });
    describe('Scenario: min property', () => {
      it('should have min property', () => {
        // Assert
        expect(mutation.MOD_BIAS.min).toBe(-1);
      });
    });
    describe('Scenario: max property', () => {
      it('should have max property', () => {
        // Assert
        expect(mutation.MOD_BIAS.max).toBe(1);
      });
    });
  });

  describe('MOD_ACTIVATION', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.MOD_ACTIVATION).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.MOD_ACTIVATION.name).toBe('MOD_ACTIVATION');
      });
    });
    describe('Scenario: mutateOutput property', () => {
      it('should have mutateOutput property', () => {
        // Assert
        expect(mutation.MOD_ACTIVATION.mutateOutput).toBe(true);
      });
    });
    describe('Scenario: allowed property', () => {
      it('should have allowed property as an array', () => {
        // Assert
        expect(Array.isArray(mutation.MOD_ACTIVATION.allowed)).toBe(true);
      });
      describe('Scenario: Contains expected activation functions', () => {
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
        expectedActivations.forEach((act) => {
          it(`should contain ${act.name} in allowed`, () => {
            // Assert
            expect(mutation.MOD_ACTIVATION.allowed).toContain(act);
          });
        });
        it('should have allowed property with correct length', () => {
          // Assert
          expect(mutation.MOD_ACTIVATION.allowed.length).toBe(expectedActivations.length);
        });
        it('should not contain unexpected activation functions', () => {
          // Arrange
          const notExpected: any[] = [];
          // Assert
          notExpected.forEach((act) => {
            expect(mutation.MOD_ACTIVATION.allowed).not.toContain(act);
          });
        });
      });
    });
  });

  describe('ADD_SELF_CONN', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.ADD_SELF_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_SELF_CONN.name).toBe('ADD_SELF_CONN');
      });
    });
  });

  describe('SUB_SELF_CONN', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.SUB_SELF_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_SELF_CONN.name).toBe('SUB_SELF_CONN');
      });
    });
  });

  describe('ADD_GATE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.ADD_GATE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_GATE.name).toBe('ADD_GATE');
      });
    });
  });

  describe('SUB_GATE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.SUB_GATE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_GATE.name).toBe('SUB_GATE');
      });
    });
  });

  describe('ADD_BACK_CONN', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.ADD_BACK_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.ADD_BACK_CONN.name).toBe('ADD_BACK_CONN');
      });
    });
  });

  describe('SUB_BACK_CONN', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.SUB_BACK_CONN).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.SUB_BACK_CONN.name).toBe('SUB_BACK_CONN');
      });
    });
  });

  describe('SWAP_NODES', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(mutation.SWAP_NODES).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(mutation.SWAP_NODES.name).toBe('SWAP_NODES');
      });
    });
    describe('Scenario: mutateOutput property', () => {
      it('should have mutateOutput property', () => {
        // Assert
        expect(mutation.SWAP_NODES.mutateOutput).toBe(true);
      });
    });
  });

  describe('ADD_LSTM_NODE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        expect(mutation.ADD_LSTM_NODE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        expect(mutation.ADD_LSTM_NODE.name).toBe('ADD_LSTM_NODE');
      });
    });
  });

  describe('ADD_GRU_NODE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        expect(mutation.ADD_GRU_NODE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        expect(mutation.ADD_GRU_NODE.name).toBe('ADD_GRU_NODE');
      });
    });
  });

  describe('ALL', () => {
    describe('Scenario: Existence', () => {
      describe('when mutation.ALL is defined', () => {
        it('should be defined', () => {
          // Assert
          expect(mutation.ALL).toBeDefined();
        });
      });
      describe('when mutation.ALL is an array', () => {
        it('should be an array', () => {
          // Assert
          expect(Array.isArray(mutation.ALL)).toBe(true);
        });
      });
    });
    describe('Scenario: Contains all mutation methods', () => {
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
        mutation.REINIT_WEIGHT,
        mutation.BATCH_NORM,
        mutation.ADD_LSTM_NODE,
        mutation.ADD_GRU_NODE,
      ];
      expectedMethods.forEach((method) => {
        describe(`when ALL contains ${method.name}`, () => {
          it(`should contain ${method.name}`, () => {
            // Assert
            expect(mutation.ALL).toContain(method);
          });
        });
      });
      it('should have correct length', () => {
        // Assert
        expect(mutation.ALL.length).toBe(18);
      });
      it('should not contain unexpected mutation methods', () => {
        // Arrange
        const notExpected: any[] = [];
        // Assert
        notExpected.forEach((method) => {
          expect(mutation.ALL).not.toContain(method);
        });
      });
    });
  });

  describe('FFW', () => {
    describe('Scenario: Existence', () => {
      describe('when mutation.FFW is defined', () => {
        it('should be defined', () => {
          // Assert
          expect(mutation.FFW).toBeDefined();
        });
      });
      describe('when mutation.FFW is an array', () => {
        it('should be an array', () => {
          // Assert
          expect(Array.isArray(mutation.FFW)).toBe(true);
        });
      });
    });
    describe('Scenario: Contains only feedforward-compatible mutation methods', () => {
      const expectedMethods = [
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
      expectedMethods.forEach((method) => {
        describe(`when FFW contains ${method.name}`, () => {
          it(`should contain ${method.name}`, () => {
            // Assert
            expect(mutation.FFW).toContain(method);
          });
        });
      });
      it('should have correct length', () => {
        // Assert
        expect(mutation.FFW.length).toBe(10);
      });
      describe('Scenario: Does not contain recurrent mutation methods', () => {
        const recurrentMethods = [
          mutation.ADD_GATE,
          mutation.SUB_GATE,
          mutation.ADD_SELF_CONN,
          mutation.SUB_SELF_CONN,
          mutation.ADD_BACK_CONN,
          mutation.SUB_BACK_CONN,
        ];
        recurrentMethods.forEach((method) => {
          describe(`when FFW does not contain ${method.name}`, () => {
            it(`should not contain ${method.name}`, () => {
              // Assert
              expect(mutation.FFW).not.toContain(method);
            });
          });
        });
      });
    });
  });
});
