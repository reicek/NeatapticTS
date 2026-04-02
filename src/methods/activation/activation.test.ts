import Activation, { registerCustomActivation } from './activation';

describe('Activation', () => {
  describe('logistic()', () => {
    describe('given a non-zero input', () => {
      describe('when the activation is evaluated', () => {
        it('returns the expected logistic value', () => {
          // Arrange
          const inputValue = 1;
          const expectedValue = 1 / (1 + Math.exp(-inputValue));

          // Act
          const actualValue = Activation.logistic(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });

      describe('when the derivative is evaluated', () => {
        it('returns the expected logistic derivative', () => {
          // Arrange
          const inputValue = 1;
          const logisticValue = 1 / (1 + Math.exp(-inputValue));
          const expectedValue = logisticValue * (1 - logisticValue);

          // Act
          const actualValue = Activation.logistic(inputValue, true);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('sigmoid()', () => {
    describe('given a shared input with logistic()', () => {
      describe('when both aliases are evaluated', () => {
        it('matches the logistic activation output', () => {
          // Arrange
          const inputValue = -0.5;
          const expectedValue = Activation.logistic(inputValue);

          // Act
          const actualValue = Activation.sigmoid(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('tanh()', () => {
    describe('given a non-zero input', () => {
      describe('when the derivative is evaluated', () => {
        it('returns one minus tanh squared', () => {
          // Arrange
          const inputValue = 0.5;
          const expectedValue = 1 - Math.tanh(inputValue) ** 2;

          // Act
          const actualValue = Activation.tanh(inputValue, true);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('identity()', () => {
    describe('given an arbitrary input', () => {
      describe('when the activation is evaluated', () => {
        it('returns the input unchanged', () => {
          // Arrange
          const inputValue = -3;

          // Act
          const actualValue = Activation.identity(inputValue);

          // Assert
          expect(actualValue).toBe(inputValue);
        });
      });
    });
  });

  describe('step()', () => {
    describe('given a positive input', () => {
      describe('when the activation is evaluated', () => {
        it('returns one', () => {
          // Arrange
          const inputValue = 2;

          // Act
          const actualValue = Activation.step(inputValue);

          // Assert
          expect(actualValue).toBe(1);
        });
      });
    });

    describe('given a non-positive input', () => {
      describe('when the activation is evaluated', () => {
        it('returns zero', () => {
          // Arrange
          const inputValue = 0;

          // Act
          const actualValue = Activation.step(inputValue);

          // Assert
          expect(actualValue).toBe(0);
        });
      });
    });
  });

  describe('relu()', () => {
    describe('given a negative input', () => {
      describe('when the activation is evaluated', () => {
        it('clamps the output to zero', () => {
          // Arrange
          const inputValue = -2;

          // Act
          const actualValue = Activation.relu(inputValue);

          // Assert
          expect(actualValue).toBe(0);
        });
      });
    });

    describe('given a positive input', () => {
      describe('when the derivative is evaluated', () => {
        it('returns one', () => {
          // Arrange
          const inputValue = 2;

          // Act
          const actualValue = Activation.relu(inputValue, true);

          // Assert
          expect(actualValue).toBe(1);
        });
      });
    });
  });

  describe('softsign()', () => {
    describe('given a non-zero input', () => {
      describe('when the activation is evaluated', () => {
        it('returns the expected softsign value', () => {
          // Arrange
          const inputValue = -2;
          const expectedValue = inputValue / (1 + Math.abs(inputValue));

          // Act
          const actualValue = Activation.softsign(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('sinusoid()', () => {
    describe('given a non-zero input', () => {
      describe('when the derivative is evaluated', () => {
        it('returns the cosine of the input', () => {
          // Arrange
          const inputValue = 0.5;
          const expectedValue = Math.cos(inputValue);

          // Act
          const actualValue = Activation.sinusoid(inputValue, true);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('gaussian()', () => {
    describe('given a non-zero input', () => {
      describe('when the derivative is evaluated', () => {
        it('returns minus two x times the gaussian value', () => {
          // Arrange
          const inputValue = 1;
          const expectedValue = -2 * inputValue * Math.exp(-(inputValue ** 2));

          // Act
          const actualValue = Activation.gaussian(inputValue, true);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('bentIdentity()', () => {
    describe('given a non-zero input', () => {
      describe('when the activation is evaluated', () => {
        it('returns the expected bent identity value', () => {
          // Arrange
          const inputValue = 0.5;
          const expectedValue =
            (Math.sqrt(inputValue ** 2 + 1) - 1) / 2 + inputValue;

          // Act
          const actualValue = Activation.bentIdentity(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('bipolar()', () => {
    describe('given a positive input', () => {
      describe('when the activation is evaluated', () => {
        it('returns one', () => {
          // Arrange
          const inputValue = 3;

          // Act
          const actualValue = Activation.bipolar(inputValue);

          // Assert
          expect(actualValue).toBe(1);
        });
      });
    });

    describe('given a non-positive input', () => {
      describe('when the activation is evaluated', () => {
        it('returns minus one', () => {
          // Arrange
          const inputValue = 0;

          // Act
          const actualValue = Activation.bipolar(inputValue);

          // Assert
          expect(actualValue).toBe(-1);
        });
      });
    });
  });

  describe('bipolarSigmoid()', () => {
    describe('given a non-zero input', () => {
      describe('when the activation is evaluated', () => {
        it('returns the expected bipolar-sigmoid value', () => {
          // Arrange
          const inputValue = 1;
          const expectedValue = 2 / (1 + Math.exp(-inputValue)) - 1;

          // Act
          const actualValue = Activation.bipolarSigmoid(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('hardTanh()', () => {
    describe('given an input below the lower clamp', () => {
      describe('when the activation is evaluated', () => {
        it('returns minus one', () => {
          // Arrange
          const inputValue = -4;

          // Act
          const actualValue = Activation.hardTanh(inputValue);

          // Assert
          expect(actualValue).toBe(-1);
        });
      });
    });

    describe('given an input inside the linear region', () => {
      describe('when the derivative is evaluated', () => {
        it('returns one', () => {
          // Arrange
          const inputValue = 0.25;

          // Act
          const actualValue = Activation.hardTanh(inputValue, true);

          // Assert
          expect(actualValue).toBe(1);
        });
      });
    });
  });

  describe('absolute()', () => {
    describe('given a negative input', () => {
      describe('when the derivative is evaluated', () => {
        it('returns minus one', () => {
          // Arrange
          const inputValue = -2;

          // Act
          const actualValue = Activation.absolute(inputValue, true);

          // Assert
          expect(actualValue).toBe(-1);
        });
      });
    });
  });

  describe('inverse()', () => {
    describe('given an arbitrary input', () => {
      describe('when the activation is evaluated', () => {
        it('returns one minus the input', () => {
          // Arrange
          const inputValue = 0.25;
          const expectedValue = 1 - inputValue;

          // Act
          const actualValue = Activation.inverse(inputValue);

          // Assert
          expect(actualValue).toBe(expectedValue);
        });
      });
    });
  });

  describe('selu()', () => {
    describe('given a negative input', () => {
      describe('when the activation is evaluated', () => {
        it('uses the scaled exponential negative branch', () => {
          // Arrange
          const inputValue = -1;
          const alpha = 1.6732632423543772;
          const scale = 1.0507009873554805;
          const expectedValue = scale * (alpha * Math.exp(inputValue) - alpha);

          // Act
          const actualValue = Activation.selu(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('softplus()', () => {
    describe('given a very large positive input', () => {
      describe('when the activation is evaluated', () => {
        it('uses the stable linear approximation', () => {
          // Arrange
          const inputValue = 40;

          // Act
          const actualValue = Activation.softplus(inputValue);

          // Assert
          expect(actualValue).toBe(inputValue);
        });
      });
    });
  });

  describe('swish()', () => {
    describe('given a non-zero input', () => {
      describe('when the activation is evaluated', () => {
        it('returns x multiplied by logistic(x)', () => {
          // Arrange
          const inputValue = 1;
          const expectedValue = inputValue * (1 / (1 + Math.exp(-inputValue)));

          // Act
          const actualValue = Activation.swish(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('gelu()', () => {
    describe('given a non-zero input', () => {
      describe('when the activation is evaluated', () => {
        it('returns the tanh-based gelu approximation', () => {
          // Arrange
          const inputValue = 1;
          const tanhArgument =
            Math.sqrt(2 / Math.PI) * (inputValue + 0.044715 * inputValue ** 3);
          const expectedValue =
            inputValue * 0.5 * (1 + Math.tanh(tanhArgument));

          // Act
          const actualValue = Activation.gelu(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('mish()', () => {
    describe('given a non-zero input', () => {
      describe('when the activation is evaluated', () => {
        it('returns x multiplied by tanh(softplus(x))', () => {
          // Arrange
          const inputValue = 1;
          const softplusValue =
            Math.max(0, inputValue) +
            Math.log(1 + Math.exp(-Math.abs(inputValue)));
          const expectedValue = inputValue * Math.tanh(softplusValue);

          // Act
          const actualValue = Activation.mish(inputValue);

          // Assert
          expect(actualValue).toBeCloseTo(expectedValue, 12);
        });
      });
    });
  });

  describe('registerCustomActivation()', () => {
    describe('given a custom activation name and implementation', () => {
      describe('when the activation is registered', () => {
        it('adds the new function to the shared registry', () => {
          // Arrange
          const activationName = 'copilotCube';
          registerCustomActivation(
            activationName,
            (inputValue, shouldDerivative) =>
              shouldDerivative ? 3 * inputValue ** 2 : inputValue ** 3,
          );

          // Act
          const actualValue = Activation[activationName](2);

          // Assert
          expect(actualValue).toBe(8);
        });
      });
    });
  });
});
