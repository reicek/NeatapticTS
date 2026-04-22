import * as methods from '../../../methods/methods';
import {
  resolveActivationFunction,
  resolveActivationKey,
} from './network.serialize.activation.utils';
import { FALLBACK_ACTIVATION_KEY } from './network.serialize.utils.types';

type SerializeActivationFunction = typeof methods.Activation.identity;

function createNamedCustomActivation(): SerializeActivationFunction {
  function customHydrationActivation(
    inputValue: number,
    shouldComputeDerivative = false,
  ): number {
    return shouldComputeDerivative ? 1 : inputValue;
  }

  return customHydrationActivation as SerializeActivationFunction;
}

function createBlankNameActivation(): SerializeActivationFunction {
  const customActivation = createNamedCustomActivation();

  Object.defineProperty(customActivation, 'name', {
    configurable: true,
    value: '',
  });

  return customActivation;
}

describe('network serialize activation utilities chapter', () => {
  describe('resolveActivationKey', () => {
    describe('given the runtime squash function matches a registered activation reference', () => {
      it('returns the canonical registry key', () => {
        // Arrange
        const squashFunction = methods.Activation.relu;

        // Act
        const activationKey = resolveActivationKey(squashFunction);

        // Assert
        expect(activationKey).toBe('relu');
      });
    });

    describe('given a custom squash function is not registered but exposes a stable function name', () => {
      it('returns the runtime function name', () => {
        // Arrange
        const squashFunction = createNamedCustomActivation();

        // Act
        const activationKey = resolveActivationKey(squashFunction);

        // Assert
        expect(activationKey).toBe('customHydrationActivation');
      });
    });

    describe('given the runtime squash function is missing', () => {
      it('returns the identity fallback key', () => {
        // Arrange
        const squashFunction = undefined as unknown as SerializeActivationFunction;

        // Act
        const activationKey = resolveActivationKey(squashFunction);

        // Assert
        expect(activationKey).toBe(FALLBACK_ACTIVATION_KEY);
      });
    });

    describe('given a custom squash function has an empty runtime name', () => {
      it('returns the identity fallback key', () => {
        // Arrange
        const squashFunction = createBlankNameActivation();

        // Act
        const activationKey = resolveActivationKey(squashFunction);

        // Assert
        expect(activationKey).toBe(FALLBACK_ACTIVATION_KEY);
      });
    });
  });

  describe('resolveActivationFunction', () => {
    describe('given the stored squash name matches a canonical registry key', () => {
      it('returns the registered activation function', () => {
        // Arrange
        const squashName = 'relu';

        // Act
        const squashFunction = resolveActivationFunction(squashName);

        // Assert
        expect(squashFunction).toBe(methods.Activation.relu);
      });
    });

    describe('given the stored squash name matches a runtime function name instead of a registry key', () => {
      it('returns the activation by matching function.name', () => {
        // Arrange
        const squashName = methods.Activation.tanh.name;

        // Act
        const squashFunction = resolveActivationFunction(squashName);

        // Assert
        expect(squashFunction).toBe(methods.Activation.tanh);
      });
    });

    describe('given the stored squash name is missing', () => {
      it('returns the identity fallback activation', () => {
        // Arrange
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => undefined);

        try {
          // Act
          const squashFunction = resolveActivationFunction(undefined);

          // Assert
          expect(squashFunction).toBe(methods.Activation.identity);
        } finally {
          warnSpy.mockRestore();
        }
      });
    });
  });
});