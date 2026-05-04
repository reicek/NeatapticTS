import { config } from '../../../config';
import mutation from '../../../methods/mutation/mutation';
import {
  resolveMutationKey,
  warnUnknownMutation,
} from './network.mutate.dispatch.utils';
import { UNKNOWN_MUTATION_WARNING_PREFIX } from './network.mutate.utils.types';

describe('network mutate dispatch utility chapter', () => {
  const originalWarnings = config.warnings ?? false;

  afterEach(() => {
    config.warnings = originalWarnings;
    jest.restoreAllMocks();
  });

  describe('resolveMutationKey', () => {
    describe('given the method is already a mutation-key string', () => {
      it('returns the same key string', () => {
        // Arrange
        const method = 'ADD_NODE';

        // Act
        const resolvedMutationKey = resolveMutationKey(method);

        // Assert
        expect(resolvedMutationKey).toBe('ADD_NODE');
      });
    });

    describe('given the method object provides a name field', () => {
      it('prefers the name field over other direct key fields', () => {
        // Arrange
        const method = {
          identity: 'SUB_CONN',
          name: 'ADD_GATE',
          type: 'MOD_WEIGHT',
        };

        // Act
        const resolvedMutationKey = resolveMutationKey(method);

        // Assert
        expect(resolvedMutationKey).toBe('ADD_GATE');
      });
    });

    describe('given the method object omits name but provides a type field', () => {
      it('returns the type field', () => {
        // Arrange
        const method = {
          identity: 'SUB_CONN',
          type: 'MOD_WEIGHT',
        };

        // Act
        const resolvedMutationKey = resolveMutationKey(method);

        // Assert
        expect(resolvedMutationKey).toBe('MOD_WEIGHT');
      });
    });

    describe('given the method object provides only an identity field', () => {
      it('returns the identity field', () => {
        // Arrange
        const method = {
          identity: 'SUB_GATE',
        };

        // Act
        const resolvedMutationKey = resolveMutationKey(method);

        // Assert
        expect(resolvedMutationKey).toBe('SUB_GATE');
      });
    });

    describe('given the method object has no direct key fields but matches a known mutation reference', () => {
      it('falls back to the identity-reference match', () => {
        // Arrange
        const originalMutationName = mutation.ADD_CONN.name;
        Reflect.deleteProperty(mutation.ADD_CONN, 'name');

        try {
          // Act
          const resolvedMutationKey = resolveMutationKey(mutation.ADD_CONN);

          // Assert
          expect(resolvedMutationKey).toBe('ADD_CONN');
        } finally {
          Reflect.set(mutation.ADD_CONN, 'name', originalMutationName);
        }
      });
    });

    describe('given the method object matches no direct or identity fallback', () => {
      it('returns undefined', () => {
        // Arrange
        const method = {
          max: 0.25,
          min: -0.25,
        };

        // Act
        const resolvedMutationKey = resolveMutationKey(method);

        // Assert
        expect(resolvedMutationKey).toBeUndefined();
      });
    });
  });

  describe('warnUnknownMutation', () => {
    describe('given warning mode is disabled', () => {
      it('does not emit a console warning', () => {
        // Arrange
        config.warnings = false;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);

        // Act
        warnUnknownMutation('UNKNOWN_MUTATION');

        // Assert
        expect(warnSpy.mock.calls.length).toBe(0);
      });
    });

    describe('given warning mode is enabled', () => {
      it('emits one unknown-mutation warning with the unresolved key', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);

        // Act
        warnUnknownMutation('UNKNOWN_MUTATION');

        // Assert
        expect(warnSpy.mock.calls).toEqual([
          [UNKNOWN_MUTATION_WARNING_PREFIX, 'UNKNOWN_MUTATION'],
        ]);
      });
    });
  });
});
