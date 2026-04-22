import { safeStructuredClone } from './safeStructuredClone';

function withStructuredClone<T>(
  structuredCloneImplementation: typeof globalThis.structuredClone | undefined,
  run: () => T,
): T {
  const originalDescriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    'structuredClone',
  );

  Object.defineProperty(globalThis, 'structuredClone', {
    configurable: true,
    value: structuredCloneImplementation,
    writable: true,
  });

  try {
    return run();
  } finally {
    if (originalDescriptor) {
      Object.defineProperty(globalThis, 'structuredClone', originalDescriptor);
    } else {
      Reflect.deleteProperty(globalThis, 'structuredClone');
    }
  }
}

describe('utils chapter', () => {
  describe('safeStructuredClone', () => {
    describe('given native structuredClone is available', () => {
      describe('when cloning nested plain data', () => {
        it('returns a detached deep clone from the native path', () => {
          // Arrange
          const sourceValue = { nested: { count: 1 } };

          // Act
          const clonedValue = withStructuredClone(
            (value) => JSON.parse(JSON.stringify(value)),
            () => {
              const clonedPayload = safeStructuredClone(sourceValue);

              sourceValue.nested.count = 2;

              return clonedPayload;
            },
          );

          // Assert
          expect(clonedValue).toEqual({ nested: { count: 1 } });
        });
      });
    });

    describe('given native structuredClone is unavailable', () => {
      describe('when cloning nested plain data', () => {
        it('falls back to the JSON clone path', () => {
          // Arrange
          const sourceValue = { nested: { count: 3 } };

          // Act
          const clonedValue = withStructuredClone(undefined, () => {
            const clonedPayload = safeStructuredClone(sourceValue);

            sourceValue.nested.count = 4;

            return clonedPayload;
          });

          // Assert
          expect(clonedValue).toEqual({ nested: { count: 3 } });
        });
      });
    });

    describe('given native structuredClone throws', () => {
      describe('when cloning nested plain data', () => {
        it('retries with the JSON clone fallback', () => {
          // Arrange
          const sourceValue = { nested: { count: 5 } };

          // Act
          const clonedValue = withStructuredClone(
            () => {
              throw new Error('structuredClone failed');
            },
            () => {
              const clonedPayload = safeStructuredClone(sourceValue);

              sourceValue.nested.count = 6;

              return clonedPayload;
            },
          );

          // Assert
          expect(clonedValue).toEqual({ nested: { count: 5 } });
        });
      });
    });
  });
});