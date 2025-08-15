// Mock console methods to suppress unwanted output during tests
const originalLog = console.log;
const originalWarn = console.warn;
const originalError = console.error;

// Override console methods
console.log = jest.fn();
console.warn = jest.fn();
console.error = jest.fn();

// Add custom matchers
expect.extend({
  toBeCloseToArray(received: any[], expected: any[], precision = 5) {
    if (!Array.isArray(received) || !Array.isArray(expected)) {
      return {
        pass: false,
        message: () => `Expected ${received} and ${expected} to be arrays`,
      };
    }

    if (received.length !== expected.length) {
      return {
        pass: false,
        message: () =>
          `Expected arrays to have same length but got ${received.length} and ${expected.length}`,
      };
    }

    for (let i = 0; i < received.length; i++) {
      const diff = Math.abs(received[i] - expected[i]);
      const epsilon = Math.pow(10, -precision) / 2;
      if (diff > epsilon) {
        return {
          pass: false,
          message: () =>
            `Expected ${received[i]} to be close to ${expected[i]} (at index ${i})`,
        };
      }
    }

    return {
      pass: true,
      message: () => `Expected arrays not to be close`,
    };
  },
});

// Add this line to prevent "Your test suite must contain at least one test." error
describe('Setup', () => {
  it('Jest setup file loaded correctly', () => {
    expect(true).toBe(true);
  });
});

// Restore original console methods after tests
afterAll(() => {
  console.log = originalLog;
  console.warn = originalWarn;
  console.error = originalError;
});
