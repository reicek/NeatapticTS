// Suppress noisy console output during tests unless explicitly requested.
const originalLog = console.log;
const originalWarn = console.warn;
const originalError = console.error;

const ALLOW_ALL = process.env.JEST_ALLOW_ALL_LOGS === '1';

console.log = (...args: unknown[]) => {
  if (ALLOW_ALL) return originalLog(...(args as [unknown]));
};
console.warn = (...args: unknown[]) => {
  if (ALLOW_ALL) return originalWarn(...(args as [unknown]));
};
console.error = (...args: unknown[]) => {
  if (ALLOW_ALL) return originalError(...(args as [unknown]));
};

afterAll(() => {
  console.log = originalLog;
  console.warn = originalWarn;
  console.error = originalError;
});
