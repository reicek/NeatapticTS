// Polyfill structuredClone for jsdom test environments that do not expose it.
if (typeof globalThis.structuredClone === 'undefined') {
  // Use Node.js v8 serialization as a spec-compliant substitute.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { serialize, deserialize } = require('v8') as typeof import('v8');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (globalThis as any).structuredClone = <T>(obj: T): T =>
    deserialize(serialize(obj)) as T;
}

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
