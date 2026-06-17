// Polyfill structuredClone for jsdom test environments that do not expose it.
// Use async IIFE for ESM-compatible dynamic import of Node.js v8 module.
(async () => {
  if (typeof globalThis.structuredClone === 'undefined') {
    // Use Node.js v8 serialization as a spec-compliant substitute.
    const v8 = await import('v8');
    globalThis.structuredClone = <T>(obj: T): T =>
      v8.deserialize(v8.serialize(obj)) as T;
  }
})();

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
