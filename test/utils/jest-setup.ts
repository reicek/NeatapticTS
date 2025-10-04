// Console filtering: suppress noisy logs while allowing structured benchmark lines through.
const originalLog = console.log;
const originalWarn = console.warn;
const originalError = console.error;

const ALLOW_ALL = process.env.JEST_ALLOW_ALL_LOGS === '1';
const BENCH_PREFIX = '[BENCH]';
const BENCH_PRETTY = process.env.JEST_BENCH_PRETTY === '1';

let __benchRunSeq = 0;

const humanBytes = (v: unknown): string | unknown => {
  if (typeof v !== 'number' || !isFinite(v)) return v;
  const units = ['B', 'KB', 'MB', 'GB'];
  let u = 0;
  let n = v as number;
  while (n >= 1024 && u < units.length - 1) {
    n /= 1024;
    u++;
  }
  return `${n.toFixed(n < 10 && u > 0 ? 2 : 1)}${units[u]}`;
};

const detectCaller = (): string => {
  try {
    const raw = new Error().stack || '';
    const lines = raw.split(/\n+/).slice(2);
    for (const line of lines) {
      if (line.includes('jest-setup')) continue;
      const m = line.match(/\(([^)]+):(\d+):(\d+)\)/);
      if (m) {
        const file = m[1].split(/[\\/]/).pop();
        return `${file}:${m[2]}`;
      }
    }
  } catch (e) {
    void e;
  }
  return 'unknown';
};

// Global structured benchmark logger (phase 0 formatting)
const _benchLog = (
  tag: string,
  section: string,
  kv: Record<string, unknown>,
) => {
  const caller = detectCaller();
  const seq = ++__benchRunSeq;
  const timestamp = new Date().toISOString();
  // Stable key order for primary metrics first
  const primaryOrder = [
    'mode',
    'variant',
    'size',
    'buildMs',
    'fwdAvgMs',
    'fwdTotalMs',
    'conn',
    'nodes',
    'estBytes',
    'bytesPerConn',
  ];
  const ordered: [string, unknown][] = [];
  const keys = Object.keys(kv);
  for (const k of primaryOrder) if (keys.includes(k)) ordered.push([k, kv[k]]);
  for (const k of keys) if (!primaryOrder.includes(k)) ordered.push([k, kv[k]]);

  // Derive human-friendly extras
  const extras: Record<string, string> = {};
  const estBytes = kv['estBytes'];
  if (typeof estBytes === 'number')
    extras.estBytesHuman = String(humanBytes(estBytes));
  const bytesPerConn = kv['bytesPerConn'];
  if (typeof bytesPerConn === 'number')
    extras.bytesPerConnHuman = String(humanBytes(bytesPerConn)).replace(
      /B$/,
      'B/conn',
    );

  if (!BENCH_PRETTY) {
    const parts = ordered
      .map(([k, v]) => `${k}=${v}`)
      .concat(Object.entries(extras).map(([k, v]) => `${k}=${v}`))
      .join(' ');
    originalLog(
      `${BENCH_PREFIX}[${tag}][${section}] seq=${seq} ts=${timestamp} at=${caller} ${parts}`,
    );
    return;
  }

  // Pretty multi-line formatting
  originalLog(`${BENCH_PREFIX}────────────────────────────────────────`);
  originalLog(`${BENCH_PREFIX} Tag       : ${tag}`);
  originalLog(`${BENCH_PREFIX} Section   : ${section}`);
  originalLog(`${BENCH_PREFIX} Sequence  : ${seq}`);
  originalLog(`${BENCH_PREFIX} Timestamp : ${timestamp}`);
  originalLog(`${BENCH_PREFIX} Caller    : ${caller}`);
  originalLog(`${BENCH_PREFIX} Metrics:`);
  for (const [k, v] of ordered) {
    originalLog(`${BENCH_PREFIX}   • ${k.padEnd(12)} = ${v}`);
  }
  for (const [k, v] of Object.entries(extras)) {
    originalLog(`${BENCH_PREFIX}   • ${k.padEnd(12)} = ${v}`);
  }
  originalLog(`${BENCH_PREFIX}────────────────────────────────────────`);
};

console.log = (...args: unknown[]) => {
  if (ALLOW_ALL) return originalLog(...(args as [unknown]));
  if (
    typeof args[0] === 'string' &&
    (args[0] as string).startsWith(BENCH_PREFIX)
  ) {
    return originalLog(...(args as [unknown])); // pass through benchmark output
  }
  // otherwise swallow (kept as noop for test clarity)
};
console.warn = (...args: unknown[]) => {
  if (ALLOW_ALL) return originalWarn(...(args as [unknown]));
};
console.error = (...args: unknown[]) => {
  if (ALLOW_ALL) return originalError(...(args as [unknown]));
};

// Add custom matchers
expect.extend({
  toBeCloseToArray: (received: unknown, expected: unknown, precision = 5) => {
    const r = received as number[];
    const e = expected as number[];
    if (!Array.isArray(r) || !Array.isArray(e)) {
      return {
        pass: false,
        message: () => `Expected ${received} and ${expected} to be arrays`,
      };
    }

    if (r.length !== e.length) {
      return {
        pass: false,
        message: () =>
          `Expected arrays to have same length but got ${r.length} and ${e.length}`,
      };
    }

    for (let i = 0; i < r.length; i++) {
      const diff = Math.abs(r[i] - e[i]);
      const epsilon = Math.pow(10, -precision) / 2;
      if (diff > epsilon) {
        return {
          pass: false,
          message: () =>
            `Expected ${r[i]} to be close to ${e[i]} (at index ${i})`,
        };
      }
    }

    return {
      pass: true,
      message: () => `Expected arrays not to be close`,
    };
  },
});

// Expose bench logger on globalThis with a precise type
(
  globalThis as unknown as {
    benchLog?: typeof _benchLog;
  }
).benchLog = _benchLog;

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
