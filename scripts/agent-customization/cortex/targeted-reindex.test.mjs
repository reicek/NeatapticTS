/**
 * @module targeted-reindex.test
 * @description Jest unit tests for the targeted reindex engine.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly. The spawn calls are mocked so tests never touch
 * the real embedding model or database.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const mockSpawn = jest.fn();

jest.unstable_mockModule('node:child_process', () => ({
  spawn: (...args) => mockSpawn(...args),
}));

const { isEligibleFile, reindexFiles, resetReindexState } =
  await import('./targeted-reindex.mjs');

function makeExitingChild(code, stderr = '') {
  const child = {
    stderr: {
      on(event, cb) {
        if (event === 'data' && stderr) setImmediate(() => cb(stderr));
      },
    },
    on(event, cb) {
      if (event === 'exit') setImmediate(() => cb(code));
    },
  };
  return child;
}

function makeErrorChild(errorMessage) {
  const child = {
    stderr: {
      on() {},
    },
    on(event, cb) {
      if (event === 'error') setImmediate(() => cb(new Error(errorMessage)));
    },
  };
  return child;
}

describe('isEligibleFile', () => {
  it('accepts .md under plans/', () => {
    assert.strictEqual(isEligibleFile('plans/foo.plans.md'), true);
  });

  it('accepts .ts under src/', () => {
    assert.strictEqual(isEligibleFile('src/neat/neat.ts'), true);
  });

  it('rejects .md under .github/skills/ (excluded from index)', () => {
    assert.strictEqual(
      isEligibleFile('.github/skills/execute/SKILL.md'),
      false,
    );
  });

  it('accepts .mjs under scripts/agent-customization/', () => {
    assert.strictEqual(
      isEligibleFile('scripts/agent-customization/cortex/targeted-reindex.mjs'),
      true,
    );
  });

  it('accepts .js under examples/', () => {
    assert.strictEqual(isEligibleFile('examples/flappy/main.js'), true);
  });

  it('rejects .md under .github/agents/ (excluded from index)', () => {
    assert.strictEqual(
      isEligibleFile('.github/agents/01-planning.agent.md'),
      false,
    );
  });

  it('rejects files outside eligible roots', () => {
    assert.strictEqual(isEligibleFile('README.md'), false);
    assert.strictEqual(isEligibleFile('docs/guide.md'), false);
    assert.strictEqual(isEligibleFile('package.json'), false);
  });

  it('rejects ineligible extensions even under eligible roots', () => {
    assert.strictEqual(isEligibleFile('src/image.png'), false);
    assert.strictEqual(isEligibleFile('src/data.csv'), false);
  });

  it('rejects non-string and empty inputs', () => {
    assert.strictEqual(isEligibleFile(''), false);
    assert.strictEqual(isEligibleFile(null), false);
    assert.strictEqual(isEligibleFile(undefined), false);
    assert.strictEqual(isEligibleFile(123), false);
  });
});

describe('reindexFiles', () => {
  beforeEach(() => {
    mockSpawn.mockReset();
    resetReindexState();
  });

  it('returns empty result for an empty input list', async () => {
    const result = await reindexFiles([]);
    assert.deepStrictEqual(result, { reindexed: [], errors: [] });
    assert.strictEqual(mockSpawn.mock.calls.length, 0);
  });

  it('returns empty result for non-array input', async () => {
    const result = await reindexFiles(null);
    assert.deepStrictEqual(result, { reindexed: [], errors: [] });
    assert.strictEqual(mockSpawn.mock.calls.length, 0);
  });

  it('skips ineligible files without spawning', async () => {
    const result = await reindexFiles(['README.md', 'package.json']);
    assert.deepStrictEqual(result, { reindexed: [], errors: [] });
    assert.strictEqual(mockSpawn.mock.calls.length, 0);
  });

  it('reindexes eligible files by spawning build-index then embed-index', async () => {
    mockSpawn.mockReturnValue(makeExitingChild(0));

    const result = await reindexFiles(['src/neat/neat.ts']);
    assert.deepStrictEqual(result.reindexed, ['src/neat/neat.ts']);
    assert.deepStrictEqual(result.errors, []);
    assert.strictEqual(mockSpawn.mock.calls.length, 2);
  });

  it('normalizes absolute paths to repo-relative form', async () => {
    mockSpawn.mockReturnValue(makeExitingChild(0));

    const result = await reindexFiles(['C:\\NeatapticTS\\src\\neat\\neat.ts']);
    assert.deepStrictEqual(result.reindexed, ['src/neat/neat.ts']);
  });

  it('deduplicates and skips blanks', async () => {
    mockSpawn.mockReturnValue(makeExitingChild(0));

    const result = await reindexFiles([
      'src/neat/neat.ts',
      '',
      'src/neat/neat.ts',
    ]);
    assert.deepStrictEqual(result.reindexed, ['src/neat/neat.ts']);
  });

  it('records errors when build-index exits non-zero', async () => {
    mockSpawn.mockReturnValue(makeExitingChild(1, 'build failed'));

    const result = await reindexFiles(['src/neat/neat.ts']);
    assert.deepStrictEqual(result.reindexed, []);
    assert.strictEqual(result.errors.length, 1);
    assert.match(result.errors[0], /build-index failed/);
  });

  it('records errors when embed-index exits non-zero', async () => {
    let call = 0;
    mockSpawn.mockImplementation(() => {
      call += 1;
      return makeExitingChild(
        call === 1 ? 0 : 1,
        call === 2 ? 'embed failed' : '',
      );
    });

    const result = await reindexFiles(['src/neat/neat.ts']);
    assert.deepStrictEqual(result.reindexed, []);
    assert.strictEqual(result.errors.length, 1);
    assert.match(result.errors[0], /embed-index failed/);
  });

  it('never throws even when spawn itself raises', async () => {
    mockSpawn.mockImplementation(() => {
      throw new Error('spawn boom');
    });

    const result = await reindexFiles(['src/neat/neat.ts']);
    assert.deepStrictEqual(result.reindexed, []);
    assert.ok(result.errors.length >= 1);
  });

  it('records errors when the child process emits an error event', async () => {
    mockSpawn.mockReturnValue(makeErrorChild('child boom'));

    const result = await reindexFiles(['src/neat/neat.ts']);
    assert.deepStrictEqual(result.reindexed, []);
    assert.ok(result.errors.length >= 1);
    assert.match(result.errors[0], /child error/);
  });

  it('treats a null exit code as -1 (signal kill)', async () => {
    const child = {
      stderr: { on() {} },
      on(event, cb) {
        if (event === 'exit') setImmediate(() => cb(null));
      },
    };
    mockSpawn.mockReturnValue(child);

    const result = await reindexFiles(['src/neat/neat.ts']);
    assert.deepStrictEqual(result.reindexed, []);
    assert.ok(result.errors.length >= 1);
    assert.match(result.errors[0], /code -1/);
  });
});
