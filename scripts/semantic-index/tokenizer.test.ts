import { execFileSync } from 'node:child_process';
import path from 'node:path';

// Tests run from the repository root, so cwd is a stable anchor for repo-relative
// paths without needing import.meta.url (which currently breaks ts-jest for the
// semantic-index-scripts project).
const REPO_ROOT = path.resolve();

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 * Used to import `.mjs` implementation modules that do not yet exist in the
 * red phase and to run them against real fixtures.
 *
 * @param source - ESM source string executed as `--input-type=module --eval`.
 * @returns The JSON-parsed value the snippet wrote to stdout.
 */
const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );
  const trimmed = output.trim();
  if (trimmed.length === 0) {
    throw new Error('Module evaluation produced empty output');
  }
  return JSON.parse(trimmed) as Result;
};

describe('tokenizer behavior (via cortex-db.mjs sanitizeFtsQuery)', () => {
  describe('code identifier tokenization', () => {
    it('preserves dotted identifiers as a searchable term', () => {
      const result = runModuleEvaluation<string>(`
        import { sanitizeFtsQuery } from './scripts/mcp-semantic/tools/cortex-db.mjs';
        console.log(JSON.stringify(sanitizeFtsQuery('network.activate')));
      `);
      expect(result).toMatch(/(?:^|\s)"?network\.activate"?\*?(?:$|\s)/);
    });

    it('preserves file-extension identifiers as a searchable term', () => {
      const result = runModuleEvaluation<string>(`
        import { sanitizeFtsQuery } from './scripts/mcp-semantic/tools/cortex-db.mjs';
        console.log(JSON.stringify(sanitizeFtsQuery('network.ts')));
      `);
      expect(result).toMatch(/(?:^|\s)"?network\.ts"?\*?(?:$|\s)/);
    });
  });

  describe('query wildcard discipline', () => {
    it('does not add wildcard prefix to every token of a mixed identifier phrase', () => {
      const result = runModuleEvaluation<string>(`
        import { sanitizeFtsQuery } from './scripts/mcp-semantic/tools/cortex-db.mjs';
        console.log(JSON.stringify(sanitizeFtsQuery('NEAT selection code')));
      `);
      const wildcardedTokenCount = (result.match(/\w+\*/g) ?? []).length;
      const totalTokenCount = result.trim().split(/\s+/).length;
      expect(wildcardedTokenCount).toBeLessThan(totalTokenCount);
    });
  });
});
