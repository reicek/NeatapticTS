import { execFileSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '..', '..');

const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(process.execPath, ['--input-type=module', '--eval', source], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });

  return JSON.parse(output) as Result;
};

describe('validate-index.mjs', () => {
  describe('red fixHint contract', () => {
    it('includes a stale-index fixHint when freshness proofs do not match', () => {
      const result = runModuleEvaluation<Record<string, unknown>>(`
        import { validateSemanticIndex } from './scripts/semantic-index/validate-index.mjs';
        const result = await validateSemanticIndex({
          documents: [
            { file_path: 'README.md', mtime_ms: 1, file_size: 1, sha256: 'stale-proof' },
          ],
          freshnessChecks: [
            { file_path: 'README.md', mtime_ms: 2, size: 1, sha256: 'fresh-proof' },
          ],
        });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(expect.objectContaining({
        pass: false,
        fixHint: 'Stale paths detected. Run: node scripts/semantic-index/build-index.mjs',
      }));
    });
  });
});