import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface CodeQualityContractReport {
  evidence: Array<{ issue: string; symbol: string }>;
  pass: boolean;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('code-quality-scanner.mjs', () => {
  describe('red JSDoc quality contract', () => {
    it('returns a failing report for an exported function with missing JSDoc', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'semantic-code-quality-red-'));
      const fixtureSourceDirectory = path.join(fixtureDirectory, 'src');
      const fixtureSourcePath = path.join(fixtureSourceDirectory, 'mutation.ts');
      await mkdir(fixtureSourceDirectory, { recursive: true });
      await writeFile(fixtureSourcePath, [
        'export function undocumentedMutation(rate: number) {',
        '  return rate * 2;',
        '}',
        '',
        '/**',
        ' * Returns a documented selection rate fixture with enough words to pass the JSDoc threshold.',
        ' * @param rate - Raw selection rate from the fixture.',
        ' * @returns The unchanged selection rate for comparison.',
        ' */',
        'export function documentedSelection(rate: number) {',
        '  return rate;',
        '}',
      ].join('\n'), 'utf8');

      // Act
      const result = runModuleEvaluation<CodeQualityContractReport>(`
        import { scanCodeQuality } from './scripts/semantic-index/code-quality-scanner.mjs';

        const report = await scanCodeQuality({
          complexityThreshold: 10,
          minJsdocWords: 10,
          sourcePaths: [${JSON.stringify(fixtureSourcePath)}],
        });
        console.log(JSON.stringify({
          evidence: report.evidence.map(({ issue, symbol }) => ({ issue, symbol })),
          pass: report.pass,
        }));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true });

      // Assert
      expect(result).toEqual(expect.objectContaining({
        report: {
          evidence: [{ issue: 'missing JSDoc', symbol: 'undocumentedMutation' }],
          pass: false,
        },
        status: 0,
      }));
    });
  });
});

function runModuleEvaluation<ReportType>(source: string): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(process.execPath, ['--input-type=module', '--eval', source], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });

  return {
    report: tryParseJson<ReportType>(spawned.stdout ?? ''),
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout: spawned.stdout ?? '',
  };
}

function tryParseJson<ReportType>(stdout: string): ReportType | null {
  if (!stdout.trim()) return null;

  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}