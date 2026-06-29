import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface TypeScriptChunkContractReport {
  chunks: Array<{
    doc_family: string;
    jsdoc_text: string;
    symbol_name: string;
  }>;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('ts-chunker.mjs', () => {
  describe('red ts-source chunk contract', () => {
    it('emits one ts-source chunk per exported function with symbol name and JSDoc text', async () => {
      // Arrange
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-ts-chunker-red-'),
      );
      const fixtureSourceDirectory = path.join(fixtureDirectory, 'src');
      const fixtureSourcePath = path.join(fixtureSourceDirectory, 'fitness.ts');
      await mkdir(fixtureSourceDirectory, { recursive: true });
      await writeFile(
        fixtureSourcePath,
        [
          '/**',
          ' * Adds an evolved fitness score bonus while preserving deterministic ranking inputs.',
          ' * @param currentScore - Current raw score before novelty pressure is applied.',
          ' * @param bonus - Stable deterministic bonus from the evaluator.',
          ' * @returns The adjusted score used by the ranking pass.',
          ' */',
          'export function addFitnessScore(currentScore: number, bonus: number) {',
          '  return currentScore + bonus;',
          '}',
          '',
          '/**',
          ' * Normalizes novelty into a bounded value for hybrid retrieval fixtures.',
          ' * @param novelty - Raw novelty distance for the candidate.',
          ' * @returns A value clamped into the zero-to-one interval.',
          ' */',
          'export function normalizeNovelty(novelty: number) {',
          '  return Math.max(0, Math.min(1, novelty));',
          '}',
        ].join('\n'),
        'utf8',
      );

      // Act
      const result = runModuleEvaluation<TypeScriptChunkContractReport>(`
        import { chunkTypeScriptSources } from './rag-index/ts-chunker.mjs';

        const chunks = await chunkTypeScriptSources({ sourcePaths: [${JSON.stringify(fixtureSourcePath)}] });
        console.log(JSON.stringify({
          chunks: chunks.map(({ doc_family, jsdoc_text, symbol_name }) => ({
            doc_family,
            jsdoc_text,
            symbol_name,
          })),
        }));
      `);
      await rm(fixtureDirectory, { recursive: true, force: true });

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: {
            chunks: [
              {
                doc_family: 'ts-source',
                jsdoc_text:
                  'Adds an evolved fitness score bonus while preserving deterministic ranking inputs.',
                symbol_name: 'addFitnessScore',
              },
              {
                doc_family: 'ts-source',
                jsdoc_text:
                  'Normalizes novelty into a bounded value for hybrid retrieval fixtures.',
                symbol_name: 'normalizeNovelty',
              },
            ],
          },
          status: 0,
        }),
      );
    });
  });
});

function runModuleEvaluation<ReportType>(
  source: string,
): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );

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
