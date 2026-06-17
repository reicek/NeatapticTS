import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface EmbedReadinessReport {
  cached?: boolean;
  latency_ms?: number;
  reason?: string;
  ready: boolean;
  state: 'cold' | 'model-only' | 'warm';
}

interface EmbedReadinessSequenceReport {
  firstLatency?: number;
  firstState: string;
  secondCached: boolean;
  secondLatency?: number;
  secondState: string;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('embed-readiness.mjs', () => {
  describe('red cold-start latency contract', () => {
    it('measures and caches the first probe so the second call is faster', () => {
      const result = runModuleEvaluation<EmbedReadinessSequenceReport>(`
        import { mkdtemp } from 'node:fs/promises';
        import { tmpdir } from 'node:os';
        import path from 'node:path';
        import { getEmbedReadiness } from './scripts/semantic-index/embed-readiness.mjs';

        const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'embed-readiness-latency-'));
        const options = {
          corpusDatabasePath: path.join(fixtureDirectory, 'semantic-index.sqlite'),
          embeddingsDatabasePath: path.join(fixtureDirectory, 'embeddings.sqlite'),
          modelDirectory: path.join(fixtureDirectory, 'models'),
        };

        const first = await getEmbedReadiness(options);
        const second = await getEmbedReadiness(options);

        console.log(JSON.stringify({
          firstLatency: first.latency_ms,
          firstState: first.state,
          secondCached: second.cached === true,
          secondLatency: second.latency_ms,
          secondState: second.state,
        }));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          report: {
            firstLatency: expect.any(Number),
            firstState: 'cold',
            secondCached: true,
            secondLatency: expect.any(Number),
            secondState: 'cold',
          },
          status: 0,
        }),
      );
    });

    it('reports a finite latency_ms for every readiness probe', () => {
      const result = runModuleEvaluation<EmbedReadinessReport>(`
        import { mkdtemp } from 'node:fs/promises';
        import { tmpdir } from 'node:os';
        import path from 'node:path';
        import { getEmbedReadiness } from './scripts/semantic-index/embed-readiness.mjs';

        const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'embed-readiness-latency-bound-'));
        const report = await getEmbedReadiness({
          corpusDatabasePath: path.join(fixtureDirectory, 'semantic-index.sqlite'),
          embeddingsDatabasePath: path.join(fixtureDirectory, 'embeddings.sqlite'),
          modelDirectory: path.join(fixtureDirectory, 'models'),
        });

        console.log(JSON.stringify({
          latency_ms: report.latency_ms,
          ready: report.ready,
          state: report.state,
        }));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          report: {
            latency_ms: expect.any(Number),
            ready: false,
            state: 'cold',
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
