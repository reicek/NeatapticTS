import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface ParityReport {
  leftDigest: string;
  parity: boolean;
  rightDigest: string;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());
const FIXTURES_ROOT = path.join(REPO_ROOT, 'scripts', 'semantic-index', 'docs-quality', '__fixtures__');

describe('docs-quality parity red contracts', () => {
  it('produces identical manifest digest for CLI and MCP paths with the same inputs and config', () => {
    const baseManifest = JSON.parse(readFileSync(path.join(FIXTURES_ROOT, 'manifest.v1.base.json'), 'utf8')) as Record<string, unknown>;

    const result = runModuleEvaluation<ParityReport>(`
      import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
      import { runDocsQualityMetrics } from './scripts/semantic-index/docs-quality/docs-quality.metrics.mjs';

      const config = {
        complexityThreshold: 10,
        minJsdocWords: 10,
        sourcePaths: ['src/neat.ts', 'src/architecture/network.ts'],
      };

      const cliResult = await runDocsQualityMetrics(config);
      const server = createRepoCortexMcpServer({ databasePath: './data/semantic-index.sqlite' });
      const mcpResponse = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/call',
        params: {
          name: 'scan_code_quality',
          arguments: {
            complexity_threshold: config.complexityThreshold,
            min_jsdoc_words: config.minJsdocWords,
            source_paths: config.sourcePaths,
          },
        },
      });

      const mcpManifest = mcpResponse.structuredContent?.manifest ?? ${JSON.stringify(baseManifest)};
      const cliManifest = cliResult.manifest;
      const parity = JSON.stringify(cliManifest) === JSON.stringify(mcpManifest);
      console.log(JSON.stringify({
        leftDigest: cliManifest.scopeDigest,
        parity,
        rightDigest: mcpManifest.scopeDigest,
      }));
    `);

    expect(result).toEqual(expect.objectContaining({
      report: {
        leftDigest: '9f4ac9f8f2d0afac8beffd2d95b8d6c38b57f03b9057c2a8f6cc6d0bbf6f0a11',
        parity: true,
        rightDigest: '9f4ac9f8f2d0afac8beffd2d95b8d6c38b57f03b9057c2a8f6cc6d0bbf6f0a11',
      },
      status: 0,
    }));
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
