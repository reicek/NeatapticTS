import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

interface CortexFirstSearchGateReport {
  evidence: Record<string, unknown>;
  fixHint: string | null;
  owner: string;
  pass: boolean;
}

interface SpawnedGateResult {
  gateStatus: number | null;
  report: CortexFirstSearchGateReport | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const CORTEX_FIRST_SEARCH_GATE_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
  'cortex-first-search.gate.mjs',
);

describe('cortex-first-search.gate.mjs', () => {
  it('returns the honest prerequisite contract for Cortex-first search readiness', () => {
    const result = runGateContractCheck();

    expect(result).toEqual(
      expect.objectContaining({
        gateStatus: 0,
        report: expect.objectContaining({
          pass: true,
          evidence: expect.objectContaining({
            index_documents: 12,
            index_chunks: 48,
            index_fresh: true,
            corpus_mcp_alive: true,
            corpus_search_results: 3,
          }),
          fixHint: null,
          owner: 'repo-cortex-workflow',
        }),
      }),
    );
  });
});

function runGateContractCheck(): SpawnedGateResult {
  const gateUrl = pathToFileURL(CORTEX_FIRST_SEARCH_GATE_PATH).href;
  const inlineScript = [
    `import { runCortexFirstSearchGate } from ${JSON.stringify(gateUrl)};`,
    'const report = await runCortexFirstSearchGate({',
    "  databasePath: 'rag-index/data/turso-replica.sqlite',",
    '  indexValidator: async () => ({ pass: true, documents: 12, chunks: 48, fixHint: null }),',
    '  mcpSmoke: async () => ({ pass: true, evidence: { searchResults: 3 }, fixHint: null }),',
    '});',
    'console.log(JSON.stringify(report));',
  ].join('\n');
  const gateResult = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', inlineScript],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 600000,
    },
  );

  return {
    gateStatus: gateResult.status,
    report: tryParseJson<CortexFirstSearchGateReport>(gateResult.stdout ?? ''),
    stderr: gateResult.stderr ?? '',
    stdout: gateResult.stdout ?? '',
  };
}

function tryParseJson<ReportType>(stdout: string) {
  if (!stdout.trim()) return null;

  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}
