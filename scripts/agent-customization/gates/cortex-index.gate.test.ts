import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface CortexIndexGateReport {
  evidence: Record<string, unknown>;
  fixHint: string | null;
  owner: string;
  pass: boolean;
}

interface SpawnedGateResult {
  gateStatus: number | null;
  report: CortexIndexGateReport | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const CORTEX_INDEX_GATE_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
  'cortex-index.gate.mjs',
);

describe('cortex-index.gate.mjs', () => {
  describe('red gate contract', () => {
    it('returns the helping-owned passing contract when all sub-checks are green', () => {
      const result = runGateContractCheck();

      expect(result).toEqual(
        expect.objectContaining({
          gateStatus: 0,
          report: expect.objectContaining({
            pass: true,
            evidence: expect.objectContaining({
              index_documents: expect.any(Number),
              index_fresh: true,
              corpus_mcp_alive: true,
              workflow_mcp_alive: true,
            }),
            fixHint: null,
            owner: '00-helping',
          }),
        }),
      );
    });
  });
});

function runGateContractCheck(): SpawnedGateResult {
  const gateResult = spawnSync(
    process.execPath,
    [
      CORTEX_INDEX_GATE_PATH,
      '--json',
      '--plan=plans/mcp-active-binding.plans.md',
    ],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 600000,
    },
  );

  return {
    gateStatus: gateResult.status,
    report: tryParseJson<CortexIndexGateReport>(gateResult.stdout ?? ''),
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
