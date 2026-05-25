import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface DocsQualityMetricsGateReport {
  evidence: Record<string, unknown>;
  fixHint: string | null;
  gateType?: string;
  owner: string;
  pass: boolean;
  probeScope?: string;
  repoWideDebtCommand?: string;
}

interface SpawnedGateResult {
  gateStatus: number | null;
  report: DocsQualityMetricsGateReport | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const DOCS_QUALITY_GATE_PATH = path.join(REPO_ROOT, 'scripts', 'agent-customization', 'gates', 'docs-quality-metrics.gate.mjs');

describe('docs-quality-metrics.gate.mjs', () => {
  describe('red gate contract', () => {
    it('reports additive mechanism-only metadata instead of implying repo-wide debt status', () => {
      const result = runGateContractCheck();

      expect(result).toEqual(expect.objectContaining({
        gateStatus: 0,
        report: expect.objectContaining({
          pass: true,
          gateType: 'mechanism-only',
          probeScope: expect.stringMatching(/docs-quality/i),
          repoWideDebtCommand: 'npm run docs:quality:metrics',
          owner: '05-green-testing',
        }),
      }));
    });
  });
});

function runGateContractCheck(): SpawnedGateResult {
  const gateResult = spawnSync(process.execPath, [DOCS_QUALITY_GATE_PATH, '--json'], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 600000,
  });

  return {
    gateStatus: gateResult.status,
    report: tryParseJson<DocsQualityMetricsGateReport>(gateResult.stdout ?? ''),
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