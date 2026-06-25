import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

interface AgentReport {
  name: string;
  hasSkill: boolean;
  missingSpecialists: string[];
  skills: string[];
  agents: string[];
}

interface GateReport {
  evidence: {
    requiredAgents: string[];
    requiredSkill: string;
    requiredSpecialists: string[];
    agentReports: AgentReport[];
    missingSkillAgents: string[];
    missingSpecialistAgents: { agent: string; specialist: string }[];
  };
  fixHint: string | null;
  owner: string;
  pass: boolean;
}

interface SpawnedResult {
  gateStatus: number | null;
  report: GateReport | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const GATE_URL = pathToFileURL(
  path.resolve(__dirname, 'chrome-devtools-mcp-coverage.gate.mjs'),
).href;
const GATE_PATH = path.resolve(
  __dirname,
  'chrome-devtools-mcp-coverage.gate.mjs',
);

function makeFullAgent(name: string): string {
  return [
    '---',
    `name: '${name}'`,
    'tier: 1',
    'skills: [execute, chrome-devtools-mcp]',
    'agents: [performance-trace-specialist, browser-ui-specialist, browser-memory-specialist]',
    '---',
    '',
    'Agent body.',
  ].join('\n');
}

function makeAgentMissingSkill(name: string): string {
  return [
    '---',
    `name: '${name}'`,
    'tier: 1',
    'skills: [execute]',
    'agents: [performance-trace-specialist, browser-ui-specialist, browser-memory-specialist]',
    '---',
    '',
    'Agent body.',
  ].join('\n');
}

function makeAgentMissingSpecialist(name: string): string {
  return [
    '---',
    `name: '${name}'`,
    'tier: 1',
    'skills: [execute, chrome-devtools-mcp]',
    'agents: [performance-trace-specialist, browser-ui-specialist]',
    '---',
    '',
    'Agent body.',
  ].join('\n');
}

function buildLoaderSource(inventory: Record<string, string>): string {
  return Object.entries(inventory)
    .map(([key, value]) => `    '${key}': ${JSON.stringify(value)},`)
    .join('\n');
}

function runGate(loaderSource: string): SpawnedResult {
  const inlineScript = [
    `import { runChromeDevToolsMcpCoverageGate } from ${JSON.stringify(GATE_URL)};`,
    'const report = await runChromeDevToolsMcpCoverageGate({',
    '  agentLoader: async (name) => ({',
    loaderSource,
    '  })[name],',
    '});',
    'console.log(JSON.stringify(report));',
  ].join('\n');

  const gateResult = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', inlineScript],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 600_000,
    },
  );

  return {
    gateStatus: gateResult.status,
    report: tryParseJson<GateReport>(gateResult.stdout ?? ''),
    stderr: gateResult.stderr ?? '',
    stdout: gateResult.stdout ?? '',
  };
}

function runCliJson(): SpawnedResult {
  const cliResult = spawnSync(process.execPath, [GATE_PATH, '--json'], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 600_000,
  });

  return {
    gateStatus: cliResult.status,
    report: tryParseJson<GateReport>(cliResult.stdout ?? ''),
    stderr: cliResult.stderr ?? '',
    stdout: cliResult.stdout ?? '',
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

describe('chrome-devtools-mcp-coverage.gate.mjs', () => {
  it('passes when both required agents have the skill and all specialists', () => {
    const result = runGate(
      buildLoaderSource({
        '03-red-testing': makeFullAgent('03-red-testing'),
        '05-green-testing': makeFullAgent('05-green-testing'),
      }),
    );

    expect(result.gateStatus).toBe(0);
    expect(result.report).toEqual(
      expect.objectContaining({
        pass: true,
        fixHint: null,
        owner: 'chrome-devtools-mcp-workflow',
        evidence: expect.objectContaining({
          missingSkillAgents: [],
          missingSpecialistAgents: [],
        }),
      }),
    );
  });

  it('fails when a required agent is missing the chrome-devtools-mcp skill', () => {
    const result = runGate(
      buildLoaderSource({
        '03-red-testing': makeAgentMissingSkill('03-red-testing'),
        '05-green-testing': makeFullAgent('05-green-testing'),
      }),
    );

    expect(result.report).toEqual(
      expect.objectContaining({
        pass: false,
        evidence: expect.objectContaining({
          missingSkillAgents: ['03-red-testing'],
        }),
      }),
    );
  });

  it('fails when a required agent is missing a specialist in the agents list', () => {
    const result = runGate(
      buildLoaderSource({
        '03-red-testing': makeFullAgent('03-red-testing'),
        '05-green-testing': makeAgentMissingSpecialist('05-green-testing'),
      }),
    );

    expect(result.report).toEqual(
      expect.objectContaining({
        pass: false,
        evidence: expect.objectContaining({
          missingSpecialistAgents: [
            {
              agent: '05-green-testing',
              specialist: 'browser-memory-specialist',
            },
          ],
        }),
      }),
    );
  });

  it('emits valid JSON contract from CLI --json mode', () => {
    const result = runCliJson();

    expect(result.report).toEqual(
      expect.objectContaining({
        pass: expect.any(Boolean),
        owner: 'chrome-devtools-mcp-workflow',
        evidence: expect.objectContaining({
          requiredAgents: ['03-red-testing', '05-green-testing'],
          requiredSkill: 'chrome-devtools-mcp',
        }),
      }),
    );
  });
});
