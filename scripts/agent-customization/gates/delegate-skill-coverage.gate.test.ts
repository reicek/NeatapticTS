import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

interface AgentReport {
  name: string;
  tier: number;
  hasDelegate: boolean;
  skills: string[];
}

interface GateReport {
  evidence: {
    requiredSkill: string;
    requiredTiers: number[];
    checkedAgentCount: number;
    agentReports: AgentReport[];
    missingDelegateAgents: string[];
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
  path.resolve(__dirname, 'delegate-skill-coverage.gate.mjs'),
).href;
const GATE_PATH = path.resolve(__dirname, 'delegate-skill-coverage.gate.mjs');

function makeAgent(
  name: string,
  tier: number,
  skills: string[],
  agents: string[] = [],
): string {
  return [
    '---',
    `name: '${name}'`,
    `tier: ${tier}`,
    `skills: [${skills.join(', ')}]`,
    `agents: [${agents.join(', ')}]`,
    '---',
    '',
    'Agent body.',
  ].join('\n');
}

function buildInventorySource(
  descriptors: { name: string; contents: string }[],
): string {
  return descriptors
    .map(
      (descriptor) =>
        `    { name: ${JSON.stringify(descriptor.name)}, relativePath: ${JSON.stringify(`.github/agents/${descriptor.name}.agent.md`)}, contents: ${JSON.stringify(descriptor.contents)} },`,
    )
    .join('\n');
}

function runGate(inventorySource: string): SpawnedResult {
  const inlineScript = [
    `import { runDelegateSkillCoverageGate } from ${JSON.stringify(GATE_URL)};`,
    'const report = await runDelegateSkillCoverageGate({',
    '  inventoryLoader: async () => [',
    inventorySource,
    '  ],',
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

describe('delegate-skill-coverage.gate.mjs', () => {
  it('passes when all Tier 1 and Tier 2 agents have the execute skill', () => {
    const result = runGate(
      buildInventorySource([
        {
          name: '01-planning',
          contents: makeAgent('01-planning', 1, ['execute', 'plan-alignment']),
        },
        {
          name: 'some-specialist-tier2',
          contents: makeAgent('some-specialist-tier2', 2, ['execute']),
        },
        {
          name: 'tier3-specialist',
          contents: makeAgent('tier3-specialist', 3, ['unrelated-skill']),
        },
      ]),
    );

    expect(result.gateStatus).toBe(0);
    expect(result.report).toEqual(
      expect.objectContaining({
        pass: true,
        fixHint: null,
        owner: 'delegate-skill-workflow',
        evidence: expect.objectContaining({
          checkedAgentCount: 2,
          missingDelegateAgents: [],
        }),
      }),
    );
  });

  it('fails when one Tier 1 agent is missing the execute skill', () => {
    const result = runGate(
      buildInventorySource([
        {
          name: '01-planning',
          contents: makeAgent('01-planning', 1, ['plan-alignment']),
        },
        {
          name: 'some-specialist-tier2',
          contents: makeAgent('some-specialist-tier2', 2, ['execute']),
        },
      ]),
    );

    expect(result.report).toEqual(
      expect.objectContaining({
        pass: false,
        evidence: expect.objectContaining({
          missingDelegateAgents: ['01-planning'],
        }),
      }),
    );
  });

  it('emits valid JSON contract from CLI --json mode', () => {
    const result = runCliJson();

    expect(result.report).toEqual(
      expect.objectContaining({
        pass: expect.any(Boolean),
        owner: 'delegate-skill-workflow',
        evidence: expect.objectContaining({
          requiredSkill: 'execute',
          requiredTiers: [1, 2],
        }),
      }),
    );
  });
});
