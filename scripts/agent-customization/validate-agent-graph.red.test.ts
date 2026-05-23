import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

interface AgentGraphIssue {
  message: string;
  path: string;
  severity: string;
}

interface AgentGraphReport {
  graph: Array<{ agents: string[]; name: string; path: string }>;
  issues: AgentGraphIssue[];
  ok: boolean;
}

interface TierEnforcementGateReport {
  evidence: Record<string, unknown>;
  fixHint: string;
  owner: string;
  pass: boolean;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

interface AgentFixtureOptions {
  agents?: string[];
  name: string;
  tier: number;
  userInvocable: boolean;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const VALIDATE_AGENT_GRAPH_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'validate-agent-graph.mjs',
);
const TIER_ENFORCEMENT_GATE_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
  'tier-enforcement-gate.mjs',
);

describe('tier enforcement red contracts', () => {
  describe('validate-agent-graph.mjs', () => {
    it('reports a violation when a non-Tier-1 agent is user-invocable', async () => {
      const report = await runValidateAgentGraph({
        '.github/agents/coverage-scout.agent.md': createAgentFrontmatter({
          name: 'Coverage Scout',
          tier: 3,
          userInvocable: true,
        }),
      });

      expect(report).toEqual(expect.objectContaining({
        ok: false,
        issues: expect.arrayContaining([
          expect.objectContaining({
            path: '.github/agents/coverage-scout.agent.md',
            message: expect.stringContaining('user-invocable'),
          }),
        ]),
      }));
    });

    it('does not report a user-invocable violation for a Tier-1 orchestrator', async () => {
      const report = await runValidateAgentGraph({
        '.github/agents/01-planning.agent.md': createAgentFrontmatter({
          name: '01-planning',
          tier: 1,
          userInvocable: true,
        }),
      });

      expect(report).toEqual(expect.objectContaining({
        ok: true,
        issues: [],
      }));
    });

    it('reports a violation for an upward or lateral delegation edge', async () => {
      const report = await runValidateAgentGraph({
        '.github/agents/boundary-mapper.agent.md': createAgentFrontmatter({
          name: 'Boundary Mapper',
          tier: 3,
          userInvocable: false,
          agents: ['planning-context-coordinator'],
        }),
        '.github/agents/planning-context-coordinator.agent.md': createAgentFrontmatter({
          name: 'planning-context-coordinator',
          tier: 2,
          userInvocable: false,
        }),
      });

      expect(report).toEqual(expect.objectContaining({
        ok: false,
        issues: expect.arrayContaining([
          expect.objectContaining({
            path: '.github/agents/boundary-mapper.agent.md',
            message: expect.stringContaining('tier'),
          }),
        ]),
      }));
    });
  });

  describe('tier-enforcement-gate.mjs', () => {
    it('returns a structured failing gate report when tier violations exist', async () => {
      const gateResult = await runTierEnforcementGate({
        '.github/agents/coverage-scout.agent.md': createAgentFrontmatter({
          name: 'Coverage Scout',
          tier: 3,
          userInvocable: true,
        }),
      });

      expect(gateResult).toEqual(expect.objectContaining({
        status: 1,
        report: expect.objectContaining({
          pass: false,
          evidence: expect.any(Object),
        }),
      }));
    });
  });
});

async function runValidateAgentGraph(agentFiles: Record<string, string>) {
  const { report } = await runJsonScript<AgentGraphReport>(VALIDATE_AGENT_GRAPH_PATH, agentFiles);
  return report as AgentGraphReport;
}

async function runTierEnforcementGate(agentFiles: Record<string, string>) {
  return runJsonScript<TierEnforcementGateReport>(TIER_ENFORCEMENT_GATE_PATH, agentFiles);
}

async function runJsonScript<ReportType>(scriptPath: string, workspaceFiles: Record<string, string>) {
  const workspacePath = await createFixtureWorkspace(workspaceFiles);

  try {
    const spawned = spawnSync(process.execPath, [scriptPath, '--json'], {
      cwd: workspacePath,
      encoding: 'utf8',
    });

    return {
      report: tryParseJson<ReportType>(spawned.stdout),
      status: spawned.status,
      stderr: spawned.stderr ?? '',
      stdout: spawned.stdout ?? '',
    } satisfies SpawnedJsonResult<ReportType>;
  } finally {
    await rm(workspacePath, { recursive: true, force: true });
  }
}

async function createFixtureWorkspace(workspaceFiles: Record<string, string>) {
  const workspacePath = await mkdtemp(path.join(tmpdir(), 'agent-graph-red-'));

  for (const [relativePath, fileContents] of Object.entries(workspaceFiles)) {
    const absolutePath = path.join(workspacePath, relativePath);
    await mkdir(path.dirname(absolutePath), { recursive: true });
    await writeFile(absolutePath, fileContents, 'utf8');
  }

  return workspacePath;
}

function createAgentFrontmatter({ agents = [], name, tier, userInvocable }: AgentFixtureOptions) {
  const serializedAgents = agents.length === 0
    ? '[]'
    : `[${agents.map((agentName) => `'${agentName}'`).join(', ')}]`;

  return [
    '---',
    `name: '${name}'`,
    `tier: ${tier}`,
    `user-invocable: ${userInvocable ? 'true' : 'false'}`,
    `agents: ${serializedAgents}`,
    '---',
    '',
    '# Test fixture',
  ].join('\n');
}

function tryParseJson<ReportType>(stdout: string) {
  if (!stdout.trim()) return null;

  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}