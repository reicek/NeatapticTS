#!/usr/bin/env node
import {
  issue,
  listMarkdownFiles,
  parseArgs,
  parseFrontmatter,
  printUsage,
  readWorkspaceFile,
  summarizeIssues,
  writeReport,
} from './customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Validate NeatapticTS custom agent delegation graph.',
    usage: 'node scripts/agent-customization/validate-agent-graph.mjs [--json]',
  });
  process.exit(0);
}

const agents = await collectAgents();
const byName = new Map(agents.map((agent) => [agent.name, agent]));
const issues = [];

for (const agent of agents) {
  for (const childName of agent.children) {
    if (!byName.has(childName)) {
      issues.push(issue('error', agent.path, `Unknown subagent '${childName}'.`));
    }
  }
}

for (const cycle of findCycles(agents, byName)) {
  issues.push(issue('error', '.github/agents', `Delegation cycle detected: ${cycle.join(' -> ')}`));
}

const report = {
  ...summarizeIssues('agent graph', issues),
  graph: agents.map((agent) => ({ name: agent.name, path: agent.path, agents: agent.children })),
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;

async function collectAgents() {
  const paths = await listMarkdownFiles('.github/agents', (relativePath) => relativePath.endsWith('.agent.md'));
  return Promise.all(
    paths.map(async (relativePath) => {
      const { data } = parseFrontmatter(await readWorkspaceFile(relativePath), relativePath);
      return {
        path: relativePath,
        name: data.name ?? relativePath.split('/').at(-1)?.replace('.agent.md', ''),
        children: Array.isArray(data.agents) ? data.agents : [],
      };
    }),
  );
}

function findCycles(agents, byName) {
  const cycles = [];
  const visiting = new Set();
  const visited = new Set();

  for (const agent of agents) {
    visit(agent.name, []);
  }

  return cycles;

  function visit(name, stack) {
    if (visiting.has(name)) {
      const cycleStart = stack.indexOf(name);
      cycles.push([...stack.slice(cycleStart), name]);
      return;
    }
    if (visited.has(name)) return;

    const agent = byName.get(name);
    if (!agent) return;

    visiting.add(name);
    for (const child of agent.children) visit(child, [...stack, name]);
    visiting.delete(name);
    visited.add(name);
  }
}