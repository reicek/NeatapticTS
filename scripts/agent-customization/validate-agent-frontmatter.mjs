#!/usr/bin/env node
import {
  issue,
  knownAgentTools,
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
    title: 'Validate NeatapticTS custom agent frontmatter.',
    usage: 'node scripts/agent-customization/validate-agent-frontmatter.mjs [--json] [--strict]',
  });
  process.exit(0);
}

const agents = await collectAgents();
const issues = agents.flatMap((agent) => validateAgent(agent, agents, options));
issues.push(...validateGlobalAgentRules(agents, options));

const report = {
  ...summarizeIssues('agent frontmatter', issues),
  agents: agents.map(({ body: _body, raw: _raw, ...agent }) => agent),
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;

async function collectAgents() {
  const paths = await listMarkdownFiles('.github/agents', (relativePath) => relativePath.endsWith('.agent.md'));
  return Promise.all(
    paths.map(async (relativePath) => {
      const text = await readWorkspaceFile(relativePath);
      const parsed = parseFrontmatter(text, relativePath);
      return {
        path: relativePath,
        data: parsed.data,
        raw: parsed.raw,
        body: parsed.body,
        parseIssues: parsed.issues,
        name: parsed.data.name ?? relativePath.split('/').at(-1)?.replace('.agent.md', ''),
      };
    }),
  );
}

function validateAgent(agent, agents, { strict }) {
  const issues = [...agent.parseIssues];
  const { data, path: relativePath } = agent;

  if (!data.description) issues.push(issue('error', relativePath, 'Agent description is required.'));
  if (!data.name) issues.push(issue('warning', relativePath, 'Agent name should be explicit for stable delegation.'));
  if ('user-invocable' in data && typeof data['user-invocable'] !== 'boolean') {
    issues.push(issue('error', relativePath, '`user-invocable` must be a boolean.'));
  }
  if ('disable-model-invocation' in data && typeof data['disable-model-invocation'] !== 'boolean') {
    issues.push(issue('error', relativePath, '`disable-model-invocation` must be a boolean.'));
  }
  if ('tools' in data && !Array.isArray(data.tools)) {
    issues.push(issue('error', relativePath, '`tools` must be an inline array.'));
  }
  if ('agents' in data && !Array.isArray(data.agents)) {
    issues.push(issue('error', relativePath, '`agents` must be an inline array.'));
  }

  for (const tool of Array.isArray(data.tools) ? data.tools : []) {
    if (!knownAgentTools.has(tool) && !tool.includes('/')) {
      issues.push(issue('warning', relativePath, `Unknown tool alias '${tool}'.`));
    }
  }

  const knownNames = new Set(agents.map((knownAgent) => knownAgent.name));
  for (const childAgent of Array.isArray(data.agents) ? data.agents : []) {
    if (!knownNames.has(childAgent)) {
      issues.push(issue('error', relativePath, `Unknown subagent '${childAgent}'.`));
    }
    if (childAgent === agent.name) {
      issues.push(issue('error', relativePath, 'Agent cannot list itself as a subagent.'));
    }
  }

  if (strict && data['user-invocable'] !== false && !/^\.github\/agents\/0[1-7]-/.test(relativePath)) {
    issues.push(issue('error', relativePath, 'Strict mode allows only numbered phase agents to be user-invocable.'));
  }

  if (strict && /^\.github\/agents\/0[1-7]-/.test(relativePath)) {
    if (!data.model) issues.push(issue('error', relativePath, 'Phase agents must declare a model or fallback array.'));
    if (data['user-invocable'] !== true) issues.push(issue('error', relativePath, 'Phase agents must be user-invocable.'));
  }

  if (data.model && !isQualifiedModel(data.model)) {
    issues.push(issue('error', relativePath, 'Model must be a qualified model string or fallback array like GPT-5.4 (copilot).'));
  }

  return issues;
}

function validateGlobalAgentRules(agents, { strict }) {
  if (!strict) return [];

  const visibleAgents = agents.filter((agent) => agent.data['user-invocable'] !== false);
  if (visibleAgents.length !== 7) {
    return [issue('error', '.github/agents', `Strict mode expected exactly 7 user-invocable agents, found ${visibleAgents.length}.`)];
  }

  return [];
}

function isQualifiedModel(model) {
  if (Array.isArray(model)) return model.every(isQualifiedModel);
  return typeof model === 'string' && /^[A-Za-z0-9 ._-]+ \([A-Za-z0-9 ._-]+\)$/.test(model);
}