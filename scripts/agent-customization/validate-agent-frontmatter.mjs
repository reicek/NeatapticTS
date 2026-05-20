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

const strictVisibleAgentPathsByName = new Map([
  ['00-helping', '.github/agents/00-helping.agent.md'],
  ['01-planning', '.github/agents/01-planning.agent.md'],
  ['02-researching', '.github/agents/02-researching.agent.md'],
  ['03-red-testing', '.github/agents/03-red-testing.agent.md'],
  ['04-implementing', '.github/agents/04-implementing.agent.md'],
  ['05-green-testing', '.github/agents/05-green-testing.agent.md'],
  ['06-documenting', '.github/agents/06-documenting.agent.md'],
  ['07-logging', '.github/agents/07-logging.agent.md'],
]);
const strictVisibleAgentNames = new Set(strictVisibleAgentPathsByName.keys());
const strictVisibleAgentPathPattern = /^\.github\/agents\/0[0-7]-/;
const strictAllowedModels = new Set([
  'GPT-5.4 (copilot)',
  'GPT-5.4-mini (copilot)',
  'Claude Sonnet 4.6 (copilot)',
  'Claude Haiku 4.6 (copilot)',
]);

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

  if (Array.isArray(data.agents) && data.agents.length > 0 && !data.tools?.includes('agent')) {
    issues.push(issue('error', relativePath, 'Agents that list subagents must include the `agent` tool.'));
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

  if (strict && data['user-invocable'] !== false && !strictVisibleAgentPathPattern.test(relativePath)) {
    issues.push(issue('error', relativePath, 'Strict mode allows only numbered SDLC orchestrators to be user-invocable.'));
  }

  if (strict && strictVisibleAgentPathPattern.test(relativePath)) {
    if (!data.model) issues.push(issue('error', relativePath, 'SDLC orchestrators must declare a model or fallback array.'));
    if (data['user-invocable'] !== true) issues.push(issue('error', relativePath, 'SDLC orchestrators must be user-invocable.'));
    if (!strictVisibleAgentNames.has(data.name)) {
      issues.push(issue('error', relativePath, `Strict mode expected one of the public SDLC agent names, found '${data.name ?? 'NONE'}'.`));
    }
    const expectedPath = strictVisibleAgentPathsByName.get(data.name);
    if (expectedPath && relativePath !== expectedPath) {
      issues.push(issue('error', relativePath, `Strict mode expected '${data.name}' to live at '${expectedPath}'.`));
    }
  }

  if (strict && data['user-invocable'] === false && !hasOutputContract(agent.body)) {
    issues.push(issue('error', relativePath, 'Hidden agents must define a compact output contract.'));
  }

  if (data.model && !isQualifiedModel(data.model)) {
    issues.push(issue('error', relativePath, 'Model must be a qualified model string or fallback array like GPT-5.4 (copilot).'));
  }
  if (strict && data.model && !usesOnlyAllowedModels(data.model)) {
    issues.push(issue('error', relativePath, 'Strict mode allows only the configured SDLC model pool.'));
  }

  return issues;
}

function validateGlobalAgentRules(agents, { strict }) {
  if (!strict) return [];

  const visibleAgents = agents.filter((agent) => agent.data['user-invocable'] !== false);
  const issues = [];

  if (visibleAgents.length !== strictVisibleAgentNames.size) {
    issues.push(issue('error', '.github/agents', `Strict mode expected exactly ${strictVisibleAgentNames.size} user-invocable agents, found ${visibleAgents.length}.`));
  }

  const visibleNames = new Set(visibleAgents.map((agent) => agent.name));
  for (const expectedName of strictVisibleAgentNames) {
    if (!visibleNames.has(expectedName)) issues.push(issue('error', '.github/agents', `Missing user-invocable SDLC agent '${expectedName}'.`));
  }

  for (const visibleAgent of visibleAgents) {
    if (!strictVisibleAgentNames.has(visibleAgent.name)) {
      issues.push(issue('error', visibleAgent.path, `Unexpected user-invocable agent '${visibleAgent.name}'.`));
    }
  }

  return issues;
}

function isQualifiedModel(model) {
  if (Array.isArray(model)) return model.every(isQualifiedModel);
  return typeof model === 'string' && /^[A-Za-z0-9 ._-]+ \([A-Za-z0-9 ._-]+\)$/.test(model);
}

function usesOnlyAllowedModels(model) {
  if (Array.isArray(model)) return model.every(usesOnlyAllowedModels);
  return strictAllowedModels.has(model);
}

function hasOutputContract(body) {
  return /(^|\n)(## Output Format|Return:|Return only:)/.test(body);
}