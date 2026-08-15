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
const strictTier1StructuredFields = [
  'OUTPUT_CONTRACT',
  'TASK_STATUS',
  'TIER',
  'ROLE',
  'TASK_RECEIVED',
  'FILES_READ',
  'FILES_CHANGED',
  'KEY_FINDINGS',
  'ACTIONS_TAKEN',
  'VALIDATION_EVIDENCE',
  'BLOCKERS',
  'RISKS_OR_GAPS',
  'LEARNING_EVENT_NEEDED',
  'SUGGESTED_NEXT_AGENT',
  'PHASE_COMPLETE',
  'SUB_ORCHESTRATORS_USED',
  'SUMMARY',
];
const strictTier2CoordinatorPathsByName = new Map([
  [
    'flappy-architecture-polish',
    '.github/agents/flappy-architecture-polish.agent.md',
  ],
  [
    'green-test-failure-triage-coordinator',
    '.github/agents/green-test-failure-triage-coordinator.agent.md',
  ],
  [
    'helping-agent-maintenance-coordinator',
    '.github/agents/helping-agent-maintenance-coordinator.agent.md',
  ],
  [
    'helping-gap-resolution-coordinator',
    '.github/agents/helping-gap-resolution-coordinator.agent.md',
  ],
  [
    'implementation-pattern-coordinator',
    '.github/agents/implementation-pattern-coordinator.agent.md',
  ],
  [
    'planning-context-coordinator',
    '.github/agents/planning-context-coordinator.agent.md',
  ],
  [
    'planning-risk-coordinator',
    '.github/agents/planning-risk-coordinator.agent.md',
  ],
  ['solid-split', '.github/agents/solid-split.agent.md'],
]);
const strictTier2CoordinatorPaths = new Set(
  strictTier2CoordinatorPathsByName.values(),
);
const strictTier2StructuredFields = [
  'OUTPUT_CONTRACT',
  'TASK_STATUS',
  'TIER',
  'ROLE',
  'TASK_RECEIVED',
  'FILES_READ',
  'FILES_CHANGED',
  'KEY_FINDINGS',
  'ACTIONS_TAKEN',
  'VALIDATION_EVIDENCE',
  'SPECIALISTS_USED',
  'HANDOFF',
  'BLOCKERS',
  'RISKS_OR_GAPS',
  'LEARNING_EVENT_NEEDED',
  'SUGGESTED_NEXT_AGENT',
  'SUMMARY',
];
const strictAllowedModels = new Set(['glm-5.2:cloud', 'kimi-k2.7-code:cloud']);

if (options.help) {
  printUsage({
    title: 'Validate NeatapticTS custom agent frontmatter.',
    usage:
      'node scripts/agent-customization/validate-agent-frontmatter.mjs [--json] [--strict]',
  });
  process.exit(0);
}

const agents = await collectAgents();
const skillNames = await collectSkillNames();
const issues = agents.flatMap((agent) =>
  validateAgent(agent, agents, skillNames, options),
);
issues.push(...validateGlobalAgentRules(agents, options));

const report = {
  ...summarizeIssues('agent frontmatter', issues),
  agents: agents.map(({ body: _body, raw: _raw, ...agent }) => agent),
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;

async function collectAgents() {
  const paths = await listMarkdownFiles('.github/agents', (relativePath) =>
    relativePath.endsWith('.agent.md'),
  );
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
        name:
          parsed.data.name ??
          relativePath.split('/').at(-1)?.replace('.agent.md', ''),
      };
    }),
  );
}

async function collectSkillNames() {
  const paths = await listMarkdownFiles('.github/skills', (relativePath) =>
    relativePath.endsWith('/SKILL.md'),
  );
  const skills = await Promise.all(
    paths.map(async (relativePath) => {
      const text = await readWorkspaceFile(relativePath);
      const parsed = parseFrontmatter(text, relativePath);
      return parsed.data.name ?? relativePath.split('/').at(-2) ?? '';
    }),
  );

  return new Set(skills.filter(Boolean));
}

function validateAgent(agent, agents, skillNames, { strict }) {
  const issues = [...agent.parseIssues];
  const { data, path: relativePath } = agent;

  if (!data.description)
    issues.push(issue('error', relativePath, 'Agent description is required.'));
  if (!data.name)
    issues.push(
      issue(
        'warning',
        relativePath,
        'Agent name should be explicit for stable delegation.',
      ),
    );
  if ('user-invocable' in data && typeof data['user-invocable'] !== 'boolean') {
    issues.push(
      issue('error', relativePath, '`user-invocable` must be a boolean.'),
    );
  }
  if (
    'disable-model-invocation' in data &&
    typeof data['disable-model-invocation'] !== 'boolean'
  ) {
    issues.push(
      issue(
        'error',
        relativePath,
        '`disable-model-invocation` must be a boolean.',
      ),
    );
  }
  if ('tools' in data && !Array.isArray(data.tools)) {
    issues.push(
      issue('error', relativePath, '`tools` must be an inline array.'),
    );
  }
  if ('agents' in data && !Array.isArray(data.agents)) {
    issues.push(
      issue('error', relativePath, '`agents` must be an inline array.'),
    );
  }
  if (!('skills' in data)) {
    issues.push(
      issue(
        'error',
        relativePath,
        '`skills` is required and must be an inline array.',
      ),
    );
  }
  if ('skills' in data && !Array.isArray(data.skills)) {
    issues.push(
      issue('error', relativePath, '`skills` must be an inline array.'),
    );
  }

  if (
    Array.isArray(data.agents) &&
    data.agents.length > 0 &&
    !(Array.isArray(data.tools) && data.tools.includes('agent'))
  ) {
    issues.push(
      issue(
        'error',
        relativePath,
        'Agents that list subagents must include the `agent` tool.',
      ),
    );
  }

  for (const tool of Array.isArray(data.tools) ? data.tools : []) {
    if (!knownAgentTools.has(tool) && !tool.includes('/')) {
      issues.push(
        issue('warning', relativePath, `Unknown tool alias '${tool}'.`),
      );
    }
  }

  const knownNames = new Set(agents.map((knownAgent) => knownAgent.name));
  for (const childAgent of Array.isArray(data.agents) ? data.agents : []) {
    if (!knownNames.has(childAgent)) {
      issues.push(
        issue('error', relativePath, `Unknown subagent '${childAgent}'.`),
      );
    }
    if (childAgent === agent.name) {
      issues.push(
        issue('error', relativePath, 'Agent cannot list itself as a subagent.'),
      );
    }
  }

  for (const skillName of Array.isArray(data.skills) ? data.skills : []) {
    if (!skillNames.has(skillName)) {
      issues.push(
        issue('error', relativePath, `Unknown skill '${skillName}'.`),
      );
    }
  }

  if (
    strict &&
    data['user-invocable'] !== false &&
    !strictVisibleAgentPathPattern.test(relativePath)
  ) {
    issues.push(
      issue(
        'error',
        relativePath,
        'Strict mode allows only numbered SDLC orchestrators to be user-invocable.',
      ),
    );
  }

  if (strict && strictVisibleAgentPathPattern.test(relativePath)) {
    if (!data.model)
      issues.push(
        issue(
          'error',
          relativePath,
          'SDLC orchestrators must declare a model.',
        ),
      );
    if (data['user-invocable'] !== true)
      issues.push(
        issue(
          'error',
          relativePath,
          'SDLC orchestrators must be user-invocable.',
        ),
      );
    if (!strictVisibleAgentNames.has(data.name)) {
      issues.push(
        issue(
          'error',
          relativePath,
          `Strict mode expected one of the public SDLC agent names, found '${data.name ?? 'NONE'}'.`,
        ),
      );
    }
    const expectedPath = strictVisibleAgentPathsByName.get(data.name);
    if (expectedPath && relativePath !== expectedPath) {
      issues.push(
        issue(
          'error',
          relativePath,
          `Strict mode expected '${data.name}' to live at '${expectedPath}'.`,
        ),
      );
    }
  }

  const structuredPromptContract = resolveStructuredPromptContract(agent);
  if (strict && structuredPromptContract) {
    issues.push(
      ...validateStructuredV1PromptContract(agent, structuredPromptContract),
    );
  }

  if (
    strict &&
    data['user-invocable'] === false &&
    !structuredPromptContract &&
    !hasOutputContract(agent.body)
  ) {
    issues.push(
      issue(
        'error',
        relativePath,
        'Hidden agents must define a compact output contract.',
      ),
    );
  }

  if (data.model && !isQualifiedModel(data.model)) {
    issues.push(
      issue(
        'error',
        relativePath,
        'Model must be a qualified model string like glm-5.2:cloud.',
      ),
    );
  }
  if (strict && data.model && !usesOnlyAllowedModels(data.model)) {
    issues.push(
      issue(
        'error',
        relativePath,
        'Strict mode allows only the configured SDLC model pool.',
      ),
    );
  }

  return issues;
}

function validateGlobalAgentRules(agents, { strict }) {
  if (!strict) return [];

  const visibleAgents = agents.filter(
    (agent) => agent.data['user-invocable'] !== false,
  );
  const issues = [];

  if (visibleAgents.length !== strictVisibleAgentNames.size) {
    issues.push(
      issue(
        'error',
        '.github/agents',
        `Strict mode expected exactly ${strictVisibleAgentNames.size} user-invocable agents, found ${visibleAgents.length}.`,
      ),
    );
  }

  const visibleNames = new Set(visibleAgents.map((agent) => agent.name));
  for (const expectedName of strictVisibleAgentNames) {
    if (!visibleNames.has(expectedName))
      issues.push(
        issue(
          'error',
          '.github/agents',
          `Missing user-invocable SDLC agent '${expectedName}'.`,
        ),
      );
  }

  for (const visibleAgent of visibleAgents) {
    if (!strictVisibleAgentNames.has(visibleAgent.name)) {
      issues.push(
        issue(
          'error',
          visibleAgent.path,
          `Unexpected user-invocable agent '${visibleAgent.name}'.`,
        ),
      );
    }
  }

  return issues;
}

function isQualifiedModel(model) {
  return (
    typeof model === 'string' &&
    (/^[A-Za-z0-9 .:_-]+ \([A-Za-z0-9 ._-]+\)$/.test(model) ||
      strictAllowedModels.has(model))
  );
}

function usesOnlyAllowedModels(model) {
  return strictAllowedModels.has(model);
}

function hasOutputContract(body) {
  return /(^|\n)(## Output [Ff]ormat|Return:|Return only:)/.test(body);
}

function resolveStructuredPromptContract(agent) {
  if (strictVisibleAgentPathPattern.test(agent.path)) {
    return {
      tier: '1',
      requiredFields: strictTier1StructuredFields,
      expectedRole: agent.name,
    };
  }

  if (strictTier2CoordinatorPaths.has(agent.path)) {
    return {
      tier: '2',
      requiredFields: strictTier2StructuredFields,
      expectedRole: agent.name,
    };
  }

  return null;
}

function validateStructuredV1PromptContract(agent, contract) {
  const issues = [];
  const structuredFenceMatches = [
    ...agent.body.matchAll(/```structured-v1\r?\n(?<body>[\s\S]*?)\r?\n```/gu),
  ];

  if (structuredFenceMatches.length !== 1) {
    issues.push(
      issue(
        'error',
        agent.path,
        'Strict mode requires exactly one fenced ```structured-v1``` prompt template.',
      ),
    );
    return issues;
  }

  const fenceBody = structuredFenceMatches[0].groups?.body ?? '';
  const parsedFields = parsePromptFields(fenceBody);
  const detectedFields = parsedFields.map(({ field }) => field);

  if (
    detectedFields.length !== contract.requiredFields.length ||
    detectedFields.some(
      (field, index) => field !== contract.requiredFields[index],
    )
  ) {
    issues.push(
      issue(
        'error',
        agent.path,
        `Structured-v1 prompt fields must match the exact ${contract.tier === '1' ? 'Tier-1' : 'Tier-2'} order: ${contract.requiredFields.join(', ')}.`,
      ),
    );
  }

  for (const requiredField of contract.requiredFields) {
    if (!detectedFields.includes(requiredField)) {
      issues.push(
        issue(
          'error',
          agent.path,
          `Structured-v1 prompt template is missing required field '${requiredField}'.`,
        ),
      );
    }
  }

  const outputContractField = parsedFields.find(
    ({ field }) => field === 'OUTPUT_CONTRACT',
  );
  if (outputContractField?.value !== 'structured-v1') {
    issues.push(
      issue(
        'error',
        agent.path,
        "Structured-v1 prompt template must set 'OUTPUT_CONTRACT: structured-v1'.",
      ),
    );
  }

  const tierField = parsedFields.find(({ field }) => field === 'TIER');
  if (tierField?.value !== contract.tier) {
    issues.push(
      issue(
        'error',
        agent.path,
        `Structured-v1 prompt template must set 'TIER: ${contract.tier}'.`,
      ),
    );
  }

  const roleField = parsedFields.find(({ field }) => field === 'ROLE');
  if (roleField?.value !== contract.expectedRole) {
    issues.push(
      issue(
        'error',
        agent.path,
        `Structured-v1 prompt template must set 'ROLE: ${contract.expectedRole}'.`,
      ),
    );
  }

  return issues;
}

function parsePromptFields(fenceBody) {
  const parsedFields = [];

  for (const rawLine of fenceBody.split(/\r?\n/u)) {
    const trimmedLine = rawLine.trim();
    if (!trimmedLine || trimmedLine.startsWith('- ')) {
      continue;
    }

    const fieldMatch = /^(?<field>[A-Z_]+):\s*(?<value>.*)$/u.exec(trimmedLine);
    if (!fieldMatch?.groups) {
      continue;
    }

    parsedFields.push({
      field: fieldMatch.groups.field,
      value: fieldMatch.groups.value.trim(),
    });
  }

  return parsedFields;
}
