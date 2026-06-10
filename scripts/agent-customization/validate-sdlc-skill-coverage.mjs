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
    title: 'Validate NeatapticTS SDLC skill coverage.',
    usage:
      'node scripts/agent-customization/validate-sdlc-skill-coverage.mjs [--json]',
  });
  process.exit(0);
}

const requiredCapabilities = [
  ['acceptance criteria', ['planning-acceptance-criteria']],
  ['test creation', ['creating-unit-tests', 'red-test-contracts']],
  ['test execution', ['running-unit-tests', 'green-validation-gates']],
  ['test failure triage', ['triaging-test-failures', 'test-fix-workflow']],
  ['docs audit', ['auditing-js-docs', 'docs-academic-citation-audit']],
  ['docs update', ['updating-js-docs', 'educational-docs']],
  [
    'agent frontmatter',
    ['agent-frontmatter-standards', 'updating-agent-frontmatter'],
  ],
  [
    'skill frontmatter',
    ['skill-frontmatter-standards', 'updating-skill-frontmatter'],
  ],
  ['specialist creation', ['creating-specialist-agent']],
  ['agent splitting', ['splitting-monolithic-agent']],
  ['learning events', ['capturing-learning-event']],
  ['session summaries', ['summarizing-session-log']],
  ['customization inventory', ['agent-inventory-audit']],
  ['script tooling', ['agent-script-tooling']],
  ['trigger evals', ['skill-description-evals']],
  ['output evals', ['skill-output-evals']],
  ['subagent delegation', ['subagent-delegation-patterns']],
  ['model routing', ['model-routing-and-budget']],
];

const skills = await collectSkills();
const skillsByName = new Map(skills.map((skill) => [skill.name, skill]));
const issues = [];

for (const [capability, skillNames] of requiredCapabilities) {
  const missing = skillNames.filter(
    (skillName) => !skillsByName.has(skillName),
  );
  if (missing.length > 0) {
    issues.push(
      issue(
        'error',
        '.github/skills',
        `${capability} is missing required skill(s): ${missing.join(', ')}`,
      ),
    );
  }
}

for (const skill of skills) {
  if (!skill.argumentHint) {
    issues.push(
      issue(
        'warning',
        skill.path,
        'Skill has no argument-hint; routing prompts may be underspecified.',
      ),
    );
  }
}

const report = {
  ...summarizeIssues('SDLC skill coverage', issues),
  capabilities: requiredCapabilities.map(([capability, skillNames]) => ({
    capability,
    skills: skillNames,
    present: skillNames.every((skillName) => skillsByName.has(skillName)),
  })),
  summary: {
    requiredCapabilities: requiredCapabilities.length,
    skills: skills.length,
  },
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;

async function collectSkills() {
  const paths = await listMarkdownFiles('.github/skills', (relativePath) =>
    relativePath.endsWith('/SKILL.md'),
  );
  return Promise.all(
    paths.map(async (relativePath) => {
      const { data } = parseFrontmatter(
        await readWorkspaceFile(relativePath),
        relativePath,
      );
      return {
        path: relativePath,
        name: data.name ?? '',
        argumentHint: data['argument-hint'] ?? null,
      };
    }),
  );
}
