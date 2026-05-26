#!/usr/bin/env node
import path from 'node:path';
import {
  extractMarkdownLinks,
  fileExists,
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
    title: 'Validate NeatapticTS Agent Skill frontmatter and local resources.',
    usage: 'node scripts/agent-customization/validate-skill-frontmatter.mjs [--json] [--strict]',
  });
  process.exit(0);
}

const skills = await collectSkills();
const issues = (await Promise.all(skills.map((skill) => validateSkill(skill, options)))).flat();
const report = {
  ...summarizeIssues('skill frontmatter', issues),
  skills: skills.map(({ body: _body, ...skill }) => skill),
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;

async function collectSkills() {
  const paths = await listMarkdownFiles('.github/skills', (relativePath) => relativePath.endsWith('/SKILL.md'));
  return Promise.all(
    paths.map(async (relativePath) => {
      const text = await readWorkspaceFile(relativePath);
      const parsed = parseFrontmatter(text, relativePath);
      return {
        path: relativePath,
        data: parsed.data,
        body: parsed.body,
        parseIssues: parsed.issues,
      };
    }),
  );
}

async function validateSkill(skill, { strict }) {
  const issues = [...skill.parseIssues];
  const { data, path: relativePath, body } = skill;
  const folderName = relativePath.split('/').at(-2);

  if (!data.name) issues.push(issue('error', relativePath, 'Skill name is required.'));
  if (data.name && !/^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/.test(data.name)) {
    issues.push(issue('error', relativePath, 'Skill name must be lowercase alphanumeric with hyphens and at most 64 characters.'));
  }
  if (data.name && data.name !== folderName) {
    issues.push(issue('error', relativePath, `Skill name '${data.name}' must match folder '${folderName}'.`));
  }
  if (!data.description) issues.push(issue('error', relativePath, 'Skill description is required.'));
  if (typeof data.description === 'string' && data.description.length > 1024) {
    issues.push(issue('error', relativePath, 'Skill description exceeds the Agent Skills 1024-character limit.'));
  }
  if ('user-invocable' in data && typeof data['user-invocable'] !== 'boolean') {
    issues.push(issue('error', relativePath, '`user-invocable` must be a boolean.'));
  }
  if ('disable-model-invocation' in data && typeof data['disable-model-invocation'] !== 'boolean') {
    issues.push(issue('error', relativePath, '`disable-model-invocation` must be a boolean.'));
  }
  if (data.compatibility && String(data.compatibility).length > 500) {
    issues.push(issue('error', relativePath, 'Compatibility text exceeds the 500-character Agent Skills limit.'));
  }
  if (strict && !('user-invocable' in data)) {
    issues.push(issue('warning', relativePath, 'Strict mode prefers an explicit `user-invocable` decision.'));
  }
  if (strict && !data['argument-hint']) {
    issues.push(issue('error', relativePath, 'Strict mode requires an `argument-hint` for discoverable task shaping.'));
  }

  for (const localLink of extractMarkdownLinks(body)) {
    const target = path.posix.normalize(path.posix.join(path.posix.dirname(relativePath), localLink));
    if (!(await fileExists(target))) {
      issues.push(issue('error', relativePath, `Referenced local resource does not exist: ${localLink}`));
    }
  }

  return issues;
}