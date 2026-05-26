#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  listMarkdownFiles,
  parseArgs,
  parseFrontmatter,
  printUsage,
  readWorkspaceFile,
  writeReport,
} from './customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Inventory NeatapticTS agent customizations.',
    usage: 'node scripts/agent-customization/inventory-customizations.mjs [--json]',
  });
  process.exit(0);
}

export async function runCustomizationInventory() {
  const agents = await collectAgents();
  const skills = await collectSkills();
  const report = {
    name: 'customization inventory',
    ok: true,
    summary: {
      agents: agents.length,
      userInvocableAgents: agents.filter((agent) => agent.userInvocable !== false).length,
      skills: skills.length,
      userInvocableSkills: skills.filter((skill) => skill.userInvocable !== false).length,
    },
    agents,
    skills,
  };

  report.summaryText = [
    `PASS customization inventory`,
    `agents=${report.summary.agents}`,
    `userInvocableAgents=${report.summary.userInvocableAgents}`,
    `skills=${report.summary.skills}`,
    `userInvocableSkills=${report.summary.userInvocableSkills}`,
  ].join(' ');

  return report;
}

async function main() {
  const report = await runCustomizationInventory();
  writeReport(report, options);
}

async function collectAgents() {
  const paths = await listMarkdownFiles('.github/agents', (relativePath) => relativePath.endsWith('.agent.md'));
  return Promise.all(paths.map(readAgent));
}

async function collectSkills() {
  const paths = await listMarkdownFiles('.github/skills', (relativePath) => relativePath.endsWith('/SKILL.md'));
  return Promise.all(paths.map(readSkill));
}

async function readAgent(relativePath) {
  const { data, raw } = parseFrontmatter(await readWorkspaceFile(relativePath), relativePath);
  return {
    path: relativePath,
    name: data.name ?? relativePath.split('/').at(-1)?.replace('.agent.md', ''),
    description: data.description ?? '',
    tier: data.tier ?? null,
    tools: Array.isArray(data.tools) ? data.tools : [],
    agents: Array.isArray(data.agents) ? data.agents : [],
    skills: Array.isArray(data.skills) ? data.skills : [],
    model: data.model ?? null,
    handoffs: raw.includes('\nhandoffs:'),
    userInvocable: data['user-invocable'] ?? true,
    disableModelInvocation: data['disable-model-invocation'] ?? false,
  };
}

async function readSkill(relativePath) {
  const { data, body } = parseFrontmatter(await readWorkspaceFile(relativePath), relativePath);
  return {
    path: relativePath,
    name: data.name ?? '',
    description: data.description ?? '',
    argumentHint: data['argument-hint'] ?? null,
    userInvocable: data['user-invocable'] ?? true,
    disableModelInvocation: data['disable-model-invocation'] ?? false,
    context: data.context ?? null,
    license: data.license ?? null,
    bodyLines: body.split(/\r?\n/).length,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}