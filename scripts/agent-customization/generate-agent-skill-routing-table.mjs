#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  fileExists,
  listMarkdownFiles,
  parseArgs,
  printUsage,
  readWorkspaceFile,
  repoRoot,
  writeReport,
} from './customization-utils.mjs';
import { runCustomizationInventory } from './inventory-customizations.mjs';

export const ROUTING_TABLE_PATH = '.github/agent-skill-routing-table.md';

const options = parseArgs(process.argv.slice(2));

const isMainModule =
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;

if (options.help && isMainModule) {
  printUsage({
    title: 'Generate the canonical NeatapticTS agent and skill routing table.',
    usage:
      'node scripts/agent-customization/generate-agent-skill-routing-table.mjs [--json]',
  });
  process.exit(0);
}

export async function collectCustomizationRoutingTable() {
  const inventory = await runCustomizationInventory();
  const sourceFiles = await collectRoutingSourceFiles();
  const sourceHash = await computeRoutingSourceHash(sourceFiles);
  const agentRows = createAgentRows(inventory.agents);
  const skillRows = createSkillRows(inventory.skills, inventory.agents);
  const markdown = renderRoutingTableMarkdown({
    sourceHash,
    sourceFiles,
    agentRows,
    skillRows,
  });

  return {
    inventory,
    sourceFiles,
    sourceHash,
    agentRows,
    skillRows,
    markdown,
  };
}

export async function runGenerateCustomizationRoutingTable({
  write = true,
} = {}) {
  const table = await collectCustomizationRoutingTable();
  const currentMarkdown = (await fileExists(ROUTING_TABLE_PATH))
    ? await readWorkspaceFile(ROUTING_TABLE_PATH)
    : null;
  const changed = currentMarkdown !== table.markdown;

  if (write && changed) {
    await writeFile(
      path.join(repoRoot, ROUTING_TABLE_PATH),
      table.markdown,
      'utf8',
    );
  }

  return {
    name: 'customization routing table',
    ok: true,
    path: ROUTING_TABLE_PATH,
    changed,
    sourceHash: table.sourceHash,
    sourceFileCount: table.sourceFiles.length,
    agentCount: table.agentRows.length,
    skillCount: table.skillRows.length,
    summaryText: [
      'PASS customization routing table',
      `path=${ROUTING_TABLE_PATH}`,
      `changed=${changed}`,
      `sourceHash=${table.sourceHash}`,
      `agents=${table.agentRows.length}`,
      `skills=${table.skillRows.length}`,
    ].join(' '),
  };
}

export function extractRoutingTableSourceHash(markdown) {
  return (
    /<!-- source-hash: (?<sourceHash>[a-f0-9]{64}) -->/u.exec(markdown)?.groups
      ?.sourceHash ?? null
  );
}

async function collectRoutingSourceFiles() {
  const agentPaths = await listMarkdownFiles('.github/agents', (relativePath) =>
    relativePath.endsWith('.agent.md'),
  );
  const skillPaths = await listMarkdownFiles('.github/skills', (relativePath) =>
    relativePath.endsWith('/SKILL.md'),
  );
  return [...agentPaths, ...skillPaths].toSorted();
}

async function computeRoutingSourceHash(sourceFiles) {
  const hasher = createHash('sha256');

  for (const relativePath of sourceFiles) {
    const fileContents = await readWorkspaceFile(relativePath);
    hasher.update(`>>> ${relativePath}\n${normalizeForHash(fileContents)}\n`);
  }

  return hasher.digest('hex');
}

export function createAgentRows(agents) {
  return agents
    .toSorted((leftAgent, rightAgent) =>
      leftAgent.name.localeCompare(rightAgent.name),
    )
    .map((agent) => ({
      name: agent.name,
      tier: agent.tier ?? '-',
      model: formatModel(agent.model),
      agents: formatList(agent.agents),
      skills: formatList(agent.skills),
    }));
}

export function createSkillRows(skills, agents) {
  const agentsBySkill = new Map(
    skills.map((skill) => [
      skill.name,
      agents
        .filter((agent) => agent.skills.includes(skill.name))
        .map((agent) => agent.name)
        .toSorted(),
    ]),
  );

  return skills
    .toSorted((leftSkill, rightSkill) =>
      leftSkill.name.localeCompare(rightSkill.name),
    )
    .map((skill) => ({
      name: skill.name,
      tier: 'skill',
      model: '-',
      agents: formatList(agentsBySkill.get(skill.name)),
      skills: 'self',
    }));
}

export function formatModel(model) {
  if (Array.isArray(model)) return model.join('<br>');
  return model ?? '-';
}

export function formatList(items) {
  return items.length > 0 ? items.join('<br>') : '-';
}

function normalizeForHash(text) {
  return text.replace(/\r\n/g, '\n');
}

function renderRoutingTableMarkdown({
  sourceHash,
  sourceFiles,
  agentRows,
  skillRows,
}) {
  return [
    '<!-- generated-by: scripts/agent-customization/generate-agent-skill-routing-table.mjs -->',
    `<!-- source-hash: ${sourceHash} -->`,
    `<!-- source-file-count: ${sourceFiles.length} -->`,
    '# Canonical Agent and Skill Routing Table',
    '',
    '> Generated file. Do not edit manually.',
    '> Refresh with `npm run agents:routing-table`.',
    '> Validate freshness with `npm run agents:routing-table:gate`.',
    '',
    '## Agents',
    '',
    '| Name | Tier | Model | Agents | Skills |',
    '| --- | --- | --- | --- | --- |',
    ...agentRows.map(renderTableRow),
    '',
    '## Skills',
    '',
    '| Name | Tier | Model | Agents | Skills |',
    '| --- | --- | --- | --- | --- |',
    ...skillRows.map(renderTableRow),
    '',
  ].join('\n');
}

function renderTableRow(row) {
  return `| ${row.name} | ${row.tier} | ${row.model} | ${row.agents} | ${row.skills} |`;
}

export async function main() {
  const report = await runGenerateCustomizationRoutingTable();
  writeReport(report, options);
}

export function handleMainError(error) {
  console.error(error);
  process.exitCode = 1;
}

if (isMainModule) {
  main().then(() => {}, handleMainError);
}
