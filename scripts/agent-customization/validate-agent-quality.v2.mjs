#!/usr/bin/env node
/**
 * validate-agent-quality.v2.mjs
 * Stricter validator enforcing exact '## Output format' header and that the
 * triple-fenced ```structured-v1``` block is the file tail. Works with
 * --fix (delegates to validate-agent-quality.fix.mjs) and --json output.
 */

import path from 'node:path';
import { pathToFileURL } from 'node:url';

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

const AGENT_QUALITY_CONTRACT_PATH = '.github/AGENT_QUALITY_CONTRACT.md';

const tierContracts = {
  1: {
    label: 'Tier-1 orchestrator',
    requiredSections: [
      'Mission',
      'Constraints',
      'Default Flow',
      'If Blocked',
      'Output format',
    ],
    requiredFields: [
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
    ],
  },
  2: {
    label: 'Tier-2 coordinator',
    requiredSections: [
      'Mission',
      'Constraints',
      'Required Workflow',
      'If Blocked',
      'Output format',
    ],
    requiredFields: [
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
    ],
  },
  3: {
    label: 'Tier-3 scout',
    requiredSections: [
      'Mission',
      'Constraints',
      'Approach',
      'If Blocked',
      'Output format',
    ],
    requiredFields: [
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
      'HANDOFF',
      'BLOCKERS',
      'RISKS_OR_GAPS',
      'LEARNING_EVENT_NEEDED',
      'SUGGESTED_NEXT_AGENT',
      'SUMMARY',
    ],
  },
  4: {
    label: 'Tier-4 auxiliary',
    requiredSections: [
      'Mission',
      'Constraints',
      'Default Flow',
      'If Blocked',
      'Output format',
    ],
    requiredFields: [
      'OUTPUT_CONTRACT',
      'TASK_STATUS',
      'TIER',
      'ROLE',
      'TASK_RECEIVED',
      'FILES_READ',
      'FILES_CHANGED',
      'KEY_FINDINGS',
      'ACTIONS_TAKEN',
      'BLOCKERS',
      'RISKS_OR_GAPS',
      'LEARNING_EVENT_NEEDED',
      'SUGGESTED_NEXT_AGENT',
      'SUMMARY',
    ],
  },
};

const options = parseArgs(process.argv.slice(2));
options.fix = process.argv.slice(2).includes('--fix');

if (options.fix) {
  const { runFix } = await import('./validate-agent-quality.fix.mjs');
  const fixReport = await runFix({ json: options.json });
  writeReport(fixReport, options);
  process.exitCode = fixReport.ok ? 0 : 1;
  process.exit();
}

if (options.help) {
  printUsage({
    title: 'Validate NeatapticTS agent quality contract compliance (v2).',
    usage: 'node scripts/agent-customization/validate-agent-quality.v2.mjs [--json] [--fix]',
  });
  process.exit(0);
}

export async function runValidateAgentQuality() {
  const agentReports = await collectAgentReports();
  const issues = agentReports.flatMap((agentReport) => agentReport.issues);
  const report = {
    ...summarizeIssues('agent quality (v2)', issues),
    contractDocument: AGENT_QUALITY_CONTRACT_PATH,
    agents: agentReports.map(({ path: relativePath, name, tier, issues: currentIssues }) => ({
      path: relativePath,
      name,
      tier,
      counts: {
        errors: currentIssues.filter((ci) => ci.severity === 'error').length,
        warnings: currentIssues.filter((ci) => ci.severity === 'warning').length,
      },
      issues: currentIssues,
    })),
  };

  return report;
}

async function main() {
  const report = await runValidateAgentQuality();
  writeReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  await main();
}

async function collectAgentReports() {
  const agentPaths = await listMarkdownFiles('.github/agents', (relativePath) =>
    relativePath.endsWith('.agent.md'),
  );

  return Promise.all(
    agentPaths.map(async (relativePath) => {
      const text = await readWorkspaceFile(relativePath);
      const parsed = parseFrontmatter(text, relativePath);
      const tier = normalizeTier(parsed.data.tier);
      const name = parsed.data.name ?? relativePath.split('/').at(-1)?.replace('.agent.md', '') ?? relativePath;
      const issues = validateAgent({
        path: relativePath,
        name,
        tier,
        body: parsed.body,
        data: parsed.data,
        parseIssues: parsed.issues,
      });

      return {
        path: relativePath,
        name,
        tier,
        issues,
      };
    }),
  );
}

function validateAgent(agent) {
  const issues = [...agent.parseIssues];

  if (!agent.tier) {
    issues.push(issue('error', agent.path, 'Agent frontmatter must define a valid numeric tier between 1 and 4.'));
    return issues;
  }

  const contract = tierContracts[agent.tier];
  if (!contract) {
    issues.push(issue('error', agent.path, `No agent quality contract is defined for tier '${agent.tier}'.`));
    return issues;
  }

  const sections = extractSections(agent.body);
  // Require only the canonical Output format section (case-sensitive) as a footer.
  const outputSection = sections.get('Output format');
  if (!outputSection) {
    issues.push(issue('error', agent.path, "Agent must define section '## Output format' with a trailing ```structured-v1``` block."));
    return issues;
  }

  issues.push(...validateStructuredOutputContract(agent, contract));
  return issues;
}

function validateStructuredOutputContract(agent, contract) {
  const issues = [];

  // Find exact '## Output format' heading (case-sensitive) in the document body.
  const headingRegex = /^##\s+Output format\s*$/m;
  const headingMatch = headingRegex.exec(agent.body);
  if (!headingMatch) {
    issues.push(issue('error', agent.path, "Missing required heading '## Output format' (case-sensitive)."));
    return issues;
  }

  const outputStartIndex = headingMatch.index + headingMatch[0].length;
  const afterHeading = agent.body.slice(outputStartIndex);

  // Find fences
  const fenceRegex = /```structured-v1\r?\n([\s\S]*?)\r?\n```/g;
  const matches = [...afterHeading.matchAll(fenceRegex)];
  if (matches.length !== 1) {
    issues.push(issue('error', agent.path, 'Output format must contain exactly one fenced ```structured-v1``` block.'));
    return issues;
  }

  const match = matches[0];
  const fenceBody = match[1] ?? '';

  // Ensure closing fence is at file tail (no non-whitespace after it)
  const closingFenceIndex = outputStartIndex + match.index + match[0].length;
  const tail = agent.body.slice(closingFenceIndex);
  if (tail.trim() !== '') {
    issues.push(issue('error', agent.path, 'The structured-v1 block must be the final content of the file (no other content after the closing fence).'));
  }

  const firstNonBlankLine = fenceBody.split(/\r?\n/).find((line) => line.trim());
  if (firstNonBlankLine?.trim() !== 'OUTPUT_CONTRACT: structured-v1') {
    issues.push(issue('error', agent.path, "The first non-blank line inside the structured-v1 block must be 'OUTPUT_CONTRACT: structured-v1'."));
  }

  const parsedFields = parsePromptFields(fenceBody);
  const detectedFields = parsedFields.map(({ field }) => field);

  if (
    detectedFields.length !== contract.requiredFields.length ||
    detectedFields.some((field, index) => field !== contract.requiredFields[index])
  ) {
    issues.push(issue('error', agent.path, `${contract.label} structured-v1 fields must match the exact order: ${contract.requiredFields.join(', ')}.`));
  }

  for (const requiredField of contract.requiredFields) {
    if (!detectedFields.includes(requiredField)) {
      issues.push(issue('error', agent.path, `Structured-v1 output contract is missing required field '${requiredField}'.`));
    }
  }

  const outputContractField = parsedFields.find(({ field }) => field === 'OUTPUT_CONTRACT');
  if (outputContractField?.value !== 'structured-v1') {
    issues.push(issue('error', agent.path, "Structured-v1 output contract must set 'OUTPUT_CONTRACT: structured-v1'."));
  }

  const tierField = parsedFields.find(({ field }) => field === 'TIER');
  if ((tierField?.value ?? '').toString() !== (agent.tier ?? '').toString()) {
    issues.push(issue('error', agent.path, `Structured-v1 output contract must set 'TIER: ${agent.tier}'.`));
  }

  const roleField = parsedFields.find(({ field }) => field === 'ROLE');
  if ((roleField?.value ?? '') !== String(agent.name)) {
    issues.push(issue('error', agent.path, `Structured-v1 output contract must set 'ROLE: ${agent.name}'.`));
  }

  return issues;
}

function extractSections(body) {
  const sectionMatches = [...body.matchAll(/^##\s+(?<name>.+)$/gmu)];
  const sections = new Map();

  for (const [index, currentMatch] of sectionMatches.entries()) {
    const sectionName = currentMatch.groups?.name?.trim();
    if (!sectionName) continue;

    const sectionStart = (currentMatch.index ?? 0) + currentMatch[0].length;
    const sectionEnd = sectionMatches[index + 1]?.index ?? body.length;
    sections.set(sectionName, body.slice(sectionStart, sectionEnd).trim());
  }

  return sections;
}

function parsePromptFields(fenceBody) {
  const parsedFields = [];

  for (const rawLine of fenceBody.split(/\r?\n/u)) {
    const trimmedLine = rawLine.trim();
    if (!trimmedLine || trimmedLine.startsWith('- ')) continue;

    const fieldMatch = /^(?<field>[A-Z_]+):\s*(?<value>.*)$/u.exec(trimmedLine);
    if (!fieldMatch?.groups) continue;

    parsedFields.push({
      field: fieldMatch.groups.field,
      value: fieldMatch.groups.value.trim(),
    });
  }

  return parsedFields;
}

function normalizeTier(rawTier) {
  if (rawTier === null || rawTier === undefined) return null;
  const tier = String(rawTier).trim();
  return /^[1-4]$/.test(tier) ? tier : null;
}
