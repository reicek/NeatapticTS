#!/usr/bin/env node
/**
 * Validate the NeatapticTS agent quality contract.
 *
 * Checks every `.github/agents/*.agent.md` file for:
 * - a valid frontmatter `tier:` field,
 * - the mandatory section set for that tier,
 * - a single `structured-v1` block under `## Output Format`,
 * - the exact tier-specific field order,
 * - `OUTPUT_CONTRACT: structured-v1` as the first field,
 * - and `ROLE` / `TIER` values that match frontmatter.
 *
 * Usage:
 *   node scripts/agent-customization/validate-agent-quality.mjs [--json]
 *   node scripts/agent-customization/validate-agent-quality.mjs --help
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
      'Output Format',
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
    documentedExpectations: [
      'Document the governing skills instead of copying durable policy into the agent body.',
      'Document MCP and gate usage when the orchestrator relies on MCP-only evidence or tooling.',
      'Keep delegation and downstream handoff boundaries explicit in the workflow text.',
    ],
  },
  2: {
    label: 'Tier-2 coordinator',
    requiredSections: [
      'Mission',
      'Constraints',
      'Required Workflow',
      'If Blocked',
      'Output Format',
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
    documentedExpectations: [
      'Document which companion skills own durable policy when the coordinator stays intentionally thin.',
      'Document MCP or gate-server usage when the coordinator depends on those boundaries.',
      'Keep reroute responsibility explicit through HANDOFF and SUGGESTED_NEXT_AGENT.',
    ],
  },
  3: {
    label: 'Tier-3 scout',
    requiredSections: [
      'Mission',
      'Constraints',
      'Approach',
      'If Blocked',
      'Output Format',
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
    documentedExpectations: [
      'Stay thin and read-only unless the frontmatter explicitly allows edits or execution.',
      'Document the smallest useful handoff instead of broad recommendations.',
      'Document MCP evidence dependencies only when the scout actually relies on them.',
    ],
  },
  4: {
    label: 'Tier-4 auxiliary',
    requiredSections: [
      'Mission',
      'Constraints',
      'Default Flow',
      'If Blocked',
      'Output Format',
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
    documentedExpectations: [
      'Keep the helper purpose narrow and one-shot.',
      'Document exactly when the helper should stop and hand back control.',
      'Document MCP or file-write expectations only when they materially affect the helper boundary.',
    ],
  },
};

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Validate NeatapticTS agent quality contract compliance.',
    usage:
      'node scripts/agent-customization/validate-agent-quality.mjs [--json]',
  });
  process.exit(0);
}

export async function runValidateAgentQuality() {
  const agentReports = await collectAgentReports();
  const issues = agentReports.flatMap((agentReport) => agentReport.issues);
  const report = {
    ...summarizeIssues('agent quality', issues),
    contractDocument: AGENT_QUALITY_CONTRACT_PATH,
    enforcedRules: {
      mandatorySectionsByTier: Object.fromEntries(
        Object.entries(tierContracts).map(([tier, contract]) => [
          tier,
          contract.requiredSections,
        ]),
      ),
      structuredFieldOrderByTier: Object.fromEntries(
        Object.entries(tierContracts).map(([tier, contract]) => [
          tier,
          contract.requiredFields,
        ]),
      ),
    },
    documentedExpectations: Object.fromEntries(
      Object.entries(tierContracts).map(([tier, contract]) => [
        tier,
        contract.documentedExpectations,
      ]),
    ),
    agents: agentReports.map(
      ({ path: relativePath, name, tier, issues: currentIssues }) => ({
        path: relativePath,
        name,
        tier,
        counts: {
          errors: currentIssues.filter(
            (currentIssue) => currentIssue.severity === 'error',
          ).length,
          warnings: currentIssues.filter(
            (currentIssue) => currentIssue.severity === 'warning',
          ).length,
        },
        issues: currentIssues,
      }),
    ),
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
      const name =
        parsed.data.name ??
        relativePath.split('/').at(-1)?.replace('.agent.md', '') ??
        relativePath;
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
    issues.push(
      issue(
        'error',
        agent.path,
        'Agent frontmatter must define a valid numeric tier between 1 and 4.',
      ),
    );
    return issues;
  }

  const contract = tierContracts[agent.tier];
  if (!contract) {
    issues.push(
      issue(
        'error',
        agent.path,
        `No agent quality contract is defined for tier '${agent.tier}'.`,
      ),
    );
    return issues;
  }

  const sections = extractSections(agent.body);
  for (const requiredSection of contract.requiredSections) {
    if (!sections.has(requiredSection)) {
      issues.push(
        issue(
          'error',
          agent.path,
          `${contract.label} agents must define section '## ${requiredSection}'.`,
        ),
      );
    }
  }

  const outputSection = sections.get('Output Format');
  if (!outputSection) {
    return issues;
  }

  issues.push(
    ...validateStructuredOutputContract(agent, contract, outputSection),
  );
  return issues;
}

function validateStructuredOutputContract(agent, contract, outputSection) {
  const issues = [];
  const structuredFenceMatches = [
    ...outputSection.matchAll(
      /```structured-v1\r?\n(?<body>[\s\S]*?)\r?\n```/gu,
    ),
  ];

  if (structuredFenceMatches.length !== 1) {
    issues.push(
      issue(
        'error',
        agent.path,
        'Output Format must contain exactly one fenced ```structured-v1``` block.',
      ),
    );
    return issues;
  }

  const fenceBody = structuredFenceMatches[0].groups?.body ?? '';
  const firstNonBlankLine = fenceBody
    .split(/\r?\n/u)
    .find((line) => line.trim());
  if (firstNonBlankLine?.trim() !== 'OUTPUT_CONTRACT: structured-v1') {
    issues.push(
      issue(
        'error',
        agent.path,
        "The first non-blank line inside the structured-v1 block must be 'OUTPUT_CONTRACT: structured-v1'.",
      ),
    );
  }

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
        `${contract.label} structured-v1 fields must match the exact order: ${contract.requiredFields.join(', ')}.`,
      ),
    );
  }

  for (const requiredField of contract.requiredFields) {
    if (!detectedFields.includes(requiredField)) {
      issues.push(
        issue(
          'error',
          agent.path,
          `Structured-v1 output contract is missing required field '${requiredField}'.`,
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
        "Structured-v1 output contract must set 'OUTPUT_CONTRACT: structured-v1'.",
      ),
    );
  }

  const tierField = parsedFields.find(({ field }) => field === 'TIER');
  if (tierField?.value !== agent.tier) {
    issues.push(
      issue(
        'error',
        agent.path,
        `Structured-v1 output contract must set 'TIER: ${agent.tier}'.`,
      ),
    );
  }

  const roleField = parsedFields.find(({ field }) => field === 'ROLE');
  if (roleField?.value !== agent.name) {
    issues.push(
      issue(
        'error',
        agent.path,
        `Structured-v1 output contract must set 'ROLE: ${agent.name}'.`,
      ),
    );
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
