#!/usr/bin/env node
/**
 * validate-agent-quality.mjs
 *
 * Enforces the canonical agent footer contract for all .github/agents/*.agent.md files.
 * Requirement: the final section of each agent must be an exact heading
 *   ## Output Format
 * followed by a single triple-fenced ```structured-v1``` block that is the file tail.
 *
 * This script supports:
 *   --json  : machine-readable JSON report
 *   --fix   : attempt auto-fixes (delegates to validate-agent-quality.fix.mjs)
 *   --help  : show usage
 *
 * Exports:
 *   runValidateAgentQuality(): Promise<object>  -- programmatic entrypoint returning the report
 *
 * Implementation notes:
 * - Field order, OUTPUT_CONTRACT, TIER and ROLE are validated against a tier-specific contract.
 * - Uses customization-utils.mjs helpers for frontmatter parsing and file IO.
 *
 * Additions: JSDoc on public functions and inline comments for maintainers.
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

/*
 * Per-tier required structured-v1 field order and expectations.
 * Keep this small, explicit, and easy to review when the structured-v1 contract evolves.
 */
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
  },
};

const options = parseArgs(process.argv.slice(2));
options.fix = process.argv.slice(2).includes('--fix');

// Delegate fixes to the fixer module when --fix is requested.
if (options.fix) {
  const { runFix } = await import('./validate-agent-quality.fix.mjs');
  const fixReport = await runFix({ json: options.json });
  writeReport(fixReport, options);
  /* istanbul ignore next */
  process.exitCode = fixReport.ok ? 0 : 1;
  process.exit();
}

if (options.help) {
  printUsage({
    title: 'Validate NeatapticTS agent quality contract compliance.',
    usage:
      'node scripts/agent-customization/validate-agent-quality.mjs [--json] [--fix]',
  });
  process.exit(0);
}

/**
 * Run the full validation pass and return a structured report object.
 * @returns {Promise<object>} - report information suitable for JSON output
 */
export async function runValidateAgentQuality() {
  const agentReports = await collectAgentReports();
  const issues = agentReports.flatMap((agentReport) => agentReport.issues);

  const report = {
    ...summarizeIssues('agent quality', issues),
    contractDocument: AGENT_QUALITY_CONTRACT_PATH,
    agents: agentReports.map(
      ({ path: relativePath, name, tier, issues: currentIssues }) => ({
        path: relativePath,
        name,
        tier,
        counts: {
          errors: currentIssues.filter((ci) => ci.severity === 'error').length,
          warnings: currentIssues.filter((ci) => ci.severity === 'warning')
            .length,
        },
        issues: currentIssues,
      }),
    ),
  };

  return report;
}

// If invoked directly from the CLI, run and print the report.
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

/**
 * Collect reports for every agent file under .github/agents.
 * Uses customization-utils to read files and parse frontmatter.
 */
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
        /* istanbul ignore next */
        relativePath;

      const issues = validateAgent({
        path: relativePath,
        name,
        tier,
        body: parsed.body,
        data: parsed.data,
        parseIssues: parsed.issues,
      });

      return { path: relativePath, name, tier, issues };
    }),
  );
}

/**
 * Validate a single agent's body and structured output contract.
 * @param {object} agent - { path, name, tier, body, data, parseIssues }
 * @returns {Array<object>} - list of issue objects
 */
function validateAgent(agent) {
  // Start with any parse-time frontmatter issues.
  const issues = [...agent.parseIssues];

  // The tier must be present and normalized to 1-4.
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
  /* istanbul ignore next -- unreachable because normalizeTier only returns 1-4 */
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

  // Extract ## sections (map name -> content) and require the canonical Output Format section.
  const sections = extractSections(agent.body);
  const outputSection = sections.get('Output Format');
  if (!outputSection) {
    issues.push(
      issue(
        'error',
        agent.path,
        "Agent must define section '## Output Format' with a trailing ```structured-v1``` block.",
      ),
    );
    return issues;
  }

  // Validate the structured-v1 block and its fields.
  issues.push(...validateStructuredOutputContract(agent, contract));
  return issues;
}

/**
 * Validate the fenced structured-v1 block under the exact '## Output Format' heading.
 * Enforces:
 *  - a single ```structured-v1``` fence
 *  - the fence is the file tail (no other content after closing fence)
 *  - first non-blank line is 'OUTPUT_CONTRACT: structured-v1'
 *  - field presence and exact order per-tier
 *  - TIER and ROLE values match frontmatter
 *
 * @param {object} agent - agent descriptor
 * @param {object} contract - tier contract
 * @returns {Array<object>} - issues found
 */
function validateStructuredOutputContract(agent, contract) {
  const issues = [];

  // Case-sensitive search for the canonical heading.
  const headingRegex = /^##\s+Output Format\s*$/m;
  const headingMatch = headingRegex.exec(agent.body);
  /* istanbul ignore next -- unreachable because validateAgent already requires the section */
  if (!headingMatch) {
    issues.push(
      issue(
        'error',
        agent.path,
        "Missing required heading '## Output Format' (case-sensitive).",
      ),
    );
    return issues;
  }

  const outputStartIndex = headingMatch.index + headingMatch[0].length;
  const afterHeading = agent.body.slice(outputStartIndex);

  // Match the fenced structured-v1 block. Use a global regex and count matches.
  const fenceRegex = /```structured-v1\r?\n([\s\S]*?)\r?\n```/g;
  const matches = [...afterHeading.matchAll(fenceRegex)];
  if (matches.length !== 1) {
    issues.push(
      issue(
        'error',
        agent.path,
        'Output format must contain exactly one fenced ```structured-v1``` block.',
      ),
    );
    return issues;
  }

  const match = matches[0];
  /* istanbul ignore next */
  const fenceBody = match[1] ?? '';

  // Ensure nothing (non-whitespace) exists after the closing fence: the block must be the file tail.
  const closingFenceIndex = outputStartIndex + match.index + match[0].length;
  const tail = agent.body.slice(closingFenceIndex);
  if (tail.trim() !== '') {
    issues.push(
      issue(
        'error',
        agent.path,
        'The structured-v1 block must be the final content of the file (no other content after the closing fence).',
      ),
    );
  }

  // First non-blank line must declare the contract marker.
  const firstNonBlankLine = fenceBody
    .split(/\r?\n/)
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

  // Parse uppercase FIELD: value lines inside the fence body.
  const parsedFields = parsePromptFields(fenceBody);
  const detectedFields = parsedFields.map(({ field }) => field);

  // Enforce exact field count and ordering per contract.
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

  // Ensure each required field is present (redundant but provides clear missing-field errors).
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

  // OUTPUT_CONTRACT value check
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

  // TIER must match the parsed frontmatter tier
  const tierField = parsedFields.find(({ field }) => field === 'TIER');
  const tierValueRaw = tierField?.value;
  /* istanbul ignore next */
  const tierValue = tierValueRaw ?? '';
  /* istanbul ignore next */
  const agentTier = agent.tier ?? '';
  if (tierValue.toString() !== agentTier.toString()) {
    issues.push(
      issue(
        'error',
        agent.path,
        `Structured-v1 output contract must set 'TIER: ${agent.tier}'.`,
      ),
    );
  }

  // ROLE must match the agent name
  const roleField = parsedFields.find(({ field }) => field === 'ROLE');
  const roleValueRaw = roleField?.value;
  /* istanbul ignore next */
  const roleValue = roleValueRaw ?? '';
  if (roleValue !== String(agent.name)) {
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

/**
 * Extracts all top-level '##' sections from a markdown body.
 * Returns a Map(sectionName -> sectionContent).
 * @param {string} body - markdown body
 * @returns {Map<string,string>} sections
 */
function extractSections(body) {
  const sectionMatches = [...body.matchAll(/^##\s+(?<name>.+)$/gmu)];
  const sections = new Map();

  for (const [index, currentMatch] of sectionMatches.entries()) {
    const sectionName = currentMatch.groups?.name?.trim();
    /* istanbul ignore next */
    if (!sectionName) continue;

    /* istanbul ignore next */
    const sectionStart = (currentMatch.index ?? 0) + currentMatch[0].length;
    /* istanbul ignore next */
    const sectionEnd = sectionMatches[index + 1]?.index ?? body.length;
    sections.set(sectionName, body.slice(sectionStart, sectionEnd).trim());
  }

  return sections;
}

/**
 * Parse uppercase FIELD: value pairs from the structured-v1 fence body.
 * Ignores bullet lines and blank lines.
 * @param {string} fenceBody
 * @returns {Array<{field:string,value:string}>}
 */
function parsePromptFields(fenceBody) {
  const parsedFields = [];

  for (const rawLine of fenceBody.split(/\r?\n/u)) {
    const trimmedLine = rawLine.trim();
    if (!trimmedLine || trimmedLine.startsWith('- ')) continue; // ignore bullets and blank lines

    const fieldMatch = /^(?<field>[A-Z_]+):\s*(?<value>.*)$/u.exec(trimmedLine);
    if (!fieldMatch?.groups) continue;

    parsedFields.push({
      field: fieldMatch.groups.field,
      value: fieldMatch.groups.value.trim(),
    });
  }

  return parsedFields;
}

/**
 * Normalize and validate a tier value from frontmatter.
 * Returns '1'..'4' or null when invalid.
 */
function normalizeTier(rawTier) {
  if (rawTier === null || rawTier === undefined) return null;
  const tier = String(rawTier).trim();
  return /^[1-4]$/.test(tier) ? tier : null;
}
