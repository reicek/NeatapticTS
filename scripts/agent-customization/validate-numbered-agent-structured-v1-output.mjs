#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  issue,
  parseArgs,
  printUsage,
  summarizeIssues,
  writeReport,
} from './customization-utils.mjs';

const STRUCTURED_V1_FENCE_LABEL = 'structured-v1';
const STRUCTURED_V1_FIELD_PATTERN = /^(?<field>[A-Z_]+):\s*(?<value>.*)$/u;
const STRUCTURED_V1_LIST_ITEM_PATTERN = /^-\s+(?<value>.+)$/u;
const STRUCTURED_V1_CONTRACTS = {
  tier0: {
    name: 'tier0',
    tier: '0',
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
    listCapableFields: new Set([
      'FILES_READ',
      'FILES_CHANGED',
      'KEY_FINDINGS',
      'ACTIONS_TAKEN',
      'VALIDATION_EVIDENCE',
      'BLOCKERS',
      'RISKS_OR_GAPS',
      'SUB_ORCHESTRATORS_USED',
    ]),
  },
  tier1: {
    name: 'tier1',
    tier: '1',
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
    listCapableFields: new Set([
      'FILES_READ',
      'FILES_CHANGED',
      'KEY_FINDINGS',
      'ACTIONS_TAKEN',
      'VALIDATION_EVIDENCE',
      'SPECIALISTS_USED',
      'BLOCKERS',
      'RISKS_OR_GAPS',
    ]),
  },
};
const STRUCTURED_V1_CONTRACT_ALIASES = new Map([
  ['tier0', 'tier0'],
  ['tier-0', 'tier0'],
  ['numbered', 'tier0'],
  ['tier1', 'tier1'],
  ['tier-1', 'tier1'],
  ['coordinator', 'tier1'],
]);

const options = parseArgs(process.argv.slice(2));
const contract = resolveContract(options.contract);

if (options.help) {
  printUsage({
    title: 'Validate Tier-0 and Tier-1 agent structured-v1 output envelopes.',
    usage:
      'node scripts/agent-customization/validate-numbered-agent-structured-v1-output.mjs [--json] [--contract=tier0|tier1] --input=scripts/agent-customization/fixtures/numbered-agent-structured-v1.valid.md',
    options: [
      [
        '--contract=<tier0|tier1>',
        'Structured-v1 contract to validate. Defaults to tier0.',
      ],
      ['--input=<path>', 'Structured-v1 output fixture to validate.'],
    ],
  });
  process.exit(0);
}

const inputPath = typeof options.input === 'string' ? options.input.trim() : '';
const issues = [];
let outputText = '';

if (!contract) {
  issues.push(
    issue(
      'error',
      'scripts/agent-customization/validate-numbered-agent-structured-v1-output.mjs',
      `Unsupported --contract value '${options.contract}'. Use tier0 or tier1.`,
    ),
  );
}

if (!inputPath) {
  issues.push(
    issue(
      'error',
      'scripts/agent-customization/validate-numbered-agent-structured-v1-output.mjs',
      'The validator requires --input=<path>.',
    ),
  );
} else {
  try {
    outputText = await readFile(path.resolve(process.cwd(), inputPath), 'utf8');
  } catch (error) {
    if (
      error &&
      typeof error === 'object' &&
      'code' in error &&
      error.code === 'ENOENT'
    ) {
      issues.push(
        issue(
          'error',
          inputPath,
          'Structured-v1 output fixture was not found.',
        ),
      );
    } else {
      throw error;
    }
  }
}

const validationResult =
  outputText && contract
    ? validateStructuredV1Output(outputText, inputPath, contract)
    : { issues: [], detectedFields: [], parsedFields: {} };

issues.push(...validationResult.issues);

const report = {
  ...summarizeIssues(
    `${contract?.name ?? 'unknown'} structured-v1 output`,
    issues,
  ),
  input: inputPath || null,
  contract: contract?.name ?? null,
  fence: STRUCTURED_V1_FENCE_LABEL,
  requiredFields: contract?.requiredFields ?? [],
  detectedFields: validationResult.detectedFields,
  parsedFields: validationResult.parsedFields,
};

writeReport(report, options);
process.exitCode = report.ok ? 0 : 1;

function resolveContract(contractOption) {
  const normalized =
    typeof contractOption === 'string'
      ? contractOption.trim().toLowerCase()
      : 'tier0';
  const contractName = STRUCTURED_V1_CONTRACT_ALIASES.get(normalized);
  return contractName ? STRUCTURED_V1_CONTRACTS[contractName] : null;
}

function validateStructuredV1Output(outputText, outputPath, contract) {
  const issues = [];
  const parsedFields = {};
  const detectedFields = [];
  const structuredFenceMatch =
    /^```structured-v1\r?\n(?<body>[\s\S]*?)\r?\n```$/u.exec(outputText.trim());

  if (!structuredFenceMatch?.groups?.body) {
    issues.push(
      issue(
        'error',
        outputPath,
        'Output must be only one fenced ```structured-v1``` block with no surrounding prose.',
      ),
    );
    return { issues, detectedFields, parsedFields };
  }

  let activeField = null;

  for (const rawLine of structuredFenceMatch.groups.body.split(/\r?\n/u)) {
    const trimmedLine = rawLine.trim();
    if (!trimmedLine) {
      continue;
    }

    const fieldMatch = STRUCTURED_V1_FIELD_PATTERN.exec(trimmedLine);
    if (!fieldMatch?.groups) {
      const listItemMatch = STRUCTURED_V1_LIST_ITEM_PATTERN.exec(trimmedLine);
      if (!listItemMatch?.groups?.value) {
        issues.push(
          issue(
            'error',
            outputPath,
            `Structured-v1 line must match FIELD: value or list-item syntax: ${trimmedLine}`,
          ),
        );
        continue;
      }

      if (!activeField) {
        issues.push(
          issue(
            'error',
            outputPath,
            `List item must belong to a preceding field: ${trimmedLine}`,
          ),
        );
        continue;
      }

      if (!contract.listCapableFields.has(activeField)) {
        issues.push(
          issue(
            'error',
            outputPath,
            `Field ${activeField} does not accept list items.`,
          ),
        );
        continue;
      }

      const listValue = listItemMatch.groups.value.trim();
      /* istanbul ignore if -- regex requires .+ so value is never empty after trim */
      if (!listValue) {
        issues.push(
          issue(
            'error',
            outputPath,
            `${activeField} list items must not be empty.`,
          ),
        );
        continue;
      }

      if (typeof parsedFields[activeField] === 'string') {
        issues.push(
          issue(
            'error',
            outputPath,
            `Field ${activeField} cannot mix inline values with list items.`,
          ),
        );
        continue;
      }

      /* istanbul ignore if -- list-capable fields always get [] at line 305 or string (caught above) */
      if (!Array.isArray(parsedFields[activeField])) {
        parsedFields[activeField] = [];
      }

      parsedFields[activeField].push(listValue);
      continue;
    }

    const { field, value } = fieldMatch.groups;
    detectedFields.push(field);
    activeField = field;

    if (!contract.requiredFields.includes(field)) {
      issues.push(
        issue('error', outputPath, `Unexpected structured-v1 field: ${field}.`),
      );
      continue;
    }

    if (Object.hasOwn(parsedFields, field)) {
      issues.push(
        issue('error', outputPath, `Duplicate structured-v1 field: ${field}.`),
      );
      continue;
    }

    if (value.trim()) {
      parsedFields[field] = value.trim();
      continue;
    }

    parsedFields[field] = contract.listCapableFields.has(field) ? [] : '';
  }

  for (const requiredField of contract.requiredFields) {
    if (!Object.hasOwn(parsedFields, requiredField)) {
      issues.push(
        issue(
          'error',
          outputPath,
          `Missing required structured-v1 field: ${requiredField}.`,
        ),
      );
    }
  }

  if (
    detectedFields.some(
      (field, index) => field !== contract.requiredFields[index],
    )
  ) {
    issues.push(
      issue(
        'error',
        outputPath,
        `Structured-v1 fields must appear in exact ${contract.name} order: ${contract.requiredFields.join(', ')}.`,
      ),
    );
  }

  validateScalarField(parsedFields, 'OUTPUT_CONTRACT', outputPath, issues, {
    exactValue: STRUCTURED_V1_FENCE_LABEL,
  });
  validateScalarField(parsedFields, 'TASK_STATUS', outputPath, issues, {
    allowedValues: ['SUCCESS', 'PARTIAL', 'FAILED'],
  });
  validateScalarField(parsedFields, 'TIER', outputPath, issues, {
    exactValue: contract.tier,
  });
  validateScalarField(parsedFields, 'ROLE', outputPath, issues);
  validateScalarField(parsedFields, 'TASK_RECEIVED', outputPath, issues);
  validateFieldContent(parsedFields, 'FILES_READ', outputPath, issues);
  validateFieldContent(parsedFields, 'FILES_CHANGED', outputPath, issues);
  validateFieldContent(parsedFields, 'KEY_FINDINGS', outputPath, issues);
  validateFieldContent(parsedFields, 'ACTIONS_TAKEN', outputPath, issues);
  validateFieldContent(parsedFields, 'VALIDATION_EVIDENCE', outputPath, issues);
  if (contract.name === 'tier1') {
    validateFieldContent(parsedFields, 'SPECIALISTS_USED', outputPath, issues);
    validateScalarField(parsedFields, 'HANDOFF', outputPath, issues);
  }
  validateFieldContent(parsedFields, 'BLOCKERS', outputPath, issues);
  validateFieldContent(parsedFields, 'RISKS_OR_GAPS', outputPath, issues);
  validateScalarField(
    parsedFields,
    'LEARNING_EVENT_NEEDED',
    outputPath,
    issues,
    { allowedValues: ['true', 'false'] },
  );
  validateScalarField(parsedFields, 'SUGGESTED_NEXT_AGENT', outputPath, issues);
  if (contract.name === 'tier0') {
    validateScalarField(parsedFields, 'PHASE_COMPLETE', outputPath, issues, {
      allowedValues: ['true', 'false'],
    });
    validateFieldContent(
      parsedFields,
      'SUB_ORCHESTRATORS_USED',
      outputPath,
      issues,
    );
  }
  validateScalarField(parsedFields, 'SUMMARY', outputPath, issues);

  return {
    issues,
    detectedFields,
    parsedFields,
  };
}

function validateFieldContent(parsedFields, fieldName, outputPath, issues) {
  const value = parsedFields[fieldName];
  if (typeof value === 'string') {
    /* istanbul ignore if -- string values are always non-empty from line 301 */
    if (!value.trim()) {
      issues.push(
        issue('error', outputPath, `${fieldName} must not be empty.`),
      );
    }
    return;
  }

  if (!Array.isArray(value) || value.length === 0) {
    issues.push(
      issue(
        'error',
        outputPath,
        `${fieldName} must include at least one value.`,
      ),
    );
  }
}

function validateScalarField(
  parsedFields,
  fieldName,
  outputPath,
  issues,
  { exactValue, allowedValues } = {},
) {
  const value = parsedFields[fieldName];
  if (typeof value !== 'string' || !value.trim()) {
    issues.push(
      issue(
        'error',
        outputPath,
        `${fieldName} must be a non-empty scalar value.`,
      ),
    );
    return;
  }

  if (exactValue && value !== exactValue) {
    issues.push(
      issue('error', outputPath, `${fieldName} must equal '${exactValue}'.`),
    );
  }

  if (allowedValues && !allowedValues.includes(value)) {
    issues.push(
      issue(
        'error',
        outputPath,
        `${fieldName} must be one of: ${allowedValues.join(', ')}.`,
      ),
    );
  }
}
