#!/usr/bin/env node
import { pathToFileURL } from 'node:url';
import {
  extractStatus,
  fileExists,
  issue,
  normalizePath,
  parseArgs,
  parsePlanYamlBlock,
  printUsage,
  readWorkspaceFile,
  summarizeIssues,
  writeReport,
} from './customization-utils.mjs';

const GOAL_VALUES = new Set([
  'planning',
  'researching',
  'red-testing',
  'implementing',
  'green-testing',
  'documenting',
  'logging',
  'helping',
]);

const TDD_SEQUENCE_VALUES = new Set(['red-green', 'green-only']);
const EXPANSION_PHASE = 'steps';
const EXPANSION_STEP_VALUES = new Set(['none', 'slices']);
const SLICE_GOAL_VALUES = new Set([
  'red-testing',
  'implementing',
  'green-testing',
  'documenting',
  'researching',
  'helping',
]);

const PHASE_REQUIRED_KEYS = [
  'phase',
  'title',
  'status',
  'goal',
  'expansion',
  'auto_expand',
  'mode',
  'source_of_truth',
  'copy_paste',
  'next_phase',
  'skills',
  'validation',
  'acceptance_criteria',
  'placeholder_steps',
];

const STEP_REQUIRED_KEYS = [
  'phase',
  'step',
  'title',
  'status',
  'goal',
  'mode',
  'source_of_truth',
  'copy_paste',
  'next_step',
  'skills',
  'validation',
  'acceptance_criteria',
];

const STEP_OPTIONAL_KEYS = new Set([
  'tdd_sequence',
  'expansion',
  'auto_expand',
  'slices',
  'specialists',
  'owner',
  'reviewer',
  'evidence',
]);

const PHASE_REQUIRED_SECTIONS = [
  '**Phase objective:**',
  '**Stop conditions:**',
  '**Required validation:**',
];

const STEP_REQUIRED_SECTIONS = [
  '**User instruction:**',
  '**Step objective:**',
  '**Stop conditions:**',
  '**Required validation:**',
];

const FORBIDDEN_SECTIONS = ['**Copy-paste prompt:**'];

const MIGRATION_HINT =
  'Legacy plan block detected. Run: node scripts/agent-customization/migrate-plan-format.mjs --plan=<plan-file>';

const STALE_TESTPATHPATTERN_FLAG = '--testPathPattern';
const VALIDATION_PATH_HINT =
  'Store validation entries as file paths (e.g. scripts/foo.test.ts), not Jest CLI flags.';

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Validate copy-pasteable plan phase/step packets.',
    usage:
      'node scripts/agent-customization/validate-plan-phase-packets.mjs [--json] [--plan=plans/PlanName.plans.md]',
    options: [
      [
        '--plan=<path>',
        'Plan file whose implementation phases should be validated.',
      ],
    ],
  });
  process.exit(0);
}

let planPath = options.plan;
let normalizedPlanPath = normalizePath(planPath);
let planText = '';
let planStatus = null;
let isArchivedClosedPlan = false;
const issues = [];
let phases = [];

async function main() {
  if (options.help) {
    printUsage({
      title: 'Validate copy-pasteable plan phase/step packets.',
      usage:
        'node scripts/agent-customization/validate-plan-phase-packets.mjs [--json] [--plan=plans/PlanName.plans.md]',
      options: [
        [
          '--plan=<path>',
          'Plan file whose implementation phases should be validated.',
        ],
      ],
    });
    return;
  }

  planText = await readWorkspaceFile(planPath);
  const report = await validatePlanText(planText, planPath);
  writeReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
}

/**
 * Validate the implementation-phase packets in a plan document.
 *
 * Exported so tests can exercise the validator without spawning a child
 * process. The function resets internal state, parses phase/step YAML packets,
 * and returns the same report shape the CLI emits, including `ok`, `counts`,
 * `issues`, and a `phases` summary.
 *
 * @param text - Plan markdown content.
 * @param planFilePath - Repo-relative path used for messages and source_of_truth checks.
 * @returns Validation report with `ok`, `counts`, `issues`, and `phases` fields.
 */
export async function validatePlanText(text, planFilePath) {
  planPath = planFilePath;
  normalizedPlanPath = normalizePath(planPath);
  planText = text;
  planStatus = extractStatus(planText);
  isArchivedClosedPlan =
    planStatus === 'DONE' && normalizedPlanPath.startsWith('plans/completed/');
  issues.length = 0;
  phases = [...extractPhaseBlocks(planText)].map((phaseBlock) => ({
    ...phaseBlock,
    stepBlocks: [...extractStepBlocks(phaseBlock.body)],
  }));

  if (phases.length === 0 && !isArchivedClosedPlan) {
    issues.push(
      issue('error', planPath, 'No implementation phase packets found.'),
    );
  }

  for (const [phaseIndex, phaseBlock] of phases.entries()) {
    const previousPhase = phaseIndex === 0 ? null : phases.at(phaseIndex - 1);
    await validatePhase(phaseBlock, previousPhase?.headingPhase ?? null);
  }

  const wipCount = phases.filter(
    (phaseBlock) => phaseBlock.headingStatus === 'WIP',
  ).length;

  if (isArchivedClosedPlan && wipCount !== 0) {
    issues.push(
      issue(
        'error',
        planPath,
        `Archived [DONE] plans must not contain [WIP] phases, found ${wipCount}.`,
      ),
    );
  }

  if (!isArchivedClosedPlan && wipCount !== 1) {
    issues.push(
      issue(
        'warning',
        planPath,
        `Expected exactly one [WIP] phase, found ${wipCount}.`,
      ),
    );
  }

  return {
    ...summarizeIssues('plan phase packets', issues),
    plan: planPath,
    phases: phases.map((phaseBlock) => ({
      phase: phaseBlock.headingPhase,
      title: phaseBlock.headingTitle,
      status: phaseBlock.headingStatus,
      schema: phaseBlock.metadata?.step !== undefined ? 'step' : 'phase',
      goal:
        phaseBlock.stepBlocks.find(
          (stepBlock) => stepBlock.headingStatus === 'WIP',
        )?.metadata?.goal ??
        phaseBlock.metadata?.goal ??
        null,
    })),
  };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

function* extractPhaseBlocks(text) {
  const implementationMatch =
    /^## Implementation phases\s*(?<body>[\s\S]*?)(?=^## [^\n]*\bvalidation gates\b[^\n]*$)/imu.exec(
      text,
    );
  if (!implementationMatch?.groups) return;

  const implementationBody = implementationMatch.groups.body;
  const phasePattern =
    /^### Phase (?<phase>[A-Z0-9]+) — (?<title>.+?) \[(?<status>PLANNED|WIP|DONE)\]\s*$/gmu;
  const matches = [...implementationBody.matchAll(phasePattern)];

  for (const [matchIndex, match] of matches.entries()) {
    if (!match.groups) continue;
    const bodyStart = (match.index ?? 0) + match[0].length;
    const nextMatch = matches.at(matchIndex + 1);
    const bodyEnd = nextMatch?.index ?? implementationBody.length;
    const phaseBody = implementationBody.slice(bodyStart, bodyEnd);
    const metadata = parseMetadataBlock(phaseBody);
    const phaseLabel = match.groups.phase;
    yield {
      headingPhase: /^\d+$/u.test(phaseLabel) ? Number(phaseLabel) : phaseLabel,
      headingTitle: match.groups.title,
      headingStatus: match.groups.status,
      body: phaseBody,
      metadata,
    };
  }
}

function* extractStepBlocks(phaseBody) {
  const stepPattern =
    /^#### Step (?<step>\d{2,})\s*[:\-—]\s*(?<title>.+?) \[(?<status>PLANNED|WIP|DONE)\]\s*$/gmu;
  const matches = [...phaseBody.matchAll(stepPattern)];

  for (const [matchIndex, match] of matches.entries()) {
    if (!match.groups) continue;
    const bodyStart = (match.index ?? 0) + match[0].length;
    const nextMatch = matches.at(matchIndex + 1);
    const bodyEnd = nextMatch?.index ?? phaseBody.length;
    const stepBody = phaseBody.slice(bodyStart, bodyEnd);
    yield {
      headingStep: Number(match.groups.step),
      headingTitle: match.groups.title,
      headingStatus: match.groups.status,
      body: stepBody,
      metadata: parseMetadataBlock(stepBody),
    };
  }
}

function parseMetadataBlock(body) {
  const match = /^\s*```yaml\r?\n(?<yaml>[\s\S]*?)\r?\n```/mu.exec(body);
  if (!match?.groups) return null;
  try {
    return parsePlanYamlBlock(match.groups.yaml);
  } catch (error) {
    return { __parseError: String(error) };
  }
}

async function validatePhase(phaseBlock, previousPhaseLabel) {
  const phasePath = `${planPath}#phase-${phaseBlock.headingPhase}`;

  if (previousPhaseLabel !== null) {
    const expectedPhaseLabel = getNextPhaseLabel(previousPhaseLabel);
    if (
      expectedPhaseLabel !== null &&
      normalizePhaseLabel(phaseBlock.headingPhase) !== expectedPhaseLabel
    ) {
      issues.push(
        issue(
          'error',
          phasePath,
          `Expected phase ${expectedPhaseLabel}, found phase ${phaseBlock.headingPhase}.`,
        ),
      );
    }
  }

  const { metadata } = phaseBlock;

  if (metadata === null) {
    if (phaseBlock.headingStatus !== 'DONE') {
      issues.push(
        issue(
          'error',
          phasePath,
          'Active or planned phase must contain a YAML metadata block.',
        ),
      );
    }
    return;
  }

  if (metadata.__parseError) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Could not parse phase YAML: ${metadata.__parseError}`,
      ),
    );
    return;
  }

  if (isLegacyBlock(metadata, phaseBlock.headingStatus)) {
    issues.push(issue('error', phasePath, `${MIGRATION_HINT} (phase-level)`));
    return;
  }

  const isStepPacket = metadata.step !== undefined;

  if (isStepPacket) {
    await validateStepPacketInPhase(phaseBlock, phasePath);
    return;
  }

  await validatePhasePacket(phaseBlock, phasePath);
}

async function validatePhasePacket(phaseBlock, phasePath) {
  const { metadata, body, headingPhase, headingTitle, headingStatus } =
    phaseBlock;

  for (const key of PHASE_REQUIRED_KEYS) {
    if (!Object.hasOwn(metadata, key)) {
      issues.push(
        issue('error', phasePath, `Missing phase metadata key: ${key}.`),
      );
    }
  }

  if (metadata.title !== headingTitle) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Metadata title ${String(metadata.title)} does not match heading title.`,
      ),
    );
  }

  const metadataPhase = normalizePhaseLabel(metadata.phase);
  if (metadataPhase !== normalizePhaseLabel(headingPhase)) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Metadata phase ${String(metadata.phase)} does not match heading.`,
      ),
    );
  }

  const metadataStatus = stripStatus(metadata.status);
  if (metadataStatus !== headingStatus) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Metadata status ${String(metadata.status)} does not match heading.`,
      ),
    );
  }

  if (metadata.goal !== 'planning') {
    issues.push(
      issue(
        'error',
        phasePath,
        `Phase-level goal must be 'planning', found ${String(metadata.goal)}.`,
      ),
    );
  }

  if (metadata.expansion !== EXPANSION_PHASE) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Phase-level expansion must be '${EXPANSION_PHASE}', found ${String(metadata.expansion)}.`,
      ),
    );
  }

  if (metadata.auto_expand !== false) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Phase-level auto_expand must be false, found ${String(metadata.auto_expand)}.`,
      ),
    );
  }

  if (!['fresh-session', 'perpetual'].includes(metadata.mode)) {
    issues.push(
      issue(
        'error',
        phasePath,
        "Metadata mode must be 'fresh-session' or 'perpetual'.",
      ),
    );
  }

  const normalizedSource = normalizePath(
    String(metadata.source_of_truth ?? ''),
  );
  if (normalizedSource !== normalizedPlanPath) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Metadata source_of_truth must be ${normalizedPlanPath}.`,
      ),
    );
  }

  if (!booleanValue(metadata.copy_paste)) {
    issues.push(issue('error', phasePath, 'Metadata copy_paste must be true.'));
  }

  if (!Array.isArray(metadata.skills) || metadata.skills.length === 0) {
    issues.push(
      issue('error', phasePath, 'Metadata skills must be a non-empty list.'),
    );
  }

  if (!Array.isArray(metadata.validation) || metadata.validation.length === 0) {
    issues.push(
      issue(
        'error',
        phasePath,
        'Metadata validation must be a non-empty list.',
      ),
    );
  } else {
    validateValidationList(metadata.validation, `${phasePath}/validation`);
  }

  if (
    !Array.isArray(metadata.acceptance_criteria) ||
    metadata.acceptance_criteria.length === 0
  ) {
    issues.push(
      issue(
        'error',
        phasePath,
        'Metadata acceptance_criteria must be a non-empty list.',
      ),
    );
  }

  if (
    !Array.isArray(metadata.placeholder_steps) ||
    metadata.placeholder_steps.length === 0
  ) {
    issues.push(
      issue(
        'error',
        phasePath,
        'Metadata placeholder_steps must be a non-empty list.',
      ),
    );
  }

  if (headingStatus !== 'DONE') {
    for (const section of PHASE_REQUIRED_SECTIONS) {
      if (!body.includes(section)) {
        issues.push(
          issue('error', phasePath, `Missing required section: ${section}`),
        );
      }
    }
  }

  pushForbiddenSectionIssues(body, phasePath);

  for (const stepBlock of phaseBlock.stepBlocks) {
    await validateStepPacketInPhase(
      phaseBlock,
      `${phasePath}-step-${String(stepBlock.headingStep).padStart(2, '0')}`,
      stepBlock,
    );
  }
}

async function validateStepPacketInPhase(
  phaseBlock,
  phasePath,
  explicitStepBlock = null,
) {
  const stepBlocks = explicitStepBlock
    ? [explicitStepBlock]
    : phaseBlock.stepBlocks;
  const shouldRequireFullStepSequence =
    phaseBlock.headingStatus !== 'DONE' && !explicitStepBlock;

  if (
    shouldRequireFullStepSequence &&
    phaseBlock.stepBlocks.length > 0 &&
    phaseBlock.stepBlocks[0]?.headingStep !== 1
  ) {
    issues.push(
      issue(
        'error',
        phasePath,
        'Step-based phases must start with Step 01 when active.',
      ),
    );
  }

  const wipSteps = phaseBlock.stepBlocks.filter(
    (stepBlock) => stepBlock.headingStatus === 'WIP',
  ).length;

  if (
    phaseBlock.headingStatus === 'WIP' &&
    wipSteps !== 1 &&
    !explicitStepBlock
  ) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Expected exactly one [WIP] step in active phase ${phaseBlock.headingPhase}, found ${wipSteps}.`,
      ),
    );
  }

  if (
    phaseBlock.headingStatus !== 'WIP' &&
    wipSteps !== 0 &&
    !explicitStepBlock
  ) {
    issues.push(
      issue(
        'error',
        phasePath,
        `Only [WIP] phases may contain [WIP] steps; found ${wipSteps} in phase ${phaseBlock.headingPhase}.`,
      ),
    );
  }

  for (const [stepIndex, stepBlock] of stepBlocks.entries()) {
    const expectedStep = explicitStepBlock
      ? stepBlock.headingStep
      : shouldRequireFullStepSequence
        ? stepIndex + 1
        : stepBlock.headingStep;
    const stepPath = `${phasePath}-step-${String(stepBlock.headingStep).padStart(2, '0')}`;
    const { metadata, body, headingStep, headingTitle, headingStatus } =
      stepBlock;

    if (stepBlock.headingStep !== expectedStep) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Expected step ${String(expectedStep).padStart(2, '0')}, found step ${String(headingStep).padStart(2, '0')}.`,
        ),
      );
    }

    if (metadata === null) {
      if (headingStatus !== 'DONE') {
        issues.push(
          issue(
            'error',
            stepPath,
            'Active or planned steps must keep a YAML metadata block.',
          ),
        );
      }
      continue;
    }

    if (metadata.__parseError) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Could not parse step YAML: ${metadata.__parseError}`,
        ),
      );
      continue;
    }

    if (isLegacyBlock(metadata, headingStatus)) {
      issues.push(issue('error', stepPath, `${MIGRATION_HINT} (step-level)`));
      continue;
    }

    for (const key of STEP_REQUIRED_KEYS) {
      if (
        headingStatus === 'DONE' &&
        (key === 'title' || key === 'acceptance_criteria')
      ) {
        continue;
      }
      if (!Object.hasOwn(metadata, key)) {
        issues.push(
          issue('error', stepPath, `Missing step metadata key: ${key}.`),
        );
      }
    }

    const unknownKeys = Object.keys(metadata).filter(
      (key) =>
        !STEP_REQUIRED_KEYS.includes(key) &&
        !STEP_OPTIONAL_KEYS.has(key) &&
        !PHASE_REQUIRED_KEYS.includes(key),
    );
    for (const key of unknownKeys) {
      issues.push(
        issue('warning', stepPath, `Unexpected metadata key: ${key}.`),
      );
    }

    if (metadata.title !== undefined && metadata.title !== headingTitle) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Metadata title ${String(metadata.title)} does not match heading title.`,
        ),
      );
    }

    const metadataPhase = normalizePhaseLabel(metadata.phase);
    if (metadataPhase !== normalizePhaseLabel(phaseBlock.headingPhase)) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Metadata phase ${String(metadata.phase)} does not match phase heading.`,
        ),
      );
    }

    const metadataStep = Number(metadata.step);
    if (!Number.isNaN(metadataStep) && metadataStep !== headingStep) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Metadata step ${String(metadata.step)} does not match step heading.`,
        ),
      );
    }

    const metadataStatus = stripStatus(metadata.status);
    if (metadataStatus !== headingStatus) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Metadata status ${String(metadata.status)} does not match step heading.`,
        ),
      );
    }

    if (!GOAL_VALUES.has(metadata.goal)) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Invalid goal '${String(metadata.goal)}'. Must be one of: ${[...GOAL_VALUES].join(', ')}.`,
        ),
      );
    }

    if (
      metadata.tdd_sequence !== undefined &&
      !TDD_SEQUENCE_VALUES.has(metadata.tdd_sequence)
    ) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Invalid tdd_sequence '${String(metadata.tdd_sequence)}'.`,
        ),
      );
    }

    if (!['fresh-session', 'perpetual'].includes(metadata.mode)) {
      issues.push(
        issue(
          'error',
          stepPath,
          "Metadata mode must be 'fresh-session' or 'perpetual'.",
        ),
      );
    }

    const normalizedSource = normalizePath(
      String(metadata.source_of_truth ?? ''),
    );
    if (normalizedSource !== normalizedPlanPath) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Metadata source_of_truth must be ${normalizedPlanPath}.`,
        ),
      );
    }

    if (!booleanValue(metadata.copy_paste)) {
      issues.push(
        issue('error', stepPath, 'Metadata copy_paste must be true.'),
      );
    }

    if (!Array.isArray(metadata.skills) || metadata.skills.length === 0) {
      issues.push(
        issue('error', stepPath, 'Metadata skills must be a non-empty list.'),
      );
    }

    if (
      !Array.isArray(metadata.validation) ||
      metadata.validation.length === 0
    ) {
      issues.push(
        issue(
          'error',
          stepPath,
          'Metadata validation must be a non-empty list.',
        ),
      );
    } else {
      validateValidationList(metadata.validation, `${stepPath}/validation`);
    }

    if (
      headingStatus !== 'DONE' &&
      (!Array.isArray(metadata.acceptance_criteria) ||
        metadata.acceptance_criteria.length === 0)
    ) {
      issues.push(
        issue(
          'error',
          stepPath,
          'Metadata acceptance_criteria must be a non-empty list.',
        ),
      );
    }

    const expansion = metadata.expansion;
    if (expansion !== undefined && !EXPANSION_STEP_VALUES.has(expansion)) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Invalid expansion '${String(expansion)}'. Must be 'none' or 'slices'.`,
        ),
      );
    }

    if (expansion === 'slices') {
      if (metadata.auto_expand !== true) {
        issues.push(
          issue(
            'error',
            stepPath,
            "Expansion 'slices' requires auto_expand: true.",
          ),
        );
      }

      if (!metadata.tdd_sequence) {
        issues.push(
          issue(
            'error',
            stepPath,
            "Expansion 'slices' requires a tdd_sequence field.",
          ),
        );
      }

      if (!Array.isArray(metadata.slices) || metadata.slices.length === 0) {
        issues.push(
          issue(
            'error',
            stepPath,
            "Expansion 'slices' requires a non-empty slices list.",
          ),
        );
      } else {
        await validateSlices(metadata.slices, stepPath, metadata.tdd_sequence);
      }
    }

    if (headingStatus !== 'DONE') {
      for (const section of STEP_REQUIRED_SECTIONS) {
        if (!body.includes(section)) {
          issues.push(
            issue('error', stepPath, `Missing required section: ${section}`),
          );
        }
      }
    }

    pushForbiddenSectionIssues(body, stepPath);

    if (metadata.agent_file && !(await fileExists(metadata.agent_file))) {
      issues.push(
        issue(
          'error',
          stepPath,
          `Metadata agent_file does not exist: ${String(metadata.agent_file)}.`,
        ),
      );
    }
  }
}

/**
 * Heuristic check for whether a plan validation entry looks like a repo-relative
 * file path rather than a Jest CLI flag or free-form command.
 *
 * Accepts strings that contain a path separator or a file extension, plus the
 * special relative paths `.` and `..`. Rejects multi-line strings and strings
 * that start with `-` so CLI flags are never treated as paths.
 *
 * @param value - Candidate validation entry.
 * @returns `true` when the entry should be treated as a file path.
 */
export function looksLikeFilePath(value) {
  if (typeof value !== 'string') return false;
  if (value.includes('\n')) return false;
  if (value.startsWith('-')) return false;
  // Treat a plain dot or dot-dot as a path. A slash/backslash or a file
  // extension fragment are good enough heuristic for our plan entries.
  return /[\\/]|\.[^.\s]|^\.{1,2}$/u.test(value);
}

/**
 * Validate a list of plan validation entries.
 *
 * Each entry must be a string. Entries that contain the stale
 * `--testPathPattern` / `--testPathPatterns` CLI flag are rejected as errors,
 * and entries that do not look like a repo-relative file path produce a warning
 * reminding authors to store file paths instead of shell commands.
 *
 * @param validation - Validation strings from a phase or step packet.
 * @param contextPath - Repo-relative path used to annotate any issues.
 * @param issuesList - Optional issue sink; defaults to the module-level issues array.
 * @returns The issue sink that was populated.
 */
export function validateValidationList(
  validation,
  contextPath,
  issuesList = issues,
) {
  for (const [entryIndex, entry] of validation.entries()) {
    const entryPath = `${contextPath}[${entryIndex}]`;
    if (typeof entry !== 'string') {
      issuesList.push(
        issue(
          'error',
          entryPath,
          `Validation entry must be a string, found ${typeof entry}.`,
        ),
      );
      continue;
    }

    if (entry.includes(STALE_TESTPATHPATTERN_FLAG)) {
      issuesList.push(
        issue(
          'error',
          entryPath,
          `Validation entry contains stale ${STALE_TESTPATHPATTERN_FLAG} flag; ${VALIDATION_PATH_HINT}: ${entry}`,
        ),
      );
      continue;
    }

    if (!looksLikeFilePath(entry)) {
      issuesList.push(
        issue(
          'warning',
          entryPath,
          `Validation entry does not look like a file path: ${entry}. ${VALIDATION_PATH_HINT}`,
        ),
      );
    }
  }
  return issuesList;
}

async function validateSlices(slices, stepPath, tddSequence) {
  const expectedFirstGoal =
    tddSequence === 'green-only' ? 'implementing' : 'red-testing';
  const lastSliceIndex = slices.length - 1;

  for (const [sliceIndex, slice] of slices.entries()) {
    const slicePath = `${stepPath}-slice-${sliceIndex}`;
    const sliceId = slice?.slice_id ?? '<missing>';
    const sliceIdPath = `${slicePath}(${sliceId})`;

    const requiredSliceKeys = [
      'slice_id',
      'title',
      'status',
      'goal',
      'estimate_hours',
      'files_to_change',
      'acceptance_criteria',
      'parallelizable',
      'dependencies',
    ];

    for (const key of requiredSliceKeys) {
      if (!Object.hasOwn(slice ?? {}, key)) {
        issues.push(issue('error', sliceIdPath, `Missing slice key: ${key}.`));
      }
    }

    if (slice && !SLICE_GOAL_VALUES.has(slice.goal)) {
      issues.push(
        issue(
          'error',
          sliceIdPath,
          `Invalid slice goal '${String(slice.goal)}'.`,
        ),
      );
    }

    if (slice && !Array.isArray(slice.files_to_change)) {
      issues.push(
        issue('error', sliceIdPath, 'Slice files_to_change must be a list.'),
      );
    }

    if (slice && !Array.isArray(slice.acceptance_criteria)) {
      issues.push(
        issue(
          'error',
          sliceIdPath,
          'Slice acceptance_criteria must be a list.',
        ),
      );
    }

    if (slice && typeof slice.parallelizable !== 'boolean') {
      issues.push(
        issue('error', sliceIdPath, 'Slice parallelizable must be a boolean.'),
      );
    }

    if (slice && !Array.isArray(slice.dependencies)) {
      issues.push(
        issue('error', sliceIdPath, 'Slice dependencies must be a list.'),
      );
    }

    if (slice && typeof slice.estimate_hours !== 'number') {
      issues.push(
        issue('error', sliceIdPath, 'Slice estimate_hours must be a number.'),
      );
    }

    // Enforce the TDD boundary: the first slice starts the right phase and the
    // last slice is green validation. Intermediate slices may decompose the
    // implementation work without breaking the sequence.
    if (sliceIndex === 0 && slice && slice.goal !== expectedFirstGoal) {
      issues.push(
        issue(
          'error',
          sliceIdPath,
          `Expected slice 0 goal to be '${expectedFirstGoal}', found '${String(slice.goal)}'.`,
        ),
      );
    }

    if (
      sliceIndex === lastSliceIndex &&
      slice &&
      slice.goal !== 'green-testing'
    ) {
      issues.push(
        issue(
          'error',
          sliceIdPath,
          `Expected final slice goal to be 'green-testing', found '${String(slice.goal)}'.`,
        ),
      );
    }
  }
}

function isLegacyBlock(metadata, headingStatus) {
  if (headingStatus === 'DONE') return false;
  if (!metadata || typeof metadata !== 'object') return false;
  if (metadata.agent !== undefined || metadata.agent_file !== undefined)
    return true;
  if (metadata.expansion === undefined) return true;
  return false;
}

function booleanValue(value) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') return value === 'true';
  return false;
}

function pushForbiddenSectionIssues(body, sectionPath) {
  for (const section of FORBIDDEN_SECTIONS) {
    if (body.includes(section)) {
      issues.push(
        issue(
          'error',
          sectionPath,
          'Packet must not contain a separate Copy-paste prompt section; the whole packet is the prompt.',
        ),
      );
    }
  }
}

function normalizePhaseLabel(value) {
  return value == null ? null : String(value).trim().toUpperCase();
}

function stripStatus(value) {
  return value?.replace(/^\[/u, '').replace(/\]$/u, '') ?? null;
}

function getNextPhaseLabel(phaseLabel) {
  const normalizedPhaseLabel = normalizePhaseLabel(phaseLabel);
  if (normalizedPhaseLabel === null) return null;

  if (/^\d+$/u.test(normalizedPhaseLabel)) {
    const numericPhase = Number(normalizedPhaseLabel);
    if (numericPhase === 0) return 'A';
    return String(numericPhase + 1);
  }

  if (/^[A-Z]$/u.test(normalizedPhaseLabel)) {
    return normalizedPhaseLabel === 'Z'
      ? null
      : String.fromCharCode(normalizedPhaseLabel.charCodeAt(0) + 1);
  }

  return null;
}
