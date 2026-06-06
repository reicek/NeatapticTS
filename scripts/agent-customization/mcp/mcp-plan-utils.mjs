/**
 * @module mcp-plan-utils
 * @description Markdown plan parser and workflow-snapshot builders for the NeatapticTS MCP servers.
 *
 * Extracts the single active `[WIP]` phase and step from a `plans/*.plans.md`
 * file and converts the structured context into typed snapshot objects that
 * the workflow and validation MCP tools expose to AI agents.
 *
 * @remarks
 * ### Plan Parsing Pipeline
 *
 * ```mermaid
 * flowchart TD
 *   A[Plan markdown text] --> B[extractPhaseBlocks<br/>PHASE_PATTERN]
 *   B --> C{Exactly one WIP phase?}
 *   C -- no  --> D[throw Error]
 *   C -- yes --> E[extractStepBlocks<br/>STEP_PATTERN]
 *   E --> F{Exactly one WIP step?}
 *   F -- no  --> G[throw Error]
 *   F -- yes --> H[parseStepMetadata<br/>YAML fenced block]
 *   H --> I[extractRequiredValidationCommands<br/>prose backtick scan]
 *   I --> J[Active plan context]
 *   J --> K[createWorkflowSnapshot]
 *   J --> L[createValidationAllowlistSnapshot]
 * ```
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { parseFrontmatterValue } from '../customization-utils.mjs';
import {
  MCP_REPO_ROOT,
  requireString,
  resolveExplicitPlanPath,
} from './mcp-utils.mjs';

/** Matches a phase header line, e.g. `### Phase 2 — Title [WIP]` or `### Phase A — Title [WIP]`. */
const PHASE_PATTERN = /^### Phase (?<phase>[A-Z0-9]+) — (?<title>.+?) \[(?<status>PLANNED|WIP|DONE)\]\s*$/gmu;
/** Matches a step header line, e.g. `#### Step 03 — Title [PLANNED]`. */
const STEP_PATTERN = /^#### Step (?<step>\d{2})\s*[:\-—]\s*(?<title>.+?) \[(?<status>PLANNED|WIP|DONE)\]\s*$/gmu;
/** Captures the body of the `## Implementation phases` section up to the first validation-gates heading. */
const IMPLEMENTATION_SECTION_PATTERN = /^## Implementation phases\s*(?<body>[\s\S]*?)(?=^## [^\n]*\bvalidation gates\b[^\n]*$)/imu;
const PLANS_ROOT = path.join(MCP_REPO_ROOT, 'plans');
const SESSION_OVERRIDE_PATH = path.join(MCP_REPO_ROOT, 'data', 'mcp-session-override.json');

/**
 * Load the single active phase and step from the workflow plan.
 *
 * @param {string} planPath - Relative plan path.
 * @returns {Promise<{ planPath: string, activePhase: { number: number | string, title: string, status: string }, activeStep: { number: number, title: string, status: string, metadata: Record<string, unknown>, stepObjective: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean } }>} Active plan context.
 */
export async function loadActivePlanContext(planPath) {
  const resolvedPlanPath = resolveExplicitPlanPath(planPath);
  let planText;

  try {
    planText = await readFile(resolvedPlanPath.absolutePath, 'utf8');
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      throw new Error(`Plan file not found: ${resolvedPlanPath.displayPath}`);
    }

    throw error;
  }

  const phases = [...extractPhaseBlocks(planText)];
  const activePhases = phases.filter((phaseBlock) => phaseBlock.status === 'WIP');

  if (activePhases.length !== 1) {
    throw new Error(`Expected exactly one [WIP] phase in ${planPath}, found ${activePhases.length}.`);
  }

  const activePhase = activePhases[0];
  const steps = [...extractStepBlocks(activePhase.body)];
  const activeSteps = steps.filter((stepBlock) => stepBlock.status === 'WIP');

  if (activeSteps.length !== 1) {
    throw new Error(`Expected exactly one [WIP] step in Phase ${activePhase.number}, found ${activeSteps.length}.`);
  }

  const activeStep = activeSteps[0];
  const metadata = parseStepMetadata(activeStep.body);
  const validationCommands = normalizeCommands(Array.isArray(metadata.validation) ? metadata.validation : []);
  const requiredValidationCommands = normalizeCommands(extractRequiredValidationCommands(activeStep.body));

  return {
    planPath: resolvedPlanPath.displayPath,
    activePhase: {
      number: activePhase.number,
      title: activePhase.title,
      status: activePhase.status,
    },
    activeStep: {
      number: activeStep.number,
      title: activeStep.title,
      status: activeStep.status,
      metadata,
      stepObjective: extractSectionSummary(activeStep.body, 'Step objective'),
      validationCommands,
      requiredValidationCommands,
      validationCommandsMatch: compareCommands(validationCommands, requiredValidationCommands),
    },
  };
}

/**
 * Resolve the effective active plan path using the shared precedence chain.
 *
 * Priority order: (1) per-call `plan_path` argument, (2) session override file
 * at `data/mcp-session-override.json`, (3) startup `planPath`.
 *
 * @param {Record<string, unknown>} argumentsObject - Tool call arguments.
 * @param {string} startupPlanPath - Startup plan path to fall back to.
 * @returns {Promise<string>} Resolved effective plan path.
 */
export async function resolveEffectivePlanPath(argumentsObject, startupPlanPath) {
  if (argumentsObject?.plan_path !== undefined) {
    return resolvePlansScopedPath(argumentsObject.plan_path, 'plan_path');
  }

  const sessionOverridePlanPath = await readSessionOverridePlanPath();
  return sessionOverridePlanPath ?? startupPlanPath;
}

/**
 * Build the repo-static workflow snapshot exposed by the workflow MCP.
 *
 * @param {{ planPath: string, activePhase: { number: number | string, title: string, status: string }, activeStep: { number: number, title: string, status: string, metadata: Record<string, unknown>, stepObjective: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean } }} activePlanContext - Active plan context.
 * @returns {{ scope: string, plan: string, activePhase: { number: number | string, title: string, status: string }, activeStep: { number: number, title: string, status: string, agent: unknown, agentFile: unknown, objective: string, nextStep: unknown, validationCommands: string[] }, allowlistAuthority: string, sourceBoundary: string[] }} Workflow snapshot.
 */
export function createWorkflowSnapshot(activePlanContext) {
  return {
    scope: 'repo-static',
    plan: activePlanContext.planPath,
    activePhase: activePlanContext.activePhase,
    activeStep: {
      number: activePlanContext.activeStep.number,
      title: activePlanContext.activeStep.title,
      status: activePlanContext.activeStep.status,
      agent: activePlanContext.activeStep.metadata.agent ?? null,
      agentFile: activePlanContext.activeStep.metadata.agent_file ?? null,
      objective: activePlanContext.activeStep.stepObjective,
      nextStep: activePlanContext.activeStep.metadata.next_step ?? null,
      validationCommands: activePlanContext.activeStep.validationCommands,
    },
    allowlistAuthority: 'active-step.validation',
    sourceBoundary: [
      'plan metadata',
      'deterministic customization scripts',
    ],
  };
}

/**
 * Build the active validation allow-list snapshot exposed by the validation MCP.
 *
 * @param {{ planPath: string, activePhase: { number: number | string, title: string, status: string }, activeStep: { number: number, title: string, status: string, metadata: Record<string, unknown>, stepObjective: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean } }} activePlanContext - Active plan context.
 * @returns {{ scope: string, plan: string, activePhase: { number: number | string, title: string, status: string }, activeStep: { number: number, title: string, status: string, agent: unknown }, allowlistAuthority: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean }} Validation allow-list snapshot.
 */
export function createValidationAllowlistSnapshot(activePlanContext) {
  return {
    scope: 'direct-MCP',
    plan: activePlanContext.planPath,
    activePhase: activePlanContext.activePhase,
    activeStep: {
      number: activePlanContext.activeStep.number,
      title: activePlanContext.activeStep.title,
      status: activePlanContext.activeStep.status,
      agent: activePlanContext.activeStep.metadata.agent ?? null,
    },
    allowlistAuthority: 'active-step.validation',
    validationCommands: activePlanContext.activeStep.validationCommands,
    requiredValidationCommands: activePlanContext.activeStep.requiredValidationCommands,
    validationCommandsMatch: activePlanContext.activeStep.validationCommandsMatch,
  };
}

/**
 * Yield structured phase descriptors from the implementation-phases section of a plan file.
 *
 * Slices each phase body as the text between one phase heading and the next,
 * so step extraction can operate on a bounded substring without rescanning the
 * full document.
 *
 * @param {string} planText - Full plan file text.
 * @yields {{ number: number | string, title: string, status: string, body: string }} Phase descriptors.
 */
function* extractPhaseBlocks(planText) {
  const implementationSection = IMPLEMENTATION_SECTION_PATTERN.exec(planText)?.groups?.body;
  if (!implementationSection) {
    return;
  }

  const phaseMatches = [...implementationSection.matchAll(PHASE_PATTERN)];
  for (const [phaseIndex, phaseMatch] of phaseMatches.entries()) {
    if (!phaseMatch.groups) {
      continue;
    }

    const phaseBodyStart = (phaseMatch.index ?? 0) + phaseMatch[0].length;
    const nextPhaseMatch = phaseMatches.at(phaseIndex + 1);
    const phaseBodyEnd = nextPhaseMatch?.index ?? implementationSection.length;
    const phaseLabel = phaseMatch.groups.phase;
    yield {
      number: /^\d+$/.test(phaseLabel) ? Number(phaseLabel) : phaseLabel,
      title: phaseMatch.groups.title,
      status: phaseMatch.groups.status,
      body: implementationSection.slice(phaseBodyStart, phaseBodyEnd),
    };
  }
}

/**
 * Yield structured step descriptors from the body of a single phase.
 *
 * @param {string} phaseBody - Phase body text extracted by {@link extractPhaseBlocks}.
 * @yields {{ number: number, title: string, status: string, body: string }} Step descriptors.
 */
function* extractStepBlocks(phaseBody) {
  const stepMatches = [...phaseBody.matchAll(STEP_PATTERN)];
  for (const [stepIndex, stepMatch] of stepMatches.entries()) {
    if (!stepMatch.groups) {
      continue;
    }

    const stepBodyStart = (stepMatch.index ?? 0) + stepMatch[0].length;
    const nextStepMatch = stepMatches.at(stepIndex + 1);
    const stepBodyEnd = nextStepMatch?.index ?? phaseBody.length;
    yield {
      number: Number(stepMatch.groups.step),
      title: stepMatch.groups.title,
      status: stepMatch.groups.status,
      body: phaseBody.slice(stepBodyStart, stepBodyEnd),
    };
  }
}

/**
 * Parse the YAML metadata block from a step body into a key–value map.
 *
 * Looks for a fenced ` ```yaml ` block and walks it line by line. Values are
 * parsed with {@link parseFrontmatterValue} so quoted strings, booleans, and
 * YAML list items are handled correctly. Throws if no YAML block is found,
 * because a missing metadata block is always a plan-authoring error.
 *
 * @param {string} stepBody - Step body text extracted by {@link extractStepBlocks}.
 * @returns {Record<string, unknown>} Parsed metadata map.
 * @throws {Error} When no YAML metadata block is present in the step.
 */
function parseStepMetadata(stepBody) {
  const yamlBlock = /```yaml\r?\n(?<yaml>[\s\S]*?)```/u.exec(stepBody)?.groups?.yaml;
  if (!yamlBlock) {
    throw new Error('Active step packet is missing a YAML metadata block.');
  }

  const metadata = {};
  let activeListKey = null;

  for (const rawLine of yamlBlock.split(/\r?\n/u)) {
    if (!rawLine.trim()) {
      continue;
    }

    const listItemMatch = /^\s+-\s+(?<value>.+)$/u.exec(rawLine);
    if (listItemMatch?.groups && activeListKey) {
      metadata[activeListKey].push(parseFrontmatterValue(listItemMatch.groups.value.trim()));
      continue;
    }

    const keyValueMatch = /^(?<key>[A-Za-z0-9_-]+):(?<value>.*)$/u.exec(rawLine.trimStart());
    if (!keyValueMatch?.groups) {
      continue;
    }

    const key = keyValueMatch.groups.key;
    const value = keyValueMatch.groups.value.trim();
    if (!value) {
      metadata[key] = [];
      activeListKey = key;
      continue;
    }

    metadata[key] = parseFrontmatterValue(value);
    activeListKey = Array.isArray(metadata[key]) ? key : null;
  }

  return metadata;
}

/**
 * Extract validation command strings from the `Required validation` prose section.
 *
 * Scans backtick-delimited code spans in the section body so the validation
 * MCP can compare the structured YAML list against the prose description and
 * detect drift between the two sources of truth.
 *
 * @param {string} stepBody - Step body text.
 * @returns {string[]} Array of required validation command strings.
 */
function extractRequiredValidationCommands(stepBody) {
  const requiredValidationBody = extractSectionBody(stepBody, 'Required validation');
  return [...requiredValidationBody.matchAll(/`([^`]+)`/gu)].map((match) => match[1]);
}

/**
 * Return a single-line summary of a step section by collapsing internal whitespace.
 *
 * @param {string} stepBody - Step body text.
 * @param {string} sectionName - Bold-key section heading to locate (e.g. `Step objective`).
 * @returns {string} Collapsed single-line summary, or empty string when the section is absent.
 */
function extractSectionSummary(stepBody, sectionName) {
  return extractSectionBody(stepBody, sectionName).replace(/\s+/gu, ' ').trim();
}

/**
 * Extract the raw body of a named section from a step body.
 *
 * Sections are delimited by `**Name:**` bold markers. The body extends from
 * the opening marker to the next such marker or end of text.
 *
 * @param {string} stepBody - Step body text.
 * @param {string} sectionName - Section heading without the `**...**:` wrapping.
 * @returns {string} Trimmed section body, or empty string if the section is absent.
 */
function extractSectionBody(stepBody, sectionName) {
  const marker = `**${sectionName}:**`;
  const markerIndex = stepBody.indexOf(marker);
  if (markerIndex === -1) {
    return '';
  }

  const afterMarker = stepBody.slice(markerIndex + marker.length);
  const nextSectionMatch = /\n\*\*[^\n*]+:\*\*/u.exec(afterMarker);
  if (!nextSectionMatch?.index && nextSectionMatch?.index !== 0) {
    return afterMarker.trim();
  }

  return afterMarker.slice(0, nextSectionMatch.index).trim();
}

/**
 * Filter and trim a raw command array, dropping non-strings and blank values.
 *
 * @param {unknown[]} commands - Raw command array from parsed YAML or prose.
 * @returns {string[]} Normalized non-empty command strings.
 */
function normalizeCommands(commands) {
  return commands
    .filter((command) => typeof command === 'string')
    .map((command) => command.trim())
    .filter(Boolean);
}

/**
 * Check whether the left command list is a prefix-match of the right list.
 *
 * Returns `true` when every command in `leftCommands` appears at the same
 * position in `rightCommands`. The right list may have additional trailing
 * commands. Returns `false` when the right list is shorter than the left.
 *
 * @param {string[]} leftCommands - Canonical YAML command list.
 * @param {string[]} rightCommands - Required-validation prose command list.
 * @returns {boolean} `true` when the YAML list is a prefix of the prose list.
 */
function compareCommands(leftCommands, rightCommands) {
  if (rightCommands.length < leftCommands.length) {
    return false;
  }

  return leftCommands.every((command, index) => command === rightCommands[index]);
}

/**
 * Read the active session override plan path from the session override file.
 *
 * Returns `null` when the file is absent, malformed JSON, or does not contain
 * a usable `plan_path` string, so callers can fall back to the startup plan.
 *
 * @returns {Promise<string | null>} Resolved plan path from the session override, or `null`.
 */
async function readSessionOverridePlanPath() {
  try {
    const rawOverride = await readFile(SESSION_OVERRIDE_PATH, 'utf8');
    const overridePayload = JSON.parse(rawOverride);
    if (typeof overridePayload?.plan_path !== 'string' || !overridePayload.plan_path.trim()) {
      return null;
    }

    return resolvePlansScopedPath(overridePayload.plan_path, 'session override plan_path');
  } catch {
    return null;
  }
}

/**
 * Resolve and validate a plan path so it stays within the `plans/` directory.
 *
 * @param {string} candidatePath - Raw plan path from the caller or session override.
 * @param {string} fieldName - Human-readable field name for error messages.
 * @returns {string} Normalized repo-relative plan path within `plans/`.
 */
function resolvePlansScopedPath(candidatePath, fieldName) {
  const requestedPlanPath = requireString(candidatePath, fieldName);
  const absolutePlanPath = path.isAbsolute(requestedPlanPath)
    ? path.normalize(requestedPlanPath)
    : path.resolve(MCP_REPO_ROOT, requestedPlanPath);
  const relativeToPlans = path.relative(PLANS_ROOT, absolutePlanPath);
  const staysWithinPlans = relativeToPlans !== ''
    && !relativeToPlans.startsWith('..')
    && !path.isAbsolute(relativeToPlans);

  if (!staysWithinPlans) {
    const error = new Error(`${fieldName} must resolve within plans/. Received: ${requestedPlanPath}`);
    error.jsonRpcCode = -32602;
    throw error;
  }

  return path.relative(MCP_REPO_ROOT, absolutePlanPath).replaceAll(path.sep, '/');
}