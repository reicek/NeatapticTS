import { readFile } from 'node:fs/promises';

import {
  parseFrontmatterValue,
} from '../customization-utils.mjs';
import { resolveExplicitPlanPath } from './mcp-utils.mjs';

const PHASE_PATTERN = /^### Phase (?<phase>\d+) — (?<title>.+?) \[(?<status>PLANNED|WIP|DONE)\]\s*$/gmu;
const STEP_PATTERN = /^#### Step (?<step>\d{2})\s*[:\-—]\s*(?<title>.+?) \[(?<status>PLANNED|WIP|DONE)\]\s*$/gmu;
const IMPLEMENTATION_SECTION_PATTERN = /^## Implementation phases\s*(?<body>[\s\S]*?)(?=^## [^\n]*\bvalidation gates\b[^\n]*$)/imu;

/**
 * Load the single active phase and step from the workflow plan.
 *
 * @param {string} planPath - Relative plan path.
 * @returns {Promise<{ planPath: string, activePhase: { number: number, title: string, status: string }, activeStep: { number: number, title: string, status: string, metadata: Record<string, unknown>, stepObjective: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean } }>} Active plan context.
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
 * Build the repo-static workflow snapshot exposed by the workflow MCP.
 *
 * @param {{ planPath: string, activePhase: { number: number, title: string, status: string }, activeStep: { number: number, title: string, status: string, metadata: Record<string, unknown>, stepObjective: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean } }} activePlanContext - Active plan context.
 * @returns {{ scope: string, plan: string, activePhase: { number: number, title: string, status: string }, activeStep: { number: number, title: string, status: string, agent: unknown, agentFile: unknown, objective: string, nextStep: unknown, validationCommands: string[] }, allowlistAuthority: string, sourceBoundary: string[] }} Workflow snapshot.
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
 * @param {{ planPath: string, activePhase: { number: number, title: string, status: string }, activeStep: { number: number, title: string, status: string, metadata: Record<string, unknown>, stepObjective: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean } }} activePlanContext - Active plan context.
 * @returns {{ scope: string, plan: string, activePhase: { number: number, title: string, status: string }, activeStep: { number: number, title: string, status: string, agent: unknown }, allowlistAuthority: string, validationCommands: string[], requiredValidationCommands: string[], validationCommandsMatch: boolean }} Validation allow-list snapshot.
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
    yield {
      number: Number(phaseMatch.groups.phase),
      title: phaseMatch.groups.title,
      status: phaseMatch.groups.status,
      body: implementationSection.slice(phaseBodyStart, phaseBodyEnd),
    };
  }
}

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

function extractRequiredValidationCommands(stepBody) {
  const requiredValidationBody = extractSectionBody(stepBody, 'Required validation');
  return [...requiredValidationBody.matchAll(/`([^`]+)`/gu)].map((match) => match[1]);
}

function extractSectionSummary(stepBody, sectionName) {
  return extractSectionBody(stepBody, sectionName).replace(/\s+/gu, ' ').trim();
}

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

function normalizeCommands(commands) {
  return commands
    .filter((command) => typeof command === 'string')
    .map((command) => command.trim())
    .filter(Boolean);
}

function compareCommands(leftCommands, rightCommands) {
  if (rightCommands.length < leftCommands.length) {
    return false;
  }

  return leftCommands.every((command, index) => command === rightCommands[index]);
}