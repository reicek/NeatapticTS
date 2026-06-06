import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { repoRoot } from '../customization-utils.mjs';

export const RUNTIME_CONTEXT_TTL_MS = 60 * 60 * 1000;
export const LEARNING_LOG_PATH = path.join(repoRoot, '.github', 'ai-learning', 'learning-log.jsonl');
export const RUNTIME_CONTEXT_DIR = path.join(repoRoot, 'data');
export const RUNTIME_PROOF_TOOL_PATTERN =
  /^(apply_patch|powershell|task|edit|create|create_file|createFile|editFiles|replace_string_in_file|writeFile|vscode_renameSymbol)$/i;
export const RUNTIME_CONTEXT_PREPARE_PATTERN =
  /runtime-enforcement-context\.mjs\b[\s\S]*--prepare\b/i;

/**
 * Resolve the current session identifier from hook payload or environment.
 *
 * @param {object} [input]
 * @returns {string}
 */
export function resolveSessionId(input = {}) {
  const candidateSessionIds = [
    input?.session_id,
    input?.sessionId,
    process.env.COPILOT_SESSION_ID,
    process.env.GITHUB_COPILOT_SESSION_ID,
    process.env.COPILOT_CLI_SESSION_ID,
    'cli-hook-session',
  ];
  const sessionId = candidateSessionIds.find(
    (candidateValue) => typeof candidateValue === 'string' && candidateValue.trim().length > 0,
  ) ?? 'cli-hook-session';
  return sanitizeSessionId(String(sessionId));
}

/**
 * Determine whether the given tool name requires a strict runtime proof context.
 *
 * @param {string} toolName
 * @returns {boolean}
 */
export function requiresRuntimeProof(toolName) {
  return RUNTIME_PROOF_TOOL_PATTERN.test(String(toolName ?? ''));
}

/**
 * Derive the action class enforced for the given tool.
 *
 * @param {string} toolName
 * @returns {'write'|'execute'|null}
 */
export function inferActionClass(toolName) {
  const normalizedToolName = String(toolName ?? '');
  if (/^(apply_patch|edit|create|create_file|createFile|editFiles|replace_string_in_file|writeFile|vscode_renameSymbol)$/i.test(normalizedToolName)) {
    return 'write';
  }
  if (/^(powershell|task)$/i.test(normalizedToolName)) {
    return 'execute';
  }

  return null;
}

/**
 * Decide whether a powershell payload is preparing a runtime proof context.
 *
 * @param {string} candidateText
 * @returns {boolean}
 */
export function isRuntimeContextPreparation(candidateText) {
  return RUNTIME_CONTEXT_PREPARE_PATTERN.test(String(candidateText ?? ''));
}

/**
 * Return the absolute context file path for the current session.
 *
 * @param {string} sessionId
 * @returns {string}
 */
export function getRuntimeContextPath(sessionId) {
  return path.join(RUNTIME_CONTEXT_DIR, `hook-context-${sanitizeSessionId(sessionId)}.json`);
}

/**
 * Ensure the runtime state directory exists.
 *
 * @returns {Promise<void>}
 */
export async function ensureRuntimeStateDir() {
  await mkdir(RUNTIME_CONTEXT_DIR, { recursive: true });
}

/**
 * Initialize the baseline runtime context carrier for a session.
 *
 * @param {string} sessionId
 * @param {string} [carrierSource]
 * @returns {Promise<object>}
 */
export async function initializeRuntimeContextCarrier(sessionId, carrierSource = 'session-start') {
  const existingCarrier = await readRuntimeContext(sessionId).catch(() => null);
  const initializedCarrier = {
    schemaVersion: 1,
    sessionId,
    sessionStartedAt: existingCarrier?.sessionStartedAt ?? new Date().toISOString(),
    carrierSource,
    carrierExpiresAt: new Date(Date.now() + RUNTIME_CONTEXT_TTL_MS).toISOString(),
    preparedAction: null,
  };

  await writeRuntimeContext(initializedCarrier);
  return initializedCarrier;
}

/**
 * Read the runtime context carrier for the current session.
 *
 * @param {string} sessionId
 * @returns {Promise<object | null>}
 */
export async function readRuntimeContext(sessionId) {
  try {
    const fileText = await readFile(getRuntimeContextPath(sessionId), 'utf8');
    return JSON.parse(fileText);
  } catch {
    return null;
  }
}

/**
 * Persist the runtime context carrier for the current session.
 *
 * @param {object} carrier
 * @returns {Promise<void>}
 */
export async function writeRuntimeContext(carrier) {
  await ensureRuntimeStateDir();
  await writeFile(
    getRuntimeContextPath(carrier.sessionId),
    `${JSON.stringify(carrier, null, 2)}\n`,
    'utf8',
  );
}

/**
 * Prepare a one-shot runtime proof context for the next write/execute action.
 *
 * @param {object} options
 * @returns {Promise<object>}
 */
export async function prepareRuntimeContext(options) {
  const sessionId = resolveSessionId(options);
  const baseCarrier = await initializeRuntimeContextCarrier(sessionId, 'prepare');
  const preparedAt = new Date();
  const preparedAction = {
    actionId: crypto.randomUUID(),
    flowId: requireNonEmptyValue(options.flowId, 'flowId'),
    currentAgent: requireNonEmptyValue(options.currentAgent, 'currentAgent'),
    delegatorChain: normalizeStringArray(options.delegatorChain),
    requiredSkills: normalizeStringArray(options.requiredSkills),
    requiredSpecialists: normalizeStringArray(options.requiredSpecialists),
    planPath: requireNonEmptyValue(options.planPath, 'planPath'),
    activePhase: String(options.activePhase ?? '').trim() || null,
    activeStep: String(options.activeStep ?? '').trim() || null,
    allowedActionClass: requireNonEmptyValue(options.allowedActionClass, 'allowedActionClass'),
    expectedToolName: String(options.expectedToolName ?? '').trim() || null,
    preparedAt: preparedAt.toISOString(),
    expiresAt: new Date(preparedAt.getTime() + RUNTIME_CONTEXT_TTL_MS).toISOString(),
  };

  if (preparedAction.delegatorChain.length === 0) {
    throw new Error('delegatorChain must include at least one agent or orchestrator.');
  }

  baseCarrier.carrierExpiresAt = preparedAction.expiresAt;
  baseCarrier.preparedAction = preparedAction;
  await writeRuntimeContext(baseCarrier);
  return baseCarrier;
}

/**
 * Clear the prepared action from the runtime context carrier after a post-action pass.
 *
 * @param {string} sessionId
 * @param {string | null} actionId
 * @returns {Promise<object | null>}
 */
export async function clearPreparedRuntimeContext(sessionId, actionId = null) {
  const carrier = await readRuntimeContext(sessionId);
  if (!carrier) {
    return null;
  }

  if (
    actionId &&
    carrier.preparedAction?.actionId &&
    carrier.preparedAction.actionId !== actionId
  ) {
    return carrier;
  }

  carrier.carrierExpiresAt = new Date(Date.now() + RUNTIME_CONTEXT_TTL_MS).toISOString();
  carrier.preparedAction = null;
  await writeRuntimeContext(carrier);
  return carrier;
}

/**
 * Validate the currently prepared runtime context against the attempted tool action.
 *
 * @param {object} options
 * @returns {{
 *   ok: boolean,
 *   reason: string,
 *   recoveryHint: string,
 *   actionClass: 'write'|'execute'|null,
 *   preparedAction: object | null,
 * }}
 */
export function validatePreparedRuntimeContext(options) {
  return diagnosePreparedRuntimeContext(options);
}

/**
 * Diagnose the currently prepared runtime context against the attempted tool action.
 *
 * @param {object} options
 * @returns {{
 *   ok: boolean,
 *   reason: string,
 *   recoveryHint: string,
 *   actionClass: 'write'|'execute'|null,
 *   preparedAction: object | null,
 * }}
 */
export function diagnosePreparedRuntimeContext(options) {
  const actionClass = inferActionClass(options.toolName);
  if (!actionClass) {
    return {
      ok: true,
      reason: 'Tool does not require strict runtime proof.',
      recoveryHint: 'No runtime proof preparation is required for this tool.',
      actionClass,
      preparedAction: null,
    };
  }

  const sessionId = resolveSessionId(options);
  const expectedPlanPath = String(options.planPath ?? '').trim() || null;

  if (!options.carrier) {
    return {
      ok: false,
      reason: `Missing runtime enforcement context carrier for session ${sessionId} and strict ${actionClass} action.`,
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction: null,
    };
  }

  const preparedAction = options.carrier.preparedAction;
  if (!preparedAction) {
    return {
      ok: false,
      reason: `Runtime enforcement context carrier exists for session ${sessionId}, but no prepared action is present.`,
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction: null,
    };
  }

  if (!preparedAction.flowId) {
    return {
      ok: false,
      reason: 'Prepared action is missing flowId.',
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (!preparedAction.currentAgent) {
    return {
      ok: false,
      reason: 'Prepared action is missing currentAgent.',
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (!Array.isArray(preparedAction.delegatorChain) || preparedAction.delegatorChain.length === 0) {
    return {
      ok: false,
      reason: 'Prepared action is missing delegatorChain entries.',
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (!preparedAction.planPath) {
    return {
      ok: false,
      reason: 'Prepared action is missing planPath.',
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (!preparedAction.allowedActionClass) {
    return {
      ok: false,
      reason: 'Prepared action is missing allowedActionClass.',
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (Date.parse(preparedAction.expiresAt ?? '') < Date.now()) {
    return {
      ok: false,
      reason: 'Prepared action runtime context has expired.',
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (preparedAction.allowedActionClass !== actionClass) {
    return {
      ok: false,
      reason: `Prepared action allows ${preparedAction.allowedActionClass}, but tool requires ${actionClass}.`,
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (preparedAction.expectedToolName && preparedAction.expectedToolName !== options.toolName) {
    return {
      ok: false,
      reason: `Prepared action expects tool ${preparedAction.expectedToolName}, but observed ${options.toolName}.`,
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  if (expectedPlanPath && normalizeRepoPath(preparedAction.planPath) !== normalizeRepoPath(expectedPlanPath)) {
    return {
      ok: false,
      reason: `Prepared action planPath ${preparedAction.planPath} does not match active plan ${expectedPlanPath}.`,
      recoveryHint: buildRuntimeRecoveryHint({
        sessionId,
        expectedPlanPath,
        toolName: options.toolName,
        actionClass,
      }),
      actionClass,
      preparedAction,
    };
  }

  return {
    ok: true,
    reason: `Runtime enforcement proof validated for ${actionClass} action.`,
    recoveryHint: 'Runtime proof is ready; proceed with the strict action.',
    actionClass,
    preparedAction,
  };
}

/**
 * Append a structured learning event to the repo learning log.
 *
 * @param {object} event
 * @returns {Promise<void>}
 */
export async function appendLearningEvent(event) {
  await appendFile(LEARNING_LOG_PATH, `${JSON.stringify(event)}\n`, 'utf8');
}

/**
 * Record a structured runtime enforcement action event.
 *
 * @param {object} options
 * @returns {Promise<object>}
 */
export async function recordRuntimeActionEvent(options) {
  const event = {
    timestamp: new Date().toISOString(),
    eventType: options.eventType,
    category: 'runtime-enforcement',
    sessionId: resolveSessionId(options),
    actionId: options.actionId ?? null,
    toolName: String(options.toolName ?? '').trim() || null,
    actionClass: String(options.actionClass ?? '').trim() || null,
    flowId: String(options.flowId ?? '').trim() || null,
    agent: String(options.currentAgent ?? options.agent ?? '').trim() || null,
    delegatorChain: normalizeStringArray(options.delegatorChain),
    requiredSkills: normalizeStringArray(options.requiredSkills),
    requiredSpecialists: normalizeStringArray(options.requiredSpecialists),
    planPath: String(options.planPath ?? '').trim() || null,
    activePhase: String(options.activePhase ?? '').trim() || null,
    activeStep: String(options.activeStep ?? '').trim() || null,
    reason: String(options.reason ?? '').trim() || null,
    status: String(options.status ?? '').trim() || null,
  };

  await appendLearningEvent(event);
  return event;
}

/**
 * Build the next-step runtime proof preparation hint for a blocked strict action.
 *
 * @param {object} options
 * @returns {string}
 */
function buildRuntimeRecoveryHint(options) {
  const commandParts = [
    'node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs',
    '--prepare',
    `--session-id=${options.sessionId}`,
  ];

  if (options.expectedPlanPath) {
    commandParts.push(`--plan=${options.expectedPlanPath}`);
  }

  if (options.toolName) {
    commandParts.push(`--tool-name=${options.toolName}`);
  }

  if (options.actionClass) {
    commandParts.push(`--action-class=${options.actionClass}`);
  }

  return `Refresh the runtime proof with \`${commandParts.join(' ')}\` before retrying the strict action.`;
}

/**
 * Load and parse all learning-log JSONL events.
 *
 * @returns {Promise<object[]>}
 */
export async function loadLearningLogEvents() {
  let fileText = '';
  try {
    fileText = await readFile(LEARNING_LOG_PATH, 'utf8');
  } catch {
    return [];
  }

  const parsedEvents = [];
  for (const line of fileText.split('\n')) {
    const trimmedLine = line.trim();
    if (!trimmedLine) {
      continue;
    }

    try {
      parsedEvents.push(JSON.parse(trimmedLine));
    } catch {
      // Ignore malformed lines so auditing remains resilient.
    }
  }

  return parsedEvents;
}

/**
 * Count the trailing run of gate exceptions for the given session.
 *
 * @param {object[]} events
 * @param {string} sessionId
 * @returns {number}
 */
export function countTrailingGateFailures(events, sessionId) {
  const relevantEvents = events
    .filter((event) => event?.sessionId === sessionId || event?.['session-id'] === sessionId)
    .filter((event) => String(event?.gateId ?? event?.['gate-id'] ?? '').trim() || isFailureResetEvent(event))
    .toSorted((leftEvent, rightEvent) =>
      String(leftEvent?.timestamp ?? '').localeCompare(String(rightEvent?.timestamp ?? '')),
    );

  let failureCount = 0;
  for (let index = relevantEvents.length - 1; index >= 0; index -= 1) {
    const event = relevantEvents[index];
    if (event?.eventType === 'gate-exception' || event?.category === 'gate-exception') {
      failureCount += 1;
      continue;
    }
    if (isFailureResetEvent(event)) {
      break;
    }
  }

  return failureCount;
}

/**
 * Determine whether the event resets the consecutive failure streak.
 *
 * @param {object} event
 * @returns {boolean}
 */
export function isFailureResetEvent(event) {
  return (
    event?.eventType === 'runtime-action-prepass' ||
    event?.eventType === 'runtime-action-postpass'
  );
}

/**
 * Normalize repo-relative path strings for stable comparisons.
 *
 * @param {string} repoPath
 * @returns {string}
 */
export function normalizeRepoPath(repoPath) {
  return String(repoPath ?? '').replace(/\\/g, '/').trim();
}

/**
 * Parse a JSON array string or comma-separated list into a stable string array.
 *
 * @param {unknown} value
 * @returns {string[]}
 */
export function normalizeStringArray(value) {
  if (Array.isArray(value)) {
    return value
      .map((arrayEntry) => String(arrayEntry ?? '').trim())
      .filter(Boolean);
  }

  if (typeof value !== 'string') {
    return [];
  }

  const trimmedValue = value.trim();
  if (!trimmedValue) {
    return [];
  }

  if (trimmedValue.startsWith('[') && trimmedValue.endsWith(']')) {
    try {
      const parsedValue = JSON.parse(trimmedValue);
      if (Array.isArray(parsedValue)) {
        return parsedValue
          .map((arrayEntry) => String(arrayEntry ?? '').trim())
          .filter(Boolean);
      }
    } catch {
      // Fall back to comma-splitting when the JSON is malformed.
    }
  }

  return trimmedValue
    .split(',')
    .map((segment) => segment.trim())
    .filter(Boolean);
}

function requireNonEmptyValue(value, fieldName) {
  const normalizedValue = String(value ?? '').trim();
  if (!normalizedValue) {
    throw new Error(`Missing required non-empty ${fieldName}.`);
  }

  return normalizedValue;
}

function sanitizeSessionId(sessionId) {
  return String(sessionId ?? 'cli-hook-session').replace(/[^A-Za-z0-9._-]/g, '_');
}
