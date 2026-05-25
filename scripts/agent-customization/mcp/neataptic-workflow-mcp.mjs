#!/usr/bin/env node
/**
 * @module neataptic-workflow-mcp
 * @description Workflow MCP server — exposes repo-static workflow facts as MCP tools.
 *
 * Reads the active `[WIP]` phase and step from the plan file specified at
 * startup and serves two read-only tools to AI agents:
 * `get_active_workflow_snapshot` and `get_customization_inventory`. All facts
 * are derived from deterministic file-based sources; live host state (selected
 * model, active agent, tool-picker state) is explicitly banned via
 * {@link BANNED_LIVE_FACT_KEYS}.
 *
 * @remarks
 * ### Plan-Path Resolution Chain
 *
 * ```mermaid
 * flowchart TD
 *   A[Tool call] --> B{plan_path arg provided?}
 *   B -- yes --> C[resolvePlansScopedPath<br/>validate within plans/]
 *   B -- no  --> D{SESSION_OVERRIDE_PATH exists?}
 *   D -- yes --> E[readSessionOverridePlanPath<br/>parse JSON override]
 *   E --> F{Valid plan_path in override?}
 *   F -- yes --> G[resolvePlansScopedPath]
 *   F -- no  --> H[Use startup planPath]
 *   D -- no  --> H
 *   C & G & H --> I[loadActivePlanContext<br/>parse WIP phase + step]
 * ```
 */
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { issue } from '../customization-utils.mjs';
import {
  createMcpServer,
  createSelfCheckReport,
  createTool,
  emitSelfCheckReport,
  invokeServerRequest,
  parseMcpCliArgs,
  printMcpUsage,
  requireExplicitPlanPath,
  requireString,
  runShellFreeCommand,
  runStdioMcpServer,
  selfCheckError,
  MCP_PROTOCOL_VERSION,
  MCP_REPO_ROOT,
} from './mcp-utils.mjs';
import {
  createWorkflowSnapshot,
  loadActivePlanContext,
} from './mcp-plan-utils.mjs';

const SERVER_NAME = 'neataptic-workflow-mcp';
const SERVER_VERSION = '0.1.0';
const INVENTORY_COMMAND = 'node scripts/agent-customization/inventory-customizations.mjs --json';
const PLANS_ROOT = path.join(MCP_REPO_ROOT, 'plans');
const SESSION_OVERRIDE_PATH = path.join(MCP_REPO_ROOT, 'data', 'mcp-session-override.json');
/**
 * Keys that must never appear in the workflow snapshot payload.
 *
 * These represent live host-state facts (selected model, active agent, tool-picker
 * state, hook observations) that are only available via a bridge and cannot be
 * reliably served by a file-based MCP server. Exposing them would mislead agents
 * into treating stale or missing data as authoritative live context.
 *
 * The self-check asserts that none of these keys are present in a snapshot
 * returned by `get_active_workflow_snapshot`.
 */
const BANNED_LIVE_FACT_KEYS = [
  'selectedActiveAgent',
  'currentSelectedModel',
  'toolPickerState',
  'liveAgentList',
  'hookObservations',
  'modelSnapshots',
];

const options = parseMcpCliArgs(process.argv.slice(2));

if (options.help) {
  printMcpUsage({
    title: 'Expose repo-static workflow facts as a direct MCP server.',
    entrypoint: 'scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs',
    summary: 'Without flags this script starts a dependency-light stdio MCP server. Use --self-check for a machine-checkable local audit of the current active workflow packet and deterministic customization inventory.',
    tools: createWorkflowTools('plans/help-only.md'),
  });
  process.exit(0);
}

const planPath = requireExplicitPlanPath(options.plan);
const server = createMcpServer({
  serverName: SERVER_NAME,
  serverVersion: SERVER_VERSION,
  tools: createWorkflowTools(planPath),
});

if (options.selfCheck) {
  const report = await runWorkflowSelfCheck({ server, planPath });
  emitSelfCheckReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
} else {
  await runStdioMcpServer(server);
}

/**
 * Build the tool list for the workflow MCP server.
 *
 * @param {string} planPath - Repo-relative plan path used as the startup default.
 * @returns {Array<{ name: string, description: string, inputSchema: Record<string, unknown>, handler: Function }>} Tool list.
 */
function createWorkflowTools(planPath) {
  return [
    createTool({
      name: 'get_active_workflow_snapshot',
      description: 'Return the current repo-static workflow snapshot from the active phase and step packet.',
      annotations: { readOnlyHint: true },
      inputSchema: {
        type: 'object',
        properties: {
          plan_path: {
            type: 'string',
            description: 'Optional repo-relative plan path within plans/ to load for this call only.',
          },
        },
        additionalProperties: false,
      },
      handler: async (argumentsObject) => {
        const effectivePlanPath = await resolveEffectivePlanPath(argumentsObject, planPath);
        try {
          return createWorkflowSnapshot(await loadActivePlanContext(effectivePlanPath));
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          const isNoWipPhase = message.includes('[WIP] phase') || message.includes('[WIP] step');
          if (!isNoWipPhase) {
            throw error;
          }

          // Graceful degradation: plan exists but no [WIP] phase/step is set yet.
          // Return a structured "no-active-phase" snapshot instead of an error so
          // agents can detect session-start state and fall back to direct plan reads
          // rather than spinning on a hard tool failure.
          return {
            scope: 'no-active-phase',
            plan: effectivePlanPath,
            activePhase: null,
            activeStep: null,
            reason: message,
            fallbackAdvice: [
              'No [WIP] phase or step found in the plan.',
              'If starting a new workstream: mark Phase 1 as [WIP] and Step 01 as [WIP] in the plan file, then retry.',
              'If redirecting the session: run plan-session-redirect.mjs --clear to revert to the perpetual binding (plans/mcp-active-binding.plans.md).',
              'Use direct plan file read as the fallback for current phase/step context.',
            ],
          };
        }
      },
    }),
    createTool({
      name: 'get_customization_inventory',
      description: 'Return the deterministic customization inventory generated by the repo-side inventory script.',
      annotations: { readOnlyHint: true },
      handler: async () => await loadCustomizationInventory(),
    }),
  ];
}

/**
 * Run an end-to-end self-check of the workflow MCP server.
 *
 * Validates protocol version, tool count, snapshot correctness, absence of
 * banned live-fact keys, and inventory response shape. Returns a structured
 * report compatible with the standard self-check format.
 *
 * @param {{ server: object, planPath: string }} params - Server instance and plan path.
 * @returns {Promise<Record<string, unknown>>} Self-check report.
 */
async function runWorkflowSelfCheck({ server, planPath }) {
  const issues = [];
  const effectivePlanPath = await resolveEffectivePlanPath({}, planPath);
  const activePlanContext = await loadActivePlanContext(effectivePlanPath);
  const initializeResult = await invokeServerRequest(server, {
    method: 'initialize',
    params: {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: 'self-check', version: SERVER_VERSION },
    },
  });
  const toolListResult = await invokeServerRequest(server, { method: 'tools/list' });
  const workflowSnapshotResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'get_active_workflow_snapshot',
      arguments: {},
    },
  });
  const inventoryResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'get_customization_inventory',
      arguments: {},
    },
  });

  if (initializeResult.protocolVersion !== MCP_PROTOCOL_VERSION) {
    issues.push(selfCheckError(planPath, `Expected protocol version ${MCP_PROTOCOL_VERSION}, received ${String(initializeResult.protocolVersion)}.`));
  }

  if (!Array.isArray(toolListResult.tools) || toolListResult.tools.length !== 2) {
    issues.push(selfCheckError(planPath, `Expected 2 workflow tools, found ${Array.isArray(toolListResult.tools) ? toolListResult.tools.length : 'none'}.`));
  }

  if (workflowSnapshotResult.isError) {
    issues.push(selfCheckError(planPath, 'Workflow snapshot tool returned an error during self-check.'));
  }

  const workflowSnapshot = workflowSnapshotResult.structuredContent ?? {};
  if (workflowSnapshot.activePhase?.number !== activePlanContext.activePhase.number) {
    issues.push(selfCheckError(planPath, 'Workflow snapshot phase did not match the active plan phase.'));
  }

  if (workflowSnapshot.activeStep?.number !== activePlanContext.activeStep.number) {
    issues.push(selfCheckError(planPath, 'Workflow snapshot step did not match the active plan step.'));
  }

  if (BANNED_LIVE_FACT_KEYS.some((key) => key in workflowSnapshot)) {
    issues.push(selfCheckError(planPath, 'Workflow snapshot exposed a bridge-required or manual-only fact key.'));
  }

  if (inventoryResult.isError) {
    issues.push(selfCheckError(planPath, 'Customization inventory tool returned an error during self-check.'));
  }

  const inventory = inventoryResult.structuredContent ?? {};
  if (typeof inventory.summary?.agents !== 'number' || typeof inventory.summary?.skills !== 'number') {
    issues.push(selfCheckError(planPath, 'Customization inventory did not include the expected summary counts.'));
  }

  return createSelfCheckReport('neataptic-workflow-mcp self-check', issues, {
    server: { name: SERVER_NAME, version: SERVER_VERSION },
    plan: effectivePlanPath,
    toolNames: server.tools.map((tool) => tool.name),
    snapshot: {
      phase: workflowSnapshot.activePhase?.number ?? null,
      step: workflowSnapshot.activeStep?.number ?? null,
      agent: workflowSnapshot.activeStep?.agent ?? null,
      validationCommandCount: Array.isArray(workflowSnapshot.activeStep?.validationCommands)
        ? workflowSnapshot.activeStep.validationCommands.length
        : 0,
    },
    inventorySummary: inventory.summary ?? null,
  });
}

/**
 * Load the deterministic customization inventory by running the inventory script.
 *
 * Executes `node scripts/agent-customization/inventory-customizations.mjs --json`
 * via {@link runShellFreeCommand} and parses the JSON output. Throws on non-zero
 * exit or invalid JSON so callers receive a clear error rather than silent data loss.
 *
 * @returns {Promise<Record<string, unknown>>} Parsed inventory payload.
 * @throws {Error} When the command fails or its output is not valid JSON.
 */
async function loadCustomizationInventory() {
  const commandResult = await runShellFreeCommand(INVENTORY_COMMAND, { maxOutputBytes: 200_000 });
  if (commandResult.exitCode !== 0) {
    throw new Error(`Customization inventory command failed with exit code ${commandResult.exitCode}.`);
  }

  try {
    return JSON.parse(commandResult.stdout);
  } catch {
    throw new Error('Customization inventory command did not return valid JSON.');
  }
}

/**
 * Determine the effective plan path for a tool call.
 *
 * Priority order: (1) per-call `plan_path` argument, (2) session override file
 * at `data/mcp-session-override.json`, (3) startup `planPath`.
 *
 * @param {Record<string, unknown>} argumentsObject - Tool call arguments.
 * @param {string} startupPlanPath - Startup plan path to fall back to.
 * @returns {Promise<string>} Resolved effective plan path.
 */
async function resolveEffectivePlanPath(argumentsObject, startupPlanPath) {
  if (argumentsObject?.plan_path !== undefined) {
    return resolvePlansScopedPath(argumentsObject.plan_path, 'plan_path');
  }

  const sessionOverridePlanPath = await readSessionOverridePlanPath();
  return sessionOverridePlanPath ?? startupPlanPath;
}

/**
 * Read the active session override plan path from the session override file.
 *
 * Returns `null` when the file is absent, malformed JSON, or does not contain
 * a usable `plan_path` string, so the caller falls back to the startup plan path.
 *
 * @returns {Promise<string | null>} Resolved plan path from the session override, or `null`.
 */
async function readSessionOverridePlanPath() {
  if (!existsSync(SESSION_OVERRIDE_PATH)) {
    return null;
  }

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
 * Throws a JSON-RPC -32602 invalid-params error when the resolved path escapes
 * `plans/`, preventing callers from using the workflow tool to read arbitrary
 * filesystem paths through plan-path injection.
 *
 * @param {string} candidatePath - Raw plan path from the caller or session override.
 * @param {string} fieldName - Human-readable field name for error messages.
 * @returns {string} Normalized repo-relative plan path within `plans/`.
 * @throws {Error} When the path resolves outside `plans/` (JSON-RPC error code -32602).
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
    throw invalidParamsError(`${fieldName} must resolve within plans/. Received: ${requestedPlanPath}`);
  }

  return normalizePath(path.relative(MCP_REPO_ROOT, absolutePlanPath));
}

function invalidParamsError(message) {
  const error = new Error(message);
  error.jsonRpcCode = -32602;
  return error;
}

function normalizePath(value) {
  return value.replaceAll(path.sep, '/');
}
