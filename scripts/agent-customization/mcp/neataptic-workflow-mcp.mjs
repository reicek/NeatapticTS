#!/usr/bin/env node
/**
 * @module neataptic-workflow-mcp
 * @description Workflow MCP server — exposes repo-static workflow facts as MCP tools.
 *
 * Reads the active `[WIP]` phase and step from the plan file specified at
 * startup and serves three read-only tools to AI agents:
 * `get_active_workflow_snapshot`, `get_customization_inventory`, and
 * `get_slice_context`. All facts are derived from deterministic file-based
 * sources; live host state (selected model, active agent, tool-picker state) is
 * explicitly banned via {@link BANNED_LIVE_FACT_KEYS}.
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
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parsePlanYamlBlock } from '../customization-utils.mjs';
import {
  createMcpServer,
  createSelfCheckReport,
  createTool,
  emitSelfCheckReport,
  formatToolResult,
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
  resolveEffectivePlanPath,
} from './mcp-plan-utils.mjs';

/* global Buffer */

const SERVER_NAME = 'neataptic-workflow-mcp';
const SERVER_VERSION = '0.1.0';
const INVENTORY_COMMAND =
  'node scripts/agent-customization/inventory-customizations.mjs --json';
/** Maximum byte size for a `get_slice_context` payload (instructions + RAG context). */
const COMPACT_RESPONSE_LIMIT_BYTES = 16_384;
/**
 * Overhead factor applied to the wire-format envelope before measuring against
 * {@link COMPACT_RESPONSE_LIMIT_BYTES}. JSON/string serialization adds bytes
 * for quotes, newlines, escaping, and the MCP `content` wrapper. Using a 15%
 * headroom margin keeps the measured payload under the documented cap even
 * after wire formatting.
 */
const WIRE_OVERHEAD_FACTOR = 1.025;
/** Effective byte budget used to trim the structured payload before formatting. */
const ENFORCEMENT_BUDGET = Math.floor(
  COMPACT_RESPONSE_LIMIT_BYTES / WIRE_OVERHEAD_FACTOR,
);
/** Token budget for the Cortex `search_context` call embedded in `get_slice_context`. */
const CONTEXT_TOKEN_BUDGET = 2_500;
/** Maximum number of corpus chunks to retain in the `context.chunks` array. */
const CONTEXT_RESULT_LIMIT = 10;
/** Score bonus applied to `*.test.ts` chunks in TDD slices so they outrank implementation chunks. */
const TEST_FILE_PRIORITY_BONUS = 1_000_000;
/** Reserved token budget for the dedicated `*.test.ts` search in TDD slices. */
const TEST_FILE_RESERVED_BUDGET = 500;
/** Maximum number of acceptance-criterion queries to issue beyond the primary query. */
const MAX_ACCEPTANCE_CRITERIA_QUERIES = 5;
/** Soft byte cap for the synthesized `instructions` string. */
const INSTRUCTIONS_SOFT_LIMIT_BYTES = 2_048;
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

/**
 * Lazy-load the Cortex `search_context` implementation.
 *
 * Kept as a dynamic import so the workflow MCP server remains dependency-light
 * at startup. The heavy embedding/ONNX modules are only loaded when a
 * `get_slice_context` call actually needs live corpus search.
 */
/* istanbul ignore next */
async function loadSearchContext() {
  const { searchContext } =
    await import('../../mcp-semantic/tools/search-context.mjs');
  return searchContext;
}

/** Captures the body of the `## Implementation phases` section up to the first validation-gates heading. */
const IMPLEMENTATION_SECTION_PATTERN =
  /^## Implementation phases\s*(?<body>[\s\S]*?)(?=^## [^\n]*\bvalidation gates\b[^\n]*$)/imu;
/** Matches a step header line, e.g. `#### Step 03 — Title [PLANNED]` or `#### Step B1 — Title [PLANNED]`. */
const STEP_PATTERN =
  /^#### Step (?<step>[A-Z0-9]+(?:\.\d+)?)\s*[:\-—]\s*(?<title>.+?) \[(?<status>PLANNED|WIP|DONE)\]\s*$/gmu;

/**
 * Build the tool list for the workflow MCP server.
 *
 * @param {{ planPath?: string, searchContextFn?: (options: Record<string, unknown>) => Promise<Record<string, unknown>>, inventoryCommandRunner?: (command: string, options: Record<string, unknown>) => Promise<{ exitCode: number, stdout: string, stderr?: string }> }} [options={}] - Startup options.
 * @param {string} [options.planPath] - Repo-relative plan path used as the startup default.
 * @param {Function} [options.searchContextFn] - Injectable Cortex search_context implementation for testing.
 * @param {Function} [options.inventoryCommandRunner] - Injectable shell runner for the customization inventory command.
 * @returns {Array<{ name: string, description: string, inputSchema: Record<string, unknown>, handler: Function }>} Tool list.
 */
export function createWorkflowTools({
  planPath,
  searchContextFn = async (options) => (await loadSearchContext())(options),
  inventoryCommandRunner,
} = {}) {
  const loadInventory = inventoryCommandRunner
    ? () => loadCustomizationInventory(inventoryCommandRunner)
    : loadCustomizationInventory;

  return [
    createTool({
      name: 'get_active_workflow_snapshot',
      description:
        'Return the current repo-static workflow snapshot from the active phase and step packet.',
      annotations: { readOnlyHint: true },
      inputSchema: {
        type: 'object',
        properties: {
          plan_path: {
            type: 'string',
            description:
              'Optional repo-relative plan path within plans/ to load for this call only.',
          },
        },
        additionalProperties: false,
      },
      handler: async (argumentsObject) => {
        const effectivePlanPath = await resolveEffectivePlanPath(
          argumentsObject,
          planPath,
        );
        try {
          return createWorkflowSnapshot(
            await loadActivePlanContext(effectivePlanPath),
          );
        } catch (error) {
          const message = String(error);
          const isNoWipPhase =
            message.includes('[WIP] phase') || message.includes('[WIP] step');
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
      description:
        'Return the deterministic customization inventory generated by the repo-side inventory script.',
      annotations: { readOnlyHint: true },
      handler: async () => await loadInventory(),
    }),
    createTool({
      name: 'get_slice_context',
      description:
        'Return a deterministic, self-contained context window for a slice_id from the active plan step plus Cortex RAG context. The response contains the full dispatch packet an agent needs in one call: slice boundary fields, step-level skills/validation/tdd metadata, a synthesized `instructions` directive string, and a bounded `context` block with the most relevant corpus chunks pulled via search_context. Total payload is capped at ~16 KB so an agent can read it directly without loading additional files. To retrieve the full plan document, use neataptic-cortex-mcp:load_document instead.',
      annotations: { readOnlyHint: true },
      inputSchema: {
        type: 'object',
        properties: {
          slice_id: {
            type: 'string',
            description:
              'Unique slice identifier (exact slice_id) or the symbolic step label (e.g. B1) for the active step.',
          },
          plan_path: {
            type: 'string',
            description:
              'Optional repo-relative plan path within plans/ to load for this call only.',
          },
        },
        required: ['slice_id'],
        additionalProperties: false,
      },
      handler: async (argumentsObject) => {
        return buildSliceContextWindow(
          argumentsObject,
          planPath,
          searchContextFn,
        );
      },
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
export async function runWorkflowSelfCheck({ server, planPath }) {
  const issues = [];
  const effectivePlanPath = await resolveEffectivePlanPath(
    { plan_path: planPath },
    planPath,
  );
  const activePlanContext = await loadActivePlanContext(effectivePlanPath);
  const initializeResult = await invokeServerRequest(server, {
    method: 'initialize',
    params: {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: 'self-check', version: SERVER_VERSION },
    },
  });
  const toolListResult = await invokeServerRequest(server, {
    method: 'tools/list',
  });
  const workflowSnapshotResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'get_active_workflow_snapshot',
      arguments: { plan_path: effectivePlanPath },
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
    issues.push(
      selfCheckError(
        planPath,
        `Expected protocol version ${MCP_PROTOCOL_VERSION}, received ${String(initializeResult.protocolVersion)}.`,
      ),
    );
  }

  if (
    !Array.isArray(toolListResult.tools) ||
    toolListResult.tools.length !== 3
  ) {
    issues.push(
      selfCheckError(
        planPath,
        `Expected 3 workflow tools, found ${Array.isArray(toolListResult.tools) ? toolListResult.tools.length : 'none'}.`,
      ),
    );
  }

  if (workflowSnapshotResult.isError) {
    issues.push(
      selfCheckError(
        planPath,
        'Workflow snapshot tool returned an error during self-check.',
      ),
    );
  }

  const workflowSnapshot = workflowSnapshotResult.structuredContent ?? {};
  const expectedPhase = activePlanContext.activePhase;
  const expectedStep = activePlanContext.activeStep;

  if (!expectedPhase || !expectedStep) {
    issues.push(
      selfCheckError(
        planPath,
        'No active WIP phase or step found in the active plan.',
      ),
    );
  }

  if (
    expectedPhase &&
    workflowSnapshot.activePhase?.number !== expectedPhase.number
  ) {
    issues.push(
      selfCheckError(
        planPath,
        'Workflow snapshot phase did not match the active plan phase.',
      ),
    );
  }

  if (
    expectedStep &&
    workflowSnapshot.activeStep?.number !== expectedStep.number
  ) {
    issues.push(
      selfCheckError(
        planPath,
        'Workflow snapshot step did not match the active plan step.',
      ),
    );
  }

  if (BANNED_LIVE_FACT_KEYS.some((key) => key in workflowSnapshot)) {
    issues.push(
      selfCheckError(
        planPath,
        'Workflow snapshot exposed a bridge-required or manual-only fact key.',
      ),
    );
  }

  if (inventoryResult.isError) {
    issues.push(
      selfCheckError(
        planPath,
        'Customization inventory tool returned an error during self-check.',
      ),
    );
  }

  const inventory = inventoryResult.structuredContent ?? {};
  if (
    typeof inventory.summary?.agents !== 'number' ||
    typeof inventory.summary?.skills !== 'number'
  ) {
    issues.push(
      selfCheckError(
        planPath,
        'Customization inventory did not include the expected summary counts.',
      ),
    );
  }

  return createSelfCheckReport('neataptic-workflow-mcp self-check', issues, {
    server: { name: SERVER_NAME, version: SERVER_VERSION },
    plan: effectivePlanPath,
    toolNames: server.tools.map((tool) => tool.name),
    snapshot: {
      phase: workflowSnapshot.activePhase?.number ?? null,
      step: workflowSnapshot.activeStep?.number ?? null,
      agent: workflowSnapshot.activeStep?.agent ?? null,
      validationCommandCount: Array.isArray(
        workflowSnapshot.activeStep?.validationCommands,
      )
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
async function loadCustomizationInventory(commandRunner = runShellFreeCommand) {
  const commandResult = await commandRunner(INVENTORY_COMMAND, {
    maxOutputBytes: 200_000,
  });
  if (commandResult.exitCode !== 0) {
    throw new Error(
      `Customization inventory command failed with exit code ${commandResult.exitCode}.`,
    );
  }

  try {
    return JSON.parse(commandResult.stdout);
  } catch {
    throw new Error(
      'Customization inventory command did not return valid JSON.',
    );
  }
}

/**
 * Assemble a deterministic, self-contained context window for a slice_id.
 *
 * Loads the active plan, resolves the slice descriptor, pulls the most relevant
 * corpus chunks via Cortex `search_context`, and returns a bounded payload
 * containing the full dispatch packet (boundary fields, step-level metadata,
 * synthesized `instructions`, and a `context` block) so an agent can act on a
 * single call without loading additional files. Use
 * neataptic-cortex-mcp:load_document to retrieve the full plan document.
 *
 * @param {Record<string, unknown>} argumentsObject - Tool call arguments.
 * @param {string | undefined} startupPlanPath - Startup plan path fallback.
 * @param {Function} [searchContextFn] - Injectable Cortex search_context implementation.
 * @returns {Promise<Record<string, unknown>>} JSON-RPC tool result with `structuredContent`.
 */
async function buildSliceContextWindow(
  argumentsObject,
  startupPlanPath,
  searchContextFn,
) {
  const sliceId = requireString(argumentsObject.slice_id, 'slice_id');
  const effectivePlanPath = await resolveEffectivePlanPath(
    argumentsObject,
    startupPlanPath,
  );

  let activePlanContext;
  try {
    activePlanContext = await loadActivePlanContext(effectivePlanPath);
  } catch {
    activePlanContext = null;
  }

  const descriptor = activePlanContext
    ? await findSliceDescriptor(activePlanContext, effectivePlanPath, sliceId)
    : null;

  if (!descriptor) {
    return {
      compact: true,
      notFound: true,
      slice_id: sliceId,
      plan: effectivePlanPath,
      message: `No step packet found for slice_id '${sliceId}' in ${effectivePlanPath}. Use neataptic-cortex-mcp:load_document to retrieve the full plan context.`,
    };
  }

  const ragContext = await fetchSliceContext(
    descriptor,
    sliceId,
    searchContextFn,
    effectivePlanPath,
  );

  return buildCompactSliceResponse(
    descriptor,
    sliceId,
    effectivePlanPath,
    ragContext,
  );
}

/**
 * Build a self-contained slice dispatch packet for `get_slice_context`.
 *
 * The response contains the full set of fields an agent needs to act on a
 * single call: slice boundary fields, step-level skills/validation/tdd
 * metadata, a synthesized `instructions` directive string, and a bounded
 * `context` block with the most relevant corpus chunks. The total payload is
 * capped at {@link COMPACT_RESPONSE_LIMIT_BYTES}; when it would exceed the
 * limit, the `context.text` is truncated first, then `instructions`, and a
 * `truncated` flag plus `fallback_message` are set.
 *
 * @param {Record<string, unknown>} descriptor - Resolved slice descriptor.
 * @param {string} sliceId - Exact slice identifier requested.
 * @param {string} planPath - Repo-relative plan path used for the lookup.
 * @param {{ query?: string, text?: string, chunks?: Array<Record<string, unknown>>, token_count?: number, dense_state?: string, truncated?: boolean, follow_up_refs?: Array<Record<string, unknown>> }} [ragContext] - RAG context from Cortex.
 * @returns {Record<string, unknown>} Self-contained slice context window.
 */
function buildCompactSliceResponse(descriptor, sliceId, planPath, ragContext) {
  /** @type {Record<string, unknown>} */
  const notes = descriptor.boundaryNotes || {};
  const stepMeta = descriptor.stepMetadata || {};
  const contracts = Array.isArray(descriptor.testContracts)
    ? descriptor.testContracts
    : [];
  const instructions = buildSliceInstructions(descriptor, sliceId);
  const rag = ragContext || {};

  const basePayload = {
    compact: true,
    slice_id: sliceId,
    plan: planPath,
    phase: notes.phase ?? null,
    phase_status: notes.phase_status ?? null,
    phase_title: notes.phase_title ?? null,
    step_number: descriptor.stepNumber,
    step_title: String(stepMeta.title ?? ''),
    step_status: notes.step_status ?? null,
    title: notes.title ?? descriptor.sliceTitle ?? sliceId,
    status: notes.status ?? 'unknown',
    goal: notes.goal ?? null,
    estimate_hours: notes.estimate_hours ?? null,
    parallelizable: notes.parallelizable ?? null,
    tdd_sequence: stepMeta.tdd_sequence ?? null,
    mode: stepMeta.mode ?? null,
    skills: Array.isArray(stepMeta.skills) ? stepMeta.skills : [],
    validation: Array.isArray(stepMeta.validation) ? stepMeta.validation : [],
    files_to_change: Array.isArray(notes.files_to_change)
      ? notes.files_to_change
      : [],
    acceptance_criteria: contracts.map((contract) => ({
      id: contract.id ?? '',
      text: contract.text ?? '',
      validation: contract.validation ?? null,
    })),
    dependencies: Array.isArray(notes.dependencies) ? notes.dependencies : [],
    next_slice: notes.next_slice ?? null,
    next_step: stepMeta.next_step ?? null,
    slice_history: Array.isArray(notes.slice_history)
      ? notes.slice_history
      : [],
    instructions,
    context: {
      query: rag.query ?? null,
      text: rag.text ?? '',
      chunks: Array.isArray(rag.chunks) ? rag.chunks : [],
      token_count: rag.token_count ?? 0,
      dense_state: rag.dense_state ?? null,
      truncated: rag.truncated === true,
      follow_up_refs: Array.isArray(rag.follow_up_refs)
        ? rag.follow_up_refs
        : [],
    },
  };

  const serialized = JSON.stringify(basePayload);
  if (Buffer.byteLength(serialized, 'utf8') <= ENFORCEMENT_BUDGET) {
    return basePayload;
  }

  // Truncate the context text first, then instructions, to stay under the
  // enforcement budget. The budget is intentionally smaller than the documented
  // 16 KB cap to leave headroom for the MCP envelope (quotes, escaping, and the
  // `content` wrapper) after JSON serialization.
  const contextText = String(basePayload.context.text ?? '');
  const trimmedContext = truncateToBudget(
    basePayload,
    'context.text',
    contextText,
    ENFORCEMENT_BUDGET,
  );
  if (trimmedContext.fits) {
    return trimmedContext.payload;
  }

  const trimmedInstructions = truncateToBudget(
    trimmedContext.payload,
    'instructions',
    instructions,
    ENFORCEMENT_BUDGET,
  );
  if (trimmedInstructions.fits) {
    return {
      ...trimmedInstructions.payload,
      truncated: true,
      fallback_message:
        'Slice context exceeded the 16 KB limit; context.text and instructions were truncated. Use neataptic-cortex-mcp:load_document for the full plan or load_chunk for complete chunks.',
    };
  }

  return {
    compact: true,
    truncated: true,
    fallback_message:
      'Slice context exceeded the 16 KB limit and could not be trimmed in place. Use neataptic-cortex-mcp:load_document to retrieve the full plan context.',
    slice_id: sliceId,
    plan: planPath,
    step_number: descriptor.stepNumber,
    title: basePayload.title,
    status: basePayload.status,
    goal: truncateStringToBytes(String(basePayload.goal ?? ''), 256),
    instructions: trimmedInstructions.value,
  };
}

/**
 * Synthesize a concise, actionable `instructions` directive for the agent.
 *
 * Combines the slice goal, title, status, acceptance criteria, validation
 * commands, skills, tdd sequence, and handoff into a single string the agent
 * can read first to understand exactly what to do.
 *
 * @param {Record<string, unknown>} descriptor - Resolved slice descriptor.
 * @param {string} sliceId - Exact slice identifier.
 * @returns {string} Synthesized instructions directive.
 */
function buildSliceInstructions(descriptor, sliceId) {
  const notes = descriptor.boundaryNotes || {};
  const stepMeta = descriptor.stepMetadata || {};
  const contracts = Array.isArray(descriptor.testContracts)
    ? descriptor.testContracts
    : [];
  const phase = notes.phase != null ? `Phase ${notes.phase} ` : '';
  const stepPart =
    descriptor.stepNumber != null ? `Step ${descriptor.stepNumber} ` : '';
  const lines = [];
  const displayTitle =
    notes.title ??
    descriptor.sliceTitle ??
    String(stepMeta.title ?? '') ??
    sliceId;
  lines.push(`SLICE: ${sliceId} — ${displayTitle}`);
  lines.push(
    `${phase}${stepPart}| STATUS: ${notes.status ?? 'unknown'} | GOAL: ${notes.goal ?? 'unspecified'}`,
  );
  if (notes.phase_status) {
    lines.push(`PHASE STATUS: ${notes.phase_status}`);
  }
  if (stepMeta.tdd_sequence) {
    lines.push(
      `TDD: ${stepMeta.tdd_sequence}${stepMeta.mode ? ` | MODE: ${stepMeta.mode}` : ''}`,
    );
  }
  if (contracts.length > 0) {
    lines.push('ACCEPTANCE CRITERIA:');
    for (const contract of contracts) {
      const idPart = contract.id ? `${contract.id}: ` : '';
      lines.push(`- ${idPart}${contract.text ?? ''}`);
      if (contract.validation) {
        lines.push(`  Validation: ${contract.validation}`);
      }
    }
  }
  const files = Array.isArray(notes.files_to_change)
    ? notes.files_to_change
    : [];
  if (files.length > 0) {
    lines.push(`FILES TO CHANGE (${files.length}): ${files.join(', ')}`);
  }
  const validation = Array.isArray(stepMeta.validation)
    ? stepMeta.validation
    : [];
  if (validation.length > 0) {
    lines.push(`VALIDATION: ${validation.join(' ; ')}`);
  }
  const skills = Array.isArray(stepMeta.skills) ? stepMeta.skills : [];
  if (skills.length > 0) {
    lines.push(`SKILLS: ${skills.join(', ')}`);
  }
  const deps = Array.isArray(notes.dependencies) ? notes.dependencies : [];
  if (deps.length > 0) {
    lines.push(`DEPENDENCIES: ${deps.join(', ')}`);
  }
  const sliceHistory = Array.isArray(notes.slice_history)
    ? notes.slice_history
    : [];
  if (sliceHistory.length > 0) {
    const doneCount = sliceHistory.filter(
      (s) => String(s.status).toUpperCase() === '[DONE]',
    ).length;
    const wipCount = sliceHistory.filter(
      (s) => String(s.status).toUpperCase() === '[WIP]',
    ).length;
    lines.push(
      `SLICE HISTORY: ${sliceHistory.length} slices (${doneCount} DONE, ${wipCount} WIP)`,
    );
    const summary = sliceHistory
      .slice(0, 10)
      .map((s) => `${s.slice_id}:${s.status}`)
      .join(', ');
    lines.push(`  ${summary}${sliceHistory.length > 10 ? '…' : ''}`);
  }
  if (notes.next_slice) {
    lines.push(`NEXT SLICE: ${notes.next_slice}`);
  }
  if (stepMeta.next_step) {
    lines.push(`NEXT STEP: ${stepMeta.next_step}`);
  }
  if (
    String(notes.status).toUpperCase() === '[WIP]' ||
    String(notes.step_status).toUpperCase() === '[WIP]'
  ) {
    lines.push(
      'STOP: This step is [WIP]. Complete your assigned task and STOP for orchestrator review. Do not advance to the next step autonomously.',
    );
  }
  const instructions = lines.join('\n');
  if (Buffer.byteLength(instructions, 'utf8') > INSTRUCTIONS_SOFT_LIMIT_BYTES) {
    return `${instructions.slice(0, INSTRUCTIONS_SOFT_LIMIT_BYTES - 1)}…`;
  }
  return instructions;
}

/**
 * Fetch the most relevant corpus context for a slice via Cortex `search_context`.
 *
 * Builds a query from the slice title, goal, acceptance criteria, and files to
 * change, then calls the injectable `searchContextFn`. Normalizes the response
 * into a bounded `{ query, text, chunks, token_count, dense_state, truncated,
 * follow_up_refs }` shape. When the search function is unavailable or returns
 * no usable results, an empty (but well-formed) context block is returned so
 * the dispatch packet remains self-contained.
 *
 * @param {Record<string, unknown>} descriptor - Resolved slice descriptor.
 * @param {string} sliceId - Exact slice identifier.
 * @param {Function} [searchContextFn] - Injectable Cortex search_context implementation.
 * @returns {Promise<{ query: string, text: string, chunks: Array<Record<string, unknown>>, token_count: number, dense_state: string | null, truncated: boolean, follow_up_refs: Array<Record<string, unknown>> }>} RAG context block.
 */
async function fetchSliceContext(
  descriptor,
  sliceId,
  searchContextFn,
  planPath,
) {
  const {
    query: primaryQuery,
    queries,
    metadata,
  } = buildSliceQuery(descriptor, sliceId, planPath);
  const notes = descriptor.boundaryNotes || {};
  const testFiles = getTestFilePaths(descriptor);
  const isTdd = isTddSlice(descriptor);

  if (typeof searchContextFn !== 'function') {
    return emptyRagContext(primaryQuery);
  }

  const baseOptions = {
    limit: CONTEXT_RESULT_LIMIT,
    budget: CONTEXT_TOKEN_BUDGET,
    include_metadata: true,
    context_format: 'json',
    expand_query: true,
    use_rerank: true,
    metadata,
  };

  /** @type {Array<Record<string, unknown>>} */
  const responses = [];
  try {
    for (const query of queries) {
      responses.push(await searchContextFn({ ...baseOptions, query }));
    }
  } catch {
    return emptyRagContext(primaryQuery);
  }

  /** @type {Record<string, unknown> | null} */
  let testResponse = null;
  if (isTdd && testFiles.length > 0) {
    const testQuery = testFiles.map((p) => p.split('/').pop()).join(' ');
    try {
      testResponse = await searchContextFn({
        ...baseOptions,
        query: testQuery,
        budget: TEST_FILE_RESERVED_BUDGET,
        limit: Math.max(testFiles.length, CONTEXT_RESULT_LIMIT),
      });
    } catch {
      testResponse = null;
    }
  }

  /** @type {Map<string | number, Record<string, unknown>>} */
  const resultById = new Map();
  /** @type {Map<string | number, Record<string, unknown>>} */
  const assembledById = new Map();
  let denseState = null;
  let tokenCount = 0;
  let truncated = false;
  /** @type {Record<string, unknown> | null} */
  let searchRef = null;

  /**
   * Ingest a single `search_context` response, merging raw results and any
   * assembled chunks into the shared maps.
   *
   * @param {Record<string, unknown>} response - A single search response.
   */
  function ingestResponse(response) {
    if (!response || typeof response !== 'object') {
      return;
    }
    if (typeof response.dense_state === 'string') {
      denseState = response.dense_state;
    }
    tokenCount +=
      typeof response.token_count === 'number' ? response.token_count : 0;
    truncated = truncated || response.truncated === true;

    const rawResults = Array.isArray(response.results) ? response.results : [];
    for (const result of rawResults) {
      const id =
        result.chunk_id ??
        `${String(result.file_path ?? result.metadata?.file_path ?? 'unknown')}:${String(result.char_start ?? result.metadata?.char_start ?? 'none')}`;
      const score = Number(result.score ?? 0);
      const existing = resultById.get(id);
      if (!existing || score > (existing.score ?? 0)) {
        resultById.set(id, { ...result, _score: score });
      }
    }

    const assembled =
      response.context && typeof response.context === 'object'
        ? response.context
        : null;
    const chunks = Array.isArray(assembled?.chunks) ? assembled.chunks : [];
    for (const chunk of chunks) {
      const id =
        chunk.chunk_id ??
        `${String(chunk.file_path ?? 'unknown')}:${String(chunk.char_start ?? 'none')}`;
      if (!assembledById.has(id)) {
        assembledById.set(id, chunk);
      }
    }

    if (!searchRef && Array.isArray(response.follow_up_refs)) {
      const ref = response.follow_up_refs.find(
        (r) => r?.tool === 'search_context' && typeof r.reason === 'string',
      );
      if (ref) {
        searchRef = ref;
      }
    }
  }

  for (const response of responses) {
    ingestResponse(response);
  }
  if (testResponse) {
    ingestResponse(testResponse);
  }

  /** @type {Array<Record<string, unknown>>} */
  let mergedResults = Array.from(resultById.values());

  // Drop chunks that point at the same file and overlapping or identical ranges.
  mergedResults = deduplicateLocationChunks(mergedResults);

  // In TDD slices, elevate test file chunks so they survive the top-N cut.
  if (isTdd && testFiles.length > 0) {
    mergedResults = boostTestFileChunks(mergedResults, testFiles);
  }

  // Sort by effective score and keep the strongest chunks up to the limit.
  mergedResults.sort((a, b) => (b._score ?? 0) - (a._score ?? 0));

  /**
   * Normalize a raw search result into a context chunk, back-filling body text
   * from any assembled chunk returned by Cortex.
   *
   * @param {Record<string, unknown>} result - Raw search result.
   * @returns {Record<string, unknown>} Normalized chunk.
   */
  function mapResultToChunk(result) {
    const resultId =
      result.chunk_id ??
      `${String(result.file_path ?? result.metadata?.file_path ?? 'unknown')}:${String(result.char_start ?? result.metadata?.char_start ?? 'none')}`;
    const fallbackChunk = assembledById.get(resultId);
    const resultPath =
      result.file_path ??
      result.metadata?.file_path ??
      result.path ??
      fallbackChunk?.file_path ??
      null;
    const resultHeading =
      result.heading_path ||
      result.metadata?.heading_path ||
      result.metadata?.context_header ||
      fallbackChunk?.heading_path ||
      fallbackChunk?.context_header ||
      null;
    const start =
      result.char_start ??
      result.metadata?.char_start ??
      fallbackChunk?.char_start ??
      null;
    const end =
      result.char_end ??
      result.metadata?.char_end ??
      fallbackChunk?.char_end ??
      null;
    // Prefer the assembled chunk content (which has full body text) over the
    // metadata-only result fields.
    const rawText =
      result.text ??
      result.content ??
      result.snippet ??
      fallbackChunk?.content ??
      fallbackChunk?.text ??
      '';
    // Drop chunks that were truncated by budget enforcement — a partial chunk
    // is misleading. The agent can use follow_up_refs to load the full chunk.
    const isTruncated =
      result.truncated === true || fallbackChunk?.truncated === true;
    return {
      chunk_id: result.chunk_id ?? null,
      file_path: resultPath,
      heading_path: resultHeading,
      char_start: start,
      char_end: end,
      text: isTruncated ? '' : rawText,
    };
  }

  const allMapped = mergedResults.map(mapResultToChunk);
  const allChunks = allMapped.filter(
    (chunk) => typeof chunk.text === 'string' && chunk.text.trim().length > 0,
  );
  const droppedChunks = allMapped.filter(
    (chunk) =>
      chunk.chunk_id != null &&
      !(typeof chunk.text === 'string' && chunk.text.trim().length > 0),
  );
  const chunks = allChunks.slice(0, CONTEXT_RESULT_LIMIT);

  // Detect partial-file chunks: a chunk may cover only part of its source file
  // (char_end < file length). Reading the file is the only reliable way to
  // detect this — the search engine's `truncated` flag only marks chunks cut
  // by the token budget, not chunks that are simply the first slice.
  const repoRoot = process.cwd();
  for (const chunk of chunks) {
    if (
      chunk.file_path &&
      typeof chunk.char_end === 'number' &&
      chunk.char_start === 0
    ) {
      try {
        const fullPath = path.resolve(repoRoot, chunk.file_path);
        const content = await readFile(fullPath, 'utf8');
        const fileChars = content.length;
        if (chunk.char_end < fileChars) {
          chunk.partial_file = true;
          chunk.file_chars = fileChars;
        }
      } catch {
        // File not found or not readable — skip
      }
    }
  }

  // Build follow-up refs.
  const followUpRefs = [];
  // Add refs for dropped (truncated) chunks so the agent can retrieve them.
  for (const chunk of droppedChunks.slice(0, 3)) {
    if (chunk.chunk_id != null) {
      followUpRefs.push({
        tool: 'load_chunk',
        args: { chunk_id: chunk.chunk_id, query: primaryQuery },
        reason: `Full text of ${chunk.file_path ?? 'chunk'} (truncated by budget, not included in context)`,
      });
    }
  }
  // In TDD slices, explicitly retrieve any required test files still missing.
  if (isTdd && testFiles.length > 0) {
    followUpRefs.push(
      ...buildMissingTestFileRefs(chunks, droppedChunks, testFiles),
    );
  }

  const chunkPaths = new Set(chunks.map((c) => c.file_path).filter(Boolean));
  const droppedPaths = new Set(
    droppedChunks.map((c) => c.file_path).filter(Boolean),
  );
  const allFiles = Array.isArray(notes.files_to_change)
    ? notes.files_to_change
    : Array.isArray(descriptor.files_to_change)
      ? descriptor.files_to_change
      : [];
  const missingFiles = allFiles
    .map((entry) => (typeof entry === 'string' ? entry : entry?.path))
    .filter(
      (p) =>
        typeof p === 'string' &&
        p.length > 0 &&
        !p.endsWith('.test.ts') &&
        !chunkPaths.has(p) &&
        !droppedPaths.has(p),
    );
  if (missingFiles.length > 0) {
    followUpRefs.push({
      tool: 'search_context',
      args: {
        query: missingFiles.map((p) => p.split('/').pop()).join(' '),
        limit: 5,
      },
      reason: `${missingFiles.length} file(s) from files_to_change not in context: ${missingFiles.map((p) => p.split('/').pop()).join(', ')}`,
    });
  }

  // Add refs for partial-file chunks so the agent can retrieve the rest.
  const partialFiles = chunks.filter((c) => c.partial_file === true);
  for (const chunk of partialFiles.slice(0, 3)) {
    const basename = (chunk.file_path ?? 'unknown').split('/').pop();
    followUpRefs.push({
      tool: 'search_context',
      args: { query: basename, limit: 3 },
      reason: `${basename} is partial in context (chunk covers chars 0-${chunk.char_end} of ${chunk.file_chars}); load remaining content`,
    });
  }
  if (searchRef) {
    followUpRefs.push(searchRef);
  }

  /**
   * Pick a fallback text summary from the first response that carries one.
   *
   * @returns {string} Fallback context text.
   */
  function firstResponseContextText() {
    for (const response of [testResponse, ...responses]) {
      if (!response || typeof response !== 'object') {
        continue;
      }
      if (typeof response.context === 'string' && response.context.length > 0) {
        return response.context;
      }
      const assembled =
        response.context && typeof response.context === 'object'
          ? response.context
          : null;
      if (
        typeof assembled?.context === 'string' &&
        assembled.context.length > 0
      ) {
        return assembled.context;
      }
    }
    return '';
  }

  // Prefer the full assembled context text from the search response — it
  // contains the actual plan/source content an agent needs. Fall back to a
  // stitched chunk summary only if no assembled context was returned.
  const fullContextText = firstResponseContextText();
  const text =
    fullContextText.length > 0
      ? fullContextText
      : chunks
          .map((chunk) => {
            const fullText = chunk.text || '';
            const basename = (chunk.file_path ?? 'unknown').split('/').pop();
            const partialNote =
              chunk.partial_file === true
                ? ` [PARTIAL: chars 0-${chunk.char_end} of ${chunk.file_chars}]`
                : '';
            return `${basename} — ${fullText.split(/\r?\n/).find((l) => l.trim().length > 0) ?? ''}${partialNote}`;
          })
          .join('\n');

  return {
    query: primaryQuery,
    text,
    chunks: chunks.map(({ text: _text, ...meta }) => meta),
    token_count: tokenCount,
    dense_state: denseState,
    truncated,
    follow_up_refs: followUpRefs,
  };
}

/**
 * Build a file-path metadata filter that constrains RAG results to chunks
 * whose `file_path` matches one of the slice's `files_to_change` entries.
 *
 * @param {Array<string | { path?: string }>} files - Files to change.
 * @returns {Record<string, unknown> | undefined} Filter object, or undefined if no valid paths.
 */
function buildFilePathFilter(files) {
  const paths = files
    .map((entry) => (typeof entry === 'string' ? entry : entry?.path))
    .filter((p) => typeof p === 'string' && p.length > 0)
    .map((p) => p.replace(/\\/g, '/'));
  if (paths.length === 0) {
    return undefined;
  }

  const predicates = paths.map((p) => ({
    op: 'like',
    field: 'file_path',
    value: `%${p}%`,
  }));
  return paths.length === 1 ? predicates[0] : { op: 'or', predicates };
}

/**
 * Build a file_path LIKE predicate that scopes the RAG search to the active
 * plan file and its companion logs file. For a plan path like
 * `plans/Neon_Shooter_NGE_Demo.plans.md`, this produces a LIKE filter on
 * `Neon_Shooter_NGE_Demo` which matches both `.plans.md` and `.logs.md`.
 *
 * @param {string} [planPath] - Repo-relative path to the active plan file.
 * @returns {Record<string, unknown> | undefined} Predicate tree or undefined.
 */
function buildPlanFilePathFilter(planPath) {
  if (typeof planPath !== 'string' || planPath.trim().length === 0) {
    return undefined;
  }
  const normalized = planPath.replace(/\\/g, '/');
  // Extract the base name without the `.plans.md` suffix so the LIKE
  // predicate also matches the companion `.logs.md` file.
  const baseName = normalized
    .split('/')
    .pop()
    .replace(/\.plans\.md$/i, '');
  if (baseName.length === 0) {
    return undefined;
  }
  return {
    op: 'like',
    field: 'file_path',
    value: `%${baseName}%`,
  };
}

/**
 * Stop words removed from synthesized RAG queries.
 */
const STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'has',
  'in',
  'is',
  'it',
  'its',
  'of',
  'on',
  'or',
  'that',
  'the',
  'to',
  'with',
  'all',
  'any',
  'into',
  'their',
  'then',
  'they',
  'this',
  'will',
  'with',
]);

/**
 * Extract the most useful tokens from a raw text fragment for use as a
 * `search_context` query. Removes punctuation, stop words, and very short
 * tokens, then deduplicates and caps the token count.
 *
 * @param {string} rawText - Source text (title, goal, acceptance criterion, etc.).
 * @param {number} [maxTokens=3] - Maximum number of query tokens to retain.
 * @returns {Array<string>} Cleaned, ordered query tokens.
 */
function extractQueryTokens(rawText, maxTokens = 3) {
  if (typeof rawText !== 'string' || rawText.length === 0) {
    return [];
  }
  return rawText
    .toLowerCase()
    .replace(/[^a-z0-9_\-/\s]/g, ' ')
    .split(/\s+/)
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token))
    .filter((token, index, arr) => arr.indexOf(token) === index)
    .slice(0, maxTokens);
}

/**
 * Determine whether a slice is part of a TDD (red-green-refactor) sequence.
 *
 * @param {Record<string, unknown>} descriptor - Resolved slice descriptor.
 * @returns {boolean} True when the slice metadata indicates a TDD red phase.
 */
function isTddSlice(descriptor) {
  const notes = descriptor.boundaryNotes || {};
  const stepMeta = descriptor.stepMetadata || {};
  const sequence =
    stepMeta.tdd_sequence ?? notes.tdd_sequence ?? descriptor.tdd_sequence;
  return typeof sequence === 'string' && sequence.toLowerCase().includes('red');
}

/**
 * Resolve the list of `*.test.ts` paths from a slice descriptor.
 *
 * @param {Record<string, unknown>} descriptor - Resolved slice descriptor.
 * @returns {Array<string>} Test file paths from `files_to_change`.
 */
function getTestFilePaths(descriptor) {
  const notes = descriptor.boundaryNotes || {};
  const files = Array.isArray(notes.files_to_change)
    ? notes.files_to_change
    : Array.isArray(descriptor.files_to_change)
      ? descriptor.files_to_change
      : [];
  return files
    .map((entry) => (typeof entry === 'string' ? entry : entry?.path))
    .filter(
      (p) => typeof p === 'string' && p.length > 0 && p.endsWith('.test.ts'),
    )
    .map((p) => p.replace(/\\/g, '/'));
}

/**
 * Deduplicate corpus chunks that point at the same file and overlapping or
 * identical character ranges. Keeps the broadest surviving chunk per starting
 * offset; chunks that are fully contained within an already-kept chunk are
 * dropped.
 *
 * @param {Array<Record<string, unknown>>} chunks - Raw corpus chunks.
 * @returns {Array<Record<string, unknown>>} Deduplicated chunks.
 */
function deduplicateLocationChunks(chunks) {
  /** @type {Map<string, Array<Record<string, unknown>>>} */
  const grouped = new Map();
  /** @type {Array<Record<string, unknown>>} */
  const withoutPath = [];
  for (const chunk of chunks) {
    const filePath =
      chunk.file_path ??
      chunk.metadata?.file_path ??
      chunk.path ??
      chunk.metadata?.path;
    if (typeof filePath !== 'string' || filePath.length === 0) {
      withoutPath.push(chunk);
      continue;
    }
    const key = filePath.replace(/\\/g, '/');
    const list = grouped.get(key) ?? [];
    list.push(chunk);
    grouped.set(key, list);
  }

  /** @type {Array<Record<string, unknown>>} */
  const deduped = [];
  for (const [, list] of grouped) {
    const scored = list.map((chunk) => {
      const start = Number(chunk.char_start ?? chunk.metadata?.char_start ?? 0);
      const rawEnd = chunk.char_end ?? chunk.metadata?.char_end;
      const end =
        typeof rawEnd === 'number' && !Number.isNaN(rawEnd)
          ? rawEnd
          : Number.MAX_SAFE_INTEGER;
      const score = Number(chunk.score ?? chunk._score ?? 0);
      return { chunk, start, end, score };
    });
    // Broadest ranges first for a given start offset, then by score.
    scored.sort(
      (a, b) => a.start - b.start || b.end - a.end || b.score - a.score,
    );
    /** @type {Array<{ start: number, end: number, chunk: Record<string, unknown> }>} */
    const kept = [];
    for (const item of scored) {
      const contained = kept.some(
        (k) => k.start <= item.start && k.end >= item.end,
      );
      if (!contained) {
        kept.push(item);
      }
    }
    deduped.push(...kept.map((item) => item.chunk));
  }

  return [...deduped, ...withoutPath];
}

/**
 * Boost the score of every chunk whose file path matches a test file.
 *
 * @param {Array<Record<string, unknown>>} chunks - Raw corpus chunks.
 * @param {Array<string>} testFiles - Paths ending in `.test.ts`.
 * @param {number} [bonus=TEST_FILE_PRIORITY_BONUS] - Score bonus.
 * @returns {Array<Record<string, unknown>>} Chunks with adjusted scores.
 */
function boostTestFileChunks(
  chunks,
  testFiles,
  bonus = TEST_FILE_PRIORITY_BONUS,
) {
  if (testFiles.length === 0) {
    return chunks;
  }
  const testPaths = new Set(testFiles.map((p) => p.replace(/\\/g, '/')));
  return chunks.map((chunk) => {
    const filePath =
      chunk.file_path ??
      chunk.metadata?.file_path ??
      chunk.path ??
      chunk.metadata?.path;
    if (typeof filePath !== 'string' || filePath.length === 0) {
      return chunk;
    }
    if (!testPaths.has(filePath.replace(/\\/g, '/'))) {
      return chunk;
    }
    const baseScore = Number(chunk.score ?? chunk._score ?? 0);
    return { ...chunk, score: baseScore + bonus, _score: baseScore + bonus };
  });
}

/**
 * Build targeted `load_chunk` or `search_context` follow-up refs for test files
 * that are listed in `files_to_change` but have no chunk present in the context.
 *
 * @param {Array<Record<string, unknown>>} contextChunks - Chunks kept in the context window.
 * @param {Array<Record<string, unknown>>} droppedChunks - Chunks that were dropped (empty/truncated) during assembly.
 * @param {Array<string>} testFiles - Required test file paths.
 * @returns {Array<Record<string, unknown>>} Follow-up refs for missing test files.
 */
function buildMissingTestFileRefs(contextChunks, droppedChunks, testFiles) {
  if (testFiles.length === 0) {
    return [];
  }
  const presentPaths = new Set(
    contextChunks
      .map((c) => c.file_path)
      .filter((p) => typeof p === 'string' && p.length > 0)
      .map((p) => p.replace(/\\/g, '/')),
  );
  const refs = [];
  for (const testFile of testFiles) {
    const normalized = testFile.replace(/\\/g, '/');
    if (presentPaths.has(normalized)) {
      continue;
    }
    const dropped = droppedChunks.find(
      (c) =>
        typeof c.file_path === 'string' &&
        c.file_path.replace(/\\/g, '/') === normalized &&
        c.chunk_id != null,
    );
    if (dropped) {
      refs.push({
        tool: 'load_chunk',
        args: { chunk_id: dropped.chunk_id },
        reason: `Required test file ${normalized} is not in context; load chunk ${dropped.chunk_id}`,
      });
    } else {
      const basename = normalized.split('/').pop() ?? normalized;
      refs.push({
        tool: 'search_context',
        args: { query: basename, limit: 3 },
        reason: `Required test file ${normalized} has no chunks in context; run targeted search`,
      });
    }
  }
  return refs;
}

/**
 * Build a Cortex search query and optional metadata filter from the slice descriptor.
 *
 * The query blends semantic slice metadata (title, goal, contracts) with exact
 * path tokens derived from `files_to_change` so the search targets both the
 * intent and the concrete files in the slice boundary. A `metadata.filter`
 * constrains results to the same file paths when paths are available.
 *
 * @param {Record<string, unknown>} descriptor - Resolved slice descriptor.
 * @param {string} sliceId - Exact slice identifier.
 * @returns {{ query: string, queries: Array<string>, metadata?: { filter: Record<string, unknown> } }} Search options.
 */
function buildSliceQuery(descriptor, _sliceId, planPath) {
  const notes = descriptor.boundaryNotes || {};
  const files = Array.isArray(notes.files_to_change)
    ? notes.files_to_change
    : Array.isArray(descriptor.files_to_change)
      ? descriptor.files_to_change
      : [];

  const stepMeta = descriptor.stepMetadata || {};
  const rawSemantic = [
    notes.title ?? descriptor.sliceTitle ?? '',
    stepMeta.title ?? '',
    notes.goal ?? '',
  ].join(' ');

  const primaryTokens = extractQueryTokens(rawSemantic, 3);
  const primaryQuery = primaryTokens.join(' ').slice(0, 256);

  const contracts = Array.isArray(descriptor.testContracts)
    ? descriptor.testContracts
    : Array.isArray(notes.acceptance_criteria)
      ? notes.acceptance_criteria
      : Array.isArray(descriptor.acceptance_criteria)
        ? descriptor.acceptance_criteria
        : [];
  const acQueries = contracts
    .slice(0, MAX_ACCEPTANCE_CRITERIA_QUERIES)
    .map((contract) => {
      const raw =
        typeof contract === 'string'
          ? contract
          : (contract?.text ?? contract?.description ?? '');
      return extractQueryTokens(raw, 3).join(' ').slice(0, 256);
    })
    .filter((q) => q.length > 0 && q !== primaryQuery);

  const queries = [primaryQuery, ...acQueries];

  // Build a combined file_path filter that scopes the RAG search to the
  // active plan file, its corresponding logs file, and any declared
  // files_to_change. This prevents the search from returning generic
  // chunks from unrelated plans or skill docs.
  const planFilter = buildPlanFilePathFilter(planPath);
  const filesFilter = buildFilePathFilter(files);
  let filter;
  if (planFilter && filesFilter) {
    filter = { op: 'or', predicates: [planFilter, filesFilter] };
  } else {
    filter = planFilter ?? filesFilter;
  }
  const metadata = filter ? { filter } : undefined;
  return { query: primaryQuery, queries, metadata };
}

/**
 * Return an empty (well-formed) RAG context block.
 *
 * @param {string} query - The query that was attempted.
 * @returns {{ query: string, text: string, chunks: Array, token_count: number, dense_state: null, truncated: boolean, follow_up_refs: Array }} Empty RAG context.
 */
function emptyRagContext(query) {
  return {
    query,
    text: '',
    chunks: [],
    token_count: 0,
    dense_state: null,
    truncated: false,
    follow_up_refs: [],
  };
}

/**
 * Truncate a string to a maximum byte length (UTF-8 safe, ellipsis-appended).
 *
 * @param {string} value - String to truncate.
 * @param {number} maxBytes - Maximum byte length.
 * @returns {string} Truncated string.
 */
function truncateStringToBytes(value, maxBytes) {
  if (Buffer.byteLength(value, 'utf8') <= maxBytes) {
    return value;
  }
  let truncated = value;
  while (
    Buffer.byteLength(truncated, 'utf8') > maxBytes - 1 &&
    truncated.length > 0
  ) {
    truncated = truncated.slice(0, Math.max(0, truncated.length - 128));
  }
  return `${truncated}…`;
}

/**
 * Truncate a single string field within a payload to fit the byte budget.
 *
 * @param {Record<string, unknown>} payload - Payload to mutate.
 * @param {string} fieldPath - Dotted field path (e.g. `context.text` or `instructions`).
 * @param {string} currentValue - Current value of the field.
 * @param {number} budget - Total byte budget for the serialized payload.
 * @returns {{ payload: Record<string, unknown>, value: string, fits: boolean }} Trimmed payload and whether it fits the budget.
 */
function truncateToBudget(payload, fieldPath, currentValue, budget) {
  const pathParts = fieldPath.split('.');
  let parent = payload;
  for (let i = 0; i < pathParts.length - 1; i += 1) {
    parent = parent[pathParts[i]];
  }
  const leafKey = pathParts[pathParts.length - 1];

  let trimmed = currentValue;
  parent[leafKey] = trimmed;
  while (
    Buffer.byteLength(JSON.stringify(payload), 'utf8') > budget &&
    trimmed.length > 0
  ) {
    trimmed = trimmed.slice(0, Math.max(0, trimmed.length - 256));
    parent[leafKey] = trimmed;
  }
  const fits = Buffer.byteLength(JSON.stringify(payload), 'utf8') <= budget;
  return { payload, value: trimmed, fits };
}

/**
 * Derive a step label from a slice_id.
 *
 * Exact slice identifiers such as `B1-tool-impl` belong to the step labelled
 * `B1`. A standalone label such as `B1` is returned unchanged.
 *
 * @param {string} sliceId - Requested slice identifier.
 * @returns {string} Step label, or empty string when no label can be derived.
 */
function deriveStepLabel(sliceId) {
  const dashIndex = sliceId.indexOf('-');
  return dashIndex > 0 ? sliceId.slice(0, dashIndex) : sliceId;
}

/**
 * Normalize a step number string to a numeric value when possible.
 *
 * @param {number | string} stepNumber - Raw step number or label.
 * @returns {number | string} Numeric value for purely numeric labels, original otherwise.
 */
function normalizeStepNumber(stepNumber) {
  if (typeof stepNumber === 'number') {
    return stepNumber;
  }
  if (typeof stepNumber === 'string' && /^\d+$/u.test(stepNumber)) {
    return Number(stepNumber);
  }
  return stepNumber;
}

/**
 * Scan every step packet in the plan for an exact slice_id match.
 *
 * @param {string} planText - Full plan markdown text.
 * @param {string} sliceId - Requested slice identifier.
 * @param {Awaited<ReturnType<typeof loadActivePlanContext>>} activePlanContext - Active plan context.
 * @returns {Record<string, unknown> | null} Slice descriptor, or null when not found.
 */
async function findSliceDescriptorAcrossSteps(
  planText,
  sliceId,
  activePlanContext,
) {
  const implementationSection = IMPLEMENTATION_SECTION_PATTERN.exec(planText);
  const text = implementationSection.groups.body;
  const matches = [...text.matchAll(STEP_PATTERN)];

  for (const stepMatch of matches) {
    const start = stepMatch.index + stepMatch[0].length;
    const nextMatch = matches.find(
      (candidate) => candidate.index > stepMatch.index,
    );
    const end = nextMatch ? nextMatch.index : text.length;
    const stepPacket = text.slice(start, end).trim();
    const metadata = parseStepPacketMetadata(stepPacket);
    const slices = Array.isArray(metadata.slices) ? metadata.slices : [];
    const slice = slices.find(
      (candidateSlice) => String(candidateSlice.slice_id ?? '') === sliceId,
    );

    if (slice) {
      const stepNumber = normalizeStepNumber(stepMatch.groups.step);
      return buildDescriptorFromSlice(
        slice,
        stepPacket,
        stepNumber,
        activePlanContext,
      );
    }
  }

  return null;
}

/**
 * Locate the step packet and slice metadata for a slice_id in the active plan.
 *
 * Matches exact slice_ids first, then falls back to the step's symbolic
 * label (e.g. `B1`) so callers can request a whole step packet by step label
 * even when the active step parser only supplies a phase-level context.
 *
 * @param {Awaited<ReturnType<typeof loadActivePlanContext>>} activePlanContext - Active plan context.
 * @param {string} planPath - Repo-relative plan path.
 * @param {string} sliceId - Requested slice identifier.
 * @returns {Promise<Record<string, unknown> | null>} Slice descriptor, or null when not found.
 */
async function findSliceDescriptor(activePlanContext, planPath, sliceId) {
  const absolutePlanPath = path.resolve(MCP_REPO_ROOT, planPath);
  const planText = await readFile(absolutePlanPath, 'utf8');

  const activeStepNumber = activePlanContext.activeStep?.number ?? null;
  const activeStepTitle = activePlanContext.activeStep?.title ?? '';
  const stepLabel = deriveStepLabel(sliceId);

  let stepPacket = null;
  let matchedStepNumber = null;

  if (
    activeStepNumber != null &&
    (String(activeStepNumber) === sliceId ||
      String(activeStepNumber) === stepLabel ||
      titleMatchesStepId(activeStepTitle, sliceId) ||
      titleMatchesStepId(activeStepTitle, stepLabel))
  ) {
    stepPacket = extractStepPacketText(planText, activeStepNumber);
    matchedStepNumber = activeStepNumber;
  }

  if (stepPacket == null) {
    for (const label of new Set([sliceId, stepLabel])) {
      const candidate = extractStepPacketText(planText, label);
      if (candidate != null) {
        stepPacket = candidate;
        matchedStepNumber = label;
        break;
      }
    }
  }

  if (stepPacket == null) {
    const scanned = await findSliceDescriptorAcrossSteps(
      planText,
      sliceId,
      activePlanContext,
    );
    return scanned;
  }

  const metadata = parseStepPacketMetadata(stepPacket);
  const slices = Array.isArray(metadata.slices) ? metadata.slices : [];
  const slice = slices.find(
    (candidateSlice) => String(candidateSlice.slice_id ?? '') === sliceId,
  );

  if (slice) {
    return buildDescriptorFromSlice(
      slice,
      stepPacket,
      normalizeStepNumber(matchedStepNumber),
      activePlanContext,
    );
  }

  if (
    String(matchedStepNumber) === sliceId ||
    String(matchedStepNumber) === stepLabel ||
    titleMatchesStepId(String(metadata.title ?? ''), sliceId) ||
    titleMatchesStepId(String(metadata.title ?? ''), stepLabel)
  ) {
    return buildDescriptorFromStep(
      metadata,
      stepPacket,
      normalizeStepNumber(matchedStepNumber),
      activePlanContext,
    );
  }

  return null;
}

/**
 * Extract the raw step packet text for a given step number or symbolic label.
 *
 * @param {string} planText - Full plan markdown text.
 * @param {number | string} stepNumber - Numeric step number or symbolic label from the step header.
 * @returns {string | null} Step body text, or null when the step is not found.
 */
function extractStepPacketText(planText, stepNumber) {
  const implementationSection = IMPLEMENTATION_SECTION_PATTERN.exec(planText);
  const text = implementationSection.groups.body;
  const matches = [...text.matchAll(STEP_PATTERN)];
  const normalizedTarget = String(stepNumber);
  const numericTarget =
    typeof stepNumber === 'number' ? stepNumber : Number(stepNumber);
  const match = matches.find((stepMatch) => {
    const step = stepMatch.groups?.step;
    if (step === normalizedTarget) {
      return true;
    }
    const numericStep = Number(step);
    return Number.isFinite(numericStep) && numericStep === numericTarget;
  });

  if (!match) {
    return null;
  }

  const start = match.index + match[0].length;
  const nextMatch = matches.find((stepMatch) => stepMatch.index > match.index);
  const end = nextMatch ? nextMatch.index : text.length;
  return text.slice(start, end).trim();
}

/**
 * Parse the YAML metadata block embedded in a step packet.
 *
 * @param {string} stepPacket - Step packet markdown text.
 * @returns {Record<string, unknown>} Parsed metadata, or an empty object on failure.
 */
function parseStepPacketMetadata(stepPacket) {
  const yamlBlock = /```yaml\r?\n(?<yaml>[\s\S]*?)```/u.exec(stepPacket)?.groups
    ?.yaml;
  if (!yamlBlock) {
    return {};
  }

  return parsePlanYamlBlock(yamlBlock);
}

/**
 * Build a descriptor from an exact slice match.
 *
 * @param {Record<string, unknown>} slice - Slice metadata from the step packet.
 * @param {string} stepPacket - Full step packet text.
 * @param {number} stepNumber - Active step number.
 * @param {Awaited<ReturnType<typeof loadActivePlanContext>>} activePlanContext - Active plan context.
 * @returns {Record<string, unknown>} Descriptor for context-window assembly.
 */
function buildDescriptorFromSlice(
  slice,
  stepPacket,
  stepNumber,
  activePlanContext,
) {
  const stepMetadata = parseStepPacketMetadata(stepPacket);
  const allSlices = Array.isArray(stepMetadata.slices)
    ? stepMetadata.slices
    : [];
  return {
    stepPacket,
    stepNumber,
    stepMetadata,
    sliceTitle: String(slice.title ?? ''),
    boundaryNotes: {
      slice_id: String(slice.slice_id),
      title: String(slice.title ?? ''),
      status: String(slice.status ?? ''),
      goal: String(slice.goal ?? ''),
      phase: activePlanContext.activePhase?.number ?? null,
      phase_status: activePlanContext.activePhase?.status ?? null,
      phase_title: activePlanContext.activePhase?.title ?? null,
      step: stepNumber,
      step_status: String(stepMetadata.status ?? ''),
      estimate_hours: slice.estimate_hours ?? null,
      parallelizable: slice.parallelizable ?? null,
      files_to_change: Array.isArray(slice.files_to_change)
        ? slice.files_to_change
        : [],
      dependencies: Array.isArray(slice.dependencies) ? slice.dependencies : [],
      next_slice: slice.next_slice ?? null,
      slice_history: allSlices.map((s) => ({
        slice_id: String(s.slice_id ?? ''),
        status: String(s.status ?? ''),
        title: String(s.title ?? ''),
      })),
    },
    testContracts: normalizeTestContracts(slice.acceptance_criteria),
  };
}

/**
 * Build a descriptor from a symbolic step label match (e.g. `B1`).
 *
 * @param {Record<string, unknown>} metadata - Parsed step metadata.
 * @param {string} stepPacket - Full step packet text.
 * @param {number} stepNumber - Active step number.
 * @param {Awaited<ReturnType<typeof loadActivePlanContext>>} activePlanContext - Active plan context.
 * @returns {Record<string, unknown>} Descriptor for context-window assembly.
 */
function buildDescriptorFromStep(
  metadata,
  stepPacket,
  stepNumber,
  activePlanContext,
) {
  const allSlices = Array.isArray(metadata.slices) ? metadata.slices : [];
  return {
    stepPacket,
    stepNumber,
    stepMetadata: metadata,
    sliceTitle: null,
    boundaryNotes: {
      phase: activePlanContext.activePhase?.number ?? null,
      phase_status: activePlanContext.activePhase?.status ?? null,
      phase_title: activePlanContext.activePhase?.title ?? null,
      step: stepNumber,
      step_status: String(metadata.status ?? ''),
      status: String(metadata.status ?? ''),
      goal: String(metadata.goal ?? ''),
      source_boundary: Array.isArray(metadata.source_boundary)
        ? metadata.source_boundary
        : [],
      files_to_change: Array.isArray(metadata.files_to_change)
        ? metadata.files_to_change
        : [],
      skills: Array.isArray(metadata.skills) ? metadata.skills : [],
      estimate_hours: metadata.estimate_hours ?? null,
      parallelizable: metadata.parallelizable ?? null,
      dependencies: Array.isArray(metadata.dependencies)
        ? metadata.dependencies
        : [],
      next_slice: metadata.next_slice ?? null,
      slice_history: allSlices.map((s) => ({
        slice_id: String(s.slice_id ?? ''),
        status: String(s.status ?? ''),
        title: String(s.title ?? ''),
      })),
    },
    testContracts: normalizeTestContracts(metadata.acceptance_criteria),
  };
}

/**
 * Normalize an acceptance_criteria list into test-contract objects.
 *
 * @param {unknown} criteria - Raw acceptance criteria from plan metadata.
 * @returns {Array<{ id: string, text: string, validation?: string }>} Test contracts.
 */
function normalizeTestContracts(criteria) {
  if (!Array.isArray(criteria)) {
    return [];
  }

  return criteria.map((criterion) => {
    const text = String(criterion.text ?? '');
    const explicitId = String(criterion.id ?? '').trim();
    const extractedId = explicitId || (text.match(/AC-\d+/)?.[0] ?? '');
    return {
      id: extractedId,
      text,
      validation:
        typeof criterion.validation === 'string'
          ? criterion.validation
          : undefined,
    };
  });
}

/**
 * Check whether a step title begins with a symbolic step identifier.
 *
 * @param {string} title - Step title from the step header.
 * @param {string} sliceId - Requested slice identifier.
 * @returns {boolean} True when the title starts with `sliceId:` or similar delimiter.
 */
function titleMatchesStepId(title, sliceId) {
  const pattern = new RegExp('^' + escapeRegExp(sliceId) + '[:\\s\\-—]', 'u');
  return pattern.test(title.trim());
}

/**
 * Escape a string for safe inclusion in a RegExp character class.
 *
 * @param {string} value - Raw string.
 * @returns {string} Escaped string.
 */
function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

/**
 * Entry point for the workflow MCP server.
 *
 * Parses CLI arguments, builds the server, and either runs the stdio server or
 * executes the local self-check. Exported so tests can exercise startup without
 * spawning a child process.
 *
 * @param {string[]} [argv=process.argv.slice(2)] - Raw CLI arguments.
 * @param {{ runSelfCheck?: typeof runWorkflowSelfCheck, runStdio?: typeof runStdioMcpServer }} [deps={}] - Injectable dependencies.
 * @returns {Promise<void>}
 */
export async function main(
  argv = process.argv.slice(2),
  { runSelfCheck = runWorkflowSelfCheck, runStdio = runStdioMcpServer } = {},
) {
  const options = parseMcpCliArgs(argv);

  if (options.help) {
    printMcpUsage({
      title: 'Expose repo-static workflow facts as a direct MCP server.',
      entrypoint: 'scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs',
      summary:
        'Without flags this script starts a dependency-light stdio MCP server. Use --self-check for a machine-checkable local audit of the current active workflow packet and deterministic customization inventory.',
      tools: createWorkflowTools({ planPath: 'plans/help-only.md' }),
    });
    process.exit(0);
  }

  const effectivePlanPath = requireExplicitPlanPath(options.plan);
  const server = createMcpServer({
    serverName: SERVER_NAME,
    serverVersion: SERVER_VERSION,
    tools: createWorkflowTools({
      planPath: effectivePlanPath,
    }),
  });

  if (options.selfCheck) {
    const report = await runSelfCheck({ server, planPath: effectivePlanPath });
    emitSelfCheckReport(report, options);
    process.exitCode = report.ok ? 0 : 1;
  } else {
    await runStdio(server);
  }
}

/**
 * Bootstrap the entry point and attach a top-level error handler.
 *
 * @param {{ runSelfCheck?: typeof runWorkflowSelfCheck, runStdio?: typeof runStdioMcpServer }} [deps={}] - Injectable dependencies.
 * @returns {Promise<void>}
 */
export function bootstrapMain(deps = {}) {
  return main(undefined, deps).catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

/* istanbul ignore next */
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  bootstrapMain();
}
