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
    step_number: descriptor.stepNumber,
    step_title: String(stepMeta.title ?? ''),
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
  lines.push(
    `SLICE: ${sliceId} — ${notes.title ?? descriptor.sliceTitle ?? sliceId}`,
  );
  lines.push(
    `${phase}${stepPart}| STATUS: ${notes.status ?? 'unknown'} | GOAL: ${notes.goal ?? 'unspecified'}`,
  );
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
  if (notes.next_slice) {
    lines.push(`NEXT SLICE: ${notes.next_slice}`);
  }
  if (stepMeta.next_step) {
    lines.push(`NEXT STEP: ${stepMeta.next_step}`);
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
async function fetchSliceContext(descriptor, sliceId, searchContextFn) {
  const { query, metadata } = buildSliceQuery(descriptor, sliceId);
  const notes = descriptor.boundaryNotes || {};
  if (typeof searchContextFn !== 'function') {
    return emptyRagContext(query);
  }

  let response;
  try {
    response = await searchContextFn({
      query,
      limit: CONTEXT_RESULT_LIMIT,
      budget: CONTEXT_TOKEN_BUDGET,
      include_metadata: true,
      context_format: 'json',
      expand_query: true,
      use_rerank: true,
      metadata,
    });
  } catch {
    return emptyRagContext(query);
  }

  if (!response || typeof response !== 'object') {
    return emptyRagContext(query);
  }

  // When `context_format: 'json'` is honored, the assembled context object is
  // returned in `response.context`. Map the assembled chunk bodies back into
  // the individual `chunks` entries so callers never see empty `text` fields.
  const assembled =
    response.context && typeof response.context === 'object'
      ? response.context
      : null;
  const assembledChunks = Array.isArray(assembled?.chunks)
    ? assembled.chunks
    : [];

  const rawResults = response.results;
  if (!Array.isArray(rawResults) && assembledChunks.length === 0) {
    return emptyRagContext(query);
  }

  const sourceResults = Array.isArray(rawResults) ? rawResults : [];
  const allChunks = sourceResults
    .slice(0, CONTEXT_RESULT_LIMIT)
    .map((result, index) => {
      const fallbackChunk = assembledChunks[index];
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
      // Prefer the assembled chunk content (which has full body text) over
      // the metadata-only result fields.
      const rawText =
        result.text ??
        result.content ??
        result.snippet ??
        fallbackChunk?.content ??
        '';
      // Drop chunks that were truncated by budget enforcement — a partial
      // chunk is misleading. The agent can use follow_up_refs to load the
      // full chunk via load_chunk if needed.
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
    });

  // Keep only chunks with non-empty text (drops truncated and empty chunks).
  const chunks = allChunks.filter(
    (chunk) => typeof chunk.text === 'string' && chunk.text.trim().length > 0,
  );

  // Capture dropped (truncated) chunks so we can generate follow-up refs for
  // them — the agent can retrieve their full text via load_chunk.
  const droppedChunks = allChunks.filter(
    (chunk) =>
      chunk.chunk_id != null &&
      !(typeof chunk.text === 'string' && chunk.text.trim().length > 0),
  );

  // Detect partial-file chunks: a chunk may cover only part of its source
  // file (char_end < file length).  Reading the file to get its character
  // count is the only reliable way to detect this — the search engine's
  // `truncated` flag only marks chunks cut by the token budget, not chunks
  // that are simply the first slice of a multi-chunk file.
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

  // Build follow-up refs. We skip redundant load_chunk refs for complete
  // chunks (the full text is already in the `chunks` array) and instead
  // generate a targeted search_context ref for files_to_change that have no
  // chunk in the context, plus keep refs for dropped (truncated) chunks.
  const followUpRefs = [];
  // Add refs for dropped (truncated) chunks so the agent can retrieve them.
  for (const chunk of droppedChunks.slice(0, 3)) {
    if (chunk.chunk_id != null) {
      followUpRefs.push({
        tool: 'load_chunk',
        args: { chunk_id: chunk.chunk_id, query },
        reason: `Full text of ${chunk.file_path ?? 'chunk'} (truncated by budget, not included in context)`,
      });
    }
  }
  // Generate a targeted search_context ref for files_to_change that have no
  // chunk in the context, so the agent knows exactly what's missing.
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
  // Keep the search_context suggestion from the original refs if present.
  const searchRef = Array.isArray(response.follow_up_refs)
    ? response.follow_up_refs.find(
        (ref) =>
          ref?.tool === 'search_context' && typeof ref.reason === 'string',
      )
    : null;
  if (searchRef) {
    followUpRefs.push(searchRef);
  }

  // Build a brief stitched summary: basename + first sentence of JSDoc (or
  // first meaningful line) of each chunk. This gives agents a quick overview
  // of what context is available without duplicating the full chunk texts
  // (which are already in the `chunks` array).
  const text =
    chunks.length > 0
      ? chunks
          .map((chunk) => {
            const fullText = chunk.text || '';
            const docMatch = fullText.match(/\/\*\*([\s\S]*?)\*\//);
            let description = docMatch
              ? docMatch[1]
                  .split(/\r?\n/)
                  .map((l) => l.replace(/^\s*\*\s?/, '').trim())
                  .filter(Boolean)
                  .join(' ')
              : (fullText.split(/\r?\n/).find((l) => l.trim().length > 0) ??
                '');
            // Shorten the module path in the description to just the basename.
            description = description.replace(
              /examples\/neatenstein\/browser-entry\/host\/game\//g,
              '',
            );
            const basename = (chunk.file_path ?? 'unknown').split('/').pop();
            const partialNote =
              chunk.partial_file === true
                ? ` [PARTIAL: chars 0-${chunk.char_end} of ${chunk.file_chars}]`
                : '';
            return `${basename} — ${description}${partialNote}`;
          })
          .join('\n')
      : typeof response.context === 'string' && response.context.length > 0
        ? response.context
        : typeof assembled?.context === 'string' && assembled.context.length > 0
          ? assembled.context
          : '';

  return {
    query,
    text,
    chunks,
    token_count:
      typeof response.token_count === 'number'
        ? response.token_count
        : typeof assembled?.tokenCount === 'number'
          ? assembled.tokenCount
          : 0,
    dense_state:
      typeof response.dense_state === 'string' ? response.dense_state : null,
    truncated: response.truncated === true,
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
 * Build a Cortex search query and optional metadata filter from the slice descriptor.
 *
 * The query blends semantic slice metadata (title, goal, contracts) with exact
 * path tokens derived from `files_to_change` so the search targets both the
 * intent and the concrete files in the slice boundary. A `metadata.filter`
 * constrains results to the same file paths when paths are available.
 *
 * @param {Record<string, unknown>} descriptor - Resolved slice descriptor.
 * @param {string} sliceId - Exact slice identifier.
 * @returns {{ query: string, metadata?: { filter: Record<string, unknown> } }} Search options.
 */
function buildSliceQuery(descriptor, _sliceId) {
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

  const stopWords = new Set([
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
  const semanticTokens = rawSemantic
    .toLowerCase()
    .replace(/[^a-z0-9_\-/\s]/g, ' ')
    .split(/\s+/)
    .filter((token) => token.length > 1 && !stopWords.has(token))
    .filter((token, index, arr) => arr.indexOf(token) === index)
    .slice(0, 3);

  const query = semanticTokens.join(' ').slice(0, 256);
  const filter = buildFilePathFilter(files);
  return filter ? { query, metadata: { filter } } : { query };
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
      step: stepNumber,
      estimate_hours: slice.estimate_hours ?? null,
      parallelizable: slice.parallelizable ?? null,
      files_to_change: Array.isArray(slice.files_to_change)
        ? slice.files_to_change
        : [],
      dependencies: Array.isArray(slice.dependencies) ? slice.dependencies : [],
      next_slice: slice.next_slice ?? null,
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
  return {
    stepPacket,
    stepNumber,
    stepMetadata: metadata,
    sliceTitle: null,
    boundaryNotes: {
      phase: activePlanContext.activePhase?.number ?? null,
      step: stepNumber,
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
