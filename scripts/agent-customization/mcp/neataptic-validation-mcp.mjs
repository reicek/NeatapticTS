#!/usr/bin/env node
/**
 * @module neataptic-validation-mcp
 * @description Validation MCP server — exposes the active-step allow-listed commands as MCP tools.
 *
 * Reads the active `[WIP]` step from the plan file and restricts runnable
 * commands to the exact set declared in that step's YAML metadata block.
 * This server is the sole allow-list authority for AI-driven validation in
 * the NeatapticTS SDLC workflow — no command outside the list can be executed.
 *
 * Commands are run shell-free via {@link runShellFreeCommand} using `spawn`
 * with `shell: false`, so shell metacharacters (`|`, `&`, `;`, `<`, `>`) are
 * tokenizer-rejected before any process is launched.
 */
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
  tokenizeShellSafeCommand,
  MCP_PROTOCOL_VERSION,
} from './mcp-utils.mjs';
import {
  createValidationAllowlistSnapshot,
  loadActivePlanContext,
} from './mcp-plan-utils.mjs';

const SERVER_NAME = 'neataptic-validation-mcp';
const SERVER_VERSION = '0.1.0';
const SELF_CHECK_COMMAND_PATTERN = /neataptic-(workflow|validation)-mcp\.mjs|--self-check/iu;

const options = parseMcpCliArgs(process.argv.slice(2));

if (options.help) {
  printMcpUsage({
    title: 'Expose active-step allow-listed validation commands as a direct MCP server.',
    entrypoint: 'scripts/agent-customization/mcp/neataptic-validation-mcp.mjs',
    summary: 'Without flags this script starts a dependency-light stdio MCP server. Use --self-check to confirm that the current active step packet is the only allow-list authority and that bounded validation commands still execute without a shell.',
    tools: createValidationTools('plans/help-only.md'),
  });
  process.exit(0);
}

const planPath = requireExplicitPlanPath(options.plan);
const server = createMcpServer({
  serverName: SERVER_NAME,
  serverVersion: SERVER_VERSION,
  tools: createValidationTools(planPath),
});

if (options.selfCheck) {
  const report = await runValidationSelfCheck({ server, planPath });
  emitSelfCheckReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
} else {
  await runStdioMcpServer(server);
}

/**
 * Build the tool list for the validation MCP server.
 *
 * Provides two tools:
 * - `get_active_validation_allowlist` — returns the step-packet command list.
 * - `run_allowlisted_validation` — executes one exact allow-listed command.
 *
 * **Security:** `run_allowlisted_validation` checks the requested command
 * against `validationCommands` from the active step packet before any process
 * is launched. Any command not in the list is rejected with an error regardless
 * of whether it is otherwise a valid or safe command string.
 *
 * @param {string} planPath - Repo-relative plan path.
 * @returns {Array<{ name: string, description: string, inputSchema: Record<string, unknown>, handler: Function }>} Tool list.
 */
function createValidationTools(planPath) {
  return [
    createTool({
      name: 'get_active_validation_allowlist',
      description: 'Return the exact allow-listed validation commands from the active step packet.',
      annotations: { readOnlyHint: true },
      handler: async () => createValidationAllowlistSnapshot(await loadActivePlanContext(planPath)),
    }),
    createTool({
      name: 'run_allowlisted_validation',
      description: 'Run one exact command already named by the active step packet without using a shell.',
      annotations: { readOnlyHint: false },
      inputSchema: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'Exact command string returned by get_active_validation_allowlist.',
          },
        },
        required: ['command'],
        additionalProperties: false,
      },
      handler: async (argumentsObject) => {
        const requestedCommand = requireString(argumentsObject.command, 'command');
        const activePlanContext = await loadActivePlanContext(planPath);
        if (!activePlanContext.activeStep.validationCommands.includes(requestedCommand)) {
          throw new Error('Command is not allow-listed by the active step packet.');
        }

        const commandResult = await runShellFreeCommand(requestedCommand);
        return {
          scope: 'direct-MCP',
          plan: planPath,
          allowlistAuthority: 'active-step.validation',
          ...commandResult,
        };
      },
    }),
  ];
}

/**
 * Run an end-to-end self-check of the validation MCP server.
 *
 * Validates protocol version, tool count, allow-list snapshot correctness, and
 * executes up to two non-recursive allow-listed commands. Also verifies that:
 * - An out-of-list command is rejected with an error.
 * - The tokenizer rejects shell metacharacters.
 *
 * Both verifications confirm the security boundary is intact before marking
 * the self-check as passing.
 *
 * @param {{ server: object, planPath: string }} params - Server instance and plan path.
 * @returns {Promise<Record<string, unknown>>} Self-check report.
 */
async function runValidationSelfCheck({ server, planPath }) {
  const issues = [];
  const activePlanContext = await loadActivePlanContext(planPath);
  const initializeResult = await invokeServerRequest(server, {
    method: 'initialize',
    params: {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: 'self-check', version: SERVER_VERSION },
    },
  });
  const toolListResult = await invokeServerRequest(server, { method: 'tools/list' });
  const allowlistResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'get_active_validation_allowlist',
      arguments: {},
    },
  });

  if (initializeResult.protocolVersion !== MCP_PROTOCOL_VERSION) {
    issues.push(selfCheckError(planPath, `Expected protocol version ${MCP_PROTOCOL_VERSION}, received ${String(initializeResult.protocolVersion)}.`));
  }

  if (!Array.isArray(toolListResult.tools) || toolListResult.tools.length !== 2) {
    issues.push(selfCheckError(planPath, `Expected 2 validation tools, found ${Array.isArray(toolListResult.tools) ? toolListResult.tools.length : 'none'}.`));
  }

  if (allowlistResult.isError) {
    issues.push(selfCheckError(planPath, 'Validation allow-list tool returned an error during self-check.'));
  }

  const allowlistSnapshot = allowlistResult.structuredContent ?? {};
  if (!allowlistSnapshot.validationCommandsMatch) {
    issues.push(selfCheckError(planPath, 'Validation allow-list disagrees with the Required validation prose in the active step packet.'));
  }

  if (allowlistSnapshot.activeStep?.number !== activePlanContext.activeStep.number) {
    issues.push(selfCheckError(planPath, 'Validation allow-list did not resolve the current active step.'));
  }

  const executableValidationCommands = (allowlistSnapshot.validationCommands ?? [])
    .filter((command) => typeof command === 'string')
    .filter((command) => !SELF_CHECK_COMMAND_PATTERN.test(command))
    .slice(0, 2);

  if (executableValidationCommands.length === 0) {
    issues.push(selfCheckError(planPath, 'Validation self-check could not find any non-recursive allow-listed commands to run.'));
  }

  const commandResults = [];
  for (const command of executableValidationCommands) {
    const toolCallResult = await invokeServerRequest(server, {
      method: 'tools/call',
      params: {
        name: 'run_allowlisted_validation',
        arguments: { command },
      },
    });

    commandResults.push(toolCallResult.structuredContent ?? null);
    if (toolCallResult.isError || toolCallResult.structuredContent?.exitCode !== 0) {
      issues.push(selfCheckError(planPath, `Allow-listed validation command failed during self-check: ${command}`));
    }
  }

  const rejectedUnknownCommand = await invokeServerRequest(server, {
    method: 'tools/call',
    params: {
      name: 'run_allowlisted_validation',
      arguments: {
        command: 'node scripts/agent-customization/not-approved.mjs',
      },
    },
  });
  if (!rejectedUnknownCommand.isError) {
    issues.push(selfCheckError(planPath, 'Validation MCP did not reject a command outside the active allow-list.'));
  }

  let rejectedUnsafeCommand = false;
  try {
    tokenizeShellSafeCommand('node scripts/agent-customization/validate-plan-sync.mjs ; injected');
  } catch {
    rejectedUnsafeCommand = true;
  }

  if (!rejectedUnsafeCommand) {
    issues.push(selfCheckError(planPath, 'Validation command tokenizer did not reject shell metacharacters.'));
  }

  return createSelfCheckReport('neataptic-validation-mcp self-check', issues, {
    server: { name: SERVER_NAME, version: SERVER_VERSION },
    plan: planPath,
    toolNames: server.tools.map((tool) => tool.name),
    activeStep: {
      phase: allowlistSnapshot.activePhase?.number ?? null,
      step: allowlistSnapshot.activeStep?.number ?? null,
      agent: allowlistSnapshot.activeStep?.agent ?? null,
    },
    allowlistCommandCount: Array.isArray(allowlistSnapshot.validationCommands)
      ? allowlistSnapshot.validationCommands.length
      : 0,
    sampledCommandResults: commandResults,
    rejectedUnknownCommand: Boolean(rejectedUnknownCommand.isError),
    rejectedUnsafeCommand,
  });
}