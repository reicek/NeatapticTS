#!/usr/bin/env node
import {
  clearPreparedRuntimeContext,
  diagnosePreparedRuntimeContext,
  inferActionClass,
  prepareRuntimeContext,
  readRuntimeContext,
  resolveSessionId,
} from './runtime-enforcement.mjs';

const options = parseCliArguments(process.argv.slice(2));

if (options.help || (!options.prepare && !options.show && !options.clear && !options.diagnose)) {
  printUsage();
  process.exit(options.help ? 0 : 1);
}

try {
  const sessionId = resolveSessionId({ sessionId: options.sessionId });
  let result;

  if (options.prepare) {
    result = await prepareRuntimeContext({
      sessionId,
      flowId: options.flowId,
      currentAgent: options.currentAgent,
      delegatorChain: options.delegatorChain,
      requiredSkills: options.requiredSkills,
      requiredSpecialists: options.requiredSpecialists,
      planPath: options.planPath,
      activePhase: options.activePhase,
      activeStep: options.activeStep,
      allowedActionClass: options.actionClass || inferActionClass(options.expectedToolName),
      expectedToolName: options.expectedToolName,
    });
  } else if (options.diagnose) {
    const carrier = await readRuntimeContext(sessionId);
    result = {
      carrier,
      diagnosis: diagnosePreparedRuntimeContext({
        carrier,
        sessionId,
        toolName: options.expectedToolName,
        planPath: options.planPath,
      }),
    };
  } else if (options.clear) {
    result = await clearPreparedRuntimeContext(sessionId, options.actionId);
  } else {
    result = await readRuntimeContext(sessionId);
  }

  console.log(JSON.stringify({ ok: true, sessionId, context: result }, null, 2));
  process.exitCode = 0;
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.log(JSON.stringify({ ok: false, error: message }, null, 2));
  process.exitCode = 1;
}

function parseCliArguments(argv) {
  const parsed = {
    prepare: false,
    show: false,
    clear: false,
    help: false,
    diagnose: false,
    sessionId: '',
    flowId: '',
    currentAgent: '',
    delegatorChain: '',
    requiredSkills: '',
    requiredSpecialists: '',
    planPath: '',
    activePhase: '',
    activeStep: '',
    actionClass: '',
    expectedToolName: '',
    actionId: '',
  };

  for (const argument of argv) {
    if (argument === '--prepare') parsed.prepare = true;
    else if (argument === '--show') parsed.show = true;
    else if (argument === '--clear') parsed.clear = true;
    else if (argument === '--diagnose') parsed.diagnose = true;
    else if (argument === '--help' || argument === '-h') parsed.help = true;
    else if (argument.startsWith('--session-id=')) parsed.sessionId = argument.slice('--session-id='.length);
    else if (argument.startsWith('--flow-id=')) parsed.flowId = argument.slice('--flow-id='.length);
    else if (argument.startsWith('--agent=')) parsed.currentAgent = argument.slice('--agent='.length);
    else if (argument.startsWith('--delegator-chain=')) parsed.delegatorChain = argument.slice('--delegator-chain='.length);
    else if (argument.startsWith('--required-skills=')) parsed.requiredSkills = argument.slice('--required-skills='.length);
    else if (argument.startsWith('--required-specialists=')) parsed.requiredSpecialists = argument.slice('--required-specialists='.length);
    else if (argument.startsWith('--plan=')) parsed.planPath = argument.slice('--plan='.length);
    else if (argument.startsWith('--phase=')) parsed.activePhase = argument.slice('--phase='.length);
    else if (argument.startsWith('--step=')) parsed.activeStep = argument.slice('--step='.length);
    else if (argument.startsWith('--action-class=')) parsed.actionClass = argument.slice('--action-class='.length);
    else if (argument.startsWith('--tool-name=')) parsed.expectedToolName = argument.slice('--tool-name='.length);
    else if (argument.startsWith('--action-id=')) parsed.actionId = argument.slice('--action-id='.length);
  }

  return parsed;
}

function printUsage() {
  console.log(`runtime-enforcement-context

Usage:
  node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --prepare --flow-id=<id> --agent=<agent> --delegator-chain=<json-or-csv> --required-skills=<json-or-csv> --required-specialists=<json-or-csv> --plan=<path> [--phase=<phase>] [--step=<step>] --tool-name=<tool> [--action-class=<write|execute>] [--session-id=<id>]
  node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --show [--session-id=<id>]
  node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --diagnose --tool-name=<tool> [--plan=<path>] [--session-id=<id>]
  node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --clear [--action-id=<id>] [--session-id=<id>]
`);
}
