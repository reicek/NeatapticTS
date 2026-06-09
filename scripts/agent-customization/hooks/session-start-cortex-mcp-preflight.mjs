#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  initializeRuntimeContextCarrier,
  resolveSessionId,
} from '../enforcement/runtime-enforcement.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const preflightSteps = Object.freeze([
  {
    name: 'session-start-index',
    args: ['scripts/semantic-index/session-start-index.mjs', '--json'],
  },
  {
    name: 'cortex-index-gate',
    args: ['scripts/agent-customization/gates/cortex-index.gate.mjs', '--json'],
  },
]);

main().catch((error) => {
  process.stderr.write(
    `[cortex-session-preflight] runtime context initialization failed: ${error.message}\n`,
  );
  process.exit(2);
});

async function main() {
  const stepSummaries = [];
  for (const step of preflightSteps) {
    const stepResult = runNodeStep(step.args);
    if (stepResult.status !== 0) {
      process.stderr.write(formatFailure(step.name, stepResult));
      process.exit(2);
    }

    stepSummaries.push({
      name: step.name,
      summary: summarizeStdout(stepResult.stdout),
    });
  }

  const sessionId = resolveSessionId();
  const runtimeContext = await initializeRuntimeContextCarrier(sessionId);
  stepSummaries.push({
    name: 'runtime-context',
    summary: runtimeContext?.preparedAction ? 'prepared' : 'initialized',
  });

  writeHookOutput({
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'SessionStart',
      additionalContext: formatSuccessContext(stepSummaries),
    },
  });
}

function runNodeStep(stepArgs) {
  return spawnSync(process.execPath, stepArgs, {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 240_000,
  });
}

function summarizeStdout(stdoutText) {
  const trimmedText = String(stdoutText ?? '').trim();
  if (!trimmedText) {
    return 'ok';
  }

  const parsedOutput = parseTrailingJsonPayload(trimmedText);
  if (parsedOutput !== null) {
    return summarizeParsedOutput(parsedOutput);
  }

  return trimmedText.split(/\r?\n/).at(-1) ?? 'ok';
}

function parseTrailingJsonPayload(stdoutText) {
  const firstJsonBraceIndex = stdoutText.indexOf('{');
  const lastJsonBraceIndex = stdoutText.lastIndexOf('}');
  if (firstJsonBraceIndex === -1 || lastJsonBraceIndex === -1) {
    return null;
  }

  const candidateJson = stdoutText.slice(
    firstJsonBraceIndex,
    lastJsonBraceIndex + 1,
  );
  try {
    return JSON.parse(candidateJson);
  } catch {
    return null;
  }
}

function summarizeParsedOutput(parsedOutput) {
  if (typeof parsedOutput.pass === 'boolean') {
    return parsedOutput.pass ? 'pass' : 'fail';
  }
  if (typeof parsedOutput.ok === 'boolean') {
    return parsedOutput.ok ? 'ok' : 'not-ok';
  }
  if (typeof parsedOutput.status === 'string') {
    return parsedOutput.status;
  }
  if (
    typeof parsedOutput.documents === 'number' &&
    typeof parsedOutput.chunks === 'number'
  ) {
    return `${parsedOutput.documents} docs / ${parsedOutput.chunks} chunks`;
  }
  if (Array.isArray(parsedOutput.steps)) {
    return parsedOutput.steps
      .map((step) => `${step.name}:${step.status}`)
      .join(', ');
  }

  return 'ok';
}

function formatSuccessContext(stepSummaries) {
  const compactSummary = stepSummaries
    .map((stepSummary) => `${stepSummary.name}=${stepSummary.summary}`)
    .join(' | ');
  return `Session-start Cortex/MCP preflight completed: ${compactSummary}`;
}

function formatFailure(stepName, stepResult) {
  const stderrText = String(stepResult.stderr ?? '').trim();
  const stdoutText = String(stepResult.stdout ?? '').trim();
  const details =
    stderrText ||
    stdoutText ||
    `${stepName} exited with status ${stepResult.status ?? 'unknown'}.`;
  return `[cortex-session-preflight] ${stepName} failed: ${details}\n`;
}

function writeHookOutput(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}
