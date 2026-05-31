#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const writeToolPattern = /(apply_patch|create_file|vscode_renameSymbol|editFiles|replace_string_in_file|createFile|writeFile)/i;
const refreshSteps = Object.freeze([
  {
    name: 'build-index',
    args: ['scripts/semantic-index/build-index.mjs', '--json-health'],
  },
  {
    name: 'build-browser-snapshot',
    args: ['scripts/semantic-index/build-browser-snapshot.mjs', '--json'],
  },
  {
    name: 'prewarm-dense',
    args: ['scripts/semantic-index/prewarm-dense.mjs', '--json'],
  },
  {
    name: 'cortex-index-gate',
    args: ['scripts/agent-customization/gates/cortex-index.gate.mjs', '--json'],
  },
]);

main();

function main() {
  const hookInput = readHookInput();
  if (!shouldRefreshCortex(hookInput)) {
    writeHookOutput({ continue: true });
    return;
  }

  const stepSummaries = [];
  for (const step of refreshSteps) {
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

  writeHookOutput({
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'PostToolUse',
      additionalContext: formatSuccessContext(stepSummaries),
    },
  });
}

function readHookInput() {
  const rawInput = readFileSync(0, 'utf8').trim();
  if (!rawInput) {
    return {};
  }

  try {
    return JSON.parse(rawInput);
  } catch {
    return {};
  }
}

function shouldRefreshCortex(hookInput) {
  const toolName = String(hookInput.tool_name ?? hookInput.toolName ?? '');
  return writeToolPattern.test(toolName);
}

function runNodeStep(stepArgs) {
  return spawnSync(process.execPath, stepArgs, {
    cwd: repoRoot,
    encoding: 'utf8',
    timeout: 180000,
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

  const candidateJson = stdoutText.slice(firstJsonBraceIndex, lastJsonBraceIndex + 1);
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
  return `Cortex auto-refresh completed after file write: ${compactSummary}`;
}

function formatFailure(stepName, stepResult) {
  const stderrText = String(stepResult.stderr ?? '').trim();
  const stdoutText = String(stepResult.stdout ?? '').trim();
  const details = stderrText || stdoutText || `${stepName} exited with status ${stepResult.status ?? 'unknown'}.`;
  return `[cortex-auto-refresh] ${stepName} failed: ${details}\n`;
}

function writeHookOutput(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}
