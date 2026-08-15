#!/usr/bin/env node
/**
 * @module post-write-reindex-hook
 * @description PostToolUse hook that triggers a fire-and-forget targeted
 *   reindex of the file touched by a write tool call.
 *
 * The hook reads the tool call arguments from stdin (the standard hook input
 * JSON), extracts the file path from the `path` / `file_path` field, and
 * spawns a detached background process that runs `targeted-reindex.mjs` for
 * that single file. The hook never blocks the host tool and never throws into
 * the host: every failure path is swallowed and logged to
 * `scripts/agent-customization/hooks/post-write-reindex.log`.
 *
 * Eligibility (`.md`/`.ts`/`.mjs`/`.js` under corpus roots) is enforced inside
 * `targeted-reindex.mjs`, so the hook simply forwards the file path. The
 * hook only acts on write-style tool names (edit, create, apply_patch, etc.).
 */
import { spawn } from 'node:child_process';
import { appendFileSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const reindexScriptPath = path.join(
  repoRoot,
  'scripts',
  'agent-customization',
  'cortex',
  'targeted-reindex.mjs',
);
const logPath = path.join(
  repoRoot,
  'scripts',
  'agent-customization',
  'hooks',
  'post-write-reindex.log',
);

/**
 * Tool names that produce file writes and should trigger a reindex.
 */
const WRITE_TOOL_PATTERN =
  /^(edit|create|create_file|createFile|writeFile|apply_patch|editFiles|replace_string_in_file|vscode_renameSymbol)$/i;

/**
 * Hook entry point. Reads stdin, extracts the file path, spawns a background
 * reindex process, and immediately returns control to the host.
 *
 * @param {Record<string, unknown>} hookInput - Parsed hook input JSON.
 */
export function main(hookInput) {
  const toolName = String(hookInput.tool_name ?? hookInput.toolName ?? '');
  if (!WRITE_TOOL_PATTERN.test(toolName)) {
    writeHookOutput({ continue: true });
    return;
  }

  const filePath = extractFilePath(hookInput);
  if (!filePath) {
    writeHookOutput({ continue: true });
    return;
  }

  triggerBackgroundReindex(filePath);
  writeHookOutput({
    continue: true,
    hookSpecificOutput: {
      hookEventName: 'PostToolUse',
      additionalContext: `Post-write reindex triggered for ${filePath}`,
    },
  });
}

main(readHookInput());

/**
 * Extract the file path from the hook input. The hook receives the tool call
 * arguments in various shapes (`tool_input`, `arguments`, or the top-level
 * object). This looks for `path` or `file_path` fields anywhere in the
 * arguments object.
 *
 * @param {Record<string, unknown>} hookInput - Parsed hook input JSON.
 * @returns {string|null} The extracted file path or null.
 */
export function extractFilePath(hookInput) {
  if (!hookInput || typeof hookInput !== 'object') return null;
  const candidates = [
    hookInput.tool_input,
    hookInput.arguments,
    hookInput.input,
    hookInput,
  ];
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== 'object') continue;
    const raw =
      candidate.path ??
      candidate.file_path ??
      candidate.filePath ??
      candidate.filename;
    if (typeof raw === 'string' && raw.trim() !== '') return raw;
  }
  return null;
}

/**
 * Spawn a detached background node process that imports and runs
 * `reindexFiles([filePath])`. The process is fully detached (unref'd) so the
 * hook returns immediately without waiting for completion.
 *
 * @param {string} filePath - Repo-relative or absolute file path.
 */
export function triggerBackgroundReindex(filePath) {
  const launcher = `
import { reindexFiles } from ${JSON.stringify(reindexScriptPath)};
const { reindexed, errors } = await reindexFiles([${JSON.stringify(filePath)}]);
if (errors.length > 0) {
  for (const err of errors) console.error('reindex error:', err);
}
`;
  try {
    const child = spawn(
      process.execPath,
      ['--input-type=module', '-e', launcher],
      {
        cwd: repoRoot,
        stdio: 'ignore',
        detached: true,
      },
    );
    child.unref();
    safeLog(
      `[post-write-reindex] triggered for ${filePath} (pid ${child.pid ?? '?'})`,
    );
  } catch (spawnError) {
    safeLog(
      `[post-write-reindex] spawn failed for ${filePath}: ${spawnError.message}`,
    );
  }
}

/**
 * Append a timestamped line to the reindex log. Silently no-ops on filesystem
 * errors so the hook never throws.
 *
 * @param {string} message - Log message.
 */
export function safeLog(message) {
  try {
    mkdirSync(path.dirname(logPath), { recursive: true });
    appendFileSync(logPath, `${new Date().toISOString()} ${message}\n`, 'utf8');
  } catch {
    // Logging is best-effort; never throw.
  }
}

/**
 * Read and parse the hook input JSON from stdin (fd 0).
 *
 * @returns {Record<string, unknown>} Parsed hook input (or empty object).
 */
export function readHookInput() {
  try {
    const rawInput = readFileSync(0, 'utf8').trim();
    if (!rawInput) return {};
    return JSON.parse(rawInput);
  } catch {
    return {};
  }
}

/**
 * Write the hook output JSON to stdout.
 *
 * @param {object} payload - Hook output payload.
 */
export function writeHookOutput(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}
