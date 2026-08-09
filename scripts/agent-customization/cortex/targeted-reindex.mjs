/**
 * @module targeted-reindex
 * @description Targeted Cortex reindex engine that re-chunks and re-embeds a
 *   bounded set of repo-relative file paths.
 *
 * Exposes `reindexFiles(filePaths)` which orchestrates two steps per eligible
 * file:
 *   1. `rag-index/build-index.mjs --files=<path> ...` — re-chunk the changed
 *      file into the corpus SQLite database.
 *   2. `rag-index/embed-index.mjs --files=<path> ...` — re-embed the changed
 *      chunks into the dense embedding index.
 *
 * Only `.md`, `.ts`, `.mjs`, `.js` files under `plans/`, `src/`, `examples/`,
 * `scripts/agent-customization/`, `rag-index/`, and `scripts/mcp-semantic/` are
 * eligible; everything else is
 * silently skipped. Paths are resolved relative to the repo root and
 * normalized to POSIX-style repo-relative form before being passed to the
 * underlying CLIs.
 *
 * The function never throws: it returns `{ reindexed, errors }` so callers
 * (especially the post-write hook) can invoke it without a surrounding
 * try/catch that risks leaking exceptions into the host process.
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');

/**
 * Eligible top-level corpus roots (repo-relative, POSIX-style). A file must
 * start with one of these prefixes to be considered for reindexing.
 */
const ELIGIBLE_ROOTS = [
  'plans/',
  'src/',
  'examples/',
  'scripts/agent-customization/',
  'rag-index/',
  'scripts/mcp-semantic/',
];

/**
 * Eligible file extensions (lowercase, including the leading dot).
 */
const ELIGIBLE_EXTENSIONS = ['.md', '.ts', '.mjs', '.js'];

/**
 * Normalize a raw file path into a canonical POSIX-style repo-relative path.
 * Absolute paths are resolved relative to the repo root; relative paths are
 * resolved against the repo root as well.
 *
 * @param {string} filePath - Raw file path (absolute or relative).
 * @returns {string} Repo-relative POSIX-style path.
 */
function toRepoRelative(filePath) {
  const absolute = path.isAbsolute(filePath)
    ? filePath
    : path.resolve(repoRoot, filePath);
  return path.relative(repoRoot, absolute).replaceAll(path.sep, '/');
}

/**
 * Determine whether a single repo-relative file path is eligible for
 * reindexing. A file is eligible when it lives under an eligible root AND has
 * an eligible extension.
 *
 * @param {string} filePath - Repo-relative POSIX-style path.
 * @returns {boolean} `true` when the file should be reindexed.
 *
 * @example
 * ```js
 * import { isEligibleFile } from './targeted-reindex.mjs';
 * isEligibleFile('src/neat/neat.ts'); // true
 * isEligibleFile('README.md');       // false
 * ```
 */
export function isEligibleFile(filePath) {
  if (typeof filePath !== 'string' || filePath.trim() === '') return false;
  const normalized = toRepoRelative(filePath);
  const underEligibleRoot = ELIGIBLE_ROOTS.some((root) =>
    normalized.startsWith(root),
  );
  if (!underEligibleRoot) return false;
  const ext = path.posix.extname(normalized).toLowerCase();
  return ELIGIBLE_EXTENSIONS.includes(ext);
}

/**
 * Spawn a node child process and resolve on exit. Resolves with the exit code
 * and captured stderr on completion; never rejects (errors are returned as
 * `{ code: -1, stderr }`).
 *
 * @param {string[]} args - Node script arguments (relative to repo root).
 * @returns {Promise<{ code: number, stderr: string }>} Exit result.
 */
function spawnNode(args) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(process.execPath, args, {
        cwd: repoRoot,
        stdio: ['ignore', 'ignore', 'pipe'],
        detached: false,
      });
    } catch (spawnError) {
      resolve({ code: -1, stderr: `spawn failed: ${spawnError.message}` });
      return;
    }

    let stderr = '';
    child.stderr?.on('data', (chunk) => {
      stderr += String(chunk);
    });
    child.on('error', (err) => {
      resolve({ code: -1, stderr: `child error: ${err.message}` });
    });
    child.on('exit', (code) => {
      resolve({ code: code ?? -1, stderr });
    });
  });
}

/**
 * Build the CLI argument list for one reindex step.
 *
 * @param {string} scriptPath - Repo-relative script path (e.g.
 *   `rag-index/build-index.mjs`).
 * @param {string[]} files - Repo-relative file paths to reindex.
 * @returns {string[]} Node argument list.
 */
function buildArgs(scriptPath, files) {
  const args = [scriptPath, '--json'];
  for (const file of files) {
    args.push(`--files=${file}`);
  }
  return args;
}

/**
 * Reindex a bounded set of file paths. Eligible files are re-chunked via
 * `build-index.mjs` and re-embedded via `embed-index.mjs`. Ineligible files
 * are silently skipped.
 *
 * The function never throws. On success it returns `{ reindexed, errors }`
 * where `reindexed` lists the files that completed both steps cleanly and
 * `errors` lists per-file failure messages.
 *
 * @param {string[]} filePaths - Raw file paths (absolute or repo-relative).
 * @returns {Promise<{ reindexed: string[], errors: string[] }>} Result.
 *
 * @example
 * ```js
 * import { reindexFiles } from './targeted-reindex.mjs';
 * const { reindexed, errors } = await reindexFiles(['src/neat/neat.ts']);
 * console.log(reindexed.length, errors.length);
 * ```
 */
export async function reindexFiles(filePaths) {
  const result = { reindexed: [], errors: [] };
  if (!Array.isArray(filePaths)) return result;

  const eligibleSet = new Set();
  for (const raw of filePaths) {
    if (typeof raw !== 'string' || raw.trim() === '') continue;
    const normalized = toRepoRelative(raw);
    if (isEligibleFile(normalized)) {
      eligibleSet.add(normalized);
    }
  }
  const eligible = [...eligibleSet].sort();

  if (eligible.length === 0) return result;

  const buildResult = await spawnNode(
    buildArgs('rag-index/build-index.mjs', eligible),
  );
  if (buildResult.code !== 0) {
    result.errors.push(
      `build-index failed (code ${buildResult.code}): ${buildResult.stderr.trim()}`,
    );
    return result;
  }

  const embedResult = await spawnNode(
    buildArgs('rag-index/embed-index.mjs', eligible),
  );
  if (embedResult.code !== 0) {
    result.errors.push(
      `embed-index failed (code ${embedResult.code}): ${embedResult.stderr.trim()}`,
    );
    return result;
  }

  result.reindexed = eligible;
  return result;
}

/**
 * Reset any module-level state. Currently a no-op placeholder kept for test
 * symmetry and future extension (e.g. an internal cooldown cache).
 */
export function resetReindexState() {
  // Intentionally empty; reserved for future cooldown/debounce state.
}
