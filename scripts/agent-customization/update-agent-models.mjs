#!/usr/bin/env node
/**
 * @module update-agent-models
 *
 * Bulk-update or remove model strings across all `.github/agents/*.agent.md`
 * files.
 *
 * ## Mode 1: Replace (default)
 *
 * Replaces every occurrence of a `--from` model string with a `--to` model
 * string in both the frontmatter `model:` field and any handoff `model:`
 * sub-fields.
 *
 * ## Mode 2: Remove (`--remove`)
 *
 * Removes all top-level frontmatter `model:` lines from every agent file.
 * This makes custom agents inherit the session model (like `general-purpose`
 * does), avoiding the `agentsResolveCustomAgentModel` native function that
 * appends a `(ollama)` provider suffix — which causes "400 invalid model name"
 * errors on the Ollama API.
 *
 * Usage:
 *   # Remove all model fields (recommended — agents inherit session model)
 *   node scripts/agent-customization/update-agent-models.mjs --remove
 *
 *   # Replace specific model string
 *   node scripts/agent-customization/update-agent-models.mjs \
 *     --from='glm-5.2:cloud (ollama)' \
 *     --to='glm-5.2:cloud'
 *
 *   # Replace ALL model strings with a single target (omit --from)
 *   node scripts/agent-customization/update-agent-models.mjs \
 *     --to='glm-5.2:cloud'
 *
 * Options:
 *   --remove          Remove all top-level frontmatter model: fields
 *   --from=<string>   Model string to find (omit to replace any model: value)
 *   --to=<string>     Replacement model string (required unless --remove)
 *   --dry-run         Preview changes without writing files
 *   --json            Machine-readable JSON output
 *   --help, -h        Show help
 */

import { readFile, writeFile } from 'node:fs/promises';
import { glob } from 'node:fs/promises';
import path from 'node:path';

const repoRoot = process.cwd();

/** Parse CLI arguments. */
function parseArgs(argv) {
  const opts = {
    from: null,
    to: null,
    remove: false,
    dryRun: false,
    json: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') opts.help = true;
    else if (arg === '--remove') opts.remove = true;
    else if (arg === '--dry-run' || arg === '--dry_run') opts.dryRun = true;
    else if (arg === '--json') opts.json = true;
    else if (arg.startsWith('--from=')) opts.from = arg.slice('--from='.length);
    else if (arg === '--from' && argv[i + 1] !== undefined)
      opts.from = argv[++i];
    else if (arg.startsWith('--to=')) opts.to = arg.slice('--to='.length);
    else if (arg === '--to' && argv[i + 1] !== undefined) opts.to = argv[++i];
  }
  return opts;
}

function printHelp() {
  console.log(`Update or remove agent model strings in bulk.

Usage:
  # Remove all frontmatter model: fields (recommended — agents inherit session model)
  node scripts/agent-customization/update-agent-models.mjs --remove

  # Replace specific model string
  node scripts/agent-customization/update-agent-models.mjs --from='<old>' --to='<new>'

  # Replace ALL model strings with a single target
  node scripts/agent-customization/update-agent-models.mjs --to='<new>' --dry-run

Options:
  --remove           Remove all top-level frontmatter model: fields
  --from=<string>    Model string to find (omit to replace any model: value)
  --to=<string>      Replacement model string (required unless --remove)
  --dry-run          Preview changes without writing files
  --json             Machine-readable JSON output
  --help, -h         Show this help text

Why --remove is recommended:
  When a custom agent has a \`model:\` field in frontmatter, the CLI resolves
  it via agentsResolveCustomAgentModel, which appends a provider suffix like
  "(ollama)". The suffixed name (e.g. "glm-5.2:cloud (ollama)") is sent to
  the Ollama API, which rejects it with "400 invalid model name". Removing
  the model: field makes the agent inherit the session model directly,
  bypassing the problematic native resolver.

After running, refresh the routing table:
  npm run agents:routing-table
`);
}

/**
 * Discover all `.agent.md` files under `.github/agents/`.
 * @returns {Promise<string[]>} Array of absolute file paths.
 */
async function findAgentFiles() {
  const dir = path.join(repoRoot, '.github', 'agents');
  const { readdir } = await import('node:fs/promises');
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isFile() && e.name.endsWith('.agent.md'))
    .map((e) => path.join(dir, e.name))
    .toSorted();
}

/**
 * Extract the current model value from a `model:` line, handling
 * both top-level frontmatter and indented handoff entries.
 *
 * @param {string} line - A single line from the file.
 * @returns {string|null} The model value without quotes, or null if not a model line.
 */
function extractModelValue(line) {
  // Match `model: 'value'` or `model: value` (with optional surrounding quotes)
  const match = /^\s*model:\s*['"]?(.*?)['"]?\s*$/.exec(line);
  if (!match) return null;
  return match[1];
}

/**
 * Build a replacement line with the same indentation and quote style as the original.
 *
 * @param {string} originalLine - The original `model:` line.
 * @param {string} newModel - The new model string.
 * @returns {string} The replacement line.
 */
function buildReplacementLine(originalLine, newModel) {
  const indentMatch = /^(\s*)/.exec(originalLine);
  const indent = indentMatch ? indentMatch[1] : '';
  // Detect quote style: single, double, or none
  const valueMatch = /^(\s*model:\s*)(['"]?)(.*?)(['"]?)\s*$/.exec(
    originalLine,
  );
  const prefix = valueMatch ? valueMatch[1] : `${indent}model: `;
  const openQuote = valueMatch?.[2] || "'";
  const closeQuote = valueMatch?.[4] || "'";
  return `${prefix}${openQuote}${newModel}${closeQuote}`;
}

/**
 * Process a single agent file: find and replace model strings, or remove
 * top-level frontmatter model: lines.
 *
 * @param {string} filePath - Absolute path to the .agent.md file.
 * @param {string|null} from - Model string to find (null = any model: value).
 * @param {string} to - Replacement model string.
 * @param {boolean} dryRun - If true, don't write changes.
 * @param {boolean} removeMode - If true, remove top-level model: lines instead of replacing.
 * @returns {Promise<{file: string, changes: Array<{line: number, from: string, to: string}>, skipped: boolean}>}
 */
async function processFile(filePath, from, to, dryRun, removeMode) {
  const content = await readFile(filePath, 'utf8');
  const lines = content.split(/\r?\n/);
  const changes = [];
  let modified = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const currentValue = extractModelValue(line);
    if (currentValue === null) continue;

    if (removeMode) {
      // Only remove top-level (non-indented) model: lines
      if (/^\s/.test(line)) continue;
      changes.push({
        line: i + 1,
        from: currentValue,
        to: '(removed)',
      });
      lines.splice(i, 1);
      i--; // Adjust index after splice
      modified = true;
    } else {
      // Replace mode
      const shouldReplace =
        from !== null ? currentValue === from : currentValue !== to;
      if (!shouldReplace) continue;
      changes.push({
        line: i + 1,
        from: currentValue,
        to,
      });
      lines[i] = buildReplacementLine(line, to);
      modified = true;
    }
  }

  if (modified && !dryRun) {
    await writeFile(filePath, lines.join('\n'), 'utf8');
  }

  return {
    file: path.relative(repoRoot, filePath),
    changes,
    skipped: !modified,
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  if (!opts.remove && !opts.to) {
    console.error(
      'Error: --to=<model-string> is required (or use --remove to remove model fields).',
    );
    console.error('Run with --help for usage.');
    process.exit(1);
  }

  const files = await findAgentFiles();
  const results = [];
  let totalChanges = 0;

  for (const filePath of files) {
    const result = await processFile(
      filePath,
      opts.from,
      opts.to ?? '',
      opts.dryRun,
      opts.remove,
    );
    results.push(result);
    totalChanges += result.changes.length;
  }

  const report = {
    name: 'update-agent-models',
    ok: true,
    dryRun: opts.dryRun,
    mode: opts.remove ? 'remove' : 'replace',
    from: opts.remove
      ? '(top-level model: fields)'
      : (opts.from ?? '(any model: value)'),
    to: opts.remove ? '(removed)' : opts.to,
    filesScanned: files.length,
    filesModified: results.filter((r) => !r.skipped).length,
    totalChanges,
    results,
  };

  report.summaryText = [
    `${opts.dryRun ? '[DRY RUN] ' : ''}update-agent-models`,
    `scanned=${report.filesScanned}`,
    `modified=${report.filesModified}`,
    `changes=${totalChanges}`,
  ].join(' ');

  if (opts.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(report.summaryText);
    for (const result of results) {
      if (result.skipped) continue;
      for (const change of result.changes) {
        console.log(
          `  ${result.file}:${change.line}  '${change.from}' → '${change.to}'`,
        );
      }
    }
    if (totalChanges > 0 && !opts.dryRun) {
      console.log(
        '\nNext: refresh the routing table with `npm run agents:routing-table`',
      );
    }
  }
}

main().catch((error) => {
  console.error('Fatal error:', error.message);
  process.exit(1);
});
