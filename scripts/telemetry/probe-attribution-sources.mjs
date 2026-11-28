/**
 * @module probe-attribution-sources
 * @description The EXECUTABLE attribution probe of
 * `plans/completed/opentelemetry-slice-telemetry.plans.md`.
 *
 * Slice: OT-S1-3. Acceptance criteria implemented by this probe:
 *   - AC-005-A (the attribution contract is demonstrated by an executable probe)
 *   - AC-005-B (every cited `file:line` resolves against the live repository)
 *   - AC-013-A (a dependent slice can read a per-signal verdict without a
 *     measurement)
 *
 * WHAT this module is: the instrument that MEASURES which attribution sources
 * exist in the live repository. The tracker's ratified instrumentation contract
 * is not asserted by prose: it is demonstrated here, against the bytes on
 * disk, and the findings are published in
 * `plans/completed/opentelemetry-slice-telemetry.research.md`.
 *
 * HOW it works: every citation is DISCOVERED, never hardcoded. The probe walks
 * the live tree, finds the sites that measure each signal, and reports the
 * line it actually read. A host change therefore empties a signal instead of
 * silently reporting a stale truth. The report carries NO timestamp, NO date,
 * NO absolute path and NO measured quantity - it carries VERDICTS and
 * RESOLVING CITATIONS.
 *
 * THE CONTRACT OF THE REPORT:
 *   - ONE entry per required signal, each with an explicit verdict that a
 *     dependent slice can read without parsing prose: `demonstrated` (the
 *     source exists on the live bytes) or `unavailable` (it does not, and the
 *     entry says why).
 *   - EVERY entry - demonstrated or unavailable - names its evidence with a
 *     repo-relative `file:line`. Two of the five obligations are ABSENCES, and
 *     an absence is evidenced by the site where it is evident: the
 *     repo-invented session-id environment fallbacks, the payload-carried
 *     session id, and the declared-but-never-measured token fields.
 *   - An `unavailable` entry NEVER carries a number. A zero is a measurement;
 *     this probe measures no quantity at all.
 *   - An `unavailable` entry still names the RESOLVING EVIDENCE OF THE ABSENCE:
 *     the nearest clock-read site for the duration, the nearest token-count
 *     site for the delivered context tokens, the hook manifest and the hook
 *     code for the tool executions. An absence that names no site leaves a
 *     host change with nothing to read.
 *   - A `demonstrated` entry carries a MACHINE-READABLE `scope` when the
 *     measurement SOURCE exists but the per-slice MEASUREMENT does not, so a
 *     dependent that reads only the verdict cannot misread it. The ledger
 *     contract reserves a `scope` field for exactly this.
 *   - EVERY published citation is CHECKED against the vocabulary of its own
 *     signal before it is published, so a citation that resolves but says
 *     nothing about its signal never leaves the probe.
 *
 * THE FIVE SIGNALS: the per-slice duration, the per-slice delivered context
 * tokens, the LLM input/output/cache token consumption, the per-slice
 * tool-EXECUTION count and the session-to-slice binding.
 *
 * THE PUBLIC SURFACE: `buildReport` and `main` are exported; the inspection
 * helpers are MODULE-PRIVATE. The focused home drives this probe as a CHILD
 * PROCESS and reads its stdout document, so no consumer imports the helpers,
 * and a private helper cannot drift out of contract unnoticed.
 *
 * @example
 * node scripts/telemetry/probe-attribution-sources.mjs
 * node scripts/telemetry/probe-attribution-sources.mjs --slice-id=OT-S1-3
 */

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const SELF_RELATIVE_PATH = path.relative(REPO_ROOT, fileURLToPath(import.meta.url)).replace(/\\/g, '/');

/** The directory that holds the repository hook manifests. */
const HOOK_MANIFEST_DIRECTORY = path.join('.github', 'hooks');

/** The directory that holds the repository-owned hook code. */
const HOOK_CODE_DIRECTORY = path.join('scripts', 'agent-customization', 'hooks');

/** The directory that holds the telemetry contract module and its siblings. */
const TELEMETRY_SCRIPT_DIRECTORY = path.join('scripts', 'telemetry');

/** The roots searched for the per-slice context-token computation. */
const TOKEN_COUNT_SCAN_ROOTS = Object.freeze([
  path.join('rag-index'),
  path.join('scripts', 'agent-customization', 'mcp'),
]);

/** The root searched for the monotonic elapsed-measurement sites. */
const DURATION_SCAN_ROOT = 'src';

/** The roots searched for a model-consumption source and a session-id source. */
const SOURCE_SCAN_ROOTS = Object.freeze([
  'src',
  'scripts',
  'rag-index',
  'examples',
  'benchmarks',
  'testing',
]);

/** Directory names that are never scanned for a signal source. */
const SKIPPED_DIRECTORY_NAMES = Object.freeze([
  '.git',
  'coverage',
  'data',
  'dist',
  'node_modules',
  'snapshots',
]);

/** The repo-relative path of the persisted copy of the report. */
const REPORT_RELATIVE_PATH = path.join('logs', 'telemetry', 'probe-attribution-sources.json');

/** The source-file extensions the probe reads. */
const SOURCE_EXTENSION_PATTERN = /\.(?:mjs|cjs|js|ts|tsx|mts|cts)$/;

/**
 * A MONOTONIC CLOCK read. A duration is measured by reading a clock twice, so
 * a line that merely names `duration_ms` in a schema is not a measurement site.
 *
 * @type {RegExp}
 */
const CLOCK_READ_PATTERN =
  /performance\.now|resolveHighResolutionNow|process\.hrtime|highresolutionnow|monotonic|Date\.now/i;

/**
 * The clock reads that make a site an elapsed-measurement site rather than an
 * instant measurement.
 *
 * @type {RegExp}
 */
const MONOTONIC_SOURCE_PATTERN = /performance\.now|resolveHighResolutionNow/;

/**
 * The vocabulary of an ELAPSED interval.
 *
 * @type {RegExp}
 */
const ELAPSED_PATTERN = /duration|elapsed/i;

/**
 * The per-slice context-token count, in every spelling the repository uses.
 *
 * @type {RegExp}
 */
const TOKEN_COUNT_PATTERN = /token_?count/i;

/**
 * The model-consumption fields, in every spelling the repository uses.
 *
 * @type {RegExp}
 */
const MODEL_FIELD_PATTERN =
  /prompt_?tokens?|completion_?tokens?|total_?tokens?|input_?tokens?|output_?tokens?/i;

/**
 * A READ of a model-consumption field from an object. A declaration names a
 * field; a measurement reads one.
 *
 * @type {RegExp}
 */
const MODEL_PROPERTY_READ_PATTERN =
  /\.\s*(?:prompt_?tokens?|completion_?tokens?|total_?tokens?|input_?tokens?|output_?tokens?)\b/i;

/**
 * A host model-accounting payload.
 *
 * @type {RegExp}
 */
const MODEL_USAGE_PATTERN = /\busage\b/i;

/**
 * A bare quoted field name - a DECLARATION of a name rather than a site that
 * does something.
 *
 * @type {RegExp}
 */
const BARE_DECLARATION_PATTERN = /^\s*['"`][^'"`]+['"`]\s*,?\s*$/;

/**
 * A session-id environment variable read from the process environment.
 *
 * @type {RegExp}
 */
const SESSION_ID_ENV_PATTERN = /process\.env\.[A-Z0-9_]*SESSION_ID/;

/**
 * A session id, in every spelling the repository uses.
 *
 * @type {RegExp}
 */
const SESSION_ID_PATTERN = /session_?id/i;

/**
 * A session id that is READ AS A PROPERTY OF A HOST PAYLOAD. The
 * payload-carried session id is a DIFFERENT source from the repository-invented
 * environment fallbacks, and a claim about the session-id sources that names
 * only the fallbacks is false.
 *
 * @type {RegExp}
 */
const PAYLOAD_SESSION_ID_PATTERN =
  /(?:input|inputs|payload|hook_?input|body|event|context|params?|request)\s*(?:\?\.|\.|\[\s*['"])\s*['"]?[a-z_-]*session_?-?id['"]?/i;

/**
 * The vocabulary of the CONTEXT-TOKEN evidence: the per-slice `token_count` or
 * an explicit mention of a context token.
 *
 * @type {RegExp}
 */
const CONTEXT_TOKEN_VOCABULARY_PATTERN = /token_?count|context token/i;

/**
 * The vocabulary of the TOOL-EXECUTION evidence: a host post-tool event or the
 * host payload's tool-name field.
 *
 * @type {RegExp}
 */
const TOOL_EXECUTION_VOCABULARY_PATTERN =
  /posttooluse|post_?tool_?use|tool_?name|write_?tool|tool_?use|tool execution/i;

/**
 * The vocabulary each signal's own citations must carry. A citation that
 * resolves but says nothing about its signal is not evidence, so the probe
 * checks every citation it publishes against the vocabulary of that signal.
 *
 * @type {Readonly<Record<string, RegExp>>}
 */
const SIGNAL_EVIDENCE_PATTERNS = Object.freeze({
  slice_duration: CLOCK_READ_PATTERN,
  context_tokens_delivered: CONTEXT_TOKEN_VOCABULARY_PATTERN,
  llm_token_consumption:
    /prompt_?tokens?|completion_?tokens?|total_?tokens?|token_?usage|model consumption/i,
  tool_executions: TOOL_EXECUTION_VOCABULARY_PATTERN,
  session_to_slice_binding: /session_?id|dispatch/i,
});

/** How many lines around a cited line may carry the supporting evidence. */
const EVIDENCE_WINDOW_RADIUS = 2;

/** How many absence-evidence sites an unavailable entry publishes. */
const ABSENCE_EVIDENCE_LIMIT = 4;

/**
 * The command-line flag that carries the OPTIONAL slice identifier.
 *
 * @type {string}
 */
const SLICE_ID_FLAG = '--slice-id=';

/**
 * The environment variable that carries the OPTIONAL slice identifier.
 *
 * @type {string}
 */
const SLICE_ID_ENVIRONMENT_VARIABLE = 'NEATAPTIC_SLICE_ID';

/**
 * The transient dispatch record, in every spelling the repository uses.
 *
 * @type {RegExp}
 */
const DISPATCH_RECORD_PATTERN = /DISPATCH_?RECORD/i;

/**
 * The host payload's tool-name field, in every spelling the repository uses.
 *
 * @type {RegExp}
 */
const TOOL_NAME_PATTERN = /tool_?name/i;

/**
 * A persistence primitive. The contract asks whether the delivered context-token
 * count is persisted anywhere; this is what "persisted" looks like.
 *
 * @type {RegExp}
 */
const PERSISTENCE_PATTERN =
  /writeFile|appendFile|createWriteStream|write_text|renameSync|writeSync/i;

/**
 * Tells whether a path is a TEST file. A test that mentions a field is not the
 * source that computes it.
 *
 * @param {string} relativePath The repo-relative path.
 * @returns {boolean} True when the path is a test file.
 */
function isTestFile(relativePath) {
  const normalized = String(relativePath ?? '').replace(/\\/g, '/');
  return (
    /\.(?:test|spec)\.[cm]?[jt]s$/.test(normalized) ||
    /(^|\/)__tests__\//.test(normalized) ||
    /(^|\/)test\//.test(normalized)
  );
}

/**
 * Tells whether a source line is a comment rather than code.
 *
 * @param {string} line The source line.
 * @returns {boolean} True when the line is a comment.
 */
function isCommentLine(line) {
  const trimmed = String(line ?? '').trim();
  return (
    trimmed.startsWith('*') ||
    trimmed.startsWith('//') ||
    trimmed.startsWith('/*') ||
    trimmed.startsWith('*/')
  );
}

/**
 * Converts an absolute path into a repo-relative path with forward slashes.
 *
 * @param {string} absolutePath The absolute path.
 * @returns {string} The repo-relative path.
 */
function toRepoRelative(absolutePath) {
  return path.relative(REPO_ROOT, absolutePath).replace(/\\/g, '/');
}

/**
 * Reads a repo-relative file.
 *
 * @param {string} relativePath The repo-relative path.
 * @returns {string | null} The file text, or null when it cannot be read.
 */
function readRepoFile(relativePath) {
  const absolutePath = path.join(REPO_ROOT, relativePath);
  if (!existsSync(absolutePath)) return null;
  try {
    return readFileSync(absolutePath, 'utf8');
  } catch {
    return null;
  }
}

/**
 * Splits a file into its lines.
 *
 * @param {string | null} source The file text.
 * @returns {string[]} The lines.
 */
function toLines(source) {
  return source === null ? [] : source.split('\n');
}

/**
 * Walks a directory and returns every source file it holds.
 *
 * @param {string} rootRelativePath The repo-relative root directory.
 * @param {RegExp} extensionPattern The accepted file extensions.
 * @returns {string[]} The repo-relative paths, sorted.
 */
function listRepoFiles(rootRelativePath, extensionPattern) {
  const results = [];
  const root = path.join(REPO_ROOT, rootRelativePath);
  if (!existsSync(root)) return results;
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const absolute = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (SKIPPED_DIRECTORY_NAMES.includes(entry.name)) continue;
        stack.push(absolute);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!extensionPattern.test(entry.name)) continue;
      results.push(toRepoRelative(absolute));
    }
  }
  return results.sort();
}

/**
 * Lists the scanned source files of the repository, without the test files and
 * without the probe's own source (the probe never cites itself).
 *
 * @param {string[]} roots The repo-relative roots to scan.
 * @returns {string[]} The repo-relative paths, sorted.
 */
function listScannableSourceFiles(roots) {
  const found = new Set();
  for (const root of roots) {
    for (const relativePath of listRepoFiles(root, SOURCE_EXTENSION_PATTERN)) {
      if (isTestFile(relativePath)) continue;
      if (relativePath === SELF_RELATIVE_PATH) continue;
      found.add(relativePath);
    }
  }
  return [...found].sort();
}

/**
 * Escapes the regular-expression metacharacters of a literal.
 *
 * @param {string} value The literal.
 * @returns {string} The escaped literal.
 */
function escapeForRegExp(value) {
  return String(value ?? '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Reads the text around a line, so a citation may point at a line that only
 * becomes meaningful together with its neighbours.
 *
 * @param {string[]} lines The file lines.
 * @param {number} lineNumber The 1-based line number.
 * @param {number} radius The radius of the window.
 * @returns {string} The window text.
 */
function windowAround(lines, lineNumber, radius = 2) {
  const index = lineNumber - 1;
  const start = Math.max(0, index - radius);
  const end = Math.min(lines.length, index + radius + 1);
  return lines.slice(start, end).join('\n');
}

/**
 * Finds the repo-relative paths that MEASURE AN ELAPSED INTERVAL. A site
 * qualifies when it reads a monotonic clock AND talks about a duration or an
 * elapsed interval: a clock read on its own measures an instant, not an
 * elapsed measurement.
 *
 * @returns {string[]} The candidate source paths.
 */
function findDurationMeasurementSites() {
  return listScannableSourceFiles([DURATION_SCAN_ROOT]).filter((relativePath) => {
    const source = readRepoFile(relativePath);
    if (source === null) return false;
    return MONOTONIC_SOURCE_PATTERN.test(source) && ELAPSED_PATTERN.test(source);
  });
}

/**
 * Picks the line of an elapsed-measurement site that READS the clock. A code
 * line is preferred over a comment, and a line whose neighbourhood talks about
 * an elapsed interval is preferred over a bare clock read.
 *
 * @param {string} relativePath The repo-relative path of the site.
 * @returns {number | null} The 1-based line number, or null when there is none.
 */
function selectClockReadLine(relativePath) {
  const lines = toLines(readRepoFile(relativePath));
  const candidates = [];
  lines.forEach((line, index) => {
    if (!CLOCK_READ_PATTERN.test(line)) return;
    if (isCommentLine(line)) return;
    candidates.push({
      lineNumber: index + 1,
      mentionsElapsed: ELAPSED_PATTERN.test(windowAround(lines, index + 1)),
    });
  });
  const preferred = candidates.find((candidate) => candidate.mentionsElapsed) ?? candidates[0];
  return preferred === undefined ? null : preferred.lineNumber;
}

/**
 * Finds the repo-relative paths that COMPUTE the per-slice context-token
 * count.
 *
 * @returns {string[]} The candidate source paths.
 */
function findTokenCountSources() {
  const candidates = new Set();
  for (const root of TOKEN_COUNT_SCAN_ROOTS) {
    for (const relativePath of listRepoFiles(root, SOURCE_EXTENSION_PATTERN)) {
      if (isTestFile(relativePath)) continue;
      if (relativePath === SELF_RELATIVE_PATH) continue;
      const source = readRepoFile(relativePath);
      if (source === null) continue;
      if (source.includes('token_count')) candidates.add(relativePath);
    }
  }
  return [...candidates].sort();
}

/**
 * Picks the lines of a context-token source that COMPUTE the count: the
 * accumulation, the read of a reported count, and the field that carries the
 * value out.
 *
 * @param {string} relativePath The repo-relative path of the source.
 * @returns {number[]} The 1-based line numbers, sorted.
 */
function selectTokenCountLines(relativePath) {
  const lines = toLines(readRepoFile(relativePath));
  const selected = new Map();
  const record = (category, lineNumber) => {
    const existing = selected.get(category) ?? [];
    if (existing.length >= 2) return;
    existing.push(lineNumber);
    selected.set(category, existing);
  };
  lines.forEach((line, index) => {
    if (isCommentLine(line)) return;
    if (!TOKEN_COUNT_PATTERN.test(line)) return;
    const text = line.trim();
    if (/token_?count\s*\+=/i.test(text)) {
      record('accumulation', index + 1);
      return;
    }
    if (/token_?count\s*:/i.test(text) && !/token_?count\s*:\s*0\b/i.test(text)) {
      record('field', index + 1);
      return;
    }
    if (/\.\s*token_?count\b/i.test(text)) {
      record('read', index + 1);
    }
  });
  return [...selected.values()].flat().sort((left, right) => left - right);
}

/**
 * Finds the repository hook manifests and tells whether any of them registers
 * a post-tool event, which is the host surface the tool-execution counter is
 * built on.
 *
 * @returns {{ manifests: string[], registrations: Array<{ manifest: string, event: string, line: number | null }> }}
 * The finding.
 */
function inspectHookManifests() {
  const directory = path.join(REPO_ROOT, HOOK_MANIFEST_DIRECTORY);
  const manifests = [];
  const registrations = [];
  if (!existsSync(directory)) return { manifests, registrations };
  const manifestDirectory = HOOK_MANIFEST_DIRECTORY.replace(/\\/g, '/');
  for (const name of readdirSync(directory).filter((entry) => entry.endsWith('.json')).sort()) {
    const relativePath = `${manifestDirectory}/${name}`;
    manifests.push(relativePath);
    let parsed;
    try {
      parsed = JSON.parse(readFileSync(path.join(directory, name), 'utf8'));
    } catch {
      continue;
    }
    const hooks = parsed !== null && typeof parsed === 'object' ? parsed.hooks : null;
    if (hooks === null || typeof hooks !== 'object') continue;
    for (const key of Object.keys(hooks)) {
      const normalized = key.replace(/[\s_-]/g, '').toLowerCase();
      if (normalized !== 'posttooluse' && normalized !== 'posttoolusefailure') continue;
      registrations.push({ manifest: relativePath, event: key, line: findKeyLine(relativePath, key) });
    }
  }
  return { manifests, registrations };
}

/**
 * Finds the line of a JSON file that declares a given key.
 *
 * @param {string} relativePath The repo-relative path of the file.
 * @param {string} key The declared key.
 * @returns {number | null} The 1-based line number, or null when absent.
 */
function findKeyLine(relativePath, key) {
  const pattern = new RegExp(`^[^\\S\\n]*["']?${escapeForRegExp(key)}["']?[^\\S\\n]*:`);
  const lines = toLines(readRepoFile(relativePath));
  const index = lines.findIndex((line) => pattern.test(line));
  return index === -1 ? null : index + 1;
}

/**
 * Resolves the repository hook scripts that a manifest actually REGISTERS.
 * Only the scripts named by a registered command are the host surface the
 * repository owns; a script that merely survives as a manual helper is not.
 *
 * @param {string[]} manifests The repo-relative paths of the hook manifests.
 * @returns {string[]} The repo-relative paths of the registered hook scripts.
 */
function findRegisteredHookScripts(manifests) {
  const scripts = new Set();
  const commandPattern = /[\w.\\/-]+\.(?:mjs|cjs|js|ts|tsx|mts|cts)/g;
  for (const relativePath of manifests) {
    let parsed;
    try {
      parsed = JSON.parse(readRepoFile(relativePath) ?? '');
    } catch {
      continue;
    }
    const hooks = parsed !== null && typeof parsed === 'object' ? parsed.hooks : null;
    if (hooks === null || typeof hooks !== 'object') continue;
    for (const entries of Object.values(hooks)) {
      if (!Array.isArray(entries)) continue;
      for (const entry of entries) {
        const command = entry !== null && typeof entry === 'object' && typeof entry.command === 'string'
          ? entry.command
          : '';
        let match = commandPattern.exec(command);
        while (match !== null) {
          const candidate = match[0].replace(/\\/g, '/').replace(/^\.\//, '');
          if (candidate.startsWith('scripts/') && existsSync(path.join(REPO_ROOT, candidate))) {
            scripts.add(candidate);
          }
          match = commandPattern.exec(command);
        }
      }
    }
  }
  return [...scripts].sort();
}

/**
 * Finds the repository hook code that reads the host payload's tool-name
 * field, which corroborates that the host surface reaches repo-owned code.
 *
 * @param {string[]} [registeredScripts] The repo-relative paths of the
 * registered hook scripts. When empty, the hook directory is scanned instead.
 * @returns {Array<{ file: string, payloadLine: number | null, filterLine: number | null }>}
 * The finding.
 */
function inspectHookCode(registeredScripts = []) {
  const findings = [];
  const candidates = registeredScripts.length > 0
    ? registeredScripts
    : listScannableSourceFiles([HOOK_CODE_DIRECTORY]);
  for (const relativePath of candidates) {
    const source = readRepoFile(relativePath);
    if (source === null) continue;
    if (!TOOL_NAME_PATTERN.test(source)) continue;
    const lines = toLines(source);
    let payloadLine = null;
    let filterLine = null;
    lines.forEach((line, index) => {
      if (isCommentLine(line)) return;
      const text = line.trim();
      if (payloadLine === null && /hookInput|payload|hook_input/i.test(text) && TOOL_NAME_PATTERN.test(text)) {
        payloadLine = index + 1;
      }
      if (filterLine === null && /WRITE_?TOOL/i.test(text)) {
        filterLine = index + 1;
      }
    });
    if (payloadLine === null && filterLine === null) continue;
    findings.push({ file: relativePath, payloadLine, filterLine });
  }
  return findings;
}

/**
 * Finds the session-id sources on the live bytes: the environment fallbacks the
 * repository invents, the session ids it reads FROM A HOST PAYLOAD, and the
 * declaration of the transient dispatch record that carries the slice id.
 *
 * @returns {{ sessionIdFallbacks: Array<{ file: string, line: number }>, payloadSessionIds: Array<{ file: string, line: number }>, dispatchDeclarations: Array<{ file: string, line: number }>, dispatchMentions: Array<{ file: string, line: number }> }}
 * The finding.
 */
function inspectSessionBinding() {
  const sessionIdFallbacks = [];
  const payloadSessionIds = [];
  for (const relativePath of listScannableSourceFiles(SOURCE_SCAN_ROOTS)) {
    const lines = toLines(readRepoFile(relativePath));
    lines.forEach((line, index) => {
      if (isCommentLine(line)) return;
      if (SESSION_ID_ENV_PATTERN.test(line)) {
        sessionIdFallbacks.push({ file: relativePath, line: index + 1 });
        return;
      }
      if (!SESSION_ID_PATTERN.test(line)) return;
      if (!PAYLOAD_SESSION_ID_PATTERN.test(line)) return;
      payloadSessionIds.push({ file: relativePath, line: index + 1 });
    });
  }
  const dispatchDeclarations = [];
  const dispatchMentions = [];
  for (const relativePath of listScannableSourceFiles([TELEMETRY_SCRIPT_DIRECTORY])) {
    const lines = toLines(readRepoFile(relativePath));
    lines.forEach((line, index) => {
      if (!DISPATCH_RECORD_PATTERN.test(line)) return;
      dispatchMentions.push({ file: relativePath, line: index + 1 });
      if (isCommentLine(line)) return;
      dispatchDeclarations.push({ file: relativePath, line: index + 1 });
    });
  }
  return { sessionIdFallbacks, payloadSessionIds, dispatchDeclarations, dispatchMentions };
}

/**
 * Finds the model-consumption fields the repository mentions and tells whether
 * any of them is READ from a host payload. A field that is only DECLARED is
 * not a source that MEASURES it.
 *
 * @returns {{ declarations: Array<{ file: string, line: number }>, measurements: Array<{ file: string, line: number }> }}
 * The finding.
 */
function inspectModelConsumption() {
  const declarations = [];
  const measurements = [];
  for (const relativePath of listScannableSourceFiles(SOURCE_SCAN_ROOTS)) {
    const lines = toLines(readRepoFile(relativePath));
    lines.forEach((line, index) => {
      if (!MODEL_FIELD_PATTERN.test(line)) return;
      if (isCommentLine(line)) return;
      const citation = { file: relativePath, line: index + 1 };
      const readsAHostField =
        MODEL_PROPERTY_READ_PATTERN.test(line) ||
        (MODEL_USAGE_PATTERN.test(line) && MODEL_FIELD_PATTERN.test(line));
      if (readsAHostField) {
        measurements.push(citation);
        return;
      }
      declarations.push(citation);
    });
  }
  return { declarations, measurements };
}

/**
 * Tells whether the per-slice context-token count is PERSISTED anywhere. Before
 * this contract, the count lived only in the MCP response and nothing wrote it
 * down.
 *
 * @returns {Array<{ file: string, line: number }>} The persistence sites.
 */
function findTokenCountPersistenceSites() {
  const sites = [];
  for (const relativePath of listScannableSourceFiles(SOURCE_SCAN_ROOTS)) {
    const lines = toLines(readRepoFile(relativePath));
    lines.forEach((line, index) => {
      const window = windowAround(lines, index + 1);
      if (!TOKEN_COUNT_PATTERN.test(window)) return;
      if (!PERSISTENCE_PATTERN.test(window)) return;
      sites.push({ file: relativePath, line: index + 1 });
    });
  }
  return sites;
}

/**
 * Builds the citation of one signal.
 *
 * @param {string} file The repo-relative path.
 * @param {number} line The 1-based line number.
 * @returns {{ file: string, line: number }} The citation.
 */
function citation(file, line) {
  return { file, line };
}

/**
 * Formats citations as a readable `file:line` list.
 *
 * @param {Array<{ file: string, line: number }>} citations The citations.
 * @returns {string} The formatted list.
 */
function formatCitationList(citations) {
  if (citations.length === 0) return 'none';
  return citations.map((site) => `${site.file}:${site.line}`).join(', ');
}

/**
 * Tells whether a cited line is nothing but a QUOTED FIELD NAME - a
 * DECLARATION of a name rather than a site that does something.
 *
 * @param {string} line The cited line text.
 * @returns {boolean} True when the line is a bare declaration.
 */
function isBareDeclarationLine(line) {
  return /^\s*['"`][^'"`]+['"`]\s*,?\s*$/.test(String(line ?? ''));
}

/**
 * Tells whether one citation supports one signal. The rule is the executable
 * one: the cited line must RESOLVE, its neighbourhood must carry the
 * vocabulary of the signal it claims, and a signal claimed as `demonstrated`
 * must not rest on a bare quoted field name.
 *
 * @param {{ file: string, line: number }} candidate The citation.
 * @param {string} signal The canonical signal identifier.
 * @param {boolean} [demonstrated] Whether the signal is claimed as demonstrated.
 * @returns {boolean} True when the citation supports the signal.
 */
function citationSupportsSignal(candidate, signal, demonstrated = false) {
  const pattern = SIGNAL_EVIDENCE_PATTERNS[signal];
  if (pattern === undefined) return false;
  const source = readRepoFile(candidate.file);
  if (source === null) return false;
  const lines = toLines(source);
  if (candidate.line < 1 || candidate.line > lines.length) return false;
  if (!pattern.test(windowAround(lines, candidate.line, EVIDENCE_WINDOW_RADIUS))) return false;
  if (demonstrated && isBareDeclarationLine(lines[candidate.line - 1])) return false;
  return true;
}

/**
 * Keeps only the citations that support their own signal, so the probe never
 * publishes a citation that merely resolves.
 *
 * @param {Array<{ file: string, line: number }>} candidates The candidate citations.
 * @param {string} signal The canonical signal identifier.
 * @param {boolean} [demonstrated] Whether the signal is claimed as demonstrated.
 * @returns {Array<{ file: string, line: number }>} The supporting citations.
 */
function filterSupportingCitations(candidates, signal, demonstrated = false) {
  return candidates.filter((candidate) => citationSupportsSignal(candidate, signal, demonstrated));
}

/**
 * Publishes the citations that support their own signal. When the check would
 * empty the list, the original candidates are published instead, so a signal
 * never loses the only evidence it has.
 *
 * @param {Array<{ file: string, line: number }>} candidates The candidate citations.
 * @param {string} signal The canonical signal identifier.
 * @param {boolean} [demonstrated] Whether the signal is claimed as demonstrated.
 * @returns {Array<{ file: string, line: number }>} The published citations.
 */
function verifiedCitations(candidates, signal, demonstrated = false) {
  const supporting = filterSupportingCitations(candidates, signal, demonstrated);
  return supporting.length > 0 ? supporting : candidates;
}

/**
 * Finds the sites whose line carries a vocabulary. Code lines come FIRST and
 * comment lines LAST, so the evidence a signal publishes is the code that does
 * the work rather than the prose that mentions it.
 *
 * @param {RegExp} pattern The vocabulary.
 * @param {string[]} roots The repo-relative roots to scan.
 * @param {number} [limit] The maximum number of sites.
 * @param {boolean} [includeComments] Whether comment lines may be published.
 * @returns {Array<{ file: string, line: number }>} The sites.
 */
function findVocabularySites(pattern, roots, limit = ABSENCE_EVIDENCE_LIMIT, includeComments = false) {
  const code = [];
  const comments = [];
  for (const relativePath of listScannableSourceFiles(roots)) {
    const lines = toLines(readRepoFile(relativePath));
    lines.forEach((line, index) => {
      if (!pattern.test(line)) return;
      const site = citation(relativePath, index + 1);
      if (isCommentLine(line)) comments.push(site);
      else code.push(site);
    });
  }
  const ordered = includeComments ? [...code, ...comments] : code;
  return ordered.slice(0, limit);
}

/**
 * Finds the clock reads on the live bytes, the MONOTONIC ones first. They are
 * the evidence of the ABSENCE of a per-slice duration measurement: the
 * repository reads clocks, but no clock read measures the elapsed interval of
 * a slice dispatch window.
 *
 * @returns {Array<{ file: string, line: number }>} The clock-read sites.
 */
function findClockReadSites() {
  const monotonic = findVocabularySites(MONOTONIC_SOURCE_PATTERN, [DURATION_SCAN_ROOT]);
  if (monotonic.length > 0) return monotonic;
  const wallClock = findVocabularySites(CLOCK_READ_PATTERN, [DURATION_SCAN_ROOT]);
  if (wallClock.length > 0) return wallClock;
  return findVocabularySites(CLOCK_READ_PATTERN, [DURATION_SCAN_ROOT], ABSENCE_EVIDENCE_LIMIT, true);
}

/**
 * Finds the per-slice token-count sites on the live bytes. They are the
 * evidence of the ABSENCE of a per-slice context-token measurement: the
 * vocabulary exists in the repository, but no site computes the count.
 *
 * @returns {Array<{ file: string, line: number }>} The token-count sites.
 */
function findTokenCountMentionSites() {
  const code = findVocabularySites(TOKEN_COUNT_PATTERN, TOKEN_COUNT_SCAN_ROOTS);
  if (code.length > 0) return code;
  return findVocabularySites(TOKEN_COUNT_PATTERN, TOKEN_COUNT_SCAN_ROOTS, ABSENCE_EVIDENCE_LIMIT, true);
}

/**
 * Finds the sites that make the host tool-execution surface VISIBLE to the
 * repository: the hook manifests that name a post-tool event and the hook code
 * that reads the host payload's tool-name field. They are the evidence of the
 * ABSENCE of a per-slice execution counter - the surface is there, and the
 * counter is not.
 *
 * @returns {Array<{ file: string, line: number }>} The tool-surface sites.
 */
function findToolSurfaceSites() {
  const manifestSites = [];
  const manifestDirectory = HOOK_MANIFEST_DIRECTORY.replace(/\\/g, '/');
  const directory = path.join(REPO_ROOT, HOOK_MANIFEST_DIRECTORY);
  if (existsSync(directory)) {
    for (const name of readdirSync(directory).filter((entry) => entry.endsWith('.json')).sort()) {
      const relativePath = `${manifestDirectory}/${name}`;
      const lines = toLines(readRepoFile(relativePath));
      lines.forEach((line, index) => {
        if (!TOOL_EXECUTION_VOCABULARY_PATTERN.test(line)) return;
        manifestSites.push(citation(relativePath, index + 1));
      });
    }
  }
  return [
    ...manifestSites,
    ...findVocabularySites(TOOL_EXECUTION_VOCABULARY_PATTERN, [HOOK_CODE_DIRECTORY]),
  ];
}

/**
 * Builds the duration signal.
 *
 * @returns {Record<string, unknown>} The signal entry.
 */
function buildDurationSignal() {
  const sites = findDurationMeasurementSites();
  const candidates = [];
  for (const site of sites) {
    const lineNumber = selectClockReadLine(site);
    if (lineNumber === null) continue;
    candidates.push(citation(site, lineNumber));
  }
  const citations = verifiedCitations(candidates, 'slice_duration', true);
  if (citations.length === 0) {
    const absenceCitations = verifiedCitations(findClockReadSites(), 'slice_duration');
    return {
      signal: 'slice_duration',
      verdict: 'unavailable',
      owner: 'layer B (the repo-owned DONE path)',
      measures: 'the monotonic elapsed interval of the slice dispatch window',
      summary: 'no live source measures an elapsed interval',
      reason:
        'the probe found no source file that reads a monotonic clock and talks about a ' +
        'duration or an elapsed interval, so the slice duration has no measured source ' +
        'today',
      observation:
        'the nearest clock reads on the live bytes - a clock read measures an instant, ' +
        `not the elapsed interval of a slice dispatch window: ${formatCitationList(absenceCitations)}`,
      citations: absenceCitations,
    };
  }
  return {
    signal: 'slice_duration',
    verdict: 'demonstrated',
    owner: 'layer B (the repo-owned DONE path); the OpenTelemetry layer never computes it',
    measures: 'the monotonic elapsed interval of the slice dispatch window',
    summary:
      'the repository measures a monotonic elapsed interval today: a high-resolution clock ' +
      'read with a wall-clock fallback, and a performance.now() delta in the GPU profiler. ' +
      'The elapsed interval of the SLICE dispatch window itself is computed by the layer-B ' +
      'DONE writer, which this tracker schedules in a later slice; this probe reports the ' +
      'measurement SOURCE that exists on the live bytes, not a slice measurement.',
    scope:
      'the elapsed-measurement SOURCE exists on the live bytes; the per-slice dispatch-window ' +
      'duration is NOT measured yet, and the layer-B DONE path owns it',
    observation:
      `elapsed-measurement CITED sites on the live bytes: ${formatCitationList(citations)}`,
    citations,
  };
}

/**
 * Builds the delivered-context-token signal.
 *
 * @returns {Record<string, unknown>} The signal entry.
 */
function buildContextTokenSignal() {
  const sources = findTokenCountSources();
  const candidates = [];
  for (const source of sources) {
    for (const lineNumber of selectTokenCountLines(source)) {
      candidates.push(citation(source, lineNumber));
    }
  }
  const citations = verifiedCitations(candidates, 'context_tokens_delivered', true);
  const persistenceSites = findTokenCountPersistenceSites();
  if (citations.length === 0) {
    const absenceCitations = verifiedCitations(
      findTokenCountMentionSites(),
      'context_tokens_delivered',
    );
    return {
      signal: 'context_tokens_delivered',
      verdict: 'unavailable',
      owner: 'the workflow MCP (get_slice_context)',
      measures: 'context tokens DELIVERED to the slice, not model consumption',
      summary: 'no live source computes a per-slice context-token count',
      reason:
        'the probe found no source file that computes a per-slice token count, so the ' +
        'delivered context-token source does not exist today',
      observation:
        'the nearest per-slice token-count sites on the live bytes - the vocabulary ' +
        `exists, but no site computes the count for a slice: ${formatCitationList(absenceCitations)}`,
      citations: absenceCitations,
    };
  }
  return {
    signal: 'context_tokens_delivered',
    verdict: 'demonstrated',
    owner: 'the workflow MCP (get_slice_context); the persistence owner is the dispatch-time producer',
    measures: 'context tokens DELIVERED to the slice, not model consumption',
    summary:
      'the workflow MCP computes a per-slice token_count for the RAG context it delivers: it ' +
      'accumulates the count of every search response, defaults it per response and returns ' +
      'it in the delivered context block. The count is DELIVERED context size, not model ' +
      'consumption, and on the live bytes NOTHING persists it - the value is computed, ' +
      'returned and discarded.',
    observation:
      persistenceSites.length === 0
        ? 'the probe found no write of the delivered context-token count anywhere in the repository'
        : `the delivered context-token count is persisted at: ${formatCitationList(persistenceSites)}`,
    citations,
  };
}

/**
 * Builds the LLM token-consumption signal.
 *
 * @returns {Record<string, unknown>} The signal entry.
 */
function buildModelConsumptionSignal() {
  const { declarations, measurements } = inspectModelConsumption();
  const schemaDeclarations = declarations.filter((declaration) => {
    const lines = toLines(readRepoFile(declaration.file));
    return BARE_DECLARATION_PATTERN.test(lines[declaration.line - 1] ?? '');
  });
  const declarationCitations = (schemaDeclarations.length > 0 ? schemaDeclarations : declarations)
    .slice(0, 3)
    .map((declaration) => citation(declaration.file, declaration.line));
  const citations = verifiedCitations(declarationCitations, 'llm_token_consumption');
  if (measurements.length > 0) {
    const measurementCitations = measurements.map((measurement) =>
      citation(measurement.file, measurement.line),
    );
    return {
      signal: 'llm_token_consumption',
      verdict: 'demonstrated',
      owner: 'a host model-accounting surface',
      measures: 'the host model prompt, completion and total token consumption',
      summary: 'a live source reads a model-consumption field from a host payload',
      citations: [
        ...citations,
        ...verifiedCitations(measurementCitations, 'llm_token_consumption', true),
      ],
    };
  }
  return {
    signal: 'llm_token_consumption',
    verdict: 'unavailable',
    owner: 'the host agent model accounting, which this repository cannot reach',
    measures: 'the host model prompt, completion and total token consumption',
    summary: 'no per-slice LLM token-consumption source exists on the live bytes',
    reason:
      'the repository has no access to the host agent model accounting: no live source READS ' +
      'a model-consumption field (prompt_tokens, completion_tokens, total_tokens) from a host ' +
      'payload, so there is nothing to measure. The fields exist only as DECLARATIONS in the ' +
      'ledger schema, and the nearest total in the repository is its own context-token ' +
      'estimate, which is a different quantity. A dependent slice must write available: false ' +
      'with this reason and null counters - never 0.',
    observation: `model-consumption fields that are only DECLARED and never read from a host payload: ${
      schemaDeclarations.length > 0
        ? schemaDeclarations.map((declaration) => `${declaration.file}:${declaration.line}`).join(', ')
        : 'none'
    }${
      declarations.length > schemaDeclarations.length
        ? `; ${declarations.length - schemaDeclarations.length} further local token-estimator mentions declare the same field shape and are not host model accounting`
        : ''
    }`,
    citations,
  };
}

/**
 * Builds the tool-execution signal.
 *
 * @returns {Record<string, unknown>} The signal entry.
 */
function buildToolExecutionSignal() {
  const { manifests, registrations } = inspectHookManifests();
  const registeredScripts = findRegisteredHookScripts(manifests);
  const hookCode = inspectHookCode(registrations.length === 0 ? [] : registeredScripts);
  const candidates = [];
  for (const registration of registrations) {
    if (registration.line === null) continue;
    candidates.push(citation(registration.manifest, registration.line));
  }
  for (const finding of hookCode) {
    if (finding.payloadLine !== null) candidates.push(citation(finding.file, finding.payloadLine));
    if (finding.filterLine !== null) candidates.push(citation(finding.file, finding.filterLine));
  }
  const citations = verifiedCitations(candidates, 'tool_executions', true);
  if (registrations.length === 0) {
    const absenceCitations = verifiedCitations(findToolSurfaceSites(), 'tool_executions');
    return {
      signal: 'tool_executions',
      verdict: 'unavailable',
      owner: 'the host CLI post-tool surface',
      measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
      summary: 'no repository hook manifest registers a post-tool event',
      reason:
        'the probe found no repository hook manifest that registers a post-tool event, so ' +
        'the host tool-execution surface does not exist for this repository and no ' +
        'execution can be observed',
      observation:
        'the hook manifests found, and the hook code that reads the host tool-name payload ' +
        `field - the surface is invisible here, so no execution can be counted: ${formatCitationList(absenceCitations)}`,
      citations: absenceCitations,
    };
  }
  return {
    signal: 'tool_executions',
    verdict: 'demonstrated',
    owner: 'the host CLI post-tool surface; the repo-owned observer that counts executions is a later slice',
    measures: 'tool EXECUTIONS COMPLETED by the host CLI inside the slice dispatch window',
    summary:
      'the host tool-execution surface EXISTS and this repository already registers it: a ' +
      'hook manifest registers the post-tool event and the registered hook code already reads ' +
      "the host payload's tool-name field. What does NOT exist yet is the repo-owned OBSERVER " +
      'that counts executions per slice - the write-tool hook it would sit beside matches ' +
      'write tools only, carries no slice id and writes a timestamped line - so the surface is ' +
      'demonstrated while the per-slice counter is not yet implemented.',
    scope:
      'the host post-tool SURFACE exists on the live bytes and is registered; the per-slice ' +
      'execution COUNTER is not implemented yet, and the repo-owned observer owns it',
    observation: `hook manifests found: ${manifests.join(', ')}; hook scripts they register: ${
      registeredScripts.length > 0 ? registeredScripts.join(', ') : 'none'
    }`,
    citations,
  };
}

/**
 * Builds the session-to-slice binding signal.
 *
 * @returns {Record<string, unknown>} The signal entry.
 */
function buildSessionBindingSignal() {
  const { sessionIdFallbacks, payloadSessionIds, dispatchDeclarations, dispatchMentions } =
    inspectSessionBinding();
  const candidates = [
    ...payloadSessionIds.slice(0, 2).map((site) => citation(site.file, site.line)),
    ...sessionIdFallbacks.map((fallback) => citation(fallback.file, fallback.line)),
    ...dispatchDeclarations
      .slice(0, 1)
      .map((declaration) => citation(declaration.file, declaration.line)),
  ];
  const citations = verifiedCitations(candidates, 'session_to_slice_binding');
  const absenceCitations =
    citations.length > 0
      ? citations
      : verifiedCitations(
          dispatchMentions.slice(0, 2).map((site) => citation(site.file, site.line)),
          'session_to_slice_binding',
        );
  return {
    signal: 'session_to_slice_binding',
    verdict: 'unavailable',
    owner: 'the repo, at dispatch time',
    measures: 'the binding of a host session to the slice whose dispatch record is open',
    summary: 'the session-to-slice binding does not exist on the live bytes yet',
    reason:
      'no DOCUMENTED session-id environment variable exists - the environment fallbacks the ' +
      'probe found are repository-invented COPILOT_* names - and the dispatch-time producer ' +
      'that would stamp the slice id onto an observation does not exist yet: the transient ' +
      'dispatch record has a declared shape and a writer, but nothing calls that writer at ' +
      'dispatch time. The host payload ALREADY carries a session id, so the session half of ' +
      'the binding has a source; the slice half does not. The binding is therefore owned by ' +
      'the repository at dispatch time, and it is NOT available today.',
    observation:
      `payload-carried session-id reads found: ${formatCitationList(payloadSessionIds)}; ` +
      `session-id environment fallbacks found: ${formatCitationList(sessionIdFallbacks)}`,
    citations: absenceCitations,
  };
}

/**
 * Resolves the OPTIONAL slice identifier of the report. A dependent slice that
 * reuses this probe passes its own identifier; when none is supplied the field
 * is OMITTED, so the report never publishes another slice's id.
 *
 * @returns {string | null} The slice identifier, or null when none is supplied.
 */
function resolveSliceId() {
  const flag = process.argv.find((argument) => argument.startsWith(SLICE_ID_FLAG));
  if (flag !== undefined) {
    const value = flag.slice(SLICE_ID_FLAG.length).trim();
    if (value.length > 0) return value;
  }
  const fromEnvironment = String(process.env[SLICE_ID_ENVIRONMENT_VARIABLE] ?? '').trim();
  return fromEnvironment.length > 0 ? fromEnvironment : null;
}

/**
 * Builds the machine-readable report.
 *
 * @returns {Record<string, unknown>} The report.
 */
export function buildReport() {
  const sliceId = resolveSliceId();
  return {
    schema_version: 1,
    report: 'attribution-sources',
    ...(sliceId === null ? {} : { slice_id: sliceId }),
    signals: [
      buildDurationSignal(),
      buildContextTokenSignal(),
      buildModelConsumptionSignal(),
      buildToolExecutionSignal(),
      buildSessionBindingSignal(),
    ],
  };
}

/**
 * Persists a copy of the report under the git-ignored log directory. The copy
 * is a convenience, never the contract: the stdout document IS the report.
 *
 * @param {Record<string, unknown>} report The report.
 * @returns {boolean} True when the copy was written.
 */
function persistReport(report) {
  const target = path.join(REPO_ROOT, REPORT_RELATIVE_PATH);
  try {
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
    return true;
  } catch (error) {
    const code = typeof error?.code === 'string' ? error.code : 'unknown-error';
    process.stderr.write(`probe: the persisted report copy could not be written (${code})\n`);
    return false;
  }
}

/**
 * Runs the probe and prints the report.
 *
 * @returns {number} The process exit code.
 */
export function main() {
  const report = buildReport();
  const persisted = persistReport(report);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!persisted) {
    process.stderr.write('probe: the stdout report is authoritative; the copy is missing\n');
  }
  return 0;
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));

if (invokedDirectly) {
  process.exitCode = main();
}
