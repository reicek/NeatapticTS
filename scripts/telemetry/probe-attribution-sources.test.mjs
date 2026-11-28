/**
 * @module probe-attribution-sources.test
 * @description RED contract home for the executable attribution probe.
 *
 * Slice: OT-S1-3 of `plans/opentelemetry-slice-telemetry.plans.md`.
 * Acceptance criteria pinned by this home:
 *   - AC-005-A (the executable demonstration)
 *   - AC-005-B (the evidence resolves)
 *   - AC-013-A (the dependents are gated)
 *
 * THE SUBJECT UNDER TEST IS `scripts/telemetry/probe-attribution-sources.mjs`,
 * which the GREEN lane implements. This home resolves it LAZILY inside every
 * assertion, so an unimplemented probe is an ASSERTION FAILURE that names the
 * missing contract - never an import failure that would abort the whole file.
 *
 * THE CONTRACT THIS HOME DEMANDS OF THE PROBE
 * --------------------------------------------
 * 1. The probe is a RUNNABLE SCRIPT. This home executes it as a child process
 *    (`node scripts/telemetry/probe-attribution-sources.mjs`, cwd = the
 *    repository root) and reads the report from its STANDARD OUTPUT. The
 *    report MUST be a single JSON document on stdout; diagnostics belong on
 *    stderr. Persisting a copy of the report is OPTIONAL and its location is
 *    NOT pinned by this home - the stdout document IS the machine-readable
 *    report this home asserts against.
 * 2. The report carries ONE ENTRY PER REQUIRED SIGNAL, and every entry carries
 *    an EXPLICIT VERDICT that a dependent slice can read without parsing
 *    prose: either `demonstrated` or `unavailable` (AC-013-A).
 * 3. The report carries ONE ENTRY PER REQUIRED SIGNAL, and every entry carries
 *    an EXPLICIT VERDICT that a dependent slice can read without parsing
 *    prose: either `demonstrated` or `unavailable` (AC-013-A).
 * 4. EVERY signal - demonstrated OR unavailable - names its evidence with a
 *    repo-relative `file:line`. The tracker's ratified contract demands that
 *    the probe "must name, with a resolving `file:line`" each of the five
 *    obligations, and two of those obligations are ABSENCES (the absence of a
 *    documented session-id environment variable, and the absence of a
 *    per-slice LLM token-consumption source). An absence is named by the site
 *    where it is evident: the repo-invented environment fallbacks for the
 *    session id, and the declared-but-never-measured token fields for the LLM
 *    consumption.
 * 5. EVERY signal - demonstrated OR unavailable - names at least one citation
 *    whose line ACTUALLY SUPPORTS THE CLAIM, not merely one that resolves.
 *    Three rules make "supports" precise:
 *      (a) the cited line must be ABOUT the signal it claims - a monotonic
 *          clock read for the duration, a `token_count` for the delivered
 *          context tokens, the declared model-consumption fields for the LLM
 *          consumption, a host post-tool event or a tool-name payload field
 *          for the tool executions, a session id or the dispatch binding for
 *          the session-to-slice binding;
 *      (b) a citation into the tracker (`plans/`) never counts, because the
 *          tracker and the research artifact DESCRIBE the sources rather than
 *          being sources, and a citation into a TEST file never counts,
 *          because a test is not a source either;
 *      (c) a signal claimed as `demonstrated` may not cite a bare quoted
 *          field name, because "a schema field that DECLARES `prompt_tokens`
 *          is not a source that MEASURES them" - and the same reasoning
 *          disqualifies a `duration_ms` schema entry as a duration source.
 *          A signal claimed as `unavailable` MAY cite a declaration, because
 *          naming the declared-but-never-measured field is exactly how an
 *          absence is evidenced.
 * 6. An `unavailable` signal names a NON-EMPTY REASON and NEVER a number
 *    (AC-004, AC-013-A).
 * 7. EVERY `file:line` the report cites resolves to an existing file and an
 *    existing line in the LIVE repository. A citation that does not resolve
 *    fails this home (AC-005-B).
 * 8. The probe INSPECTS the live repository. It must not hardcode the
 *    citations it reports, and its verdicts must agree with what the live
 *    repository actually contains (AC-005-A).
 * 9. The research artifact
 *    `plans/completed/opentelemetry-slice-telemetry.research.md`
 *    agrees with the probe report on EVERY signal (AC-005-B).
 *
 * NOTE ON PERSISTENCE. The tracker's ratified contract also says the probe
 * "persists a machine-readable report". The tracker does NOT pin the path of
 * that persisted copy, so this home verifies the report CONTENT through the
 * probe's stdout document - which is the same report - rather than inventing
 * a path the tracker never agreed on. The GREEN lane should still persist the
 * report, as the contract asks.
 *
 * THE REQUIRED SIGNALS AND THEIR CANONICAL IDENTIFIERS
 * -----------------------------------------------------
 * The canonical identifier of a signal is the identifier this home searches for
 * in the research artifact; the probe may report a signal under any of its
 * documented ALIASES and this home normalises the alias onto the canonical id.
 * The five signals are the ones the tracker's acceptance criteria name:
 *   - the per-slice duration                  -> `slice_duration`
 *   - the per-slice delivered context tokens  -> `context_tokens_delivered`
 *   - the LLM input/output/cache consumption  -> `llm_token_consumption`
 *   - the per-slice tool-EXECUTION count      -> `tool_executions`
 *   - the session-to-slice binding            -> `session_to_slice_binding`
 * `context_tokens_delivered` and `tool_executions` are the tracker's own field
 * names; the ledger field is `usage.tool_executions` and NEVER `usage.tool_calls`.
 *
 * THE VERDICTS THIS HOME PINS AS ORACLES (rather than as free choices)
 * ---------------------------------------------------------------------
 * The tracker's ratified instrumentation contract and its measured-evidence
 * table decide FOUR of the five signals, so the home checks the probe against
 * an INDEPENDENTLY COMPUTED expectation read from the live repository instead
 * of trusting the probe:
 *   - the tool-EXECUTION source: if a repository hook manifest registers a
 *     post-tool event, the source EXISTS and the signal MUST be `demonstrated`
 *     with a citation into the host hook surface. If no manifest registers one,
 *     the signal MUST be `unavailable` with a reason.
 *   - the LLM token-consumption signal: the tracker's measured answer is
 *     "There is NO per-slice LLM token-consumption source in this repository
 *     today, and none is coming", so the signal MUST be `unavailable` with a
 *     reason. A schema field that DECLARES `prompt_tokens` is not a source
 *     that MEASURES them, and this home says so in its failure detail.
 *   - the delivered-context-token signal: the tracker's table records the
 *     per-slice `token_count` as "available now", so when a live source
 *     computes it the signal MUST be `demonstrated` and must cite that source.
 *   - the duration signal: the tracker's table records a monotonic elapsed
 *     measurement as "available now" (owned by layer B), so when the live
 *     repository contains a monotonic elapsed-measurement site the signal MUST
 *     be `demonstrated` and must cite a line about a duration or a monotonic
 *     clock.
 * The FIFTH signal - the session-to-slice binding - is asserted for SHAPE
 * only, and this is DELIBERATE. The tracker decides that no DOCUMENTED
 * session-id environment variable exists (so the repo owns the binding at
 * dispatch time), but it does NOT state whether the binding itself counts as
 * available today, and the two readings (the payload carries the session id,
 * so the source exists; the dispatch-time producer that stamps the slice id
 * does not exist yet, so it does not) are both defensible. Pinning either one
 * would make this home a plan change rather than a plan check. The GREEN lane
 * must therefore decide that verdict DELIBERATELY, and whichever way it goes
 * the signal must still name its evidence and, when demonstrated, cite a line
 * that supports the claim.
 *
 * THE TOLERANCES ARE DELIBERATE AND ARE DOCUMENTED WHERE THEY ARE USED
 * ---------------------------------------------------------------------
 * The tracker pins the CONTENT of the report, not its descriptor shape, so the
 * home normalises: the container of the signals (array or keyed map), the name
 * of the verdict field, the name of the citation field, the name of the reason
 * field, and the spelling of a signal identifier. The assertions on the
 * CONTENT are not tolerant: a missing signal, an unresolvable citation, a
 * missing reason, a fabricated number, a hardcoded citation or a research
 * artifact that disagrees all fail this home.
 *
 * @example
 * npm run jest:mjs -- --testPathPatterns="scripts/telemetry/probe-attribution-sources"
 */

import { describe, it, expect } from '@jest/globals';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const PROBE_RELATIVE_PATH = path.join('scripts', 'telemetry', 'probe-attribution-sources.mjs');
const PROBE_PATH = path.join(REPO_ROOT, PROBE_RELATIVE_PATH);
const RESEARCH_ARTIFACT_RELATIVE_PATH = path.join(
  'plans',
  'completed',
  'opentelemetry-slice-telemetry.research.md',
);
const RESEARCH_ARTIFACT_PATH = path.join(REPO_ROOT, RESEARCH_ARTIFACT_RELATIVE_PATH);
const WORKFLOW_MCP_RELATIVE_PATH = path.join(
  'scripts',
  'agent-customization',
  'mcp',
  'neataptic-workflow-mcp.mjs',
);

/**
 * The required signals and the identifiers this home accepts for each of them.
 * `canonical` is the identifier the research artifact must publish.
 */
const REQUIRED_SIGNALS = Object.freeze([
  {
    canonical: 'slice_duration',
    label: 'the per-slice duration',
    aliases: ['slice_duration', 'duration', 'per_slice_duration', 'duration_ms'],
  },
  {
    canonical: 'context_tokens_delivered',
    label: 'the per-slice delivered context tokens',
    aliases: [
      'context_tokens_delivered',
      'context_tokens',
      'delivered_context_tokens',
      'rag_context_tokens',
    ],
  },
  {
    canonical: 'llm_token_consumption',
    label: 'the LLM input/output/cache token consumption',
    aliases: [
      'llm_token_consumption',
      'llm_tokens',
      'model_token_consumption',
      'prompt_completion_tokens',
    ],
  },
  {
    canonical: 'tool_executions',
    label: 'the per-slice tool-EXECUTION count',
    aliases: ['tool_executions', 'tool_execution', 'tool_execution_count'],
  },
  {
    canonical: 'session_to_slice_binding',
    label: 'the session-to-slice binding',
    aliases: ['session_to_slice_binding', 'session_slice_binding', 'slice_binding'],
  },
]);

/** The two verdicts a dependent slice may read. */
const DEMONSTRATED = 'demonstrated';
const UNAVAILABLE = 'unavailable';

/** Keys that would carry a measured quantity if a signal had one. */
const MEASUREMENT_KEYS = Object.freeze([
  'value',
  'count',
  'quantity',
  'measurement',
  'measure',
  'duration_ms',
  'durationMs',
  'tokens',
  'executions',
  'total',
  'number',
  'amount',
  'observed',
  'observed_count',
]);

/** Field names that may hold the signals collection of the report. */
const SIGNAL_COLLECTION_KEYS = Object.freeze([
  'signals',
  'findings',
  'results',
  'entries',
  'verdicts',
  'sources',
  'attribution_sources',
  'attributionSources',
]);

/** Field names that may hold the citations of one signal. */
const CITATION_KEYS = Object.freeze([
  'citations',
  'sources',
  'evidence',
  'cited_sources',
  'citedSources',
  'locations',
  'references',
]);

/** Field names that may hold the reason of an unavailable signal. */
const REASON_KEYS = Object.freeze([
  'reason',
  'unavailable_reason',
  'unavailableReason',
  'explanation',
  'detail',
  'details',
  'note',
  'notes',
]);

/**
 * The concept each signal's evidence must be ABOUT, for EVERY verdict. A
 * citation that resolves but says nothing about the signal is not evidence.
 *
 * `slice_duration` - a MONOTONIC CLOCK read. A duration is measured by
 *   reading a clock twice, so a line that merely NAMES `duration_ms` in a
 *   schema is not a duration measurement site.
 * `context_tokens_delivered` - the per-slice `token_count`.
 * `llm_token_consumption` - the model-consumption fields (declared, never
 *   measured).
 * `tool_executions` - the host hook event or the tool-name payload field.
 * `session_to_slice_binding` - a session id or the dispatch-record binding.
 */
const SIGNAL_EVIDENCE_PATTERNS = Object.freeze({
  slice_duration:
    /performance\.now|resolveHighResolutionNow|process\.hrtime|highresolutionnow|monotonic|Date\.now/i,
  context_tokens_delivered: /token_?count|context token/i,
  llm_token_consumption:
    /prompt_?tokens?|completion_?tokens?|total_?tokens?|token_?usage|model consumption/i,
  tool_executions:
    /posttooluse|post_?tool_?use|tool_?name|write_?tool|tool_?use|tool execution/i,
  session_to_slice_binding: /session_?id|dispatch/i,
});

/** How many lines around a cited line may carry the supporting evidence. */
const EVIDENCE_WINDOW_LINES = 2;

/**
 * Normalises an identifier: lowercase, every run of non-alphanumerics folded
 * into a single underscore, and no leading or trailing underscore.
 *
 * @param {unknown} value The value to normalise.
 * @returns {string} The normalised identifier.
 */
function normalizeId(value) {
  return String(value ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

/**
 * Tells whether a citation points outside the repository (a documentation URL).
 * An external reference is not a `file:line` and is therefore not resolved.
 *
 * @param {string} value The citation target.
 * @returns {boolean} True when the target is external.
 */
function isExternalReference(value) {
  return /^https?:\/\//i.test(String(value ?? ''));
}

/**
 * Reads a report-shaped value out of a record, tolerating the several field
 * names the tracker uses for the same concept.
 *
 * @param {Record<string, unknown>} record The signal record.
 * @param {string[]} keys The candidate field names, in priority order.
 * @returns {unknown} The first present value, or undefined.
 */
function pick(record, keys) {
  for (const key of keys) {
    if (record && Object.prototype.hasOwnProperty.call(record, key)) return record[key];
  }
  return undefined;
}

/**
 * Normalises one citation into `{ file, line, external }`.
 *
 * @param {unknown} citation The raw citation.
 * @returns {{ file: string, line: number | null, external: boolean } | null} The citation.
 */
function normalizeCitation(citation) {
  if (typeof citation === 'string') {
    const trimmed = citation.trim();
    if (trimmed.length === 0) return null;
    const match = trimmed.match(/^(.+?):(\d+)(?:\s*[-–]\s*(\d+))?$/);
    if (match === null) {
      return { file: trimmed, line: null, endLine: null, external: isExternalReference(trimmed) };
    }
    return {
      file: match[1],
      line: Number(match[2]),
      endLine: match[3] === undefined ? null : Number(match[3]),
      external: isExternalReference(match[1]),
    };
  }
  if (citation === null || typeof citation !== 'object') return null;
  const file = pick(citation, ['file', 'path', 'file_path', 'filePath', 'source', 'ref', 'url']);
  if (typeof file !== 'string' || file.trim().length === 0) return null;
  const rawLine = pick(citation, ['line', 'line_number', 'lineNumber', 'ln', 'start_line']);
  let line = null;
  if (typeof rawLine === 'number' && Number.isInteger(rawLine)) line = rawLine;
  else if (typeof rawLine === 'string' && /^\d+$/.test(rawLine.trim())) line = Number(rawLine.trim());
  const rawEnd = pick(citation, ['end_line', 'endLine', 'end', 'to_line', 'toLine']);
  let endLine = null;
  if (typeof rawEnd === 'number' && Number.isInteger(rawEnd)) endLine = rawEnd;
  else if (typeof rawEnd === 'string' && /^\d+$/.test(rawEnd.trim())) endLine = Number(rawEnd.trim());
  return { file: file.trim(), line, endLine, external: isExternalReference(file) };
}

/**
 * Normalises the verdict of a signal record into `demonstrated` or
 * `unavailable`. The tracker pins the OUTCOME, not the descriptor shape, so a
 * boolean, a word or a status string are all accepted.
 *
 * @param {Record<string, unknown>} record The signal record.
 * @returns {string | null} The verdict, or null when it cannot be read.
 */
function verdictOf(record) {
  const raw = pick(record, ['verdict', 'status', 'availability', 'available', 'state']);
  if (typeof raw === 'boolean') return raw ? DEMONSTRATED : UNAVAILABLE;
  if (typeof raw !== 'string') return null;
  const normalized = raw.toLowerCase().replace(/[_\s]+/g, '-');
  if (normalized.includes('unavailabl')) return UNAVAILABLE;
  if (normalized.includes('demonstrat')) return DEMONSTRATED;
  if (normalized.includes('available')) return DEMONSTRATED;
  if (normalized.includes('missing') || normalized.includes('absent')) return UNAVAILABLE;
  return null;
}

/**
 * Reads the reason of an unavailable signal.
 *
 * @param {Record<string, unknown>} record The signal record.
 * @returns {string} The reason, empty when there is none.
 */
function reasonOf(record) {
  const raw = pick(record, REASON_KEYS);
  if (typeof raw === 'string') return raw.trim();
  if (raw !== null && typeof raw === 'object') return JSON.stringify(raw);
  return '';
}

/**
 * Collects the citations of a signal record.
 *
 * @param {Record<string, unknown>} record The signal record.
 * @returns {Array<{ file: string, line: number | null, external: boolean }>} The citations.
 */
function citationsOf(record) {
  const raw = pick(record, CITATION_KEYS);
  if (Array.isArray(raw)) {
    return raw.map(normalizeCitation).filter((entry) => entry !== null);
  }
  if (typeof raw === 'string') {
    const single = normalizeCitation(raw);
    return single === null ? [] : [single];
  }
  return [];
}

/**
 * Collects every numeric measurement a record carries, so an unavailable
 * signal can be checked for a fabricated number.
 *
 * @param {Record<string, unknown>} record The signal record.
 * @returns {Array<{ key: string, value: number }>} The numeric measurements.
 */
function numericMeasurementsOf(record) {
  const found = [];
  for (const key of MEASUREMENT_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(record, key)) continue;
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value)) found.push({ key, value });
  }
  return found;
}

/**
 * Reads the signals collection of a report, tolerating an array or a keyed map.
 *
 * @param {unknown} report The report.
 * @returns {Array<{ key: string | null, record: Record<string, unknown> }>} The entries.
 */
function collectSignalEntries(report) {
  if (Array.isArray(report)) {
    return report.map((value) => ({ key: null, record: toRecord(value) }));
  }
  if (report === null || typeof report !== 'object') return [];
  for (const key of SIGNAL_COLLECTION_KEYS) {
    const collection = report[key];
    if (Array.isArray(collection)) {
      return collection.map((value) => ({ key: null, record: toRecord(value) }));
    }
    if (collection !== null && typeof collection === 'object') {
      return Object.entries(collection).map(([entryKey, value]) => ({
        key: entryKey,
        record: toRecord(value),
      }));
    }
  }
  return [];
}

/**
 * Coerces a raw entry into a record object.
 *
 * @param {unknown} value The raw entry.
 * @returns {Record<string, unknown>} The record.
 */
function toRecord(value) {
  if (value !== null && typeof value === 'object') return value;
  if (typeof value === 'string') return { verdict: value };
  return {};
}

/**
 * Reads the identifier of a signal entry.
 *
 * @param {{ key: string | null, record: Record<string, unknown> }} entry The entry.
 * @returns {string | null} The normalised identifier, or null when absent.
 */
function identifierOf(entry) {
  const raw = pick(entry.record, ['signal', 'signal_id', 'signalId', 'name', 'id', 'key']);
  const candidate = typeof raw === 'string' && raw.trim().length > 0 ? raw : entry.key;
  if (typeof candidate !== 'string') return null;
  return normalizeId(candidate);
}

/**
 * Maps a report onto the canonical signal identifiers this home requires.
 *
 * @param {unknown} report The report.
 * @returns {Map<string, { entry: { key: string | null, record: Record<string, unknown> }, identifier: string }>} The signals.
 */
function signalsByCanonicalId(report) {
  const resolved = new Map();
  for (const entry of collectSignalEntries(report)) {
    const identifier = identifierOf(entry);
    if (identifier === null) continue;
    const match = REQUIRED_SIGNALS.find((signal) =>
      signal.aliases.map(normalizeId).includes(identifier),
    );
    if (match === undefined) continue;
    if (!resolved.has(match.canonical)) {
      resolved.set(match.canonical, { entry, identifier });
    }
  }
  return resolved;
}

/**
 * Extracts the first JSON object that carries a signals collection out of a
 * chunk of text, so a probe that prints a preamble still yields its report.
 *
 * @param {string} text The text.
 * @returns {unknown} The parsed report, or null when there is none.
 */
function extractReportJson(text) {
  const trimmed = String(text ?? '').trim();
  if (trimmed.length === 0) return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed !== null && typeof parsed === 'object') return parsed;
  } catch {
    // Not a bare JSON document; fall through to the scan below.
  }
  for (let index = trimmed.indexOf('{'); index !== -1; index = trimmed.indexOf('{', index + 1)) {
    try {
      const parsed = JSON.parse(trimmed.slice(index));
      if (parsed !== null && typeof parsed === 'object' && collectSignalEntries(parsed).length > 0) {
        return parsed;
      }
    } catch {
      // This brace does not start a parseable document; keep scanning.
    }
  }
  return null;
}

/**
 * Runs the probe as a child process against the live repository.
 *
 * @returns {{ report?: unknown, missing?: string, stdout?: string }} The outcome.
 */
function runProbe() {
  if (!existsSync(PROBE_PATH)) {
    return {
      missing:
        'the probe file scripts/telemetry/probe-attribution-sources.mjs does not exist yet, ' +
        'so no attribution source can be demonstrated',
    };
  }
  let stdout = '';
  try {
    stdout = execFileSync(process.execPath, [PROBE_PATH], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 120000,
      stdio: ['ignore', 'pipe', 'pipe'],
      maxBuffer: 32 * 1024 * 1024,
    });
  } catch (error) {
    const stderr = String(error?.stderr ?? '');
    const detail = stderr.trim().length > 0 ? stderr.trim() : String(error?.message ?? error);
    return {
      missing: `the probe scripts/telemetry/probe-attribution-sources.mjs failed to run (${detail})`,
    };
  }
  const report = extractReportJson(stdout);
  if (report === null) {
    return {
      missing:
        'the probe did not print a machine-readable report on stdout; the report must be a ' +
        'single JSON document on stdout (diagnostics belong on stderr)',
    };
  }
  return { report, stdout };
}

let probeOutcomePromise = null;

/**
 * Runs the probe once per test file and reuses the outcome.
 *
 * @returns {Promise<{ report?: unknown, missing?: string }>} The outcome.
 */
function probeOutcome() {
  if (probeOutcomePromise === null) probeOutcomePromise = Promise.resolve(runProbe());
  return probeOutcomePromise;
}

/**
 * Resolves the probe report inside the calling lane and asserts that it
 * exists. A missing or unrunnable probe becomes an ASSERTION FAILURE that
 * names the missing contract.
 *
 * @returns {Promise<unknown>} The report.
 */
async function requireReport() {
  const outcome = await probeOutcome();
  if (outcome.missing !== undefined) {
    expect(`the probe contract is unmet: ${outcome.missing}`).toBe('');
  }
  expect(outcome.report).toBeDefined();
  return outcome.report;
}

/**
 * Asserts a condition and, when it fails, reports a readable detail instead of
 * a bare `false`.
 *
 * @param {boolean} condition The condition.
 * @param {string} detail The failure detail.
 */
function expectWithDetail(condition, detail) {
  expect(condition ? 'ok' : detail).toBe('ok');
}

/**
 * Resolves a cited `file:line` against the live repository.
 *
 * @param {{ file: string, line: number | null, external: boolean }} citation The citation.
 * @returns {{ ok: boolean, detail: string }} The verdict.
 */
function resolveCitation(citation) {
  if (citation.external) {
    return { ok: false, detail: 'external reference (not a repo-relative file:line)' };
  }
  const relative = citation.file.replace(/\\/g, '/').replace(/^\.\//, '');
  const absolute = path.resolve(REPO_ROOT, relative);
  if (!absolute.startsWith(REPO_ROOT + path.sep)) {
    return { ok: false, detail: `the cited path ${relative} is outside the repository` };
  }
  if (!existsSync(absolute)) {
    return { ok: false, detail: `the cited file ${relative} does not exist` };
  }
  if (statSync(absolute).isDirectory()) {
    return { ok: false, detail: `the cited path ${relative} is a directory, not a file` };
  }
  if (citation.line === null) {
    return { ok: false, detail: `the citation for ${relative} carries no line number` };
  }
  if (citation.line < 1) {
    return { ok: false, detail: `the cited line ${citation.line} of ${relative} is not positive` };
  }
  const lineCount = readFileSync(absolute, 'utf8').split('\n').length;
  if (citation.line > lineCount) {
    return {
      ok: false,
      detail: `the cited line ${citation.line} of ${relative} is beyond the file's ${lineCount} lines`,
    };
  }
  if (citation.endLine !== null && citation.endLine !== undefined) {
    if (citation.endLine < citation.line) {
      return {
        ok: false,
        detail: `the cited range ${citation.line}-${citation.endLine} of ${relative} ends before it starts`,
      };
    }
    if (citation.endLine > lineCount) {
      return {
        ok: false,
        detail: `the cited range ${citation.line}-${citation.endLine} of ${relative} runs past the file's ${lineCount} lines`,
      };
    }
  }
  return { ok: true, detail: '' };
}

/**
 * Resolves a repo-relative citation target to an absolute path.
 *
 * @param {string} file The cited path.
 * @returns {string} The absolute path.
 */
function absolutePathFor(file) {
  return path.resolve(REPO_ROOT, file.replace(/\\/g, '/').replace(/^\.\//, ''));
}

/**
 * Reads the text around a cited line, so a citation may point at a line that
 * only becomes meaningful together with its neighbours (a field name on one
 * line and its assignment on the next).
 *
 * @param {string} absolutePath The absolute path of the cited file.
 * @param {number} line The 1-based cited line.
 * @returns {string} The window text.
 */
function evidenceWindowText(absolutePath, line, endLine) {
  const lines = readFileSync(absolutePath, 'utf8').split('\n');
  const index = line - 1;
  const lastIndex =
    endLine === null || endLine === undefined ? index : Math.max(index, endLine - 1);
  const start = Math.max(0, index - EVIDENCE_WINDOW_LINES);
  const end = Math.min(lines.length, lastIndex + EVIDENCE_WINDOW_LINES + 1);
  return lines.slice(start, end).join('\n');
}

/**
 * Reads the exact text of a cited line.
 *
 * @param {string} absolutePath The absolute path of the cited file.
 * @param {number} line The 1-based cited line.
 * @returns {string} The line text, empty when the line is out of range.
 */
function citedLineText(absolutePath, line, endLine) {
  const lines = readFileSync(absolutePath, 'utf8').split('\n');
  const index = line - 1;
  if (index < 0 || index >= lines.length) return '';
  const lastIndex = endLine === null || endLine === undefined ? index : endLine - 1;
  return lines.slice(index, lastIndex + 1).join('\n');
}

/**
 * Tells whether a cited line is nothing but a QUOTED FIELD NAME - a
 * DECLARATION of a name rather than a site that does something.
 *
 * This is the distinction the tracker itself draws: "a schema field that
 * DECLARES `prompt_tokens` is not a source that MEASURES them". A `declared`
 * signal may legitimately cite a declaration - that is how an absence is shown
 * - but a `demonstrated` signal must cite a site that actually does the work.
 *
 * @param {string} line The cited line text.
 * @returns {boolean} True when the line is a bare declaration.
 */
function isBareDeclarationLine(line) {
  return /^\s*['"`][^'"`]+['"`]\s*,?\s*$/.test(String(line ?? ''));
}

/**
 * Tells whether a cited RANGE is nothing but quoted field names - a block of
 * DECLARATIONS rather than a site that does something. A range that contains
 * any executable line is not a bare declaration range.
 *
 * @param {string} text The cited range text.
 * @returns {boolean} True when every line of the range is a bare declaration.
 */
function isBareDeclarationRange(text) {
  const lines = String(text ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  if (lines.length === 0) return false;
  return lines.every((line) => isBareDeclarationLine(line));
}

/**
 * Tells whether a citation points into the tracker itself (`plans/`). The
 * tracker and the research artifact DESCRIBE the sources; they are not
 * sources, so citing them is circular and never counts as supporting
 * evidence.
 *
 * @param {string} file The cited path.
 * @returns {boolean} True when the citation points into the tracker.
 */
function isTrackerDocument(file) {
  const relative = String(file ?? '')
    .replace(/\\/g, '/')
    .replace(/^\.\//, '');
  return relative.startsWith('plans/');
}

/**
 * Tells whether one citation supports one signal.
 *
 * @param {{ file: string, line: number | null, external: boolean }} citation The citation.
 * @param {string} canonical The canonical signal identifier.
 * @param {boolean} demonstrated Whether the signal is claimed as demonstrated.
 * @returns {boolean} True when the citation supports the signal.
 */
function citationSupportsSignal(citation, canonical, demonstrated) {
  if (citation.external || citation.line === null) return false;
  if (isTrackerDocument(citation.file)) return false;
  if (isTestFile(String(citation.file).replace(/\\/g, '/'))) return false;
  if (!resolveCitation(citation).ok) return false;
  const pattern = SIGNAL_EVIDENCE_PATTERNS[canonical];
  if (pattern === undefined) return false;
  const absolutePath = absolutePathFor(citation.file);
  if (!pattern.test(evidenceWindowText(absolutePath, citation.line, citation.endLine))) {
    return false;
  }
  if (
    demonstrated &&
    isBareDeclarationRange(citedLineText(absolutePath, citation.line, citation.endLine))
  ) {
    return false;
  }
  return true;
}

/**
 * Walks a directory and returns every text file whose source contains the
 * given literal.
 *
 * @param {string} root The absolute directory root.
 * @param {string} literal The literal to look for.
 * @param {string[]} skipDirectoryNames Directory names that are never scanned.
 * @returns {string[]} The repo-relative paths of the matching files.
 */
function findFilesContaining(root, literal, skipDirectoryNames) {
  const matches = [];
  if (!existsSync(root)) return matches;
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
        if (skipDirectoryNames.includes(entry.name)) continue;
        stack.push(absolute);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!/\.(mjs|cjs|js|ts|mts|cts)$/.test(entry.name)) continue;
      let source;
      try {
        source = readFileSync(absolute, 'utf8');
      } catch {
        continue;
      }
      if (source.includes(literal)) {
        matches.push(path.relative(REPO_ROOT, absolute).replace(/\\/g, '/'));
      }
    }
  }
  return matches;
}

/**
 * Finds the repository hook manifests and tells whether any of them registers
 * a post-tool event, which is the host surface the tool-execution counter is
 * built on.
 *
 * @returns {{ manifests: string[], registersPostToolEvent: boolean }} The finding.
 */
function inspectHookManifests() {
  const hooksDirectory = path.join(REPO_ROOT, '.github', 'hooks');
  if (!existsSync(hooksDirectory)) return { manifests: [], registersPostToolEvent: false };
  const manifests = readdirSync(hooksDirectory)
    .filter((name) => name.endsWith('.json'))
    .map((name) => path.join('.github', 'hooks', name));
  let registersPostToolEvent = false;
  for (const manifest of manifests) {
    let parsed;
    try {
      parsed = JSON.parse(readFileSync(path.join(REPO_ROOT, manifest), 'utf8'));
    } catch {
      continue;
    }
    const hooks = parsed !== null && typeof parsed === 'object' ? parsed.hooks : null;
    if (hooks === null || typeof hooks !== 'object') continue;
    const eventKeys = Object.keys(hooks).map((key) => key.replace(/[\s_-]/g, '').toLowerCase());
    if (eventKeys.some((key) => key === 'posttooluse' || key === 'posttoolusefailure')) {
      registersPostToolEvent = true;
    }
  }
  return { manifests, registersPostToolEvent };
}

/**
 * Tells whether a path is a TEST file. A test that mentions `token_count` is
 * not the source that computes it, so the oracle ignores test files.
 *
 * @param {string} relativePath The repo-relative path.
 * @returns {boolean} True when the path is a test file.
 */
function isTestFile(relativePath) {
  return (
    /\.(?:test|spec)\.[cm]?[jt]s$/.test(relativePath) ||
    /(^|\/)__tests__\//.test(relativePath) ||
    /(^|\/)test\//.test(relativePath)
  );
}

/**
 * Finds the repo-relative paths that compute the per-slice context-token count,
 * which is the source of the delivered-context-tokens signal.
 *
 * @returns {string[]} The candidate source paths.
 */
function inspectTokenCountSources() {
  const candidates = [];
  const workflowMcp = WORKFLOW_MCP_RELATIVE_PATH.replace(/\\/g, '/');
  const workflowMcpPath = path.join(REPO_ROOT, WORKFLOW_MCP_RELATIVE_PATH);
  if (existsSync(workflowMcpPath) && readFileSync(workflowMcpPath, 'utf8').includes('token_count')) {
    candidates.push(workflowMcp);
  }
  for (const relative of findFilesContaining(
    path.join(REPO_ROOT, 'rag-index'),
    'token_count',
    ['node_modules', 'data', 'snapshots', 'dist', 'coverage'],
  )) {
    candidates.push(relative);
  }
  for (const relative of findFilesContaining(
    path.join(REPO_ROOT, 'scripts', 'agent-customization', 'mcp'),
    'token_count',
    ['node_modules', 'dist', 'coverage'],
  )) {
    candidates.push(relative);
  }
  return [...new Set(candidates)].filter((relative) => !isTestFile(relative));
}

/**
 * Finds the repo-relative paths that MEASURE AN ELAPSED INTERVAL, which is the
 * source of the per-slice duration signal. A site qualifies when it reads a
 * monotonic clock AND talks about a duration or an elapsed interval: a clock
 * read on its own measures an instant, not an elapsed measurement.
 *
 * @returns {string[]} The candidate source paths.
 */
function inspectDurationMeasurementSites() {
  const matches = [];
  const root = path.join(REPO_ROOT, 'src');
  if (!existsSync(root)) return matches;
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
        if (['node_modules', 'dist', 'coverage', 'data', 'snapshots'].includes(entry.name)) continue;
        stack.push(absolute);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!/\.(?:ts|tsx|mts|cts|mjs|cjs|js)$/.test(entry.name)) continue;
      let source;
      try {
        source = readFileSync(absolute, 'utf8');
      } catch {
        continue;
      }
      const readsMonotonicClock = /performance\.now|resolveHighResolutionNow/.test(source);
      const measuresElapsed = /duration|elapsed/i.test(source);
      const relative = path.relative(REPO_ROOT, absolute).replace(/\\/g, '/');
      if (readsMonotonicClock && measuresElapsed && !isTestFile(relative)) {
        matches.push(relative);
      }
    }
  }
  return matches;
}

/**
 * Reads the research artifact.
 *
 * @returns {{ missing: boolean, text: string }} The artifact.
 */
function readResearchArtifact() {
  if (!existsSync(RESEARCH_ARTIFACT_PATH)) {
    return { missing: true, text: '' };
  }
  return { missing: false, text: readFileSync(RESEARCH_ARTIFACT_PATH, 'utf8') };
}

/**
 * Finds the lines of the research artifact that publish a signal.
 *
 * @param {string} text The artifact text.
 * @param {string} canonical The canonical signal identifier.
 * @returns {string[]} The matching lines.
 */
function artifactLinesForSignal(text, canonical) {
  const aliases = REQUIRED_SIGNALS.find((signal) => signal.canonical === canonical).aliases.map(
    normalizeId,
  );
  return text
    .split('\n')
    .filter((line) => aliases.some((alias) => normalizeId(line).includes(alias)));
}

/**
 * Extracts the fenced JSON blocks of the research artifact.
 *
 * @param {string} text The artifact text.
 * @returns {string[]} The block bodies.
 */
function extractFencedJsonBlocks(text) {
  const blocks = [];
  const pattern = /```(?:json|jsonc)?[ \t]*\r?\n([\s\S]*?)```/g;
  let match = pattern.exec(text);
  while (match !== null) {
    blocks.push(match[1]);
    match = pattern.exec(text);
  }
  return blocks;
}

describe('probe-attribution-sources (OT-S1-3)', () => {
  describe('the probe subject (AC-005-A)', () => {
    it('runs against the live repository and emits a machine-readable report', async () => {
      const report = await requireReport();

      expect(typeof report).toBe('object');
      expect(Array.isArray(report)).toBe(false);
      expect(collectSignalEntries(report).length > 0).toBe(true);
    });

    it('reports an explicit verdict for every required signal', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const missing = REQUIRED_SIGNALS.filter((signal) => !signals.has(signal.canonical)).map(
        (signal) => signal.canonical,
      );

      expectWithDetail(
        missing.length === 0,
        `the probe report does not carry a verdict for: ${missing.join(', ')}; the required ` +
          `signals are ${REQUIRED_SIGNALS.map((signal) => signal.canonical).join(', ')}`,
      );

      const unreadable = REQUIRED_SIGNALS.filter((signal) => {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) return false;
        return verdictOf(resolved.entry.record) === null;
      }).map((signal) => signal.canonical);

      expectWithDetail(
        unreadable.length === 0,
        `these signals carry no verdict this home can read (expected "${DEMONSTRATED}" or ` +
          `"${UNAVAILABLE}"): ${unreadable.join(', ')}`,
      );
    });

    it('states a demonstrated signal with a file:line source and an unavailable signal with a reason', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);

      for (const signal of REQUIRED_SIGNALS) {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) continue;
        const verdict = verdictOf(resolved.entry.record);
        if (verdict === DEMONSTRATED) {
          const citations = citationsOf(resolved.entry.record);
          const fileCitations = citations.filter((citation) => !citation.external);
          expectWithDetail(
            fileCitations.length > 0,
            `the signal "${signal.canonical}" is demonstrated but names no repo-relative ` +
              `file:line source; a demonstrated source must carry its file:line`,
          );
          expectWithDetail(
            fileCitations.some((citation) => citation.line !== null),
            `the signal "${signal.canonical}" is demonstrated but no citation carries a line number`,
          );
          continue;
        }
        if (verdict === UNAVAILABLE) {
          expectWithDetail(
            reasonOf(resolved.entry.record).length > 0,
            `the signal "${signal.canonical}" is unavailable but carries no reason; an ` +
              `unavailable signal must state WHY it is unavailable`,
          );
        }
      }
    });

    it('never reports a fabricated number for an unavailable signal', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const offenders = [];

      for (const signal of REQUIRED_SIGNALS) {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) continue;
        if (verdictOf(resolved.entry.record) !== UNAVAILABLE) continue;
        const measurements = numericMeasurementsOf(resolved.entry.record);
        if (measurements.length > 0) {
          offenders.push(
            `${signal.canonical} (${measurements
              .map((measurement) => `${measurement.key}=${measurement.value}`)
              .join(', ')})`,
          );
        }
      }

      expectWithDetail(
        offenders.length === 0,
        `an unavailable signal must never carry a number: ${offenders.join('; ')}`,
      );
    });
  });

  describe('every citation resolves against the live repository (AC-005-B)', () => {
    it('resolves every cited file:line to an existing file and an existing line', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const unresolved = [];
      let checked = 0;

      for (const entry of collectSignalEntries(report)) {
        for (const citation of citationsOf(entry.record)) {
          if (citation.external) continue;
          checked += 1;
          const resolution = resolveCitation(citation);
          if (!resolution.ok) {
            unresolved.push(`${citation.file}:${citation.line ?? '?'} (${resolution.detail})`);
          }
        }
      }

      expectWithDetail(
        checked > 0,
        'the probe report cites no repo-relative file:line at all, so nothing can be verified',
      );
      expectWithDetail(
        unresolved.length === 0,
        `these cited file:line references do not resolve in the live repository: ` +
          `${unresolved.join('; ')}`,
      );
      expect(signals.size).toBeGreaterThan(0);
    });

    it('names the evidence of every signal with a repo-relative file:line', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const unnamed = [];

      for (const signal of REQUIRED_SIGNALS) {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) continue;
        const citations = citationsOf(resolved.entry.record).filter(
          (citation) => !citation.external,
        );
        if (citations.length === 0) unnamed.push(signal.canonical);
      }

      expectWithDetail(
        unnamed.length === 0,
        `these signals name no repo-relative evidence: ${unnamed.join(', ')}. The tracker's ` +
          `ratified contract requires the probe to "name, with a resolving file:line" every ` +
          `obligation, and two of them are ABSENCES: an absence is named by the site where it ` +
          `is evident - the repo-invented session-id environment fallbacks for the session ` +
          `binding, and the declared-but-never-measured token fields for the LLM consumption`,
      );
    });

    it('cites a line that actually supports the signal it claims', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const unsupported = [];

      for (const signal of REQUIRED_SIGNALS) {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) continue;
        const demonstrated = verdictOf(resolved.entry.record) === DEMONSTRATED;
        const supported = citationsOf(resolved.entry.record).some((citation) =>
          citationSupportsSignal(citation, signal.canonical, demonstrated),
        );
        if (!supported) unsupported.push(signal.canonical);
      }

      expectWithDetail(
        unsupported.length === 0,
        `these signals cite no line that supports the claim: ${unsupported.join(', ')}. A ` +
          `resolving citation is not enough: the cited line (or a line within ` +
          `${EVIDENCE_WINDOW_LINES} of it) must be ABOUT the signal it claims - a monotonic ` +
          `clock read for the duration, a token_count for the delivered context tokens, the ` +
          `declared model-consumption fields for the LLM consumption, a host post-tool event or ` +
          `a tool-name payload field for the tool executions, and a session id or the dispatch ` +
          `binding for the session-to-slice binding. A citation into the tracker (plans/) never ` +
          `counts, because the tracker DESCRIBES the sources rather than being one. A signal ` +
          `claimed as DEMONSTRATED may not cite a bare quoted field name either, because a ` +
          `schema field that DECLARES a name is not a source that MEASURES it`,
      );
    });

    it('does not hardcode the citations it reports', async () => {
      const outcome = await probeOutcome();
      if (outcome.missing !== undefined) {
        expect(`the probe contract is unmet: ${outcome.missing}`).toBe('');
      }
      const source = readFileSync(PROBE_PATH, 'utf8');
      const hardcoded = source.match(
        /['"`]([^'"`\n]*\/[^'"`\n]*\.(?:mjs|cjs|js|ts|tsx|mts|cts|json|md)):(\d+)/g,
      );

      expectWithDetail(
        hardcoded === null,
        `the probe source hardcodes ${hardcoded === null ? '' : hardcoded.join(', ')}; the probe ` +
          `must INSPECT the live repository, so a host change breaks the probe instead of ` +
          `silently emptying the counter. (This lane is a best-effort source scan: it catches a ` +
          `literal file:line. The lane "cites a line that actually supports the signal it ` +
          `claims" is the behavioural backstop, because a hardcoded citation still has to point ` +
          `at a line that supports the claim.)`,
      );
    });
  });

  describe('the probe inspects the live host surface (AC-005-A)', () => {
    it('agrees with the live hook manifest about the tool-execution source', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const resolved = signals.get('tool_executions');
      if (resolved === undefined) {
        expect('the probe report carries no tool_executions signal').toBe('');
        return;
      }
      const verdict = verdictOf(resolved.entry.record);
      const { manifests, registersPostToolEvent } = inspectHookManifests();

      if (!registersPostToolEvent) {
        expectWithDetail(
          verdict === UNAVAILABLE,
          `no repository hook manifest registers a post-tool event, so the tool-execution ` +
            `source does not exist and the signal must be "${UNAVAILABLE}"`,
        );
        expectWithDetail(
          reasonOf(resolved.entry.record).length > 0,
          'the tool_executions signal is unavailable but carries no reason',
        );
        return;
      }

      expectWithDetail(
        verdict === DEMONSTRATED,
        `a repository hook manifest (${manifests.join(', ')}) registers a post-tool event, so ` +
          `the host tool-execution surface EXISTS and the signal must be "${DEMONSTRATED}" ` +
          `with the manifest as its evidence, not "${UNAVAILABLE}"`,
      );
      const citations = citationsOf(resolved.entry.record);
      const citesHookSurface = citations.some((citation) => {
        const relative = citation.file.replace(/\\/g, '/');
        return (
          relative.startsWith('.github/hooks/') ||
          relative.startsWith('scripts/agent-customization/hooks/')
        );
      });
      expectWithDetail(
        citesHookSurface,
        'the tool_executions signal is demonstrated but cites no host hook surface ' +
          '(expected a citation under .github/hooks/ or scripts/agent-customization/hooks/)',
      );
    });

    it('agrees with the live token-count site about the delivered-context-token source', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const resolved = signals.get('context_tokens_delivered');
      if (resolved === undefined) {
        expect('the probe report carries no context_tokens_delivered signal').toBe('');
        return;
      }
      const verdict = verdictOf(resolved.entry.record);
      const candidates = inspectTokenCountSources();

      if (candidates.length === 0) {
        expectWithDetail(
          verdict === UNAVAILABLE,
          `no live source computes a per-slice token count, so the delivered-context-token ` +
            `signal must be "${UNAVAILABLE}"`,
        );
        return;
      }

      expectWithDetail(
        verdict === DEMONSTRATED,
        `the live repository computes a per-slice token count (${candidates.join(', ')}), so ` +
          `the delivered-context-token signal must be "${DEMONSTRATED}", not "${UNAVAILABLE}"`,
      );
      const citations = citationsOf(resolved.entry.record);
      const citesSource = citations.some((citation) => {
        const relative = citation.file.replace(/\\/g, '/');
        return candidates.some((candidate) => relative === candidate || relative.endsWith(candidate));
      });
      expectWithDetail(
        citesSource,
        `the context_tokens_delivered signal is demonstrated but cites no live token-count ` +
          `source (expected one of: ${candidates.join(', ')})`,
      );
    });

    it('agrees with the live elapsed-measurement site about the duration signal', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const resolved = signals.get('slice_duration');
      if (resolved === undefined) {
        expect('the probe report carries no slice_duration signal').toBe('');
        return;
      }
      const verdict = verdictOf(resolved.entry.record);
      const sites = inspectDurationMeasurementSites();

      if (sites.length === 0) {
        expectWithDetail(
          verdict === UNAVAILABLE,
          `no live source measures an elapsed interval, so the duration signal must be ` +
            `"${UNAVAILABLE}"`,
        );
        return;
      }

      expectWithDetail(
        verdict === DEMONSTRATED,
        `the live repository measures an elapsed interval (${sites
          .slice(0, 3)
          .join(', ')}), so the duration signal must be "${DEMONSTRATED}", not "${UNAVAILABLE}"; ` +
          `the tracker records the monotonic elapsed measurement as "available now", owned by ` +
          `layer B`,
      );
      const supports = citationsOf(resolved.entry.record).some((citation) =>
        citationSupportsSignal(citation, 'slice_duration', true),
      );
      expectWithDetail(
        supports,
        'the duration signal is demonstrated but cites no line that supports the claim ' +
          '(expected a line that READS a monotonic clock - performance.now, ' +
          'resolveHighResolutionNow, process.hrtime or Date.now - not a line that merely ' +
          'names duration_ms in a schema)',
      );
    });

    it('reports the LLM token-consumption signal as unavailable with a reason', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const resolved = signals.get('llm_token_consumption');
      if (resolved === undefined) {
        expect('the probe report carries no llm_token_consumption signal').toBe('');
        return;
      }

      expectWithDetail(
        verdictOf(resolved.entry.record) === UNAVAILABLE,
        'the measured answer is that this repository has NO per-slice LLM token-consumption ' +
          `source today, so the llm_token_consumption signal must be "${UNAVAILABLE}"; a schema ` +
          'field that DECLARES prompt_tokens is not a source that MEASURES them',
      );
      expectWithDetail(
        reasonOf(resolved.entry.record).length > 0,
        'the llm_token_consumption signal is unavailable but carries no reason',
      );
    });
  });

  describe('the research artifact agrees with the probe report (AC-005-B)', () => {
    it('publishes every signal verdict in the research artifact', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const artifact = readResearchArtifact();

      expectWithDetail(
        !artifact.missing,
        `the research artifact plans/completed/opentelemetry-slice-telemetry.research.md does not exist yet, ` +
          `so the probe's findings are not published`,
      );

      for (const signal of REQUIRED_SIGNALS) {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) continue;
        const verdict = verdictOf(resolved.entry.record);
        const candidates = artifactLinesForSignal(artifact.text, signal.canonical);
        const pattern = verdict === DEMONSTRATED ? /demonstrat/i : /unavailabl/i;
        const matching = candidates.filter((line) => pattern.test(line));
        expectWithDetail(
          matching.length > 0,
          `the research artifact does not publish the verdict of "${signal.canonical}" ` +
            `(${verdict}); expected a line that names the signal and states "${verdict}"`,
        );
      }
    });

    it('points the research artifact at the same evidence for every demonstrated signal', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const artifact = readResearchArtifact();

      expectWithDetail(
        !artifact.missing,
        'the research artifact plans/completed/opentelemetry-slice-telemetry.research.md does not exist yet',
      );

      for (const signal of REQUIRED_SIGNALS) {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) continue;
        if (verdictOf(resolved.entry.record) !== DEMONSTRATED) continue;
        const citations = citationsOf(resolved.entry.record).filter(
          (citation) => !citation.external,
        );
        const candidates = artifactLinesForSignal(artifact.text, signal.canonical);
        const matching = candidates.filter((line) =>
          citations.some((citation) => {
            const relative = citation.file.replace(/\\/g, '/');
            const base = path.posix.basename(relative);
            return line.includes(relative) || (base.length > 0 && line.includes(base));
          }),
        );
        expectWithDetail(
          matching.length > 0,
          `the research artifact publishes the verdict of "${signal.canonical}" but never points ` +
            `at the same evidence the probe cites ` +
            `(${citations.map((citation) => citation.file).join(', ')})`,
        );
      }
    });

    it('agrees with any machine-readable report embedded in the research artifact', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const artifact = readResearchArtifact();

      expectWithDetail(
        !artifact.missing,
        'the research artifact plans/completed/opentelemetry-slice-telemetry.research.md does not exist yet',
      );

      const embeddedReports = [];
      for (const block of extractFencedJsonBlocks(artifact.text)) {
        let parsed;
        try {
          parsed = JSON.parse(block);
        } catch {
          continue;
        }
        if (parsed === null || typeof parsed !== 'object') continue;
        if (collectSignalEntries(parsed).length > 0) embeddedReports.push(parsed);
      }

      for (const embedded of embeddedReports) {
        const embeddedSignals = signalsByCanonicalId(embedded);
        for (const [canonical, resolved] of embeddedSignals.entries()) {
          const probeSignal = signals.get(canonical);
          if (probeSignal === undefined) continue;
          const embeddedVerdict = verdictOf(resolved.entry.record);
          const probeVerdict = verdictOf(probeSignal.entry.record);
          expectWithDetail(
            embeddedVerdict === probeVerdict,
            `the research artifact embeds a report that says "${canonical}" is ` +
              `"${embeddedVerdict}" while the probe says "${probeVerdict}"; the artifact and the ` +
              `probe must agree on every signal`,
          );
        }
      }
    });
  });

  describe('the dependents are gated (AC-013-A)', () => {
    it('carries a per-signal verdict a dependent slice can read without a measurement', async () => {
      const report = await requireReport();
      const signals = signalsByCanonicalId(report);
      const verdicts = [];

      for (const signal of REQUIRED_SIGNALS) {
        const resolved = signals.get(signal.canonical);
        if (resolved === undefined) continue;
        const verdict = verdictOf(resolved.entry.record);
        verdicts.push(verdict);
        expectWithDetail(
          verdict === DEMONSTRATED || verdict === UNAVAILABLE,
          `the signal "${signal.canonical}" carries the verdict ${JSON.stringify(verdict)}; a ` +
            `dependent slice must be able to read exactly "${DEMONSTRATED}" or "${UNAVAILABLE}"`,
        );
      }

      expect(verdicts.length).toBe(REQUIRED_SIGNALS.length);
    });
  });
});
