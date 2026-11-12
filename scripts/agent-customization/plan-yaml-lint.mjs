/**
 * @module plan-yaml-lint
 *
 * @description Structural lint for plan YAML metadata blocks.
 *
 * This module is the INDEPENDENT falsifier for the defect class recorded in
 * learning-log event
 * `learn-20261109-f9-plan-parser-silent-skip-overindented-id-items`: a `- id:`
 * list item whose indentation exceeded the indentation of the object that owns
 * it was skipped silently by the plan parser, so a corrupted step packet still
 * parsed "cleanly" and every shape-only gate passed while the packet was
 * broken.
 *
 * The lint combines two independent views of the SAME block:
 *
 * 1. the PARSER view — a block that cannot be parsed without dropping content
 *    is reported with the stable code of the parser error
 *    (`PLAN_YAML_OVER_INDENTED_LIST_ITEM`), including the source file, the
 *    line number and the observed indentation;
 * 2. the RAW-TEXT view — the acceptance criteria of every slice are
 *    re-derived from the raw YAML and compared with the parsed slice:
 *    every `- id:` item must sit at the indentation of its sibling items
 *    (`PLAN_YAML_OVER_INDENTED_LIST_ITEM`), every `- id:` item must appear in
 *    the parsed `acceptance_criteria`
 *    (`PLAN_YAML_CRITERION_ID_MISSING`), the criterion count must match
 *    (`PLAN_YAML_CRITERION_COUNT_MISMATCH`), every criterion field must
 *    survive the parse unchanged
 *    (`PLAN_YAML_CRITERION_CONTENT_MISMATCH`), and a slice that owns a red
 *    test must declare its `red_test` key
 *    (`PLAN_YAML_SLICE_KEY_MISSING`).
 *
 * The raw-text checks do not trust the parser: a parser that silently drops a
 * criterion is caught by the count/id comparison, and a parser that folds a
 * criterion into the wrong parent is caught by the content comparison.
 */

import {
  PlanYamlParseError,
  normalizeYamlScalar,
  parsePlanYamlBlock,
} from './customization-utils.mjs';

const LIST_ITEM_PATTERN = /^-\s/u;
const FIELD_PATTERN = /^(?<key>[A-Za-z0-9_]+):(?<rest>.*)$/u;
const YAML_BLOCK_PATTERN = /```yaml\r?\n(?<yaml>[\s\S]*?)\r?\n```/gu;
const YAML_BLOCK_HEAD_PATTERN = /^\s*```yaml\r?\n(?<yaml>[\s\S]*?)\r?\n```/mu;

/** Stable machine-readable codes emitted by {@link lintPlanYamlBlock}. */
export const PLAN_YAML_LINT_CODES = Object.freeze({
  overIndentedListItem: 'PLAN_YAML_OVER_INDENTED_LIST_ITEM',
  criterionCountMismatch: 'PLAN_YAML_CRITERION_COUNT_MISMATCH',
  criterionIdMissing: 'PLAN_YAML_CRITERION_ID_MISSING',
  criterionContentMismatch: 'PLAN_YAML_CRITERION_CONTENT_MISMATCH',
  sliceKeyMissing: 'PLAN_YAML_SLICE_KEY_MISSING',
});

/** Slice keys a red-testing slice must declare. */
export const DEFAULT_REQUIRED_SLICE_KEYS = Object.freeze(['red_test']);

/** Slice goals that oblige a slice to declare its `red_test` key. */
export const DEFAULT_RED_TEST_GOALS = Object.freeze(['red-testing']);

/**
 * Returns the indentation (number of leading spaces) of a line.
 *
 * @param {string} line - Raw line text.
 * @returns {number} Indentation width.
 */
function indentOf(line) {
  // The indentation width is the number of leading whitespace characters.
  // `trimStart()` removes exactly the whitespace that `^\s*` matches, so the
  // difference of the two lengths is the indentation width. The expression is
  // deliberately free of conditional branches: every line the callers pass is
  // a string.
  return line.length - line.trimStart().length;
}

/**
 * Determines whether a trimmed line starts a YAML sequence item.
 *
 * @param {string} trimmed - Trimmed line text.
 * @returns {boolean} `true` for `- item` lines.
 */
function isListItem(trimmed) {
  return LIST_ITEM_PATTERN.test(trimmed);
}

/**
 * Parses a trimmed line into a YAML field.
 *
 * @param {string} trimmed - Trimmed line text.
 * @returns {{key: string, rest: string}|null} Field descriptor, or `null` when the line is not a field.
 */
function parseField(trimmed) {
  const match = FIELD_PATTERN.exec(trimmed);
  if (!match?.groups) return null;
  return { key: match.groups.key, rest: match.groups.rest.trim() };
}

/**
 * Determines whether a thrown value is a plan YAML parse error.
 *
 * The check is duck-typed on purpose. `customization-utils.mjs` is mocked in
 * several gate test suites; a mocked module hands out a DIFFERENT class
 * identity, so an `instanceof` test would reject a genuine parse error and
 * the lint would rethrow it and crash the gate.
 *
 * @param {unknown} error - Thrown value.
 * @returns {boolean} `true` for a plan YAML parse error.
 */
function isPlanYamlParseError(error) {
  if (error instanceof PlanYamlParseError) return true;
  return Boolean(error) && error.name === 'PlanYamlParseError';
}

/**
 * Collects every fenced YAML block of a plan document.
 *
 * @param {string} text - Plan markdown text.
 * @returns {Array<{yaml: string, startLine: number, contentStartLine: number}>} YAML
 *   blocks in document order. `startLine` is the 1-based line of the opening fence;
 *   `contentStartLine` is the 1-based line of the first YAML content line.
 */
export function collectPlanYamlBlocks(text) {
  const blocks = [];
  const source = String(text ?? '');
  for (const match of source.matchAll(YAML_BLOCK_PATTERN)) {
    const startLine = source.slice(0, match.index).split(/\r?\n/u).length;
    blocks.push({
      yaml: match.groups.yaml,
      startLine,
      contentStartLine: startLine + 1,
    });
  }
  return blocks;
}

/**
 * Extracts the first fenced YAML block of a markdown body.
 *
 * @param {string} body - Markdown body.
 * @returns {string|null} Raw YAML text, or `null` when the body has no YAML block.
 */
export function extractYamlBlock(body) {
  const match = YAML_BLOCK_HEAD_PATTERN.exec(String(body ?? ''));
  return match?.groups?.yaml ?? null;
}

/**
 * Determines whether a raw YAML block looks like a plan phase/step packet.
 *
 * Plan packets always carry a top-level `phase:` key; PlanUpdate and evidence
 * blocks do not. The distinction keeps the lint focused on the packets the
 * workflow dispatches.
 *
 * @param {string} rawYaml - Raw YAML text.
 * @returns {boolean} `true` for a phase/step packet block.
 */
export function looksLikePlanPacket(rawYaml) {
  return /^phase\s*:/mu.test(String(rawYaml ?? ''));
}

/**
 * Removes a trailing inline comment from a raw YAML scalar.
 *
 * The comment marker is only honored OUTSIDE a quoted value, so a `#` inside a
 * quoted scalar (`status: '[DONE - issue #42]'`) survives.
 *
 * @param {string} value - Raw scalar text.
 * @returns {string} Scalar text without a trailing inline comment.
 */
function stripInlineComment(value) {
  const quote = value[0];
  if (quote === "'" || quote === '"') {
    const closing = value.indexOf(quote, 1);
    if (closing !== -1) return value.slice(0, closing + 1);
  }
  const commentIndex = value.search(/\s#/u);
  return commentIndex === -1 ? value : value.slice(0, commentIndex).trim();
}

/**
 * Reads the declared status of a raw YAML block without parsing it.
 *
 * A malformed packet cannot be parsed, but its status line is still readable
 * in the raw text; the lint uses it to tell a closed packet from an active
 * one.
 *
 * The repo writes the status as a QUOTED AND BRACKETED scalar
 * (`status: '[WIP]'`, `status: '[DONE code-only ...]'`). The quotes are
 * stripped by {@link normalizeYamlScalar}; the brackets are stripped here, so
 * the returned value is the plain status token (`WIP`, `PLANNED`,
 * `DONE code-only ...`). A trailing inline comment (`status: '[WIP]' # active`)
 * is stripped as well. Without this normalization every real packet compared
 * against a bare status list as CLOSED, and the lint silently checked ZERO
 * packets.
 *
 * @param {string} rawYaml - Raw YAML text.
 * @returns {string|null} Status value, or `null` when the block declares none.
 */
export function readPacketStatus(rawYaml) {
  const match = /^status\s*:\s*(?<status>.+)$/mu.exec(String(rawYaml ?? ''));
  if (!match?.groups) return null;
  const scalar = String(
    normalizeYamlScalar(stripInlineComment(match.groups.status.trim())),
  ).trim();
  const unbracketed = scalar.replace(/^\[/u, '').replace(/\]$/u, '').trim();
  return unbracketed === '' ? scalar : unbracketed;
}

/**
 * Statuses that mark a packet as still dispatchable.
 *
 * A closed packet is an archival record: its body is history and must not
 * fail a gate. An open packet is work that will be dispatched, so it must be
 * structurally sound before it reaches an implementing agent.
 */
export const ACTIVE_PACKET_STATUSES = Object.freeze(['PLANNED', 'WIP']);

/**
 * Status prefixes that mark a packet as CLOSED.
 *
 * The repo records a closed packet as `'[DONE]'` or as a decorated variant
 * such as `'[DONE code-only 2026-10-29 - the wave container closes ...]'`.
 * A prefix match treats every `DONE ...` variant as closed, so a decorated
 * archival record can never fail a gate.
 */
export const CLOSED_PACKET_STATUS_PREFIXES = Object.freeze([
  'DONE',
  'CLOSED',
  'COMPLETE',
  'CANCELLED',
  'SUPERSEDED',
]);

/**
 * Determines whether a raw YAML block is a plan packet that is still open.
 *
 * Historical packets (and PlanUpdate evidence blocks) are excluded so a gate
 * that scans every YAML block in a plan does not fail on an archival record.
 * A block that declares no status is treated as open, because an undeclared
 * status is a malformed packet rather than a closed one. A status that is
 * declared but not recognized is also treated as open: an unknown status must
 * not silently archive a packet that is still live.
 *
 * @param {string} rawYaml - Raw YAML text.
 * @returns {boolean} `true` for an open phase/step packet block.
 */
export function isActivePlanPacket(rawYaml) {
  if (!looksLikePlanPacket(rawYaml)) return false;
  const status = readPacketStatus(rawYaml);
  if (status === null) return true;
  const token = status.trim().toUpperCase();
  if (token === '') return true;
  // A closed packet is an archival record: it must never fail a gate.
  if (
    CLOSED_PACKET_STATUS_PREFIXES.some((prefix) => token.startsWith(prefix))
  ) {
    return false;
  }
  // `PLANNED` and `WIP` are the live statuses the repo writes, so they are
  // open. A status that is neither live nor closed is UNRECOGNIZED, and it is
  // treated as OPEN as well: an unknown status is a malformed packet, not a
  // closed one, and it must not silently archive a packet that is still live.
  return true;
}

/**
 * Describes one raw slice block of a plan metadata block.
 *
 * @typedef {object} RawSliceBlock
 * @property {string|null} sliceId - Slice identifier when the block declares one.
 * @property {number} indent - Indentation of the `- ` marker that starts the block.
 * @property {number} startLine - 1-based line number of the block start.
 * @property {Array<{key: string, indent: number, line: number, value: string}>} keys - Slice-level keys.
 * @property {Array<{id: string, text: string|null, validation: string|null, line: number, indent: number}>} criteria
 *   Raw `- id:` acceptance criteria.
 * @property {number} acceptanceCriteriaItems - Number of raw list items under `acceptance_criteria`.
 * @property {boolean} hasAcceptanceCriteriaKey - Whether the block declares `acceptance_criteria`.
 * @property {number|null} acceptanceCriteriaBaseIndent - Indentation of the first raw
 *   `acceptance_criteria` list item.
 * @property {Array<{indent: number, line: number}>} acceptanceCriteriaItemIndents -
 *   Indentation of every raw `acceptance_criteria` list item.
 */

/**
 * Extracts the raw slice blocks of a plan metadata block.
 *
 * @param {string} rawYaml - Raw YAML text of a plan metadata block.
 * @returns {RawSliceBlock[]} Raw slice blocks in document order.
 */
export function extractRawSliceBlocks(rawYaml) {
  const lines = String(rawYaml ?? '').split(/\r?\n/u);
  const slicesKeyIndex = lines.findIndex((line) =>
    /^\s*slices:\s*$/u.test(line),
  );
  if (slicesKeyIndex === -1) return [];

  const keyIndent = indentOf(lines[slicesKeyIndex]);
  const itemStarts = [];
  let listBaseIndent = null;

  for (let index = slicesKeyIndex + 1; index < lines.length; index++) {
    const line = lines[index];
    const trimmed = line.trim();
    if (trimmed === '' || trimmed.startsWith('#')) continue;
    const indent = indentOf(line);
    if (!isListItem(trimmed)) {
      if (indent <= keyIndent) break;
      continue;
    }
    if (listBaseIndent === null) listBaseIndent = indent;
    if (indent !== listBaseIndent) continue;
    itemStarts.push(index);
  }

  return itemStarts.map((start, order) => {
    const end =
      order + 1 < itemStarts.length ? itemStarts[order + 1] : lines.length;
    return describeRawSlice(lines, start, end);
  });
}

/**
 * Describes the raw slice block that starts at `startIndex`.
 *
 * @param {string[]} lines - Full line array.
 * @param {number} startIndex - Index of the `- ` line that starts the slice.
 * @param {number} endIndex - Index of the first line after the slice block.
 * @returns {RawSliceBlock} Raw slice descriptor.
 */
function describeRawSlice(lines, startIndex, endIndex) {
  const itemIndent = indentOf(lines[startIndex]);
  const inlineRemainder = lines[startIndex]
    .trim()
    .slice(lines[startIndex].trim().indexOf('-') + 1)
    .trim();
  const inlineField = parseField(inlineRemainder);

  const keys = [];
  const criteria = [];
  /** Slice-level key whose (empty) value opens a nested list. */
  let pendingListKey = null;
  /** Slice-level key that owns the nested list currently being read. */
  let currentListKey = null;
  /** Indentation of the first item of the nested list currently being read. */
  let currentListBaseIndent = null;
  /** Indentation of the nested list item currently being read. */
  let currentListItemIndent = null;
  let acceptanceCriteriaItems = 0;
  let hasAcceptanceCriteriaKey = false;
  let acceptanceCriteriaBaseIndent = null;
  const acceptanceCriteriaItemIndents = [];
  /** @type {{id: string, text: string|null, validation: string|null, line: number, indent: number}|null} */
  let currentCriterion = null;

  if (inlineField !== null) {
    keys.push({
      key: inlineField.key,
      indent: itemIndent,
      line: startIndex + 1,
      value: inlineField.rest,
    });
    if (inlineField.rest === '') pendingListKey = inlineField.key;
  }

  for (let index = startIndex + 1; index < endIndex; index++) {
    const rawLine = lines[index];
    const trimmed = rawLine.trim();
    if (trimmed === '' || trimmed.startsWith('#')) continue;
    const indent = indentOf(rawLine);

    if (isListItem(trimmed)) {
      // A list item belongs to the nested list opened by the most recent
      // slice-level key whose value was empty.
      if (currentListKey === null && pendingListKey !== null) {
        currentListKey = pendingListKey;
        currentListBaseIndent = indent;
      }
      if (currentListKey === null || currentListBaseIndent === null) continue;
      if (indent < currentListBaseIndent) continue;

      currentListItemIndent = indent;
      currentCriterion = null;

      if (currentListKey === 'acceptance_criteria') {
        acceptanceCriteriaItems += 1;
        if (acceptanceCriteriaBaseIndent === null) {
          acceptanceCriteriaBaseIndent = indent;
        }
        acceptanceCriteriaItemIndents.push({ indent, line: index + 1 });

        const remainder = trimmed.slice(trimmed.indexOf('-') + 1).trim();
        const field = parseField(remainder);
        if (field?.key === 'id') {
          currentCriterion = {
            id: String(normalizeYamlScalar(field.rest)),
            text: null,
            validation: null,
            line: index + 1,
            indent,
          };
          criteria.push(currentCriterion);
        }
      }
      continue;
    }

    const field = parseField(trimmed);
    if (field === null) continue;

    if (currentListItemIndent !== null && indent > currentListItemIndent) {
      // A field deeper than the current nested list item belongs to that item.
      if (
        currentCriterion !== null &&
        (field.key === 'text' || field.key === 'validation')
      ) {
        currentCriterion[field.key] = String(normalizeYamlScalar(field.rest));
      }
      continue;
    }

    // Slice-level field: it closes the nested list that was being read.
    currentListKey = null;
    currentListBaseIndent = null;
    currentListItemIndent = null;
    currentCriterion = null;
    pendingListKey = field.rest === '' ? field.key : null;

    if (field.key === 'acceptance_criteria') {
      hasAcceptanceCriteriaKey = true;
    }
    keys.push({
      key: field.key,
      indent,
      line: index + 1,
      value: field.rest,
    });
  }

  const inlineSliceId =
    inlineField?.key === 'slice_id'
      ? String(normalizeYamlScalar(inlineField.rest))
      : null;
  const declaredSliceId = keys.find((entry) => entry.key === 'slice_id');

  return {
    sliceId:
      inlineSliceId ??
      (declaredSliceId === undefined
        ? null
        : String(normalizeYamlScalar(declaredSliceId.value))),
    indent: itemIndent,
    startLine: startIndex + 1,
    keys,
    criteria,
    acceptanceCriteriaItems,
    hasAcceptanceCriteriaKey,
    acceptanceCriteriaBaseIndent,
    acceptanceCriteriaItemIndents,
  };
}

/**
 * Compares the raw slice blocks with the parsed slice objects.
 *
 * @param {RawSliceBlock[]} rawSlices - Raw slice blocks.
 * @param {Record<string, unknown>|null|undefined} parsed - Parsed metadata object.
 * @param {object} options - Lint options. The only caller (`lintPlanYamlBlock`)
 *   always supplies an object, so this parameter has no default.
 * @param {string[]} [options.requiredSliceKeys] - Slice keys that must survive the parse.
 * @param {string[]} [options.redTestGoals] - Slice goals that oblige a `red_test` key.
 * @returns {Array<object>} Lint findings.
 */
function lintSlices(rawSlices, parsed, options) {
  const requiredSliceKeys =
    options.requiredSliceKeys ?? DEFAULT_REQUIRED_SLICE_KEYS;
  const redTestGoals = options.redTestGoals ?? DEFAULT_RED_TEST_GOALS;
  const findings = [];
  const parsedSlices = Array.isArray(parsed?.slices) ? parsed.slices : [];

  rawSlices.forEach((rawSlice, order) => {
    // RAW-ONLY CHECK: every acceptance criterion must sit at the indentation
    // of its sibling criteria. This is the cheap, deterministic falsifier for
    // the F9 defect class: an over-indented `- id:` item is dropped by the
    // parser, so the criterion count and ids no longer match.
    if (
      rawSlice.acceptanceCriteriaBaseIndent !== null &&
      rawSlice.acceptanceCriteriaItemIndents.length > 0
    ) {
      for (const item of rawSlice.acceptanceCriteriaItemIndents) {
        if (item.indent <= rawSlice.acceptanceCriteriaBaseIndent) continue;
        findings.push({
          code: PLAN_YAML_LINT_CODES.overIndentedListItem,
          severity: 'error',
          message: `Acceptance criterion of slice '${rawSlice.sliceId ?? order}' is indented ${item.indent} spaces but its sibling criteria sit at ${rawSlice.acceptanceCriteriaBaseIndent} spaces.`,
          line: item.line,
          indent: item.indent,
          baseIndent: rawSlice.acceptanceCriteriaBaseIndent,
          sliceId: rawSlice.sliceId,
        });
      }
    }

    const parsedSlice =
      parsedSlices.find(
        (candidate) =>
          rawSlice.sliceId !== null &&
          String(candidate?.slice_id ?? '') === rawSlice.sliceId,
      ) ?? parsedSlices[order];
    if (!parsedSlice || typeof parsedSlice !== 'object') return;

    const parsedCriteria = Array.isArray(parsedSlice.acceptance_criteria)
      ? parsedSlice.acceptance_criteria
      : [];
    const parsedIds = parsedCriteria.map((criterion) =>
      String(criterion?.id ?? ''),
    );

    if (
      rawSlice.hasAcceptanceCriteriaKey &&
      rawSlice.acceptanceCriteriaItems !== parsedCriteria.length
    ) {
      findings.push({
        code: PLAN_YAML_LINT_CODES.criterionCountMismatch,
        severity: 'error',
        message: `Slice '${rawSlice.sliceId ?? order}' declares ${rawSlice.acceptanceCriteriaItems} acceptance criteria in the YAML but the parsed packet carries ${parsedCriteria.length}.`,
        line: rawSlice.startLine,
        indent: rawSlice.indent,
        baseIndent: null,
        sliceId: rawSlice.sliceId,
      });
    }

    for (const criterion of rawSlice.criteria) {
      if (!parsedIds.includes(criterion.id)) {
        findings.push({
          code: PLAN_YAML_LINT_CODES.criterionIdMissing,
          severity: 'error',
          message: `Acceptance criterion '${criterion.id}' is present in the YAML but missing from the parsed acceptance_criteria.`,
          line: criterion.line,
          indent: null,
          baseIndent: null,
          sliceId: rawSlice.sliceId,
        });
        continue;
      }

      const parsedCriterion = parsedCriteria.find(
        (candidate) => String(candidate?.id ?? '') === criterion.id,
      );
      for (const field of ['text', 'validation']) {
        const rawValue = criterion[field];
        if (rawValue === null) continue;
        const parsedValue = parsedCriterion?.[field];
        if (parsedValue === undefined) {
          findings.push({
            code: PLAN_YAML_LINT_CODES.criterionContentMismatch,
            severity: 'error',
            message: `Acceptance criterion '${criterion.id}' declares '${field}' in the YAML but the parsed criterion lost it.`,
            line: criterion.line,
            indent: null,
            baseIndent: null,
            sliceId: rawSlice.sliceId,
          });
          continue;
        }
        if (String(parsedValue) !== rawValue) {
          findings.push({
            code: PLAN_YAML_LINT_CODES.criterionContentMismatch,
            severity: 'error',
            message: `Acceptance criterion '${criterion.id}' has a different '${field}' in the parsed packet than in the YAML.`,
            line: criterion.line,
            indent: null,
            baseIndent: null,
            sliceId: rawSlice.sliceId,
          });
        }
      }
    }

    const sliceGoal = String(
      normalizeYamlScalar(
        parsedSlice.goal ??
          rawSlice.keys.find((entry) => entry.key === 'goal')?.value ??
          '',
      ),
    );
    const goalRequiresRequiredKeys = redTestGoals.includes(sliceGoal);

    for (const key of requiredSliceKeys) {
      const declaration = rawSlice.keys.find((entry) => entry.key === key);
      const carriedByParse = Object.hasOwn(parsedSlice, key);

      if (declaration !== undefined && !carriedByParse) {
        findings.push({
          code: PLAN_YAML_LINT_CODES.sliceKeyMissing,
          severity: 'error',
          message: `Slice '${rawSlice.sliceId ?? order}' declares key '${key}' in the YAML but the parsed slice does not carry it.`,
          line: declaration.line,
          indent: declaration.indent,
          baseIndent: null,
          sliceId: rawSlice.sliceId,
        });
        continue;
      }

      if (declaration === undefined && goalRequiresRequiredKeys) {
        findings.push({
          code: PLAN_YAML_LINT_CODES.sliceKeyMissing,
          severity: 'error',
          message: `Slice '${rawSlice.sliceId ?? order}' has goal '${sliceGoal}' but declares no '${key}' key.`,
          line: rawSlice.startLine,
          indent: rawSlice.indent,
          baseIndent: null,
          sliceId: rawSlice.sliceId,
        });
      }
    }
  });

  return findings;
}

/**
 * Lints a plan YAML metadata block against its parsed representation.
 *
 * @param {string} rawYaml - Raw YAML text of the metadata block.
 * @param {Record<string, unknown>|null} [parsed] - Parsed metadata object, when available.
 * @param {object} [options] - Lint options.
 * @param {string|null} [options.file] - Source file path used in parse errors.
 * @param {number} [options.startLine] - 1-based line number in the source file of the
 *   first content line of the block; findings are reported with file-absolute lines.
 * @param {string[]} [options.requiredSliceKeys] - Slice keys that must survive the parse.
 * @param {string[]} [options.redTestGoals] - Slice goals that oblige a `red_test` key.
 * @returns {Array<{code: string, severity: string, message: string, line: number|null, indent: number|null, baseIndent: number|null, sliceId: string|null}>}
 *   Lint findings; an empty array means the block is structurally sound.
 */
export function lintPlanYamlBlock(rawYaml, parsed = null, options = {}) {
  if (typeof rawYaml !== 'string' || rawYaml.trim() === '') return [];

  const findings = [];
  let effectiveParsed = parsed ?? null;

  // Line numbers inside a plan YAML block are relative to the block. Shift them
  // into the source file so a finding names a location a human can open.
  const firstContentLine =
    Number.isInteger(options.startLine) && options.startLine > 0
      ? options.startLine
      : 1;
  const lineOffset = firstContentLine - 1;
  const sourceFile = options.file ?? null;

  if (effectiveParsed === null) {
    try {
      effectiveParsed = parsePlanYamlBlock(rawYaml, {
        file: sourceFile,
        startLine: firstContentLine,
      });
    } catch (error) {
      if (!isPlanYamlParseError(error)) throw error;
      findings.push({
        code: error.code,
        severity: 'error',
        message: `Plan YAML parse error: ${error.message}`,
        line: error.line,
        blockLine: error.blockLine,
        indent: error.indent,
        baseIndent: error.baseIndent,
        sliceId: null,
        file: sourceFile,
      });
    }
  }

  const sliceFindings = lintSlices(
    extractRawSliceBlocks(rawYaml),
    effectiveParsed,
    options,
  );
  for (const finding of sliceFindings) {
    findings.push({
      ...finding,
      // Every slice finding carries a block-relative line number; shift it
      // into the source file so a finding names a location a human can open.
      line: finding.line + lineOffset,
      blockLine: finding.line,
      file: sourceFile,
    });
  }

  return findings;
}
