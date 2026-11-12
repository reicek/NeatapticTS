/**
 * @module plan-yaml-lint.test
 *
 * @description Falsification tests for the F9 plan-YAML lint.
 *
 * The lint is the INDEPENDENT falsifier of the defect class recorded in
 * learning-log event
 * `learn-20261109-f9-plan-parser-silent-skip-overindented-id-items`: an
 * over-indented `- id:` acceptance criterion used to be skipped by the plan
 * parser, so a corrupted step packet still parsed "cleanly" and every
 * shape-only gate passed.
 *
 * The tests below prove the three properties the fix packet demands:
 *
 * 1. the parser THROWS on an over-indented structured list item, and the
 *    throw names the source file and the line;
 * 2. the lint reports the defect for EVERY packet of a document, not only for
 *    the first one;
 * 3. the raw-text view catches the defects that the parser view cannot see
 *    (a criterion that the parse dropped, changed or lost).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

// The real module MUST be imported BEFORE the mock is registered: the mock
// factory re-exports the real namespace, and a mock registered first would
// shadow this very import.
const utils = await import('./customization-utils.mjs');

/**
 * Parser used by the mocked `parsePlanYamlBlock`. A test can swap it to
 * simulate a parser that throws something other than a parse error.
 *
 * @type {Function}
 */
let parseImpl = utils.parsePlanYamlBlock;

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  ...utils,
  parsePlanYamlBlock: (...args) => parseImpl(...args),
}));

const {
  ACTIVE_PACKET_STATUSES,
  CLOSED_PACKET_STATUS_PREFIXES,
  DEFAULT_RED_TEST_GOALS,
  DEFAULT_REQUIRED_SLICE_KEYS,
  PLAN_YAML_LINT_CODES,
  collectPlanYamlBlocks,
  extractRawSliceBlocks,
  extractYamlBlock,
  isActivePlanPacket,
  lintPlanYamlBlock,
  looksLikePlanPacket,
  readPacketStatus,
} = await import('./plan-yaml-lint.mjs');

const CANARY_FILE = 'plans/canary.plans.md';

/** A packet whose acceptance criteria all sit at the same indentation. */
const WELL_FORMED_PACKET = [
  'phase: A',
  'step: 1',
  "status: '[WIP]'",
  'slices:',
  '  - slice_id: P1-S1',
  '    goal: red-testing',
  '    red_test:',
  '      - path: test/foo.test.ts',
  '    acceptance_criteria:',
  '      - id: AC-001',
  '        text: First criterion',
  '        validation: npm test',
  '      - id: AC-002',
  '        text: Second criterion',
].join('\n');

/** The F9 defect: `AC-002` is indented deeper than its sibling criteria. */
const OVER_INDENTED_PACKET = [
  'phase: A',
  'step: 1',
  "status: '[WIP]'",
  'slices:',
  '  - slice_id: P1-S1',
  '    goal: red-testing',
  '    red_test:',
  '      - path: test/foo.test.ts',
  '    acceptance_criteria:',
  '      - id: AC-001',
  '        text: First criterion',
  '        validation: npm test',
  '        - id: AC-002',
  '          text: Second criterion',
].join('\n');

/** A packet whose over-indented list item carries a PLAIN scalar. */
const PLAIN_SCALAR_OVER_INDENT_PACKET = [
  'phase: A',
  'step: 1',
  "status: '[WIP]'",
  'slices:',
  '  - slice_id: P1-S1',
  '    goal: red-testing',
  '    red_test:',
  '      - path: test/foo.test.ts',
  '    acceptance_criteria:',
  '      - id: AC-001',
  '        text: First criterion',
  '        - a plain scalar',
].join('\n');

/**
 * Runs the lint over every OPEN packet of a plan document, exactly like the
 * `plan-command-lint` gate does.
 *
 * @param {string} text - Plan markdown text.
 * @param {string} file - Repo-relative plan path.
 * @returns {Array<{startLine: number, finding: object}>} Findings per packet.
 */
function lintDocument(text, file) {
  const issues = [];
  for (const block of collectPlanYamlBlocks(text)) {
    if (!isActivePlanPacket(block.yaml)) continue;
    for (const finding of lintPlanYamlBlock(block.yaml, null, {
      file,
      startLine: block.contentStartLine,
    })) {
      issues.push({ startLine: block.contentStartLine, finding });
    }
  }
  return issues;
}

/**
 * Returns the findings of one code.
 *
 * @param {Array<object>} findings - Lint findings.
 * @param {string} code - Finding code.
 * @returns {Array<object>} Matching findings.
 */
function findingsWithCode(findings, code) {
  return findings.filter((finding) => finding.code === code);
}

afterEach(() => {
  parseImpl = utils.parsePlanYamlBlock;
});

describe('plan-yaml-lint: exported contract', () => {
  it('exposes the stable finding codes', () => {
    assert.equal(
      PLAN_YAML_LINT_CODES.overIndentedListItem,
      'PLAN_YAML_OVER_INDENTED_LIST_ITEM',
    );
    assert.equal(
      PLAN_YAML_LINT_CODES.criterionCountMismatch,
      'PLAN_YAML_CRITERION_COUNT_MISMATCH',
    );
    assert.equal(
      PLAN_YAML_LINT_CODES.criterionIdMissing,
      'PLAN_YAML_CRITERION_ID_MISSING',
    );
    assert.equal(
      PLAN_YAML_LINT_CODES.criterionContentMismatch,
      'PLAN_YAML_CRITERION_CONTENT_MISMATCH',
    );
    assert.equal(
      PLAN_YAML_LINT_CODES.sliceKeyMissing,
      'PLAN_YAML_SLICE_KEY_MISSING',
    );
  });

  it('exposes the status and slice-key vocabularies', () => {
    assert.deepEqual([...ACTIVE_PACKET_STATUSES], ['PLANNED', 'WIP']);
    assert.deepEqual([...DEFAULT_REQUIRED_SLICE_KEYS], ['red_test']);
    assert.deepEqual([...DEFAULT_RED_TEST_GOALS], ['red-testing']);
    assert.deepEqual(
      [...CLOSED_PACKET_STATUS_PREFIXES],
      ['DONE', 'CLOSED', 'COMPLETE', 'CANCELLED', 'SUPERSEDED'],
    );
  });
});

describe('plan-yaml-lint: block collection', () => {
  it('collects every fenced YAML block with its source line', () => {
    const text = [
      '# Plan',
      '',
      '```yaml',
      'phase: A',
      '```',
      '',
      'text',
      '',
      '```yaml',
      'phase: B',
      '```',
    ].join('\n');
    const blocks = collectPlanYamlBlocks(text);
    assert.equal(blocks.length, 2);
    assert.equal(blocks[0].yaml, 'phase: A');
    assert.equal(blocks[0].startLine, 3);
    assert.equal(blocks[0].contentStartLine, 4);
    assert.equal(blocks[1].yaml, 'phase: B');
    assert.equal(blocks[1].startLine, 9);
    assert.equal(blocks[1].contentStartLine, 10);
  });

  it('returns no blocks for an empty or missing document', () => {
    assert.deepEqual(collectPlanYamlBlocks('no yaml here'), []);
    assert.deepEqual(collectPlanYamlBlocks(undefined), []);
    assert.deepEqual(collectPlanYamlBlocks(null), []);
  });

  it('extracts the first YAML block of a markdown body', () => {
    assert.equal(
      extractYamlBlock('text\n```yaml\nphase: A\n```\n'),
      'phase: A',
    );
    assert.equal(extractYamlBlock('no yaml here'), null);
    assert.equal(extractYamlBlock(undefined), null);
  });

  it('recognizes a plan packet by its `phase:` key', () => {
    assert.equal(looksLikePlanPacket('phase: A\nstatus: x'), true);
    assert.equal(looksLikePlanPacket('type: PlanUpdate\nstatus: x'), false);
    assert.equal(looksLikePlanPacket(undefined), false);
  });
});

describe('plan-yaml-lint: status normalization', () => {
  it('strips the quotes and the brackets the repo writes', () => {
    assert.equal(readPacketStatus("status: '[WIP]'"), 'WIP');
    assert.equal(readPacketStatus("status: '[PLANNED]'"), 'PLANNED');
    assert.equal(readPacketStatus('status: WIP'), 'WIP');
  });

  it('strips a trailing inline comment', () => {
    assert.equal(readPacketStatus("status: '[WIP]' # active"), 'WIP');
    assert.equal(readPacketStatus('status: WIP # active'), 'WIP');
  });

  it('honors a double-quoted status', () => {
    assert.equal(readPacketStatus('status: "[DONE]"'), 'DONE');
  });

  it('keeps a `#` that lives inside a quoted scalar', () => {
    assert.equal(
      readPacketStatus("status: '[DONE - issue #42]'"),
      'DONE - issue #42',
    );
  });

  it('falls back to the comment strip for an unterminated quote', () => {
    // The quote is never closed, so the value keeps its opening quote; the
    // bracket normalization still removes the trailing bracket.
    assert.equal(readPacketStatus("status: '[WIP] # active"), "'[WIP");
  });

  it('returns null when the block declares no status', () => {
    assert.equal(readPacketStatus('phase: A\nstep: 1'), null);
    assert.equal(readPacketStatus(undefined), null);
  });

  it('returns an empty status for an empty status declaration', () => {
    assert.equal(readPacketStatus("status: ''"), '');
  });
});

describe('plan-yaml-lint: open vs closed packets', () => {
  it('rejects a block that is not a plan packet', () => {
    assert.equal(isActivePlanPacket('type: PlanUpdate\nstatus: open'), false);
  });

  it('treats a packet without a status as open', () => {
    assert.equal(isActivePlanPacket('phase: A\nstep: 1'), true);
  });

  it('treats an empty status as open', () => {
    assert.equal(isActivePlanPacket("phase: A\nstatus: ''"), true);
  });

  it('treats the live statuses as open', () => {
    assert.equal(isActivePlanPacket("phase: A\nstatus: '[WIP]'"), true);
    assert.equal(isActivePlanPacket("phase: A\nstatus: '[PLANNED]'"), true);
  });

  it('treats every closed status prefix as closed', () => {
    for (const prefix of CLOSED_PACKET_STATUS_PREFIXES) {
      assert.equal(
        isActivePlanPacket(`phase: A\nstatus: '[${prefix}]'`),
        false,
        `${prefix} must be closed`,
      );
    }
  });

  it('treats a decorated DONE status as closed', () => {
    assert.equal(
      isActivePlanPacket(
        "phase: A\nstatus: '[DONE code-only 2026-10-29 - the wave closes]'",
      ),
      false,
    );
  });

  it('treats an UNRECOGNIZED status as open so it cannot be archived', () => {
    assert.equal(isActivePlanPacket('phase: A\nstatus: green-light'), true);
  });
});

describe('plan-yaml-lint: F9 canary — the parser throws', () => {
  it('throws on an over-indented `- id:` item and names file and line', () => {
    let error = null;
    try {
      utils.parsePlanYamlBlock(OVER_INDENTED_PACKET, {
        file: CANARY_FILE,
        startLine: 4,
      });
    } catch (caught) {
      error = caught;
    }
    assert.ok(error, 'the parser must throw on an over-indented list item');
    assert.ok(error instanceof utils.PlanYamlParseError);
    assert.equal(error.code, 'PLAN_YAML_OVER_INDENTED_LIST_ITEM');
    assert.equal(error.file, CANARY_FILE);
    // The offending `- id: AC-002` is the 13th line of the block, and the
    // block starts at source line 4, so the source line is 4 + 12 = 16.
    assert.equal(error.blockLine, 13);
    assert.equal(error.line, 16);
    assert.equal(error.indent, 8);
    assert.equal(error.baseIndent, 6);
    assert.ok(error.message.includes(`${CANARY_FILE}:16`));
    assert.ok(error.message.includes('indent 8'));
  });

  it('tolerates an over-indented PLAIN scalar (documented residual)', () => {
    let error = null;
    try {
      utils.parsePlanYamlBlock(PLAIN_SCALAR_OVER_INDENT_PACKET, {
        file: CANARY_FILE,
        startLine: 1,
      });
    } catch (caught) {
      error = caught;
    }
    assert.equal(error, null);
  });

  it('parses a well-formed packet without dropping a criterion', () => {
    const parsed = utils.parsePlanYamlBlock(WELL_FORMED_PACKET);
    assert.equal(parsed.slices.length, 1);
    assert.equal(parsed.slices[0].slice_id, 'P1-S1');
    assert.equal(parsed.slices[0].goal, 'red-testing');
    assert.deepEqual(parsed.slices[0].red_test, [{ path: 'test/foo.test.ts' }]);
    assert.equal(parsed.slices[0].acceptance_criteria.length, 2);
    assert.equal(parsed.slices[0].acceptance_criteria[0].id, 'AC-001');
    assert.equal(parsed.slices[0].acceptance_criteria[1].id, 'AC-002');
  });
});

describe('plan-yaml-lint: raw slice extraction', () => {
  it('returns no slices when the block declares no `slices:` key', () => {
    assert.deepEqual(extractRawSliceBlocks('phase: A\nstep: 1'), []);
    assert.deepEqual(extractRawSliceBlocks(undefined), []);
  });

  it('describes the criteria of a well-formed slice', () => {
    const slices = extractRawSliceBlocks(WELL_FORMED_PACKET);
    assert.equal(slices.length, 1);
    assert.equal(slices[0].sliceId, 'P1-S1');
    assert.equal(slices[0].acceptanceCriteriaItems, 2);
    assert.equal(slices[0].acceptanceCriteriaBaseIndent, 6);
    assert.equal(slices[0].hasAcceptanceCriteriaKey, true);
    assert.deepEqual(
      slices[0].criteria.map((criterion) => criterion.id),
      ['AC-001', 'AC-002'],
    );
    assert.equal(slices[0].criteria[0].text, 'First criterion');
    assert.equal(slices[0].criteria[0].validation, 'npm test');
    assert.equal(slices[0].criteria[1].text, 'Second criterion');
    assert.equal(slices[0].criteria[1].validation, null);
  });

  it('records the over-indented criterion and its base indentation', () => {
    const slices = extractRawSliceBlocks(OVER_INDENTED_PACKET);
    assert.equal(slices.length, 1);
    assert.equal(slices[0].acceptanceCriteriaBaseIndent, 6);
    assert.deepEqual(
      slices[0].acceptanceCriteriaItemIndents.map((item) => item.indent),
      [6, 8],
    );
  });

  it('describes several slices of one packet', () => {
    const raw = [
      'slices:',
      '  - slice_id: P1-S1',
      '    goal: implementing',
      '  - slice_id: P1-S2',
      '    goal: red-testing',
    ].join('\n');
    const slices = extractRawSliceBlocks(raw);
    assert.equal(slices.length, 2);
    assert.equal(slices[0].sliceId, 'P1-S1');
    assert.equal(slices[1].sliceId, 'P1-S2');
  });

  it('describes a slice whose key list opens inline', () => {
    const raw = [
      'slices:',
      '  - acceptance_criteria:',
      '      - id: AC-010',
      '        text: Inline criterion',
    ].join('\n');
    const slices = extractRawSliceBlocks(raw);
    assert.equal(slices.length, 1);
    assert.equal(slices[0].sliceId, null);
    assert.equal(slices[0].acceptanceCriteriaItems, 1);
    assert.deepEqual(
      slices[0].criteria.map((criterion) => criterion.id),
      ['AC-010'],
    );
  });

  it('describes a slice that declares its slice_id on its own line', () => {
    const raw = [
      'slices:',
      '  - goal: implementing',
      '    slice_id: P1-S9',
    ].join('\n');
    const slices = extractRawSliceBlocks(raw);
    assert.equal(slices[0].sliceId, 'P1-S9');
  });

  it('ignores blank lines, comments and the lines after the slice list', () => {
    const raw = [
      'slices:',
      '  - slice_id: P1-S1',
      '    # a comment',
      '',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
      'status: done',
    ].join('\n');
    const slices = extractRawSliceBlocks(raw);
    assert.equal(slices.length, 1);
    assert.equal(slices[0].criteria.length, 1);
  });
});

describe('plan-yaml-lint: findings', () => {
  it('reports nothing for a well-formed packet', () => {
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET);
    assert.deepEqual(findings, []);
  });

  it('reports the parse error and the raw over-indent for a broken packet', () => {
    const findings = lintPlanYamlBlock(OVER_INDENTED_PACKET, null, {
      file: CANARY_FILE,
      startLine: 4,
    });
    const overIndented = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.overIndentedListItem,
    );
    assert.equal(overIndented.length, 2);

    const parseFinding = overIndented.find((finding) =>
      finding.message.startsWith('Plan YAML parse error:'),
    );
    assert.ok(parseFinding, 'the parser view must report the parse error');
    assert.equal(parseFinding.file, CANARY_FILE);
    assert.equal(parseFinding.line, 16);
    assert.equal(parseFinding.blockLine, 13);
    assert.equal(parseFinding.indent, 8);
    assert.equal(parseFinding.baseIndent, 6);

    const rawFinding = overIndented.find(
      (finding) => !finding.message.startsWith('Plan YAML parse error:'),
    );
    assert.ok(rawFinding, 'the raw view must report the over-indent');
    assert.equal(rawFinding.line, 16);
    assert.equal(rawFinding.blockLine, 13);
    assert.equal(rawFinding.indent, 8);
    assert.equal(rawFinding.baseIndent, 6);
    assert.ok(rawFinding.message.includes('indented 8 spaces'));
  });

  it('reports a criterion count mismatch', () => {
    const findings = lintPlanYamlBlock(
      WELL_FORMED_PACKET,
      {
        slices: [
          {
            slice_id: 'P1-S1',
            goal: 'red-testing',
            red_test: [{ path: 'test/foo.test.ts' }],
            acceptance_criteria: [
              { id: 'AC-001', text: 'First criterion', validation: 'npm test' },
            ],
          },
        ],
      },
      { file: CANARY_FILE },
    );
    const mismatches = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.criterionCountMismatch,
    );
    assert.equal(mismatches.length, 1);
    assert.ok(mismatches[0].message.includes('declares 2 acceptance criteria'));
    assert.ok(mismatches[0].message.includes('carries 1'));
  });

  it('reports a criterion id that the parse dropped', () => {
    const findings = lintPlanYamlBlock(
      WELL_FORMED_PACKET,
      {
        slices: [
          {
            slice_id: 'P1-S1',
            goal: 'red-testing',
            red_test: [{ path: 'test/foo.test.ts' }],
            acceptance_criteria: [
              { id: 'AC-001', text: 'First criterion', validation: 'npm test' },
              { id: 'AC-003', text: 'Second criterion' },
            ],
          },
        ],
      },
      { file: CANARY_FILE },
    );
    const missing = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.criterionIdMissing,
    );
    assert.equal(missing.length, 1);
    assert.ok(missing[0].message.includes("'AC-002'"));
  });

  it('reports a criterion whose content changed during the parse', () => {
    const findings = lintPlanYamlBlock(
      WELL_FORMED_PACKET,
      {
        slices: [
          {
            slice_id: 'P1-S1',
            goal: 'red-testing',
            red_test: [{ path: 'test/foo.test.ts' }],
            acceptance_criteria: [
              {
                id: 'AC-001',
                text: 'A DIFFERENT criterion',
                validation: 'npm test',
              },
              { id: 'AC-002', text: 'Second criterion' },
            ],
          },
        ],
      },
      { file: CANARY_FILE },
    );
    const mismatches = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.criterionContentMismatch,
    );
    assert.equal(mismatches.length, 1);
    assert.ok(mismatches[0].message.includes("different 'text'"));
  });

  it('reports a criterion field the parse lost', () => {
    const findings = lintPlanYamlBlock(
      WELL_FORMED_PACKET,
      {
        slices: [
          {
            slice_id: 'P1-S1',
            goal: 'red-testing',
            red_test: [{ path: 'test/foo.test.ts' }],
            acceptance_criteria: [
              { id: 'AC-001', text: 'First criterion' },
              { id: 'AC-002', text: 'Second criterion' },
            ],
          },
        ],
      },
      { file: CANARY_FILE },
    );
    const mismatches = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.criterionContentMismatch,
    );
    assert.equal(mismatches.length, 1);
    assert.ok(mismatches[0].message.includes("declares 'validation'"));
    assert.ok(mismatches[0].message.includes('lost it'));
  });

  it('reports a slice key the parse did not carry', () => {
    const findings = lintPlanYamlBlock(
      WELL_FORMED_PACKET,
      {
        slices: [
          {
            slice_id: 'P1-S1',
            goal: 'red-testing',
            acceptance_criteria: [
              { id: 'AC-001', text: 'First criterion', validation: 'npm test' },
              { id: 'AC-002', text: 'Second criterion' },
            ],
          },
        ],
      },
      { file: CANARY_FILE },
    );
    const missing = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.sliceKeyMissing,
    );
    assert.equal(missing.length, 1);
    assert.ok(missing[0].message.includes("declares key 'red_test'"));
  });

  it('reports a red-testing slice that declares no red_test key', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - slice_id: P1-S1',
      '    goal: red-testing',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, {
      slices: [
        {
          slice_id: 'P1-S1',
          goal: 'red-testing',
          acceptance_criteria: [{ id: 'AC-001', text: 'First criterion' }],
        },
      ],
    });
    const missing = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.sliceKeyMissing,
    );
    assert.equal(missing.length, 1);
    assert.ok(missing[0].message.includes("goal 'red-testing'"));
    assert.ok(missing[0].message.includes("declares no 'red_test' key"));
  });

  it('reads the goal from the raw YAML when the parse dropped it', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - slice_id: P1-S1',
      '    goal: implementing',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, {
      slices: [
        {
          slice_id: 'P1-S1',
          acceptance_criteria: [{ id: 'AC-001', text: 'First criterion' }],
        },
      ],
    });
    assert.deepEqual(findings, []);
  });

  it('accepts a slice without any goal', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - slice_id: P1-S1',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, {
      slices: [
        {
          slice_id: 'P1-S1',
          acceptance_criteria: [{ id: 'AC-001', text: 'First criterion' }],
        },
      ],
    });
    assert.deepEqual(findings, []);
  });

  it('honors the requiredSliceKeys and redTestGoals overrides', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - slice_id: P1-S1',
      '    goal: implementing',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(
      raw,
      {
        slices: [
          {
            slice_id: 'P1-S1',
            goal: 'implementing',
            acceptance_criteria: [{ id: 'AC-001', text: 'First criterion' }],
          },
        ],
      },
      {
        requiredSliceKeys: ['custom_key'],
        redTestGoals: ['implementing'],
      },
    );
    const missing = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.sliceKeyMissing,
    );
    assert.equal(missing.length, 1);
    assert.ok(missing[0].message.includes("declares no 'custom_key' key"));
  });

  it('ignores a parsed slice that is not an object', () => {
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET, {
      slices: ['not an object'],
    });
    assert.deepEqual(findings, []);
  });

  it('ignores a parsed packet without slices', () => {
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET, { phase: 'A' });
    assert.deepEqual(findings, []);
  });

  it('ignores a parsed criterion that is not an object', () => {
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET, {
      slices: [
        {
          slice_id: 'P1-S1',
          goal: 'red-testing',
          red_test: [{ path: 'test/foo.test.ts' }],
          acceptance_criteria: [null],
        },
      ],
    });
    // The parsed criterion has no id, so the raw criterion is reported as
    // missing and the count does not match.
    assert.equal(
      findingsWithCode(findings, PLAN_YAML_LINT_CODES.criterionIdMissing)
        .length,
      2,
    );
  });

  it('falls back to the positional slice when no slice_id matches', () => {
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET, {
      slices: [
        {
          slice_id: 'OTHER',
          goal: 'red-testing',
          red_test: [{ path: 'test/foo.test.ts' }],
          acceptance_criteria: [
            { id: 'AC-001', text: 'First criterion', validation: 'npm test' },
            { id: 'AC-002', text: 'Second criterion' },
          ],
        },
      ],
    });
    assert.deepEqual(findings, []);
  });

  it('matches a slice by id even when it is not the first parsed slice', () => {
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET, {
      slices: [
        null,
        {
          slice_id: 'P1-S1',
          goal: 'red-testing',
          red_test: [{ path: 'test/foo.test.ts' }],
          acceptance_criteria: [
            { id: 'AC-001', text: 'First criterion', validation: 'npm test' },
            { id: 'AC-002', text: 'Second criterion' },
          ],
        },
      ],
    });
    assert.deepEqual(findings, []);
  });

  it('returns no findings for an empty or non-string block', () => {
    assert.deepEqual(lintPlanYamlBlock(''), []);
    assert.deepEqual(lintPlanYamlBlock('   '), []);
    assert.deepEqual(lintPlanYamlBlock(null), []);
    assert.deepEqual(lintPlanYamlBlock(undefined), []);
  });

  it('reports file-absolute line numbers', () => {
    const findings = lintPlanYamlBlock(OVER_INDENTED_PACKET, null, {
      file: CANARY_FILE,
      startLine: 100,
    });
    const overIndented = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.overIndentedListItem,
    );
    for (const finding of overIndented) {
      assert.equal(finding.file, CANARY_FILE);
      assert.equal(finding.line, 112);
      assert.equal(finding.blockLine, 13);
    }
  });

  it('falls back to the first line when the start line is invalid', () => {
    const findings = lintPlanYamlBlock(OVER_INDENTED_PACKET, null, {
      file: CANARY_FILE,
      startLine: 0,
    });
    const parseFinding = findings.find((finding) =>
      finding.message.startsWith('Plan YAML parse error:'),
    );
    assert.equal(parseFinding.line, 13);
  });

  it('omits the file when none is given', () => {
    const findings = lintPlanYamlBlock(OVER_INDENTED_PACKET);
    const parseFinding = findings.find((finding) =>
      finding.message.startsWith('Plan YAML parse error:'),
    );
    assert.equal(parseFinding.file, null);
  });
});

describe('plan-yaml-lint: raw view corner cases', () => {
  it('describes a malformed slice block without losing a criterion', () => {
    const raw = [
      'slices:',
      '  - plain scalar inline',
      '    - stray list item',
      '    just some text',
      '    note: a note',
      '    red_test:',
      '        - path: test/foo.test.ts',
      '      - stray deeper',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        note: not text or validation',
      '      - text: no id here',
    ].join('\n');
    const slices = extractRawSliceBlocks(raw);
    assert.equal(slices.length, 1);
    assert.equal(slices[0].sliceId, null);
    assert.equal(slices[0].acceptanceCriteriaItems, 2);
    assert.equal(slices[0].acceptanceCriteriaBaseIndent, 6);
    assert.deepEqual(
      slices[0].criteria.map((criterion) => criterion.id),
      ['AC-001'],
    );
  });

  it('names a slice without a slice_id by its position', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - goal: red-testing',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
      '        - id: AC-002',
      '          text: Second criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, null, {
      file: CANARY_FILE,
      startLine: 1,
    });
    const rawFinding = findings.find((finding) =>
      finding.message.includes('is indented 8 spaces'),
    );
    assert.ok(rawFinding);
    assert.ok(rawFinding.message.includes("slice '0'"));
  });

  it('reports a count mismatch for a slice without a slice_id', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - goal: implementing',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, {
      slices: [{ goal: 'implementing', acceptance_criteria: [] }],
    });
    const mismatches = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.criterionCountMismatch,
    );
    assert.equal(mismatches.length, 1);
    assert.ok(mismatches[0].message.includes("Slice '0'"));
  });

  it('reports an uncarried slice key for a slice without a slice_id', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - goal: red-testing',
      '    red_test:',
      '      - path: test/foo.test.ts',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, {
      slices: [
        {
          goal: 'red-testing',
          acceptance_criteria: [{ id: 'AC-001', text: 'First criterion' }],
        },
      ],
    });
    const missing = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.sliceKeyMissing,
    );
    assert.equal(missing.length, 1);
    assert.ok(missing[0].message.includes("Slice '0'"));
    assert.ok(missing[0].message.includes("declares key 'red_test'"));
  });

  it('reports a missing red_test for a slice without a slice_id', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - goal: red-testing',
      '    acceptance_criteria:',
      '      - id: AC-001',
      '        text: First criterion',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, {
      slices: [
        {
          goal: 'red-testing',
          acceptance_criteria: [{ id: 'AC-001', text: 'First criterion' }],
        },
      ],
    });
    const missing = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.sliceKeyMissing,
    );
    assert.equal(missing.length, 1);
    assert.ok(missing[0].message.includes("Slice '0'"));
    assert.ok(missing[0].message.includes("declares no 'red_test' key"));
  });

  it('accepts a parsed slice that carries no acceptance criteria', () => {
    const raw = [
      'phase: A',
      'slices:',
      '  - slice_id: P1-S1',
      '    goal: implementing',
    ].join('\n');
    const findings = lintPlanYamlBlock(raw, {
      slices: [{ slice_id: 'P1-S1', goal: 'implementing' }],
    });
    assert.deepEqual(findings, []);
  });

  it('matches a criterion while a parsed criterion carries no id', () => {
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET, {
      slices: [
        {
          slice_id: 'P1-S1',
          goal: 'red-testing',
          red_test: [{ path: 'test/foo.test.ts' }],
          acceptance_criteria: [
            { text: 'a criterion without an id' },
            { id: 'AC-001', text: 'First criterion', validation: 'npm test' },
          ],
        },
      ],
    });
    const missing = findingsWithCode(
      findings,
      PLAN_YAML_LINT_CODES.criterionIdMissing,
    );
    assert.equal(missing.length, 1);
    assert.ok(missing[0].message.includes("'AC-002'"));
  });
});

describe('plan-yaml-lint: error handling', () => {
  it('rethrows an error that is not a plan YAML parse error', () => {
    parseImpl = () => {
      throw new Error('boom');
    };
    assert.throws(() => lintPlanYamlBlock(WELL_FORMED_PACKET), /boom/);
  });

  it('rethrows a falsy thrown value', () => {
    parseImpl = () => {
      throw null;
    };
    let caught;
    let threw = false;
    try {
      lintPlanYamlBlock(WELL_FORMED_PACKET);
    } catch (error) {
      caught = error;
      threw = true;
    }
    assert.equal(threw, true);
    assert.equal(caught, null);
  });

  it('accepts a duck-typed parse error (the module may be mocked)', () => {
    parseImpl = () => {
      const error = new Error('mocked parse failure');
      error.name = 'PlanYamlParseError';
      error.code = 'PLAN_YAML_MOCK_ERROR';
      error.line = 7;
      throw error;
    };
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET);
    const parseFinding = findings.find((finding) =>
      finding.message.startsWith('Plan YAML parse error:'),
    );
    assert.equal(parseFinding.code, 'PLAN_YAML_MOCK_ERROR');
    assert.equal(parseFinding.line, 7);
  });

  it('does not call the parser when a parsed packet is supplied', () => {
    parseImpl = () => {
      throw new Error('the parser must not be called');
    };
    const findings = lintPlanYamlBlock(WELL_FORMED_PACKET, {
      slices: [
        {
          slice_id: 'P1-S1',
          goal: 'red-testing',
          red_test: [{ path: 'test/foo.test.ts' }],
          acceptance_criteria: [
            { id: 'AC-001', text: 'First criterion', validation: 'npm test' },
            { id: 'AC-002', text: 'Second criterion' },
          ],
        },
      ],
    });
    assert.deepEqual(findings, []);
  });
});

describe('plan-yaml-lint: per-packet coverage', () => {
  const DOCUMENT = [
    '# Plan',
    '',
    '```yaml',
    'phase: A',
    'step: 1',
    "status: '[WIP]'",
    'slices:',
    '  - slice_id: P1-S1',
    '    goal: red-testing',
    '    red_test:',
    '      - path: test/foo.test.ts',
    '    acceptance_criteria:',
    '      - id: AC-001',
    '        text: First criterion',
    '        validation: npm test',
    '```',
    '',
    '```yaml',
    'phase: B',
    'step: 2',
    "status: '[WIP]'",
    'slices:',
    '  - slice_id: P2-S1',
    '    goal: red-testing',
    '    red_test:',
    '      - path: test/bar.test.ts',
    '    acceptance_criteria:',
    '      - id: AC-010',
    '        text: Second packet criterion',
    '        - id: AC-011',
    '          text: Over-indented criterion',
    '```',
    '',
    '```yaml',
    'phase: C',
    'step: 3',
    "status: '[DONE code-only 2026-10-29 - archived]'",
    'slices:',
    '  - slice_id: P3-S1',
    '    acceptance_criteria:',
    '      - id: AC-020',
    '        text: Archived criterion',
    '```',
  ].join('\n');

  it('lints EVERY open packet, not only the first one', () => {
    const issues = lintDocument(DOCUMENT, CANARY_FILE);
    const overIndented = issues.filter(
      (issue) =>
        issue.finding.code === PLAN_YAML_LINT_CODES.overIndentedListItem,
    );
    assert.ok(overIndented.length >= 1);
    // The defect sits in the SECOND open packet; the first packet is clean.
    for (const issue of overIndented) {
      assert.equal(issue.startLine, 19);
      assert.equal(issue.finding.line, 30);
      assert.equal(issue.finding.file, CANARY_FILE);
    }
  });

  it('skips a closed packet', () => {
    const blocks = collectPlanYamlBlocks(DOCUMENT);
    assert.equal(blocks.length, 3);
    assert.equal(isActivePlanPacket(blocks[0].yaml), true);
    assert.equal(isActivePlanPacket(blocks[1].yaml), true);
    assert.equal(isActivePlanPacket(blocks[2].yaml), false);
  });

  it('reports the first packet as clean and the second one as broken', () => {
    const blocks = collectPlanYamlBlocks(DOCUMENT);
    assert.deepEqual(lintPlanYamlBlock(blocks[0].yaml), []);
    const second = lintPlanYamlBlock(blocks[1].yaml, null, {
      file: CANARY_FILE,
      startLine: blocks[1].contentStartLine,
    });
    assert.ok(second.length >= 1);
    assert.ok(
      second.some(
        (finding) =>
          finding.code === PLAN_YAML_LINT_CODES.overIndentedListItem &&
          finding.line === 30,
      ),
    );
  });
});
