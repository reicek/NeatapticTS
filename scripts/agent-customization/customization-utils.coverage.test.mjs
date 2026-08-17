/**
 * @module customization-utils.coverage.test
 * @description Targeted coverage tests for remaining uncovered lines in customization-utils.mjs.
 *
 * Covers: listMarkdownFiles readdir catch, parseFrontmatter multi-line array
 * continuation, parsePlanYamlBlock list/object edge cases, extractPlanLinks,
 * normalizePlanLink, extractPlanLinksFromRoadmap, and extractDownstreamTrackers
 * candidate filtering.
 */
import { jest } from '@jest/globals';
import { readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  listMarkdownFiles,
  parseFrontmatter,
  parsePlanYamlBlock,
  extractDownstreamTrackers,
  normalizeTestContracts,
} from './customization-utils.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../../');

describe('listMarkdownFiles readdir failure', () => {
  it('returns an empty array when the root directory does not exist', async () => {
    const files = await listMarkdownFiles('__nonexistent_dir_coverage__', () => true);
    expect(files).toEqual([]);
  });
});

describe('parseFrontmatter multi-line array continuation', () => {
  it('handles a bracketed array spanning three lines', () => {
    const result = parseFrontmatter(
      '---\ntools: [read,\n   edit,\n   search]\n---\n',
      'test.md',
    );
    expect(result.data.tools).toEqual(['read', 'edit', 'search']);
  });

  it('skips blank lines before a bracketed array on a following line', () => {
    const result = parseFrontmatter(
      '---\ntools:\n\n[read, edit]\n---\n',
      'test.md',
    );
    expect(result.data.tools).toEqual(['read', 'edit']);
  });
});

describe('parsePlanYamlBlock list edge cases', () => {
  it('skips blank lines and comments inside a list block', () => {
    const result = parsePlanYamlBlock(`items:
  - a
  # comment inside list
  - b`);
    expect(result.items).toEqual(['a', 'b']);
  });

  it('parses an object list item whose key starts on the next line', () => {
    const result = parsePlanYamlBlock(`items:
  - some-key:
      field: value`);
    expect(result.items).toEqual([{ field: 'value' }]);
  });

  it('treats an empty inline first key with no children as an empty array', () => {
    const result = parsePlanYamlBlock(`slices:
  - empty_key:`);
    expect(result.slices).toEqual([{ empty_key: [] }]);
  });

  it('skips non-key indented lines inside an object', () => {
    const result = parsePlanYamlBlock(`slices:
  - slice_id: A
    some text without colon
    title: B`);
    expect(result.slices).toEqual([{ slice_id: 'A', title: 'B' }]);
  });
});

describe('extractDownstreamTrackers coverage', () => {
  it('finds a body link with plans/ prefix that exists on disk', async () => {
    const tempFile = path.join(REPO_ROOT, 'plans/__coverage_downstream.plans.md');
    await writeFile(tempFile, '# temp', 'utf8');
    try {
      const body = 'See plans/__coverage_downstream.plans.md for details';
      const result = await extractDownstreamTrackers(
        body,
        'plans/__coverage_active.plans.md',
      );
      expect(result).toContain('plans/__coverage_downstream.plans.md');
    } finally {
      await rm(tempFile, { force: true });
    }
  });

  it('covers Roadmap section matching, normalizePlanLink branches, and candidate filtering', async () => {
    const roadmapPath = path.join(REPO_ROOT, 'plans/Roadmap.md');
    const original = await readFile(roadmapPath, 'utf8');
    try {
      await writeFile(
        roadmapPath,
        [
          '## Test Coverage Section',
          '',
          '- [Active](plans/Test_Coverage_Active.plans.md)',
          '- Completed: `completed/Test_Completed.plans.md`',
          '- Bare: `Test_Bare.plans.md`',
          '- Prefixed: `plans/Test_Prefixed.plans.md`',
          '- Roadmap ref: `plans/Roadmap.md`',
          '- Readme ref: `plans/README.md`',
          '',
          '## Other Section',
          '',
          'Nothing relevant here.',
          '',
        ].join('\n'),
        'utf8',
      );

      const result = await extractDownstreamTrackers(
        '',
        'plans/Test_Coverage_Active.plans.md',
      );
      expect(result).toEqual([]);
    } finally {
      await writeFile(roadmapPath, original, 'utf8');
    }
  });

  it('returns empty array when Roadmap.md does not exist', async () => {
    const roadmapPath = path.join(REPO_ROOT, 'plans/Roadmap.md');
    const original = await readFile(roadmapPath, 'utf8');
    try {
      await rm(roadmapPath);
      const result = await extractDownstreamTrackers(
        '',
        'plans/any-active-plan.plans.md',
      );
      expect(result).toEqual([]);
    } finally {
      await writeFile(roadmapPath, original, 'utf8');
    }
  });
});

describe('normalizeTestContracts branch coverage', () => {
  it('covers criteria with text, id, and validation defined', () => {
    const result = normalizeTestContracts([
      { text: 'Do something', id: 'AC-1', validation: 'npm test' },
    ]);
    expect(result).toEqual([
      { id: 'AC-1', text: 'Do something', validation: 'npm test' },
    ]);
  });

  it('extracts AC-n from text when id is missing', () => {
    const result = normalizeTestContracts([
      { text: 'AC-5: Do something' },
    ]);
    expect(result).toEqual([
      { id: 'AC-5', text: 'AC-5: Do something', validation: undefined },
    ]);
  });

  it('handles empty criterion with no fields', () => {
    const result = normalizeTestContracts([{}]);
    expect(result).toEqual([{ id: '', text: '', validation: undefined }]);
  });
});

describe('parsePlanYamlBlock scalar and edge-case branches', () => {
  it('skips empty lines in peekNextNonEmptyLine via top-level key', () => {
    const result = parsePlanYamlBlock('key:\n\nnext: value');
    expect(result.key).toEqual([]);
    expect(result.next).toBe('value');
  });

  it('covers normalizeYamlScalar branches for all scalar types', () => {
    const result = parsePlanYamlBlock(`slices:
  - slice_id: A1
    count: 42
    flag: true
    flag2: false
    empty_list: []
    quoted: "hello"
    single: 'world'
    negative: -3
    decimal: 3.14`);
    expect(result.slices).toEqual([
      {
        slice_id: 'A1',
        count: 42,
        flag: true,
        flag2: false,
        empty_list: [],
        quoted: 'hello',
        single: 'world',
        negative: -3,
        decimal: 3.14,
      },
    ]);
  });

  it('covers Number.isFinite false branch with overflow number', () => {
    const bigNum = '9'.repeat(310);
    const result = parsePlanYamlBlock(`slices:
  - slice_id: A1
    big: ${bigNum}`);
    expect(result.slices[0].big).toBe(bigNum);
    expect(typeof result.slices[0].big).toBe('string');
  });

  it('covers object field with empty rest and no children (while loop)', () => {
    const result = parsePlanYamlBlock(`slices:
  - slice_id: A1
    empty_field:`);
    expect(result.slices).toEqual([{ slice_id: 'A1', empty_field: [] }]);
  });

  it('covers nested object with empty field (emptyDefaultToObject true)', () => {
    const result = parsePlanYamlBlock(`slices:
  - slice_id: A1
    config:
      empty_nested:`);
    expect(result.slices).toEqual([
      { slice_id: 'A1', config: { empty_nested: {} } },
    ]);
  });
});

describe('defensive branch coverage via monkey-patching', () => {
  it('covers line 498 (line === undefined) via sparse split array', () => {
    const origSplit = String.prototype.split;
    String.prototype.split = function sep(sep) {
      const arr = origSplit.call(this, sep);
      if (
        arr.length === 4 &&
        arr[0] === 'items:' &&
        arr[1] === '  - obj_key:' &&
        arr[2] === '    - nested' &&
        arr[3] === 'stop: val'
      ) {
        const sparse = new Array(5);
        sparse[0] = arr[0];
        sparse[1] = arr[1];
        sparse[3] = arr[2];
        sparse[4] = arr[3];
        return sparse;
      }
      return arr;
    };
    let result;
    try {
      result = parsePlanYamlBlock(
        'items:\n  - obj_key:\n    - nested\nstop: val',
      );
    } finally {
      String.prototype.split = origSplit;
    }
    expect(result.items).toEqual([{ obj_key: ['nested'] }]);
    expect(result.stop).toBe('val');
  });

  it('covers line 500 (match null) in peekNextNonEmptyLine', () => {
    const origMatch = String.prototype.match;
    let count = 0;
    String.prototype.match = function match(re) {
      if (re && re.source === '^\\s*') {
        count++;
        if (count === 1) return null;
      }
      return origMatch.call(this, re);
    };
    let result;
    try {
      result = parsePlanYamlBlock('key:\nnext: value');
    } finally {
      String.prototype.match = origMatch;
    }
    expect(result.key).toEqual([]);
    expect(result.next).toBe('value');
  });

  it('covers line 529 (match null) in parseListBlock', () => {
    const origMatch = String.prototype.match;
    let count = 0;
    String.prototype.match = function match(re) {
      if (re && re.source === '^\\s*') {
        count++;
        if (count === 2) return null;
      }
      return origMatch.call(this, re);
    };
    let result;
    try {
      result = parsePlanYamlBlock('items:\n  - a');
    } finally {
      String.prototype.match = origMatch;
    }
    expect(result.items).toEqual([]);
  });

  it('covers line 617 (match null) in parseObjectFields while loop', () => {
    const origMatch = String.prototype.match;
    let count = 0;
    String.prototype.match = function match(re) {
      if (re && re.source === '^\\s*') {
        count++;
        if (count === 3) return null;
      }
      return origMatch.call(this, re);
    };
    let result;
    try {
      result = parsePlanYamlBlock('items:\n  - key: value\n    field: val');
    } finally {
      String.prototype.match = origMatch;
    }
    expect(result.items).toEqual([{ key: 'value' }]);
  });

  it('covers line 589 (firstMatch null) in parseObjectFields via exec mock', () => {
    const origExec = RegExp.prototype.exec;
    let count = 0;
    RegExp.prototype.exec = function exec(str) {
      if (this.source === '^(?<key>[A-Za-z0-9_]+):(?<rest>.*)$') {
        count++;
        if (count === 4) return null;
      }
      return origExec.call(this, str);
    };
    let result;
    try {
      result = parsePlanYamlBlock('items:\n  - a\n  - key: value');
    } finally {
      RegExp.prototype.exec = origExec;
    }
    expect(result.items).toEqual(['a', {}, { key: 'value' }]);
  });
});