/**
 * @module customization-utils.direct.test
 * @description Native-ESM contract tests for shared agent-customization utilities.
 *
 * These tests exercise the defensive branches and edge cases in
 * `customization-utils.mjs` using pure dependency injection and real filesystem
 * fixtures where practical. No module mocking is used.
 */

import { jest } from '@jest/globals';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  repoRoot,
  parseArgs,
  printUsage,
  writeReport,
  listMarkdownFiles,
  readWorkspaceFile,
  fileExists,
  parseFrontmatter,
  parseFrontmatterValue,
  parseInlineArray,
  normalizePath,
  issue,
  summarizeIssues,
  parsePlanYamlBlock,
  normalizeTestContracts,
  extractMarkdownLinks,
  extractStatus,
  extractDownstreamTrackers,
} from './customization-utils.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../../');

/**
 * Create a temporary repo-relative directory and return its path.
 *
 * @param {string} name - Unique directory name under tmp/.
 * @returns {Promise<string>} Repo-relative directory path.
 */
async function makeTempDirectory(name) {
  const relativePath = path.join('tmp', name).replace(/\\/g, '/');
  await mkdir(path.join(REPO_ROOT, relativePath), { recursive: true });
  return relativePath;
}

describe('parseArgs', () => {
  it('returns defaults for an empty argv', () => {
    const options = parseArgs([]);

    expect(options.json).toBe(false);
    expect(options.strict).toBe(false);
    expect(options.help).toBe(false);
    expect(options.all).toBe(false);
    expect(options.dryRun).toBe(false);
    expect(options.contract).toBe('tier0');
    expect(options.plan).toBe(
      'plans/completed/Agentic_Workflow_Architecture.plans.md',
    );
  });

  it('parses boolean flags', () => {
    const options = parseArgs([
      '--json',
      '--strict',
      '--all',
      '--dry-run',
      '--help',
    ]);

    expect(options.json).toBe(true);
    expect(options.strict).toBe(true);
    expect(options.all).toBe(true);
    expect(options.dryRun).toBe(true);
    expect(options.help).toBe(true);
  });

  it('parses equals-style values', () => {
    const options = parseArgs([
      '--contract=tier2',
      '--input=foo.md',
      '--plan=plans/demo.plans.md',
    ]);

    expect(options.contract).toBe('tier2');
    expect(options.input).toBe('foo.md');
    expect(options.plan).toBe('plans/demo.plans.md');
  });

  it('parses space-separated values', () => {
    const options = parseArgs([
      '--contract',
      'tier1',
      '--input',
      'bar.md',
      '--plan',
      'plans/other.plans.md',
    ]);

    expect(options.contract).toBe('tier1');
    expect(options.input).toBe('bar.md');
    expect(options.plan).toBe('plans/other.plans.md');
  });

  it('ignores unknown positional arguments', () => {
    const options = parseArgs(['--json', 'unexpected']);

    expect(options.json).toBe(true);
    expect(options.unexpected).toBeUndefined();
  });
});

describe('report helpers', () => {
  /** @type {jest.SpyInstance} */
  let logSpy;
  /** @type {jest.SpyInstance} */
  let stdoutSpy;

  beforeEach(() => {
    logSpy = jest.spyOn(global.console, 'log').mockImplementation(() => {});
    stdoutSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);
  });

  afterEach(() => {
    logSpy.mockRestore();
    stdoutSpy.mockRestore();
  });

  it('printUsage writes the title, usage, and options', () => {
    printUsage({
      title: 'test-script',
      usage: 'node test-script.mjs',
      options: [['--custom', 'Custom flag.']],
    });

    expect(logSpy).toHaveBeenCalledWith(
      'test-script\n\nUsage:\n  node test-script.mjs\n\nOptions:',
    );
    expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('--json'));
    expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('--custom'));
  });

  it('writeReport prints summaryText when json is false', () => {
    writeReport(
      { ok: true, name: 'demo', summaryText: 'PASS demo' },
      { json: false },
    );

    expect(stdoutSpy).toHaveBeenCalledWith('PASS demo\n', 'utf8');
  });

  it('writeReport serializes the full report when json is true', () => {
    const report = { ok: false, name: 'demo', issues: [] };
    writeReport(report, { json: true });

    expect(stdoutSpy).toHaveBeenCalledWith(
      JSON.stringify(report, null, 2) + '\n',
      'utf8',
    );
  });

  it('writeReport preserves non-ASCII characters in JSON output', () => {
    const report = {
      ok: true,
      name: 'charset-test',
      description: 'Verify \u22643 files \u2014 no corruption',
    };
    writeReport(report, { json: true });

    const expectedJson = JSON.stringify(report, null, 2) + '\n';
    expect(stdoutSpy).toHaveBeenCalledWith(expectedJson, 'utf8');
    expect(expectedJson).toContain('\u2264');
    expect(expectedJson).toContain('\u2014');
    expect(expectedJson).not.toContain('?');
  });
});

describe('filesystem helpers', () => {
  it('readWorkspaceFile reads a known repo-relative file', async () => {
    const contents = await readWorkspaceFile('plans/README.md');

    expect(contents).toContain('#');
  });

  it('fileExists returns true for a regular file', async () => {
    await expect(fileExists('plans/Roadmap.md')).resolves.toBe(true);
  });

  it('fileExists returns false for a missing path', async () => {
    await expect(fileExists('plans/does-not-exist.md')).resolves.toBe(false);
  });

  it('listMarkdownFiles returns an empty array when the predicate rejects everything', async () => {
    const tempDir = await makeTempDirectory(
      `utils-md-reject-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    );
    await writeFile(path.join(REPO_ROOT, tempDir, 'a.md'), 'A', 'utf8');

    try {
      const files = await listMarkdownFiles(tempDir, () => false);

      expect(files).toEqual([]);
    } finally {
      await rm(path.join(REPO_ROOT, tempDir), { recursive: true, force: true });
    }
  });

  it('listMarkdownFiles discovers and sorts matching files', async () => {
    const tempDir = await makeTempDirectory(
      `utils-md-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    );
    await writeFile(path.join(REPO_ROOT, tempDir, 'a.md'), 'A', 'utf8');
    await writeFile(path.join(REPO_ROOT, tempDir, 'b.md'), 'B', 'utf8');

    try {
      const files = await listMarkdownFiles(tempDir, () => true);

      expect(files).toContain(`${tempDir}/a.md`);
      expect(files).toContain(`${tempDir}/b.md`);
      expect(files[0]).toBe(`${tempDir}/a.md`);
    } finally {
      await rm(path.join(REPO_ROOT, tempDir), { recursive: true, force: true });
    }
  });
});

describe('parseFrontmatter', () => {
  it('reports an error when the opening fence is missing', () => {
    const result = parseFrontmatter('No frontmatter here.', 'test.md');

    expect(result.data).toEqual({});
    expect(result.body).toBe('No frontmatter here.');
    expect(result.issues).toContainEqual(
      issue('error', 'test.md', 'Missing opening YAML frontmatter fence.'),
    );
  });

  it('reports an error when the closing fence is missing', () => {
    const result = parseFrontmatter(
      '---\nfoo: bar\nNo closing fence.',
      'test.md',
    );

    expect(result.data).toEqual({});
    expect(result.issues).toContainEqual(
      issue('error', 'test.md', 'Missing closing YAML frontmatter fence.'),
    );
  });

  it('parses scalar, boolean, and quoted values', () => {
    const result = parseFrontmatter(
      `---\nname: demo\nenabled: true\ndisabled: false\ntitle: "Quoted title"\n---\nbody`,
      'test.md',
    );

    expect(result.data).toEqual({
      name: 'demo',
      enabled: true,
      disabled: false,
      title: 'Quoted title',
    });
    expect(result.body).toBe('body');
  });

  it('parses inline arrays', () => {
    const result = parseFrontmatter(
      '---\ntools: [read, edit, "quoted"]\n---\n',
      'test.md',
    );

    expect(result.data.tools).toEqual(['read', 'edit', 'quoted']);
  });

  it('parses a multi-line bracketed array', () => {
    const result = parseFrontmatter(
      '---\ntools:\n  [read, edit,\n   search]\n---\n',
      'test.md',
    );

    expect(result.data.tools).toEqual(['read', 'edit', 'search']);
  });

  it('warns on unparsable non-indented lines', () => {
    const result = parseFrontmatter(
      '---\nname: demo\nthis is garbage\n---\n',
      'test.md',
    );

    expect(result.data.name).toBe('demo');
    expect(result.issues).toContainEqual(
      issue(
        'warning',
        'test.md',
        'Could not parse frontmatter line: this is garbage',
      ),
    );
  });

  it('ignores indented structural lines and comments', () => {
    const result = parseFrontmatter(
      '---\nname: demo\n  # indented comment\n# top comment\n---\n',
      'test.md',
    );

    expect(result.data).toEqual({ name: 'demo' });
    expect(result.issues).toHaveLength(0);
  });

  it('strips a BOM if present', () => {
    const result = parseFrontmatter('\uFEFF---\nname: demo\n---\n', 'test.md');

    expect(result.data.name).toBe('demo');
  });

  it('preserves non-ASCII characters (\u2264, \u2014) in parsed values', () => {
    const result = parseFrontmatter(
      '---\nname: test\ndescription: Foo \u22643 files \u2014 bar\n---\n',
      'test.md',
    );

    expect(result.data.description).toBe('Foo \u22643 files \u2014 bar');
    expect(result.data.description).toContain('\u2264');
    expect(result.data.description).toContain('\u2014');
    expect(result.data.description).not.toContain('?');
  });

  it('parses a multi-line bracketed array starting on the same line', () => {
    const result = parseFrontmatter(
      '---\ntools: [read, edit,\n   search]\n---\n',
      'test.md',
    );

    expect(result.data.tools).toEqual(['read', 'edit', 'search']);
  });

  it('interprets a bare key with no value as true', () => {
    const result = parseFrontmatter('---\nflag:\n---\n', 'test.md');

    expect(result.data.flag).toBe(true);
  });

  it('strips trailing inline comments', () => {
    const result = parseFrontmatter(
      '---\nname: demo # inline comment\n---\n',
      'test.md',
    );

    expect(result.data.name).toBe('demo');
  });
});

describe('parseFrontmatterValue', () => {
  it('converts empty string to true', () => {
    expect(parseFrontmatterValue('')).toBe(true);
  });

  it('converts booleans', () => {
    expect(parseFrontmatterValue('true')).toBe(true);
    expect(parseFrontmatterValue('false')).toBe(false);
  });

  it('strips matching quotes', () => {
    expect(parseFrontmatterValue('"quoted"')).toBe('quoted');
    expect(parseFrontmatterValue("'single'")).toBe('single');
  });

  it('parses inline arrays', () => {
    expect(parseFrontmatterValue('[a, b, c]')).toEqual(['a', 'b', 'c']);
  });

  it('returns trimmed scalar with comment removed', () => {
    expect(parseFrontmatterValue(' demo # comment ')).toBe('demo');
  });
});

describe('parseInlineArray', () => {
  it('returns an empty array for an empty bracket pair', () => {
    expect(parseInlineArray('[]')).toEqual([]);
  });

  it('trims, unquotes, and filters empty items', () => {
    expect(parseInlineArray('[a, "b", \'c\', , d]')).toEqual([
      'a',
      'b',
      'c',
      'd',
    ]);
  });
});

describe('normalizePath', () => {
  it('replaces platform separators with forward slashes', () => {
    expect(normalizePath(path.join('a', 'b', 'c'))).toBe('a/b/c');
  });
});

describe('summarizeIssues', () => {
  it('passes when there are no errors', () => {
    const result = summarizeIssues('surface', [
      issue('warning', 'a.md', 'advisory'),
    ]);

    expect(result.ok).toBe(true);
    expect(result.counts).toEqual({ errors: 0, warnings: 1 });
    expect(result.summaryText).toBe('PASS surface: 0 errors, 1 warnings');
  });

  it('fails when there is an error', () => {
    const result = summarizeIssues('surface', [
      issue('error', 'a.md', 'blocking'),
      issue('warning', 'b.md', 'advisory'),
    ]);

    expect(result.ok).toBe(false);
    expect(result.counts).toEqual({ errors: 1, warnings: 1 });
    expect(result.summaryText).toBe('FAIL surface: 1 errors, 1 warnings');
  });
});

describe('parsePlanYamlBlock', () => {
  it('parses scalar keys', () => {
    const result = parsePlanYamlBlock('phase: 1\nstatus: WIP');

    expect(result.phase).toBe(1);
    expect(result.status).toBe('WIP');
  });

  it('parses a scalar list', () => {
    const result = parsePlanYamlBlock('files_to_change:\n  - a.ts\n  - b.ts');

    expect(result.files_to_change).toEqual(['a.ts', 'b.ts']);
  });

  it('parses an object list', () => {
    const result = parsePlanYamlBlock(`slices:
  - slice_id: A
    title: Slice A
  - slice_id: B
    title: Slice B`);

    expect(result.slices).toEqual([
      { slice_id: 'A', title: 'Slice A' },
      { slice_id: 'B', title: 'Slice B' },
    ]);
  });

  it('parses nested objects', () => {
    const result = parsePlanYamlBlock(`pre_execute_hook:
  tool: cortex/cortex
  args:
    slice_id: A`);

    expect(result.pre_execute_hook).toEqual({
      tool: 'cortex/cortex',
      args: { slice_id: 'A' },
    });
  });

  it('treats a bare key with list children as an empty list', () => {
    const result = parsePlanYamlBlock('empty_list:\n# no children');

    expect(result.empty_list).toEqual([]);
  });

  it('treats a bare key with object children as an empty object', () => {
    const result = parsePlanYamlBlock('empty_object:\n  child:\n');

    expect(result.empty_object).toEqual({ child: {} });
  });

  it('parses a nested list inside an object-list item', () => {
    const result = parsePlanYamlBlock(`slices:
  - config:
      - a
      - b`);

    expect(result.slices).toEqual([{ config: ['a', 'b'] }]);
  });

  it('parses scalar block chomping indicators as empty lists', () => {
    expect(parsePlanYamlBlock('description: >').description).toEqual([]);
    expect(parsePlanYamlBlock('description: |').description).toEqual([]);
  });

  it('parses numeric and quoted scalar values', () => {
    const result = parsePlanYamlBlock(`count: 42
ratio: 3.14
name: "quoted"`);

    expect(result.count).toBe(42);
    expect(result.ratio).toBe(3.14);
    expect(result.name).toBe('quoted');
  });

  it('ignores blank lines and comments', () => {
    const result = parsePlanYamlBlock(`# comment

phase: 2
`);

    expect(result.phase).toBe(2);
  });
});

describe('normalizeTestContracts', () => {
  it('returns an empty array for non-array input', () => {
    expect(normalizeTestContracts(null)).toEqual([]);
    expect(normalizeTestContracts('string')).toEqual([]);
  });

  it('extracts an id from the text when the id field is missing', () => {
    const result = normalizeTestContracts([
      { text: 'Verify AC-123 passes coverage.' },
    ]);

    expect(result).toEqual([
      {
        id: 'AC-123',
        text: 'Verify AC-123 passes coverage.',
        validation: undefined,
      },
    ]);
  });

  it('keeps an explicit id when present', () => {
    const result = normalizeTestContracts([
      { id: 'AC-001', text: 'Do the thing.' },
    ]);

    expect(result).toEqual([
      { id: 'AC-001', text: 'Do the thing.', validation: undefined },
    ]);
  });

  it('preserves string validation values', () => {
    const result = normalizeTestContracts([
      { id: 'AC-002', text: 'Run tests.', validation: 'npx jest' },
    ]);

    expect(result[0].validation).toBe('npx jest');
  });

  it('drops non-string validation values', () => {
    const result = normalizeTestContracts([
      { id: 'AC-003', text: 'Check.', validation: { command: 'jest' } },
    ]);

    expect(result[0].validation).toBeUndefined();
  });
});

describe('extractMarkdownLinks', () => {
  it('returns only local relative links', () => {
    const body =
      '[local](./relative/path.md) [absolute](https://example.com) [anchor](#x)';

    expect(extractMarkdownLinks(body)).toEqual(['./relative/path.md']);
  });
});

describe('extractStatus', () => {
  it('extracts DONE, WIP, and PLANNED markers', () => {
    expect(extractStatus('**Status:** [DONE]')).toBe('DONE');
    expect(extractStatus('**Status:** [WIP]')).toBe('WIP');
    expect(extractStatus('**Status:** [PLANNED]')).toBe('PLANNED');
    expect(extractStatus('no status')).toBeNull();
  });
});

describe('extractDownstreamTrackers', () => {
  it('returns an array of valid downstream tracker paths', async () => {
    const body = '[Downstream tracker](./Racing_Perception_Redesign.plans.md)';
    const result = await extractDownstreamTrackers(
      body,
      'plans/orchestration-fixes.plans.md',
    );

    expect(Array.isArray(result)).toBe(true);
    expect(result).not.toContain('plans/orchestration-fixes.plans.md');
    expect(result).not.toContain('plans/README.md');
    expect(result).not.toContain('plans/Roadmap.md');
    expect(result).toEqual([...result].sort());
  });

  it('returns an empty array for a non-existent active plan', async () => {
    const result = await extractDownstreamTrackers(
      '',
      'plans/__missing-active.plans.md',
    );

    expect(result).toEqual([]);
  });

  it('excludes body links that do not exist on disk', async () => {
    const body = '[Non-existent tracker](./NonExistent_Tracker.plans.md)';
    const result = await extractDownstreamTrackers(
      body,
      'plans/orchestration-fixes.plans.md',
    );

    expect(result).not.toContain('plans/NonExistent_Tracker.plans.md');
  });
});
