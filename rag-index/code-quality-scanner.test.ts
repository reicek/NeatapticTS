import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

interface CodeQualityIssue {
  complexity?: number;
  file: string;
  issue: string;
  symbol: string;
  tags?: string[];
  words?: number;
}

interface CodeQualityReport {
  evidence: CodeQualityIssue[];
  fixHint: string | null;
  owner: string;
  pass: boolean;
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  stderr: string;
  stdout: string;
  status: number | null;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('code-quality scanner expanded dimensions', () => {
  let tempFixtureDirectory: string;

  beforeEach(() => {
    tempFixtureDirectory = mkdtempSync(
      path.join(tmpdir(), 'code-quality-scanner-'),
    );
  });

  afterEach(() => {
    rmSync(tempFixtureDirectory, { recursive: true, force: true });
  });

  it('flags exported functions missing required JSDoc @param and @returns tags', () => {
    const fixturePath = writeTempFixture(
      tempFixtureDirectory,
      'jsdoc-tags.ts',
      [
        '/**',
        ' * Demonstrates incomplete JSDoc tag coverage for a function.',
        ' * @param a - the first number',
        ' */',
        'export function incompleteDocs(a: number, b: string): boolean {',
        '  return b.length > a;',
        '}',
      ].join('\n'),
    );

    const result = runScannerModuleEvaluation(`
      import { scanCodeQuality } from './rag-index/code-quality-scanner.mjs';
      const report = await scanCodeQuality({
        sourcePaths: [${JSON.stringify(fixturePath)}],
        complexityThreshold: 10,
        minJsdocWords: 3,
      });
      console.log(JSON.stringify(report));
    `);

    expect(result.status).toBe(0);
    expect(result.report?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          file: expect.any(String),
          issue: 'incomplete JSDoc tags',
          symbol: 'incompleteDocs',
        }),
      ]),
    );
  });

  it('counts switch statements, default clauses and finally clauses in cyclomatic complexity', () => {
    const fixturePath = writeTempFixture(
      tempFixtureDirectory,
      'switch-complexity.ts',
      [
        '/**',
        ' * Uses a switch and a finally clause to exercise ES2023 complexity counting.',
        ' */',
        'export function classify(value: number): number {',
        '  try {',
        '    switch (value) {',
        '      case 1: return 1;',
        '      case 2: return 2;',
        '      default: return 0;',
        '    }',
        '  } finally {',
        '    console.log("done");',
        '  }',
        '}',
      ].join('\n'),
    );

    const result = runScannerModuleEvaluation(`
      import { scanCodeQuality } from './rag-index/code-quality-scanner.mjs';
      const report = await scanCodeQuality({
        sourcePaths: [${JSON.stringify(fixturePath)}],
        complexityThreshold: 1,
        minJsdocWords: 3,
      });
      console.log(JSON.stringify(report));
    `);

    expect(result.status).toBe(0);
    expect(result.report?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          issue: 'high complexity',
          symbol: 'classify',
          complexity: 6,
        }),
      ]),
    );
  });

  it('counts optional chaining and ES2023 logical assignments in cyclomatic complexity', () => {
    const fixturePath = writeTempFixture(
      tempFixtureDirectory,
      'es2023-complexity.ts',
      [
        '/**',
        ' * Uses optional chaining and logical assignment operators.',
        ' */',
        'export function resolve(config: { value?: number }, fallback: number): number {',
        '  config.value ??= fallback;',
        '  return config.value?.toFixed(2)?.length ?? 0;',
        '}',
      ].join('\n'),
    );

    const result = runScannerModuleEvaluation(`
      import { scanCodeQuality } from './rag-index/code-quality-scanner.mjs';
      const report = await scanCodeQuality({
        sourcePaths: [${JSON.stringify(fixturePath)}],
        complexityThreshold: 1,
        minJsdocWords: 3,
      });
      console.log(JSON.stringify(report));
    `);

    expect(result.status).toBe(0);
    expect(result.report?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          issue: 'high complexity',
          symbol: 'resolve',
          complexity: 4,
        }),
      ]),
    );
  });

  it('counts optional chaining inside multi-operator logical chains', () => {
    const fixturePath = writeTempFixture(
      tempFixtureDirectory,
      'logical-chain.ts',
      [
        '/**',
        ' * Combines optional chaining with short-circuit operators.',
        ' */',
        'export function pick(a: number | null, b: number | null, c: number): number {',
        '  return a ?? b?.toString() || c && 0;',
        '}',
      ].join('\n'),
    );

    const result = runScannerModuleEvaluation(`
      import { scanCodeQuality } from './rag-index/code-quality-scanner.mjs';
      const report = await scanCodeQuality({
        sourcePaths: [${JSON.stringify(fixturePath)}],
        complexityThreshold: 1,
        minJsdocWords: 3,
      });
      console.log(JSON.stringify(report));
    `);

    expect(result.status).toBe(0);
    expect(result.report?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          issue: 'high complexity',
          symbol: 'pick',
          complexity: 5,
        }),
      ]),
    );
  });
});

function writeTempFixture(
  directory: string,
  fileName: string,
  content: string,
): string {
  const filePath = path.join(directory, fileName);
  writeFileSync(filePath, content, 'utf8');
  return filePath;
}

function runScannerModuleEvaluation(
  source: string,
): SpawnedJsonResult<CodeQualityReport> {
  const spawned = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );

  return {
    report: tryParseJson<CodeQualityReport>(spawned.stdout ?? ''),
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout: spawned.stdout ?? '',
  };
}

function tryParseJson<ReportType>(stdout: string): ReportType | null {
  if (!stdout.trim()) return null;

  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}
