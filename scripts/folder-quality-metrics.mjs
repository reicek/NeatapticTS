#!/usr/bin/env node
/**
 * @module folder-quality-metrics
 * @description Static folder-quality scanner for fast post-edit checks.
 *
 * Usage:
 *   node scripts/folder-quality-metrics.mjs --folder=<path> [--json]
 *   node scripts/folder-quality-metrics.mjs --help
 *
 * Checks:
 *   - TypeScript diagnostics scoped to files inside the target folder
 *   - ESLint error diagnostics scoped to files inside the target folder
 *   - JSDoc presence for exported function/class/const symbols
 *   - Missing sibling `.test.ts` files for source modules
 *   - Optional line-coverage deficits from `coverage/lcov.info`
 *
 * Exit codes:
 *   0: no blocking smells found
 *   1: blocking smells found, invalid arguments, or unexpected failure
 */

import { constants as fsConstants } from 'node:fs';
import { access, readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

import { ESLint } from 'eslint';
import typescript from 'typescript';

const SCRIPT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIRECTORY, '..');
const COVERAGE_LCOV_PATH = path.join(REPO_ROOT, 'coverage', 'lcov.info');
const execAsync = promisify(exec);
const SOURCE_FILE_SUFFIX = '.ts';
const DECLARATION_FILE_SUFFIX = '.d.ts';
const TEST_FILE_SUFFIX = '.test.ts';
const NON_MODULE_FILE_SUFFIXES = ['.types.ts', '.constants.ts'];
/** Suffix for pure-leaf util files that are tested through their parent module. */
const UTIL_FILE_SUFFIX = '.utils.ts';
const EXPORTED_SYMBOL_PATTERN =
  /^\s*export\s+(?:async\s+)?(?:(function|class|const)\s+([A-Za-z0-9_$]+))/u;
const IGNORED_DIRECTORY_NAMES = new Set([
  '.git',
  'coverage',
  'dist',
  'dist-docs',
  'node_modules',
]);

/**
 * Run the folder-quality scanner and return a structured report.
 *
 * @param {{ folderPath: string }} options - Scanner options.
 * @returns {Promise<{ evidence: string[], folderChecked: string, pass: boolean, smells: Array<{ detail: string, file: string, kind: string }> }>} Structured quality report.
 */
export async function runFolderQualityMetrics({ folderPath }) {
  const resolvedFolder = await resolveFolderPath(folderPath);
  const sourceFilePaths = await collectTypeScriptFiles(
    resolvedFolder.absolutePath,
  );
  const moduleFilePaths = sourceFilePaths.filter(isModuleOwnedSourceFile);
  const collectedSmells = [];
  const evidence = [];

  const typeScriptResult = await collectTypeScriptSmells(
    resolvedFolder,
    sourceFilePaths,
  );
  collectedSmells.push(...typeScriptResult.smells);
  evidence.push(typeScriptResult.evidence);

  const eslintResult = await collectEslintSmells(sourceFilePaths);
  collectedSmells.push(...eslintResult.smells);
  evidence.push(eslintResult.evidence);

  const jsdocResult = await collectJsdocSmells(moduleFilePaths);
  collectedSmells.push(...jsdocResult.smells);
  evidence.push(jsdocResult.evidence);

  const testPresenceResult = await collectMissingTestFileSmells(
    moduleFilePaths,
    resolvedFolder.absolutePath,
  );
  collectedSmells.push(...testPresenceResult.smells);
  evidence.push(testPresenceResult.evidence);

  const coverageResult = await collectCoverageDeficitSmells(
    resolvedFolder.relativePath,
    moduleFilePaths,
  );
  collectedSmells.push(...coverageResult.smells);
  evidence.push(coverageResult.evidence);

  const smells = deduplicateSmells(collectedSmells).toSorted(compareSmells);

  return {
    evidence,
    folderChecked: resolvedFolder.relativePath,
    pass: smells.length === 0,
    smells,
  };
}

function parseArgs(argv) {
  const options = {
    folder: null,
    help: false,
    json: false,
  };

  for (let argumentIndex = 0; argumentIndex < argv.length; argumentIndex++) {
    const rawArgument = argv[argumentIndex];

    if (rawArgument === '--help' || rawArgument === '-h') {
      options.help = true;
      continue;
    }

    if (rawArgument === '--json') {
      options.json = true;
      continue;
    }

    if (rawArgument.startsWith('--folder=')) {
      options.folder = sanitizeCliValue(rawArgument.slice('--folder='.length));
      continue;
    }

    if (rawArgument === '--folder') {
      const nextArgument = argv[argumentIndex + 1];
      if (typeof nextArgument !== 'string' || nextArgument.trim() === '') {
        throw new Error('Missing value after --folder.');
      }

      options.folder = sanitizeCliValue(nextArgument);
      argumentIndex++;
      continue;
    }

    throw new Error(`Unknown argument: ${rawArgument}`);
  }

  return options;
}

/**
 * Removes wrapping single or double quotes that can survive shell/npm forwarding.
 *
 * @param cliValue - Raw CLI token value.
 * @returns Trimmed value with one matching quote pair removed.
 */
function sanitizeCliValue(cliValue) {
  const trimmedValue = cliValue.trim();
  const startsWithSingleQuote = trimmedValue.startsWith("'");
  const endsWithSingleQuote = trimmedValue.endsWith("'");
  const startsWithDoubleQuote = trimmedValue.startsWith('"');
  const endsWithDoubleQuote = trimmedValue.endsWith('"');

  if (
    (startsWithSingleQuote && endsWithSingleQuote) ||
    (startsWithDoubleQuote && endsWithDoubleQuote)
  ) {
    return trimmedValue.slice(1, -1).trim();
  }

  return trimmedValue;
}
function resolveFolderFromNpmEnvironment() {
  const npmConfigFolder = process.env.npm_config_folder;

  if (typeof npmConfigFolder !== 'string' || npmConfigFolder.trim() === '') {
    return null;
  }

  return sanitizeCliValue(npmConfigFolder);
}

function printUsage() {
  console.log(
    [
      'Folder-quality metrics',
      '',
      'Usage:',
      '  node scripts/folder-quality-metrics.mjs --folder=<path> [--json]',
      '  node scripts/folder-quality-metrics.mjs --help',
      '',
      'Flags:',
      '  --folder=<path>  Repo-relative folder to scan.',
      '  --json           Emit the structured report to stdout as JSON.',
      '  --help           Show usage, checks, and exit codes.',
      '',
      'Checks:',
      '  - TypeScript diagnostics scoped to the target folder',
      '  - ESLint error diagnostics scoped to the target folder',
      '  - Exported-symbol JSDoc presence',
      '  - Missing sibling .test.ts files',
      '  - Optional coverage/lcov.info line-coverage deficits',
      '',
      'Exit codes:',
      '  0  No blocking smells found',
      '  1  Blocking smells found, invalid arguments, or unexpected failure',
    ].join('\n'),
  );
}

async function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    const resolvedFolder = options.folder ?? resolveFolderFromNpmEnvironment();

    if (options.help) {
      printUsage();
      return;
    }

    if (!resolvedFolder) {
      throw new Error('--folder=<path> is required.');
    }

    const report = await runFolderQualityMetrics({
      folderPath: resolvedFolder,
    });
    writeReport(report, options.json);
    process.exitCode = report.pass ? 0 : 1;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  }
}

async function resolveFolderPath(folderPath) {
  if (typeof folderPath !== 'string' || !folderPath.trim()) {
    throw new Error('The folder path must be a non-empty string.');
  }

  const trimmedFolderPath = folderPath.trim();
  const absolutePath = path.isAbsolute(trimmedFolderPath)
    ? path.normalize(trimmedFolderPath)
    : path.resolve(REPO_ROOT, trimmedFolderPath);
  const relativePath = normalizePath(path.relative(REPO_ROOT, absolutePath));

  if (
    relativePath === '' ||
    relativePath.startsWith('..') ||
    path.isAbsolute(relativePath)
  ) {
    throw new Error(
      'The target folder must resolve inside the repository root.',
    );
  }

  const folderStats = await stat(absolutePath).catch(() => null);
  if (!folderStats?.isDirectory()) {
    throw new Error(`Target folder does not exist: ${trimmedFolderPath}`);
  }

  return {
    absolutePath,
    relativePath,
  };
}

async function collectTypeScriptFiles(folderPath) {
  const discoveredFiles = [];
  const pendingDirectories = [folderPath];

  while (pendingDirectories.length > 0) {
    const currentDirectory = pendingDirectories.pop();
    const directoryEntries = await readdir(currentDirectory, {
      withFileTypes: true,
    });

    for (const directoryEntry of directoryEntries) {
      const entryPath = path.join(currentDirectory, directoryEntry.name);
      if (directoryEntry.isDirectory()) {
        if (!IGNORED_DIRECTORY_NAMES.has(directoryEntry.name)) {
          pendingDirectories.push(entryPath);
        }

        continue;
      }

      if (
        directoryEntry.isFile() &&
        entryPath.endsWith(SOURCE_FILE_SUFFIX) &&
        !entryPath.endsWith(DECLARATION_FILE_SUFFIX)
      ) {
        discoveredFiles.push(path.normalize(entryPath));
      }
    }
  }

  return discoveredFiles.toSorted((leftPath, rightPath) =>
    leftPath.localeCompare(rightPath),
  );
}

async function collectDeclarationFiles(folderPath) {
  const discoveredFiles = [];
  const pendingDirectories = [folderPath];

  while (pendingDirectories.length > 0) {
    const currentDirectory = pendingDirectories.pop();
    const directoryEntries = await readdir(currentDirectory, {
      withFileTypes: true,
    });

    for (const directoryEntry of directoryEntries) {
      const entryPath = path.join(currentDirectory, directoryEntry.name);
      if (directoryEntry.isDirectory()) {
        if (!IGNORED_DIRECTORY_NAMES.has(directoryEntry.name)) {
          pendingDirectories.push(entryPath);
        }

        continue;
      }

      if (
        directoryEntry.isFile() &&
        entryPath.endsWith(DECLARATION_FILE_SUFFIX)
      ) {
        discoveredFiles.push(path.normalize(entryPath));
      }
    }
  }

  return discoveredFiles.toSorted((leftPath, rightPath) =>
    leftPath.localeCompare(rightPath),
  );
}

function isModuleOwnedSourceFile(filePath) {
  return (
    !filePath.endsWith(TEST_FILE_SUFFIX) &&
    !NON_MODULE_FILE_SUFFIXES.some((suffix) => filePath.endsWith(suffix))
  );
}

async function collectTypeScriptSmells(resolvedFolder, sourceFilePaths) {
  if (sourceFilePaths.length === 0) {
    return {
      evidence:
        'TypeScript: no .ts files found under the target folder; skipped.',
      smells: [],
    };
  }

  const configFileName = resolvedFolder.relativePath.startsWith('src/')
    ? 'tsconfig.json'
    : 'tsconfig.test.json';
  const configFilePath = path.join(REPO_ROOT, configFileName);
  const configFile = typescript.readConfigFile(
    configFilePath,
    typescript.sys.readFile,
  );

  if (configFile.error) {
    return {
      evidence: `TypeScript (${configFileName}): failed to read config.`,
      smells: [
        {
          detail: flattenDiagnosticMessage(configFile.error.messageText),
          file: normalizePath(path.relative(REPO_ROOT, configFilePath)),
          kind: 'typescript-error',
        },
      ],
    };
  }

  const parsedConfig = typescript.parseJsonConfigFileContent(
    configFile.config,
    typescript.sys,
    REPO_ROOT,
    undefined,
    configFilePath,
  );
  const typeScriptRootNames = sourceFilePaths
    .filter((filePath) => !filePath.endsWith(TEST_FILE_SUFFIX))
    .concat(await collectDeclarationFiles(resolvedFolder.absolutePath))
    .toSorted((leftPath, rightPath) => leftPath.localeCompare(rightPath));

  const program = typescript.createProgram({
    options: parsedConfig.options,
    rootNames: typeScriptRootNames,
  });
  const inFolderDiagnostics = typescript
    .getPreEmitDiagnostics(program)
    .filter((diagnostic) =>
      isDiagnosticInsideFolder(diagnostic, resolvedFolder.absolutePath),
    );

  return {
    evidence: `TypeScript (${configFileName}): ${inFolderDiagnostics.length} in-folder diagnostic(s) across ${typeScriptRootNames.length} file(s) (tests excluded, .d.ts included).`,
    smells: inFolderDiagnostics.map((diagnostic) => ({
      detail: flattenDiagnosticMessage(diagnostic.messageText),
      file: normalizePath(
        path.relative(REPO_ROOT, diagnostic.file?.fileName ?? configFilePath),
      ),
      kind: 'typescript-error',
    })),
  };
}

async function collectEslintSmells(sourceFilePaths) {
  if (sourceFilePaths.length === 0) {
    return {
      evidence: 'ESLint: no .ts files found under the target folder; skipped.',
      smells: [],
    };
  }

  const eslint = new ESLint({
    cwd: REPO_ROOT,
    errorOnUnmatchedPattern: false,
  });
  const lintResults = await eslint.lintFiles(sourceFilePaths);
  const smells = [];

  for (const lintResult of lintResults) {
    const filePath = normalizePath(
      path.relative(REPO_ROOT, lintResult.filePath),
    );
    for (const message of lintResult.messages.filter(
      (lintMessage) => lintMessage.severity === 2,
    )) {
      const locationSuffix = Number.isFinite(message.line)
        ? ` (${message.line}:${message.column ?? 1})`
        : '';
      const ruleLabel = message.ruleId ? `${message.ruleId}: ` : '';
      smells.push({
        detail: `${ruleLabel}${message.message}${locationSuffix}`,
        file: filePath,
        kind: 'lint-error',
      });
    }
  }

  return {
    evidence: `ESLint: ${smells.length} error(s) across ${lintResults.length} file(s).`,
    smells,
  };
}

async function collectJsdocSmells(moduleFilePaths) {
  if (moduleFilePaths.length === 0) {
    return {
      evidence: 'JSDoc: no source modules required JSDoc inspection.',
      smells: [],
    };
  }

  const smells = [];
  let exportedSymbolCount = 0;
  let documentedSymbolCount = 0;

  for (const moduleFilePath of moduleFilePaths) {
    const fileText = await readFile(moduleFilePath, 'utf8');
    const fileLines = fileText.split(/\r?\n/u);

    for (const exportedSymbol of listExportedSymbols(fileLines)) {
      exportedSymbolCount += 1;
      if (hasLeadingJsdoc(fileLines, exportedSymbol.lineIndex)) {
        documentedSymbolCount += 1;
        continue;
      }

      smells.push({
        detail: `Missing JSDoc for exported ${exportedSymbol.kind} ${exportedSymbol.name}.`,
        file: normalizePath(path.relative(REPO_ROOT, moduleFilePath)),
        kind: 'missing-jsdoc',
      });
    }
  }

  return {
    evidence: `JSDoc: ${documentedSymbolCount}/${exportedSymbolCount} exported symbol(s) documented across ${moduleFilePaths.length} file(s).`,
    smells,
  };
}

async function collectMissingTestFileSmells(
  moduleFilePaths,
  folderAbsolutePath,
) {
  if (moduleFilePaths.length === 0) {
    return {
      evidence:
        'Tests: no source modules required sibling test-file inspection.',
      smells: [],
    };
  }

  const changedSet = await collectChangedFilePaths(folderAbsolutePath);
  const smells = [];
  let checkedCount = 0;

  for (const moduleFilePath of moduleFilePaths) {
    // Pure-leaf util files are tested through their parent module's test file,
    // not a dedicated sibling test, so skip the sibling-test check for them.
    // Pure constant/type definition files (`.constants.ts`, `.types.ts`) are
    // also exempt: they contain no testable logic and are verified through the
    // modules that consume them.
    if (
      moduleFilePath.endsWith(UTIL_FILE_SUFFIX) ||
      NON_MODULE_FILE_SUFFIXES.some((suffix) => moduleFilePath.endsWith(suffix))
    ) {
      continue;
    }
    const relativeFilePath = normalizePath(
      path.relative(REPO_ROOT, moduleFilePath),
    );
    if (changedSet !== null && !changedSet.has(relativeFilePath)) {
      continue;
    }

    checkedCount += 1;
    const hasSiblingTest = await hasSiblingTestFile(moduleFilePath);
    if (hasSiblingTest) {
      continue;
    }

    const baseName = path.basename(moduleFilePath, SOURCE_FILE_SUFFIX);
    smells.push({
      detail: `Missing sibling test file ${baseName}.*.test.ts.`,
      file: relativeFilePath,
      kind: 'missing-test-file',
    });
  }

  const evidenceQualifier =
    changedSet !== null
      ? `checked ${checkedCount} changed source module(s)`
      : 'checked all source modules (git status unavailable)';

  return {
    evidence: `Tests: ${smells.length} source module(s) are missing a sibling .test.ts file (${evidenceQualifier}).`,
    smells,
  };
}

async function collectChangedFilePaths(folderAbsolutePath) {
  try {
    const { stdout } = await execAsync(
      `git status --short --untracked-files=all -- "${folderAbsolutePath}"`,
      { cwd: REPO_ROOT },
    );
    const changedPaths = new Set();
    const text = stdout ?? '';
    if (text.trim() === '') {
      return null;
    }

    for (const line of text.split(/\r?\n/u)) {
      if (line.length < 2) {
        continue;
      }
      const pathSegment = line.slice(2).trim();
      if (pathSegment === '') {
        continue;
      }

      if (pathSegment.includes(' -> ')) {
        for (const segmentPart of pathSegment.split(' -> ')) {
          changedPaths.add(normalizePath(segmentPart.trim()));
        }
      } else {
        changedPaths.add(normalizePath(pathSegment));
      }
    }

    return changedPaths.size > 0 ? changedPaths : null;
  } catch {
    return null;
  }
}

async function hasSiblingTestFile(moduleFilePath) {
  const directory = path.dirname(moduleFilePath);
  const baseName = path.basename(moduleFilePath, SOURCE_FILE_SUFFIX);
  const directoryEntries = await readdir(directory, { withFileTypes: true });

  return directoryEntries.some((directoryEntry) => {
    if (!directoryEntry.isFile()) {
      return false;
    }

    const name = directoryEntry.name;
    return (
      name === `${baseName}${TEST_FILE_SUFFIX}` ||
      (name.startsWith(`${baseName}.`) && name.endsWith(TEST_FILE_SUFFIX))
    );
  });
}

async function collectCoverageDeficitSmells(
  relativeFolderPath,
  moduleFilePaths,
) {
  if (!(await pathExists(COVERAGE_LCOV_PATH))) {
    return {
      evidence: 'Coverage: coverage/lcov.info missing; deficit lookup skipped.',
      smells: [],
    };
  }

  const coverageText = await readFile(COVERAGE_LCOV_PATH, 'utf8');
  const coverageByFile = parseLcovCoverage(coverageText);
  const smells = [];
  let matchedFileCount = 0;

  for (const moduleFilePath of moduleFilePaths) {
    const relativeFilePath = normalizePath(
      path.relative(REPO_ROOT, moduleFilePath),
    );
    if (!isRelativePathInside(relativeFolderPath, relativeFilePath)) {
      continue;
    }

    const coverageEntry = coverageByFile.get(relativeFilePath);
    if (!coverageEntry) {
      continue;
    }

    matchedFileCount += 1;
    if (
      coverageEntry.linesFound === 0 ||
      coverageEntry.linesHit === coverageEntry.linesFound
    ) {
      continue;
    }

    const lineCoveragePercent = (
      (coverageEntry.linesHit / coverageEntry.linesFound) *
      100
    ).toFixed(2);
    smells.push({
      detail: `Line coverage ${lineCoveragePercent}% (${coverageEntry.linesHit}/${coverageEntry.linesFound}) is below 100%.`,
      file: relativeFilePath,
      kind: 'coverage-deficit',
    });
  }

  return {
    evidence: `Coverage: ${matchedFileCount} lcov entr${matchedFileCount === 1 ? 'y' : 'ies'} matched the folder and ${smells.length} fell below 100% line coverage.`,
    smells,
  };
}

function listExportedSymbols(fileLines) {
  const exportedSymbols = [];

  for (const [lineIndex, fileLine] of fileLines.entries()) {
    const match = EXPORTED_SYMBOL_PATTERN.exec(fileLine);
    if (!match) {
      continue;
    }

    exportedSymbols.push({
      kind: match[1],
      lineIndex,
      name: match[2],
    });
  }

  return exportedSymbols;
}

function hasLeadingJsdoc(fileLines, exportLineIndex) {
  let currentLineIndex = exportLineIndex - 1;

  while (currentLineIndex >= 0 && fileLines[currentLineIndex].trim() === '') {
    currentLineIndex -= 1;
  }

  if (
    currentLineIndex < 0 ||
    !fileLines[currentLineIndex].trim().endsWith('*/')
  ) {
    return false;
  }

  while (currentLineIndex >= 0) {
    const trimmedLine = fileLines[currentLineIndex].trim();
    if (trimmedLine.startsWith('/**')) {
      return true;
    }

    if (!trimmedLine.startsWith('*') && !trimmedLine.startsWith('*/')) {
      return false;
    }

    currentLineIndex -= 1;
  }

  return false;
}

function deduplicateSmells(smells) {
  const smellMap = new Map();

  for (const smell of smells) {
    const smellKey = `${smell.kind}::${smell.file}::${smell.detail}`;
    if (!smellMap.has(smellKey)) {
      smellMap.set(smellKey, smell);
    }
  }

  return [...smellMap.values()];
}

function compareSmells(leftSmell, rightSmell) {
  return (
    leftSmell.file.localeCompare(rightSmell.file) ||
    leftSmell.kind.localeCompare(rightSmell.kind) ||
    leftSmell.detail.localeCompare(rightSmell.detail)
  );
}

function isDiagnosticInsideFolder(diagnostic, folderPath) {
  const diagnosticFilePath = diagnostic.file?.fileName;
  return (
    typeof diagnosticFilePath === 'string' &&
    isAbsolutePathInside(folderPath, diagnosticFilePath)
  );
}

function isAbsolutePathInside(parentPath, candidatePath) {
  const relativePath = path.relative(parentPath, candidatePath);
  return (
    relativePath !== '' &&
    !relativePath.startsWith('..') &&
    !path.isAbsolute(relativePath)
  );
}

function isRelativePathInside(parentPath, candidatePath) {
  const normalizedParentPath = normalizePath(parentPath);
  const normalizedCandidatePath = normalizePath(candidatePath);
  return (
    normalizedCandidatePath === normalizedParentPath ||
    normalizedCandidatePath.startsWith(`${normalizedParentPath}/`)
  );
}

function parseLcovCoverage(coverageText) {
  const coverageByFile = new Map();
  let currentFilePath = null;
  let linesFound = 0;
  let linesHit = 0;

  for (const coverageLine of coverageText.split(/\r?\n/u)) {
    if (coverageLine.startsWith('SF:')) {
      currentFilePath = normalizeCoveragePath(coverageLine.slice(3));
      linesFound = 0;
      linesHit = 0;
      continue;
    }

    if (coverageLine.startsWith('LF:')) {
      linesFound = Number.parseInt(coverageLine.slice(3), 10) || 0;
      continue;
    }

    if (coverageLine.startsWith('LH:')) {
      linesHit = Number.parseInt(coverageLine.slice(3), 10) || 0;
      continue;
    }

    if (coverageLine === 'end_of_record' && currentFilePath) {
      coverageByFile.set(currentFilePath, {
        linesFound,
        linesHit,
      });
      currentFilePath = null;
    }
  }

  return coverageByFile;
}

function normalizeCoveragePath(rawCoveragePath) {
  const normalizedRawPath = rawCoveragePath.replace(/\\/gu, '/').trim();
  const absolutePath = path.isAbsolute(normalizedRawPath)
    ? path.normalize(normalizedRawPath)
    : path.resolve(REPO_ROOT, normalizedRawPath);

  return normalizePath(path.relative(REPO_ROOT, absolutePath));
}

function flattenDiagnosticMessage(messageText) {
  return typescript.flattenDiagnosticMessageText(messageText, ' ');
}

async function pathExists(targetPath) {
  try {
    await access(targetPath, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function normalizePath(targetPath) {
  return targetPath.split(path.sep).join('/');
}

function writeReport(report, jsonOutput) {
  if (jsonOutput) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  console.log(
    report.pass
      ? `PASS folder-quality-metrics (${report.folderChecked})`
      : `FAIL folder-quality-metrics (${report.folderChecked})`,
  );
  for (const evidenceLine of report.evidence) {
    console.log(`- ${evidenceLine}`);
  }

  if (report.smells.length === 0) {
    return;
  }

  for (const smell of report.smells) {
    console.log(`- ${smell.kind}: ${smell.file}: ${smell.detail}`);
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await main();
}
