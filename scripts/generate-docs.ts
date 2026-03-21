/*
 * Generates per-folder README.md aggregating exported symbols' JSDoc.
 * - Copies root README.md into docs/ (manual content retained)
 * - Skips generating README for src root (leave top-level README manual)
 * Usage: npm run docs:folders
 */
import {
  Project,
  type JSDocTag,
  type SourceFile,
  type Symbol as MorphSymbol,
} from 'ts-morph';
import fg from 'fast-glob';
import fs from 'fs-extra';
import * as path from 'path';

const DOCS_DIR = path.resolve('docs');
const DEFAULT_DOCS_TARGET = 'src';
const FILE_SUMMARY_SYMBOL_NAME = '__file_summary__';
const SOURCE_FILE_GLOBS = ['**/*.ts'];
const SOURCE_FILE_IGNORE_GLOBS = ['**/*.d.ts'];
const FOLDER_INDEX_FILE_NAME = 'FOLDERS.md';
const DOCS_ORDER_CONFIG_FILE_NAME = 'docs.order.json';
const WORKSPACE_ROOT_DIR = path.resolve('.');

interface DocsTargetConfig {
  name: string;
  sourceDir: string;
  docsDir: string;
  rootDocsDir: string;
  rootReadmeSource?: string;
  rootReadmeDestination?: string;
  excludeRootSourceFiles?: boolean;
  includeFolderIndex?: boolean;
  publishedRootDir?: string;
  preservePublishedEntries?: string[];
}

interface RenderedParameter {
  name: string;
  type?: string;
  doc?: string;
}

interface RenderedSymbol {
  kind: string;
  name: string;
  filePath: string;
  parent?: string;
  signature?: string;
  jsdoc: {
    summary?: string;
    description?: string;
    params?: RenderedParameter[];
    returns?: string;
    deprecated?: string;
    examples?: string[];
  };
}

interface FolderIndexNode {
  name: string;
  path: string;
  children: Map<string, FolderIndexNode>;
  fileCount?: number;
}

interface DirectoryDocsOrderConfig {
  introFile?: string;
  fileOrder?: string[];
  symbolOrder?: Record<string, string[]>;
  folderOrder?: string[];
  hiddenFiles?: string[];
  hiddenSymbols?: string[];
}

interface LoadedDirectoryDocsOrderConfig {
  configPath: string;
  config: DirectoryDocsOrderConfig;
}

interface RenderedFileSummary {
  description?: string;
  examples?: string[];
}

type DirectorySymbolMap = Map<string, Map<string, RenderedSymbol[]>>;

const DOCS_TARGETS: Record<string, DocsTargetConfig> = {
  src: {
    name: 'src',
    sourceDir: path.resolve('src'),
    docsDir: DOCS_DIR,
    rootDocsDir: path.join(DOCS_DIR, 'src'),
    rootReadmeSource: path.resolve('README.md'),
    rootReadmeDestination: path.join(DOCS_DIR, 'README.md'),
    includeFolderIndex: true,
  },
  asciiMaze: {
    name: 'asciiMaze',
    sourceDir: path.resolve('test', 'examples', 'asciiMaze'),
    docsDir: path.join(DOCS_DIR, 'examples', 'asciiMaze', 'docs'),
    rootDocsDir: path.join(DOCS_DIR, 'examples', 'asciiMaze', 'docs'),
    rootReadmeSource: path.resolve('test', 'examples', 'asciiMaze', 'README.md'),
    rootReadmeDestination: path.join(
      DOCS_DIR,
      'examples',
      'asciiMaze',
      'docs',
      'README.md',
    ),
    excludeRootSourceFiles: true,
    publishedRootDir: path.join(DOCS_DIR, 'examples', 'asciiMaze'),
    preservePublishedEntries: ['index.html'],
  },
  'flappy-bird': {
    name: 'flappy-bird',
    sourceDir: path.resolve('test', 'examples', 'flappy_bird'),
    docsDir: path.join(DOCS_DIR, 'examples', 'flappy_bird', 'docs'),
    rootDocsDir: path.join(DOCS_DIR, 'examples', 'flappy_bird', 'docs'),
    rootReadmeSource: path.resolve(
      'test',
      'examples',
      'flappy_bird',
      'README.md',
    ),
    rootReadmeDestination: path.join(
      DOCS_DIR,
      'examples',
      'flappy_bird',
      'docs',
      'README.md',
    ),
    excludeRootSourceFiles: true,
    publishedRootDir: path.join(DOCS_DIR, 'examples', 'flappy_bird'),
    preservePublishedEntries: ['index.html'],
  },
};

const project = new Project({
  tsConfigFilePath: path.resolve('tsconfig.json'),
  skipAddingFilesFromTsConfig: true,
});
const resolvedDirectoryDocsOrderConfigCache = new Map<
  string,
  LoadedDirectoryDocsOrderConfig | undefined
>();
const directoryDocsOrderConfigCache = new Map<
  string,
  Promise<LoadedDirectoryDocsOrderConfig | undefined>
>();

/**
 * Generates docs for the requested target.
 *
 * The script runs in four passes:
 * 1. Resolve and prepare the target output tree.
 * 2. Load source files into the shared ts-morph project.
 * 3. Collect and normalize rendered symbols.
 * 4. Emit directory README files and the optional folder index.
 */
async function main(): Promise<void> {
  const target = resolveDocsTarget();
  await initializeDocsTarget(target);

  const sourceFiles = await loadTargetSourceFiles(target);
  const directorySymbolMap = collectDirectorySymbols(sourceFiles);

  dedupeDirectorySymbols(directorySymbolMap);
  await emitDirectoryDocs(directorySymbolMap, target);

  if (target.includeFolderIndex) {
    await emitFolderIndex(directorySymbolMap, target);
  }

  console.log(`[docs:${target.name}] Per-folder README generation complete.`);
}

/**
 * Resolves the active docs target from CLI or environment input.
 *
 * Lookup order:
 * 1. `--target=...`
 * 2. `DOCS_TARGET`
 * 3. default target
 *
 * @returns Concrete docs target configuration.
 */
function resolveDocsTarget(): DocsTargetConfig {
  const cliTarget = process.argv
    .find((argument) => argument.startsWith('--target='))
    ?.slice('--target='.length);
  const requestedTarget =
    cliTarget || process.env.DOCS_TARGET || DEFAULT_DOCS_TARGET;
  const resolvedTarget = DOCS_TARGETS[requestedTarget];

  if (!resolvedTarget) {
    const supportedTargets = Object.keys(DOCS_TARGETS).toSorted().join(', ');
    throw new Error(
      `Unknown docs target "${requestedTarget}". Supported targets: ${supportedTargets}`,
    );
  }

  return resolvedTarget;
}

/**
 * Prepares the output tree for the selected target.
 *
 * @param target - Target configuration.
 * @returns Nothing.
 */
async function initializeDocsTarget(target: DocsTargetConfig): Promise<void> {
  await cleanupPublishedRoot(target);
  await fs.ensureDir(target.docsDir);

  if (
    !target.rootReadmeSource ||
    !target.rootReadmeDestination ||
    !(await fs.pathExists(target.rootReadmeSource))
  ) {
    return;
  }

  await fs.ensureDir(path.dirname(target.rootReadmeDestination));
  await fs.copyFile(target.rootReadmeSource, target.rootReadmeDestination);
}

/**
 * Loads target source files into the shared ts-morph project.
 *
 * @param target - Target configuration.
 * @returns Filtered source files that belong to the target tree.
 */
async function loadTargetSourceFiles(
  target: DocsTargetConfig,
): Promise<SourceFile[]> {
  const filePaths = await fg(SOURCE_FILE_GLOBS, {
    cwd: target.sourceDir,
    absolute: true,
    ignore: SOURCE_FILE_IGNORE_GLOBS,
  });

  for (const filePath of filePaths) {
    if (!shouldIncludeSourceFile(filePath, target)) {
      continue;
    }

    project.addSourceFileAtPath(filePath);
  }

  const rawSourceFiles = project.getSourceFiles();
  const sourceFiles = rawSourceFiles.filter((sourceFile) => {
    const filePath = sourceFile.getFilePath();
    return !filePath.endsWith('.d.ts') && !/node_modules/.test(filePath);
  });

  console.log(
    `[docs:${target.name}] Loaded ${sourceFiles.length} source files (raw: ${rawSourceFiles.length})`,
  );

  return sourceFiles;
}

/**
 * Determines whether a discovered file should participate in docs generation.
 *
 * @param filePath - Absolute source file path.
 * @param target - Target configuration.
 * @returns True when the file belongs in the target set.
 */
function shouldIncludeSourceFile(
  filePath: string,
  target: DocsTargetConfig,
): boolean {
  if (/\.test\.ts$/i.test(filePath)) {
    return false;
  }

  if (!target.excludeRootSourceFiles) {
    return true;
  }

  const relativeDirectory = path.dirname(
    path.relative(target.sourceDir, filePath),
  );
  return relativeDirectory !== '' && relativeDirectory !== '.';
}

/**
 * Removes previously published generated files for targets that mirror output
 * into a browsable published directory.
 *
 * @param target - Target configuration.
 * @returns Nothing.
 */
async function cleanupPublishedRoot(target: DocsTargetConfig): Promise<void> {
  if (!target.publishedRootDir) {
    return;
  }

  await fs.ensureDir(target.publishedRootDir);
  const preservedEntries = new Set(target.preservePublishedEntries ?? []);
  const childEntries = await fs.readdir(target.publishedRootDir);

  for (const childEntry of childEntries) {
    if (preservedEntries.has(childEntry)) {
      continue;
    }

    await fs.remove(path.join(target.publishedRootDir, childEntry));
  }
}

/**
 * Collects renderable symbols grouped by directory and then by file.
 *
 * @param sourceFiles - Loaded source files to scan.
 * @returns Nested symbol map keyed by directory and then source file path.
 */
function collectDirectorySymbols(
  sourceFiles: readonly SourceFile[],
): DirectorySymbolMap {
  const directorySymbolMap: DirectorySymbolMap = new Map();

  for (const sourceFile of sourceFiles) {
    collectSourceFileSymbols(sourceFile, directorySymbolMap);
  }

  return directorySymbolMap;
}

/**
 * Collects every renderable symbol from a single source file.
 *
 * @param sourceFile - Source file being scanned.
 * @param directorySymbolMap - Aggregate directory symbol map.
 * @returns Nothing.
 */
function collectSourceFileSymbols(
  sourceFile: SourceFile,
  directorySymbolMap: DirectorySymbolMap,
): void {
  const filePath = sourceFile.getFilePath();
  const fileSymbolMap = resolveFileSymbolMap(directorySymbolMap, filePath);

  addFileSummarySymbol(sourceFile, fileSymbolMap);
  collectExportedDeclarations(sourceFile, fileSymbolMap);
  collectDocumentedTopLevelDeclarations(sourceFile, fileSymbolMap);
}

/**
 * Resolves the mutable file-symbol map for the file's directory.
 *
 * @param directorySymbolMap - Aggregate directory symbol map.
 * @param filePath - Absolute source file path.
 * @returns Per-directory file symbol map.
 */
function resolveFileSymbolMap(
  directorySymbolMap: DirectorySymbolMap,
  filePath: string,
): Map<string, RenderedSymbol[]> {
  const directoryPath = path.dirname(filePath);
  let fileSymbolMap = directorySymbolMap.get(directoryPath);

  if (!fileSymbolMap) {
    fileSymbolMap = new Map();
    directorySymbolMap.set(directoryPath, fileSymbolMap);
  }

  return fileSymbolMap;
}

/**
 * Adds a synthetic file-summary symbol when the file starts with a file-level
 * JSDoc block.
 *
 * @param sourceFile - Source file being scanned.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function addFileSummarySymbol(
  sourceFile: SourceFile,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  const fileSummary = resolveFileSummary(sourceFile);
  if (!fileSummary.description && !fileSummary.examples?.length) {
    return;
  }

  appendRenderedSymbol(fileSymbolMap, sourceFile.getFilePath(), {
    kind: 'File',
    name: FILE_SUMMARY_SYMBOL_NAME,
    filePath: sourceFile.getFilePath(),
    jsdoc: {
      description: fileSummary.description,
      examples: fileSummary.examples,
    },
  });

  console.log(`[docs] Added file summary for ${sourceFile.getBaseName()}`);
}

/**
 * Resolves the renderable file-summary JSDoc for a source file.
 *
 * @param sourceFile - Source file to inspect.
 * @returns File summary description/examples when present.
 */
function resolveFileSummary(sourceFile: SourceFile): RenderedFileSummary {
  const leadingFileSummary = extractLeadingFileJsDocSummary(sourceFile);
  if (leadingFileSummary.description || leadingFileSummary.examples?.length) {
    return leadingFileSummary;
  }

  const attachedFileSummary = extractAttachedFileJsDocSummary(sourceFile);
  if (attachedFileSummary.description || attachedFileSummary.examples?.length) {
    return attachedFileSummary;
  }

  return {
    description: getFirstJsDocDescription(
      sourceFile as unknown as { getJsDocs?: () => unknown[] },
    ),
  };
}

/**
 * Collects exported declarations and supported child members.
 *
 * @param sourceFile - Source file being scanned.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectExportedDeclarations(
  sourceFile: SourceFile,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  const exportedDeclarations = sourceFile.getExportedDeclarations();
  if (exportedDeclarations.size === 0) {
    return;
  }

  console.log(
    `[docs] ${sourceFile.getBaseName()} exports: ${exportedDeclarations.size}`,
  );

  for (const [, declarations] of exportedDeclarations) {
    for (const declaration of declarations) {
      const symbol = declaration.getSymbol();
      if (!symbol) {
        continue;
      }

      const renderedSymbol = renderSymbol(
        symbol,
        declaration.getKindName(),
        sourceFile.getFilePath(),
      );
      if (renderedSymbol) {
        appendRenderedSymbol(
          fileSymbolMap,
          sourceFile.getFilePath(),
          renderedSymbol,
        );
      }

      collectExportedObjectLiteralMembers(
        declaration as unknown as Record<string, unknown>,
        symbol.getName(),
        sourceFile.getFilePath(),
        fileSymbolMap,
      );
      collectExportedClassMembers(
        declaration as unknown as Record<string, unknown>,
        symbol.getName(),
        sourceFile.getFilePath(),
        fileSymbolMap,
      );
    }
  }
}

/**
 * Collects documented properties from exported object-literal variables.
 *
 * @param declaration - Candidate declaration.
 * @param parentName - Parent exported symbol name.
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectExportedObjectLiteralMembers(
  declaration: Record<string, unknown>,
  parentName: string,
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  try {
    const declarationKindName = (
      declaration.getKindName as (() => string) | undefined
    )?.();
    if (declarationKindName !== 'VariableDeclaration') {
      return;
    }

    const initializer = (
      declaration.getInitializer as (() => any) | undefined
    )?.();
    if (initializer?.getKindName?.() !== 'ObjectLiteralExpression') {
      return;
    }

    const properties = (initializer.getProperties?.() as any[]) || [];
    for (const property of properties) {
      const propertyName =
        property.getName?.() || property.getSymbol?.()?.getName?.();
      const renderedProperty = renderDeclaration(
        property,
        String(propertyName),
        property.getKindName?.() || 'Property',
        filePath,
        parentName,
      );

      if (renderedProperty) {
        appendRenderedSymbol(fileSymbolMap, filePath, renderedProperty);
      }
    }
  } catch {
    // Best-effort introspection only.
  }
}

/**
 * Collects documented members from exported classes.
 *
 * @param declaration - Candidate declaration.
 * @param parentName - Parent exported symbol name.
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectExportedClassMembers(
  declaration: Record<string, unknown>,
  parentName: string,
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  try {
    const declarationKindName = (
      declaration.getKindName as (() => string) | undefined
    )?.();
    if (declarationKindName !== 'ClassDeclaration') {
      return;
    }

    const members =
      (declaration.getMembers as (() => any[]) | undefined)?.() || [];
    for (const member of members) {
      const memberName = member.getName?.();
      if (!memberName) {
        continue;
      }

      const renderedMember = renderDeclaration(
        member,
        String(memberName),
        member.getKindName?.() || 'ClassMember',
        filePath,
        parentName,
      );

      if (renderedMember) {
        appendRenderedSymbol(fileSymbolMap, filePath, renderedMember);
      }
    }
  } catch {
    // Best-effort introspection only.
  }
}

/**
 * Collects documented top-level non-exported declarations.
 *
 * @param sourceFile - Source file being scanned.
 * @param fileSymbolMap - Mutable symbol map for the file's directory.
 * @returns Nothing.
 */
function collectDocumentedTopLevelDeclarations(
  sourceFile: SourceFile,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): void {
  try {
    const statements =
      (
        sourceFile as unknown as { getStatements?: () => any[] }
      ).getStatements?.() || [];

    for (const statement of statements) {
      const declarations = statement.getDeclarations?.() || [statement];

      for (const declaration of declarations) {
        const jsDocs = declaration.getJsDocs?.() || [];
        if (jsDocs.length === 0) {
          continue;
        }

        const name =
          declaration.getName?.() || declaration.getSymbol?.()?.getName?.();
        if (!name) {
          continue;
        }

        if (
          hasRenderedSymbol(
            fileSymbolMap,
            sourceFile.getFilePath(),
            String(name),
          )
        ) {
          continue;
        }

        const renderedDeclaration = renderDeclaration(
          declaration,
          String(name),
          declaration.getKindName?.() || 'Declaration',
          sourceFile.getFilePath(),
        );

        if (renderedDeclaration) {
          appendRenderedSymbol(
            fileSymbolMap,
            sourceFile.getFilePath(),
            renderedDeclaration,
          );
        }
      }
    }
  } catch {
    // Best-effort introspection only.
  }
}

/**
 * Checks whether a symbol name has already been collected for a file.
 *
 * @param fileSymbolMap - Mutable symbol map for the directory.
 * @param filePath - Absolute source file path.
 * @param name - Candidate symbol name.
 * @returns True when the symbol already exists in the collected set.
 */
function hasRenderedSymbol(
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  filePath: string,
  name: string,
): boolean {
  return (fileSymbolMap.get(filePath) ?? []).some(
    (symbol) => symbol.name === name,
  );
}

/**
 * Appends a rendered symbol to the per-file collection.
 *
 * @param fileSymbolMap - Mutable symbol map for the directory.
 * @param filePath - Absolute source file path.
 * @param renderedSymbol - Symbol to append.
 * @returns Nothing.
 */
function appendRenderedSymbol(
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  filePath: string,
  renderedSymbol: RenderedSymbol,
): void {
  const renderedSymbols = fileSymbolMap.get(filePath) ?? [];
  renderedSymbols.push(renderedSymbol);
  fileSymbolMap.set(filePath, renderedSymbols);
}

/**
 * Tidies and deduplicates collected symbols before README rendering.
 *
 * @param directorySymbolMap - Directory symbol map to normalize.
 * @returns Nothing.
 */
function dedupeDirectorySymbols(directorySymbolMap: DirectorySymbolMap): void {
  for (const [, fileSymbolMap] of directorySymbolMap) {
    for (const [filePath, renderedSymbols] of fileSymbolMap) {
      fileSymbolMap.set(filePath, dedupeFileSymbols(renderedSymbols, filePath));
    }
  }
}

/**
 * Deduplicates symbols for a single file while conservatively merging JSDoc.
 *
 * @param renderedSymbols - Symbols collected for one file.
 * @param filePath - Absolute source file path.
 * @returns Deduplicated symbol list.
 */
function dedupeFileSymbols(
  renderedSymbols: readonly RenderedSymbol[],
  filePath: string,
): RenderedSymbol[] {
  const seen = new Map<string, RenderedSymbol>();
  const dedupedSymbols: RenderedSymbol[] = [];

  for (const renderedSymbol of renderedSymbols) {
    const normalizedName = normalizeName(renderedSymbol, filePath);
    const dedupeKey = `${renderedSymbol.parent || ''}::${normalizedName}::${renderedSymbol.signature || ''}`;
    const existingSymbol = seen.get(dedupeKey);

    if (existingSymbol) {
      mergeRenderedSymbols(existingSymbol, renderedSymbol);
      continue;
    }

    const clonedSymbol: RenderedSymbol = {
      ...renderedSymbol,
      name: normalizedName,
      jsdoc: {
        ...renderedSymbol.jsdoc,
        params: renderedSymbol.jsdoc.params?.slice(),
        examples: renderedSymbol.jsdoc.examples?.slice(),
      },
    };

    seen.set(dedupeKey, clonedSymbol);
    dedupedSymbols.push(clonedSymbol);
  }

  return dedupedSymbols;
}

/**
 * Merges missing JSDoc fields from a duplicate symbol into the retained one.
 *
 * @param target - Existing retained symbol.
 * @param incoming - Duplicate symbol carrying possible extra data.
 * @returns Nothing.
 */
function mergeRenderedSymbols(
  target: RenderedSymbol,
  incoming: RenderedSymbol,
): void {
  target.jsdoc.description =
    target.jsdoc.description || incoming.jsdoc.description;
  target.jsdoc.summary = target.jsdoc.summary || incoming.jsdoc.summary;
  target.jsdoc.deprecated =
    target.jsdoc.deprecated || incoming.jsdoc.deprecated;

  if (incoming.jsdoc.examples?.length) {
    target.jsdoc.examples = [
      ...(target.jsdoc.examples || []),
      ...incoming.jsdoc.examples.filter(
        (incomingExample) => !(target.jsdoc.examples || []).includes(incomingExample),
      ),
    ];
  }

  if (incoming.jsdoc.params?.length) {
    target.jsdoc.params = (target.jsdoc.params || []).slice();

    for (const parameter of incoming.jsdoc.params) {
      const alreadyPresent = target.jsdoc.params.some(
        (existingParameter) => existingParameter.name === parameter.name,
      );
      if (!alreadyPresent) {
        target.jsdoc.params.push(parameter);
      }
    }
  }

  if (!target.signature && incoming.signature) {
    target.signature = incoming.signature;
  }
}

/**
 * Emits directory README files to both docs output and the source tree.
 *
 * @param directorySymbolMap - Normalized directory symbol map.
 * @param target - Target configuration.
 * @returns Nothing.
 */
async function emitDirectoryDocs(
  directorySymbolMap: DirectorySymbolMap,
  target: DocsTargetConfig,
): Promise<void> {
  for (const [directoryPath, fileSymbolMap] of directorySymbolMap) {
    const relativeDirectory = path.relative(target.sourceDir, directoryPath);
    if (relativeDirectory.startsWith('..')) {
      continue;
    }

    const outputDirectory =
      relativeDirectory === ''
        ? target.rootDocsDir
        : path.join(target.docsDir, relativeDirectory);
    const sourceDirectory =
      relativeDirectory === ''
        ? target.sourceDir
        : path.join(target.sourceDir, relativeDirectory);

    await loadDirectoryDocsOrderConfig(sourceDirectory);

    const markdown = buildDirectoryReadme(
      relativeDirectory,
      fileSymbolMap,
      target.sourceDir,
    );

    await fs.ensureDir(outputDirectory);
    await writeFileIfChanged(path.join(outputDirectory, 'README.md'), markdown);
    await writeFileIfChanged(path.join(sourceDirectory, 'README.md'), markdown);
  }
}

/**
 * Emits the folder index for targets that want a docs landing page.
 *
 * @param directorySymbolMap - Normalized directory symbol map.
 * @param target - Target configuration.
 * @returns Nothing.
 */
async function emitFolderIndex(
  directorySymbolMap: DirectorySymbolMap,
  target: DocsTargetConfig,
): Promise<void> {
  const rootNode: FolderIndexNode = {
    name: 'src',
    path: 'src',
    children: new Map(),
    fileCount: 0,
  };

  const relativeDirectories = [...directorySymbolMap.keys()]
    .map((directoryPath) => path.relative(target.sourceDir, directoryPath))
    .filter((relativeDirectory) => !relativeDirectory.startsWith('..'));

  const directoriesWithConfigSupport = [
    target.sourceDir,
    ...relativeDirectories.map((relativeDirectory) =>
      path.join(target.sourceDir, relativeDirectory),
    ),
  ];
  await Promise.all(
    [...new Set(directoriesWithConfigSupport)].map((directoryPath) =>
      loadDirectoryDocsOrderConfig(directoryPath),
    ),
  );

  for (const relativeDirectory of relativeDirectories) {
    insertFolderIndexNode(
      rootNode,
      relativeDirectory,
      target,
      directorySymbolMap,
    );
  }

  const markdown = buildFolderIndexMarkdown(rootNode);
  await writeFileIfChanged(
    path.join(DOCS_DIR, FOLDER_INDEX_FILE_NAME),
    markdown,
  );
}

/**
 * Inserts one relative directory into the folder index tree.
 *
 * @param rootNode - Root folder index node.
 * @param relativeDirectory - Relative directory from the target source root.
 * @param target - Target configuration.
 * @param directorySymbolMap - Directory symbol map used for file counts.
 * @returns Nothing.
 */
function insertFolderIndexNode(
  rootNode: FolderIndexNode,
  relativeDirectory: string,
  target: DocsTargetConfig,
  directorySymbolMap: DirectorySymbolMap,
): void {
  const normalizedDirectory =
    relativeDirectory === '' ? 'src' : relativeDirectory.replace(/\\/g, '/');
  const pathParts = normalizedDirectory.split('/');

  let currentNode = rootNode;
  let accumulatedPath = '';

  for (const pathPart of pathParts) {
    accumulatedPath = accumulatedPath
      ? `${accumulatedPath}/${pathPart}`
      : pathPart;

    if (!currentNode.children.has(pathPart)) {
      currentNode.children.set(pathPart, {
        name: pathPart,
        path: accumulatedPath,
        children: new Map(),
      });
    }

    currentNode = currentNode.children.get(pathPart)!;
  }

  const absoluteDirectoryPath = path.join(
    target.sourceDir,
    relativeDirectory === '' ? '' : relativeDirectory,
  );
  const fileSymbolMap = directorySymbolMap.get(absoluteDirectoryPath);
  if (fileSymbolMap) {
    currentNode.fileCount = resolveVisibleDirectoryFiles(
      absoluteDirectoryPath,
      [...fileSymbolMap.keys()],
    ).length;
  }
}

/**
 * Builds markdown for the generated folder index.
 *
 * @param rootNode - Root folder index node.
 * @returns Folder index markdown.
 */
function buildFolderIndexMarkdown(rootNode: FolderIndexNode): string {
  const lines = [
    '# Docs Index',
    '',
    'Auto-generated index of source folders (click to open folder README).',
    '',
  ];

  renderFolderIndexNode(rootNode, 0, lines);
  return `${lines.join('\n')}\n`;
}

/**
 * Renders one folder index node and its children recursively.
 *
 * @param node - Current folder index node.
 * @param level - Nesting level.
 * @param lines - Output line buffer.
 * @returns Nothing.
 */
function renderFolderIndexNode(
  node: FolderIndexNode,
  level: number,
  lines: string[],
): void {
  const indent = '  '.repeat(Math.max(0, level));
  const label = node.path === 'src' && level === 0 ? 'src (root)' : node.name;
  const countSuffix = node.fileCount
    ? ` — ${node.fileCount} file${node.fileCount > 1 ? 's' : ''}`
    : '';

  lines.push(`${indent}- [${label}](${node.path}/README.md)${countSuffix}`);

  const childNames = resolveSortedFolderIndexChildNames(node);
  for (const childName of childNames) {
    renderFolderIndexNode(node.children.get(childName)!, level + 1, lines);
  }
}

/**
 * Resolves child-folder order for one folder-index node.
 *
 * Configured folder order only affects direct children of the current folder.
 * Unspecified folders remain on the previous alphabetical fallback path.
 *
 * @param node - Folder index node whose children should be ordered.
 * @returns Sorted child folder names.
 */
function resolveSortedFolderIndexChildNames(node: FolderIndexNode): string[] {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(path.resolve(node.path));
  const configuredFolderOrder = loadedConfig?.config.folderOrder;

  if (configuredFolderOrder?.length) {
    warnForUnknownConfiguredFolderIndexChildren(
      configuredFolderOrder,
      [...node.children.keys()],
      loadedConfig?.configPath,
      node.path,
    );
  }

  const explicitFolderOrderIndex = new Map(
    (configuredFolderOrder ?? []).map((folderName, index) => [folderName, index] as const),
  );

  return [...node.children.keys()].toSorted((leftName, rightName) => {
    const leftExplicitOrder = explicitFolderOrderIndex.get(leftName);
    const rightExplicitOrder = explicitFolderOrderIndex.get(rightName);

    if (leftExplicitOrder !== undefined || rightExplicitOrder !== undefined) {
      if (leftExplicitOrder === undefined) {
        return 1;
      }

      if (rightExplicitOrder === undefined) {
        return -1;
      }

      if (leftExplicitOrder !== rightExplicitOrder) {
        return leftExplicitOrder - rightExplicitOrder;
      }
    }

    return leftName.localeCompare(rightName);
  });
}

/**
 * Extracts and parses the leading file-level JSDoc block from raw source text.
 *
 * @param sourceFile - Source file to inspect.
 * @returns Parsed top-of-file summary description/examples when present.
 */
function extractLeadingFileJsDocSummary(
  sourceFile: SourceFile,
): RenderedFileSummary {
  const text = sourceFile.getFullText();
  const match = text.match(/^\s*(?:\uFEFF)?\/\*\*([\s\S]*?)\*\//);
  if (!match) {
    return {};
  }

  const cleanedText = match[1]
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*\*\s?/, ''))
    .join('\n')
    .trim();

  if (!cleanedText) {
    return {};
  }

  if (
    cleanedText.split('\n').some((line) => line.trim().startsWith('@internal'))
  ) {
    return {};
  }

  return parseFileSummaryBlock(cleanedText);
}

/**
 * Parses a cleaned file-summary JSDoc block into description and examples.
 *
 * @param cleanedText - Cleaned raw file-summary block text.
 * @returns Parsed description/examples.
 */
function parseFileSummaryBlock(cleanedText: string): RenderedFileSummary {
  const tagStartIndex = cleanedText.search(/^@\w+/m);
  const description = sanitizeTagBlockText(
    tagStartIndex >= 0 ? cleanedText.slice(0, tagStartIndex) : cleanedText,
  );
  const examples = extractFileSummaryExamples(cleanedText);

  return {
    description,
    examples,
  };
}

/**
 * Extracts file-summary example blocks from a cleaned raw JSDoc block.
 *
 * @param cleanedText - Cleaned raw file-summary block text.
 * @returns Example blocks when present.
 */
function extractFileSummaryExamples(
  cleanedText: string,
): string[] | undefined {
  const exampleMatches = [
    ...cleanedText.matchAll(/(^|\n)@example\s*([\s\S]*?)(?=\n@\w+|$)/g),
  ];

  const examples = exampleMatches
    .map((match) => sanitizeTagBlockText(match[2]))
    .filter((example): example is string => Boolean(example));

  return examples.length > 0 ? examples : undefined;
}

/**
 * Extracts module-summary JSDoc blocks that are attached to the first
 * top-level declaration after imports/exports.
 *
 * @param sourceFile - Source file to inspect.
 * @returns Parsed description/examples when attached module-summary blocks exist.
 */
function extractAttachedFileJsDocSummary(
  sourceFile: SourceFile,
): RenderedFileSummary {
  const firstStatementWithMultipleJsDocs = sourceFile
    .getStatements()
    .find((statement) => {
      const statementJsDocs = getJsDocs(statement);
      return statementJsDocs.length > 1 && !hasInternalTag(statementJsDocs);
    });

  if (!firstStatementWithMultipleJsDocs) {
    return {};
  }

  const summaryJsDocs = getJsDocs(firstStatementWithMultipleJsDocs).slice(0, -1);
  return mergeRenderedFileSummaries(summaryJsDocs.map(renderJsDocSummaryBlock));
}

/**
 * Renders one exported symbol when it belongs to a supported declaration kind
 * and is not marked internal.
 *
 * @param symbol - Exported symbol.
 * @param fallbackKind - Declaration kind used as a fallback label.
 * @param filePath - Absolute source file path.
 * @returns Rendered symbol or null when not supported.
 */
function renderSymbol(
  symbol: MorphSymbol,
  fallbackKind: string,
  filePath: string,
): RenderedSymbol | null {
  const declaration = symbol.getDeclarations()[0];
  if (!declaration) {
    return null;
  }

  const declarationKind = declaration.getKindName();
  const supportedKinds =
    /ClassDeclaration|FunctionDeclaration|InterfaceDeclaration|EnumDeclaration|TypeAliasDeclaration|VariableDeclaration/;
  if (!supportedKinds.test(declarationKind)) {
    return null;
  }

  const jsDocs = resolveRenderableJsDocs(declaration);
  if (hasInternalTag(jsDocs)) {
    return null;
  }

  const primaryJsDoc = jsDocs.at(-1);
  const fullDescription = primaryJsDoc?.getDescription().trim();

  return {
    kind: fallbackKind || declarationKind,
    name: symbol.getName(),
    parent: undefined,
    filePath,
    signature: resolveCallSignature(declaration),
    jsdoc: {
      summary: fullDescription?.split(/\r?\n\r?\n/)[0]?.trim(),
      description: fullDescription,
      params: extractParamDocs(primaryJsDoc?.getTags() || []),
      examples: extractExampleDocs(primaryJsDoc?.getTags() || []),
      returns: getTagCommentText(
        primaryJsDoc
          ?.getTags()
          .find(
            (tag: JSDocTag) =>
              tag.getTagName() === 'returns' || tag.getTagName() === 'return',
          ),
      ),
      deprecated: getTagCommentText(
        primaryJsDoc
          ?.getTags()
          .find((tag: JSDocTag) => tag.getTagName() === 'deprecated'),
      ),
    },
  };
}

/**
 * Renders a declaration-like node that may not have a directly usable symbol.
 *
 * @param declaration - Declaration-like node.
 * @param forcedName - Optional forced display name.
 * @param forcedKind - Optional forced kind label.
 * @param filePath - Absolute source file path.
 * @param parentName - Optional parent display name.
 * @returns Rendered symbol or null when not renderable.
 */
function renderDeclaration(
  declaration: any,
  forcedName?: string,
  forcedKind?: string,
  filePath?: string,
  parentName?: string,
): RenderedSymbol | null {
  if (!declaration) {
    return null;
  }

  try {
    const jsDocs = resolveRenderableJsDocs(declaration);
    if (jsDocs.length === 0 || hasInternalTag(jsDocs)) {
      return null;
    }

    const primaryJsDoc = jsDocs.at(-1);
    const fullDescription = primaryJsDoc?.getDescription?.()?.trim();

    return {
      kind:
        forcedKind ||
        declaration.getKindName?.() ||
        declaration.getKind?.() ||
        'Declaration',
      name:
        forcedName ||
        declaration.getName?.() ||
        declaration.getSymbol?.()?.getName?.() ||
        (parentName ? `${parentName}.${forcedName}` : ''),
      parent: parentName || undefined,
      filePath: filePath || '',
      signature: resolveCallSignature(declaration),
      jsdoc: {
        summary: fullDescription?.split(/\r?\n\r?\n/)[0]?.trim(),
        description: fullDescription,
        params: extractParamDocs(primaryJsDoc?.getTags() || []),
        examples: extractExampleDocs(primaryJsDoc?.getTags() || []),
        returns: getTagCommentText(
          primaryJsDoc
            ?.getTags()
            .find(
              (tag: JSDocTag) =>
                tag.getTagName() === 'returns' || tag.getTagName() === 'return',
            ),
        ),
        deprecated: getTagCommentText(
          primaryJsDoc
            ?.getTags()
            .find((tag: JSDocTag) => tag.getTagName() === 'deprecated'),
        ),
      },
    };
  } catch {
    return null;
  }
}

/**
 * Returns JSDoc blocks from a declaration-like node.
 *
 * @param node - Declaration-like node.
 * @returns JSDoc array.
 */
function getJsDocs(node: any): any[] {
  return (node.getJsDocs?.() as any[]) || [];
}

/**
 * Resolves the JSDoc blocks that should be used for a renderable declaration.
 *
 * Exported constants are commonly documented on their enclosing
 * `VariableStatement` instead of on the inner `VariableDeclaration`. This
 * helper falls back to that enclosing statement so generated README entries for
 * constants keep their descriptions.
 *
 * @param node - Declaration-like node.
 * @returns JSDoc array, preferring direct docs and falling back when needed.
 */
function resolveRenderableJsDocs(node: any): any[] {
  const directJsDocs = getJsDocs(node);
  if (directJsDocs.length > 0) {
    return directJsDocs;
  }

  if (node?.getKindName?.() !== 'VariableDeclaration') {
    return directJsDocs;
  }

  const variableStatement =
    node.getFirstAncestorByKindName?.('VariableStatement') ||
    node.getVariableStatement?.();
  return getJsDocs(variableStatement);
}

/**
 * Resolves the first JSDoc description from a declaration-like node.
 *
 * @param node - Declaration-like node.
 * @returns Description text when present.
 */
function getFirstJsDocDescription(node: {
  getJsDocs?: () => unknown[];
}): string | undefined {
  const firstJsDoc = getJsDocs(node)[0];
  return firstJsDoc?.getDescription?.()?.trim() as string | undefined;
}

/**
 * Checks whether any provided JSDoc block contains an internal tag.
 *
 * @param jsDocs - JSDoc array to inspect.
 * @returns True when the declaration is marked internal.
 */
function hasInternalTag(jsDocs: readonly any[]): boolean {
  return jsDocs.some((jsDoc) =>
    jsDoc
      .getTags()
      .some((tag: { getTagName(): string }) => tag.getTagName() === 'internal'),
  );
}

/**
 * Resolves a call signature string from a declaration-like node.
 *
 * @param declaration - Declaration-like node.
 * @returns Signature string when the declaration is callable.
 */
function resolveCallSignature(declaration: any): string | undefined {
  try {
    const declarationType =
      declaration.getType?.() || declaration.getSymbol?.()?.getType?.();
    const callSignature = declarationType?.getCallSignatures?.()[0];
    if (!callSignature) {
      return undefined;
    }

    const renderedParameters = callSignature
      .getParameters()
      .map((parameter: any) => {
        const parameterDeclarations = parameter.getDeclarations();
        const parameterType = normalizeRenderedTypeText(
          parameter
          .getTypeAtLocation(parameterDeclarations[0] || declaration)
          .getText(),
        );
        return `${parameter.getName()}: ${parameterType}`;
      })
      .join(', ');

    const returnType = normalizeRenderedTypeText(
      callSignature.getReturnType().getText(),
    );
    return `(${renderedParameters}) => ${returnType}`;
  } catch {
    return undefined;
  }
}

/**
 * Formats a stored call signature into a human-friendly TypeScript block.
 *
 * @param symbolName - Symbol name shown in the heading.
 * @param signature - Stored canonical call signature.
 * @returns Fenced TypeScript block lines.
 */
function renderSignatureBlock(symbolName: string, signature: string): string[] {
  const parsedSignature = parseCallSignature(signature);
  if (!parsedSignature) {
    return ['```ts', `${symbolName}${signature}`, '```'];
  }

  if (parsedSignature.parameters.length === 0) {
    return [
      '```ts',
      `${symbolName}(): ${parsedSignature.returnType}`,
      '```',
    ];
  }

  return [
    '```ts',
    `${symbolName}(`,
    ...parsedSignature.parameters.map((parameter) => `  ${parameter},`),
    `): ${parsedSignature.returnType}`,
    '```',
  ];
}

/**
 * Parses a canonical stored call signature.
 *
 * @param signature - Stored canonical call signature.
 * @returns Parsed parameters and return type when recognized.
 */
function parseCallSignature(
  signature: string,
): { parameters: string[]; returnType: string } | undefined {
  if (!signature.startsWith('(')) {
    return undefined;
  }

  const closingParenthesisIndex = findMatchingDelimiter(signature, 0, '(', ')');
  if (closingParenthesisIndex === -1) {
    return undefined;
  }

  const parameterListText = signature.slice(1, closingParenthesisIndex);
  const returnTypePrefix = signature.slice(closingParenthesisIndex + 1).trimStart();
  if (!returnTypePrefix.startsWith('=>')) {
    return undefined;
  }

  return {
    parameters: splitTopLevelCommaSeparated(parameterListText),
    returnType: returnTypePrefix.slice(2).trim(),
  };
}

/**
 * Splits a comma-separated type list while respecting nested delimiters.
 *
 * @param value - Comma-separated text.
 * @returns Top-level list entries.
 */
function splitTopLevelCommaSeparated(value: string): string[] {
  if (!value.trim()) {
    return [];
  }

  const segments: string[] = [];
  let segmentStartIndex = 0;
  let parenthesisDepth = 0;
  let bracketDepth = 0;
  let braceDepth = 0;
  let angleDepth = 0;

  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === '(') {
      parenthesisDepth += 1;
      continue;
    }

    if (character === ')') {
      parenthesisDepth -= 1;
      continue;
    }

    if (character === '[') {
      bracketDepth += 1;
      continue;
    }

    if (character === ']') {
      bracketDepth -= 1;
      continue;
    }

    if (character === '{') {
      braceDepth += 1;
      continue;
    }

    if (character === '}') {
      braceDepth -= 1;
      continue;
    }

    if (character === '<') {
      angleDepth += 1;
      continue;
    }

    if (character === '>') {
      angleDepth = Math.max(0, angleDepth - 1);
      continue;
    }

    const isTopLevelComma =
      character === ',' &&
      parenthesisDepth === 0 &&
      bracketDepth === 0 &&
      braceDepth === 0 &&
      angleDepth === 0;
    if (!isTopLevelComma) {
      continue;
    }

    segments.push(value.slice(segmentStartIndex, index).trim());
    segmentStartIndex = index + 1;
  }

  segments.push(value.slice(segmentStartIndex).trim());
  return segments.filter(Boolean);
}

/**
 * Finds the matching closing delimiter for a starting delimiter.
 *
 * @param value - Text to inspect.
 * @param startIndex - Opening delimiter index.
 * @param openingDelimiter - Opening delimiter character.
 * @param closingDelimiter - Closing delimiter character.
 * @returns Closing delimiter index when found.
 */
function findMatchingDelimiter(
  value: string,
  startIndex: number,
  openingDelimiter: string,
  closingDelimiter: string,
): number {
  let delimiterDepth = 0;

  for (let index = startIndex; index < value.length; index += 1) {
    const character = value[index];
    if (character === openingDelimiter) {
      delimiterDepth += 1;
      continue;
    }

    if (character !== closingDelimiter) {
      continue;
    }

    delimiterDepth -= 1;
    if (delimiterDepth === 0) {
      return index;
    }
  }

  return -1;
}

/**
 * Rewrites absolute import paths from ts-morph type text into repo-relative paths.
 *
 * @param typeText - Raw type text returned by ts-morph.
 * @returns Normalized type text safe for generated docs.
 */
function normalizeRenderedTypeText(typeText: string): string {
  const normalizedImportPaths = typeText.replace(
    /import\((['"])([^'"]+)\1\)/g,
    (_match, quote: string, importPath: string) => {
      const normalizedImportPath = path.normalize(importPath);
      if (!path.isAbsolute(normalizedImportPath)) {
        return `import(${quote}${importPath}${quote})`;
      }

      const relativeImportPath = path
        .relative(WORKSPACE_ROOT_DIR, normalizedImportPath)
        .replace(/\\/g, '/');

      const portableImportPath = relativeImportPath.startsWith('..')
        ? importPath.replace(/\\/g, '/')
        : relativeImportPath;

      return `import(${quote}${portableImportPath}${quote})`;
    },
  );

  return normalizedImportPaths
    .replace(
      /\btypeof\s+import\((['"])([^'"]+)\1\)\.([A-Za-z_$][\w$]*)/g,
      'typeof $3',
    )
    .replace(
      /import\((['"])([^'"]+)\1\)\.([A-Za-z_$][\w$]*)/g,
      '$3',
    );
}

/**
 * Extracts rendered parameter docs from JSDoc tags.
 *
 * @param tags - JSDoc tags to inspect.
 * @returns Parameter docs when present.
 */
function extractParamDocs(
  tags: readonly JSDocTag[],
): RenderedParameter[] | undefined {
  const parameterDocs = tags
    .filter((tag) => tag.getTagName() === 'param')
    .map((tag) => {
      const match = tag.getText().match(/@param\s+(\w+)/);
      return {
        name: match?.[1] || '',
        doc: getTagCommentText(tag),
      } satisfies RenderedParameter;
    })
    .filter((parameter) => Boolean(parameter.name));

  return parameterDocs.length > 0 ? parameterDocs : undefined;
}

/**
 * Resolves a tag comment into plain text.
 *
 * @param tag - JSDoc tag to inspect.
 * @returns Flattened comment string when present.
 */
function getTagCommentText(tag: JSDocTag | undefined): string | undefined {
  const rawComment = tag?.getComment();
  if (typeof rawComment === 'string') {
    return sanitizeTagCommentText(rawComment);
  }

  if (Array.isArray(rawComment)) {
    return sanitizeTagCommentText(
      rawComment
      .map(
        (commentPart) =>
          (commentPart as { getText?: () => string }).getText?.() ||
          String(commentPart),
      )
      .join(' ')
      .trim(),
    );
  }

  return undefined;
}

/**
 * Remove ts-morph JSDoc tag artifacts such as standalone trailing asterisks.
 *
 * @param commentText - Flattened tag comment text.
 * @returns Cleaned comment text or undefined when nothing meaningful remains.
 */
function sanitizeTagCommentText(
  commentText: string | undefined,
): string | undefined {
  const normalizedComment = commentText
    ?.replace(/\r\n/g, '\n')
    .replace(/\n\s*\*\s*$/g, '')
    .replace(/^\s*\*\s*$/g, '')
    .trim();

  return normalizedComment ? normalizedComment : undefined;
}

/**
 * Normalizes rendered symbol names so generated headings remain stable.
 *
 * @param renderedSymbol - Symbol being normalized.
 * @param filePath - Source file path used as fallback context.
 * @returns Stable display name.
 */
function normalizeName(
  renderedSymbol: RenderedSymbol,
  filePath: string,
): string {
  let name = String(renderedSymbol.name || '');

  if (!name || name === 'default' || name === FILE_SUMMARY_SYMBOL_NAME) {
    const fileBaseName = path.basename(filePath || '', '.ts');

    if (renderedSymbol.kind === 'File' || name === FILE_SUMMARY_SYMBOL_NAME) {
      return fileBaseName;
    }

    if (renderedSymbol.parent) {
      return `${renderedSymbol.parent}.${fileBaseName}`;
    }

    if (renderedSymbol.signature) {
      return `${fileBaseName}${renderedSymbol.signature.split(')')[0]})`;
    }

    return fileBaseName;
  }

  return name.replace(/^function\s+/, '').replace(/\(\)$/, '');
}

/**
 * Builds the README markdown for one directory.
 *
 * @param relativeDirectory - Directory path relative to the target root.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @param sourceDir - Absolute source root.
 * @returns Markdown README content.
 */
function buildDirectoryReadme(
  relativeDirectory: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  sourceDir: string,
): string {
  const title = (relativeDirectory || path.basename(sourceDir)).replace(
    /\\/g,
    '/',
  );
  const directoryBaseName = path.basename(relativeDirectory || sourceDir);
  const directoryPath =
    relativeDirectory === ''
      ? sourceDir
      : path.join(sourceDir, relativeDirectory);
  const lines = [`# ${title}`, ''];

  const sortedFiles = resolveSortedDirectoryFiles(
    directoryPath,
    fileSymbolMap,
  );
  const hiddenSymbolNames = resolveHiddenDirectorySymbolNames(
    directoryPath,
    fileSymbolMap,
  );
  const visibleSortedFiles = resolveVisibleDirectoryFiles(
    directoryPath,
    sortedFiles,
  );

  const primaryDirectoryIntro = resolvePrimaryDirectoryIntro(
    directoryPath,
    visibleSortedFiles,
    fileSymbolMap,
    directoryBaseName,
  );

  if (primaryDirectoryIntro?.summary.description) {
    lines.push(primaryDirectoryIntro.summary.description, '');
  }

  if (primaryDirectoryIntro?.summary.examples?.length) {
    lines.push(
      primaryDirectoryIntro.summary.examples.length === 1
        ? 'Example:'
        : 'Examples:',
      '',
    );

    for (const example of primaryDirectoryIntro.summary.examples) {
      lines.push(example, '');
    }
  }

  for (const filePath of visibleSortedFiles) {
    renderFileReadmeSection(
      lines,
      filePath,
      fileSymbolMap,
      sourceDir,
      hiddenSymbolNames,
      primaryDirectoryIntro?.filePath === filePath,
    );
  }

  return `${lines.join('\n').trim()}\n`;
}

/**
 * Renders the README section for a single file.
 *
 * @param lines - Output line buffer.
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @param sourceDir - Absolute source root.
 * @returns Nothing.
 */
function renderFileReadmeSection(
  lines: string[],
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  sourceDir: string,
  hiddenSymbolNames: ReadonlySet<string>,
  suppressFileSummary: boolean = false,
): void {
  const visibleFileSymbols = (fileSymbolMap.get(filePath) ?? []).filter(
    (symbol) => !hiddenSymbolNames.has(symbol.name),
  );
  const fileSummarySymbol = visibleFileSymbols.find(
    (symbol) => symbol.kind === 'File',
  );
  const nonFileSummarySymbols = visibleFileSymbols.filter(
    (symbol) => symbol.kind !== 'File',
  );

  if (
    !fileSummarySymbol?.jsdoc.description &&
    !fileSummarySymbol?.jsdoc.examples?.length &&
    nonFileSummarySymbols.length === 0
  ) {
    return;
  }

  const relativeFilePath = path
    .relative(sourceDir, filePath)
    .replace(/\\/g, '/');
  lines.push(`## ${relativeFilePath}`, '');

  const fileBaseName = path.basename(filePath, '.ts');
  const sortedSymbols = visibleFileSymbols.toSorted((left, right) =>
    compareFileSectionSymbols(left, right, fileBaseName),
  );

  if (!suppressFileSummary && fileSummarySymbol?.jsdoc.description) {
    lines.push(fileSummarySymbol.jsdoc.description, '');
  }

  if (!suppressFileSummary && fileSummarySymbol?.jsdoc.examples?.length) {
    lines.push(
      fileSummarySymbol.jsdoc.examples.length === 1 ? 'Example:' : 'Examples:',
      '',
    );

    for (const example of fileSummarySymbol.jsdoc.examples) {
      lines.push(example, '');
    }
  }

  const topLevelSymbols = resolveSortedTopLevelFileSectionSymbols(
    filePath,
    nonFileSummarySymbols.filter((symbol) => !symbol.parent),
  );
  const symbolsByParent = groupSymbolsByParent(nonFileSummarySymbols);

  for (const topLevelSymbol of topLevelSymbols) {
    renderSymbolBlock(lines, topLevelSymbol, 3);

    const childSymbols = symbolsByParent.get(topLevelSymbol.name);
    if (!childSymbols) {
      continue;
    }

    childSymbols.sort((left, right) => left.name.localeCompare(right.name));
    for (const childSymbol of childSymbols) {
      renderSymbolBlock(lines, childSymbol, 4);
    }
  }

  for (const [parentName, childSymbols] of symbolsByParent) {
    if (topLevelSymbols.some((symbol) => symbol.name === parentName)) {
      continue;
    }

    lines.push(`### ${parentName}`, '');
    childSymbols.sort((left, right) => left.name.localeCompare(right.name));
    for (const childSymbol of childSymbols) {
      renderSymbolBlock(lines, childSymbol, 4);
    }
  }
}

/**
 * Compares file-section symbols using the pre-phase-3 stable fallback order.
 *
 * @param left - Left rendered symbol.
 * @param right - Right rendered symbol.
 * @param fileBaseName - Base file name without extension.
 * @returns Negative when left sorts before right.
 */
function compareFileSectionSymbols(
  left: RenderedSymbol,
  right: RenderedSymbol,
  fileBaseName: string,
): number {
  const leftIsPrimary = !left.parent && left.name === fileBaseName;
  const rightIsPrimary = !right.parent && right.name === fileBaseName;
  if (leftIsPrimary !== rightIsPrimary) {
    return leftIsPrimary ? -1 : 1;
  }

  return (
    (left.parent || left.name).localeCompare(right.parent || right.name) ||
    left.name.localeCompare(right.name)
  );
}

/**
 * Resolves top-level symbol order for one rendered file section.
 *
 * Configured symbol order applies only to top-level entries for the current
 * file. Unspecified symbols stay on the existing stable fallback path, and
 * nested member ordering remains conservative elsewhere.
 *
 * @param filePath - Absolute source file path.
 * @param topLevelSymbols - Stable fallback-sorted top-level symbols.
 * @returns Top-level symbols ordered for rendering.
 */
function resolveSortedTopLevelFileSectionSymbols(
  filePath: string,
  topLevelSymbols: readonly RenderedSymbol[],
): RenderedSymbol[] {
  const directoryPath = path.dirname(filePath);
  const loadedConfig = getCachedDirectoryDocsOrderConfig(directoryPath);
  const configuredSymbolOrder = loadedConfig?.config.symbolOrder?.[path.basename(filePath)];

  if (!configuredSymbolOrder?.length) {
    return [...topLevelSymbols];
  }

  warnForUnknownConfiguredFileSectionSymbols(
    configuredSymbolOrder,
    topLevelSymbols,
    loadedConfig?.configPath,
    filePath,
  );

  const explicitSymbolOrderIndex = new Map(
    configuredSymbolOrder.map((symbolName, index) => [symbolName, index] as const),
  );

  return [...topLevelSymbols].toSorted((leftSymbol, rightSymbol) => {
    const leftExplicitOrder = explicitSymbolOrderIndex.get(leftSymbol.name);
    const rightExplicitOrder = explicitSymbolOrderIndex.get(rightSymbol.name);

    if (leftExplicitOrder !== undefined || rightExplicitOrder !== undefined) {
      if (leftExplicitOrder === undefined) {
        return 1;
      }

      if (rightExplicitOrder === undefined) {
        return -1;
      }

      if (leftExplicitOrder !== rightExplicitOrder) {
        return leftExplicitOrder - rightExplicitOrder;
      }
    }

    return 0;
  });
}

/**
 * Resolves the file-summary block that should be promoted into the directory
 * README opening.
 *
 * @param sortedFiles - Files already ordered for directory README rendering.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns File path and summary when a promotable directory intro exists.
 */
function resolvePrimaryDirectoryIntro(
  directoryPath: string,
  sortedFiles: readonly string[],
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  directoryBaseName: string,
): { filePath: string; summary: RenderedFileSummary } | undefined {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(directoryPath);
  const configuredIntroFile = loadedConfig?.config.introFile;

  if (configuredIntroFile) {
    warnForUnknownConfiguredDirectoryIntroFile(
      configuredIntroFile,
      sortedFiles,
      loadedConfig?.configPath,
    );

    const configuredIntroFilePath = sortedFiles.find(
      (filePath) => path.basename(filePath) === configuredIntroFile,
    );
    const configuredSummary = configuredIntroFilePath
      ? resolveRenderedFileSummary(configuredIntroFilePath, fileSymbolMap)
      : undefined;

    if (
      configuredIntroFilePath &&
      configuredSummary &&
      (configuredSummary.description || configuredSummary.examples?.length)
    ) {
      return {
        filePath: configuredIntroFilePath,
        summary: configuredSummary,
      };
    }
  }

  const prioritizedFiles = sortedFiles.toSorted((leftFile, rightFile) => {
    const leftBaseName = path.basename(leftFile, '.ts');
    const rightBaseName = path.basename(rightFile, '.ts');
    const leftIsDirectoryEntry =
      leftBaseName === directoryBaseName || leftBaseName === 'index';
    const rightIsDirectoryEntry =
      rightBaseName === directoryBaseName || rightBaseName === 'index';

    if (leftIsDirectoryEntry !== rightIsDirectoryEntry) {
      return leftIsDirectoryEntry ? -1 : 1;
    }

    return sortedFiles.indexOf(leftFile) - sortedFiles.indexOf(rightFile);
  });

  for (const filePath of prioritizedFiles) {
    const summary = resolveRenderedFileSummary(filePath, fileSymbolMap);

    if (summary.description || summary.examples?.length) {
      return { filePath, summary };
    }
  }

  return undefined;
}

/**
 * Resolves the rendered file-summary block for one collected source file.
 *
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Rendered file summary values when present.
 */
function resolveRenderedFileSummary(
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): RenderedFileSummary {
  const fileSummarySymbol = (fileSymbolMap.get(filePath) ?? []).find(
    (renderedSymbol) => renderedSymbol.kind === 'File',
  );

  return {
    description: fileSummarySymbol?.jsdoc.description,
    examples: fileSummarySymbol?.jsdoc.examples,
  };
}

/**
 * Groups symbols that have parent names.
 *
 * @param renderedSymbols - Symbols for one file.
 * @returns Parent-keyed symbol map.
 */
function groupSymbolsByParent(
  renderedSymbols: readonly RenderedSymbol[],
): Map<string, RenderedSymbol[]> {
  const symbolsByParent = new Map<string, RenderedSymbol[]>();

  for (const renderedSymbol of renderedSymbols.filter(
    (symbol) => symbol.parent,
  )) {
    const childSymbols = symbolsByParent.get(renderedSymbol.parent!) ?? [];
    childSymbols.push(renderedSymbol);
    symbolsByParent.set(renderedSymbol.parent!, childSymbols);
  }

  return symbolsByParent;
}

/**
 * Renders one symbol block into the README line buffer.
 *
 * @param lines - Output line buffer.
 * @param renderedSymbol - Symbol to render.
 * @param headingLevel - Markdown heading level.
 * @returns Nothing.
 */
function renderSymbolBlock(
  lines: string[],
  renderedSymbol: RenderedSymbol,
  headingLevel: number,
): void {
  const symbolDescription =
    renderedSymbol.jsdoc.description || renderedSymbol.jsdoc.summary;

  lines.push(`${'#'.repeat(headingLevel)} ${renderedSymbol.name}`);

  if (renderedSymbol.signature) {
    lines.push('', ...renderSignatureBlock(renderedSymbol.name, renderedSymbol.signature));
  }

  if (symbolDescription) {
    lines.push('', symbolDescription);
  }

  if (renderedSymbol.jsdoc.deprecated) {
    lines.push('', `**Deprecated:** ${renderedSymbol.jsdoc.deprecated}`);
  }

  if (renderedSymbol.jsdoc.params?.length) {
    lines.push('', 'Parameters:');
    for (const parameter of renderedSymbol.jsdoc.params) {
      lines.push(
        `- \`${parameter.name}\`${parameter.doc ? ` - ${parameter.doc}` : ''}`,
      );
    }
  }

  if (renderedSymbol.jsdoc.returns) {
    lines.push('', `Returns: ${renderedSymbol.jsdoc.returns}`);
  }

  if (renderedSymbol.jsdoc.examples?.length) {
    lines.push('', renderedSymbol.jsdoc.examples.length === 1 ? 'Example:' : 'Examples:');

    for (const example of renderedSymbol.jsdoc.examples) {
      lines.push('', example);
    }
  }

  lines.push('');
}

/**
 * Extracts rendered example blocks from JSDoc tags.
 *
 * @param tags - JSDoc tags to inspect.
 * @returns Example blocks when present.
 */
function extractExampleDocs(tags: readonly JSDocTag[]): string[] | undefined {
  const examples = tags
    .filter((tag) => tag.getTagName() === 'example')
    .map((tag) => getExampleTagText(tag))
    .filter((example): example is string => Boolean(example));

  return examples.length > 0 ? examples : undefined;
}

/**
 * Resolves an example tag into markdown-friendly multiline text.
 *
 * @param tag - JSDoc example tag.
 * @returns Example block content when present.
 */
function getExampleTagText(tag: JSDocTag | undefined): string | undefined {
  const rawTagText = tag?.getText()?.replace(/\r\n/g, '\n');
  if (!rawTagText) {
    return undefined;
  }

  const exampleBody = rawTagText.replace(/^@example\s*/, '');
  return sanitizeTagBlockText(exampleBody);
}

/**
 * Renders one JSDoc block into file-summary description/examples.
 *
 * @param jsDoc - JSDoc block to render.
 * @returns Parsed description/examples.
 */
function renderJsDocSummaryBlock(jsDoc: any): RenderedFileSummary {
  const description = sanitizeTagBlockText(jsDoc?.getDescription?.()?.trim());
  const examples = extractExampleDocs(jsDoc?.getTags?.() || []);

  return {
    description,
    examples,
  };
}

/**
 * Merges multiple parsed file-summary blocks.
 *
 * @param summaries - Parsed summary blocks.
 * @returns One merged file summary.
 */
function mergeRenderedFileSummaries(
  summaries: readonly RenderedFileSummary[],
): RenderedFileSummary {
  const descriptions = summaries
    .map((summary) => summary.description)
    .filter((description): description is string => Boolean(description));
  const examples = summaries
    .flatMap((summary) => summary.examples || [])
    .filter(
      (example, index, allExamples) => allExamples.indexOf(example) === index,
    );

  return {
    description: descriptions.length > 0 ? descriptions.join('\n\n') : undefined,
    examples: examples.length > 0 ? examples : undefined,
  };
}

/**
 * Ranks files within a directory README so entrypoints appear before helpers.
 *
 * @param filePath - Absolute source file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Sort rank tuple.
 */
function rankFileForDirectoryReadme(
  filePath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): number[] {
  const fileName = path.basename(filePath).toLowerCase();
  const renderedSymbols = fileSymbolMap.get(filePath) ?? [];

  const hasFileSummary = renderedSymbols.some((renderedSymbol) => {
    if (
      renderedSymbol.kind !== 'File' ||
      renderedSymbol.name !== FILE_SUMMARY_SYMBOL_NAME
    ) {
      return false;
    }

    return Boolean(renderedSymbol.jsdoc.description?.trim());
  });

  const isIndexFile = fileName === 'index.ts';
  const isTypesFile =
    fileName.includes('.types.') || fileName.endsWith('.types.ts');
  const isUtilityLike =
    /(\.utils\.|\.export-|\.import-|\.internal\.|\.private\.)/i.test(fileName);
  const isEntrypointLike = !isUtilityLike;

  return [
    isIndexFile ? 0 : 1,
    hasFileSummary ? 0 : 1,
    isEntrypointLike ? 0 : 1,
    isTypesFile ? 0 : 1,
    fileName.length,
  ];
}

/**
 * Normalizes multiline tag blocks while preserving markdown structure.
 *
 * @param blockText - Multiline tag block text.
 * @returns Cleaned block text or undefined when empty.
 */
function sanitizeTagBlockText(
  blockText: string | undefined,
): string | undefined {
  const normalizedBlock = blockText
    ?.replace(/\r\n/g, '\n')
    .replace(/^\s*\* ?/gm, '')
    .replace(/^\n+/, '')
    .replace(/\n+$/, '')
    .trim();

  return normalizedBlock ? normalizedBlock : undefined;
}

/**
 * Loads and caches the optional per-folder docs ordering config.
 *
 * Phase 1 only validates and stores config so later phases can consume it at
 * the existing generator seams without changing default behavior today.
 *
 * @param directoryPath - Absolute directory path that may contain docs config.
 * @returns Parsed and validated config when present.
 */
async function loadDirectoryDocsOrderConfig(
  directoryPath: string,
): Promise<LoadedDirectoryDocsOrderConfig | undefined> {
  const normalizedDirectoryPath = path.resolve(directoryPath);
  if (directoryDocsOrderConfigCache.has(normalizedDirectoryPath)) {
    return directoryDocsOrderConfigCache.get(normalizedDirectoryPath)!;
  }

  const pendingConfig = readDirectoryDocsOrderConfig(normalizedDirectoryPath).then(
    (loadedConfig) => {
      resolvedDirectoryDocsOrderConfigCache.set(
        normalizedDirectoryPath,
        loadedConfig,
      );
      return loadedConfig;
    },
  );
  directoryDocsOrderConfigCache.set(normalizedDirectoryPath, pendingConfig);
  return pendingConfig;
}

/**
 * Reads and validates one directory's `docs.order.json` file when present.
 *
 * @param directoryPath - Absolute directory path.
 * @returns Parsed config or undefined when absent or invalid.
 */
async function readDirectoryDocsOrderConfig(
  directoryPath: string,
): Promise<LoadedDirectoryDocsOrderConfig | undefined> {
  const configPath = path.join(directoryPath, DOCS_ORDER_CONFIG_FILE_NAME);
  if (!(await fs.pathExists(configPath))) {
    return undefined;
  }

  try {
    const rawConfigText = await fs.readFile(configPath, 'utf8');
    const parsedConfig: unknown = JSON.parse(rawConfigText);
    const validatedConfig = validateDirectoryDocsOrderConfig(
      parsedConfig,
      configPath,
    );
    return validatedConfig ? { configPath, config: validatedConfig } : undefined;
  } catch (error) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Failed to read ${DOCS_ORDER_CONFIG_FILE_NAME}: ${getErrorMessage(error)}`,
    );
    return undefined;
  }
}

/**
 * Validates the supported phase-1 docs ordering config keys.
 *
 * @param parsedConfig - Raw parsed JSON value.
 * @param configPath - Absolute config path used in warnings.
 * @returns Sanitized config when at least one supported value is valid.
 */
function validateDirectoryDocsOrderConfig(
  parsedConfig: unknown,
  configPath: string,
): DirectoryDocsOrderConfig | undefined {
  if (!isRecord(parsedConfig)) {
    warnDirectoryDocsOrderConfig(
      configPath,
      'Config must be a JSON object. Ignoring file.',
    );
    return undefined;
  }

  const supportedKeys = new Set([
    'introFile',
    'fileOrder',
    'symbolOrder',
    'folderOrder',
    'hiddenFiles',
    'hiddenSymbols',
  ]);
  for (const configKey of Object.keys(parsedConfig)) {
    if (!supportedKeys.has(configKey)) {
      warnDirectoryDocsOrderConfig(
        configPath,
        `Unknown key "${configKey}". Supported keys: ${[...supportedKeys].join(', ')}`,
      );
    }
  }

  const validatedConfig: DirectoryDocsOrderConfig = {};
  const introFile = normalizeOptionalStringConfigValue(
    parsedConfig.introFile,
    'introFile',
    configPath,
  );
  if (introFile) {
    validatedConfig.introFile = introFile;
  }

  const fileOrder = normalizeStringArrayConfigValue(
    parsedConfig.fileOrder,
    'fileOrder',
    configPath,
  );
  if (fileOrder) {
    validatedConfig.fileOrder = fileOrder;
  }

  const symbolOrder = normalizeSymbolOrderConfigValue(
    parsedConfig.symbolOrder,
    configPath,
  );
  if (symbolOrder) {
    validatedConfig.symbolOrder = symbolOrder;
  }

  const folderOrder = normalizeStringArrayConfigValue(
    parsedConfig.folderOrder,
    'folderOrder',
    configPath,
  );
  if (folderOrder) {
    validatedConfig.folderOrder = folderOrder;
  }

  const hiddenFiles = normalizeStringArrayConfigValue(
    parsedConfig.hiddenFiles,
    'hiddenFiles',
    configPath,
  );
  if (hiddenFiles) {
    validatedConfig.hiddenFiles = hiddenFiles;
  }

  const hiddenSymbols = normalizeStringArrayConfigValue(
    parsedConfig.hiddenSymbols,
    'hiddenSymbols',
    configPath,
  );
  if (hiddenSymbols) {
    validatedConfig.hiddenSymbols = hiddenSymbols;
  }

  return Object.keys(validatedConfig).length > 0 ? validatedConfig : {};
}

/**
 * Returns the cached docs ordering config for a directory.
 *
 * The loader is called before directory emission, so this helper stays
 * synchronous for render-time consumers in later phases.
 *
 * @param directoryPath - Absolute directory path.
 * @returns Cached config when already loaded.
 */
function getCachedDirectoryDocsOrderConfig(
  directoryPath: string,
): LoadedDirectoryDocsOrderConfig | undefined {
  return resolvedDirectoryDocsOrderConfigCache.get(path.resolve(directoryPath));
}

/**
 * Resolves directory file order using config-first ordering with stable
 * fallback to the existing heuristic rank.
 *
 * @param directoryPath - Absolute directory path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Files sorted for one directory README.
 */
function resolveSortedDirectoryFiles(
  directoryPath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): string[] {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(directoryPath);
  const configuredFileOrder = loadedConfig?.config.fileOrder;

  if (configuredFileOrder?.length) {
    warnForUnknownConfiguredDirectoryFiles(
      configuredFileOrder,
      [...fileSymbolMap.keys()],
      loadedConfig?.configPath,
    );
  }

  const explicitFileOrderIndex = new Map(
    (configuredFileOrder ?? []).map((fileName, index) => [fileName, index] as const),
  );

  return [...fileSymbolMap.keys()].toSorted((leftFile, rightFile) => {
    const leftExplicitOrder = explicitFileOrderIndex.get(path.basename(leftFile));
    const rightExplicitOrder = explicitFileOrderIndex.get(path.basename(rightFile));

    if (leftExplicitOrder !== undefined || rightExplicitOrder !== undefined) {
      if (leftExplicitOrder === undefined) {
        return 1;
      }

      if (rightExplicitOrder === undefined) {
        return -1;
      }

      if (leftExplicitOrder !== rightExplicitOrder) {
        return leftExplicitOrder - rightExplicitOrder;
      }
    }

    return compareDirectoryReadmeFiles(leftFile, rightFile, fileSymbolMap);
  });
}

/**
 * Resolves visible files for one directory README after applying optional
 * `hiddenFiles` filtering.
 *
 * @param directoryPath - Absolute directory path.
 * @param sortedFiles - Already sorted directory files.
 * @returns Visible files that should remain in the generated output.
 */
function resolveVisibleDirectoryFiles(
  directoryPath: string,
  sortedFiles: readonly string[],
): string[] {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(directoryPath);
  const configuredHiddenFiles = loadedConfig?.config.hiddenFiles;

  if (configuredHiddenFiles?.length) {
    warnForUnknownConfiguredHiddenFiles(
      configuredHiddenFiles,
      sortedFiles,
      loadedConfig?.configPath,
    );
  }

  const hiddenFileNames = new Set(configuredHiddenFiles ?? []);
  return sortedFiles.filter(
    (filePath) => !hiddenFileNames.has(path.basename(filePath)),
  );
}

/**
 * Resolves hidden symbol names for one directory README.
 *
 * @param directoryPath - Absolute directory path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Hidden symbol-name set for render-time filtering.
 */
function resolveHiddenDirectorySymbolNames(
  directoryPath: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): ReadonlySet<string> {
  const loadedConfig = getCachedDirectoryDocsOrderConfig(directoryPath);
  const configuredHiddenSymbols = loadedConfig?.config.hiddenSymbols;

  if (configuredHiddenSymbols?.length) {
    warnForUnknownConfiguredHiddenSymbols(
      configuredHiddenSymbols,
      fileSymbolMap,
      loadedConfig?.configPath,
    );
  }

  return new Set(configuredHiddenSymbols ?? []);
}

/**
 * Warns when `fileOrder` references files missing from the current folder.
 *
 * @param configuredFileOrder - Configured file order entries.
 * @param filePaths - Actual files in the current directory README.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
function warnForUnknownConfiguredDirectoryFiles(
  configuredFileOrder: readonly string[],
  filePaths: readonly string[],
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownFileNames = new Set(filePaths.map((filePath) => path.basename(filePath)));
  const unknownFileNames = configuredFileOrder.filter(
    (fileName) => !knownFileNames.has(fileName),
  );

  if (unknownFileNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "fileOrder" references unknown files for this folder: ${unknownFileNames.join(', ')}`,
  );
}

/**
 * Warns when `hiddenFiles` references files missing from the current folder.
 *
 * @param configuredHiddenFiles - Configured hidden file entries.
 * @param filePaths - Actual files in the current directory README.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
function warnForUnknownConfiguredHiddenFiles(
  configuredHiddenFiles: readonly string[],
  filePaths: readonly string[],
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownFileNames = new Set(filePaths.map((filePath) => path.basename(filePath)));
  const unknownFileNames = configuredHiddenFiles.filter(
    (fileName) => !knownFileNames.has(fileName),
  );

  if (unknownFileNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "hiddenFiles" references unknown files for this folder: ${unknownFileNames.join(', ')}`,
  );
}

/**
 * Warns when `introFile` references a file missing from the current folder.
 *
 * @param configuredIntroFile - Configured intro file name.
 * @param filePaths - Actual files in the current directory README.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
function warnForUnknownConfiguredDirectoryIntroFile(
  configuredIntroFile: string,
  filePaths: readonly string[],
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownFileNames = new Set(filePaths.map((filePath) => path.basename(filePath)));
  if (knownFileNames.has(configuredIntroFile)) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "introFile" references an unknown file for this folder: ${configuredIntroFile}`,
  );
}

/**
 * Warns when `symbolOrder` references top-level symbols missing from one file
 * section.
 *
 * @param configuredSymbolOrder - Configured symbol order entries.
 * @param topLevelSymbols - Top-level symbols rendered for the current file.
 * @param configPath - Absolute config path.
 * @param filePath - Absolute source file path for warning context.
 * @returns Nothing.
 */
function warnForUnknownConfiguredFileSectionSymbols(
  configuredSymbolOrder: readonly string[],
  topLevelSymbols: readonly RenderedSymbol[],
  configPath: string | undefined,
  filePath: string,
): void {
  if (!configPath) {
    return;
  }

  const knownSymbolNames = new Set(topLevelSymbols.map((symbol) => symbol.name));
  const unknownSymbolNames = configuredSymbolOrder.filter(
    (symbolName) => !knownSymbolNames.has(symbolName),
  );

  if (unknownSymbolNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "symbolOrder.${path.basename(filePath)}" references unknown top-level symbols for this file section: ${unknownSymbolNames.join(', ')}`,
  );
}

/**
 * Warns when `hiddenSymbols` references symbols missing from the current
 * directory README.
 *
 * @param configuredHiddenSymbols - Configured hidden symbol entries.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @param configPath - Absolute config path.
 * @returns Nothing.
 */
function warnForUnknownConfiguredHiddenSymbols(
  configuredHiddenSymbols: readonly string[],
  fileSymbolMap: Map<string, RenderedSymbol[]>,
  configPath: string | undefined,
): void {
  if (!configPath) {
    return;
  }

  const knownSymbolNames = new Set(
    [...fileSymbolMap.values()].flatMap((renderedSymbols) =>
      renderedSymbols.map((symbol) => symbol.name),
    ),
  );
  const unknownSymbolNames = configuredHiddenSymbols.filter(
    (symbolName) => !knownSymbolNames.has(symbolName),
  );

  if (unknownSymbolNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "hiddenSymbols" references unknown symbols for this folder: ${unknownSymbolNames.join(', ')}`,
  );
}

/**
 * Warns when `folderOrder` references child folders missing from one folder
 * index node.
 *
 * @param configuredFolderOrder - Configured folder order entries.
 * @param childFolderNames - Actual child folder names under the current node.
 * @param configPath - Absolute config path.
 * @param nodePath - Relative folder-index node path for warning context.
 * @returns Nothing.
 */
function warnForUnknownConfiguredFolderIndexChildren(
  configuredFolderOrder: readonly string[],
  childFolderNames: readonly string[],
  configPath: string | undefined,
  nodePath: string,
): void {
  if (!configPath) {
    return;
  }

  const knownFolderNames = new Set(childFolderNames);
  const unknownFolderNames = configuredFolderOrder.filter(
    (folderName) => !knownFolderNames.has(folderName),
  );

  if (unknownFolderNames.length === 0) {
    return;
  }

  warnDirectoryDocsOrderConfig(
    configPath,
    `Field "folderOrder" references unknown child folders for ${nodePath}: ${unknownFolderNames.join(', ')}`,
  );
}

/**
 * Compares two files using the pre-phase-2 directory README heuristic.
 *
 * @param leftFile - Left file path.
 * @param rightFile - Right file path.
 * @param fileSymbolMap - Symbols grouped by file within the directory.
 * @returns Negative when left sorts before right.
 */
function compareDirectoryReadmeFiles(
  leftFile: string,
  rightFile: string,
  fileSymbolMap: Map<string, RenderedSymbol[]>,
): number {
  const leftRank = rankFileForDirectoryReadme(leftFile, fileSymbolMap);
  const rightRank = rankFileForDirectoryReadme(rightFile, fileSymbolMap);

  for (
    let index = 0;
    index < Math.max(leftRank.length, rightRank.length);
    index += 1
  ) {
    const leftValue = leftRank[index] ?? 0;
    const rightValue = rightRank[index] ?? 0;
    if (leftValue !== rightValue) {
      return leftValue - rightValue;
    }
  }

  return leftFile.localeCompare(rightFile);
}

/**
 * Normalizes a single optional string config field.
 *
 * @param value - Raw config value.
 * @param fieldName - Config field name.
 * @param configPath - Absolute config path used in warnings.
 * @returns Trimmed string when valid.
 */
function normalizeOptionalStringConfigValue(
  value: unknown,
  fieldName: string,
  configPath: string,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== 'string' || value.trim().length === 0) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Field "${fieldName}" must be a non-empty string. Ignoring value.`,
    );
    return undefined;
  }

  return value.trim();
}

/**
 * Normalizes one string-array config field.
 *
 * @param value - Raw config value.
 * @param fieldName - Config field name.
 * @param configPath - Absolute config path used in warnings.
 * @returns Unique non-empty strings when valid.
 */
function normalizeStringArrayConfigValue(
  value: unknown,
  fieldName: string,
  configPath: string,
): string[] | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (!Array.isArray(value)) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Field "${fieldName}" must be an array of non-empty strings. Ignoring value.`,
    );
    return undefined;
  }

  const normalizedEntries = value
    .filter((entry): entry is string => typeof entry === 'string')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);

  if (normalizedEntries.length !== value.length) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Field "${fieldName}" must contain only non-empty strings. Dropping invalid entries.`,
    );
  }

  return normalizedEntries.length > 0
    ? [...new Set(normalizedEntries)]
    : undefined;
}

/**
 * Normalizes the per-file symbol ordering map.
 *
 * @param value - Raw config value.
 * @param configPath - Absolute config path used in warnings.
 * @returns Sanitized symbol-order map when valid.
 */
function normalizeSymbolOrderConfigValue(
  value: unknown,
  configPath: string,
): Record<string, string[]> | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (!isRecord(value)) {
    warnDirectoryDocsOrderConfig(
      configPath,
      'Field "symbolOrder" must be an object keyed by file name. Ignoring value.',
    );
    return undefined;
  }

  const normalizedSymbolOrderEntries = Object.entries(value)
    .map(([fileName, symbolNames]) => {
      const normalizedFileName = fileName.trim();
      if (normalizedFileName.length === 0) {
        warnDirectoryDocsOrderConfig(
          configPath,
          'Field "symbolOrder" contains an empty file key. Dropping entry.',
        );
        return undefined;
      }

      const normalizedSymbols = normalizeStringArrayConfigValue(
        symbolNames,
        `symbolOrder.${normalizedFileName}`,
        configPath,
      );
      if (!normalizedSymbols) {
        return undefined;
      }

      return [normalizedFileName, normalizedSymbols] as const;
    })
    .filter(
      (
        entry,
      ): entry is readonly [string, string[]] => entry !== undefined,
    );

  return normalizedSymbolOrderEntries.length > 0
    ? Object.fromEntries(normalizedSymbolOrderEntries)
    : undefined;
}

/**
 * Writes a scoped warning for one docs ordering config file.
 *
 * @param configPath - Absolute config path.
 * @param message - Warning message.
 * @returns Nothing.
 */
function warnDirectoryDocsOrderConfig(
  configPath: string,
  message: string,
): void {
  const relativeConfigPath = path
    .relative(WORKSPACE_ROOT_DIR, configPath)
    .replace(/\\/g, '/');
  console.warn(`[docs] ${relativeConfigPath}: ${message}`);
}

/**
 * Narrows unknown values to simple object records.
 *
 * @param value - Value to inspect.
 * @returns True when the value is a plain record-like object.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Extracts a readable message from unknown thrown values.
 *
 * @param error - Thrown value.
 * @returns Human-readable error text.
 */
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

/**
 * Writes a file only when its content changed.
 *
 * @param filePath - Output file path.
 * @param content - Desired file content.
 * @returns Nothing.
 */
async function writeFileIfChanged(
  filePath: string,
  content: string,
): Promise<void> {
  if (await fs.pathExists(filePath)) {
    const previousContent = await fs.readFile(filePath, 'utf8');
    if (previousContent === content) {
      return;
    }
  }

  await fs.writeFile(filePath, content, 'utf8');
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
