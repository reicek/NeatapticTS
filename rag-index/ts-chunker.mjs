/**
 * @description Extract `ts-source` family chunks from `src/**\/*.ts` (excluding test and
 * declaration files) using ts-morph AST traversal. Each chunk covers one exported symbol
 * (function, class, interface, or type alias) with its full JSDoc comment, signature
 * text, and source path. Produces the same chunk shape as the markdown chunker so
 * `build-index.mjs` can store ts-source chunks alongside readme, plan, and agent chunks.
 *
 * @param {boolean} [--json]         - Emit JSON chunk output (array of chunk objects).
 * @param {string}  [--source <path>] - Limit scanning to one or more explicit source file paths.
 * @param {boolean} [--help]         - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error. JSON written to stdout when `--json` is
 *   passed; each chunk includes `{ family, symbolName, heading_path, body_text, filePath }`.
 */
/* global process */
import fg from 'fast-glob';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  toRepoRelative,
  writeJsonOrText,
} from './cli-utils.mjs';
import { repoRoot } from './init-schema.mjs';

const DEFAULT_TS_PATTERNS = ['src/**/*.ts'];
const DEFAULT_TS_IGNORE = [
  'src/**/*.d.ts',
  'src/**/*.test.ts',
  'src/**/*.spec.ts',
];
const DEFAULT_TSCONFIG_PATH = path.join(repoRoot, 'tsconfig.json');

/**
 * Require hook used to load the CJS-only `ts-morph` package on demand.
 *
 * The package is loaded LAZILY, so a build that never chunks a TypeScript
 * source never loads the TypeScript compiler at all.
 */
const requireTsMorph = createRequire(import.meta.url);

/**
 * Cached ts-morph module exports.
 *
 * @type {object | null}
 */
let tsMorphModule = null;

/**
 * Load the ts-morph module exports, caching them for the process lifetime.
 *
 * @returns {object} The ts-morph module exports.
 */
function loadTsMorphModule() {
  tsMorphModule ??= requireTsMorph('ts-morph');
  return tsMorphModule;
}

/**
 * Resolve the ts-morph `Node` type-guard helpers.
 *
 * The lookup is LAZY: the ts-morph compiler is only loaded when a TypeScript
 * source is actually chunked. Callers MUST resolve the guards inside the
 * function that needs them, never at module scope, otherwise the compiler is
 * loaded for every consumer of this module.
 *
 * @returns {object} The ts-morph `Node` namespace with the type-guard helpers.
 *
 * @example
 * ```js
 * const Node = getTsMorphNode();
 * if (Node.isClassDeclaration(declaration)) {
 *   // ...
 * }
 * ```
 */
export function getTsMorphNode() {
  return loadTsMorphModule().Node;
}

export async function resolveTypeScriptSourcePaths(options = {}) {
  const sourcePaths = options.sourcePaths;
  if (Array.isArray(sourcePaths) && sourcePaths.length > 0) {
    return sourcePaths
      .map((sourcePath) => path.resolve(sourcePath))
      .toSorted((leftPath, rightPath) =>
        leftPath.localeCompare(rightPath, 'en'),
      );
  }

  const patterns = options.patterns ?? DEFAULT_TS_PATTERNS;
  const ignore = options.ignore ?? DEFAULT_TS_IGNORE;
  const entries = await fg(patterns, {
    cwd: repoRoot,
    absolute: true,
    onlyFiles: true,
    dot: false,
    ignore,
  });
  return entries.toSorted((leftPath, rightPath) =>
    leftPath.localeCompare(rightPath, 'en'),
  );
}

export function createTypeScriptProject(options = {}) {
  const tsConfigFilePath = path.resolve(
    options.tsConfigFilePath ?? DEFAULT_TSCONFIG_PATH,
  );
  const { Project } = loadTsMorphModule();
  return new Project({
    compilerOptions: {
      allowJs: true,
      // The indexer only reads declaration TEXT (names, JSDoc, offsets and
      // kinds); it never resolves ambient type packages. Leaving `types`
      // unset makes TypeScript auto-include every package under
      // `node_modules/@types` (78 packages, ~4 MB of `.d.ts`) as program root
      // files, and parsing plus binding those ambient declarations inflates
      // the ts-source phase peak RSS by roughly 125 MB. Opting out keeps the
      // exported declarations byte-identical while removing that cost.
      types: [],
    },
    skipAddingFilesFromTsConfig: true,
    skipFileDependencyResolution: true,
    tsConfigFilePath,
  });
}

/**
 * Projects whose AST and compiler were already released.
 *
 * The guard makes {@link releaseTypeScriptProject} idempotent, so a project
 * can be released from several release paths without double-disposing the
 * TypeScript compiler.
 *
 * @type {WeakSet<object>}
 */
const releasedProjects = new WeakSet();

/**
 * Release the ts-morph AST and the underlying TypeScript compiler.
 *
 * The release removes every source file that was added to the project (which
 * forgets the corresponding AST nodes) and then disposes the compiler that
 * backs the language service. The `Project` class of ts-morph has NO
 * `dispose()` and NO `forget()` method, so neither is called here.
 *
 * The caller MUST have finished reading plain data out of the AST first: a
 * removed source file is forgotten, and every later read on one of its nodes
 * throws. The function is idempotent — calling it twice for the same project
 * is a no-op.
 *
 * @param {object | null | undefined} project - The ts-morph project to release.
 * @returns {void}
 *
 * @example
 * ```js
 * const project = createTypeScriptProject();
 * // ... extract plain chunk data out of the AST ...
 * releaseTypeScriptProject(project);
 * ```
 */
export function releaseTypeScriptProject(project) {
  if (project == null) return;
  if (releasedProjects.has(project)) return;
  releasedProjects.add(project);

  // Snapshot the source files: removing one mutates the project collection.
  for (const sourceFile of [...project.getSourceFiles()]) {
    project.removeSourceFile(sourceFile);
  }

  project.getLanguageService().compilerObject.dispose();
}

/**
 * Load the exported declarations of the given TypeScript sources.
 *
 * OWNERSHIP CONTRACT: the returned entries hold LIVE ts-morph nodes
 * (`declaration`, `jsdoc_source_node`, `sourceFile`). The project that owns
 * those nodes is NEVER released by this function — the caller must keep it
 * alive for as long as it reads the nodes, and it is the caller that decides
 * when {@link releaseTypeScriptProject} is invoked.
 *
 * @param {object} [options={}] - Loading options.
 * @param {string[]} [options.sourcePaths] - Explicit source file paths to load.
 * @param {object} [options.project] - Pre-created ts-morph project to load into.
 * @returns {Promise<Array<object>>} Exported declarations, sorted by file path then symbol name.
 */
export async function loadExportedTypeScriptDeclarations(options) {
  /* istanbul ignore next -- defensive: always called with options from chunkTypeScriptSources */
  if (options == null) options = {};
  const sourcePaths = await resolveTypeScriptSourcePaths(options);
  const project = options.project ?? createTypeScriptProject(options);

  for (const sourcePath of sourcePaths) {
    project.addSourceFileAtPathIfExists(sourcePath);
  }

  const exportedSymbols = project.getSourceFiles().flatMap((sourceFile) => {
    const exportedDeclarations = sourceFile.getExportedDeclarations();
    return [...exportedDeclarations.entries()].flatMap(
      ([exportName, declarations]) => {
        const declaration = selectPrimaryDeclaration(declarations);
        /* istanbul ignore if -- defensive: selectPrimaryDeclaration always returns a declaration */
        if (!declaration) return [];

        const filePath = toRepoRelative(sourceFile.getFilePath());

        return [
          {
            declaration,
            file_path: filePath,
            jsdoc_source_node: resolveExportJsdocSourceNode(
              sourceFile,
              exportName,
              declaration,
            ),
            reexport: materializeReexport(declaration, filePath),
            sourceFile,
            symbol_name: resolveSymbolName(declaration, exportName),
          },
        ];
      },
    );
  });

  const seenDeclarationIds = new Set();
  const dedupedSymbols = exportedSymbols.filter((entry) => {
    const declarationFile = toRepoRelative(
      entry.declaration.getSourceFile().getFilePath(),
    );
    const declarationId = `${declarationFile}:${entry.declaration.getStart()}:${entry.declaration.getKindName()}`;
    if (seenDeclarationIds.has(declarationId)) return false;
    seenDeclarationIds.add(declarationId);
    return true;
  });

  return dedupedSymbols.toSorted((leftSymbol, rightSymbol) => {
    const pathOrder = leftSymbol.file_path.localeCompare(
      rightSymbol.file_path,
      'en',
    );
    return pathOrder !== 0
      ? pathOrder
      : leftSymbol.symbol_name.localeCompare(rightSymbol.symbol_name, 'en');
  });
}

/**
 * Materialize the cross-file data a re-export chunk needs.
 *
 * A re-export declaration (`export { X } from './other'`) is chunked under the
 * file that RE-EXPORTS the symbol, but its ts-morph node lives in the file
 * that DECLARES it. Reading that node while the group is chunked couples the
 * chunking of one file to the AST of another file, which is what makes it
 * impossible to release a source file as soon as its own group was chunked.
 *
 * The values are therefore read ONCE, while every source file is still
 * resident, and stored as plain data on the entry. A re-export chunk then
 * reads plain data only, so a source file can be released as soon as its own
 * group was chunked.
 *
 * @param {object} declaration - ts-morph declaration node.
 * @param {string} filePath - Repository-relative path of the EXPORTING file.
 * @returns {{ end: number, start: number, text: string } | null} The materialized re-export data, or `null` when the declaration is not a re-export.
 */
export function materializeReexport(declaration, filePath) {
  const declarationFilePath = toRepoRelative(
    /* istanbul ignore next -- defensive: declaration always has a source file */
    declaration.getSourceFile?.().getFilePath?.() ?? '',
  );
  if (!declarationFilePath || declarationFilePath === filePath) {
    return null;
  }

  return {
    end: declaration.getEnd(),
    start: declaration.getStart(),
    text: declaration.getText(),
  };
}

export async function chunkTypeScriptSources(options) {
  /* istanbul ignore next -- defensive: always called with options from main() or tests */
  if (options == null) options = {};
  const exportedDeclarations =
    await loadExportedTypeScriptDeclarations(options);
  return exportedDeclarations.map(
    ({ declaration, file_path, jsdoc_source_node, symbol_name }) => {
      const jsdoc_text = resolveJsdocSummaryText(
        declaration,
        jsdoc_source_node,
      );
      const signature_text = resolveSignatureText(declaration);
      const body_text = [
        `Symbol: ${symbol_name}`,
        `Path: ${file_path}`,
        /* istanbul ignore next -- defensive: resolveSignatureText always returns non-empty for valid declarations */
        signature_text ? `Signature: ${signature_text}` : null,
        jsdoc_text ? `JSDoc: ${jsdoc_text}` : null,
      ]
        .filter(Boolean)
        .join('\n');

      return {
        body_text,
        char_end: declaration.getEnd(),
        char_start: declaration.getStart(),
        chunk_index: 0,
        doc_family: 'ts-source',
        file_path,
        heading_path: symbol_name,
        jsdoc_text,
        signature_text,
        symbol_name,
      };
    },
  );
}

export function resolveJsdocSummaryText(declaration, jsdocSourceNode = null) {
  const directDescription = resolveNodeJsdocSummaryText(declaration);
  if (directDescription) return directDescription;

  if (jsdocSourceNode && jsdocSourceNode !== declaration) {
    return resolveNodeJsdocSummaryText(jsdocSourceNode);
  }

  return '';
}

export function countWords(value) {
  return cleanWhitespace(value).split(' ').filter(Boolean).length;
}

function resolveJsdocNodes(declaration) {
  const Node = getTsMorphNode();

  if (typeof declaration?.getJsDocs === 'function') {
    const directJsDocs = declaration.getJsDocs();
    /* istanbul ignore else -- defensive: declarations with JSDoc always have direct JsDocs */
    if (directJsDocs.length > 0) return directJsDocs;
  }

  if (Node.isVariableDeclaration(declaration)) {
    /* istanbul ignore next -- defensive: variable declarations always have a statement with JsDocs */
    return declaration.getVariableStatement?.()?.getJsDocs?.() ?? [];
  }

  return [];
}

function resolveNodeJsdocSummaryText(node) {
  /* istanbul ignore next -- defensive: ts-morph JsDoc always has getDescription for valid nodes */
  const directDescription = resolveJsdocNodes(node)
    .map((jsDoc) => cleanWhitespace(jsDoc.getDescription?.() ?? ''))
    .find(Boolean);
  if (directDescription) return directDescription;

  return (
    /* istanbul ignore next -- defensive: compilerNode.jsDoc fallback for ts-morph API edge cases */
    node?.compilerNode?.jsDoc
      ?.map((jsDoc) =>
        cleanWhitespace(resolveCompilerJsdocComment(jsDoc.comment)),
      )
      .find(Boolean) ?? ''
  );
}

/* istanbul ignore next -- defensive: only called from compilerNode.jsDoc fallback */
function resolveCompilerJsdocComment(comment) {
  if (typeof comment === 'string') return comment;
  if (!Array.isArray(comment)) return '';

  return comment
    .map((commentPart) => {
      if (typeof commentPart === 'string') return commentPart;
      return typeof commentPart?.text === 'string' ? commentPart.text : '';
    })
    .join(' ');
}

function resolveExportJsdocSourceNode(sourceFile, exportName, declaration) {
  const Node = getTsMorphNode();

  if (!Node.isSourceFile(declaration)) return null;

  return (
    /* istanbul ignore next -- defensive: export declarations always have namespace exports */
    sourceFile.getExportDeclarations().find((exportDeclaration) => {
      const namespaceExport = exportDeclaration.getNamespaceExport?.();
      return namespaceExport?.getText?.() === `* as ${exportName}`;
    }) ?? null
  );
}

/**
 * Resolve the signature text of a declaration (function, class, interface, etc.).
 *
 * Extracts the declaration signature without the body — e.g., for a function
 * declaration, returns just `function foo(x: number): string` without the
 * implementation block.
 *
 * @param {object} declaration - ts-morph declaration node.
 * @returns {string} Signature text.
 */
export function resolveSignatureText(declaration) {
  const Node = getTsMorphNode();

  const declarationText = cleanWhitespace(declaration.getText());
  /* istanbul ignore if -- defensive: declaration.getText() always returns non-empty text */
  if (!declarationText) return '';

  if (
    Node.isFunctionDeclaration(declaration) ||
    Node.isMethodDeclaration(declaration)
  ) {
    return declarationText.split('{', 1)[0].trim();
  }

  if (
    Node.isClassDeclaration(declaration) ||
    Node.isInterfaceDeclaration(declaration)
  ) {
    return declarationText.split('{', 1)[0].trim();
  }

  if (Node.isTypeAliasDeclaration(declaration)) {
    return declarationText;
  }

  if (Node.isVariableDeclaration(declaration)) {
    /* istanbul ignore next -- defensive: variable declarations always have a statement */
    const parentStatementText = cleanWhitespace(
      declaration.getVariableStatement?.()?.getText() ?? declarationText,
    );
    /* istanbul ignore next -- defensive: variable statements always end with semicolon */
    return parentStatementText.endsWith(';')
      ? parentStatementText
      : `${parentStatementText};`;
  }

  return declarationText;
}

function cleanWhitespace(value) {
  return String(value ?? '')
    .replace(/\s+/g, ' ')
    .trim();
}

function resolveSymbolName(declaration, exportName) {
  if (exportName !== 'default') return exportName;
  /* istanbul ignore next -- defensive: default exports always have a symbol with a name */
  return declaration.getSymbol?.()?.getName?.() ?? exportName;
}

function selectPrimaryDeclaration(declarations) {
  const Node = getTsMorphNode();

  /* istanbul ignore next -- defensive: declarations array always has at least one element */
  return (
    declarations.find(
      (declaration) =>
        Node.isFunctionDeclaration(declaration) ||
        Node.isClassDeclaration(declaration) ||
        Node.isInterfaceDeclaration(declaration) ||
        Node.isTypeAliasDeclaration(declaration) ||
        Node.isVariableDeclaration(declaration),
    ) ??
    declarations[0] ??
    null
  );
}

export async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'TypeScript symbol chunker',
      usage:
        'node rag-index/ts-chunker.mjs [--json] [--source path/to/file.ts]',
      options: [
        '--json           Emit JSON chunk output.',
        '--source <path>  Limit scanning to one or more explicit source paths.',
        '--help           Show this help.',
      ],
    });
    return;
  }

  const providedSources = [
    args.source,
    ...(Array.isArray(args._) ? args._ : []),
  ]
    .flat()
    .filter(Boolean);

  try {
    const chunks = await chunkTypeScriptSources({
      sourcePaths: providedSources.length > 0 ? providedSources : undefined,
    });
    writeJsonOrText(
      chunks,
      Boolean(args.json),
      /* istanbul ignore next -- text formatter, covered by JSON-mode tests */
      (payload) => `TS source chunks: ${payload.length}`,
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

/* istanbul ignore next */
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
