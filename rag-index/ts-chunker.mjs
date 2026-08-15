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
import fg from 'fast-glob';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Node, Project } from 'ts-morph';
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
  return new Project({
    compilerOptions: {
      allowJs: true,
    },
    skipAddingFilesFromTsConfig: true,
    skipFileDependencyResolution: true,
    tsConfigFilePath,
  });
}

export async function loadExportedTypeScriptDeclarations(options = {}) {
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
        if (!declaration) return [];

        return [
          {
            declaration,
            file_path: toRepoRelative(sourceFile.getFilePath()),
            jsdoc_source_node: resolveExportJsdocSourceNode(
              sourceFile,
              exportName,
              declaration,
            ),
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
    const declarationId = `${declarationFile}:${entry.declaration.getStart()}`;
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

export async function chunkTypeScriptSources(options = {}) {
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
  if (typeof declaration?.getJsDocs === 'function') {
    const directJsDocs = declaration.getJsDocs();
    if (directJsDocs.length > 0) return directJsDocs;
  }

  if (Node.isVariableDeclaration(declaration)) {
    return declaration.getVariableStatement?.()?.getJsDocs?.() ?? [];
  }

  return [];
}

function resolveNodeJsdocSummaryText(node) {
  const directDescription = resolveJsdocNodes(node)
    .map((jsDoc) => cleanWhitespace(jsDoc.getDescription?.() ?? ''))
    .find(Boolean);
  if (directDescription) return directDescription;

  return (
    node?.compilerNode?.jsDoc
      ?.map((jsDoc) =>
        cleanWhitespace(resolveCompilerJsdocComment(jsDoc.comment)),
      )
      .find(Boolean) ?? ''
  );
}

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
  if (!Node.isSourceFile(declaration)) return null;

  return (
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
  const declarationText = cleanWhitespace(declaration.getText());
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
    const parentStatementText = cleanWhitespace(
      declaration.getVariableStatement?.()?.getText() ?? declarationText,
    );
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
  return declaration.getSymbol?.()?.getName?.() ?? exportName;
}

function selectPrimaryDeclaration(declarations) {
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

async function main() {
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
      (payload) => `TS source chunks: ${payload.length}`,
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
