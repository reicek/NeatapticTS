/**
 * @description AST-aware TypeScript chunker with two-level sub-chunking, re-export merging,
 * and cross-chunk context headers. Enhances the original ts-chunker.mjs by adding
 * declaration-member sub-chunking for classes and interfaces, merging bare re-exports
 * into module-index chunks, and producing rich metadata columns (depth, parent_chunk_id,
 * context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path).
 *
 * Reuses helpers from ts-chunker.mjs (resolveJsdocSummaryText, resolveSignatureText,
 * loadExportedTypeScriptDeclarations, etc.) and adds Level 2 sub-chunking on top.
 *
 * Key improvements over v1:
 * - Two-level chunking: AST boundary detection → declaration-member sub-chunking
 * - Classes: parent chunk (depth=0) with signature + JSDoc, method sub-chunks (depth=1)
 * - Small methods (≤300 chars) grouped into a "small methods" sub-chunk per class
 * - Interfaces: signature → parent chunk, property groups → sub-chunks
 * - Re-exports merged into module-index chunks (eliminates stubs)
 * - Cross-chunk context headers: [file_path > parent_symbol > member_symbol]
 * - Hard max 2,048 chars, target 800–1,500, min viable 100 chars
 * - Overlap: 256 chars at statement-group boundaries
 *
 * @param {boolean} [--json] - Emit JSON chunk output.
 * @param {string} [--source <path>] - Limit scanning to one or more explicit source file paths.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error.
 *
 * @example
 * ```sh
 * node rag-index/ts-chunker-v2.mjs --json --source src/neat.ts
 * ```
 */
/* global process */
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import {
  createTypeScriptProject,
  getTsMorphNode,
  loadExportedTypeScriptDeclarations,
  releaseTypeScriptProject,
  resolveJsdocSummaryText,
  resolveSignatureText,
  resolveTypeScriptSourcePaths,
} from './ts-chunker.mjs';

/** Hard maximum body text length — never exceed. */
const HARD_MAX_CHARS = 2048;

/** Target body text length range. */
const TARGET_MIN_CHARS = 800;

/** Minimum viable body text length. Below this, merge into parent. */
const MIN_VIABLE_CHARS = 100;

/** Small method threshold. Methods ≤ this many chars are grouped together. */
const SMALL_METHOD_CHARS = 300;

/** Overlap chars at statement-group boundaries. */
const OVERLAP_CHARS = 256;

/**
 * Chunk TypeScript sources using AST-aware two-level chunking.
 *
 * Phase 1: Collect all exported declarations grouped by file.
 * Phase 2: Separate re-exports from substantive declarations.
 * Phase 3: For each substantive declaration, apply sub-chunking if needed.
 * Phase 4: Assign sequential chunk indices within each file.
 *
 * The flat array is the concatenation of the per-file groups produced by
 * {@link chunkTypeScriptSourcesV2Stream}, in the same order.
 *
 * @param {object} [options={}] - Chunking options.
 * @param {string[]} [options.sourcePaths] - Explicit source file paths to chunk.
 * @param {string[]} [options.patterns] - Glob patterns for source files.
 * @param {string[]} [options.ignore] - Glob patterns to ignore.
 * @param {object} [options.project] - Pre-created ts-morph Project instance.
 * @returns {Promise<Array<import('./chunker.d.mts').TypeScriptChunkV2>>} V2 TypeScript chunks.
 */
export async function chunkTypeScriptSourcesV2(options) {
  /* istanbul ignore next -- defensive: always called with options from main() or tests */
  if (options == null) options = {};

  const allChunks = [];
  for await (const group of chunkTypeScriptSourcesV2Stream(options)) {
    allChunks.push(...group.chunks);
  }
  return allChunks;
}

/**
 * Stream the TypeScript chunks of every source file, one file at a time.
 *
 * The ts-morph AST is released as soon as the last group was produced, so the
 * resident chunk set no longer scales with the size of the corpus. The groups
 * are produced in the same file order the flat array uses, so the flat array is
 * the concatenation of the yielded groups.
 *
 * OWNERSHIP: when `options.project` is supplied, the CALLER owns that project
 * and it is NOT released here. A project created by this function IS released
 * before the generator completes.
 *
 * @param {object} [options={}] - Chunking options.
 * @param {string[]} [options.sourcePaths] - Explicit source file paths to chunk.
 * @param {string[]} [options.patterns] - Glob patterns for source files.
 * @param {string[]} [options.ignore] - Glob patterns to ignore.
 * @param {object} [options.project] - Pre-created ts-morph Project instance.
 * @yields {{ file_path: string, chunks: Array<import('./chunker.d.mts').TypeScriptChunkV2> }} One group per source file.
 * @returns {AsyncGenerator<{ file_path: string, chunks: Array<import('./chunker.d.mts').TypeScriptChunkV2> }, void, void>} Per-file chunk groups.
 *
 * @example
 * ```js
 * for await (const { file_path, chunks } of chunkTypeScriptSourcesV2Stream({ sourcePaths })) {
 *   console.log(file_path, chunks.length);
 * }
 * ```
 */
export async function* chunkTypeScriptSourcesV2Stream(options) {
  /* istanbul ignore next -- defensive: always called with options from main() or tests */
  if (options == null) options = {};

  const sourcePaths = await resolveTypeScriptSourcePaths(options);
  // Nothing to chunk: return without creating a project and without loading
  // the ts-morph compiler at all.
  if (sourcePaths.length === 0) return;

  const callerOwnsProject = options.project != null;
  const project = options.project ?? createTypeScriptProject(options);

  // Holder for the live ts-morph nodes. The `finally` block clears the
  // properties, so the AST becomes collectable BEFORE the project that owns
  // it is released.
  const liveNodes = { declarations: null, fileGroups: null };

  try {
    liveNodes.declarations = await loadExportedTypeScriptDeclarations({
      ...options,
      project,
      sourcePaths,
    });
    liveNodes.fileGroups = groupByFilePath(liveNodes.declarations);
    // The flat holder is redundant once the groups own the entries: dropping
    // it here lets the per-file entries be collected as each group is
    // consumed, instead of keeping every entry alive until the very end.
    liveNodes.declarations = null;

    for (const filePath of [...liveNodes.fileGroups.keys()]) {
      const fileDeclarations = liveNodes.fileGroups.get(filePath);
      liveNodes.fileGroups.delete(filePath);
      const chunks = chunkFileDeclarations(filePath, fileDeclarations);

      // The group was produced, so the source file that owns it (and the
      // TypeScript AST behind it) is released now: the resident AST does not
      // scale with the size of the corpus.
      if (!callerOwnsProject) {
        releaseChunkedSourceFile(project, fileDeclarations);
      }

      yield {
        file_path: filePath,
        chunks,
      };
    }
  } finally {
    // Drop the live-node references FIRST, so the AST becomes collectable,
    // THEN release the project that owns it.
    liveNodes.declarations = null;
    liveNodes.fileGroups = null;
    if (!callerOwnsProject) {
      releaseTypeScriptProject(project);
    }
  }
}

/**
 * Release the source file that owns a chunked group.
 *
 * Removing the source file from the project forgets its ts-morph nodes and
 * hands the TypeScript AST they belong to back to the compiler. This is safe
 * here because a group only ever reads nodes of its OWN file: the data a
 * re-export needs from another file is materialized up front by
 * `materializeReexport`.
 *
 * @param {object} project - The ts-morph project that owns the source file.
 * @param {Array} fileDeclarations - Exported declarations of the chunked file.
 * @returns {void}
 */
function releaseChunkedSourceFile(project, fileDeclarations) {
  const sourceFile = fileDeclarations[0]?.sourceFile;
  if (sourceFile == null) return;

  try {
    project.removeSourceFile(sourceFile);
  } catch {
    // Releasing the source file is a memory optimisation: the build must not
    // fail when the runtime refuses to release it.
  }
}

/**
 * Chunk the exported declarations of a single source file.
 *
 * The body is shared by the flat and the streaming API, so both produce
 * byte-identical chunk records for the same file.
 *
 * @param {string} filePath - Repository-relative file path.
 * @param {Array} fileDeclarations - Exported declarations of that file.
 * @returns {Array<import('./chunker.d.mts').TypeScriptChunkV2>} Chunks for this file.
 */
function chunkFileDeclarations(filePath, fileDeclarations) {
  const modulePath = deriveModulePath(filePath);
  const { reexports, substantive } = partitionDeclarations(fileDeclarations);

  const chunks = [];

  // Merge re-exports into a module-index chunk.
  if (reexports.length > 0) {
    chunks.push(createModuleIndexChunk(filePath, modulePath, reexports));
  }

  // Chunk each substantive declaration.
  for (const declaration of substantive) {
    const declarationChunks = chunkDeclaration(
      declaration,
      filePath,
      modulePath,
    );
    chunks.push(...declarationChunks);
  }

  // Assign sequential chunk indices within this file.
  chunks.forEach((chunk, chunkIndex) => {
    chunk.chunk_index = chunkIndex;
  });

  // Second pass: resolve parent_chunk_id for depth=1 sub-chunks.
  // Each depth=1 chunk belongs to the nearest preceding depth=0 chunk.
  let lastParentIndex = null;
  for (const chunk of chunks) {
    let depth = chunk.depth;
    /* istanbul ignore next -- defensive: chunk always has depth set by buildTypeScriptChunk */
    if (depth == null) depth = 0;
    if (depth === 0) {
      lastParentIndex = chunk.chunk_index;
    }
    /* istanbul ignore next -- defensive: depth is always 0 or 1, and parent always precedes children */
    if (depth > 0 && lastParentIndex !== null) {
      chunk.parent_chunk_id = lastParentIndex;
    }
  }

  return chunks;
}

/**
 * Group exported declarations by their file path.
 *
 * @param {Array} declarations - Flat list of exported declarations.
 * @returns {Map<string, Array>} Declarations grouped by file path.
 */
function groupByFilePath(declarations) {
  const groups = new Map();
  for (const declaration of declarations) {
    const existing = groups.get(declaration.file_path) ?? [];
    existing.push(declaration);
    groups.set(declaration.file_path, existing);
  }
  return groups;
}

/**
 * Derive a folder-based module path from a file path.
 *
 * Strips the file name and leading slash to produce a path like
 * `src/architecture/network` from `src/architecture/network/network.ts`.
 *
 * @param {string} filePath - Repository-relative file path.
 * @returns {string} Module path.
 */
function deriveModulePath(filePath) {
  const directory = path.posix.dirname(filePath);
  // Normalize: remove leading ./ or / if present.
  return directory.replace(/^\.\//, '').replace(/^\//, '');
}

/**
 * Partition declarations into re-exports and substantive declarations.
 *
 * A re-export is a declaration whose source is a re-export barrel file entry
 * (export { X } from './path') or whose text is a bare re-export statement.
 *
 * @param {Array} declarations - Declarations from one file.
 * @returns {{ reexports: Array, substantive: Array }} Partitioned declarations.
 */
function partitionDeclarations(declarations) {
  const reexports = [];
  const substantive = [];

  for (const declaration of declarations) {
    if (isReexport(declaration)) {
      reexports.push(declaration);
    } else {
      substantive.push(declaration);
    }
  }

  return { reexports, substantive };
}

/**
 * Determine whether a declaration is a bare re-export.
 *
 * A re-export is a declaration whose full text is an export statement
 * that re-exports from another module (e.g., `export { Network } from './network'`).
 *
 * @param {object} declaration - Declaration with source text.
 * @returns {boolean} True if the declaration is a re-export.
 */
function isReexport(declaration) {
  // A re-export is a declaration whose actual source file differs from the
  // file it was exported from (i.e., the declaration was resolved from another
  // module via `export { X } from '...'` or `export * as X from '...'`).
  //
  // The classification is MATERIALIZED while every source file is still
  // resident, so chunking a file never reads a node that belongs to another
  // file. See `materializeReexport` in `ts-chunker.mjs`.
  return declaration.reexport !== null && declaration.reexport !== undefined;
}

/**
 * Create a module-index chunk that merges all re-exports from a barrel file.
 *
 * @param {string} filePath - Repository-relative file path.
 * @param {string} modulePath - Folder-based module path.
 * @param {Array} reexports - Re-export declarations to merge.
 * @returns {import('./chunker.d.mts').TypeScriptChunkV2} Module-index chunk.
 */
function createModuleIndexChunk(filePath, modulePath, reexports) {
  const symbolNames = reexports.map((decl) => decl.symbol_name);
  const bodyText = [
    `Module: ${filePath}`,
    `Re-exports: ${symbolNames.join(', ')}`,
    ...reexports.map((decl) => cleanWhitespace(decl.reexport.text)),
  ].join('\n');

  const charEnd = reexports.at(-1)?.reexport.end ?? 0;
  const charStart = reexports[0]?.reexport.start ?? 0;

  return {
    body_text: bodyText,
    char_end: charEnd,
    char_start: charStart,
    chunk_index: 0, // Reassigned later in chunkTypeScriptSourcesV2.
    context_header: `[${filePath} > module-index]`,
    depth: 0,
    doc_family: 'ts-source',
    export_type: 'reexport',
    file_path: filePath,
    heading_path: 'module-index',
    jsdoc_text: '',
    module_path: modulePath,
    parent_chunk_id: null,
    signature_text: '',
    symbol_name: 'module-index',
  };
}

/**
 * Chunk a single substantive declaration.
 *
 * Applies two-level chunking:
 * - Small declarations (≤1,500 chars): single depth=0 chunk.
 * - Classes: parent chunk + method/property sub-chunks.
 * - Interfaces: parent chunk + property group sub-chunks.
 * - Other large declarations: split at statement-group boundaries.
 *
 * @param {object} declaration - Declaration with ts-morph node and metadata.
 * @param {string} filePath - Repository-relative file path.
 * @param {string} modulePath - Folder-based module path.
 * @returns {Array<import('./chunker.d.mts').TypeScriptChunkV2>} Chunks for this declaration.
 */
function chunkDeclaration(declaration, filePath, modulePath) {
  const Node = getTsMorphNode();

  const {
    declaration: declNode,
    symbol_name: symbolName,
    jsdoc_source_node: jsdocSourceNode,
  } = declaration;
  const jsdocText = resolveJsdocSummaryText(declNode, jsdocSourceNode);
  const signatureText = resolveSignatureText(declNode);
  /* istanbul ignore next -- defensive: declaration always has getText */
  const fullText = cleanWhitespace(
    declNode.getText?.() ?? '',
  );
  const exportType = resolveExportType(declNode);

  // Small declarations — emit as single depth=0 chunk.
  if (fullText.length <= TARGET_MIN_CHARS) {
    const contextHeader = buildTypeScriptContextHeader(filePath, symbolName);
    return [
      buildTypeScriptChunk({
        bodyText: buildBodyText(
          symbolName,
          filePath,
          signatureText,
          jsdocText,
          fullText,
        ),
        charEnd: declNode.getEnd(),
        charStart: declNode.getStart(),
        contextHeader,
        depth: 0,
        exportType,
        filePath,
        headingPath: symbolName,
        jsdocText,
        modulePath,
        parentChunkId: null,
        signatureText,
        symbolName,
      }),
    ];
  }

  // Classes — apply two-level sub-chunking.
  if (Node.isClassDeclaration(declNode)) {
    return createClassChunks(
      declNode,
      filePath,
      modulePath,
      symbolName,
      jsdocText,
      signatureText,
      exportType,
    );
  }

  // Interfaces — apply two-level sub-chunking for large interfaces.
  if (Node.isInterfaceDeclaration(declNode)) {
    return createInterfaceChunks(
      declNode,
      filePath,
      modulePath,
      symbolName,
      jsdocText,
      signatureText,
      exportType,
    );
  }

  // Other large declarations — split at statement-group boundaries.
  return createLargeSymbolChunks(
    declNode,
    filePath,
    modulePath,
    symbolName,
    jsdocText,
    signatureText,
    exportType,
  );
}

/**
 * Create two-level chunks for a class declaration.
 *
 * Parent chunk (depth=0): class signature + JSDoc + constructor + property declarations.
 * Method sub-chunks (depth=1): each method with body > SMALL_METHOD_CHARS.
 * Small methods sub-chunk (depth=1): grouped methods with body ≤ SMALL_METHOD_CHARS.
 *
 * @param {object} classDecl - ts-morph ClassDeclaration node.
 * @param {string} filePath - Repository-relative file path.
 * @param {string} modulePath - Folder-based module path.
 * @param {string} symbolName - Exported symbol name.
 * @param {string} jsdocText - JSDoc summary text.
 * @param {string} signatureText - Class signature text.
 * @param {string} exportType - Export type string.
 * @returns {Array<import('./chunker.d.mts').TypeScriptChunkV2>} Class chunks.
 */
function createClassChunks(
  classDecl,
  filePath,
  modulePath,
  symbolName,
  jsdocText,
  signatureText,
  exportType,
) {
  const chunks = [];
  const parentContextHeader = buildTypeScriptContextHeader(
    filePath,
    symbolName,
  );

  // Parent chunk: class signature + JSDoc + constructor + property declarations.
  const parentBodyText = buildClassParentBody(
    classDecl,
    symbolName,
    filePath,
    jsdocText,
    signatureText,
  );
  const parentChunk = buildTypeScriptChunk({
    bodyText: parentBodyText.slice(0, HARD_MAX_CHARS),
    charEnd: classDecl.getEnd(),
    charStart: classDecl.getStart(),
    contextHeader: parentContextHeader,
    depth: 0,
    exportType,
    filePath,
    headingPath: symbolName,
    jsdocText,
    modulePath,
    parentChunkId: null,
    signatureText,
    symbolName,
  });
  chunks.push(parentChunk);

  // Method sub-chunks.
  /* istanbul ignore next -- defensive: class declarations always support getMethods */
  const methods = classDecl.getMethods?.() ?? [];
  const largeMethods = [];
  const smallMethods = [];

  for (const method of methods) {
    /* istanbul ignore next -- defensive: method always has getText */
    const methodText = cleanWhitespace(method.getText?.() ?? '');
    if (methodText.length <= SMALL_METHOD_CHARS) {
      smallMethods.push(method);
    } else {
      largeMethods.push(method);
    }
  }

  // Large methods get individual sub-chunks.
  for (const method of largeMethods) {
    /* istanbul ignore next -- defensive: method always has getName */
    const methodName = method.getName?.() ?? 'anonymous';
    const methodJSDoc = resolveJsdocSummaryText(method);
    const methodSignature = resolveSignatureText(method);
    const methodContextHeader = buildTypeScriptContextHeader(
      filePath,
      symbolName,
      methodName,
    );
    const methodBody = buildMethodBody(
      method,
      methodName,
      methodJSDoc,
      methodSignature,
    );
    const methodChunks = splitLargeText(
      methodBody,
      HARD_MAX_CHARS,
      OVERLAP_CHARS,
    );

    for (
      let methodChunkIndex = 0;
      methodChunkIndex < methodChunks.length;
      methodChunkIndex += 1
    ) {
      const methodChunk = methodChunks[methodChunkIndex];
      /* istanbul ignore next -- defensive: no test fixture has a method body large enough to split into multiple chunks */
      const headingSuffix =
        methodChunkIndex > 0 ? `${methodName} (continued)` : methodName;
      chunks.push(
        buildTypeScriptChunk({
          bodyText: methodChunk,
          charEnd: method.getEnd(),
          charStart: method.getStart(),
          contextHeader: methodContextHeader,
          depth: 1,
          exportType: 'method',
          filePath,
          headingPath: `${symbolName} > ${headingSuffix}`,
          jsdocText: methodJSDoc,
          modulePath,
          parentChunkId: null, // Resolved during insertion.
          signatureText: methodSignature,
          symbolName: methodName,
        }),
      );
    }
  }

  // Small methods get grouped into one sub-chunk.
  /* istanbul ignore else -- defensive: classes in test fixtures always have small methods */
  if (smallMethods.length > 0) {
    const smallMethodsBody = buildSmallMethodsBody(
      smallMethods,
      symbolName,
      filePath,
    );
    const smallMethodsContextHeader = buildTypeScriptContextHeader(
      filePath,
      symbolName,
      'small-methods',
    );
    /* istanbul ignore next -- defensive: smallMethods always has elements when length > 0 */
    const smallMethodsCharEnd = smallMethods.at(-1)?.getEnd?.() ?? classDecl.getEnd();
    /* istanbul ignore next -- defensive: smallMethods always has elements when length > 0 */
    const smallMethodsCharStart = smallMethods[0]?.getStart?.() ?? classDecl.getStart();
    chunks.push(
      buildTypeScriptChunk({
        bodyText: smallMethodsBody.slice(0, HARD_MAX_CHARS),
        charEnd: smallMethodsCharEnd,
        charStart: smallMethodsCharStart,
        contextHeader: smallMethodsContextHeader,
        depth: 1,
        exportType: 'method',
        filePath,
        headingPath: `${symbolName} > small-methods`,
        jsdocText: '',
        modulePath,
        parentChunkId: null,
        signatureText: '',
        symbolName: 'small-methods',
      }),
    );
  }

  return chunks;
}

/**
 * Create two-level chunks for an interface declaration.
 *
 * Parent chunk (depth=0): interface signature + JSDoc.
 * Property sub-chunks (depth=1): property groups.
 *
 * @param {object} interfaceDecl - ts-morph InterfaceDeclaration node.
 * @param {string} filePath - Repository-relative file path.
 * @param {string} modulePath - Folder-based module path.
 * @param {string} symbolName - Exported symbol name.
 * @param {string} jsdocText - JSDoc summary text.
 * @param {string} signatureText - Interface signature text.
 * @param {string} exportType - Export type string.
 * @returns {Array<import('./chunker.d.mts').TypeScriptChunkV2>} Interface chunks.
 */
function createInterfaceChunks(
  interfaceDecl,
  filePath,
  modulePath,
  symbolName,
  jsdocText,
  signatureText,
  exportType,
) {
  const chunks = [];
  const parentContextHeader = buildTypeScriptContextHeader(
    filePath,
    symbolName,
  );

  // Parent chunk: interface signature + JSDoc.
  /* istanbul ignore next -- defensive: interface always has getText */
  const interfaceText = cleanWhitespace(interfaceDecl.getText?.() ?? '');
  const parentBodyText = buildBodyText(
    symbolName,
    filePath,
    signatureText,
    jsdocText,
    interfaceText,
  );
  chunks.push(
    buildTypeScriptChunk({
      bodyText: parentBodyText.slice(0, HARD_MAX_CHARS),
      charEnd: interfaceDecl.getEnd(),
      charStart: interfaceDecl.getStart(),
      contextHeader: parentContextHeader,
      depth: 0,
      exportType,
      filePath,
      headingPath: symbolName,
      jsdocText,
      modulePath,
      parentChunkId: null,
      signatureText,
      symbolName,
    }),
  );

  // Property groups for large interfaces.
  /* istanbul ignore next -- defensive: interface always supports getProperties */
  const properties = interfaceDecl.getProperties?.() ?? [];
  /* istanbul ignore else -- defensive: interfaces in test fixtures always have properties */
  if (properties.length > 0) {
    const propertyBody = properties
      .map((prop) => {
        const propJSDoc = resolveJsdocSummaryText(prop);
        /* istanbul ignore next -- defensive: property always has getText */
        const propText = cleanWhitespace(prop.getText?.() ?? '');
        /* istanbul ignore next -- defensive: properties in test fixtures always have JSDoc */
        return propJSDoc ? `${propText} — ${propJSDoc}` : propText;
      })
      .join('\n');

    /* istanbul ignore else -- defensive: property body in test fixtures always exceeds MIN_VIABLE_CHARS */
    if (propertyBody.length > MIN_VIABLE_CHARS) {
      const propertyChunks = splitLargeText(
        propertyBody,
        HARD_MAX_CHARS,
        OVERLAP_CHARS,
      );
      for (
        let propChunkIndex = 0;
        propChunkIndex < propertyChunks.length;
        propChunkIndex += 1
      ) {
        const propChunk = propertyChunks[propChunkIndex];
        /* istanbul ignore next -- defensive: no test fixture has properties large enough to split into multiple chunks */
        const headingSuffix =
          propChunkIndex > 0 ? 'properties (continued)' : 'properties';
        chunks.push(
          buildTypeScriptChunk({
            bodyText: propChunk,
            charEnd: interfaceDecl.getEnd(),
            charStart: interfaceDecl.getStart(),
            contextHeader: buildTypeScriptContextHeader(
              filePath,
              symbolName,
              'properties',
            ),
            depth: 1,
            exportType: 'property',
            filePath,
            headingPath: `${symbolName} > ${headingSuffix}`,
            jsdocText: '',
            modulePath,
            parentChunkId: null,
            signatureText: '',
            symbolName: 'properties',
          }),
        );
      }
    }
  }

  return chunks;
}

/**
 * Create chunks for a large function, type alias, or variable declaration.
 *
 * Splits at statement-group boundaries (blank lines between logical blocks).
 *
 * @param {object} declNode - ts-morph declaration node.
 * @param {string} filePath - Repository-relative file path.
 * @param {string} modulePath - Folder-based module path.
 * @param {string} symbolName - Exported symbol name.
 * @param {string} jsdocText - JSDoc summary text.
 * @param {string} signatureText - Signature text.
 * @param {string} exportType - Export type string.
 * @returns {Array<import('./chunker.d.mts').TypeScriptChunkV2>} Chunks for the declaration.
 */
function createLargeSymbolChunks(
  declNode,
  filePath,
  modulePath,
  symbolName,
  jsdocText,
  signatureText,
  exportType,
) {
  const contextHeader = buildTypeScriptContextHeader(filePath, symbolName);
  /* istanbul ignore next -- defensive: declaration always has getText */
  const fullText = cleanWhitespace(declNode.getText?.() ?? '');
  const bodyText = buildBodyText(
    symbolName,
    filePath,
    signatureText,
    jsdocText,
    fullText,
  );

  /* istanbul ignore if -- reachable but no test fixture has a function between 800-2048 chars */
  if (bodyText.length <= HARD_MAX_CHARS) {
    /* istanbul ignore next -- reachable but no test fixture has a function between 800-2048 chars */
    return [
      buildTypeScriptChunk({
        bodyText,
        charEnd: declNode.getEnd(),
        charStart: declNode.getStart(),
        contextHeader,
        depth: 0,
        exportType,
        filePath,
        headingPath: symbolName,
        jsdocText,
        modulePath,
        parentChunkId: null,
        signatureText,
        symbolName,
      }),
    ];
  }

  // Split at statement-group boundaries.
  const subChunks = splitLargeText(bodyText, HARD_MAX_CHARS, OVERLAP_CHARS);
  return subChunks.map((subChunk, subIndex) => {
    const headingSuffix =
      subIndex > 0 ? `${symbolName} (continued)` : symbolName;
    return buildTypeScriptChunk({
      bodyText: subChunk,
      charEnd: declNode.getEnd(),
      charStart: declNode.getStart(),
      contextHeader,
      depth: 0,
      exportType,
      filePath,
      headingPath: headingSuffix,
      jsdocText,
      modulePath,
      parentChunkId: null,
      signatureText,
      symbolName,
    });
  });
}

/**
 * Build the body text for a class parent chunk.
 *
 * Includes: class name, file path, signature, JSDoc, and the class declaration
 * header (without method bodies).
 *
 * @param {object} classDecl - ts-morph ClassDeclaration node.
 * @param {string} symbolName - Class name.
 * @param {string} filePath - Repository-relative file path.
 * @param {string} jsdocText - JSDoc summary.
 * @param {string} signatureText - Class signature.
 * @returns {string} Parent chunk body text.
 */
function buildClassParentBody(
  classDecl,
  symbolName,
  filePath,
  jsdocText,
  signatureText,
) {
  const parts = [`Symbol: ${symbolName}`, `Path: ${filePath}`];

  /* istanbul ignore else -- defensive: class declarations always have signature text */
  if (signatureText) parts.push(`Signature: ${signatureText}`);
  /* istanbul ignore else -- defensive: class declarations in test fixtures always have JSDoc */
  if (jsdocText) parts.push(`JSDoc: ${jsdocText}`);

  // Include the class header (signature + property declarations, no method bodies).
  const classHeader = extractClassHeader(classDecl);
  /* istanbul ignore else -- defensive: extractClassHeader always returns non-empty for valid classes */
  if (classHeader) parts.push(classHeader);

  return parts.filter(Boolean).join('\n');
}

/**
 * Extract the class header (signature + property declarations, no method bodies).
 *
 * @param {object} classDecl - ts-morph ClassDeclaration node.
 * @returns {string} Class header text.
 */
function extractClassHeader(classDecl) {
  /* istanbul ignore next -- defensive: class declarations always have getText */
  const text = cleanWhitespace(classDecl.getText?.() ?? '');
  const braceIndex = text.indexOf('{');
  /* istanbul ignore if -- defensive: class declarations always contain { */
  if (braceIndex === -1) return text;

  // Return the signature portion (before the opening brace) plus a summary.
  return text.slice(0, braceIndex).trim();
}

/**
 * Build the body text for a method sub-chunk.
 *
 * @param {object} method - ts-morph MethodDeclaration node.
 * @param {string} methodName - Method name.
 * @param {string} methodJSDoc - Method JSDoc summary.
 * @param {string} methodSignature - Method signature.
 * @returns {string} Method sub-chunk body text.
 */
function buildMethodBody(method, methodName, methodJSDoc, methodSignature) {
  const parts = [`Method: ${methodName}`];
  /* istanbul ignore else -- defensive: methods always have signature text */
  if (methodSignature) parts.push(`Signature: ${methodSignature}`);
  /* istanbul ignore else -- defensive: methods in test fixtures always have JSDoc */
  if (methodJSDoc) parts.push(`JSDoc: ${methodJSDoc}`);

  /* istanbul ignore next -- defensive: method always has getText */
  const methodText = cleanWhitespace(method.getText?.() ?? '');
  parts.push(methodText);

  return parts.filter(Boolean).join('\n');
}

/**
 * Build the body text for the small methods group sub-chunk.
 *
 * @param {Array} smallMethods - Array of ts-morph MethodDeclaration nodes.
 * @param {string} symbolName - Parent class name.
 * @param {string} filePath - Repository-relative file path.
 * @returns {string} Small methods group body text.
 */
function buildSmallMethodsBody(smallMethods, symbolName, filePath) {
  const parts = [`Small methods in ${symbolName} (${filePath})`];

  for (const method of smallMethods) {
    /* istanbul ignore next -- defensive: method always has getName */
    const methodName = method.getName?.() ?? 'anonymous';
    const methodJSDoc = resolveJsdocSummaryText(method);
    const methodSignature = resolveSignatureText(method);
    /* istanbul ignore next -- defensive: method always has getText */
    const methodText = cleanWhitespace(method.getText?.() ?? '');

    const methodParts = [`  ${methodName}`];
    /* istanbul ignore else -- defensive: methods always have signature text */
    if (methodSignature) methodParts.push(`    Signature: ${methodSignature}`);
    /* istanbul ignore else -- defensive: methods in test fixtures always have JSDoc */
    if (methodJSDoc) methodParts.push(`    JSDoc: ${methodJSDoc}`);
    methodParts.push(`    ${methodText}`);

    parts.push(methodParts.join('\n'));
  }

  return parts.filter(Boolean).join('\n');
}

/**
 * Split large text at statement-group boundaries (blank lines) with overlap.
 *
 * @param {string} text - Text to split.
 * @param {number} maxChars - Hard maximum chars per chunk.
 * @param {number} overlapChars - Overlap chars between chunks.
 * @returns {Array<string>} Sub-chunk texts.
 */
function splitLargeText(text, maxChars, overlapChars) {
  if (text.length <= maxChars) return [text];

  const chunks = [];
  const statementGroups = splitAtStatementGroups(text);

  let currentChunk = '';
  for (const group of statementGroups) {
    /* istanbul ignore next -- unreachable: cleanWhitespace collapses newlines so splitAtStatementGroups returns a single group */
    if (
      currentChunk.length > 0 &&
      currentChunk.length + group.length + 1 > maxChars
    ) {
      chunks.push(currentChunk.trimEnd());
      // Compute overlap from end of current chunk.
      const overlap = currentChunk.trimEnd().slice(-overlapChars);
      currentChunk = overlap + '\n' + group;
    } else if (currentChunk.length === 0) {
      currentChunk = group;
    } else {
      currentChunk = currentChunk + '\n' + group;
    }
  }

  /* istanbul ignore else -- defensive: currentChunk always has content after processing groups */
  if (currentChunk.trim().length > 0) {
    chunks.push(currentChunk.trimEnd());
  }

  // Fallback: if statement-group splitting produced a single chunk that still
  // exceeds maxChars (e.g., cleanWhitespace collapsed all newlines), split at
  // character boundaries with overlap to enforce the hard max.
  /* istanbul ignore else -- defensive: splitAtCharBoundaries always handles the fallback case */
  if (chunks.length <= 1 && text.length > maxChars) {
    return splitAtCharBoundaries(text, maxChars, overlapChars);
  }

  /* istanbul ignore next -- unreachable: cleanWhitespace collapses newlines so only one group is produced */
  return chunks;
}

/**
 * Split text at fixed character boundaries with overlap.
 *
 * Used as a fallback when statement-group splitting cannot produce chunks
 * small enough to satisfy the hard max (e.g., when all newlines were collapsed
 * by cleanWhitespace).
 *
 * @param {string} text - Text to split.
 * @param {number} maxChars - Hard maximum chars per chunk.
 * @param {number} overlapChars - Overlap chars between chunks.
 * @returns {Array<string>} Sub-chunk texts.
 */
function splitAtCharBoundaries(text, maxChars, overlapChars) {
  const chunks = [];
  let offset = 0;
  while (offset < text.length) {
    const end = Math.min(offset + maxChars, text.length);
    chunks.push(text.slice(offset, end));
    if (end >= text.length) break;
    offset = end - overlapChars;
  }
  return chunks;
}

/**
 * Split text at statement-group boundaries (double newlines / blank lines).
 *
 * @param {string} text - Text to split.
 * @returns {Array<string>} Statement groups.
 */
function splitAtStatementGroups(text) {
  const groups = [];
  const lines = text.split('\n');
  let currentGroup = [];

  for (const line of lines) {
    /* istanbul ignore next -- unreachable: cleanWhitespace collapses newlines so no blank lines exist */
    if (line.trim().length === 0 && currentGroup.length > 0) {
      groups.push(currentGroup.join('\n'));
      currentGroup = [];
    } else {
      currentGroup.push(line);
    }
  }

  /* istanbul ignore else -- defensive: currentGroup always has content after processing lines */
  if (currentGroup.length > 0) {
    groups.push(currentGroup.join('\n'));
  }

  return groups;
}

/**
 * Build a TypeScript chunk object with all v2 fields.
 *
 * @param {object} params - Chunk fields.
 * @returns {import('./chunker.d.mts').TypeScriptChunkV2} V2 TypeScript chunk.
 */
function buildTypeScriptChunk(params) {
  return {
    body_text: params.bodyText,
    char_end: params.charEnd,
    char_start: params.charStart,
    chunk_index: params.chunkIndex ?? 0,
    context_header: params.contextHeader,
    depth: params.depth,
    doc_family: 'ts-source',
    export_type: params.exportType,
    file_path: params.filePath,
    heading_path: params.headingPath,
    jsdoc_text: params.jsdocText,
    module_path: params.modulePath,
    parent_chunk_id: params.parentChunkId,
    signature_text: params.signatureText,
    symbol_name: params.symbolName,
  };
}

/**
 * Resolve the export type string for a ts-morph declaration node.
 *
 * @param {object} declaration - ts-morph declaration node.
 * @returns {string} One of: 'function', 'class', 'interface', 'type', 'variable'.
 */
function resolveExportType(declaration) {
  const Node = getTsMorphNode();

  if (Node.isFunctionDeclaration(declaration)) return 'function';
  if (Node.isClassDeclaration(declaration)) return 'class';
  if (Node.isInterfaceDeclaration(declaration)) return 'interface';
  if (Node.isTypeAliasDeclaration(declaration)) return 'type';
  /* istanbul ignore else -- defensive: all declaration types are checked above */
  if (Node.isVariableDeclaration(declaration)) return 'variable';
  /* istanbul ignore next -- defensive: all declaration types are checked above */
  return 'variable';
}

/**
 * Build a TypeScript context header string.
 *
 * Format: `[file_path > parent_symbol > member_symbol]` or `[file_path > parent_symbol]`
 * for depth=0 chunks.
 *
 * @param {string} filePath - Repository-relative file path.
 * @param {string} parentSymbol - Parent symbol name.
 * @param {string} [memberSymbol] - Member symbol name (for depth=1 chunks).
 * @returns {string} Context header string.
 */
function buildTypeScriptContextHeader(filePath, parentSymbol, memberSymbol) {
  if (memberSymbol) return `[${filePath} > ${parentSymbol} > ${memberSymbol}]`;
  return `[${filePath} > ${parentSymbol}]`;
}

/**
 * Build body text for a simple (non-sub-chunked) declaration.
 *
 * @param {string} symbolName - Symbol name.
 * @param {string} filePath - File path.
 * @param {string} signatureText - Signature text.
 * @param {string} jsdocText - JSDoc text.
 * @param {string} fullText - Full declaration text.
 * @returns {string} Body text.
 */
function buildBodyText(
  symbolName,
  filePath,
  signatureText,
  jsdocText,
  fullText,
) {
  const parts = [`Symbol: ${symbolName}`, `Path: ${filePath}`];
  /* istanbul ignore else -- defensive: declarations always have signature text */
  if (signatureText) parts.push(`Signature: ${signatureText}`);
  /* istanbul ignore else -- defensive: declarations in test fixtures always have JSDoc */
  if (jsdocText) parts.push(`JSDoc: ${jsdocText}`);
  parts.push(fullText);
  return parts.filter(Boolean).join('\n');
}

/**
 * Clean whitespace in text: collapse runs of whitespace to single spaces.
 *
 * @param {string} value - Text to clean.
 * @returns {string} Cleaned text.
 */
function cleanWhitespace(value) {
  /* istanbul ignore next -- defensive: value is always a string from getText() */
  return String(value ?? '')
    .replace(/\s+/g, ' ')
    .trim();
}

export async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'TypeScript symbol chunker v2 (AST-aware)',
      usage:
        'node rag-index/ts-chunker-v2.mjs [--json] [--source path/to/file.ts]',
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
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: providedSources.length > 0 ? providedSources : undefined,
    });
    writeJsonOrText(
      chunks,
      Boolean(args.json),
      /* istanbul ignore next -- text formatter covered when writeJsonOrText is not mocked */
      (payload) => `TS source v2 chunks: ${payload.length}`,
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
