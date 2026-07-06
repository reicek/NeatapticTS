/**
 * @module extract-code-entities
 * @description Extract code entities (module, class, function, interface, type-alias,
 * variable, error-class) and their relationships (imports, exports, owns, part-of,
 * depends-on, implements) from TypeScript source files using ts-morph AST analysis.
 *
 * Reuses the existing `loadExportedTypeScriptDeclarations` and `deriveModulePath`
 * infrastructure from `ts-chunker.mjs` and `ts-chunker-v2.mjs`.
 *
 * @param {boolean} [--json] - Emit JSON output of extracted entities and edges.
 * @param {string}  [--source <path>] - Limit scanning to one or more explicit source paths.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error.
 */
import fg from 'fast-glob';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Node, SyntaxKind } from 'ts-morph';

import {
  fail,
  parseCliArgs,
  printHelp,
  toRepoRelative,
  writeJsonOrText,
} from './cli-utils.mjs';
import { repoRoot } from './init-schema.mjs';
import {
  createTypeScriptProject,
  loadExportedTypeScriptDeclarations,
  resolveSignatureText,
  resolveTypeScriptSourcePaths,
} from './ts-chunker.mjs';

/** Maximum depends-on edges per symbol to avoid O(N²) explosion. */
const MAX_DEPENDS_ON_EDGES = 20;

/** Minimum method body length (chars) to produce a separate function entity. */
const MIN_METHOD_ENTITY_CHARS = 100;

/** Maximum characters of function body to scan for call expressions. */
const MAX_CALL_EXPRESSION_SCAN_CHARS = 500;

/** Code entity types for type preference scoring. */
const CODE_ENTITY_TYPES = new Set([
  'module',
  'class',
  'function',
  'interface',
  'type-alias',
  'variable',
  'error-class',
]);

/**
 * Extract code entities and relationships from TypeScript source files.
 *
 * Phase 1: Load all exported declarations using the existing ts-chunker infrastructure.
 * Phase 2: Extract module entities from directory paths.
 * Phase 3: Extract symbol entities from declarations.
 * Phase 4: Extract relationships (imports, exports, owns, depends-on, implements).
 * Phase 5: Compute inverse relationships (part-of).
 *
 * @param {object} [options={}] - Extraction options.
 * @param {string[]} [options.sourcePaths] - Explicit source file paths.
 * @param {string[]} [options.patterns] - Glob patterns.
 * @param {string[]} [options.ignore] - Glob ignore patterns.
 * @param {object} [options.project] - Pre-created ts-morph Project instance.
 * @returns {Promise<{ entities: Array, edges: Array }>} Extracted entities and edges.
 */
export async function extractCodeEntities(options = {}) {
  const sourcePaths = await resolveTypeScriptSourcePaths(options);
  const project = options.project ?? createTypeScriptProject(options);

  for (const sourcePath of sourcePaths) {
    project.addSourceFileAtPathIfExists(sourcePath);
  }

  const exportedDeclarations = await loadExportedTypeScriptDeclarations({
    ...options,
    project,
    sourcePaths,
  });

  const entities = [];
  const edges = [];
  const moduleEntityMap = new Map(); // module_path → entity
  const symbolEntityMap = new Map(); // qualified_name → entity

  // Phase 1: Extract module entities from file paths.
  for (const declaration of exportedDeclarations) {
    const modulePath = deriveModulePath(declaration.file_path);
    if (!moduleEntityMap.has(modulePath)) {
      const moduleEntity = {
        entity_type: 'module',
        name: modulePath.split('/').at(-1),
        qualified_name: modulePath,
        file_path: declaration.file_path,
        module_path: modulePath,
        signature_text: null,
        extra_metadata: JSON.stringify({ file_count: 0, symbol_count: 0 }),
      };
      entities.push(moduleEntity);
      moduleEntityMap.set(modulePath, moduleEntity);
    }
  }

  // Count symbols per module for extra_metadata.
  const moduleSymbolCounts = new Map();
  for (const declaration of exportedDeclarations) {
    const modulePath = deriveModulePath(declaration.file_path);
    moduleSymbolCounts.set(
      modulePath,
      (moduleSymbolCounts.get(modulePath) ?? 0) + 1,
    );
  }
  for (const moduleEntity of entities) {
    if (moduleEntity.entity_type === 'module') {
      const count = moduleSymbolCounts.get(moduleEntity.module_path) ?? 0;
      moduleEntity.extra_metadata = JSON.stringify({
        file_count: countModuleFiles(
          exportedDeclarations,
          moduleEntity.module_path,
        ),
        symbol_count: count,
      });
    }
  }

  // Phase 2: Extract symbol entities from declarations.
  for (const declaration of exportedDeclarations) {
    const symbolEntities = extractSymbolEntities(declaration);
    for (const entity of symbolEntities) {
      entities.push(entity);
      symbolEntityMap.set(entity.qualified_name, entity);
    }
  }

  // Phase 3: Extract relationships from source files.
  const sourceFiles = project.getSourceFiles();
  for (const sourceFile of sourceFiles) {
    const filePath = toRepoRelative(sourceFile.getFilePath());
    const modulePath = deriveModulePath(filePath);

    // Import relationships.
    extractImportRelationships(
      sourceFile,
      modulePath,
      moduleEntityMap,
      symbolEntityMap,
      edges,
    );

    // Export relationships.
    extractExportRelationships(
      sourceFile,
      modulePath,
      moduleEntityMap,
      symbolEntityMap,
      edges,
    );

    // Ownership relationships (module → symbol).
    extractOwnershipRelationships(
      sourceFile,
      modulePath,
      moduleEntityMap,
      symbolEntityMap,
      edges,
    );
  }

  // Phase 4: Extract depends-on and implements from declarations.
  for (const declaration of exportedDeclarations) {
    const modulePath = deriveModulePath(declaration.file_path);
    extractDependsOnEdges(declaration, modulePath, symbolEntityMap, edges);
    extractImplementsEdges(declaration, modulePath, symbolEntityMap, edges);
  }

  // Phase 5: Compute inverse part-of edges from owns edges.
  const ownsEdges = edges.filter((edge) => edge.relationship === 'owns');
  for (const ownsEdge of ownsEdges) {
    edges.push({
      source_entity_id: null, // Will be resolved later by qualified_name
      target_entity_id: null,
      source_qualified_name: ownsEdge.target_qualified_name,
      target_qualified_name: ownsEdge.source_qualified_name,
      relationship: 'part-of',
      confidence: 'high',
    });
  }

  // Phase 6: Deduplicate entities by qualified_name.
  // Handles re-exported symbols that appear multiple times in exportedDeclarations.
  const uniqueEntities = [];
  const seenQNames = new Set();
  for (const entity of entities) {
    if (!seenQNames.has(entity.qualified_name)) {
      seenQNames.add(entity.qualified_name);
      uniqueEntities.push(entity);
    } else {
      // Remove from symbolEntityMap if it was overwritten by a later duplicate.
      // Keep the first occurrence — it was already set correctly in Phase 2.
    }
  }

  return { entities: uniqueEntities, edges, moduleEntityMap, symbolEntityMap };
}

/**
 * Derive a file-based module path from a file path.
 *
 * Always includes the file basename (minus extension) to ensure unique module
 * paths per source file. This avoids qualified-name collisions when multiple
 * files in the same directory export symbols with the same name (e.g. `default`).
 *
 * Examples:
 * - `src/neat.ts` → `src/neat`
 * - `src/architecture/network/network.ts` → `src/architecture/network/network`
 * - `src/architecture/architect.ts` → `src/architecture/architect`
 * - `src/methods/selection/index.ts` → `src/methods/selection/index`
 *
 * @param {string} filePath - Repository-relative file path.
 * @returns {string} Module path.
 */
export function deriveModulePath(filePath) {
  return filePath.replace(/\.ts$/, '').replace(/^\.\//, '').replace(/^\//, '');
}

/**
 * Extract symbol entities from an exported declaration.
 *
 * Maps ts-morph declaration kinds to entity types and constructs
 * qualified names following the dot-separated convention.
 *
 * @param {object} declaration - Exported declaration with ts-morph node and metadata.
 * @returns {Array<object>} Extracted entity objects.
 */
function extractSymbolEntities(declaration) {
  const {
    declaration: declNode,
    file_path: filePath,
    symbol_name: symbolName,
  } = declaration;
  const modulePath = deriveModulePath(filePath);
  const entities = [];

  if (Node.isClassDeclaration(declNode)) {
    const isErrClass = isErrorClass(declNode);
    const entityType = isErrClass ? 'error-class' : 'class';
    const qualifiedName = `${modulePath}.${symbolName}`;
    const signatureText = resolveSignatureText(declNode);

    entities.push({
      entity_type: entityType,
      name: symbolName,
      qualified_name: qualifiedName,
      file_path: filePath,
      module_path: modulePath,
      signature_text: cleanWhitespace(signatureText),
      char_start: declNode.getStart(),
      char_end: declNode.getEnd(),
      extra_metadata: '{}',
    });

    // Extract class methods as separate function entities.
    const methods = declNode.getMethods();
    for (const method of methods) {
      const methodName = method.getName?.() ?? method.getSymbol()?.getName();
      if (!methodName) continue;

      const methodBody = method.getBody?.();
      const methodText = method.getText?.() ?? '';
      // Skip short methods (accessors, trivial getters/setters).
      if (methodText.length < MIN_METHOD_ENTITY_CHARS) continue;

      const methodQualifiedName = `${modulePath}.${symbolName}.${methodName}`;
      const methodSignature = method.getText().split('{', 1)[0]?.trim() ?? '';

      entities.push({
        entity_type: 'function',
        name: methodName,
        qualified_name: methodQualifiedName,
        file_path: filePath,
        module_path: modulePath,
        signature_text: cleanWhitespace(methodSignature),
        char_start: method.getStart(),
        char_end: method.getEnd(),
        extra_metadata: JSON.stringify({ parent_class: qualifiedName }),
      });
    }
  } else if (Node.isFunctionDeclaration(declNode)) {
    const qualifiedName = `${modulePath}.${symbolName}`;
    const signatureText = resolveSignatureText(declNode);

    entities.push({
      entity_type: 'function',
      name: symbolName,
      qualified_name: qualifiedName,
      file_path: filePath,
      module_path: modulePath,
      signature_text: cleanWhitespace(signatureText),
      char_start: declNode.getStart(),
      char_end: declNode.getEnd(),
      extra_metadata: '{}',
    });
  } else if (Node.isInterfaceDeclaration(declNode)) {
    const qualifiedName = `${modulePath}.${symbolName}`;
    const signatureText = resolveSignatureText(declNode);

    entities.push({
      entity_type: 'interface',
      name: symbolName,
      qualified_name: qualifiedName,
      file_path: filePath,
      module_path: modulePath,
      signature_text: cleanWhitespace(signatureText),
      char_start: declNode.getStart(),
      char_end: declNode.getEnd(),
      extra_metadata: '{}',
    });
  } else if (Node.isTypeAliasDeclaration(declNode)) {
    const qualifiedName = `${modulePath}.${symbolName}`;
    const signatureText = resolveSignatureText(declNode);

    entities.push({
      entity_type: 'type-alias',
      name: symbolName,
      qualified_name: qualifiedName,
      file_path: filePath,
      module_path: modulePath,
      signature_text: cleanWhitespace(signatureText),
      char_start: declNode.getStart(),
      char_end: declNode.getEnd(),
      extra_metadata: '{}',
    });
  } else if (Node.isVariableDeclaration(declNode)) {
    // Determine if the variable is a function (arrow function or function expression).
    const isFunctionVariable = isVariableFunction(declNode);
    const entityType = isFunctionVariable ? 'function' : 'variable';
    const qualifiedName = `${modulePath}.${symbolName}`;
    const signatureText = resolveSignatureText(declNode);

    entities.push({
      entity_type: entityType,
      name: symbolName,
      qualified_name: qualifiedName,
      file_path: filePath,
      module_path: modulePath,
      signature_text: cleanWhitespace(signatureText),
      char_start: declNode.getStart(),
      char_end: declNode.getEnd(),
      extra_metadata: '{}',
    });
  }

  return entities;
}

/**
 * Check if a class declaration extends Error.
 *
 * @param {object} classDecl - ts-morph ClassDeclaration.
 * @returns {boolean} True if the class extends Error.
 */
function isErrorClass(classDecl) {
  const extendsClause = classDecl.getExtends?.();
  if (!extendsClause) return false;
  const extendsText = extendsClause.getText?.() ?? '';
  return extendsText.endsWith('Error');
}

/**
 * Check if a variable declaration is an arrow function or function expression.
 *
 * @param {object} variableDecl - ts-morph VariableDeclaration.
 * @returns {boolean} True if the variable is a function.
 */
function isVariableFunction(variableDecl) {
  const initializer = variableDecl.getInitializer?.();
  if (!initializer) return false;
  return (
    Node.isArrowFunction(initializer) || Node.isFunctionExpression(initializer)
  );
}

/**
 * Extract import relationships from a source file.
 *
 * For each import declaration, creates:
 * - `imports` edge from the importing module to the imported module
 * - `imports` edge from the importing module to the imported symbol (if resolvable)
 *
 * @param {object} sourceFile - ts-morph SourceFile.
 * @param {string} modulePath - Module path of the source file.
 * @param {Map} moduleEntityMap - Map of module_path → entity.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @param {Array} edges - Accumulator for extracted edges.
 */
function extractImportRelationships(
  sourceFile,
  modulePath,
  moduleEntityMap,
  symbolEntityMap,
  edges,
) {
  const importDeclarations = sourceFile.getImportDeclarations?.() ?? [];

  for (const importDecl of importDeclarations) {
    const moduleSpecifier = importDecl.getModuleSpecifierValue?.();
    if (!moduleSpecifier) continue;

    // Resolve the specifier to a module path.
    const targetModulePath = resolveImportModulePath(
      moduleSpecifier,
      sourceFile,
      modulePath,
    );
    if (!targetModulePath) continue;

    // Only extract imports within the repository (skip node_modules, external packages).
    if (
      !targetModulePath.startsWith('src/') &&
      !targetModulePath.startsWith('.')
    )
      continue;

    // Module-to-module imports edge.
    const sourceModule = moduleEntityMap.get(modulePath);
    const targetModule = moduleEntityMap.get(targetModulePath);
    if (sourceModule && targetModule) {
      edges.push({
        source_qualified_name: modulePath,
        target_qualified_name: targetModulePath,
        relationship: 'imports',
        confidence: 'high',
      });
    }

    // Named imports → module-to-symbol edges.
    const namedImports = importDecl.getNamedImports?.() ?? [];
    for (const namedImport of namedImports) {
      const importName = namedImport.getName?.();
      if (!importName) continue;

      const targetQualifiedName = `${targetModulePath}.${importName}`;
      const targetEntity = symbolEntityMap.get(targetQualifiedName);
      if (sourceModule && targetEntity) {
        edges.push({
          source_qualified_name: modulePath,
          target_qualified_name: targetQualifiedName,
          relationship: 'imports',
          confidence: 'high',
        });
      }
    }
  }
}

/**
 * Extract export relationships from a source file.
 *
 * For each exported symbol, creates an `exports` edge from the module
 * to the symbol entity.
 *
 * @param {object} sourceFile - ts-morph SourceFile.
 * @param {string} modulePath - Module path of the source file.
 * @param {Map} moduleEntityMap - Map of module_path → entity.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @param {Array} edges - Accumulator for extracted edges.
 */
function extractExportRelationships(
  sourceFile,
  modulePath,
  moduleEntityMap,
  symbolEntityMap,
  edges,
) {
  const exportedDeclarations =
    sourceFile.getExportedDeclarations?.() ?? new Map();
  const sourceModule = moduleEntityMap.get(modulePath);

  for (const [exportName, declarations] of exportedDeclarations) {
    const targetQualifiedName = `${modulePath}.${exportName}`;
    const targetEntity = symbolEntityMap.get(targetQualifiedName);
    if (sourceModule && targetEntity) {
      edges.push({
        source_qualified_name: modulePath,
        target_qualified_name: targetQualifiedName,
        relationship: 'exports',
        confidence: 'high',
      });
    }
  }
}

/**
 * Extract ownership relationships from a source file.
 *
 * For each exported symbol in the file, creates an `owns` edge from the
 * module to the symbol entity.
 *
 * @param {object} sourceFile - ts-morph SourceFile.
 * @param {string} modulePath - Module path of the source file.
 * @param {Map} moduleEntityMap - Map of module_path → entity.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @param {Array} edges - Accumulator for extracted edges.
 */
function extractOwnershipRelationships(
  sourceFile,
  modulePath,
  moduleEntityMap,
  symbolEntityMap,
  edges,
) {
  const exportedDeclarations =
    sourceFile.getExportedDeclarations?.() ?? new Map();
  const sourceModule = moduleEntityMap.get(modulePath);

  for (const [exportName, declarations] of exportedDeclarations) {
    const targetQualifiedName = `${modulePath}.${exportName}`;
    const targetEntity = symbolEntityMap.get(targetQualifiedName);
    if (sourceModule && targetEntity) {
      edges.push({
        source_qualified_name: modulePath,
        target_qualified_name: targetQualifiedName,
        relationship: 'owns',
        confidence: 'high',
      });
    }

    // Also create owns edges from class to method for method entities.
    for (const decl of declarations) {
      if (Node.isClassDeclaration(decl)) {
        const classQualifiedName = `${modulePath}.${exportName}`;
        const methods = decl.getMethods();
        for (const method of methods) {
          const methodName =
            method.getName?.() ?? method.getSymbol()?.getName();
          if (!methodName) continue;
          const methodText = method.getText?.() ?? '';
          if (methodText.length < MIN_METHOD_ENTITY_CHARS) continue;
          const methodQualifiedName = `${classQualifiedName}.${methodName}`;
          const methodEntity = symbolEntityMap.get(methodQualifiedName);
          if (methodEntity) {
            edges.push({
              source_qualified_name: classQualifiedName,
              target_qualified_name: methodQualifiedName,
              relationship: 'owns',
              confidence: 'high',
            });
          }
        }
      }
    }
  }
}

/**
 * Extract depends-on edges from type references, constructor parameters,
 * and call expressions within a declaration.
 *
 * @param {object} declaration - Exported declaration with ts-morph node.
 * @param {string} modulePath - Module path of the declaration.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @param {Array} edges - Accumulator for extracted edges.
 */
function extractDependsOnEdges(
  declaration,
  modulePath,
  symbolEntityMap,
  edges,
) {
  const { declaration: declNode, symbol_name: symbolName } = declaration;
  const sourceQualifiedName = `${modulePath}.${symbolName}`;
  const sourceEntity = symbolEntityMap.get(sourceQualifiedName);
  if (!sourceEntity) return;

  let edgeCount = 0;
  const seenTargets = new Set();

  // Extract from type references in the signature.
  const typeReferences = extractTypeReferences(
    declNode,
    modulePath,
    symbolEntityMap,
  );
  for (const targetQualifiedName of typeReferences) {
    if (edgeCount >= MAX_DEPENDS_ON_EDGES) break;
    if (seenTargets.has(targetQualifiedName)) continue;
    if (targetQualifiedName === sourceQualifiedName) continue;
    seenTargets.add(targetQualifiedName);

    edges.push({
      source_qualified_name: sourceQualifiedName,
      target_qualified_name: targetQualifiedName,
      relationship: 'depends-on',
      confidence: 'medium',
    });
    edgeCount += 1;
  }

  // Extract from constructor parameters for classes.
  if (Node.isClassDeclaration(declNode)) {
    const constructors = declNode.getConstructors?.() ?? [];
    for (const constructor of constructors) {
      const params = constructor.getParameters?.() ?? [];
      for (const param of params) {
        const paramType = param.getType?.();
        if (!paramType) continue;
        const typeSymbol = paramType.getSymbol?.();
        if (!typeSymbol) continue;
        const typeName = typeSymbol.getName?.();
        if (!typeName) continue;

        const targetQualifiedName = resolveTypeReferenceToQualifiedName(
          typeName,
          modulePath,
          symbolEntityMap,
        );
        if (!targetQualifiedName) continue;
        if (edgeCount >= MAX_DEPENDS_ON_EDGES) break;
        if (seenTargets.has(targetQualifiedName)) continue;
        if (targetQualifiedName === sourceQualifiedName) continue;
        seenTargets.add(targetQualifiedName);

        edges.push({
          source_qualified_name: sourceQualifiedName,
          target_qualified_name: targetQualifiedName,
          relationship: 'depends-on',
          confidence: 'medium',
        });
        edgeCount += 1;
      }
    }
  }

  // Extract from call expressions in the body.
  const callTargets = extractCallTargets(declNode, modulePath, symbolEntityMap);
  for (const targetQualifiedName of callTargets) {
    if (edgeCount >= MAX_DEPENDS_ON_EDGES) break;
    if (seenTargets.has(targetQualifiedName)) continue;
    if (targetQualifiedName === sourceQualifiedName) continue;
    seenTargets.add(targetQualifiedName);

    edges.push({
      source_qualified_name: sourceQualifiedName,
      target_qualified_name: targetQualifiedName,
      relationship: 'depends-on',
      confidence: 'medium',
    });
    edgeCount += 1;
  }
}

/**
 * Extract implements edges from class declarations.
 *
 * @param {object} declaration - Exported declaration with ts-morph node.
 * @param {string} modulePath - Module path of the declaration.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @param {Array} edges - Accumulator for extracted edges.
 */
function extractImplementsEdges(
  declaration,
  modulePath,
  symbolEntityMap,
  edges,
) {
  const { declaration: declNode, symbol_name: symbolName } = declaration;
  if (!Node.isClassDeclaration(declNode)) return;

  const sourceQualifiedName = `${modulePath}.${symbolName}`;
  const sourceEntity = symbolEntityMap.get(sourceQualifiedName);
  if (!sourceEntity) return;

  const implementsClauses = declNode.getImplements?.() ?? [];
  for (const implementsExpr of implementsClauses) {
    const implementsText = implementsExpr.getText?.() ?? '';
    const targetQualifiedName = resolveTypeReferenceToQualifiedName(
      implementsText,
      modulePath,
      symbolEntityMap,
    );
    if (!targetQualifiedName) continue;

    edges.push({
      source_qualified_name: sourceQualifiedName,
      target_qualified_name: targetQualifiedName,
      relationship: 'implements',
      confidence: 'high',
    });
  }
}

/**
 * Extract type references from a declaration's signature text.
 *
 * Searches the signature for known entity names and resolves them to
 * qualified names within the same module or imported modules.
 *
 * @param {object} declNode - ts-morph declaration node.
 * @param {string} modulePath - Module path of the declaration.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @returns {string[]} Array of target qualified names.
 */
function extractTypeReferences(declNode, modulePath, symbolEntityMap) {
  const signatureText = resolveSignatureText(declNode) ?? '';
  const references = [];

  // Look for type names that match existing entities.
  for (const [qualifiedName, entity] of symbolEntityMap) {
    if (entity.entity_type === 'module') continue;
    const name = entity.name;
    // Simple check: is the name mentioned in the signature?
    if (name.length > 2 && signatureText.includes(name)) {
      // Verify it's a type reference, not just substring match.
      const typeRefPattern = new RegExp(`\\b${escapeRegExp(name)}\\b`);
      if (typeRefPattern.test(signatureText)) {
        references.push(qualifiedName);
      }
    }
  }

  return references;
}

/**
 * Extract call targets from the first portion of a function body.
 *
 * @param {object} declNode - ts-morph declaration node.
 * @param {string} modulePath - Module path of the declaration.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @returns {string[]} Array of target qualified names.
 */
function extractCallTargets(declNode, modulePath, symbolEntityMap) {
  const targets = [];
  const bodyText = declNode.getText?.() ?? '';

  // Only scan first 500 chars to avoid O(N²).
  const scanText = bodyText.slice(0, MAX_CALL_EXPRESSION_SCAN_CHARS);

  for (const [qualifiedName, entity] of symbolEntityMap) {
    if (entity.entity_type === 'module') continue;
    const name = entity.name;
    if (name.length < 3) continue;

    // Look for call patterns: name( or Name. or name.
    const callPattern = new RegExp(`\\b${escapeRegExp(name)}\\s*[.(]`);
    if (callPattern.test(scanText)) {
      targets.push(qualifiedName);
    }
  }

  return targets;
}

/**
 * Resolve an import module specifier to a module path.
 *
 * Handles relative imports (./foo, ../bar) and package-internal imports
 * (src/foo/bar). Returns null for external package imports.
 *
 * @param {string} specifier - The import module specifier string.
 * @param {object} sourceFile - The ts-morph SourceFile containing the import.
 * @param {string} sourceModulePath - Module path of the importing file.
 * @returns {string | null} Resolved module path, or null for external imports.
 */
function resolveImportModulePath(specifier, sourceFile, sourceModulePath) {
  // Relative imports.
  if (specifier.startsWith('.')) {
    const sourceDir = path.posix.dirname(sourceFile.getFilePath());
    const resolvedPath = path.posix.resolve(sourceDir, specifier);
    const repoRelative = toRepoRelative(resolvedPath);
    // Strip .ts extension and /index suffixes.
    const cleaned = repoRelative.replace(/\.ts$/, '').replace(/\/index$/, '');
    return deriveModulePath(
      cleaned.endsWith('.ts') ? cleaned + '.ts' : cleaned + '.ts',
    );
  }

  // Package-internal imports starting with src/.
  if (specifier.startsWith('src/')) {
    return deriveModulePath(
      specifier.replace(/\.ts$/, '').replace(/\/index$/, ''),
    );
  }

  // External package — skip.
  return null;
}

/**
 * Resolve a type reference name to a qualified name by searching known entities.
 *
 * @param {string} typeName - Type name to resolve.
 * @param {string} modulePath - Module path for local resolution.
 * @param {Map} symbolEntityMap - Map of qualified_name → entity.
 * @returns {string | null} Qualified name if found, null otherwise.
 */
function resolveTypeReferenceToQualifiedName(
  typeName,
  modulePath,
  symbolEntityMap,
) {
  // Try local module first.
  const localQualifiedName = `${modulePath}.${typeName}`;
  if (symbolEntityMap.has(localQualifiedName)) return localQualifiedName;

  // Search all modules for the type name.
  for (const [qualifiedName, entity] of symbolEntityMap) {
    if (entity.name === typeName && entity.entity_type !== 'module') {
      return qualifiedName;
    }
  }

  return null;
}

/**
 * Count the number of unique files that contribute symbols to a module.
 *
 * @param {Array} exportedDeclarations - List of exported declarations.
 * @param {string} modulePath - Module path to count files for.
 * @returns {number} Number of unique file paths.
 */
function countModuleFiles(exportedDeclarations, modulePath) {
  const files = new Set();
  for (const decl of exportedDeclarations) {
    if (deriveModulePath(decl.file_path) === modulePath) {
      files.add(decl.file_path);
    }
  }
  return files.size;
}

/**
 * Escape special regex characters in a string.
 *
 * @param {string} value - String to escape.
 * @returns {string} Escaped string safe for use in RegExp.
 */
function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Collapse whitespace in a string for clean output.
 *
 * @param {string} value - String to clean.
 * @returns {string} Cleaned string.
 */
function cleanWhitespace(value) {
  return String(value ?? '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * CLI entrypoint for code entity extraction.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Code Entity Extractor',
      usage:
        'node rag-index/extract-code-entities.mjs [--json] [--source path]',
      options: [
        '--json           Emit JSON output of extracted entities and edges.',
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
    const result = await extractCodeEntities({
      sourcePaths: providedSources.length > 0 ? providedSources : undefined,
    });
    writeJsonOrText(result, Boolean(args.json), (payload) => {
      const entityCounts = {};
      for (const entity of payload.entities) {
        entityCounts[entity.entity_type] =
          (entityCounts[entity.entity_type] ?? 0) + 1;
      }
      return `Code entities: ${payload.entities.length} (${Object.entries(
        entityCounts,
      )
        .map(([k, v]) => `${k}=${v}`)
        .join(', ')}), edges: ${payload.edges.length}`;
    });
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
