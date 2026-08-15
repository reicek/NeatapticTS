/**
 * @description Scan `src/**\/*.ts` (non-test) for code quality signals: exported symbols
 * missing JSDoc entirely, JSDoc with fewer than 10 non-whitespace words (weak signal),
 * and functions with cyclomatic complexity above a configurable threshold. Emits the
 * standard gate JSON contract `{ pass, evidence, fixHint, owner }`. Also exposed as
 * the `scan_code_quality` MCP tool in `cortex`.
 *
 * @param {boolean} [--json]                       - Emit the standard gate JSON contract.
 * @param {number}  [--complexity-threshold <n>]   - Maximum allowed cyclomatic complexity (default: 10).
 * @param {number}  [--min-jsdoc-words <n>]        - Minimum non-whitespace words required in exported JSDoc (default: 10).
 * @param {string}  [--source <path>]              - Limit scanning to one or more explicit source file paths.
 * @param {boolean} [--help]                       - Show help and exit.
 *
 * @returns {void} Exits 0 when no quality issues are found above threshold, 1 otherwise.
 *   JSON contract written to stdout when `--json` is passed.
 */
import { Node, SyntaxKind } from 'ts-morph';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import {
  countWords,
  loadExportedTypeScriptDeclarations,
  resolveJsdocSummaryText,
} from './ts-chunker.mjs';

const DEFAULT_COMPLEXITY_THRESHOLD = 10;
const DEFAULT_MIN_JSDOC_WORDS = 10;
const DEFAULT_SCANNER_IGNORE = [
  'src/**/*.d.ts',
  'src/**/*.test.ts',
  'src/**/*.spec.ts',
  'src/**/__mocks__/*.ts',
];
const DOCUMENTATION_OWNER = '06-documenting';

/**
 * Issue dimensions emitted by this scanner.
 */
const ISSUE_DIMENSIONS = [
  'missing JSDoc',
  'weak JSDoc',
  'incomplete JSDoc tags',
  'high complexity',
];

export async function scanCodeQuality(options = {}) {
  const complexityThreshold = Number(
    options.complexityThreshold ?? DEFAULT_COMPLEXITY_THRESHOLD,
  );
  const minJsdocWords = Number(
    options.minJsdocWords ?? DEFAULT_MIN_JSDOC_WORDS,
  );
  const ignore = Array.isArray(options.ignore)
    ? options.ignore
    : DEFAULT_SCANNER_IGNORE;
  const exportedDeclarations = await loadExportedTypeScriptDeclarations({
    ...options,
    ignore,
  });
  const evidence = [];

  for (const exportedDeclaration of exportedDeclarations) {
    const documentationIssues = collectDocumentationIssues(
      exportedDeclaration,
      minJsdocWords,
    );
    const complexityIssues = collectComplexityIssues(
      exportedDeclaration,
      complexityThreshold,
    );
    evidence.push(...documentationIssues, ...complexityIssues);
  }

  return {
    pass: evidence.length === 0,
    evidence,
    issueDimensions: ISSUE_DIMENSIONS,
    fixHint:
      evidence.length === 0
        ? null
        : 'Add or strengthen JSDoc for the listed symbols, or simplify the flagged complex functions.',
    owner: DOCUMENTATION_OWNER,
  };
}

function collectDocumentationIssues(exportedDeclaration, minJsdocWords) {
  const jsdocText = resolveJsdocSummaryText(
    exportedDeclaration.declaration,
    exportedDeclaration.jsdoc_source_node,
  );
  if (!jsdocText) {
    return [
      {
        file: exportedDeclaration.file_path,
        issue: 'missing JSDoc',
        symbol: exportedDeclaration.symbol_name,
      },
    ];
  }

  const jsdocWordCount = countWords(jsdocText);
  if (jsdocWordCount < minJsdocWords) {
    return [
      {
        file: exportedDeclaration.file_path,
        issue: 'weak JSDoc',
        symbol: exportedDeclaration.symbol_name,
        words: jsdocWordCount,
      },
    ];
  }

  const functionLikeNode = resolveFunctionLikeNode(
    exportedDeclaration.declaration,
  );
  if (functionLikeNode) {
    const missingTags = collectMissingJsdocTags(functionLikeNode);
    if (missingTags.length > 0) {
      return [
        {
          file: exportedDeclaration.file_path,
          issue: 'incomplete JSDoc tags',
          symbol: exportedDeclaration.symbol_name,
          tags: missingTags,
        },
      ];
    }
  }

  return [];
}

function collectComplexityIssues(exportedDeclaration, complexityThreshold) {
  return resolveComplexityTargets(
    exportedDeclaration.declaration,
    exportedDeclaration.symbol_name,
  )
    .map(({ node, symbol }) => ({
      complexity: calculateCyclomaticComplexity(node),
      symbol,
    }))
    .filter(({ complexity }) => complexity > complexityThreshold)
    .map(({ complexity, symbol }) => ({
      complexity,
      file: exportedDeclaration.file_path,
      issue: 'high complexity',
      symbol,
    }));
}

function resolveComplexityTargets(declaration, symbolName) {
  if (Node.isFunctionDeclaration(declaration) && declaration.getBody()) {
    return [{ node: declaration, symbol: symbolName }];
  }

  if (Node.isVariableDeclaration(declaration)) {
    const initializer = declaration.getInitializer();
    if (
      initializer &&
      (Node.isArrowFunction(initializer) ||
        Node.isFunctionExpression(initializer))
    ) {
      return [{ node: initializer, symbol: symbolName }];
    }
  }

  if (Node.isClassDeclaration(declaration)) {
    return declaration
      .getMethods()
      .filter((methodDeclaration) => methodDeclaration.getBody() !== undefined)
      .map((methodDeclaration) => ({
        node: methodDeclaration,
        symbol: `${symbolName}.${methodDeclaration.getName()}`,
      }));
  }

  return [];
}

function calculateCyclomaticComplexity(node) {
  let complexity = 1;

  for (const descendant of node.forEachDescendantAsArray()) {
    switch (descendant.getKind()) {
      case SyntaxKind.IfStatement:
      case SyntaxKind.SwitchStatement:
      case SyntaxKind.ForStatement:
      case SyntaxKind.ForOfStatement:
      case SyntaxKind.ForInStatement:
      case SyntaxKind.WhileStatement:
      case SyntaxKind.DoStatement:
      case SyntaxKind.CaseClause:
      case SyntaxKind.DefaultClause:
      case SyntaxKind.CatchClause:
      case SyntaxKind.ConditionalExpression:
        complexity += 1;
        break;
      case SyntaxKind.TryStatement:
        if (descendant.getFinallyBlock?.()) {
          complexity += 1;
        }
        break;
      case SyntaxKind.PropertyAccessExpression:
      case SyntaxKind.CallExpression:
      case SyntaxKind.ElementAccessExpression: {
        if (
          Node.isQuestionDotTokenable(descendant) &&
          descendant.hasQuestionDotToken() &&
          !hasOptionalChainInLeftExpression(descendant)
        ) {
          complexity += 1;
        }
        break;
      }
      case SyntaxKind.BinaryExpression: {
        const operatorToken = descendant.getOperatorToken();
        const operatorKind = operatorToken?.getKind?.();
        if (
          operatorKind === SyntaxKind.AmpersandAmpersandToken ||
          operatorKind === SyntaxKind.BarBarToken ||
          operatorKind === SyntaxKind.QuestionQuestionToken ||
          operatorKind === SyntaxKind.AmpersandAmpersandEqualsToken ||
          operatorKind === SyntaxKind.BarBarEqualsToken ||
          operatorKind === SyntaxKind.QuestionQuestionEqualsToken
        ) {
          complexity += 1;
        }
        break;
      }
      default:
        break;
    }
  }

  return complexity;
}

function resolveFunctionLikeNode(declaration) {
  if (
    Node.isFunctionDeclaration(declaration) ||
    Node.isMethodDeclaration(declaration) ||
    Node.isArrowFunction(declaration) ||
    Node.isFunctionExpression(declaration)
  ) {
    return declaration;
  }

  if (Node.isVariableDeclaration(declaration)) {
    const initializer = declaration.getInitializer();
    if (
      initializer &&
      (Node.isArrowFunction(initializer) ||
        Node.isFunctionExpression(initializer))
    ) {
      return initializer;
    }
  }

  return null;
}

function collectMissingJsdocTags(functionLikeNode) {
  const jsDocs = resolveJsdocNodes(functionLikeNode);
  const presentTagNames = new Set(
    jsDocs.flatMap((jsDoc) => jsDoc.getTags().map((tag) => tag.getTagName())),
  );
  const missingTags = [];

  const parameters = functionLikeNode.getParameters();
  const paramTags = jsDocs
    .flatMap((jsDoc) => jsDoc.getTags())
    .filter((tag) => tag.getTagName() === 'param');
  const documentedParamNames = new Set(
    paramTags
      .map((tag) => (typeof tag.getName === 'function' ? tag.getName() : null))
      .filter(Boolean),
  );

  for (const parameter of parameters) {
    const parameterName = parameter.getName();
    if (!documentedParamNames.has(parameterName)) {
      missingTags.push(`@param ${parameterName}`);
    }
  }

  if (
    hasMeaningfulReturnType(functionLikeNode) &&
    !presentTagNames.has('returns')
  ) {
    missingTags.push('@returns');
  }

  if (
    containsThrowStatement(functionLikeNode) &&
    !presentTagNames.has('throws')
  ) {
    missingTags.push('@throws');
  }

  return missingTags;
}

function resolveJsdocNodes(node) {
  if (typeof node?.getJsDocs === 'function') {
    const directJsDocs = node.getJsDocs();
    if (directJsDocs.length > 0) return directJsDocs;
  }

  if (Node.isVariableDeclaration(node)) {
    const variableStatement = node.getVariableStatement?.();
    if (
      variableStatement &&
      typeof variableStatement.getJsDocs === 'function'
    ) {
      const statementJsDocs = variableStatement.getJsDocs();
      if (statementJsDocs.length > 0) return statementJsDocs;
    }
  }

  // ArrowFunction and FunctionExpression initializers of VariableDeclarations
  // carry their JSDoc on the enclosing VariableStatement, not on the function
  // node itself. Walk up to the parent VariableDeclaration to recover it.
  if (Node.isArrowFunction(node) || Node.isFunctionExpression(node)) {
    const parent = node.getParent?.();
    if (parent && Node.isVariableDeclaration(parent)) {
      const variableStatement = parent.getVariableStatement?.();
      if (
        variableStatement &&
        typeof variableStatement.getJsDocs === 'function'
      ) {
        const statementJsDocs = variableStatement.getJsDocs();
        if (statementJsDocs.length > 0) return statementJsDocs;
      }
    }
  }

  return [];
}

function containsThrowStatement(functionLikeNode) {
  return (
    functionLikeNode.getDescendantsOfKind?.(SyntaxKind.ThrowStatement).length >
    0
  );
}

function hasMeaningfulReturnType(functionLikeNode) {
  const returnTypeNode = functionLikeNode.getReturnTypeNode?.();
  if (returnTypeNode) {
    const returnTypeText = returnTypeNode.getText().trim();
    if (returnTypeText === 'void' || returnTypeText === 'undefined') {
      return false;
    }
  }

  return functionLikeNode
    .getDescendantsOfKind?.(SyntaxKind.ReturnStatement)
    .some(
      (returnStatement) =>
        returnStatement.getExpression?.() !== undefined &&
        returnStatement.getExpression?.() !== null,
    );
}

function hasOptionalChainInLeftExpression(optionalNode) {
  const leftExpression = optionalNode.getExpression?.();
  if (!leftExpression) return false;

  if (
    Node.isQuestionDotTokenable(leftExpression) &&
    leftExpression.hasQuestionDotToken()
  ) {
    return true;
  }

  return leftExpression
    .forEachDescendantAsArray()
    .some(
      (descendant) =>
        Node.isQuestionDotTokenable(descendant) &&
        descendant.hasQuestionDotToken(),
    );
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2), {
    repeatableFlags: ['source'],
  });
  if (args.help) {
    printHelp({
      title: 'Code quality scanner',
      usage: 'node rag-index/code-quality-scanner.mjs [--json]',
      options: [
        '--json                       Emit the standard gate JSON contract.',
        '--complexity-threshold <n>   Maximum allowed cyclomatic complexity (default: 10).',
        '--min-jsdoc-words <n>        Minimum non-whitespace words in exported JSDoc (default: 10).',
        '--source <path>              Limit scanning to one or more explicit source paths.',
        '--help                       Show this help.',
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
    const report = await scanCodeQuality({
      complexityThreshold: args['complexity-threshold'],
      minJsdocWords: args['min-jsdoc-words'],
      sourcePaths: providedSources.length > 0 ? providedSources : undefined,
    });
    writeJsonOrText(report, Boolean(args.json), (payload) =>
      payload.pass
        ? 'Code quality scan passed.'
        : `Code quality scan failed: ${payload.evidence.length} issue(s).`,
    );
    if (!report.pass) process.exitCode = 1;
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
