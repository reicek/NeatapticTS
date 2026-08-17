import { jest } from '@jest/globals';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Mock ts-morph with controlled Node static methods and SyntaxKind enum
// ---------------------------------------------------------------------------
const SyntaxKind = {
  IfStatement: 'IfStatement',
  SwitchStatement: 'SwitchStatement',
  ForStatement: 'ForStatement',
  ForOfStatement: 'ForOfStatement',
  ForInStatement: 'ForInStatement',
  WhileStatement: 'WhileStatement',
  DoStatement: 'DoStatement',
  CaseClause: 'CaseClause',
  DefaultClause: 'DefaultClause',
  CatchClause: 'CatchClause',
  ConditionalExpression: 'ConditionalExpression',
  TryStatement: 'TryStatement',
  PropertyAccessExpression: 'PropertyAccessExpression',
  CallExpression: 'CallExpression',
  ElementAccessExpression: 'ElementAccessExpression',
  BinaryExpression: 'BinaryExpression',
  AmpersandAmpersandToken: 'AmpersandAmpersandToken',
  BarBarToken: 'BarBarToken',
  QuestionQuestionToken: 'QuestionQuestionToken',
  AmpersandAmpersandEqualsToken: 'AmpersandAmpersandEqualsToken',
  BarBarEqualsToken: 'BarBarEqualsToken',
  QuestionQuestionEqualsToken: 'QuestionQuestionEqualsToken',
  ThrowStatement: 'ThrowStatement',
  ReturnStatement: 'ReturnStatement',
};

const Node = {
  isFunctionDeclaration: (n) => n?._kind === 'FunctionDeclaration',
  isVariableDeclaration: (n) => n?._kind === 'VariableDeclaration',
  isClassDeclaration: (n) => n?._kind === 'ClassDeclaration',
  isArrowFunction: (n) => n?._kind === 'ArrowFunction',
  isFunctionExpression: (n) => n?._kind === 'FunctionExpression',
  isMethodDeclaration: (n) => n?._kind === 'MethodDeclaration',
  isQuestionDotTokenable: (n) => n?._questionDotTokenable === true,
};

jest.unstable_mockModule('ts-morph', () => ({
  Node,
  SyntaxKind,
  default: { Node, SyntaxKind },
}));

// ---------------------------------------------------------------------------
// Mock ts-chunker
// ---------------------------------------------------------------------------
const mockLoadExportedTypeScriptDeclarations = jest.fn();
const mockResolveJsdocSummaryText = jest.fn();
const mockCountWords = jest.fn();
jest.unstable_mockModule('./ts-chunker.mjs', () => ({
  loadExportedTypeScriptDeclarations: mockLoadExportedTypeScriptDeclarations,
  resolveJsdocSummaryText: mockResolveJsdocSummaryText,
  countWords: mockCountWords,
  default: {
    loadExportedTypeScriptDeclarations: mockLoadExportedTypeScriptDeclarations,
    resolveJsdocSummaryText: mockResolveJsdocSummaryText,
    countWords: mockCountWords,
  },
}));

// ---------------------------------------------------------------------------
// Mock cli-utils
// ---------------------------------------------------------------------------
const mockFail = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
  parseCliArgs: (args, opts) => {
    const result = {};
    const repeatable = opts?.repeatableFlags ?? [];
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--help') result.help = true;
      else if (args[i] === '--json') result.json = true;
      else if (args[i] === '--complexity-threshold') result['complexity-threshold'] = args[++i];
      else if (args[i] === '--min-jsdoc-words') result['min-jsdoc-words'] = args[++i];
      else if (args[i] === '--source') {
        if (repeatable.includes('source')) {
          result.source = result.source || [];
          result.source.push(args[++i]);
        } else {
          result.source = args[++i];
        }
      } else if (!args[i].startsWith('--')) {
        result._ = result._ || [];
        result._.push(args[i]);
      }
    }
    return result;
  },
}));

const { scanCodeQuality } = await import('./code-quality-scanner.mjs');

// ---------------------------------------------------------------------------
// Mock node builders
// ---------------------------------------------------------------------------
function createMockNode(kind, opts = {}) {
  const node = {
    _kind: kind,
    _questionDotTokenable: opts.questionDotTokenable ?? false,
    getBody: opts.getBody ?? (() => opts.body ?? null),
    getInitializer: opts.getInitializer ?? (() => opts.initializer ?? null),
    getMethods: opts.getMethods ?? (() => opts.methods ?? []),
    forEachDescendantAsArray: opts.forEachDescendantAsArray ?? (() => opts.descendants ?? []),
    getKind: opts.getKind ?? (() => opts.syntaxKind ?? null),
    getJsDocs: opts.getJsDocs ?? (() => opts.jsDocs ?? []),
    getVariableStatement: opts.getVariableStatement ?? (() => opts.variableStatement ?? null),
    getParent: opts.getParent ?? (() => opts.parent ?? null),
    getParameters: opts.getParameters ?? (() => opts.parameters ?? []),
    getReturnTypeNode: opts.getReturnTypeNode ?? (() => opts.returnTypeNode ?? null),
    getDescendantsOfKind: opts.getDescendantsOfKind ?? (() => opts.descendantsOfKind ?? []),
    getExpression: opts.getExpression ?? (() => opts.expression ?? null),
    hasQuestionDotToken: opts.hasQuestionDotToken ?? (() => opts.questionDot ?? false),
    getOperatorToken: opts.getOperatorToken ?? (() => opts.operatorToken ?? null),
    getFinallyBlock: opts.getFinallyBlock ?? (() => opts.finallyBlock ?? null),
    getName: opts.getName ?? (() => opts.name ?? 'mockMethod'),
    getText: opts.getText ?? (() => opts.text ?? ''),
  };
  return node;
}

function createMockTag(tagName, paramName = null) {
  return {
    getTagName: () => tagName,
    getName: typeof paramName === 'string' ? () => paramName : null,
  };
}

function createMockJsDoc(tags = []) {
  return { getTags: () => tags };
}

function createExportedDeclaration(declaration, opts = {}) {
  return {
    declaration,
    jsdoc_source_node: opts.jsdocSourceNode ?? declaration,
    symbol_name: opts.symbolName ?? 'testSymbol',
    file_path: opts.filePath ?? 'src/test.ts',
  };
}

afterEach(() => {
  jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('scanCodeQuality — empty declarations', () => {
  it('returns pass=true with empty evidence', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
    expect(result.evidence).toEqual([]);
    expect(result.fixHint).toBeNull();
    expect(result.owner).toBe('06-documenting');
  });
});

describe('scanCodeQuality — documentation issues', () => {
  it('detects missing JSDoc', async () => {
    const decl = createMockNode('FunctionDeclaration', { body: {} });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('');
    const result = await scanCodeQuality();
    expect(result.pass).toBe(false);
    expect(result.evidence[0].issue).toBe('missing JSDoc');
  });

  it('detects weak JSDoc', async () => {
    const decl = createMockNode('FunctionDeclaration', { body: {} });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('short doc');
    mockCountWords.mockReturnValue(3);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('weak JSDoc');
    expect(result.evidence[0].words).toBe(3);
  });

  it('returns no issues for complete JSDoc with all tags on function declaration', async () => {
    const paramNode = { getName: () => 'myParam' };
    const tags = [
      createMockTag('param', 'myParam'),
      createMockTag('returns'),
      createMockTag('throws'),
    ];
    const jsDoc = createMockJsDoc(tags);
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [jsDoc],
      parameters: [paramNode],
      descendantsOfKind: [], // no throw statements, no return statements
    });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words to pass the threshold');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });

  it('detects missing @param tag', async () => {
    const paramNode = { getName: () => 'unDocumented' };
    const tags = [createMockTag('returns'), createMockTag('throws')];
    const jsDoc = createMockJsDoc(tags);
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [jsDoc],
      parameters: [paramNode],
      descendantsOfKind: [],
    });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('incomplete JSDoc tags');
    expect(result.evidence[0].tags).toContain('@param unDocumented');
  });

  it('detects missing @returns tag when function has meaningful return', async () => {
    const returnStmt = { getExpression: () => 'someValue' };
    const tags = [createMockTag('param', 'p1'), createMockTag('throws')];
    const jsDoc = createMockJsDoc(tags);
    const paramNode = { getName: () => 'p1' };
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [jsDoc],
      parameters: [paramNode],
      returnTypeNode: null,
      descendantsOfKind: [], // will override getDescendantsOfKind per kind
    });
    // Override getDescendantsOfKind to return different values per kind
    decl.getDescendantsOfKind = (kind) => {
      if (kind === SyntaxKind.ThrowStatement) return [];
      if (kind === SyntaxKind.ReturnStatement) return [returnStmt];
      return [];
    };
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('incomplete JSDoc tags');
    expect(result.evidence[0].tags).toContain('@returns');
  });

  it('detects missing @throws tag when function throws', async () => {
    const tags = [createMockTag('param', 'p1'), createMockTag('returns')];
    const jsDoc = createMockJsDoc(tags);
    const paramNode = { getName: () => 'p1' };
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [jsDoc],
      parameters: [paramNode],
      descendantsOfKind: [],
    });
    decl.getDescendantsOfKind = (kind) => {
      if (kind === SyntaxKind.ThrowStatement) return [{}];
      if (kind === SyntaxKind.ReturnStatement) return [];
      return [];
    };
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('incomplete JSDoc tags');
    expect(result.evidence[0].tags).toContain('@throws');
  });

  it('skips @returns when return type is void', async () => {
    const tags = [createMockTag('param', 'p1'), createMockTag('throws')];
    const jsDoc = createMockJsDoc(tags);
    const paramNode = { getName: () => 'p1' };
    const returnTypeNode = { getText: () => 'void' };
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [jsDoc],
      parameters: [paramNode],
      returnTypeNode,
    });
    decl.getDescendantsOfKind = () => [];
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    // No missing tags — void return doesn't need @returns, no throw statements
    expect(result.pass).toBe(true);
  });

  it('skips @returns when return type is undefined', async () => {
    const tags = [createMockTag('param', 'p1')];
    const jsDoc = createMockJsDoc(tags);
    const paramNode = { getName: () => 'p1' };
    const returnTypeNode = { getText: () => 'undefined' };
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [jsDoc],
      parameters: [paramNode],
      returnTypeNode,
    });
    decl.getDescendantsOfKind = () => [];
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });

  it('skips @returns when return statement has no expression', async () => {
    const returnStmt = { getExpression: () => undefined };
    const tags = [createMockTag('param', 'p1'), createMockTag('throws')];
    const jsDoc = createMockJsDoc(tags);
    const paramNode = { getName: () => 'p1' };
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [jsDoc],
      parameters: [paramNode],
    });
    decl.getDescendantsOfKind = (kind) => {
      if (kind === SyntaxKind.ThrowStatement) return [];
      if (kind === SyntaxKind.ReturnStatement) return [returnStmt];
      return [];
    };
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });

  it('handles non-function declaration with complete JSDoc (no function-like node)', async () => {
    const decl = createMockNode('InterfaceDeclaration');
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([
      createExportedDeclaration(decl),
    ]);
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });
});

describe('scanCodeQuality — resolveFunctionLikeNode branches', () => {
  beforeEach(() => {
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
  });

  it('resolves MethodDeclaration', async () => {
    const decl = createMockNode('MethodDeclaration', { jsDocs: [createMockJsDoc([])] });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });

  it('resolves ArrowFunction from VariableDeclaration', async () => {
    const arrowFn = createMockNode('ArrowFunction', {
      jsDocs: [],
      parent: createMockNode('VariableDeclaration', {
        variableStatement: { getJsDocs: () => [createMockJsDoc([createMockTag('param', 'x')])] },
      }),
      parameters: [{ getName: () => 'x' }],
    });
    arrowFn.getDescendantsOfKind = () => [];
    const decl = createMockNode('VariableDeclaration', { initializer: arrowFn });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    // Should find missing @returns via the variable statement JSDoc path
    expect(result.pass).toBe(true);
  });

  it('resolves FunctionExpression from VariableDeclaration', async () => {
    const fnExpr = createMockNode('FunctionExpression', {
      jsDocs: [],
      parent: createMockNode('VariableDeclaration', {
        variableStatement: { getJsDocs: () => [createMockJsDoc([createMockTag('param', 'x')])] },
      }),
      parameters: [{ getName: () => 'x' }],
    });
    fnExpr.getDescendantsOfKind = () => [];
    const decl = createMockNode('VariableDeclaration', { initializer: fnExpr });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });

  it('returns null for VariableDeclaration with non-function initializer', async () => {
    const decl = createMockNode('VariableDeclaration', { initializer: createMockNode('StringLiteral') });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });

  it('returns null for unknown declaration type', async () => {
    const decl = createMockNode('EnumDeclaration');
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });
});

describe('scanCodeQuality — resolveJsdocNodes branches', () => {
  beforeEach(() => {
    mockResolveJsdocSummaryText.mockReturnValue('This is a sufficiently long JSDoc description with many words');
    mockCountWords.mockReturnValue(15);
  });

  it('uses variable statement JSDoc for VariableDeclaration', async () => {
    const vs = { getJsDocs: () => [createMockJsDoc([createMockTag('param', 'x')])] };
    const arrowFn = createMockNode('ArrowFunction', {
      jsDocs: [],
      parameters: [{ getName: () => 'x' }],
    });
    arrowFn.getDescendantsOfKind = () => [];
    const decl = createMockNode('VariableDeclaration', {
      initializer: arrowFn,
      jsDocs: [],
      variableStatement: vs,
    });
    // arrowFn's parent is the decl
    arrowFn.getParent = () => decl;
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });

  it('returns empty when no JSDoc found anywhere', async () => {
    const fnDecl = createMockNode('FunctionDeclaration', {
      body: {},
      jsDocs: [],
    });
    // When resolveJsdocNodes returns empty, collectMissingJsdocTags returns []
    // (no parameters, no return, no throws) — so no issues
    fnDecl.getDescendantsOfKind = () => [];
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(fnDecl)]);
    const result = await scanCodeQuality();
    expect(result.pass).toBe(true);
  });
});

describe('scanCodeQuality — resolveComplexityTargets branches', () => {
  beforeEach(() => {
    mockResolveJsdocSummaryText.mockReturnValue('');
  });

  it('handles FunctionDeclaration without body (falls through)', async () => {
    const decl = createMockNode('FunctionDeclaration', { body: null });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    // No complexity targets, no JSDoc → missing JSDoc only
    expect(result.evidence[0].issue).toBe('missing JSDoc');
  });

  it('handles VariableDeclaration with non-function initializer', async () => {
    const decl = createMockNode('VariableDeclaration', { initializer: createMockNode('ObjectLiteral') });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('missing JSDoc');
  });

  it('handles VariableDeclaration with null initializer', async () => {
    const decl = createMockNode('VariableDeclaration', { initializer: null });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('missing JSDoc');
  });

  it('handles ClassDeclaration with methods that have bodies', async () => {
    const methodBody = {};
    const method = createMockNode('MethodDeclaration', {
      body: methodBody,
      name: 'myMethod',
      descendants: [],
    });
    const decl = createMockNode('ClassDeclaration', { methods: [method] });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('missing JSDoc');
  });

  it('handles ClassDeclaration with methods without bodies (filtered out)', async () => {
    const method = createMockNode('MethodDeclaration', {
      body: undefined,
      name: 'abstractMethod',
    });
    const decl = createMockNode('ClassDeclaration', { methods: [method] });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('missing JSDoc');
  });

  it('handles unknown declaration type (returns empty complexity targets)', async () => {
    const decl = createMockNode('TypeAliasDeclaration');
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality();
    expect(result.evidence[0].issue).toBe('missing JSDoc');
  });
});

describe('scanCodeQuality — calculateCyclomaticComplexity', () => {
  it('detects high complexity with all SyntaxKind cases', async () => {
    // Create descendants covering all SyntaxKind cases in the switch
    const descendants = [
      createMockNode(null, { syntaxKind: SyntaxKind.IfStatement }),
      createMockNode(null, { syntaxKind: SyntaxKind.SwitchStatement }),
      createMockNode(null, { syntaxKind: SyntaxKind.ForStatement }),
      createMockNode(null, { syntaxKind: SyntaxKind.ForOfStatement }),
      createMockNode(null, { syntaxKind: SyntaxKind.ForInStatement }),
      createMockNode(null, { syntaxKind: SyntaxKind.WhileStatement }),
      createMockNode(null, { syntaxKind: SyntaxKind.DoStatement }),
      createMockNode(null, { syntaxKind: SyntaxKind.CaseClause }),
      createMockNode(null, { syntaxKind: SyntaxKind.DefaultClause }),
      createMockNode(null, { syntaxKind: SyntaxKind.CatchClause }),
      createMockNode(null, { syntaxKind: SyntaxKind.ConditionalExpression }),
      // TryStatement with finally
      createMockNode(null, {
        syntaxKind: SyntaxKind.TryStatement,
        finallyBlock: {},
      }),
      // TryStatement without finally
      createMockNode(null, {
        syntaxKind: SyntaxKind.TryStatement,
        finallyBlock: null,
      }),
      // PropertyAccessExpression with question dot, no optional chain in left
      createMockNode(null, {
        syntaxKind: SyntaxKind.PropertyAccessExpression,
        questionDotTokenable: true,
        questionDot: true,
        expression: createMockNode(null, { questionDotTokenable: false, questionDot: false, descendants: [] }),
      }),
      // CallExpression with question dot, has optional chain in left → not counted
      createMockNode(null, {
        syntaxKind: SyntaxKind.CallExpression,
        questionDotTokenable: true,
        questionDot: true,
        expression: createMockNode(null, {
          questionDotTokenable: true,
          questionDot: true,
          descendants: [],
        }),
      }),
      // CallExpression without question dot → not counted
      createMockNode(null, {
        syntaxKind: SyntaxKind.CallExpression,
        questionDotTokenable: true,
        questionDot: false,
      }),
      // ElementAccessExpression with question dot, no optional chain in left
      createMockNode(null, {
        syntaxKind: SyntaxKind.ElementAccessExpression,
        questionDotTokenable: true,
        questionDot: true,
        expression: null,
      }),
      // BinaryExpression with AmpersandAmpersandToken
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: { getKind: () => SyntaxKind.AmpersandAmpersandToken },
      }),
      // BinaryExpression with BarBarToken
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: { getKind: () => SyntaxKind.BarBarToken },
      }),
      // BinaryExpression with QuestionQuestionToken
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: { getKind: () => SyntaxKind.QuestionQuestionToken },
      }),
      // BinaryExpression with AmpersandAmpersandEqualsToken
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: { getKind: () => SyntaxKind.AmpersandAmpersandEqualsToken },
      }),
      // BinaryExpression with BarBarEqualsToken
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: { getKind: () => SyntaxKind.BarBarEqualsToken },
      }),
      // BinaryExpression with QuestionQuestionEqualsToken
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: { getKind: () => SyntaxKind.QuestionQuestionEqualsToken },
      }),
      // BinaryExpression with unknown operator → not counted
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: { getKind: () => 'UnknownToken' },
      }),
      // BinaryExpression with no operator token → not counted
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: null,
      }),
      // BinaryExpression with operator token without getKind → not counted
      createMockNode(null, {
        syntaxKind: SyntaxKind.BinaryExpression,
        operatorToken: {},
      }),
      // Default case (unknown SyntaxKind)
      createMockNode(null, { syntaxKind: 'UnknownKind' }),
      // PropertyAccessExpression without questionDotTokenable
      createMockNode(null, {
        syntaxKind: SyntaxKind.PropertyAccessExpression,
        questionDotTokenable: false,
      }),
    ];

    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      descendants,
    });
    mockResolveJsdocSummaryText.mockReturnValue('');
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    // Use a low threshold to trigger complexity issues
    const result = await scanCodeQuality({ complexityThreshold: 5 });
    const complexityIssues = result.evidence.filter((e) => e.issue === 'high complexity');
    expect(complexityIssues.length).toBeGreaterThan(0);
    // Base complexity 1 + IfStatement(1) + SwitchStatement(1) + ForStatement(1) + ForOfStatement(1)
    // + ForInStatement(1) + WhileStatement(1) + DoStatement(1) + CaseClause(1) + DefaultClause(1)
    // + CatchClause(1) + ConditionalExpression(1) + TryStatement+finally(1) + TryStatement-no-finally(0)
    // + PropertyAccessExpr w/questionDot(1) + CallExpr w/optionalChain(0) + CallExpr no-questionDot(0)
    // + ElementAccessExpr w/questionDot(1) + &&(1) + ||(1) + ??(1) + &&=(1) + ||=(1) + ??=(1)
    // + unknown operator(0) + null operator(0) + no getKind(0) + default(0) + no questionDotTokenable(0)
    // = 1 + 12 + 2 + 6 = 21
    expect(complexityIssues[0].complexity).toBe(21);
  });

  it('does not flag complexity below threshold', async () => {
    const descendants = [
      createMockNode(null, { syntaxKind: SyntaxKind.IfStatement }),
    ];
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      descendants,
    });
    mockResolveJsdocSummaryText.mockReturnValue('Long enough JSDoc description with many words to pass the threshold check');
    mockCountWords.mockReturnValue(15);
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality({ complexityThreshold: 10 });
    // complexity = 2 (base 1 + 1 IfStatement), which is ≤ 10
    expect(result.pass).toBe(true);
  });
});

describe('scanCodeQuality — hasOptionalChainInLeftExpression', () => {
  it('returns false when expression is null', async () => {
    // ElementAccessExpression with questionDot but expression is null
    const descendants = [
      createMockNode(null, {
        syntaxKind: SyntaxKind.ElementAccessExpression,
        questionDotTokenable: true,
        questionDot: true,
        expression: null,
      }),
    ];
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      descendants,
    });
    mockResolveJsdocSummaryText.mockReturnValue('');
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality({ complexityThreshold: 0 });
    // expression is null → hasOptionalChainInLeftExpression returns false → complexity +1
    const complexityIssues = result.evidence.filter((e) => e.issue === 'high complexity');
    expect(complexityIssues.length).toBeGreaterThan(0);
  });

  it('returns true when descendant has question dot', async () => {
    const innerDescendant = createMockNode(null, {
      questionDotTokenable: true,
      questionDot: true,
    });
    const leftExpr = createMockNode(null, {
      questionDotTokenable: false,
      questionDot: false,
      descendants: [innerDescendant],
    });
    const descendants = [
      createMockNode(null, {
        syntaxKind: SyntaxKind.CallExpression,
        questionDotTokenable: true,
        questionDot: true,
        expression: leftExpr,
      }),
    ];
    const decl = createMockNode('FunctionDeclaration', {
      body: {},
      descendants,
    });
    mockResolveJsdocSummaryText.mockReturnValue('');
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    const result = await scanCodeQuality({ complexityThreshold: 0 });
    // hasOptionalChainInLeftExpression returns true → complexity not incremented
    const complexityIssues = result.evidence.filter((e) => e.issue === 'high complexity');
    // Only base complexity 1, which is > 0 threshold
    expect(complexityIssues.length).toBeGreaterThan(0);
    expect(complexityIssues[0].complexity).toBe(1);
  });
});

describe('scanCodeQuality — options', () => {
  it('passes custom ignore array', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    await scanCodeQuality({ ignore: ['custom/*.ts'] });
    expect(mockLoadExportedTypeScriptDeclarations).toHaveBeenCalledWith(
      expect.objectContaining({ ignore: ['custom/*.ts'] }),
    );
  });

  it('uses default ignore when not array', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    await scanCodeQuality();
    const callArg = mockLoadExportedTypeScriptDeclarations.mock.calls[0][0];
    expect(callArg.ignore).toEqual(expect.arrayContaining(['src/**/*.d.ts']));
  });
});

describe('main() CLI entry point', () => {
  it('prints help when --help flag is provided', async () => {
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), '--help'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    expect(mockPrintHelp).toHaveBeenCalled();
  });

  it('runs successfully and reports pass', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), '--json'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    expect(mockWriteJsonOrText).toHaveBeenCalled();
  });

  it('sets exitCode to 1 when scan fails', async () => {
    const origExitCode = process.exitCode;
    const decl = createMockNode('FunctionDeclaration', { body: {} });
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([createExportedDeclaration(decl)]);
    mockResolveJsdocSummaryText.mockReturnValue('');
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), '--json'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    expect(process.exitCode).toBe(1);
    process.exitCode = origExitCode;
  });

  it('handles errors via fail() with Error', async () => {
    mockLoadExportedTypeScriptDeclarations.mockRejectedValue(new Error('Scan failed'));
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), '--json'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    expect(mockFail).toHaveBeenCalledWith('Scan failed', true);
  });

  it('handles errors via fail() with non-Error', async () => {
    mockLoadExportedTypeScriptDeclarations.mockRejectedValue('string error');
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs')];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    expect(mockFail).toHaveBeenCalledWith('string error', false);
  });

  it('passes sourcePaths when --source is provided', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), '--source', 'src/foo.ts', '--source', 'src/bar.ts'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    const callArg = mockLoadExportedTypeScriptDeclarations.mock.calls[0][0];
    expect(callArg.sourcePaths).toEqual(['src/foo.ts', 'src/bar.ts']);
  });

  it('passes positional args as sourcePaths', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), 'src/baz.ts'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    const callArg = mockLoadExportedTypeScriptDeclarations.mock.calls[0][0];
    expect(callArg.sourcePaths).toEqual(['src/baz.ts']);
  });

  it('formatter callback returns success message', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), '--json'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    const formatter = mockWriteJsonOrText.mock.calls[0][2];
    expect(formatter({ pass: true, evidence: [] })).toBe('Code quality scan passed.');
  });

  it('formatter callback returns failure message', async () => {
    mockLoadExportedTypeScriptDeclarations.mockResolvedValue([]);
    process.argv = [process.argv[0], path.resolve('rag-index/code-quality-scanner.mjs'), '--json'];
    jest.resetModules();
    await import('./code-quality-scanner.mjs');
    const formatter = mockWriteJsonOrText.mock.calls[0][2];
    expect(formatter({ pass: false, evidence: [{ issue: 'test' }] })).toContain('failed');
  });
});