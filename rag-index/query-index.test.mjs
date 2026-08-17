/**
 * @module query-index.test
 * @description 100% coverage tests for query-index.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'query-index.mjs');

// Shared mock functions
const mockCreateClient = jest.fn();
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();

jest.unstable_mockModule('@libsql/client', () => ({
  createClient: mockCreateClient,
}));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: '/fake/db.sqlite',
}));

const { querySemanticIndex } = await import('./query-index.mjs');

beforeEach(() => {
  jest.clearAllMocks();
  mockCreateClient.mockReset();
});

// ---------------------------------------------------------------------------
// querySemanticIndex: validation
// ---------------------------------------------------------------------------

describe('query-index: querySemanticIndex validation', () => {
  it('throws on empty query', async () => {
    await expect(querySemanticIndex({ query: '' })).rejects.toThrow(
      'A query is required',
    );
  });

  it('throws on whitespace-only query', async () => {
    await expect(querySemanticIndex({ query: '   ' })).rejects.toThrow(
      'A query is required',
    );
  });

  it('throws on undefined query', async () => {
    await expect(querySemanticIndex({})).rejects.toThrow(
      'A query is required',
    );
  });

  it('throws when called with no arguments (default options)', async () => {
    await expect(querySemanticIndex()).rejects.toThrow(
      'A query is required',
    );
  });
});

// ---------------------------------------------------------------------------
// querySemanticIndex: with provided client
// ---------------------------------------------------------------------------

describe('query-index: querySemanticIndex with client', () => {
  it('queries with client and no family filter', async () => {
    const mockClient = {
      async execute({ sql, args }) {
        return {
          rows: [
            {
              file_path: 'src/test.ts',
              doc_family: 'ts-source',
              heading_path: '# Test',
              body_text: 'content',
              score: -2,
            },
          ],
        };
      },
    };
    const result = await querySemanticIndex({
      query: 'neat',
      client: mockClient,
      limit: 5,
    });
    expect(result).toHaveLength(1);
    expect(result[0].file_path).toBe('src/test.ts');
  });

  it('queries with client and family filter', async () => {
    const mockClient = {
      async execute({ sql, args }) {
        expect(args).toContain('readme');
        return {
          rows: [
            {
              file_path: 'src/README.md',
              doc_family: 'readme',
              heading_path: '# README',
              body_text: 'content',
              score: -1,
            },
          ],
        };
      },
    };
    const result = await querySemanticIndex({
      query: 'neat',
      client: mockClient,
      family: 'readme',
    });
    expect(result).toHaveLength(1);
  });

  it('uses default limit of 10 when not provided', async () => {
    const mockClient = {
      async execute({ args }) {
        expect(args[args.length - 1]).toBe(10);
        return { rows: [] };
      },
    };
    await querySemanticIndex({ query: 'test', client: mockClient });
  });

  it('clamps limit to minimum 1', async () => {
    const mockClient = {
      async execute({ args }) {
        expect(args[args.length - 1]).toBe(1);
        return { rows: [] };
      },
    };
    await querySemanticIndex({ query: 'test', client: mockClient, limit: -5 });
  });
});

// ---------------------------------------------------------------------------
// querySemanticIndex: without client (creates from @libsql/client)
// ---------------------------------------------------------------------------

describe('query-index: querySemanticIndex without client', () => {
  it('creates client and closes after query', async () => {
    const mockClose = jest.fn();
    mockCreateClient.mockReturnValue({
      async execute({ sql, args }) {
        return {
          rows: [
            {
              file_path: 'test.md',
              doc_family: 'readme',
              heading_path: 'H',
              body_text: 'B',
              score: -1,
            },
          ],
        };
      },
      close: mockClose,
    });
    const result = await querySemanticIndex({
      query: 'test',
      databasePath: '/custom/db.sqlite',
    });
    expect(result).toHaveLength(1);
    expect(mockClose).toHaveBeenCalledTimes(1);
    expect(mockCreateClient).toHaveBeenCalledTimes(1);
  });

  it('creates client with family filter', async () => {
    const mockClose = jest.fn();
    mockCreateClient.mockReturnValue({
      async execute({ args }) {
        expect(args).toContain('plan');
        return { rows: [] };
      },
      close: mockClose,
    });
    await querySemanticIndex({
      query: 'test',
      family: 'plan',
    });
    expect(mockClose).toHaveBeenCalledTimes(1);
  });

  it('uses defaultDatabasePath when no databasePath provided', async () => {
    mockCreateClient.mockReturnValue({
      async execute() {
        return { rows: [] };
      },
      close: jest.fn(),
    });
    await querySemanticIndex({ query: 'test' });
    expect(mockCreateClient).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('query-index: main()', () => {
  const origArgv1 = process.argv[1];

  afterEach(() => {
    process.argv[1] = origArgv1;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await import(`./query-index.mjs?cache-bust=${Date.now()}-help`);
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText).not.toHaveBeenCalled();
  });

  it('runs query and writes text output with positional args', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'neat',
      _: [],
      json: false,
    });
    mockCreateClient.mockReturnValue({
      async execute() {
        return {
          rows: [
            {
              file_path: 'test.ts',
              doc_family: 'ts-source',
              heading_path: '# Test',
              body_text: 'B',
              score: -1,
            },
          ],
        };
      },
      close: jest.fn(),
    });
    process.argv[1] = sourceFilePath;
    await import(`./query-index.mjs?cache-bust=${Date.now()}-text`);
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockFail).not.toHaveBeenCalled();
  });

  it('runs query with json output', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'neat',
      _: [],
      json: true,
      family: 'readme',
      limit: 5,
    });
    mockCreateClient.mockReturnValue({
      async execute() {
        return { rows: [] };
      },
      close: jest.fn(),
    });
    process.argv[1] = sourceFilePath;
    await import(`./query-index.mjs?cache-bust=${Date.now()}-json`);
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    const jsonFlag = mockWriteJsonOrText.mock.calls[0][1];
    expect(jsonFlag).toBe(true);
  });

  it('calls fail on Error thrown by query', async () => {
    mockParseCliArgs.mockReturnValue({
      query: '',
      _: [],
      json: false,
    });
    process.argv[1] = sourceFilePath;
    await import(`./query-index.mjs?cache-bust=${Date.now()}-err`);
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toContain('A query is required');
  });

  it('calls fail with String(error) for non-Error throws', async () => {
    mockParseCliArgs.mockReturnValue({
      query: 'test',
      _: [],
      json: true,
    });
    mockCreateClient.mockReturnValue({
      async execute() {
        throw 'string error';
      },
      close: jest.fn(),
    });
    process.argv[1] = sourceFilePath;
    await import(`./query-index.mjs?cache-bust=${Date.now()}-strerr`);
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });

  it('uses positional args when --query is not provided', async () => {
    mockParseCliArgs.mockReturnValue({
      _: ['positional', 'query'],
      json: false,
    });
    mockCreateClient.mockReturnValue({
      async execute() {
        return { rows: [] };
      },
      close: jest.fn(),
    });
    process.argv[1] = sourceFilePath;
    await import(`./query-index.mjs?cache-bust=${Date.now()}-positional`);
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockFail).not.toHaveBeenCalled();
  });
});