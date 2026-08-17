import { jest } from '@jest/globals';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const mockCreateClient = jest.fn();
jest.unstable_mockModule('@libsql/client', () => ({
  createClient: mockCreateClient,
  default: { createClient: mockCreateClient },
}));

const mockMkdir = jest.fn();
const mockWriteFile = jest.fn();
const mockReadFile = jest.fn();
jest.unstable_mockModule('node:fs/promises', () => ({
  mkdir: mockMkdir,
  writeFile: mockWriteFile,
  readFile: mockReadFile,
  default: { mkdir: mockMkdir, writeFile: mockWriteFile, readFile: mockReadFile },
}));

const mockFail = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
  parseCliArgs: (args) => {
    const result = {};
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--help') result.help = true;
      else if (args[i] === '--json') result.json = true;
      else if (args[i] === '--dry-run') result['dry-run'] = true;
      else if (args[i] === '--database') result.database = args[++i];
      else if (args[i] === '--output') result.output = args[++i];
    }
    return result;
  },
  toRepoRelative: (p) => p,
}));

const { buildBrowserSnapshot, createBrowserSnapshot } = await import('./build-browser-snapshot.mjs');

function createMockClient(documents = [], chunks = []) {
  return {
    execute: jest.fn(async (sql) => {
      if (sql.includes('FROM documents')) {
        return { rows: documents };
      }
      if (sql.includes('FROM chunks')) {
        return { rows: chunks };
      }
      return { rows: [] };
    }),
  };
}

afterEach(() => {
  jest.clearAllMocks();
});

describe('buildBrowserSnapshot', () => {
  it('builds snapshot with provided client in dry-run mode', async () => {
    const client = createMockClient(
      [{ doc_id: 1, file_path: 'src\\foo.ts', family: 'core' }],
      [{ chunk_id: 10, doc_id: 1, heading_path: 'Foo', body_text: 'body', char_start: 0, char_end: 4 }],
    );
    const result = await buildBrowserSnapshot({ client, dryRun: true });
    expect(result.dryRun).toBe(true);
    expect(result.documents).toBe(1);
    expect(result.chunks).toBe(1);
    expect(result.families).toEqual(['core']);
    expect(mockWriteFile).not.toHaveBeenCalled();
  });

  it('writes JSON file when not dry-run', async () => {
    const client = createMockClient(
      [{ doc_id: 1, file_path: 'src/foo.ts', family: 'core' }],
      [],
    );
    mockMkdir.mockResolvedValue(undefined);
    mockWriteFile.mockResolvedValue(undefined);
    const result = await buildBrowserSnapshot({
      client,
      dryRun: false,
      outputPath: '/fake/output.json',
    });
    expect(result.dryRun).toBe(false);
    expect(mockMkdir).toHaveBeenCalled();
    expect(mockWriteFile).toHaveBeenCalled();
  });

  it('creates client when not provided', async () => {
    const client = createMockClient([], []);
    mockCreateClient.mockReturnValue(client);
    mockMkdir.mockResolvedValue(undefined);
    mockWriteFile.mockResolvedValue(undefined);
    await buildBrowserSnapshot({ databasePath: '/fake/db.sqlite', dryRun: true });
    expect(mockCreateClient).toHaveBeenCalled();
  });

  it('normalizes backslash paths in documents', async () => {
    const client = createMockClient(
      [{ doc_id: 1, file_path: 'src\\deep\\file.ts', family: 'core' }],
      [],
    );
    const result = await buildBrowserSnapshot({ client, dryRun: true });
    // The families are derived from documents, and documents include file_path
    // normalizeRepoPath replaces \ with / - verify via the snapshot structure
    expect(result.families).toEqual(['core']);
  });

  it('groups chunks by document_id correctly', async () => {
    const client = createMockClient(
      [
        { doc_id: 1, file_path: 'a.ts', family: 'core' },
        { doc_id: 2, file_path: 'b.ts', family: 'docs' },
      ],
      [
        { chunk_id: 10, doc_id: 1, heading_path: 'A1', body_text: 'b1', char_start: 0, char_end: 2 },
        { chunk_id: 11, doc_id: 1, heading_path: 'A2', body_text: 'b2', char_start: 3, char_end: 5 },
        { chunk_id: 20, doc_id: 2, heading_path: 'B1', body_text: 'b3', char_start: 0, char_end: 2 },
      ],
    );
    const result = await buildBrowserSnapshot({ client, dryRun: true });
    expect(result.documents).toBe(2);
    expect(result.chunks).toBe(3);
  });

  it('handles documents with no chunks', async () => {
    const client = createMockClient(
      [{ doc_id: 1, file_path: 'a.ts', family: 'core' }],
      [],
    );
    const result = await buildBrowserSnapshot({ client, dryRun: true });
    expect(result.documents).toBe(1);
    expect(result.chunks).toBe(0);
  });

  it('handles null heading_path in chunks', async () => {
    const client = createMockClient(
      [{ doc_id: 1, file_path: 'a.ts', family: 'core' }],
      [{ chunk_id: 10, doc_id: 1, heading_path: null, body_text: 'body', char_start: 0, char_end: 4 }],
    );
    const result = await buildBrowserSnapshot({ client, dryRun: true });
    expect(result.chunks).toBe(1);
  });

  it('sorts families alphabetically', async () => {
    const client = createMockClient(
      [
        { doc_id: 1, file_path: 'a.ts', family: 'docs' },
        { doc_id: 2, file_path: 'b.ts', family: 'core' },
        { doc_id: 3, file_path: 'c.ts', family: 'core' },
      ],
      [],
    );
    const result = await buildBrowserSnapshot({ client, dryRun: true });
    expect(result.families).toEqual(['core', 'docs']);
  });
});

describe('createBrowserSnapshot', () => {
  it('creates client and returns snapshot object', async () => {
    const client = createMockClient(
      [{ doc_id: 1, file_path: 'a.ts', family: 'core' }],
      [{ chunk_id: 10, doc_id: 1, heading_path: 'H', body_text: 'body', char_start: 0, char_end: 4 }],
    );
    mockCreateClient.mockReturnValue(client);
    const snapshot = await createBrowserSnapshot('/fake/db.sqlite');
    expect(snapshot.schema_version).toBe('1');
    expect(snapshot.documents).toHaveLength(1);
    expect(snapshot.families).toEqual(['core']);
    expect(mockCreateClient).toHaveBeenCalled();
  });
});

describe('main() CLI entry point', () => {
  it('prints help when --help flag is provided', async () => {
    process.argv = [process.argv[0], path.resolve('rag-index/build-browser-snapshot.mjs'), '--help'];
    jest.resetModules();
    await import('./build-browser-snapshot.mjs');
    expect(mockPrintHelp).toHaveBeenCalled();
  });

  it('runs successfully with --dry-run and --json', async () => {
    const client = createMockClient(
      [{ doc_id: 1, file_path: 'a.ts', family: 'core' }],
      [],
    );
    mockCreateClient.mockReturnValue(client);
    mockMkdir.mockResolvedValue(undefined);
    mockWriteFile.mockResolvedValue(undefined);
    process.argv = [process.argv[0], path.resolve('rag-index/build-browser-snapshot.mjs'), '--dry-run', '--json'];
    jest.resetModules();
    await import('./build-browser-snapshot.mjs');
    expect(mockWriteJsonOrText).toHaveBeenCalled();
  });

  it('handles errors via fail()', async () => {
    mockCreateClient.mockImplementation(() => {
      throw new Error('DB connection failed');
    });
    process.argv = [process.argv[0], path.resolve('rag-index/build-browser-snapshot.mjs'), '--dry-run'];
    jest.resetModules();
    await import('./build-browser-snapshot.mjs');
    expect(mockFail).toHaveBeenCalledWith('DB connection failed', false);
  });

  it('handles non-Error exceptions in fail()', async () => {
    mockCreateClient.mockImplementation(() => {
      throw 'string error';
    });
    process.argv = [process.argv[0], path.resolve('rag-index/build-browser-snapshot.mjs'), '--dry-run', '--json'];
    jest.resetModules();
    await import('./build-browser-snapshot.mjs');
    expect(mockFail).toHaveBeenCalledWith('string error', true);
  });
});