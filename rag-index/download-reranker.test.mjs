import { jest } from '@jest/globals';
import { createHash } from 'node:crypto';
import path from 'node:path';

const mockMkdir = jest.fn().mockResolvedValue(undefined);
const mockWriteFile = jest.fn().mockResolvedValue(undefined);
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
      else if (args[i] === '--model-directory') result['model-directory'] = args[++i];
      else if (args[i] === '--model-id') result['model-id'] = args[++i];
      else if (args[i] === '--repository-id') result['repository-id'] = args[++i];
      else if (args[i] === '--expected-sha256') result['expected-sha256'] = args[++i];
      else if (args[i] === '--max-sequence-length') result['max-sequence-length'] = args[++i];
    }
    return result;
  },
}));

jest.unstable_mockModule('./reranker-readiness.mjs', () => ({
  DEFAULT_RERANKER_MODEL_DIRECTORY: '/fake/reranker-models',
  default: { DEFAULT_RERANKER_MODEL_DIRECTORY: '/fake/reranker-models' },
}));

const {
  downloadRerankerAssets,
  DEFAULT_RERANKER_REPOSITORY_ID,
  DEFAULT_RERANKER_MODEL_ID,
  DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH,
  RERANKER_ASSETS,
  RERANKER_META_FIELDS,
} = await import('./download-reranker.mjs');

const modelBuffer = Buffer.from('fake-reranker-onnx-data');
const modelSha256 = createHash('sha256').update(modelBuffer).digest('hex');
const tokenizerBuffer = Buffer.from('{"tokenizer":true}');

function mockResponse(body, { ok = true, status = 200, statusText = 'OK', url = 'https://huggingface.co/test' } = {}) {
  return {
    ok,
    status,
    statusText,
    url,
    json: async () => body,
    arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength),
    text: async () => body,
  };
}

afterEach(() => {
  jest.clearAllMocks();
  global.fetch = undefined;
});

describe('exports', () => {
  it('exports constants', () => {
    expect(DEFAULT_RERANKER_REPOSITORY_ID).toBe('cross-encoder/ms-marco-MiniLM-L-6-v2');
    expect(DEFAULT_RERANKER_MODEL_ID).toBe('cross-encoder/ms-marco-MiniLM-L-6-v2');
    expect(DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH).toBe(512);
  });

  it('exports RERANKER_ASSETS with 4 assets', () => {
    expect(RERANKER_ASSETS).toHaveLength(4);
    expect(RERANKER_ASSETS[0].verifySha256).toBe(true);
    expect(RERANKER_ASSETS[1].verifySha256).toBe(false);
  });

  it('exports RERANKER_META_FIELDS without dimension', () => {
    expect(RERANKER_META_FIELDS).not.toContain('dimension');
    expect(RERANKER_META_FIELDS).toContain('max_sequence_length');
  });
});

describe('downloadRerankerAssets', () => {
  it('downloads all assets with provided expectedSha256', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: 'abc' } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    const result = await downloadRerankerAssets({
      expectedSha256: modelSha256,
      modelDirectory: '/fake/reranker-models',
    });

    expect(result.assets).toHaveLength(4);
    expect(result.modelSha256).toBe(modelSha256);
    expect(mockWriteFile).toHaveBeenCalledTimes(5); // 4 assets + model-meta.json
  });

  it('resolves SHA-256 from repository metadata siblings', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: modelSha256 } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    const result = await downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' });
    expect(result.modelSha256).toBe(modelSha256);
  });

  it('resolves SHA-256 from LFS pointer file when metadata lacks it', async () => {
    const pointerText = `version https://git-lfs.github.com/spec/v1\noid sha256:${modelSha256}\nsize 123\n`;
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [] });
      }
      if (url.includes('/raw/main/onnx/model.onnx')) {
        return mockResponse(pointerText, { url });
      }
      if (url.includes('/resolve/main/onnx/model.onnx')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    const result = await downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' });
    expect(result.modelSha256).toBe(modelSha256);
  });

  it('throws on SHA-256 mismatch', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: 'wrongsha' } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow(/SHA-256 mismatch/);
  });

  it('throws when unable to resolve expected SHA-256', async () => {
    const emptyPointer = 'version https://git-lfs.github.com/spec/v1\n';
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [] });
      }
      if (url.includes('/raw/main/onnx/model.onnx')) {
        return mockResponse(emptyPointer, { url });
      }
      if (url.includes('/resolve/main/onnx/model.onnx')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow(/Unable to resolve the expected SHA-256/);
  });

  it('throws when fetchJson gets non-ok response', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse(null, { ok: false, status: 404, statusText: 'Not Found' });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow(/Failed to fetch/);
  });

  it('throws when downloadBinary gets non-ok response', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: modelSha256 } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(null, { ok: false, status: 500, statusText: 'Server Error' });
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow(/Failed to download/);
  });

  it('throws when downloadBinaryWithMetadata gets non-ok response', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: modelSha256 } }] });
      }
      // model.onnx returns non-ok → downloadBinaryWithMetadata throws
      if (url.includes('model.onnx')) {
        return mockResponse(null, { ok: false, status: 403, statusText: 'Forbidden' });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow(/Failed to download/);
  });

  it('retries fetch on transient failure then succeeds', async () => {
    let apiCallCount = 0;
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        apiCallCount++;
        if (apiCallCount === 1) throw new Error('Transient network error');
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: modelSha256 } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    const result = await downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' });
    expect(result.assets).toHaveLength(4);
  });

  it('throws last error after all retries fail', async () => {
    global.fetch = jest.fn(async () => {
      throw new Error('Persistent network error');
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow('Persistent network error');
  });

  it('wraps non-Error thrown by fetch', async () => {
    global.fetch = jest.fn(async () => {
      throw 'string error';
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow(/Failed to fetch/);
  });

  it('throws when resolveExpectedModelSha256FromPointer gets non-ok response', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [] });
      }
      if (url.includes('/raw/main/onnx/model.onnx')) {
        return mockResponse(null, { ok: false, status: 404, statusText: 'Not Found' });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    await expect(
      downloadRerankerAssets({ modelDirectory: '/fake/reranker-models' }),
    ).rejects.toThrow(/Failed to fetch/);
  });

  it('uses custom repositoryId, modelId, and maxSequenceLength', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: modelSha256 } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    const result = await downloadRerankerAssets({
      modelDirectory: '/fake/reranker-models',
      repositoryId: 'custom/reranker',
      modelId: 'custom-reranker',
      maxSequenceLength: 256,
    });
    expect(result.modelId).toBe('custom-reranker');
    expect(result.maxSequenceLength).toBe(256);
  });
});

describe('main() CLI entry point', () => {
  it('prints help when --help flag is provided', async () => {
    process.argv = [process.argv[0], path.resolve('rag-index/download-reranker.mjs'), '--help'];
    jest.resetModules();
    await import('./download-reranker.mjs');
    expect(mockPrintHelp).toHaveBeenCalled();
  });

  it('runs successfully with --json', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: modelSha256 } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });
    process.argv = [process.argv[0], path.resolve('rag-index/download-reranker.mjs'), '--json'];
    jest.resetModules();
    await import('./download-reranker.mjs');
    expect(mockWriteJsonOrText).toHaveBeenCalled();
  });

  it('handles errors via fail()', async () => {
    global.fetch = jest.fn(async () => { throw new Error('Network down'); });
    process.argv = [process.argv[0], path.resolve('rag-index/download-reranker.mjs'), '--json'];
    jest.resetModules();
    await import('./download-reranker.mjs');
    expect(mockFail).toHaveBeenCalledWith('Network down', true);
  });

  it('handles non-Error exceptions in fail()', async () => {
    global.fetch = jest.fn(async () => { throw 'string error'; });
    process.argv = [process.argv[0], path.resolve('rag-index/download-reranker.mjs')];
    jest.resetModules();
    await import('./download-reranker.mjs');
    expect(mockFail).toHaveBeenCalledWith('Failed to fetch https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L-6-v2', false);
  });
});