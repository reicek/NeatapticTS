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
      else if (args[i] === '--dimension') result.dimension = args[++i];
    }
    return result;
  },
}));

jest.unstable_mockModule('./embed-index.mjs', () => ({
  DEFAULT_MODEL_DIRECTORY: '/fake/models',
  DEFAULT_MODEL_ID: 'fake-model-id',
  default: { DEFAULT_MODEL_DIRECTORY: '/fake/models', DEFAULT_MODEL_ID: 'fake-model-id' },
}));

const { downloadModelAssets } = await import('./download-model.mjs');

const modelBuffer = Buffer.from('fake-model-onnx-data');
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

describe('downloadModelAssets', () => {
  it('downloads all assets with provided expectedSha256', async () => {
    const fetchCalls = [];
    global.fetch = jest.fn(async (url) => {
      fetchCalls.push(url);
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: 'abc' } }] });
      }
      if (url.includes('model.onnx')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    const result = await downloadModelAssets({
      expectedSha256: modelSha256,
      modelDirectory: '/fake/models',
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

    const result = await downloadModelAssets({ modelDirectory: '/fake/models' });
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

    const result = await downloadModelAssets({ modelDirectory: '/fake/models' });
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
      downloadModelAssets({ modelDirectory: '/fake/models' }),
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
      downloadModelAssets({ modelDirectory: '/fake/models' }),
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
      downloadModelAssets({ modelDirectory: '/fake/models' }),
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
      // tokenizer.json returns error
      return mockResponse(null, { ok: false, status: 500, statusText: 'Server Error' });
    });

    await expect(
      downloadModelAssets({ modelDirectory: '/fake/models' }),
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
      downloadModelAssets({ modelDirectory: '/fake/models' }),
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

    const result = await downloadModelAssets({ modelDirectory: '/fake/models' });
    expect(result.assets).toHaveLength(4);
  });

  it('throws last error after all retries fail', async () => {
    global.fetch = jest.fn(async () => {
      throw new Error('Persistent network error');
    });

    await expect(
      downloadModelAssets({ modelDirectory: '/fake/models' }),
    ).rejects.toThrow('Persistent network error');
  });

  it('wraps non-Error thrown by fetch', async () => {
    global.fetch = jest.fn(async () => {
      throw 'string error';
    });

    await expect(
      downloadModelAssets({ modelDirectory: '/fake/models' }),
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
      downloadModelAssets({ modelDirectory: '/fake/models' }),
    ).rejects.toThrow(/Failed to fetch/);
  });

  it('uses custom repositoryId, modelId, and dimension', async () => {
    global.fetch = jest.fn(async (url) => {
      if (url.includes('/api/models/')) {
        return mockResponse({ siblings: [{ rfilename: 'onnx/model.onnx', lfs: { oid: modelSha256 } }] });
      }
      if (url.includes('model.onnx') && url.includes('/resolve/')) {
        return mockResponse(modelBuffer, { url });
      }
      return mockResponse(tokenizerBuffer, { url });
    });

    const result = await downloadModelAssets({
      modelDirectory: '/fake/models',
      repositoryId: 'custom/repo',
      modelId: 'custom-model',
      dimension: 256,
    });
    expect(result.modelId).toBe('custom-model');
    expect(result.dimension).toBe(256);
  });
});

describe('main() CLI entry point', () => {
  it('prints help when --help flag is provided', async () => {
    process.argv = [process.argv[0], path.resolve('rag-index/download-model.mjs'), '--help'];
    jest.resetModules();
    await import('./download-model.mjs');
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
    process.argv = [process.argv[0], path.resolve('rag-index/download-model.mjs'), '--json'];
    jest.resetModules();
    await import('./download-model.mjs');
    expect(mockWriteJsonOrText).toHaveBeenCalled();
  });

  it('handles errors via fail()', async () => {
    global.fetch = jest.fn(async () => { throw new Error('Network down'); });
    process.argv = [process.argv[0], path.resolve('rag-index/download-model.mjs'), '--json'];
    jest.resetModules();
    await import('./download-model.mjs');
    expect(mockFail).toHaveBeenCalledWith('Network down', true);
  });

  it('handles non-Error exceptions in fail()', async () => {
    global.fetch = jest.fn(async () => { throw 'string error'; });
    process.argv = [process.argv[0], path.resolve('rag-index/download-model.mjs')];
    jest.resetModules();
    await import('./download-model.mjs');
    expect(mockFail).toHaveBeenCalledWith('Failed to fetch https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2', false);
  });
});