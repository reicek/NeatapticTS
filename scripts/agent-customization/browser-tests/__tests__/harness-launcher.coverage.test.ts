import http from 'node:http';
import path from 'node:path';
import { launchLocalServer } from '../harness-launcher';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');

/**
 * Make an HTTP request to a local server and resolve with the status code,
 * content-type header, and body text.
 */
function httpRequest(
  port: number,
  requestPath: string,
): Promise<{ status: number; contentType: string; body: string }> {
  return new Promise((resolve, reject) => {
    const req = http.get(
      {
        hostname: '127.0.0.1',
        port,
        path: requestPath,
      },
      (res) => {
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (chunk: string) => {
          body += chunk;
        });
        res.on('end', () => {
          resolve({
            status: res.statusCode ?? 0,
            contentType: res.headers['content-type'] ?? '',
            body,
          });
        });
      },
    );
    req.on('error', reject);
  });
}

describe('harness-launcher serveStaticFile coverage', () => {
  let handle: {
    serverUrl: string;
    scenarioUrl: string;
    teardown: () => Promise<void>;
  } | null = null;

  afterEach(async () => {
    if (handle) {
      await handle.teardown();
      handle = null;
    }
  });

  it('serves an existing HTML file with 200 and text/html content type', async () => {
    handle = await launchLocalServer({ cwd: REPO_ROOT, port: 18090 });

    const response = await httpRequest(
      18090,
      '/docs/browser-tests/webgpu-inference-smoke.html',
    );

    expect(response.status).toBe(200);
    expect(response.contentType).toBe('text/html');
  });

  it('returns 404 for a non-existent file', async () => {
    handle = await launchLocalServer({ cwd: REPO_ROOT, port: 18091 });

    const response = await httpRequest(18091, '/nonexistent-file-12345.txt');

    expect(response.status).toBe(404);
  });

  it('returns 403 for a path that resolves outside the server root', async () => {
    // Using the drive root (e.g. C:\) as cwd makes the startsWith check fail:
    // path.normalize("C:\") + path.sep = "C:\\" (double separator), but
    // path.join("C:\", "/foo") = "C:\foo" which does not start with "C:\\".
    const driveRoot = path.parse(process.cwd()).root;
    handle = await launchLocalServer({ cwd: driveRoot, port: 18092 });

    const response = await httpRequest(18092, '/foo');

    expect(response.status).toBe(403);
  });
});

describe('harness-launcher error paths', () => {
  it('rejects when the port is already in use', async () => {
    const first = await launchLocalServer({
      cwd: REPO_ROOT,
      port: 18093,
    });

    try {
      await expect(
        launchLocalServer({
          cwd: REPO_ROOT,
          port: 18093,
          startTimeoutMs: 5000,
        }),
      ).rejects.toThrow(/failed to start/);
    } finally {
      await first.teardown();
    }
  });
});

describe('harness-launcher mock-based branch coverage', () => {
  it('rejects with a timeout when the server does not become ready (no closeAllConnections)', async () => {
    await jest.isolateModulesAsync(async () => {
      const mockServer = {
        on: jest.fn(),
        listen: jest.fn(),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        // no closeAllConnections — covers the ?. undefined branch in fail()
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn(() => mockServer),
      }));

      const { launchLocalServer: mockLaunch } =
        await import('../harness-launcher');

      await expect(
        mockLaunch({
          cwd: REPO_ROOT,
          port: 18094,
          startTimeoutMs: 50,
        }),
      ).rejects.toThrow(/did not become ready/);
    });
  });

  it('rejects with a timeout when the server does not become ready (with closeAllConnections)', async () => {
    await jest.isolateModulesAsync(async () => {
      const mockServer = {
        on: jest.fn(),
        listen: jest.fn(),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        closeAllConnections: jest.fn(),
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn(() => mockServer),
      }));

      const { launchLocalServer: mockLaunch } =
        await import('../harness-launcher');

      await expect(
        mockLaunch({
          cwd: REPO_ROOT,
          port: 18095,
          startTimeoutMs: 50,
        }),
      ).rejects.toThrow(/did not become ready/);

      expect(mockServer.closeAllConnections).toHaveBeenCalled();
    });
  });

  it('starts and tears down a mock server without closeAllConnections', async () => {
    await jest.isolateModulesAsync(async () => {
      const mockServer = {
        on: jest.fn(),
        listen: jest.fn((_port: number, _host: string, cb: () => void) => cb()),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        // no closeAllConnections — covers the ?. undefined branch in teardown
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn(() => mockServer),
      }));

      const { launchLocalServer: mockLaunch } =
        await import('../harness-launcher');
      const result = await mockLaunch({
        cwd: REPO_ROOT,
        port: 18096,
      });

      expect(result.serverUrl).toBe('http://localhost:18096');
      await result.teardown();
      expect(mockServer.close).toHaveBeenCalled();
    });
  });

  it('covers the listen callback settled guard when listen fires twice', async () => {
    await jest.isolateModulesAsync(async () => {
      const mockServer = {
        on: jest.fn(),
        listen: jest.fn((_port: number, _host: string, cb: () => void) => {
          cb();
          cb();
        }),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        closeAllConnections: jest.fn(),
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn(() => mockServer),
      }));

      const { launchLocalServer: mockLaunch } =
        await import('../harness-launcher');
      const result = await mockLaunch({
        cwd: REPO_ROOT,
        port: 18097,
      });

      await result.teardown();
    });
  });

  it('covers the error handler settled guard when error fires after start', async () => {
    await jest.isolateModulesAsync(async () => {
      let errorHandler: ((err: Error) => void) | null = null;
      const mockServer = {
        on: jest.fn((event: string, handler: (err: Error) => void) => {
          if (event === 'error') errorHandler = handler;
        }),
        listen: jest.fn((_port: number, _host: string, cb: () => void) => {
          cb();
          process.nextTick(() => {
            if (errorHandler) errorHandler(new Error('late error'));
          });
        }),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        closeAllConnections: jest.fn(),
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn(() => mockServer),
      }));

      const { launchLocalServer: mockLaunch } =
        await import('../harness-launcher');
      const result = await mockLaunch({
        cwd: REPO_ROOT,
        port: 18098,
      });

      await new Promise((resolve) => setTimeout(resolve, 50));
      await result.teardown();
    });
  });

  it('covers the fail settled guard when timeout fires after successful start', async () => {
    await jest.isolateModulesAsync(async () => {
      const mockServer = {
        on: jest.fn(),
        listen: jest.fn((_port: number, _host: string, cb: () => void) => cb()),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        closeAllConnections: jest.fn(),
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn(() => mockServer),
      }));

      const clearTimeoutSpy = jest
        .spyOn(globalThis, 'clearTimeout')
        .mockImplementation(() => {});

      const { launchLocalServer: mockLaunch } =
        await import('../harness-launcher');
      const result = await mockLaunch({
        cwd: REPO_ROOT,
        port: 18099,
        startTimeoutMs: 30,
      });

      await new Promise((resolve) => setTimeout(resolve, 60));
      clearTimeoutSpy.mockRestore();
      await result.teardown();
    });
  });

  it('uses DEFAULT_SERVER_PORT when port is not specified', async () => {
    await jest.isolateModulesAsync(async () => {
      const mockServer = {
        on: jest.fn(),
        listen: jest.fn((_port: number, _host: string, cb: () => void) => cb()),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        closeAllConnections: jest.fn(),
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn(() => mockServer),
      }));

      const { launchLocalServer: mockLaunch, DEFAULT_SERVER_PORT } =
        await import('../harness-launcher');
      const result = await mockLaunch({ cwd: REPO_ROOT });

      expect(mockServer.listen).toHaveBeenCalledWith(
        DEFAULT_SERVER_PORT,
        '127.0.0.1',
        expect.any(Function),
      );
      expect(result.serverUrl).toBe(`http://localhost:${DEFAULT_SERVER_PORT}`);
      await result.teardown();
    });
  });

  it('defaults request.url to "/" when url is undefined (403 via drive root)', async () => {
    await jest.isolateModulesAsync(async () => {
      let requestHandler: ((req: any, res: any) => void) | null = null;
      const mockServer = {
        on: jest.fn(),
        listen: jest.fn((_port: number, _host: string, cb: () => void) => cb()),
        close: jest.fn((cb?: () => void) => {
          if (cb) cb();
        }),
        closeAllConnections: jest.fn(),
      };
      jest.doMock('node:http', () => ({
        createServer: jest.fn((handler: any) => {
          requestHandler = handler;
          return mockServer;
        }),
      }));
      jest.doMock('node:fs/promises', () => ({
        readFile: jest.fn(async () => Buffer.from('mock')),
      }));

      const { launchLocalServer: mockLaunch } =
        await import('../harness-launcher');
      const driveRoot = path.parse(process.cwd()).root;
      const result = await mockLaunch({ cwd: driveRoot, port: 18100 });

      // Call the request handler with url = undefined to cover the `?? '/'` branch
      const mockRes = {
        writeHead: jest.fn(),
        end: jest.fn(),
      };
      requestHandler!({ url: undefined }, mockRes);

      // With drive root, even '/' resolves outside root → 403
      expect(mockRes.writeHead).toHaveBeenCalledWith(403);
      expect(mockRes.end).toHaveBeenCalledWith('Forbidden');

      await result.teardown();
    });
  });
});
