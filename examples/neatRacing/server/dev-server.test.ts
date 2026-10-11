/**
 * RED/green contract tests for the hardened Neat Racing development server
 * (spec section 8.2, AC-018; slice P1-S5-B, step 08).
 *
 * Per the step packet this file is authored FIRST: with
 * `examples/neatRacing/server/dev-server.ts` absent every assertion here
 * fails (the module import itself cannot resolve), and it must keep passing
 * once the implementation lands without weakening any hardening gate:
 *
 * - **S1 host/origin** â€” every request needs `Host: 127.0.0.1:<port>` or
 *   `localhost:<port>`; state-changing requests additionally require a
 *   matching `Origin` when one is present; `Access-Control-*` responses are
 *   never emitted. Devtools and hostile pages cannot save, rebuild, or
 *   validate models through this server.
 * - **S2 rebuild pinning** â€” the rebuild spawner pins the fixed bins script
 *   via `spawn(process.execPath, [<script>, <name>])` with `shell: false`
 *   (source-text asserted), the model name is gated against the server's own
 *   enumeration before every spawn, and concurrent rebuilds for one model
 *   coalesce latest-wins through `RebuildCoordinator`.
 * - **S3 body limits** â€î PUT/POST require `application/json` (`Content-Type`
 *   params tolerated), declared `Content-Length` and streamed bytes over
 *   `NEATRACING_MAX_SAVE_BODY_BYTES` are both rejected with 413.
 * - **S4 name shape** â€” `:name` segments are shape-checked against the
 *   landed store pattern before any path join, so traversal payloads
 *   (`../evil`, `%2e%2e`) are rejected 403; PUT never creates models that do
 *   not already exist in the enumeration.
 * - **S5 method allowlist + hygiene** â€” per-route method allowlists return
 *   405 for anything else, static serving is confined to declared prefixes
 *   with resolve containment (no traversal, no directory listing), and every
 *   error body is exactly `{status, code}` with no filesystem or stack data.
 * - **S6 backup rotation** â€” delegated to `model-store.utils.ts`
 *   (`bak1..3`); this suite asserts the server-facing round-trip: a second
 *   PUT replaces the file and the replaced version lands in `bak1`.
 *
 * Handler fixtures are supertest-style: a tiny `invokeHandler` builds
 * `DevRequestIntent` values against contexts wired to per-test temp model
 * directories and stub spawners, so the tests exercise the pure dispatch
 * â€” the single real-socket smoke test at the end proves the runtime wiring
 * (port binding, Host acceptance, page/static serving) against the actual
 * repository tree with pages disabled.
 */
import { afterEach, describe, expect, it } from '@jest/globals';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import type { ChildProcess } from 'node:child_process';
import {
  isAllowedDevServerHost,
  buildRebuildSpawnArgv,
  createDevServerContext,
  handleDevServerRequest,
  resolveDevServerPort,
  NEATRACING_DEV_SERVER_DEFAULT_PORT,
  NEATRACING_DEV_SERVER_PORT_ENV_NAME,
  NEATRACING_MAX_SAVE_BODY_BYTES,
  type DevModelStoreApi,
  type DevRequestIntent,
  type DevResponse,
  type DevServerContext,
} from './dev-server';
import {
  isVoxelModelName,
  listVoxelModelNames,
  readVoxelModelDocument,
  ModelStoreError,
  validateVoxelModelPayload,
  NEATRACING_MODEL_STORE_BACKUP_DIR_NAME,
  writeVoxelModelDocument,
} from './model-store.utils';

/** Fixture dims: 2-cubed grid keeps every document fixture tiny. */
const FIXTURE_DIMS: readonly [number, number, number] = [2, 2, 2];

/** Full-width empty row for a 2-wide document (run 2 of palette index 0). */
const EMPTY_ROW_2 = '0200';

/** Full-width colored row for a 2-wide document (run 2 of palette index 1). */
const COLORED_ROW_2 = '0201';

/** Fixture palette slot 0 (transparent) and slot 1 (light cyan-ish). */
const FIXTURE_PALETTE: readonly (readonly number[])[] = [
  [0, 0, 0, 0, 0],
  [200, 230, 255, 255, 64],
];

/** Test port every header/origin fixture is pinned to. */
const TEST_PORT = 9001;

/** Per-test temp directories created here and removed after each test. */
let tempRoot = '';

/** Temp models directory for the current test. */
let tempModelsDir = '';

/** Temp dev root (pages + assets) for the current test. */
let tempDevRoot = '';

/**
 * Real store API adapter wired by `beforeEach` from the landed store module
 * (`model-store.utils`, P1-S5-A), mirroring `loadDefaultDevServerContext`.
 */
let testStoreApi: DevModelStoreApi;

/**
 * Creates a minimal schema-valid 2-cubed document fixture.
 *
 * @param overridden - Field overrides merged over the base fixture.
 * @returns A plain document fixture accepted by the section 7.2 schema.
 */
function makeDocument(
  overridden: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    meta: { name: 'car-f1', dims: FIXTURE_DIMS, symmetric: false },
    palette: FIXTURE_PALETTE,
    slices: [
      [EMPTY_ROW_2, EMPTY_ROW_2],
      [COLORED_ROW_2, EMPTY_ROW_2],
    ],
    ...overridden,
  };
}

/**
 * Serializes a document fixture for use as a request body.
 *
 * @param overridden - Field overrides merged over the base fixture.
 * @returns The JSON-serialized fixture as a `Uint8Array`.
 */
function makeDocumentBytes(
  overridden: Record<string, unknown> = {},
): Uint8Array {
  return Buffer.from(JSON.stringify(makeDocument(overridden)), 'utf8');
}

/**
 * Builds a dev-server context wired to the temp fixtures with a trivial
 * no-op spawner, mirroring how the runtime context is created.
 *
 * @returns The test context.
 */
function makeTestContext(): DevServerContext {
  return createDevServerContext({
    port: TEST_PORT,
    modelsDir: tempModelsDir,
    devRoot: tempDevRoot,
    assetsDir: join(tempDevRoot, 'assets'),
    pages: { '/': 'index.html', '/editor/': 'editor/editor.html' },
    spawner: () => undefined,
    store: testStoreApi,
  });
}

/**
 * Supertest-style fixture: dispatches one request intent at the handler.
 *
 * @param context - Context created by `makeTestContext`.
 * @param method - HTTP method.
 * @param rawPath - Percent-encoded request path, exactly as received.
 * @param headers - Optional lowercase header record.
 * @param chunks - Optional body chunk list (defaults to none).
 * @returns The handler response.
 */
function invoke(
  context: DevServerContext,
  method: string,
  rawPath: string,
  headers: Record<string, string> = {},
  chunks: readonly Uint8Array[] = [],
): DevResponse {
  const intent: DevRequestIntent = {
    method,
    path: rawPath,
    headers,
    bodyChunks: chunks,
  };
  return handleDevServerRequest(context, intent);
}

/**
 * Builds the standard fixture headers with the accepted test host.
 *
 * @param overrides - Header overrides merged over the base record.
 * @returns A lowercase header record.
 */
function testHeaders(
  overrides: Record<string, string> = {},
): Record<string, string> {
  return {
    host: `127.0.0.1:${TEST_PORT}`,
    ...overrides,
  };
}

/**
 * PUT-fragments the given bytes as a declared save payload.
 *
 * @param bytes - The JSON payload bytes.
 * @param split - Whether to split the payload into two chunks.
 * @returns The chunk list plus the declared content length.
 */
function toSaveChunks(
  bytes: Uint8Array,
  split: boolean = false,
): { chunks: readonly Uint8Array[]; contentLength: number } {
  if (!split) {
    return { chunks: [bytes], contentLength: bytes.length };
  }
  const half = Math.floor(bytes.length / 2);
  return {
    chunks: [bytes.subarray(0, half), bytes.subarray(half)],
    contentLength: bytes.length,
  };
}

/**
 * Asserts a failure response carries exactly `{status, code}` and nothing
 * else (the S5 hygiene contract: no paths, no messages, no stacks).
 *
 * @param response - The response to assert.
 * @param status - Expected status.
 * @param code - Expected machine-readable code.
 */
function expectErrorBody(
  response: DevResponse,
  status: number,
  code: string,
): void {
  expect(response.status).toBe(status);
  expect(typeof response.body).toBe('string');
  const parsed = JSON.parse(response.body as string) as Record<string, unknown>;
  expect(Object.keys(parsed).toSorted()).toEqual(['code', 'status']);
  expect(parsed.status).toBe(status);
  expect(parsed.code).toBe(code);
  expect(response.headers['content-type']).toBe(
    'application/json;charset=utf-8',
  );
}

/**
 * Asserts no `Access-Control-*` header appears on any response.
 *
 * @param responses - All responses gathered by the caller.
 */
function expectNoCorsHeaders(responses: readonly DevResponse[]): void {
  for (const response of responses) {
    for (const headerName of Object.keys(response.headers)) {
      expect(headerName.startsWith('access-control-')).toBe(false);
    }
  }
}

beforeEach(() => {
  tempRoot = mkdtempSync(join(tmpdir(), 'dev-server-'));
  tempModelsDir = mkdtempSync(join(tempRoot, 'models-'));
  tempDevRoot = mkdtempSync(join(tempRoot, 'devroot-'));
  // Wire the real landed store API (mirrors loadDefaultDevServerContext).
  testStoreApi = {
    isVoxelModelName: (name) => isVoxelModelName(name),
    listVoxelModelNames: (modelsDir) => listVoxelModelNames(modelsDir),
    readVoxelModelDocument: (modelsDir, name) =>
      readVoxelModelDocument(modelsDir, name),
    writeVoxelModelDocument: (modelsDir, name, candidate) =>
      writeVoxelModelDocument(modelsDir, name, candidate),
    validateVoxelModelPayload: (candidate) =>
      validateVoxelModelPayload(candidate),
    modelStoreError: ModelStoreError,
  };
  writeFileSync(join(tempDevRoot, 'index.html'), '<!DOCTYPE html>\n', 'utf8');
  mkdirSync(join(tempDevRoot, 'assets', 'models'), { recursive: true });
  writeFileSync(
    join(tempDevRoot, 'assets', 'models', 'car-f1.model.json'),
    JSON.stringify(makeDocument()),
    'utf8',
  );
  mkdirSync(join(tempDevRoot, 'assets', 'bins'), { recursive: true });
  writeFileSync(
    join(tempDevRoot, 'assets', 'bins', 'car-f1-lod128.bin'),
    Buffer.from([0, 1, 2, 3]),
  );
  // Seed the enumeration with one stored document so name-route tests can
  // exercise save/read/rebuild flows without repeating setup.
  writeVoxelModelDocument(tempModelsDir, 'car-f1', makeDocument());
});

afterEach(() => {
  rmSync(tempRoot, { recursive: true, force: true });
  tempRoot = '';
  tempModelsDir = '';
  tempDevRoot = '';
});

describe('S1 host/origin hardening', () => {
  it('rejects a forged, missing, or mismatched port Host with 403 dev-server/host', () => {
    const context = makeTestContext();
    const responses = [
      invoke(context, 'GET', '/api/models', { host: 'evil.example:9001' }),
      invoke(context, 'GET', '/api/models', {}),
      invoke(context, 'GET', '/api/models', { host: '127.0.0.1:9999' }),
      invoke(context, 'GET', '/api/models', { host: 'localhost:9999' }),
    ];
    for (const response of responses) {
      expectErrorBody(response, 403, 'dev-server/host');
    }
    expectNoCorsHeaders(responses);
  });

  it('accepts exactly 127.0.0.1:port and localhost:port for any route', () => {
    const context = makeTestContext();
    const accepted = invoke(context, 'GET', '/api/models', {
      host: `127.0.0.1:${TEST_PORT}`,
    });
    expect(accepted.status).toBe(200);
    const loopback = invoke(context, 'GET', '/api/models', {
      host: `localhost:${TEST_PORT}`,
    });
    expect(loopback.status).toBe(200);
    expectNoCorsHeaders([accepted, loopback]);
  });

  it('rejects foreign Origin on state-changing routes with 403 dev-server/origin', () => {
    const context = makeTestContext();
    const foreignOrigin = { origin: 'http://evil.example' };
    const response = invoke(
      context,
      'POST',
      '/api/models/car-f1/rebuild',
      testHeaders(foreignOrigin),
    );
    expectErrorBody(response, 403, 'dev-server/origin');
    const save = toSaveChunks(makeDocumentBytes());
    const putResponse = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        ...foreignOrigin,
        'content-type': 'application/json',
        'content-length': String(save.contentLength),
      }),
      save.chunks,
    );
    expectErrorBody(putResponse, 403, 'dev-server/origin');
  });

  it('accepts matching Origin and ignores Origin on GET routes', () => {
    const context = makeTestContext();
    const matching = invoke(
      context,
      'POST',
      '/api/models/car-f1/rebuild',
      testHeaders({ origin: `http://localhost:${TEST_PORT}` }),
    );
    expect(matching.status).toBe(202);
    const getWithForeignOrigin = invoke(
      context,
      'GET',
      '/api/models',
      testHeaders({ origin: 'http://evil.example' }),
    );
    expect(getWithForeignOrigin.status).toBe(200);
  });
});

describe('S1 no CORS responses anywhere', () => {
  it('never emits Access-Control headers across method/route mixtures', () => {
    const context = makeTestContext();
    const responses = [
      invoke(context, 'GET', '/', testHeaders()),
      invoke(context, 'GET', '/api/models', testHeaders()),
      invoke(
        context,
        'POST',
        '/api/models/car-f1/rebuild',
        testHeaders({ origin: `http://127.0.0.1:${TEST_PORT}` }),
      ),
      invoke(context, 'OPTIONS', '/api/models', testHeaders()),
      invoke(context, 'GET', '/assets/models/car-f1.model.json', testHeaders()),
    ];
    expect(responses[0].status).toBe(200);
    expectNoCorsHeaders(responses);
  });
});

describe('S2 rebuild pinning', () => {
  it('is implemented with spawn(process.execPath, args) and shell: false only', () => {
    const source = readFileSync(
      resolve(
        process.cwd(),
        'examples',
        'neatRacing',
        'server',
        'dev-server.ts',
      ),
      'utf8',
    );
    expect(source).toContain('spawn(process.execPath,');
    expect(source).toContain('shell: false');
    expect(source).not.toMatch(/\bexecSync\s*\(/);
    expect(source).not.toMatch(/\bexec\s*\(/);
    expect(source).not.toMatch(/shell\s*:\s*true/);
  });

  it('pins the fixed bins script path ahead of the model name in argv', () => {
    const argv = buildRebuildSpawnArgv(
      '/abs/repo/examples/neatRacing/scripts/build-voxel-bins.mjs',
      'car-f1',
    );
    expect(argv[0]).toBe(
      '/abs/repo/examples/neatRacing/scripts/build-voxel-bins.mjs',
    );
    expect(argv[1]).toBe('car-f1');
    expect(argv).toHaveLength(2);
  });

  it('spawns only model names that already exist in the enumeration', () => {
    const spawnArgs: string[] = [];
    const context = createDevServerContext({
      port: TEST_PORT,
      modelsDir: tempModelsDir,
      devRoot: tempDevRoot,
      assetsDir: join(tempDevRoot, 'assets'),
      pages: { '/': 'index.html', '/editor/': 'editor/editor.html' },
      spawner: (modelName) => {
        spawnArgs.push(modelName);
      },
      store: testStoreApi,
    });
    const known = invoke(
      context,
      'POST',
      '/api/models/car-f1/rebuild',
      testHeaders(),
    );
    expect(known.status).toBe(202);
    expect(spawnArgs).toEqual(['car-f1']);
    const unknown = invoke(
      context,
      'POST',
      '/api/models/ghost-model/rebuild',
      testHeaders(),
    );
    expectErrorBody(unknown, 404, 'dev-server/not-found');
    expect(spawnArgs).toEqual(['car-f1']);
  });

  it('coalesces concurrent rebuild triggers latest-wins via the coordinator', () => {
    const context = createDevServerContext({
      port: TEST_PORT,
      modelsDir: tempModelsDir,
      devRoot: tempDevRoot,
      assetsDir: join(tempDevRoot, 'assets'),
      pages: { '/': 'index.html', '/editor/': 'editor/editor.html' },
      // The stub keeps rebuilds in flight so handlers can pile up.
      spawner: () => undefined,
      store: testStoreApi,
    });
    const first = invoke(
      context,
      'POST',
      '/api/models/car-f1/rebuild',
      testHeaders(),
    );
    const second = invoke(
      context,
      'POST',
      '/api/models/car-f1/rebuild',
      testHeaders(),
    );
    expect(first.status).toBe(202);
    expect(second.status).toBe(202);
    const firstBody = JSON.parse(first.body as string) as {
      code?: string;
    };
    const secondBody = JSON.parse(second.body as string) as {
      code?: string;
    };
    expect(firstBody.code).toBe('dev-server/rebuild-started');
    expect(secondBody.code).toBe('dev-server/rebuild-coalesced');
  });
});

describe('S3 body hardening', () => {
  it('rejects Content-Length greater than the configured cap with 413', () => {
    const context = makeTestContext();
    expect(NEATRACING_MAX_SAVE_BODY_BYTES).toBe(8 * 1024 * 1024);
    const response = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        'content-type': 'application/json',
        'content-length': String(NEATRACING_MAX_SAVE_BODY_BYTES + 1),
      }),
    );
    expectErrorBody(response, 413, 'dev-server/body-too-large');
  });

  it('rejects streamed bodies exceeding the cap mid-read with 413', () => {
    const context = makeTestContext();
    const blob = new Uint8Array(NEATRACING_MAX_SAVE_BODY_BYTES / 2 + 1);
    const chunkA = blob.subarray(0, NEATRACING_MAX_SAVE_BODY_BYTES / 2);
    const chunkB = blob.subarray(
      NEATRACING_MAX_SAVE_BODY_BYTES / 2,
      NEATRACING_MAX_SAVE_BODY_BYTES / 2 + 1,
    );
    const response = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        'content-type': 'application/json',
        'content-length': String(NEATRACING_MAX_SAVE_BODY_BYTES + 1),
      }),
      [chunkA, chunkB],
    );
    expectErrorBody(response, 413, 'dev-server/body-too-large');
  });

  it('accepts a body of exactly the cap with the correct content type', () => {
    const context = makeTestContext();
    const documentBytes = makeDocumentBytes();
    const save = toSaveChunks(documentBytes);
    expect(save.contentLength).toBeLessThan(NEATRACING_MAX_SAVE_BODY_BYTES);
    const response = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        'content-type': 'application/json',
        'content-length': String(save.contentLength),
      }),
      save.chunks,
    );
    expect(response.status).toBe(200);
  });

  it('rejects wrong or missing content type on body-bearing routes with 415', () => {
    const context = makeTestContext();
    const save = toSaveChunks(makeDocumentBytes());
    const textResponse = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        'content-type': 'text/plain',
        'content-length': String(save.contentLength),
      }),
      save.chunks,
    );
    expectErrorBody(textResponse, 415, 'dev-server/content-type');
    const missingResponse = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({ 'content-length': String(save.contentLength) }),
      save.chunks,
    );
    expectErrorBody(missingResponse, 415, 'dev-server/content-type');
  });

  it('accepts application/json with parameters such as charset', () => {
    const context = makeTestContext();
    const save = toSaveChunks(makeDocumentBytes());
    const response = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        'content-type': 'application/json; charset=utf-8',
        'content-length': String(save.contentLength),
      }),
      save.chunks,
    );
    expect(response.status).toBe(200);
  });
});

describe('S4 name-shape hardening', () => {
  it('rejects traversal payloads before any path join with 403', () => {
    const context = makeTestContext();
    const save = toSaveChunks(makeDocumentBytes());
    for (const hostilePath of [
      '/api/models/%2e%2e',
      '/api/models/..%2Fsecret',
      '/api/models/../evil',
      '/api/models/x%2Fy',
    ]) {
      const response = invoke(
        context,
        'PUT',
        hostilePath,
        testHeaders({
          'content-type': 'application/json',
          'content-length': String(save.contentLength),
        }),
        save.chunks,
      );
      expectErrorBody(response, 403, 'dev-server/name-shape');
    }
  });

  it('rejects names that violate the stored pattern (uppercase, dots, empty)', () => {
    const context = makeTestContext();
    for (const hostilePath of ['/api/models/Car', '/api/models/c.a-r']) {
      const response = invoke(context, 'GET', hostilePath, testHeaders());
      expectErrorBody(response, 403, 'dev-server/name-shape');
    }
  });

  it('returns 404 for PUT against a valid but not enumerated name (no create-via-PUT)', () => {
    const context = makeTestContext();
    const save = toSaveChunks(makeDocumentBytes());
    const response = invoke(
      context,
      'PUT',
      '/api/models/car-f2',
      testHeaders({
        'content-type': 'application/json',
        'content-length': String(save.contentLength),
      }),
      save.chunks,
    );
    expectErrorBody(response, 404, 'dev-server/not-found');
    const stored = readdirSync(tempModelsDir).toSorted();
    expect(stored).toEqual(['car-f1.model.json']);
  });
});

describe('S5 method allowlist and error-body hygiene', () => {
  it('maps OPTIONS to a global 405 with no CORS headers', () => {
    const context = makeTestContext();
    const response = invoke(context, 'OPTIONS', '/api/models', testHeaders());
    expectErrorBody(response, 405, 'dev-server/method');
    expectNoCorsHeaders([response]);
  });

  it('enforces the per-route method allowlist with 405', () => {
    const context = makeTestContext();
    expectErrorBody(
      invoke(context, 'POST', '/api/models', testHeaders()),
      405,
      'dev-server/method',
    );
    expectErrorBody(
      invoke(context, 'DELETE', '/api/models/car-f1', testHeaders()),
      405,
      'dev-server/method',
    );
    expectErrorBody(
      invoke(context, 'PATCH', '/', testHeaders()),
      405,
      'dev-server/method',
    );
  });

  it('emits exactly {status, code} on every failure response', () => {
    const context = makeTestContext();
    const responses = [
      invoke(context, 'GET', '/definitely/missing', testHeaders()),
      invoke(context, 'DELETE', '/api/models/car-f1', testHeaders()),
      invoke(context, 'GET', '/api/models/evil.example:1', testHeaders()),
    ];
    for (const response of responses) {
      expect(typeof response.body).toBe('string');
      const parsed = JSON.parse(response.body as string) as Record<
        string,
        unknown
      >;
      expect(Object.keys(parsed).toSorted()).toEqual(['code', 'status']);
      expect(parsed).not.toHaveProperty('message');
      expect(parsed).not.toHaveProperty('stack');
    }
    expectNoCorsHeaders(responses);
  });
});

describe('static pages and assets', () => {
  it('serves the root page from the dev root as HTML', () => {
    const context = makeTestContext();
    const response = invoke(context, 'GET', '/', testHeaders());
    expect(response.status).toBe(200);
    expect(response.headers['content-type']).toBe('text/html;charset=utf-8');
    expect(response.body).toContain('<!DOCTYPE html>');
  });

  it('returns 404 for pages that are absent and for editor.html in a bare dev root', () => {
    const context = makeTestContext();
    const editorResponse = invoke(context, 'GET', '/editor/', testHeaders());
    expectErrorBody(editorResponse, 404, 'dev-server/not-found');
    const unknownResponse = invoke(
      context,
      'GET',
      '/definitely/missing',
      testHeaders(),
    );
    expectErrorBody(unknownResponse, 404, 'dev-server/not-found');
  });

  it('serves assets only under /assets/ with resolve containment', () => {
    const context = makeTestContext();
    const model = invoke(
      context,
      'GET',
      '/assets/models/car-f1.model.json',
      testHeaders(),
    );
    expect(model.status).toBe(200);
    expect(model.headers['content-type']).toBe(
      'application/json;charset=utf-8',
    );
    expect(
      Buffer.compare(
        Buffer.from(model.body as Uint8Array),
        Buffer.from(
          readFileSync(
            join(tempDevRoot, 'assets', 'models', 'car-f1.model.json'),
          ),
        ),
      ),
    ).toBe(0);
    const bin = invoke(
      context,
      'GET',
      '/assets/bins/car-f1-lod128.bin',
      testHeaders(),
    );
    expect(bin.status).toBe(200);
    expect(bin.headers['content-type']).toBe('application/octet-stream');
  });

  it('refuses static traversal and never lists directories', () => {
    const context = makeTestContext();
    const traversal = invoke(
      context,
      'GET',
      '/assets/%2e%2e/index.html',
      testHeaders(),
    );
    expect(traversal.status).toBe(403);
    const dotDot = invoke(
      context,
      'GET',
      '/assets/..%2Findex.html',
      testHeaders(),
    );
    expect(dotDot.status).toBe(403);
    for (const listingPath of ['/assets', '/assets/', '/assets/models/']) {
      const listing = invoke(context, 'GET', listingPath, testHeaders());
      // '/assets' and '/assets/' are directory probes: never a listing.
      if (listing.status === 404) {
        expectErrorBody(listing, 404, 'dev-server/not-found');
      } else {
        expectErrorBody(listing, 403, 'dev-server/static-path');
      }
    }
    const unknownPrefix = invoke(
      context,
      'GET',
      '/static/classified.json',
      testHeaders(),
    );
    expectErrorBody(unknownPrefix, 404, 'dev-server/not-found');
  });
});

describe('model CRUD via the landed store', () => {
  it('lists enumerated model names from the wired models directory', () => {
    writeVoxelModelDocument(tempModelsDir, 'car-f1', makeDocument());
    writeVoxelModelDocument(tempModelsDir, 'wheel', {
      ...makeDocument(),
      meta: { name: 'wheel', dims: FIXTURE_DIMS, symmetric: false },
    });
    const context = makeTestContext();
    const response = invoke(context, 'GET', '/api/models', testHeaders());
    expect(response.status).toBe(200);
    const parsed = JSON.parse(response.body as string) as { models?: unknown };
    expect(parsed.models).toEqual(['car-f1', 'wheel']);
  });

  it('round-trips PUT: saves to disk, replaces content, and rotates bak1', () => {
    writeVoxelModelDocument(tempModelsDir, 'car-f1', makeDocument());
    const context = makeTestContext();
    // Build the modified copy from an explicit override: `makeDocument()`
    // spreads the shared palette constant by reference, so an in-place
    // mutation would leak into every other fixture built afterwards.
    const modified = makeDocument({
      palette: [
        [0, 0, 0, 0, 0],
        [200, 230, 255, 255, 200],
      ],
    });
    const save = toSaveChunks(Buffer.from(JSON.stringify(modified), 'utf8'));
    const response = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        'content-type': 'application/json',
        'content-length': String(save.contentLength),
      }),
      save.chunks,
    );
    expect(response.status).toBe(200);
    const savedBody = JSON.parse(response.body as string) as { code?: string };
    expect(savedBody.code).toBe('dev-server/saved');
    const onDisk = JSON.parse(
      readFileSync(join(tempModelsDir, 'car-f1.model.json'), 'utf8'),
    ) as Record<string, unknown>;
    expect(onDisk).toEqual(modified);
    const backup = join(
      tempModelsDir,
      NEATRACING_MODEL_STORE_BACKUP_DIR_NAME,
      'car-f1.model.json.bak1',
    );
    expect(existsSync(backup)).toBe(true);
    expect(JSON.parse(readFileSync(backup, 'utf8'))).toEqual(makeDocument());
  });

  it('serves, validates, and 404s documents through the store', () => {
    writeVoxelModelDocument(tempModelsDir, 'car-f1', makeDocument());
    const context = makeTestContext();
    const fetched = invoke(context, 'GET', '/api/models/car-f1', testHeaders());
    expect(fetched.status).toBe(200);
    const fetchedBody = JSON.parse(fetched.body as string) as {
      document?: unknown;
    };
    expect(fetchedBody.document).toEqual(makeDocument());
    const invalid = toSaveChunks(
      Buffer.from(JSON.stringify({ ...makeDocument(), rogue: true }), 'utf8'),
    );
    const invalidResponse = invoke(
      context,
      'POST',
      '/api/models/car-f1/validate',
      testHeaders({
        'content-type': 'application/json',
        'content-length': String(invalid.contentLength),
      }),
      invalid.chunks,
    );
    expectErrorBody(invalidResponse, 400, 'model-store/top-level-keys');
    const unknown = invoke(
      context,
      'POST',
      '/api/models/ghost-model/validate',
      testHeaders({
        'content-type': 'application/json',
        'content-length': '2',
      }),
      toSaveChunks(Buffer.from('{}', 'utf8')).chunks,
    );
    expectErrorBody(unknown, 404, 'dev-server/not-found');
  });

  it('rejects malformed JSON bodies with 400 dev-server/invalid-json', () => {
    writeVoxelModelDocument(tempModelsDir, 'car-f1', makeDocument());
    const context = makeTestContext();
    const response = invoke(
      context,
      'PUT',
      '/api/models/car-f1',
      testHeaders({
        'content-type': 'application/json',
        'content-length': '3',
      }),
      [Buffer.from('{,}', 'utf8')],
    );
    expectErrorBody(response, 400, 'dev-server/invalid-json');
  });
});

describe('configuration: port resolution', () => {
  it(`defaults to ${NEATRACING_DEV_SERVER_DEFAULT_PORT}`, () => {
    expect(NEATRACING_DEV_SERVER_DEFAULT_PORT).toBe(8787);
  });

  it('reads NEATRACING_DEV_SERVER_PORT and rejects invalid values back to default', () => {
    expect(resolveDevServerPort({})).toBe(8787);
    expect(
      resolveDevServerPort({ [NEATRACING_DEV_SERVER_PORT_ENV_NAME]: '9001' }),
    ).toBe(9001);
    expect(
      resolveDevServerPort({ [NEATRACING_DEV_SERVER_PORT_ENV_NAME]: '65535' }),
    ).toBe(65535);
    const envName = NEATRACING_DEV_SERVER_PORT_ENV_NAME;
    for (const invalid of ['not-a-port', '0', '70000', '-5', '1.5']) {
      expect(resolveDevServerPort({ [envName]: invalid })).toBe(8787);
    }
  });

  it('exposes isAllowedDevServerHost for the runtime gate', () => {
    expect(isAllowedDevServerHost(`127.0.0.1:${TEST_PORT}`, TEST_PORT)).toBe(
      true,
    );
    expect(isAllowedDevServerHost(`localhost:${TEST_PORT}`, TEST_PORT)).toBe(
      true,
    );
    expect(isAllowedDevServerHost('evil.example:9001', TEST_PORT)).toBe(false);
    expect(isAllowedDevServerHost(undefined, TEST_PORT)).toBe(false);
  });
});

describe('runtime: real socket smoke test', () => {
  it('binds 127.0.0.1, validates Host, and serves the real index page', async () => {
    const devServer = await import('./dev-server');
    const handle = await devServer.startDevServer({
      port: 0,
      devRoot: resolve(process.cwd(), 'examples', 'neatRacing'),
      buildPages: false,
    });
    try {
      expect(handle.port).toBeGreaterThan(0);
      const { default: httpRequest } = await import('node:http');
      const fetchText = (
        uri: string,
      ): Promise<{ status: number; type: string; body: string }> =>
        new Promise((fulfil, reject) => {
          const request = httpRequest.get(
            uri,
            { headers: { host: `127.0.0.1:${handle.port}` } },
            (response) => {
              let body = '';
              response.setEncoding('utf8');
              response.on('data', (chunk: string) => {
                body += chunk;
              });
              response.on('end', () => {
                fulfil({
                  status: response.statusCode ?? 0,
                  type: String(response.headers['content-type'] ?? ''),
                  body,
                });
              });
            },
          );
          request.on('error', (error: Error) => reject(error));
        });
      const page = await fetchText(`http://127.0.0.1:${handle.port}/`);
      expect(page.status).toBe(200);
      expect(page.type).toContain('text/html');
      expect(page.body).toContain('<!DOCTYPE');
    } finally {
      await handle.close();
    }
  });
});

/**
 * Module-entry self-start contract (spec 12.4 CLI path).
 *
 * The `dev:neatRacing` package script runs this module as the Node entry
 * script (`ts-node examples/neatRacing/server/dev-server.ts`); the
 * module-level direct-run guard must detect that case and call
 * `startDevServer()` itself, producing a real listener. Plain imports of
 * the module (every other test in this suite) must stay side-effect free.
 *
 * The contract is pinned by spawning the exact public command shape
 * (`node node_modules/ts-node/dist/bin.js <script>` — identical to the
 * npm `ts-node` bin shim, cross-platform) and polling until the server
 * answers on the reserved port. `ts-node` cold-compiles before it binds,
 * so the probe deadline allows generous warm-up time.
 */

/** Wall-clock cap for the CLI child to surface its listener. */
const CLI_SELF_START_DEADLINE_MS = 180_000;

/** Jest timeout granted to the CLI self-start contract test. */
const CLI_SELF_START_TIMEOUT_MS = 210_000;

/** Delay between readiness probes of the CLI child listener. */
const CLI_SELF_START_POLL_INTERVAL_MS = 250;

/** Tail length retained from the CLI child pipes for diagnostics. */
const CLI_DIAGNOSTIC_TAIL_CHARS = 4096;

/** Grace period granted to the SIGKILL teardown before giving up. */
const CLI_CHILD_TEARDOWN_GRACE_MS = 5_000;

/**
 * Builds the child environment for the CLI self-start probe: the reserved
 * port on the documented override variable, with worker markers removed —
 * the runtime resolve shim is a no-op whenever `JEST_WORKER_ID` is
 * visible, so the CLI child must run with the real one.
 *
 * @param port - Reserved port for the child listener.
 * @returns The child process environment.
 */
function cliChildEnv(port: number): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env };
  delete env.JEST_WORKER_ID;
  env[NEATRACING_DEV_SERVER_PORT_ENV_NAME] = String(port);
  return env;
}

/**
 * Captures rolling stdout/stderr tails of the CLI child pipes.
 *
 * @param child - The CLI child process.
 * @returns A getter for the retained diagnostic tails.
 */
function pipeChildDiagnostics(child: ChildProcess): () => string {
  let stdoutTail = '';
  let stderrTail = '';
  child.stdout?.on('data', (chunk: Buffer) => {
    stdoutTail = (stdoutTail + String(chunk)).slice(-CLI_DIAGNOSTIC_TAIL_CHARS);
  });
  child.stderr?.on('data', (chunk: Buffer) => {
    stderrTail = (stderrTail + String(chunk)).slice(-CLI_DIAGNOSTIC_TAIL_CHARS);
  });
  return () =>
    `stdout: ${stdoutTail || '(none)'} | stderr: ${stderrTail || '(none)'}`;
}

/**
 * Reserves a free localhost port for the CLI probe (bind port 0, read the
 * assigned port, release it before spawn).
 *
 * @returns The reserved port.
 */
async function reserveSelfStartProbePort(): Promise<number> {
  const { createServer: createNetProbeServer } = await import('node:net');
  return await new Promise<number>((fulfil, reject) => {
    const probe = createNetProbeServer();
    probe.once('error', (error: Error) => reject(error));
    probe.listen(0, '127.0.0.1', () => {
      const address = probe.address();
      const port =
        address !== null &&
        typeof address === 'object' &&
        !Array.isArray(address)
          ? address.port
          : 0;
      probe.close(() => {
        if (port > 0) {
          fulfil(port);
        } else {
          reject(new Error('self-start probe could not reserve a port'));
        }
      });
    });
  });
}

/**
 * Polls the CLI child's listener until the root page answers once.
 *
 * Readiness polling tolerates connection refusals before the child binds;
 * an early child exit or deadline expiry fails fast with the pipe tails.
 *
 * @param port - Reserved listener port expected in the child.
 * @param child - The CLI child being probed.
 * @param diagnostics - Getter for the captured pipe tails.
 * @returns The first resolved root-page response snapshot.
 */
async function awaitSelfStartPage(
  port: number,
  child: ChildProcess,
  diagnostics: () => string,
): Promise<{ status: number }> {
  const { default: httpRequest } = await import('node:http');
  const deadline = Date.now() + CLI_SELF_START_DEADLINE_MS;
  return await new Promise((fulfil, reject) => {
    let settled = false;
    const attempt = (): void => {
      if (settled) {
        return;
      }
      if (child.exitCode !== null || child.signalCode !== null) {
        settled = true;
        reject(
          new Error(
            `CLI child exited before binding port ${port}; ${diagnostics()}`,
          ),
        );
        return;
      }
      if (Date.now() > deadline) {
        settled = true;
        reject(
          new Error(
            `CLI child did not serve 127.0.0.1:${port} within ` +
              `${CLI_SELF_START_DEADLINE_MS}ms; ${diagnostics()}`,
          ),
        );
        return;
      }
      const request = httpRequest.get(
        `http://127.0.0.1:${port}/`,
        { headers: { host: `127.0.0.1:${port}` }, timeout: 2000 },
        (response) => {
          if (settled) {
            response.resume();
            return;
          }
          settled = true;
          const status = response.statusCode ?? 0;
          response.resume();
          response.on('end', () => fulfil({ status }));
        },
      );
      request.on('timeout', () => request.destroy());
      request.on('error', () => {
        setTimeout(attempt, CLI_SELF_START_POLL_INTERVAL_MS);
      });
    };
    attempt();
  });
}

/**
 * SIGKILLs the CLI child and waits for its exit.
 *
 * @param child - The CLI child process spawned by the test.
 * @returns Resolves once the child exited (or the grace period elapsed).
 */
async function stopSelfStartChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  await new Promise<void>((fulfil) => {
    const grace = setTimeout(fulfil, CLI_CHILD_TEARDOWN_GRACE_MS);
    child.once('exit', () => {
      clearTimeout(grace);
      fulfil();
    });
    child.kill('SIGKILL');
  });
}

describe('runtime: module entry self-start (spec 12.4 CLI contract)', () => {
  it(
    'starts a live listener when dev-server.ts is the Node entry script',
    async () => {
      const { spawn } = await import('node:child_process');
      const port = await reserveSelfStartProbePort();
      const repoRoot = process.cwd();
      const child = spawn(
        process.execPath,
        [
          join(repoRoot, 'node_modules', 'ts-node', 'dist', 'bin.js'),
          join(repoRoot, 'examples', 'neatRacing', 'server', 'dev-server.ts'),
        ],
        {
          cwd: repoRoot,
          env: cliChildEnv(port),
          stdio: ['ignore', 'pipe', 'pipe'],
        },
      );
      const diagnostics = pipeChildDiagnostics(child);
      try {
        const page = await awaitSelfStartPage(port, child, diagnostics);
        // The guard contract: the CLI process itself is alive and serving.
        expect(page.status).toBe(200);
        expect(child.exitCode).toBeNull();
      } finally {
        await stopSelfStartChild(child);
      }
    },
    CLI_SELF_START_TIMEOUT_MS,
  );
});
