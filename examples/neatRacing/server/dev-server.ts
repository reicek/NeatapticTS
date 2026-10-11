/**
 * Hardened zero-dependency development server for the Neat Racing demo
 * (spec section 8.2, AC-018; slice P1-S5-B, step 08).
 *
 * The server is a thin HTTP shell over the landed model store
 * (`model-store.utils.ts`, slice P1-S5-A) and serves the demo pages/assets
 * from `examples/neatRacing/`. Every request passes through the six
 * normative hardening gates before any handler logic runs:
 *
 * - **S1 host/origin** â€” every request needs `Host: 127.0.0.1:<port>` or
 *   `localhost:<port>`; state-changing requests additionally require a
 *   matching `Origin` when one is present; `Access-Control-*` headers are
 *   never emitted, so devtools/hostile pages cannot drive saves, rebuilds,
 *   or validations.
 * - **S2 rebuild pinning** â€” rebuilds only ever run the fixed bins script
 *   through `spawn(process.execPath, [<script>, <name>])` with
 *   `shell: false`; a per-model single-flight coordinator coalesces
 *   concurrent triggers latest-wins (a superseded in-flight rebuild's
 *   completion is discarded, the newest trigger starts fresh).
 * - **S3 body limits** â€” body-bearing PUT/POST require
 *   `application/json` (params tolerated), reject declared or actual sizes
 *   above `NEATRACING_MAX_SAVE_BODY_BYTES`, and reject malformed JSON.
 * - **S4 name shape** â€” `:name` segments are decoded and checked against
 *   the store's name pattern before any path join, so traversal payloads
 *   (`../evil`, `%2e%2e`) are structurally rejected.
 * - **S5 method + hygiene** â€” per-route method allowlists (405 otherwise),
 *   static serving confined to declared prefixes with resolve containment,
 *   no directory listings, and error bodies are exactly `{status, code}`.
 * - **S6 backup rotation** â€” delegated to the store's `bak1..3` ring.
 *
 * Execution model: request handling is a pure function
 * (`handleDevServerRequest`) from `DevRequestIntent` to `DevResponse` so the
 * whole hardening surface is unit-testable without sockets. The store
 * module is loaded lazily through `loadDefaultDevServerContext` (dynamic
 * import + a resolve-hook shim registered only outside Jest) because the
 * runtime process is executed by ts-node in native ESM mode, where
 * extensionless specifiers inside the store's own import graph fail unless
 * the shim retries them with an explicit `.ts`. The single I/O seam the
 * pure core touches is the injected store API object plus the optional
 * injected rebuild spawner, keeping everything else pure dispatch.
 *
 * The module is run by the `dev:neatRacing` package script (spec section
 * 12.4: `ts-node examples/neatRacing/server/dev-server.ts`) through a
 * module-level direct-run guard: when this file is the Node entry script it
 * calls `startDevServer()` itself and keeps the process serving, so the CLI
 * path exposes a real listener instead of exiting silently. Plain imports
 * (Jest, tooling) stay side-effect free. The server binds
 * `127.0.0.1:NEATRACING_DEV_SERVER_PORT` (default 8787).
 */
import { createServer, type Server } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, readFileSync, statSync, type Stats } from 'node:fs';
import { register } from 'node:module';
import { join, resolve, extname, dirname, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

/** Type-only reference to the landed store's rejection class (S6 rotation). */
import type { ModelStoreError } from './model-store.utils';

/** Default dev-server port, overridable through the env var below (S-port). */
export const NEATRACING_DEV_SERVER_DEFAULT_PORT = 8787;

/** Environment variable consulted by `resolveDevServerPort`. */
export const NEATRACING_DEV_SERVER_PORT_ENV_NAME = 'NEATRACING_DEV_SERVER_PORT';

/** Hard cap for any request body (spec S3: 8 MiB). */
export const NEATRACING_MAX_SAVE_BODY_BYTES = 8 * 1024 * 1024;

/** Absolute file-system path of the fixed rebuild script (S2 pin). */
const REBUILD_SCRIPT_RELATIVE_PATH = join(
  'examples',
  'neatRacing',
  'scripts',
  'build-voxel-bins.mjs',
);

/** Default dev root when no override is provided (repo-relative). */
const DEV_ROOT_RELATIVE_PATH = join('examples', 'neatRacing');

/**
 * Resolve-hook source: retries TypeScript dotted-name specifiers with
 * `.ts`.
 *
 * Any `ERR_MODULE_NOT_FOUND` for a specifier whose tail is not a real Node
 * runtime extension is retried with an explicit `.ts` suffix. This covers
 * both plain extensionless names and the repo's dotted TypeScript file
 * names (for example `./model-store.utils`), which the store's own import
 * graph relies on; real runtime extensions are never retried.
 */
const TYPESCRIPT_RESOLVE_HOOK_SOURCE = String.raw`
const NON_TYPESCRIPT_RUNTIME_EXTENSION_PATTERN =
  /\.(?:js|mjs|cjs|json|node|wasm|css|html)$/i;
export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    const retryable =
      error !== null &&
      typeof error === 'object' &&
      error.code === 'ERR_MODULE_NOT_FOUND' &&
      context !== null &&
      typeof context === 'object' &&
      typeof context.parentURL === 'string' &&
      !NON_TYPESCRIPT_RUNTIME_EXTENSION_PATTERN.test(specifier);
    if (!retryable) {
      throw error;
    }
    return nextResolve(specifier + '.ts', context);
  }
}
`;

/**
 * Machine-readable rejection codes for server-level gates. Store-level
 * rejections forward their own `model-store/â€¦` and `voxel-model/â€¦` codes.
 */
export type DevServerErrorCode =
  | 'dev-server/host'
  | 'dev-server/origin'
  | 'dev-server/method'
  | 'dev-server/not-found'
  | 'dev-server/name-shape'
  | 'dev-server/static-path'
  | 'dev-server/body-too-large'
  | 'dev-server/content-type'
  | 'dev-server/invalid-json'
  | 'dev-server/internal';

/**
 * A single request as observed by the HTTP front end, normalized for the
 * pure dispatcher.
 */
export interface DevRequestIntent {
  /** Uppercased HTTP method exactly as received. */
  method: string;
  /** Percent-encoded request path (query string already stripped). */
  path: string;
  /** Lowercased header record (`host`, `origin`, `content-length`, ...). */
  headers: Readonly<Record<string, string>>;
  /** Streamed body bytes; absent for bodyless requests. */
  bodyChunks?: readonly Uint8Array[];
}

/** A complete response produced by the pure dispatcher. */
export interface DevResponse {
  /** HTTP status code. */
  status: number;
  /** Lowercased response header record. */
  headers: Readonly<Record<string, string>>;
  /** Serialized body: text for JSON/HTML, raw bytes for assets. */
  body: string | Uint8Array;
}

/**
 * The store API slice the dispatcher consumes at runtime; loaded lazily so
 * the module graph stays resolvable under ts-node's native ESM execution.
 */
export interface DevModelStoreApi {
  /** S4 name-shape gate shared with the store. */
  isVoxelModelName: (name: unknown) => boolean;
  /** Sorted enumeration of stored model names. */
  listVoxelModelNames: (modelsDir: string) => string[];
  /** Reads and validates one stored document. */
  readVoxelModelDocument: (modelsDir: string, name: unknown) => unknown;
  /** Atomically saves a document, rotating the previous version (S6). */
  writeVoxelModelDocument: (
    modelsDir: string,
    name: unknown,
    candidate: unknown,
  ) => void;
  /** Validate-only gate (no save, no backup). */
  validateVoxelModelPayload: (candidate: unknown) => unknown;
  /** The rejection class used for `instanceof` forwarding. */
  modelStoreError: typeof ModelStoreError;
}

/**
 * Spawns one rebuild flight for a model; the completion callback settles
 * the flight (success, failure, or spawn error) with the coordinator.
 */
export type DevRebuildSpawner = (
  modelName: string,
  onSettled: (error: Error | null) => void,
) => void;

/** Single-flight record for one in-flight rebuild (S2 coalescing). */
export interface DevRebuildRecord {
  /** Model being rebuilt. */
  readonly name: string;
  /** `true` once a newer flight has taken over the model. */
  superseded: boolean;
}

/** Configuration accepted by the pure context factory. */
export interface DevServerContextConfig {
  /** Port used for Host/Origin allowlisting (S1). */
  port: number;
  /** Directory holding `*.model.json` documents. */
  modelsDir: string;
  /** Root directory for pages (devRoot). */
  devRoot: string;
  /** Directory served under the `/assets/` prefix. */
  assetsDir: string;
  /** Exact route-to-page-file map served from `devRoot`. */
  pages: Readonly<Record<string, string>>;
  /** Rebuild spawner (injectable for tests). */
  spawner: DevRebuildSpawner;
  /** Optional per-request body cap override (tests only). */
  maxBodyBytes?: number;
}

/** A fully wired server context: everything `handleDevServerRequest` needs. */
export interface DevServerContext {
  /** Effective port for Host/Origin allowlisting (S1). */
  readonly port: number;
  /** Models directory handed to the store. */
  readonly modelsDir: string;
  /** Dev root for pages. */
  readonly devRoot: string;
  /** Assets directory served under `/assets/`. */
  readonly assetsDir: string;
  /** Exact route-to-page-file map. */
  readonly pages: Readonly<Record<string, string>>;
  /** Injected store API (loaded dynamically by the default factory). */
  readonly store: DevModelStoreApi;
  /** Single-flight rebuild coordinator (S2). */
  readonly rebuilds: RebuildCoordinator;
  /** Injected rebuild spawner. */
  readonly spawner: DevRebuildSpawner;
  /** Per-request body cap (S3). */
  readonly maxBodyBytes: number;
}

/**
 * Coalesces in-flight rebuild triggers per model, latest-wins (S2): a newer
 * `begin` marks the previous record superseded, and `complete` reports
 * whether the settled flight's result is still meaningful.
 */
export class RebuildCoordinator {
  /** Active flight per model name. */
  readonly #active = new Map<string, DevRebuildRecord>();

  /**
   * Opens (or takes over) the rebuild flight for one model.
   *
   * A previous in-flight record is marked superseded: its eventual
   * completion is discarded, and the caller must not treat its output as
   * applied. The returned record identifies THIS flight.
   *
   * @param name - Validated model name.
   * @returns The fresh, non-superseded record for the new flight.
   */
  public begin(name: string): DevRebuildRecord {
    const previous = this.#active.get(name);
    if (previous !== undefined) {
      previous.superseded = true;
    }
    const record: DevRebuildRecord = { name, superseded: false };
    this.#active.set(name, record);
    return record;
  }

  /**
   * Settles a started flight.
   *
   * @param record - The record returned by `begin` for this flight.
   * @returns `'discarded'` when a newer flight took over the model,
   * `'applied'` when this was the newest flight for the model.
   */
  public complete(record: DevRebuildRecord): 'applied' | 'discarded' {
    const active = this.#active.get(record.name);
    if (active === record) {
      this.#active.delete(record.name);
    }
    return record.superseded ? 'discarded' : 'applied';
  }

  /**
   * Reports whether a model currently has an active (non-superseded) flight.
   *
   * @param name - Model name to probe.
   * @returns `true` while the model's newest flight is unresolved.
   */
  public isActive(name: string): boolean {
    const active = this.#active.get(name);
    return active !== undefined && !active.superseded;
  }
}

/**
 * Builds the fixed rebuild argv for the bins script (S2 pin): the absolute
 * script path first, then the model name as the sole argument.
 *
 * @param rebuildScriptPath - Absolute path of `build-voxel-bins.mjs`.
 * @param modelName - Validated model name to rebuild.
 * @returns The argv array (`[<script>, <name>]`).
 */
export function buildRebuildSpawnArgv(
  rebuildScriptPath: string,
  modelName: string,
): string[] {
  return [rebuildScriptPath, modelName];
}

/**
 * Creates the default rebuild spawner, pinning the bins script with the
 * node interpreter, no shell, and inherited stdio (S2).
 *
 * @param rebuildScriptPath - Absolute path of the bins script.
 * @returns A spawner suitable for `createDevServerContext`.
 */
export function createDefaultRebuildSpawner(
  rebuildScriptPath: string,
): DevRebuildSpawner {
  return (modelName, onSettled): void => {
    const child = spawn(
      process.execPath,
      buildRebuildSpawnArgv(rebuildScriptPath, modelName),
      { shell: false, stdio: 'inherit' },
    );
    child.on('close', (code) => {
      onSettled(code === 0 ? null : new Error(`rebuild exit ${String(code)}`));
    });
    child.on('error', (error: Error) => {
      onSettled(error);
    });
  };
}

/**
 * Resolves the dev-server port from an environment record.
 *
 * Accepts integers in the `1..65535` range; anything else (missing, garbage,
 * out of range, fractional) falls back to the default 8787.
 *
 * @param env - Environment map, for example `process.env`.
 * @returns The resolved port.
 */
export function resolveDevServerPort(
  env: Readonly<Record<string, string | undefined>> = {},
): number {
  const raw = env[NEATRACING_DEV_SERVER_PORT_ENV_NAME];
  if (raw === undefined) {
    return NEATRACING_DEV_SERVER_DEFAULT_PORT;
  }
  const parsed = Number.parseInt(raw, 10);
  if (
    !Number.isInteger(parsed) ||
    parsed < 1 ||
    parsed > 65_535 ||
    String(parsed) !== raw
  ) {
    return NEATRACING_DEV_SERVER_DEFAULT_PORT;
  }
  return parsed;
}

/**
 * S1 host gate: exactly `127.0.0.1:<port>` or `localhost:<port>` passes.
 *
 * @param host - Raw `Host` header value (may be `undefined`).
 * @param port - Port this dev server instance is serving.
 * @returns `true` exactly when the host pair matches the loopback allowlist.
 */
export function isAllowedDevServerHost(host: unknown, port: number): boolean {
  if (host === undefined || host === null) {
    return false;
  }
  const normalized = String(host).toLowerCase();
  return (
    normalized === `127.0.0.1:${port}` || normalized === `localhost:${port}`
  );
}

/**
 * S1 origin gate for state-changing requests: an `Origin`, when present,
 * must originate from the loopback pages this server serves; a missing
 * `Origin` (curl, tests) also passes. The scheme is pinned to `http:`
 * because the dev server itself serves plaintext HTTP.
 *
 * @param origin - Raw `Origin` header value (may be `undefined`).
 * @param port - Port this dev server instance serves.
 * @returns `true` exactly when the origin is empty or allowlisted.
 */
export function isAllowedDevServerOrigin(
  origin: unknown,
  port: number,
): boolean {
  if (origin === undefined || origin === null || origin === '') {
    return true;
  }
  const normalized = String(origin).toLowerCase();
  return (
    normalized === `http://127.0.0.1:${port}` ||
    normalized === `http://localhost:${port}`
  );
}

/**
 * Pure context factory: wires configuration to the injected store API.
 *
 * Callers that need the real store use `loadDefaultDevServerContext`, which
 * performs the lazy dynamic import. Tests inject a stub object instead, so
 * context construction stays synchronous and free of module resolution.
 *
 * @param config - Explicitly provided configuration and store API.
 * @returns The wired context.
 */
export function createDevServerContext(config: {
  port: number;
  modelsDir: string;
  devRoot: string;
  assetsDir: string;
  pages: Readonly<Record<string, string>>;
  spawner: DevRebuildSpawner;
  maxBodyBytes?: number;
  store?: DevModelStoreApi;
}): DevServerContext {
  const { store } = config;
  if (store === undefined) {
    throw new Error(
      'createDevServerContext requires a store API; use loadDefaultDevServerContext at runtime',
    );
  }
  return {
    port: config.port,
    modelsDir: config.modelsDir,
    devRoot: config.devRoot,
    assetsDir: config.assetsDir,
    pages: config.pages,
    store,
    rebuilds: new RebuildCoordinator(),
    spawner: config.spawner,
    maxBodyBytes: config.maxBodyBytes ?? NEATRACING_MAX_SAVE_BODY_BYTES,
  };
}

/** Optional overrides accepted by the default (runtime) context loader. */
export interface DevServerContextOverrides {
  /** Port override for Host/Origin allowlisting. */
  port?: number;
  /** Model directory override. */
  modelsDir?: string;
  /** Dev root override (defaults to `examples/neatRacing` under cwd). */
  devRoot?: string;
  /** Assets directory override. */
  assetsDir?: string;
  /** Rebuild spawner override. */
  spawner?: DevRebuildSpawner;
  /** Absolute bins script path override (S2 pin target). */
  rebuildScriptPath?: string;
}

/**
 * Builds the page file map served from the dev root.
 *
 * `editor.html` may not exist until its slice lands; the page handler is
 * existence-gated so the route degrades to a clean 404 meanwhile.
 *
 * @returns The static route-to-relative-file map.
 */
function defaultPages(): Record<string, string> {
  return {
    '/': 'index.html',
    '/editor/': 'editor/editor.html',
  };
}

/**
 * Registers the `.ts` resolve retry shim exactly once, outside Jest.
 *
 * ts-node in native ESM mode (repo `type: module`) cannot resolve the
 * dotted-name specifiers inside the store's own import graph; the hook
 * retries `ERR_MODULE_NOT_FOUND` with an explicit `.ts` suffix for every
 * specifier that does not end with a real runtime extension. The hook
 * source is registered as a `data:` URL against the running script's URL,
 * so no temporary file or `import.meta` reference is needed (Jest safe).
 *
 * Under Jest (`JEST_WORKER_ID` defined) this is a no-op: ts-jest resolves
 * the extensionless graph itself and a module hook would fight the
 * transformer.
 */
function ensureRuntimeResolveShim(): void {
  if (runtimeResolveShimInstalled) {
    return;
  }
  runtimeResolveShimInstalled = true;
  if (process.env.JEST_WORKER_ID !== undefined) {
    return;
  }
  const parentUrl =
    process.argv[1] !== undefined
      ? pathToFileURL(process.argv[1]).href
      : 'file:///neatracing-dev-server.ts';
  register(
    `data:text/javascript,${encodeURIComponent(TYPESCRIPT_RESOLVE_HOOK_SOURCE)}`,
    parentUrl,
  );
}

/** Tracks whether the runtime resolve shim has been registered. */
let runtimeResolveShimInstalled = false;

/**
 * Runtime context factory: registers the `.ts` resolve shim outside Jest,
 * then lazily imports the landed store module.
 *
 * The dynamic import is required because the store's own graph contains
 * dotted-name specifiers, which native ESM execution (ts-node under
 * `type: module`) cannot resolve without the shim; the shim is installed
 * (plus one macrotask so Node's async resolve-hook chain is in place)
 * before the import and applies to the store's sub-graph only.
 *
 * @param overrides - Overrides for paths, port, or the spawner.
 * @returns A promise resolving to the fully wired context.
 */
export async function loadDefaultDevServerContext(
  overrides: DevServerContextOverrides = {},
): Promise<DevServerContext> {
  ensureRuntimeResolveShim();
  // One macrotask after `register()`: Node installs resolve hooks
  // asynchronously, so yielding a tick guarantees the `.ts` retry shim is
  // active before the first dynamic import of the store graph.
  await new Promise<void>((fulfil) => setImmediate(fulfil));
  const store = await import('./model-store.utils');
  const spawner =
    overrides.spawner ??
    createDefaultRebuildSpawner(
      overrides.rebuildScriptPath ??
        resolve(process.cwd(), REBUILD_SCRIPT_RELATIVE_PATH),
    );
  const port = overrides.port ?? resolveDevServerPort(process.env);
  const devRoot =
    overrides.devRoot ?? resolve(process.cwd(), DEV_ROOT_RELATIVE_PATH);
  return createDevServerContext({
    port,
    modelsDir: overrides.modelsDir ?? join(devRoot, 'assets', 'models'),
    devRoot,
    assetsDir: overrides.assetsDir ?? join(devRoot, 'assets'),
    pages: defaultPages(),
    spawner,
    store: {
      isVoxelModelName: (name) => store.isVoxelModelName(name),
      listVoxelModelNames: (modelsDir) => store.listVoxelModelNames(modelsDir),
      readVoxelModelDocument: (modelsDir, name) =>
        store.readVoxelModelDocument(modelsDir, name),
      writeVoxelModelDocument: (modelsDir, name, candidate) =>
        store.writeVoxelModelDocument(modelsDir, name, candidate),
      validateVoxelModelPayload: (candidate) =>
        store.validateVoxelModelPayload(candidate),
      modelStoreError: store.ModelStoreError,
    },
  });
}

/**
 * Starts the HTTP runtime over a context: builds the browser bundles (when
 * enabled), binds `127.0.0.1` on the resolved port, and funnels every
 * connection through the pure dispatcher with runtime body accounting.
 *
 * @param config - Start configuration (all fields optional).
 * @param config.port - Port override; defaults to the env-resolved port.
 * @param config.devRoot - Dev root override for page/static resolution.
 * @param config.modelsDir - Model directory override.
 * @param config.buildPages - Build the demo/editor bundles on start.
 * @param config.context - Prebuilt context (tests); skips store loading.
 * @returns A promise resolving to the server handle.
 */
export interface DevServerStartConfig {
  port?: number;
  devRoot?: string;
  modelsDir?: string;
  assetsDir?: string;
  buildPages?: boolean;
  context?: DevServerContext;
}

/** Server handle returned by `startDevServer`. */
export interface DevServerHandle {
  /** The bound port (resolved by the OS when requested with `0`). */
  readonly port: number;
  /** The underlying HTTP server instance. */
  readonly server: Server;
  /** Stops the server, tolerating repeat calls. */
  readonly close: () => Promise<void>;
}

/**
 * Starts the hardened dev server.
 *
 * @param config - Optional start configuration.
 * @returns A promise resolving to the server handle.
 */
export async function startDevServer(
  config: DevServerStartConfig = {},
): Promise<DevServerHandle> {
  const devRoot =
    config.devRoot ?? resolve(process.cwd(), DEV_ROOT_RELATIVE_PATH);
  const context =
    config.context ??
    (await loadDefaultDevServerContext({
      port: config.port,
      devRoot,
      modelsDir: config.modelsDir,
      assetsDir: config.assetsDir,
    }));
  const assetsDir =
    context.assetsDir !== '' ? context.assetsDir : join(devRoot, 'assets');
  if (config.buildPages ?? true) {
    buildDevBundles(devRoot, assetsDir);
  }
  const server = createServer((request, response) => {
    serveConnection(context, request, response);
  });
  const port = config.port ?? resolveDevServerPort(process.env);
  await new Promise<void>((fulfil, reject) => {
    server.once('error', (error: Error) => reject(error));
    server.listen(port, '127.0.0.1', () => {
      fulfil();
    });
  });
  const boundPort = addressPort(server);
  if (port === 0 && boundPort > 0) {
    // OS-picked port: rewire the S1 host/origin allowlist to the bound
    // value so hosts like `127.0.0.1:<actual>` pass the gate. The context
    // object is shared with every connection handler, so the mutation
    // applies to all subsequent requests.
    (context as { port: number }).port = boundPort;
  }
  return {
    port: addressPort(server),
    server,
    close: () =>
      new Promise<void>((fulfil) => {
        server.close(() => {
          fulfil();
        });
      }),
  };
}

/**
 * Compiles the demo and editor bundles into the served asset locations.
 *
 * Per spec section 8.2 the dev server rebuilds the browser bundle on every
 * start through the esbuild API; entries that a pending slice has not
 * landed yet are skipped with a console note so startup never crashes.
 * The bundle paths form the fixed contract with the host pages:
 * `assets/neatracing.bundle.js` (root page) and
 * `editor/editor.bundle.js` (editor page).
 *
 * @param devRoot - The dev root directory.
 * @param assetsDir - The assets output directory.
 */
function buildDevBundles(devRoot: string, assetsDir: string): void {
  const rootEntry = join(devRoot, 'browser-entry', 'browser-entry.ts');
  if (existsSync(rootEntry)) {
    void import('esbuild')
      .then((esbuild) => {
        esbuild.buildSync({
          entryPoints: [rootEntry],
          outfile: join(assetsDir, 'neatracing.bundle.js'),
          bundle: true,
          platform: 'browser',
          format: 'iife',
          minify: true,
          sourcemap: true,
          external: ['fs', 'child_process', 'path'],
          logLevel: 'silent',
        });
      })
      .catch(() => undefined);
  } else {
    console.log(
      'neatRacing dev server: browser entry missing, skipping bundle',
    );
  }
  const editorEntry = join(devRoot, 'editor', 'editor-entry.ts');
  if (existsSync(editorEntry)) {
    void import('esbuild')
      .then((esbuild) => {
        esbuild.buildSync({
          entryPoints: [editorEntry],
          outfile: join(devRoot, 'editor', 'editor.bundle.js'),
          bundle: true,
          platform: 'browser',
          format: 'iife',
          minify: true,
          sourcemap: true,
          external: ['fs', 'child_process', 'path'],
          logLevel: 'silent',
        });
      })
      .catch(() => undefined);
  }
}

/**
 * Serves one live HTTP connection through the pure dispatcher.
 *
 * Body bytes are accumulated up to the cap with an early 413 short-circuit;
 * once the body is fully read the intent is dispatched and the response
 * serialized. Connection-level failures produce exactly `{status, code}`.
 *
 * @param context - The wired server context.
 * @param request - Incoming request stream.
 * @param response - Server response stream.
 */
function serveConnection(
  context: DevServerContext,
  request: import('node:http').IncomingMessage,
  response: import('node:http').ServerResponse,
): void {
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  let overBudget = false;
  let settled = false;
  const maxBodyBytes = context.maxBodyBytes;
  request.setEncoding('utf8');
  request.on('data', (chunk: string) => {
    if (overBudget || settled) {
      return;
    }
    totalBytes += Buffer.byteLength(chunk, 'utf8');
    if (totalBytes > maxBodyBytes) {
      overBudget = true;
      request.destroy();
      respondJson(response, jsonError(413, 'dev-server/body-too-large'));
      return;
    }
    chunks.push(Buffer.from(chunk, 'utf8'));
  });
  request.on('error', () => {
    if (!settled) {
      settled = true;
      respondJson(response, jsonError(400, 'dev-server/invalid-json'));
    }
  });
  request.on('end', () => {
    if (settled) {
      return;
    }
    settled = true;
    try {
      const intent = readIntent(request, chunks);
      respond(response, handleDevServerRequest(context, intent));
    } catch {
      respondJson(response, jsonError(500, 'dev-server/internal'));
    }
  });
  request.on('close', () => {
    if (!settled) {
      settled = true;
    }
  });
}

/**
 * Normalizes one live request into a pure `DevRequestIntent`.
 *
 * @param request - Incoming request stream.
 * @param chunks - Collected body chunks.
 * @returns The pure intent (query string stripped, headers lowercased).
 */
function readIntent(
  request: import('node:http').IncomingMessage,
  chunks: readonly Uint8Array[],
): DevRequestIntent {
  const headers: Record<string, string> = {};
  for (const [name, value] of Object.entries(request.headers)) {
    if (typeof value === 'string') {
      headers[name.toLowerCase()] = value;
    } else if (Array.isArray(value) && value.length > 0) {
      headers[name.toLowerCase()] = value[0];
    }
  }
  return {
    method: String(request.method ?? 'GET').toUpperCase(),
    path: String(request.url ?? '/').split('?')[0],
    headers,
    bodyChunks: chunks.length > 0 ? chunks : undefined,
  };
}

/**
 * Serializes and writes one `DevResponse` to the live response.
 *
 * @param response - Live server response.
 * @param outcome - Pure response outcome.
 */
function respond(
  response: import('node:http').ServerResponse,
  outcome: DevResponse,
): void {
  response.statusCode = outcome.status;
  for (const [name, value] of Object.entries(outcome.headers)) {
    response.setHeader(name, value);
  }
  response.end(
    typeof outcome.body === 'string' ? outcome.body : Buffer.from(outcome.body),
  );
}

/**
 * Writes a JSON error response directly to a live response.
 *
 * @param response - Live server response.
 * @param outcome - Pure error response.
 */
function respondJson(
  response: import('node:http').ServerResponse,
  outcome: DevResponse,
): void {
  respond(response, outcome);
}

/**
 * Builds a failure response carrying exactly `{status, code}` (S5 hygiene:
 * no filesystem paths, messages, or stacks in the payload).
 *
 * @param status - HTTP status.
 * @param code - Machine-readable code.
 * @returns The pure error response.
 */
function jsonError(status: number, code: string): DevResponse {
  return {
    status,
    headers: {
      'content-type': 'application/json;charset=utf-8',
      'cache-control': 'no-store',
    },
    body: JSON.stringify({ status, code }),
  };
}

/**
 * Builds a success response envelope.
 *
 * @param status - HTTP status.
 * @param code - Machine-readable success code.
 * @param extra - Additional payload fields merged into the body.
 * @returns The pure success response.
 */
function jsonOk(
  status: number,
  code: string,
  extra: Readonly<Record<string, unknown>> = {},
): DevResponse {
  return {
    status,
    headers: {
      'content-type': 'application/json;charset=utf-8',
      'cache-control': 'no-store',
    },
    body: JSON.stringify({ status, code, ...extra }),
  };
}

/**
 * Builds an HTML page response.
 *
 * @param body - The page markup.
 * @returns The pure page response.
 */
function htmlResponse(body: string): DevResponse {
  return {
    status: 200,
    headers: {
      'content-type': 'text/html;charset=utf-8',
      'cache-control': 'no-store',
    },
    body,
  };
}

/**
 * Serves a raw file body with a content type derived from its extension.
 *
 * @param filePath - Absolute file path.
 * @param stats - File stats (size sanity check for the read).
 * @returns The pure asset response.
 */
function assetResponse(filePath: string, stats: Stats): DevResponse {
  const contentType = assetContentType(extname(filePath).toLowerCase());
  return {
    status: 200,
    headers: {
      'content-type': contentType,
      'cache-control': 'no-store',
      'content-length': String(stats.size),
    },
    body: new Uint8Array(readFileSync(filePath)),
  };
}

/**
 * Maps a file extension to the served content type.
 *
 * @param extension - Lowercase extension including the dot.
 * @returns The content type; unknown extensions fall back to octet-stream.
 */
function assetContentType(extension: string): string {
  switch (extension) {
    case '.json':
      return 'application/json;charset=utf-8';
    case '.html':
      return 'text/html;charset=utf-8';
    case '.js':
    case '.mjs':
      return 'text/javascript;charset=utf-8';
    case '.svg':
      return 'image/svg+xml';
    default:
      return 'application/octet-stream';
  }
}

/**
 * Extracts the raw path segments of an `/api/...` route after the prefix.
 *
 * @param rawPath - Percent-encoded request path.
 * @param prefix - Exact literal prefix, for example `/api/models/`.
 * @returns The raw segments (not yet decoded).
 */
function apiSegments(rawPath: string, prefix: string): string[] {
  return rawPath.slice(prefix.length).split('/');
}

/**
 * Decodes one raw path segment, re-routing malformed percent-escapes to the
 * caller through `null` so the dispatcher can reject them as bad names.
 *
 * @param segment - Raw (encoded) segment.
 * @returns The decoded segment, or `null` when undecodable.
 */
function decodeSegment(segment: string): string | null {
  try {
    return decodeURIComponent(segment);
  } catch {
    return null;
  }
}

/**
 * The pure dispatcher: maps one normalized intent to one response, applying
 * the S1â€“S6 gates in a fixed order before any I/O or store call runs.
 *
 * Gate order: host (S1) â†’ global `OPTIONS` rejection (S5) â†’ origin for
 * state-changing methods (S1) â†’ route/method resolution (S5/S4) â†’ name
 * shape against the store pattern (S4) â†’ body gates (S3) â†’ handler
 * delegation to the landed store (S6 rotation inside the store).
 *
 * @param context - Wired server context.
 * @param intent - Normalized request intent.
 * @returns The complete response.
 */
export function handleDevServerRequest(
  context: DevServerContext,
  intent: DevRequestIntent,
): DevResponse {
  const host = intent.headers.host;
  if (!isAllowedDevServerHost(host, context.port)) {
    return jsonError(403, 'dev-server/host');
  }
  const method = intent.method.toUpperCase();
  if (method === 'OPTIONS') {
    return jsonError(405, 'dev-server/method');
  }
  const stateChanging = method === 'PUT' || method === 'POST';
  if (
    stateChanging &&
    !isAllowedDevServerOrigin(intent.headers.origin, context.port)
  ) {
    return jsonError(403, 'dev-server/origin');
  }

  // S4 structural gate: a raw literal `.` or `..` segment is a traversal
  // attempt on every route class (api, pages, statics), rejected before any
  // routing decision or path join. Encoded payloads are caught downstream by
  // their route-specific decoders.
  for (const rawSegment of intent.path.split('/')) {
    if (rawSegment === '.' || rawSegment === '..') {
      return jsonError(403, 'dev-server/name-shape');
    }
  }

  // Exact page routes first (root and editor index), then API routes, then
  // the static prefixes; anything else falls through to 404. Pages whose
  // route ends with a slash also expose page-adjacent statics (the editor
  // page loads `editor.bundle.js` relative to its own URL).
  const pageFile = context.pages[intent.path];
  if (pageFile !== undefined) {
    if (method !== 'GET') {
      return jsonError(405, 'dev-server/method');
    }
    const pagePath = join(context.devRoot, pageFile);
    if (!existsSync(pagePath)) {
      return jsonError(404, 'dev-server/not-found');
    }
    return htmlResponse(readTextFile(pagePath));
  }
  if (intent.path === '/api/models') {
    if (method !== 'GET') {
      return jsonError(405, 'dev-server/method');
    }
    return handleListModels(context);
  }

  // Name routes apply only to the exact api prefix: without the prefix
  // guard every stray path would leak into the name-shape gate.
  const isApiModelsRoute = intent.path.startsWith('/api/models/');
  const nameRoutes = isApiModelsRoute
    ? apiSegments(intent.path, '/api/models/')
    : [];
  if (nameRoutes.length === 1 && nameRoutes[0].length > 0) {
    return handleNamedApiRoute(context, intent, nameRoutes[0]);
  }
  if (
    nameRoutes.length === 2 &&
    nameRoutes[0].length > 0 &&
    (nameRoutes[1] === 'rebuild' || nameRoutes[1] === 'validate')
  ) {
    return handleNamedApiRoute(context, intent, nameRoutes[0], nameRoutes[1]);
  }
  const pagePrefixPrefixes = getPagePrefixStaticRoutes(context);
  if (intent.path.startsWith('/assets/')) {
    return handleStaticAsset(context, intent, '/assets/', context.assetsDir);
  }
  for (const [prefix, rootDirectory] of pagePrefixPrefixes) {
    if (intent.path.startsWith(prefix) && intent.path.length > prefix.length) {
      return handleStaticAsset(context, intent, prefix, rootDirectory);
    }
  }
  return jsonError(404, 'dev-server/not-found');
}

/**
 * Derives page-adjacent static routes from the page map: a page route ending
 * with a slash (for example `/editor/`) also serves sibling files from that
 * page's directory, so `editor.bundle.js` can load next to the page.
 *
 * @param context - Wired server context.
 * @returns Prefix, to absolute root-directory entries.
 */
function getPagePrefixStaticRoutes(
  context: DevServerContext,
): Map<string, string> {
  const routes = new Map<string, string>();
  for (const [route, pageFile] of Object.entries(context.pages)) {
    // The root page route (`/`) is not a directory route: opening a static
    // root over the whole dev root would silently widen the static surface
    // (and swallow unmatched API paths); only true sub-pages (like
    // `/editor/`) expose page-adjacent statics.
    if (route.endsWith('/') && route.length > 1) {
      routes.set(route, dirname(join(context.devRoot, pageFile)));
    }
  }
  return routes;
}

/**
 * Handles the `GET /api/models` listing route.
 *
 * @param context - Wired server context.
 * @returns The `{status, code, models}` response, or a store rejection.
 */
function handleListModels(context: DevServerContext): DevResponse {
  try {
    const models = context.store.listVoxelModelNames(context.modelsDir);
    return jsonOk(200, 'dev-server/ok', { models });
  } catch (error) {
    return forwardStoreError(error);
  }
}

/**
 * Handles `/api/models/:name` and its `/rebuild` and `/validate` actions.
 *
 * Shared gating: S4 name shape (structural, before any join), then the
 * enumeration check (404 for PUT/GET/actions against names the store does
 * not list), then per-route method allowlists, then route-specific bodies.
 *
 * @param context - Wired server context.
 * @param intent - Normalized request intent.
 * @param rawName - Raw (encoded) name segment.
 * @param action - Optional action tail (`rebuild` or `validate`).
 * @returns The route-specific response.
 */
function handleNamedApiRoute(
  context: DevServerContext,
  intent: DevRequestIntent,
  rawName: string,
  action?: string,
): DevResponse {
  const decoded = decodeSegment(rawName);
  if (decoded === null) {
    return jsonError(403, 'dev-server/name-shape');
  }
  // S4: shape-check the decoded name BEFORE any path materialization; the
  // store pattern rejects separators, `%`, `..`, and mixed case outright.
  if (!context.store.isVoxelModelName(decoded)) {
    return jsonError(403, 'dev-server/name-shape');
  }
  const method = intent.method;
  if (action === undefined) {
    if (method === 'GET') {
      return handleReadModel(context, decoded);
    }
    if (method === 'PUT') {
      return handleSaveModel(context, intent, decoded);
    }
    return jsonError(405, 'dev-server/method');
  }
  if (method !== 'POST') {
    return jsonError(405, 'dev-server/method');
  }
  if (!context.store.listVoxelModelNames(context.modelsDir).includes(decoded)) {
    return jsonError(404, 'dev-server/not-found');
  }
  if (action === 'rebuild') {
    return handleRebuild(context, decoded);
  }
  return handleValidate(context, intent);
}

/**
 * Serves one stored document through the store's read+validate path.
 *
 * @param context - Wired server context.
 * @param name - Validated model name.
 * @returns The document payload, or 404 for unenumerated names, or a
 * forwarded store rejection for store-level failures.
 */
function handleReadModel(context: DevServerContext, name: string): DevResponse {
  if (!context.store.listVoxelModelNames(context.modelsDir).includes(name)) {
    return jsonError(404, 'dev-server/not-found');
  }
  try {
    const document = context.store.readVoxelModelDocument(
      context.modelsDir,
      name,
    );
    return jsonOk(200, 'dev-server/ok', { document });
  } catch (error) {
    return forwardStoreError(error);
  }
}

/**
 * Saves one document through the store, enforcing the S3 body gates and the
 * S4 enumeration gate (PUT never creates models) before the write.
 *
 * @param context - Wired server context.
 * @param intent - Normalized request intent.
 * @param name - Validated model name.
 * @returns The save acknowledgement, or a gate rejection.
 */
function handleSaveModel(
  context: DevServerContext,
  intent: DevRequestIntent,
  name: string,
): DevResponse {
  // S4 enumeration gate FIRST (PUT never creates models), then S3 size and
  // content-type gates, then parse+write delegation to the store.
  if (!context.store.listVoxelModelNames(context.modelsDir).includes(name)) {
    return jsonError(404, 'dev-server/not-found');
  }
  const budgetGate = gateRequestBody(context, intent);
  if (budgetGate !== undefined) {
    return budgetGate;
  }
  const contentTypeGate = gateRequestContentType(intent);
  if (contentTypeGate !== undefined) {
    return contentTypeGate;
  }
  try {
    const candidate = parseJsonBody(intent);
    context.store.writeVoxelModelDocument(context.modelsDir, name, candidate);
    return jsonOk(200, 'dev-server/saved');
  } catch (error) {
    if (error instanceof SyntaxError) {
      return jsonError(400, 'dev-server/invalid-json');
    }
    return forwardStoreError(error);
  }
}

/**
 * Validates one document candidate without saving (editor preflight).
 *
 * @param context - Wired server context.
 * @param intent - Normalized request intent.
 * @returns The validation acknowledgement, or a forwarded rejection.
 */
function handleValidate(
  context: DevServerContext,
  intent: DevRequestIntent,
): DevResponse {
  const budgetGate = gateRequestBody(context, intent);
  if (budgetGate !== undefined) {
    return budgetGate;
  }
  const contentTypeGate = gateRequestContentType(intent);
  if (contentTypeGate !== undefined) {
    return contentTypeGate;
  }
  try {
    const candidate = parseJsonBody(intent);
    context.store.validateVoxelModelPayload(candidate);
    return jsonOk(200, 'dev-server/valid');
  } catch (error) {
    if (error instanceof SyntaxError) {
      return jsonError(400, 'dev-server/invalid-json');
    }
    return forwardStoreError(error);
  }
}

/**
 * Triggers (or coalesces) one rebuild flight through the S2-pinned spawner.
 *
 * @param context - Wired server context.
 * @param name - Validated, enumerated model name.
 * @returns The 202 start/coalesce acknowledgement.
 */
function handleRebuild(context: DevServerContext, name: string): DevResponse {
  const activeBefore = context.rebuilds.isActive(name);
  const record = context.rebuilds.begin(name);
  try {
    context.spawner(name, (error) => {
      context.rebuilds.complete(record);
      if (error !== null) {
        console.log(
          `neatRacing dev server: rebuild failed for ${name}: ${error.message}`,
        );
      }
    });
  } catch (error) {
    context.rebuilds.complete(record);
    console.log(
      `neatRacing dev server: rebuild spawn failed: ${String(error)}`,
    );
  }
  return jsonOk(
    202,
    activeBefore
      ? 'dev-server/rebuild-coalesced'
      : 'dev-server/rebuild-started',
  );
}

/**
 * Serves one static asset under a fixed prefix with resolve containment
 * (S5): decoded segments may not traverse, the resolved path must stay
 * inside the root, and directories are never listed (404 instead).
 *
 * @param context - Wired server context.
 * @param intent - Normalized request intent.
 * @param prefix - The static prefix to serve, including the trailing slash.
 * @returns The asset response, 403 for traversal, 404 for missing/listing.
 */
function handleStaticAsset(
  context: DevServerContext,
  intent: DevRequestIntent,
  prefix: string,
  assetsDir: string,
): DevResponse {
  if (intent.method !== 'GET') {
    return jsonError(405, 'dev-server/method');
  }
  const rootDirectory = resolve(
    assetsDir === '' ? join(context.devRoot, 'assets') : assetsDir,
  );
  const tail = intent.path.slice(prefix.length);
  const rawSegments = tail === '' ? [] : tail.split('/');
  const decodedSegments: string[] = [];
  for (const rawSegment of rawSegments) {
    const decoded = decodeSegment(rawSegment);
    if (
      decoded === null ||
      decoded.length === 0 ||
      decoded === '.' ||
      decoded === '..' ||
      decoded.includes('/') ||
      decoded.includes('\\')
    ) {
      return jsonError(403, 'dev-server/static-path');
    }
    decodedSegments.push(decoded);
  }
  const joined = join(rootDirectory, ...decodedSegments);
  const resolved = resolve(joined);
  if (resolved !== rootDirectory && !resolved.startsWith(rootDirectory + sep)) {
    return jsonError(403, 'dev-server/static-path');
  }
  let stats: Stats | undefined;
  try {
    stats = statSync(resolved);
  } catch {
    stats = undefined;
  }
  if (stats === undefined || !stats.isFile()) {
    return jsonError(404, 'dev-server/not-found');
  }
  return assetResponse(resolved, stats);
}

/**
 * Applies the S3 size gates for body-bearing routes.
 *
 * @param context - Wired server context.
 * @param intent - Normalized request intent.
 * @returns A 413 response when over budget, else `undefined`.
 */
function gateRequestBody(
  context: DevServerContext,
  intent: DevRequestIntent,
): DevResponse | undefined {
  const declared = parseContentLength(intent.headers['content-length']);
  if (declared !== undefined && declared > context.maxBodyBytes) {
    return jsonError(413, 'dev-server/body-too-large');
  }
  const total = (intent.bodyChunks ?? []).reduce(
    (sum, chunk) => sum + chunk.length,
    0,
  );
  if (total > context.maxBodyBytes) {
    return jsonError(413, 'dev-server/body-too-large');
  }
  return undefined;
}

/**
 * Parses the request body as JSON.
 *
 * @param intent - Normalized request intent.
 * @returns The parsed JSON value.
 * @throws SyntaxError when the body is not valid JSON.
 */
function parseJsonBody(intent: DevRequestIntent): unknown {
  const serialized = joinChunks(intent.bodyChunks ?? []).toString('utf8');
  return JSON.parse(serialized) as unknown;
}

/**
 * Enforces the S3 content-type gate for body-bearing routes.
 *
 * Mirrors `gateRequestBody` placement so callers can reject a hostile
 * `Content-Type` before touching the payload.
 *
 * @param intent - Normalized request intent.
 * @returns A 415 response when the content type is not JSON, else `undefined`.
 */
function gateRequestContentType(
  intent: DevRequestIntent,
): DevResponse | undefined {
  const contentType = intent.headers['content-type'];
  if (contentType === undefined) {
    return jsonError(415, 'dev-server/content-type');
  }
  const mediaType = contentType.toLowerCase().split(';')[0].trim();
  if (mediaType !== 'application/json') {
    return jsonError(415, 'dev-server/content-type');
  }
  return undefined;
}

/**
 * Parses a declared `Content-Length` header.
 *
 * @param raw - Header value (may be `undefined`).
 * @returns The parsed length, or `undefined` when absent or unparseable.
 */
function parseContentLength(raw: unknown): number | undefined {
  if (typeof raw !== 'string') {
    return undefined;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isInteger(parsed) || parsed < 0) {
    return undefined;
  }
  return parsed;
}

/**
 * Concatenates body chunks into one buffer.
 *
 * @param chunks - The collected chunks.
 * @returns The concatenated buffer.
 */
function joinChunks(chunks: readonly Uint8Array[]): Buffer {
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
}

/**
 * Reads a text file, tolerating an existing-but-unreadable page by letting
 * the runtime error handler translate the failure to a 500.
 *
 * @param filePath - Absolute file path.
 * @returns The file text.
 */
function readTextFile(filePath: string): string {
  return readFileSync(filePath, 'utf8');
}

/**
 * Forwards a store/schema rejection as an HTTP failure response.
 *
 * Store codes are namespaced and stable, so they are forwarded verbatim in
 * the `{status, code}` body; unknown failures fall back to a generic
 * rejection that leaks nothing (S5).
 *
 * @param error - Caught rejection.
 * @returns The `{status, code}` failure response.
 */
function forwardStoreError(error: unknown): DevResponse {
  if (
    error !== null &&
    typeof error === 'object' &&
    'code' in error &&
    typeof (error as { code: unknown }).code === 'string'
  ) {
    const code = (error as { code: string }).code;
    return jsonError(code === 'model-store/not-found' ? 404 : 400, code);
  }
  return jsonError(500, 'dev-server/internal');
}

/**
 * Resolves the bound port from a listening server.
 *
 * @param server - The HTTP server.
 * @returns The port, or 0 when not yet listening.
 */
function addressPort(server: Server): number {
  const address = server.address();
  if (address !== null && typeof address === 'object' && 'port' in address) {
    return address.port;
  }
  return 0;
}

/**
 * True when this module is the Node entry script — the `dev:neatRacing`
 * CLI path (spec 12.4: `ts-node examples/neatRacing/server/dev-server.ts`).
 *
 * The comparison is entry-path based and module-agnostic: `import.meta.url`
 * cannot be used here because the Jest lanes compile this module with
 * `module=commonjs`, where `import.meta` is rejected with TS1343. Two
 * entry shapes are recognized:
 * - this script itself is `process.argv[1]` (the executing entry), or
 * - the `ts-node` CLI bin is `process.argv[1]` with this script as its
 *   first CLI argument (the CLI's own entry shape).
 * Importing the module (Jest, tooling) matches neither shape, so imports
 * stay side-effect free.
 *
 * @returns `true` when Node launched this file directly as the entry.
 */
function isDirectDevServerExecution(): boolean {
  const entryScriptPath = process.argv[1];
  if (entryScriptPath === undefined) {
    return false;
  }
  const entryHref = pathToFileURL(entryScriptPath).href.toLowerCase();
  if (entryHref.endsWith('/dev-server.ts')) {
    return true;
  }
  const cliArgument = process.argv[2];
  if (cliArgument === undefined) {
    return false;
  }
  const cliScriptHref = pathToFileURL(cliArgument).href.toLowerCase();
  return (
    entryHref.endsWith('/ts-node/dist/bin.js') &&
    cliScriptHref.endsWith('/dev-server.ts')
  );
}

/**
 * Module-level direct-run entry (spec 12.4): when this file is the Node
 * entry script, the module itself calls `startDevServer()`, so the
 * `dev:neatRacing` package script exposes a real `127.0.0.1` listener
 * instead of exiting silently. Startup failures set a non-zero process
 * exit code after logging the rejection.
 */
if (isDirectDevServerExecution()) {
  void startDevServer().catch((error: Error) => {
    console.error(`neatRacing dev server failed to start: ${String(error)}`);
    process.exitCode = 1;
  });
}
