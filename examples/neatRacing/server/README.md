# server

Hardened zero-dependency development server for the Neat Racing demo
(spec section 8.2, AC-018; slice P1-S5-B, step 08).

The server is a thin HTTP shell over the landed model store
(`model-store.utils.ts`, slice P1-S5-A) and serves the demo pages/assets
from `examples/neatRacing/`. Every request passes through the six
normative hardening gates before any handler logic runs:

- **S1 host/origin** â€” every request needs `Host: 127.0.0.1:<port>` or
  `localhost:<port>`; state-changing requests additionally require a
  matching `Origin` when one is present; `Access-Control-*` headers are
  never emitted, so devtools/hostile pages cannot drive saves, rebuilds,
  or validations.
- **S2 rebuild pinning** â€” rebuilds only ever run the fixed bins script
  through `spawn(process.execPath, [<script>, <name>])` with
  `shell: false`; a per-model single-flight coordinator coalesces
  concurrent triggers latest-wins (a superseded in-flight rebuild's
  completion is discarded, the newest trigger starts fresh).
- **S3 body limits** â€” body-bearing PUT/POST require
  `application/json` (params tolerated), reject declared or actual sizes
  above `NEATRACING_MAX_SAVE_BODY_BYTES`, and reject malformed JSON.
- **S4 name shape** â€” `:name` segments are decoded and checked against
  the store's name pattern before any path join, so traversal payloads
  (`../evil`, `%2e%2e`) are structurally rejected.
- **S5 method + hygiene** â€” per-route method allowlists (405 otherwise),
  static serving confined to declared prefixes with resolve containment,
  no directory listings, and error bodies are exactly `{status, code}`.
- **S6 backup rotation** â€” delegated to the store's `bak1..3` ring.

Execution model: request handling is a pure function
(`handleDevServerRequest`) from `DevRequestIntent` to `DevResponse` so the
whole hardening surface is unit-testable without sockets. The store
module is loaded lazily through `loadDefaultDevServerContext` (dynamic
import + a resolve-hook shim registered only outside Jest) because the
runtime process is executed by ts-node in native ESM mode, where
extensionless specifiers inside the store's own import graph fail unless
the shim retries them with an explicit `.ts`. The single I/O seam the
pure core touches is the injected store API object plus the optional
injected rebuild spawner, keeping everything else pure dispatch.

The module is run by the `dev:neatRacing` package script (spec section
12.4: `ts-node examples/neatRacing/server/dev-server.ts`) through a
module-level direct-run guard: when this file is the Node entry script it
calls `startDevServer()` itself and keeps the process serving, so the CLI
path exposes a real listener instead of exiting silently. Plain imports
(Jest, tooling) stay side-effect free. The server binds
`127.0.0.1:NEATRACING_DEV_SERVER_PORT` (default 8787).

## server/dev-server.ts

### addressPort

```ts
addressPort(
  server: Server<typeof IncomingMessage, typeof ServerResponse>,
): number
```

Resolves the bound port from a listening server.

Parameters:
- `server` - The HTTP server.

Returns: The port, or 0 when not yet listening.

### apiSegments

```ts
apiSegments(
  rawPath: string,
  prefix: string,
): string[]
```

Extracts the raw path segments of an `/api/...` route after the prefix.

Parameters:
- `rawPath` - Percent-encoded request path.
- `prefix` - Exact literal prefix, for example `/api/models/`.

Returns: The raw segments (not yet decoded).

### assetContentType

```ts
assetContentType(
  extension: string,
): string
```

Maps a file extension to the served content type.

Parameters:
- `extension` - Lowercase extension including the dot.

Returns: The content type; unknown extensions fall back to octet-stream.

### assetResponse

```ts
assetResponse(
  filePath: string,
  stats: Stats,
): DevResponse
```

Serves a raw file body with a content type derived from its extension.

Parameters:
- `filePath` - Absolute file path.
- `stats` - File stats (size sanity check for the read).

Returns: The pure asset response.

### buildDevBundles

```ts
buildDevBundles(
  devRoot: string,
  assetsDir: string,
): void
```

Compiles the demo and editor bundles into the served asset locations.

Per spec section 8.2 the dev server rebuilds the browser bundle on every
start through the esbuild API; entries that a pending slice has not
landed yet are skipped with a console note so startup never crashes.
The bundle paths form the fixed contract with the host pages:
`assets/neatracing.bundle.js` (root page) and
`editor/editor.bundle.js` (editor page).

Parameters:
- `devRoot` - The dev root directory.
- `assetsDir` - The assets output directory.

### buildRebuildSpawnArgv

```ts
buildRebuildSpawnArgv(
  rebuildScriptPath: string,
  modelName: string,
): string[]
```

Builds the fixed rebuild argv for the bins script (S2 pin): the absolute
script path first, then the model name as the sole argument.

Parameters:
- `rebuildScriptPath` - Absolute path of `build-voxel-bins.mjs`.
- `modelName` - Validated model name to rebuild.

Returns: The argv array (`[<script>, <name>]`).

### createDefaultRebuildSpawner

```ts
createDefaultRebuildSpawner(
  rebuildScriptPath: string,
): DevRebuildSpawner
```

Creates the default rebuild spawner, pinning the bins script with the
node interpreter, no shell, and inherited stdio (S2).

Parameters:
- `rebuildScriptPath` - Absolute path of the bins script.

Returns: A spawner suitable for `createDevServerContext`.

### createDevServerContext

```ts
createDevServerContext(
  config: { port: number; modelsDir: string; devRoot: string; assetsDir: string; pages: Readonly<Record<string, string>>; spawner: DevRebuildSpawner; maxBodyBytes?: number | undefined; store?: DevModelStoreApi | undefined; },
): DevServerContext
```

Pure context factory: wires configuration to the injected store API.

Callers that need the real store use `loadDefaultDevServerContext`, which
performs the lazy dynamic import. Tests inject a stub object instead, so
context construction stays synchronous and free of module resolution.

Parameters:
- `config` - Explicitly provided configuration and store API.

Returns: The wired context.

### decodeSegment

```ts
decodeSegment(
  segment: string,
): string | null
```

Decodes one raw path segment, re-routing malformed percent-escapes to the
caller through `null` so the dispatcher can reject them as bad names.

Parameters:
- `segment` - Raw (encoded) segment.

Returns: The decoded segment, or `null` when undecodable.

### defaultPages

```ts
defaultPages(): Record<string, string>
```

Builds the page file map served from the dev root.

`editor.html` may not exist until its slice lands; the page handler is
existence-gated so the route degrades to a clean 404 meanwhile.

Returns: The static route-to-relative-file map.

### DevModelStoreApi

The store API slice the dispatcher consumes at runtime; loaded lazily so
the module graph stays resolvable under ts-node's native ESM execution.

### DevRebuildRecord

Single-flight record for one in-flight rebuild (S2 coalescing).

### DevRebuildSpawner

```ts
DevRebuildSpawner(
  modelName: string,
  onSettled: (error: Error | null) => void,
): void
```

Spawns one rebuild flight for a model; the completion callback settles
the flight (success, failure, or spawn error) with the coordinator.

### DevRequestIntent

A single request as observed by the HTTP front end, normalized for the
pure dispatcher.

### DevResponse

A complete response produced by the pure dispatcher.

### DevServerContext

A fully wired server context: everything `handleDevServerRequest` needs.

### DevServerContextConfig

Configuration accepted by the pure context factory.

### DevServerContextOverrides

Optional overrides accepted by the default (runtime) context loader.

### DevServerErrorCode

Machine-readable rejection codes for server-level gates. Store-level
rejections forward their own `model-store/â€¦` and `voxel-model/â€¦` codes.

### DevServerHandle

Server handle returned by `startDevServer`.

### DevServerStartConfig

Starts the HTTP runtime over a context: builds the browser bundles (when
enabled), binds `127.0.0.1` on the resolved port, and funnels every
connection through the pure dispatcher with runtime body accounting.

Parameters:
- `config` - Start configuration (all fields optional).
- `config` - Port override; defaults to the env-resolved port.
- `config` - Dev root override for page/static resolution.
- `config` - Model directory override.
- `config` - Build the demo/editor bundles on start.
- `config` - Prebuilt context (tests); skips store loading.

Returns: A promise resolving to the server handle.

### ensureRuntimeResolveShim

```ts
ensureRuntimeResolveShim(): void
```

Registers the `.ts` resolve retry shim exactly once, outside Jest.

ts-node in native ESM mode (repo `type: module`) cannot resolve the
dotted-name specifiers inside the store's own import graph; the hook
retries `ERR_MODULE_NOT_FOUND` with an explicit `.ts` suffix for every
specifier that does not end with a real runtime extension. The hook
source is registered as a `data:` URL against the running script's URL,
so no temporary file or `import.meta` reference is needed (Jest safe).

Under Jest (`JEST_WORKER_ID` defined) this is a no-op: ts-jest resolves
the extensionless graph itself and a module hook would fight the
transformer.

### forwardStoreError

```ts
forwardStoreError(
  error: unknown,
): DevResponse
```

Forwards a store/schema rejection as an HTTP failure response.

Store codes are namespaced and stable, so they are forwarded verbatim in
the `{status, code}` body; unknown failures fall back to a generic
rejection that leaks nothing (S5).

Parameters:
- `error` - Caught rejection.

Returns: The `{status, code}` failure response.

### gateRequestBody

```ts
gateRequestBody(
  context: DevServerContext,
  intent: DevRequestIntent,
): DevResponse | undefined
```

Applies the S3 size gates for body-bearing routes.

Parameters:
- `context` - Wired server context.
- `intent` - Normalized request intent.

Returns: A 413 response when over budget, else `undefined`.

### gateRequestContentType

```ts
gateRequestContentType(
  intent: DevRequestIntent,
): DevResponse | undefined
```

Enforces the S3 content-type gate for body-bearing routes.

Mirrors `gateRequestBody` placement so callers can reject a hostile
`Content-Type` before touching the payload.

Parameters:
- `intent` - Normalized request intent.

Returns: A 415 response when the content type is not JSON, else `undefined`.

### getPagePrefixStaticRoutes

```ts
getPagePrefixStaticRoutes(
  context: DevServerContext,
): Map<string, string>
```

Derives page-adjacent static routes from the page map: a page route ending
with a slash (for example `/editor/`) also serves sibling files from that
page's directory, so `editor.bundle.js` can load next to the page.

Parameters:
- `context` - Wired server context.

Returns: Prefix, to absolute root-directory entries.

### handleDevServerRequest

```ts
handleDevServerRequest(
  context: DevServerContext,
  intent: DevRequestIntent,
): DevResponse
```

The pure dispatcher: maps one normalized intent to one response, applying
the S1â€“S6 gates in a fixed order before any I/O or store call runs.

Gate order: host (S1) â†’ global `OPTIONS` rejection (S5) â†’ origin for
state-changing methods (S1) â†’ route/method resolution (S5/S4) â†’ name
shape against the store pattern (S4) â†’ body gates (S3) â†’ handler
delegation to the landed store (S6 rotation inside the store).

Parameters:
- `context` - Wired server context.
- `intent` - Normalized request intent.

Returns: The complete response.

### handleListModels

```ts
handleListModels(
  context: DevServerContext,
): DevResponse
```

Handles the `GET /api/models` listing route.

Parameters:
- `context` - Wired server context.

Returns: The `{status, code, models}` response, or a store rejection.

### handleNamedApiRoute

```ts
handleNamedApiRoute(
  context: DevServerContext,
  intent: DevRequestIntent,
  rawName: string,
  action: string | undefined,
): DevResponse
```

Handles `/api/models/:name` and its `/rebuild` and `/validate` actions.

Shared gating: S4 name shape (structural, before any join), then the
enumeration check (404 for PUT/GET/actions against names the store does
not list), then per-route method allowlists, then route-specific bodies.

Parameters:
- `context` - Wired server context.
- `intent` - Normalized request intent.
- `rawName` - Raw (encoded) name segment.
- `action` - Optional action tail (`rebuild` or `validate`).

Returns: The route-specific response.

### handleReadModel

```ts
handleReadModel(
  context: DevServerContext,
  name: string,
): DevResponse
```

Serves one stored document through the store's read+validate path.

Parameters:
- `context` - Wired server context.
- `name` - Validated model name.

Returns: The document payload, or 404 for unenumerated names, or a
forwarded store rejection for store-level failures.

### handleRebuild

```ts
handleRebuild(
  context: DevServerContext,
  name: string,
): DevResponse
```

Triggers (or coalesces) one rebuild flight through the S2-pinned spawner.

Parameters:
- `context` - Wired server context.
- `name` - Validated, enumerated model name.

Returns: The 202 start/coalesce acknowledgement.

### handleSaveModel

```ts
handleSaveModel(
  context: DevServerContext,
  intent: DevRequestIntent,
  name: string,
): DevResponse
```

Saves one document through the store, enforcing the S3 body gates and the
S4 enumeration gate (PUT never creates models) before the write.

Parameters:
- `context` - Wired server context.
- `intent` - Normalized request intent.
- `name` - Validated model name.

Returns: The save acknowledgement, or a gate rejection.

### handleStaticAsset

```ts
handleStaticAsset(
  context: DevServerContext,
  intent: DevRequestIntent,
  prefix: string,
  assetsDir: string,
): DevResponse
```

Serves one static asset under a fixed prefix with resolve containment
(S5): decoded segments may not traverse, the resolved path must stay
inside the root, and directories are never listed (404 instead).

Parameters:
- `context` - Wired server context.
- `intent` - Normalized request intent.
- `prefix` - The static prefix to serve, including the trailing slash.

Returns: The asset response, 403 for traversal, 404 for missing/listing.

### handleValidate

```ts
handleValidate(
  context: DevServerContext,
  intent: DevRequestIntent,
): DevResponse
```

Validates one document candidate without saving (editor preflight).

Parameters:
- `context` - Wired server context.
- `intent` - Normalized request intent.

Returns: The validation acknowledgement, or a forwarded rejection.

### htmlResponse

```ts
htmlResponse(
  body: string,
): DevResponse
```

Builds an HTML page response.

Parameters:
- `body` - The page markup.

Returns: The pure page response.

### isAllowedDevServerHost

```ts
isAllowedDevServerHost(
  host: unknown,
  port: number,
): boolean
```

S1 host gate: exactly `127.0.0.1:<port>` or `localhost:<port>` passes.

Parameters:
- `host` - Raw `Host` header value (may be `undefined`).
- `port` - Port this dev server instance is serving.

Returns: `true` exactly when the host pair matches the loopback allowlist.

### isAllowedDevServerOrigin

```ts
isAllowedDevServerOrigin(
  origin: unknown,
  port: number,
): boolean
```

S1 origin gate for state-changing requests: an `Origin`, when present,
must originate from the loopback pages this server serves; a missing
`Origin` (curl, tests) also passes. The scheme is pinned to `http:`
because the dev server itself serves plaintext HTTP.

Parameters:
- `origin` - Raw `Origin` header value (may be `undefined`).
- `port` - Port this dev server instance serves.

Returns: `true` exactly when the origin is empty or allowlisted.

### isDirectDevServerExecution

```ts
isDirectDevServerExecution(): boolean
```

True when this module is the Node entry script — the `dev:neatRacing`
CLI path (spec 12.4: `ts-node examples/neatRacing/server/dev-server.ts`).

The comparison is entry-path based and module-agnostic: `import.meta.url`
cannot be used here because the Jest lanes compile this module with
`module=commonjs`, where `import.meta` is rejected with TS1343. Two
entry shapes are recognized:
- this script itself is `process.argv[1]` (the executing entry), or
- the `ts-node` CLI bin is `process.argv[1]` with this script as its
  first CLI argument (the CLI's own entry shape).
Importing the module (Jest, tooling) matches neither shape, so imports
stay side-effect free.

Returns: `true` when Node launched this file directly as the entry.

### joinChunks

```ts
joinChunks(
  chunks: readonly Uint8Array<ArrayBufferLike>[],
): Buffer<ArrayBufferLike>
```

Concatenates body chunks into one buffer.

Parameters:
- `chunks` - The collected chunks.

Returns: The concatenated buffer.

### jsonError

```ts
jsonError(
  status: number,
  code: string,
): DevResponse
```

Builds a failure response carrying exactly `{status, code}` (S5 hygiene:
no filesystem paths, messages, or stacks in the payload).

Parameters:
- `status` - HTTP status.
- `code` - Machine-readable code.

Returns: The pure error response.

### jsonOk

```ts
jsonOk(
  status: number,
  code: string,
  extra: Readonly<Record<string, unknown>>,
): DevResponse
```

Builds a success response envelope.

Parameters:
- `status` - HTTP status.
- `code` - Machine-readable success code.
- `extra` - Additional payload fields merged into the body.

Returns: The pure success response.

### loadDefaultDevServerContext

```ts
loadDefaultDevServerContext(
  overrides: DevServerContextOverrides,
): Promise<DevServerContext>
```

Runtime context factory: registers the `.ts` resolve shim outside Jest,
then lazily imports the landed store module.

The dynamic import is required because the store's own graph contains
dotted-name specifiers, which native ESM execution (ts-node under
`type: module`) cannot resolve without the shim; the shim is installed
(plus one macrotask so Node's async resolve-hook chain is in place)
before the import and applies to the store's sub-graph only.

Parameters:
- `overrides` - Overrides for paths, port, or the spawner.

Returns: A promise resolving to the fully wired context.

### NEATRACING_DEV_SERVER_DEFAULT_PORT

Default dev-server port, overridable through the env var below (S-port).

### NEATRACING_DEV_SERVER_PORT_ENV_NAME

Environment variable consulted by `resolveDevServerPort`.

### NEATRACING_MAX_SAVE_BODY_BYTES

Hard cap for any request body (spec S3: 8 MiB).

### parseContentLength

```ts
parseContentLength(
  raw: unknown,
): number | undefined
```

Parses a declared `Content-Length` header.

Parameters:
- `raw` - Header value (may be `undefined`).

Returns: The parsed length, or `undefined` when absent or unparseable.

### parseJsonBody

```ts
parseJsonBody(
  intent: DevRequestIntent,
): unknown
```

Parses the request body as JSON.

Parameters:
- `intent` - Normalized request intent.

Returns: The parsed JSON value.

### readIntent

```ts
readIntent(
  request: IncomingMessage,
  chunks: readonly Uint8Array<ArrayBufferLike>[],
): DevRequestIntent
```

Normalizes one live request into a pure `DevRequestIntent`.

Parameters:
- `request` - Incoming request stream.
- `chunks` - Collected body chunks.

Returns: The pure intent (query string stripped, headers lowercased).

### readTextFile

```ts
readTextFile(
  filePath: string,
): string
```

Reads a text file, tolerating an existing-but-unreadable page by letting
the runtime error handler translate the failure to a 500.

Parameters:
- `filePath` - Absolute file path.

Returns: The file text.

### RebuildCoordinator

Coalesces in-flight rebuild triggers per model, latest-wins (S2): a newer
`begin` marks the previous record superseded, and `complete` reports
whether the settled flight's result is still meaningful.

#### #active

Active flight per model name.

#### begin

```ts
begin(
  name: string,
): DevRebuildRecord
```

Opens (or takes over) the rebuild flight for one model.

A previous in-flight record is marked superseded: its eventual
completion is discarded, and the caller must not treat its output as
applied. The returned record identifies THIS flight.

Parameters:
- `name` - Validated model name.

Returns: The fresh, non-superseded record for the new flight.

#### complete

```ts
complete(
  record: DevRebuildRecord,
): "applied" | "discarded"
```

Settles a started flight.

Parameters:
- `record` - The record returned by `begin` for this flight.

Returns: `'discarded'` when a newer flight took over the model,
`'applied'` when this was the newest flight for the model.

#### isActive

```ts
isActive(
  name: string,
): boolean
```

Reports whether a model currently has an active (non-superseded) flight.

Parameters:
- `name` - Model name to probe.

Returns: `true` while the model's newest flight is unresolved.

### resolveDevServerPort

```ts
resolveDevServerPort(
  env: Readonly<Record<string, string | undefined>>,
): number
```

Resolves the dev-server port from an environment record.

Accepts integers in the `1..65535` range; anything else (missing, garbage,
out of range, fractional) falls back to the default 8787.

Parameters:
- `env` - Environment map, for example `process.env`.

Returns: The resolved port.

### respond

```ts
respond(
  response: ServerResponse<IncomingMessage>,
  outcome: DevResponse,
): void
```

Serializes and writes one `DevResponse` to the live response.

Parameters:
- `response` - Live server response.
- `outcome` - Pure response outcome.

### respondJson

```ts
respondJson(
  response: ServerResponse<IncomingMessage>,
  outcome: DevResponse,
): void
```

Writes a JSON error response directly to a live response.

Parameters:
- `response` - Live server response.
- `outcome` - Pure error response.

### serveConnection

```ts
serveConnection(
  context: DevServerContext,
  request: IncomingMessage,
  response: ServerResponse<IncomingMessage>,
): void
```

Serves one live HTTP connection through the pure dispatcher.

Body bytes are accumulated up to the cap with an early 413 short-circuit;
once the body is fully read the intent is dispatched and the response
serialized. Connection-level failures produce exactly `{status, code}`.

Parameters:
- `context` - The wired server context.
- `request` - Incoming request stream.
- `response` - Server response stream.

### startDevServer

```ts
startDevServer(
  config: DevServerStartConfig,
): Promise<DevServerHandle>
```

Starts the hardened dev server.

Parameters:
- `config` - Optional start configuration.

Returns: A promise resolving to the server handle.

## server/model-store.utils.ts

File-system model store for the Neat Racing dev server (spec section 8.2,
AC-018 support + AC-007 support; slice P1-S5-A).

Pure CRUD over `examples/neatRacing/assets/models/*.model.json` with the
section 8.2 hardening gates applied at the store boundary:

- **S4 model-name shape** — `^[a-z0-9][a-z0-9-]{0,63}$` is checked on the
  decoded name *before any path join*, so `/`, `\`, `..`, and `%` can never
  reach the filesystem; traversal rejection is structural.
- **S5 strict payload keys** — accepted documents carry exactly the
  `meta`, `palette`, and `slices` top-level keys; anything else (including
  non-object payloads) is rejected with `model-store/top-level-keys`
  before a single byte is written.
- **Atomic save + S6 backup rotation** — writes go to a
  `<name>.model.json.tmp` sibling and are swapped in with a same-directory
  rename (atomic on both Windows and POSIX), then the replaced document is
  rotated through `<modelsDir>/.backups/<name>.model.json.bak1..3` so the
  last three versions survive an editor crash.
- **Section 7.2 schema gate** — every document read or written is validated
  by `voxel/model.schema.ts` and row-decoded by
  `voxel/model.decode.utils.ts`, so on-disk payloads round-trip the exact
  schema contract (AC-007) without ever mutating the caller's objects.

Storage symmetry note: mirror-X symmetric documents store half-widths
(dims[0] / 2 hex columns per row); the store is format-agnostic about this
and delegates the width arithmetic to the row decoder.

### deleteVoxelModel

```ts
deleteVoxelModel(
  modelsDir: string,
  name: unknown,
): void
```

Deletes one stored model document.

Backup copies are intentionally kept: they are the crash-recovery net for
the editor and are only consumed by a human restore.

Parameters:
- `modelsDir` - Directory holding the stored `*.model.json` documents.
- `name` - Candidate model name (untrusted; shape-checked first).

### describePayload

```ts
describePayload(
  candidate: unknown,
): string
```

Renders an untrusted payload for rejection messages without revealing more
than its JSON kind.

Parameters:
- `candidate` - The rejected payload.

Returns: A short, deterministic description.

### isVoxelModelName

```ts
isVoxelModelName(
  name: unknown,
): boolean
```

Reports whether `name` satisfies the S4 model-name shape.

Accepts `unknown` so dev-server handlers can gate untrusted input without
pre-narrowing; non-strings (including `undefined` from URL parsing) are
rejected, never thrown.

Parameters:
- `name` - Candidate model name from any source.

Returns: `true` exactly when the name is store-safe.

### listVoxelModelNames

```ts
listVoxelModelNames(
  modelsDir: string,
): string[]
```

Lists the stored model names, sorted, ignoring foreign files.

Only regular files whose stem satisfies the S4 shape are reported; helper
files (`*.json` payloads, temp files), symlinks (whose `Dirent` entries are
not `isFile()`), and subdirectories (including `.backups/`) are skipped.
A missing directory is not an error — an empty install simply has no
models.

Parameters:
- `modelsDir` - Directory holding the stored `*.model.json` documents.

Returns: Valid model names in lexicographic order; `[]` when the directory
is missing or empty.

### ModelStoreError

Named store rejection: name-shape violations, missing files, serialization
failures, or S5 key-set violations. Extends `VoxelModelError` so server
handlers can catch the shared base class and forward `error.code` as an
HTTP status hint.

### ModelStoreErrorCode

Stable store rejection codes, namespaced under `model-store/` to keep them
distinguishable from the `voxel-model/…` schema codes they complement.

### NEATRACING_MODEL_FILE_SUFFIX

Reserved file suffix for Neat Racing model documents (section 10.2).

### NEATRACING_MODEL_NAME_PATTERN

S4 model-name shape: lowercase alphanumerics with inner hyphens, one
leading character, at most 64 characters total. The pattern cannot match
path separators, `%`, or leading dots/dashes, so name checks performed
before any `join` reject traversal payloads structurally (no `realpath`
dance needed).

### NEATRACING_MODEL_STORE_BACKUP_COUNT

Number of replaced versions kept per model in the backup directory (S6).

### NEATRACING_MODEL_STORE_BACKUP_DIR_NAME

Directory (inside the models directory) holding rotated backup copies.

### NEATRACING_MODEL_STORE_TEMP_SUFFIX

Suffix appended to the model file while an atomic save is in flight.

### readStoredBytes

```ts
readStoredBytes(
  filePath: string,
): string | undefined
```

Reads the stored bytes for pre-backup snapshotting.

Parameters:
- `filePath` - The model document path.

Returns: The previous file content, or `undefined` when no document is
stored yet.

### readVoxelModelDocument

```ts
readVoxelModelDocument(
  modelsDir: string,
  name: unknown,
): VoxelModelDocument
```

Reads and validates one stored model document.

Order of gates: S4 name shape (`model-store/name`), file presence
(`model-store/not-found`), JSON parseability and schema/row validity
(section 7.2 errors propagate). The stored bytes are never modified.

Parameters:
- `modelsDir` - Directory holding the stored `*.model.json` documents.
- `name` - Candidate model name (untrusted; shape-checked first).

Returns: The validated section 7.2 document.

### removeQuietly

```ts
removeQuietly(
  filePath: string,
): void
```

Removes a temp file, tolerating its absence.

Parameters:
- `filePath` - The temp file path.

### resolveVoxelModelFilePath

```ts
resolveVoxelModelFilePath(
  modelsDir: string,
  name: unknown,
): string
```

Builds the on-disk document path for a validated model name.

The name is re-checked here (instead of trusting callers) because this is
also the join boundary: by the time any path is assembled, the name is
proven incapable of escaping the models directory.

Parameters:
- `modelsDir` - Directory holding the stored `*.model.json` documents.
- `name` - Candidate model name (validated against the S4 shape).

Returns: The absolute path of the model document.

### rotateModelBackups

```ts
rotateModelBackups(
  modelsDir: string,
  name: string,
  replacedContent: string,
): void
```

Rotates the replaced document through the S6 backup ring.

Shifts oldest-first (`bak2` into `bak3`, then `bak1` into `bak2`) so each
`rename` lands in a free slot, then writes the replaced content as the new
`bak1`. Renames within the same directory are atomic; a rename onto an
existing slot replaces it, which is exactly how the 4th-oldest version is
dropped once the ring is full.

Parameters:
- `modelsDir` - Directory holding the stored `*.model.json` documents.
- `name` - Validated model name.
- `replacedContent` - The pre-capture content of the replaced document.

### validateVoxelModelPayload

```ts
validateVoxelModelPayload(
  candidate: unknown,
): VoxelModelDocument
```

Validates an untrusted payload against the S5 strict key set before the
section 7.2 schema gate runs.

Non-records (including arrays) and payloads holding any key beyond the
strict `meta`/`palette`/`slices` enumeration are rejected up front so the
store never persists editor-forged envelopes. Schema and row-level errors
are then delegated to the shared validators and propagate unchanged.

Parameters:
- `candidate` - Untrusted JSON value, for example a PUT request body.

Returns: The same document, narrowed to the section 7.2 schema type; the
input object is returned (not a clone) and never mutated.

### writeVoxelModelDocument

```ts
writeVoxelModelDocument(
  modelsDir: string,
  name: unknown,
  candidate: unknown,
): void
```

Atomically saves a model document, rotating the previous version into the
S6 backup ring.

Validation runs entirely before the models directory is touched, so
rejected payloads leave any stored document and the `.backups/` ring
untouched. The serialized document is written to a `.tmp` sibling and
swapped in with a same-directory rename (replace-existing on Windows,
POSIX `rename`), which is atomic against concurrent readers; only after a
successful swap is the pre-captured previous content rotated through
`bak1..3`, oldest first, so `bak1` always holds the immediately replaced
version. A first save (no previous content) creates no backup directory.

Parameters:
- `modelsDir` - Directory holding the stored `*.model.json` documents;
created on demand.
- `name` - Candidate model name (untrusted; shape-checked first).
- `candidate` - Untrusted JSON payload; never mutated on success or
rejection.
