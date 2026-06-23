/**
 * @module turso-pitr-mcp-tool
 * @description MCP tool handler for Turso point-in-time recovery (PITR) via the Turso Platform API.
 *
 * Creates a new Turso database from a timestamp by POSTing to the Turso
 * Platform API at `api.turso.tech`. The POST body carries a `seed` block
 * whose `type` is `"timestamp"` and whose `timestamp` field selects the
 * point in time to recover from. The source database is named via the
 * `seed.name` field.
 *
 * @remarks
 * The `timestamp` parameter must fall within the PITR retention window of
 * the source database. The `database_name` parameter selects the name of
 * the recovered database. Both operations require an organization, API
 * token, and the source database name.
 */

/** Base URL for the Turso Platform API. */
const TURSO_PLATFORM_API_BASE = 'https://api.turso.tech';

/**
 * Best-effort JSON read from a fetch response.
 *
 * Returns `null` when the body is empty or not valid JSON so callers can
 * build error messages without throwing on malformed payloads.
 *
 * @param {Response} response - Fetch response to read.
 * @returns {Promise<object | null>} Parsed JSON or null.
 */
async function safeReadJson(response) {
  try {
    const text = await response.text();
    if (!text) return null;
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * Create a Turso database from a timestamp via the Platform API (point-in-time recovery).
 *
 * POSTs to `/v1/organizations/{organization}/databases` with a body containing
 * the desired `name` and a `seed` block whose `type` is `"timestamp"`, `name`
 * is the source database, and `timestamp` is the recovery point.
 *
 * @param {object} params - PITR creation parameters.
 * @param {string} params.databaseName - Name of the recovered database.
 * @param {string} params.organization - Turso organization slug.
 * @param {string} params.apiToken - Turso Platform API token.
 * @param {string} params.sourceDatabaseName - Source database to recover from.
 * @param {string} params.timestamp - ISO-8601 timestamp within the PITR retention window.
 * @returns {Promise<object>} Recovery info returned by the Platform API, including the database name and hostname.
 * @throws {Error} when the Platform API returns a non-2xx status.
 */
async function createDatabaseFromTimestamp({
  databaseName,
  organization,
  apiToken,
  sourceDatabaseName,
  timestamp,
}) {
  const url = `${TURSO_PLATFORM_API_BASE}/v1/organizations/${organization}/databases`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: databaseName,
      seed: {
        type: 'timestamp',
        name: sourceDatabaseName,
        timestamp,
      },
    }),
  });

  if (!response.ok) {
    const errorBody = await safeReadJson(response);
    throw new Error(
      `Turso PITR creation failed: ${response.status} ${errorBody?.error ?? ''}`.trim(),
    );
  }

  const data = await response.json();
  return {
    databaseName: data.name ?? databaseName,
    hostname: data.hostname,
    raw: data,
  };
}

/**
 * Create a Turso database from a timestamp via the Turso Platform API (point-in-time recovery).
 *
 * POSTs to the Platform API with a `seed` block of type `"timestamp"` that
 * carries the recovery timestamp and the source database name. Returns the
 * recovered database name, hostname, and raw API response.
 *
 * @param {object} options - PITR operation options.
 * @param {string} options.databaseName - Name of the recovered database (`database_name`).
 * @param {string} options.organization - Turso organization slug.
 * @param {string} options.apiToken - Turso Platform API token.
 * @param {string} options.sourceDatabaseName - Source database to recover from.
 * @param {string} options.timestamp - ISO-8601 timestamp within the PITR retention window.
 * @returns {Promise<object>} PITR result with `databaseName`, `hostname`, and `raw` fields.
 * @throws {Error} when a required parameter is missing or the Platform API returns a non-2xx status.
 *
 * @example
 * ```js
 * const result = await tursoPitr({
 *   databaseName: 'corpus-pitr',
 *   organization: 'my-org',
 *   apiToken: process.env.TURSO_API_TOKEN,
 *   sourceDatabaseName: 'corpus',
 *   timestamp: '2026-01-01T00:00:00Z',
 * });
 * console.log(result.databaseName, result.hostname);
 * ```
 */
export async function tursoPitr({
  databaseName,
  organization,
  apiToken,
  sourceDatabaseName,
  timestamp,
} = {}) {
  if (!databaseName) {
    throw new Error('tursoPitr requires a databaseName');
  }
  if (!organization) {
    throw new Error('tursoPitr requires an organization');
  }
  if (!apiToken) {
    throw new Error('tursoPitr requires an apiToken');
  }
  if (!sourceDatabaseName) {
    throw new Error('tursoPitr requires a sourceDatabaseName');
  }
  if (!timestamp) {
    throw new Error('tursoPitr requires a timestamp');
  }

  return createDatabaseFromTimestamp({
    databaseName,
    organization,
    apiToken,
    sourceDatabaseName,
    timestamp,
  });
}
