/**
 * @module turso-branch-mcp-tool
 * @description MCP tool handler for Turso database branching via the Turso Platform API.
 *
 * Creates or deletes a branch of an existing Turso database so that index
 * changes can be tested in isolation without affecting the production
 * database. Branch creation uses the Turso Platform API at
 * `api.turso.tech` with a POST request whose body contains a `seed` block
 * that identifies the source database to branch from. Branch cleanup uses
 * a DELETE request to the same Platform API.
 *
 * @remarks
 * The `branch_name` parameter selects the name of the branch to create or
 * delete. The `action` parameter selects between `"create"` and `"delete"`.
 * Both operations require an organization, API token, and the source
 * database name.
 */

/** Base URL for the Turso Platform API. */
const TURSO_PLATFORM_API_BASE = 'https://api.turso.tech';

/**
 * Create a branch of an existing Turso database via the Platform API.
 *
 * POSTs to `/v1/organizations/{organization}/databases/{databaseName}/branches`
 * with a body containing the desired `branch` name and a `seed` block that
 * points at the source database.
 *
 * @param {object} params - Branch creation parameters.
 * @param {string} params.branchName - Name of the branch to create (the `branch_name`).
 * @param {string} params.organization - Turso organization slug.
 * @param {string} params.apiToken - Turso Platform API token.
 * @param {string} params.databaseName - Source database to branch from.
 * @returns {Promise<object>} Branch info returned by the Platform API, including the branch name and hostname.
 * @throws {Error} when the Platform API returns a non-2xx status.
 */
async function createBranch({
  branchName,
  organization,
  apiToken,
  databaseName,
}) {
  const url = `${TURSO_PLATFORM_API_BASE}/v1/organizations/${organization}/databases/${databaseName}/branches`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      branch: branchName,
      seed: {
        type: 'database',
        name: databaseName,
      },
    }),
  });

  if (!response.ok) {
    const errorBody = await safeReadJson(response);
    throw new Error(
      `Turso branch creation failed: ${response.status} ${errorBody?.error ?? ''}`.trim(),
    );
  }

  const data = await response.json();
  return {
    branchName: data.name ?? branchName,
    branch_name: data.name ?? branchName,
    hostname: data.hostname,
    raw: data,
  };
}

/**
 * Delete (cleanup) a branch of a Turso database via the Platform API.
 *
 * DELETEs `/v1/organizations/{organization}/databases/{databaseName}/branches/{branchName}`.
 *
 * @param {object} params - Branch deletion parameters.
 * @param {string} params.branchName - Name of the branch to delete (the `branch_name`).
 * @param {string} params.organization - Turso organization slug.
 * @param {string} params.apiToken - Turso Platform API token.
 * @param {string} params.databaseName - Source database the branch belongs to.
 * @returns {Promise<object>} Deletion result, including a `deleted` flag and the branch name.
 * @throws {Error} when the Platform API returns a non-2xx status.
 */
async function deleteBranch({
  branchName,
  organization,
  apiToken,
  databaseName,
}) {
  const url = `${TURSO_PLATFORM_API_BASE}/v1/organizations/${organization}/databases/${databaseName}/branches/${branchName}`;
  const response = await fetch(url, {
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${apiToken}`,
    },
  });

  if (!response.ok) {
    const errorBody = await safeReadJson(response);
    throw new Error(
      `Turso branch deletion failed: ${response.status} ${errorBody?.error ?? ''}`.trim(),
    );
  }

  const data = await safeReadJson(response);
  return {
    deleted: true,
    branchName,
    branch_name: branchName,
    raw: data ?? {},
  };
}

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
 * Create or delete a Turso database branch via the Turso Platform API.
 *
 * When `action` is `"create"` (the default), POSTs to the Platform API with
 * a `seed` block that branches from the source database. When `action` is
 * `"delete"`, DELETEs the named branch for cleanup.
 *
 * @param {object} options - Branch operation options.
 * @param {string} [options.action='create'] - Either `"create"` or `"delete"`.
 * @param {string} options.branchName - Name of the branch to create or delete (`branch_name`).
 * @param {string} options.organization - Turso organization slug.
 * @param {string} options.apiToken - Turso Platform API token.
 * @param {string} options.databaseName - Source database name.
 * @returns {Promise<object>} Branch creation or deletion result.
 * @throws {Error} when the action is unknown or the Platform API returns a non-2xx status.
 *
 * @example
 * ```js
 * const result = await tursoBranch({
 *   action: 'create',
 *   branchName: 'test-branch',
 *   organization: 'my-org',
 *   apiToken: process.env.TURSO_API_TOKEN,
 *   databaseName: 'corpus',
 * });
 * console.log(result.branchName, result.hostname);
 * ```
 */
export async function tursoBranch({
  action = 'create',
  branchName,
  organization,
  apiToken,
  databaseName,
} = {}) {
  if (!branchName) {
    throw new Error('tursoBranch requires a branch_name');
  }
  if (!organization) {
    throw new Error('tursoBranch requires an organization');
  }
  if (!apiToken) {
    throw new Error('tursoBranch requires an apiToken');
  }
  if (!databaseName) {
    throw new Error('tursoBranch requires a databaseName');
  }

  if (action === 'create') {
    return createBranch({ branchName, organization, apiToken, databaseName });
  }
  if (action === 'delete') {
    return deleteBranch({ branchName, organization, apiToken, databaseName });
  }
  throw new Error(`tursoBranch unknown action: ${action}`);
}
