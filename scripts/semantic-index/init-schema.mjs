import { createClient } from '@libsql/client';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
);
export const defaultDatabasePath = path.join(
  repoRoot,
  'data',
  'turso-replica.sqlite',
);

/**
 * Split a multi-statement SQL script into individual statements, handling
 * multi-line `CREATE TRIGGER` bodies that contain semicolons inside `BEGIN … END`.
 *
 * @param {string} sql - Raw SQL script text.
 * @returns {string[]} Array of trimmed individual SQL statements.
 */
function splitSqlStatements(sql) {
  const statements = [];
  let buffer = '';
  let inTrigger = false;
  for (const line of sql.split(/\r?\n/)) {
    const trimmedLine = line.trim();
    if (
      /^\s*(CREATE\s+(TEMP\s+|TEMPORARY\s+)?TRIGGER|CREATE\s+TRIGGER)/i.test(
        trimmedLine,
      ) &&
      /BEGIN\s*$/i.test(trimmedLine)
    ) {
      inTrigger = true;
    }
    buffer += line + '\n';
    if (inTrigger) {
      if (/^\s*END\s*;?\s*$/i.test(trimmedLine)) {
        inTrigger = false;
        const stmt = buffer.trim().replace(/;\s*$/, '');
        if (stmt) statements.push(stmt + ';');
        buffer = '';
      }
    } else if (trimmedLine.endsWith(';')) {
      const stmt = buffer.trim().replace(/;\s*$/, '');
      if (stmt) statements.push(stmt + ';');
      buffer = '';
    }
  }
  const tail = buffer.trim();
  if (tail) statements.push(tail);
  return statements;
}

/**
 * Apply a multi-statement SQL schema to a libSQL client via batch execution.
 *
 * @param {import('@libsql/client').Client} client - libSQL client instance.
 * @param {string} schemaSql - Raw SQL schema text.
 * @returns {Promise<void>}
 */
async function applySchemaToClient(client, schemaSql) {
  const statements = splitSqlStatements(schemaSql);
  const BATCH_GROUP_SIZE = 50;
  for (let i = 0; i < statements.length; i += BATCH_GROUP_SIZE) {
    const group = statements
      .slice(i, i + BATCH_GROUP_SIZE)
      .map((sql) => ({ sql, args: [] }));
    await client.batch(group, 'write');
  }
}

export async function initSemanticIndex(options = {}) {
  // When a libSQL client is provided, return it directly — the schema is
  // already initialized by the caller (e.g. createSchemaClient or init-turso).
  if (options.client) return options.client;

  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  await mkdir(path.dirname(databasePath), { recursive: true });

  const client = createClient({ url: pathToFileURL(databasePath).href });
  // Use schema-turso.sql, the production consolidated schema that merges the
  // corpus and embeddings tables into a single Turso/libSQL database.
  const schemaPath = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    'schema-turso.sql',
  );
  await applySchemaToClient(client, await readFile(schemaPath, 'utf8'));
  return client;
}
