import Database from 'better-sqlite3';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
export const defaultDatabasePath = path.join(repoRoot, 'data', 'semantic-index.sqlite');

export async function initSemanticIndex(options = {}) {
  const databasePath = path.resolve(options.databasePath ?? defaultDatabasePath);
  await mkdir(path.dirname(databasePath), { recursive: true });

  const database = new Database(databasePath);
  database.pragma('foreign_keys = ON');
  const schemaPath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'schema.sql');
  database.exec(await readFile(schemaPath, 'utf8'));
  return database;
}