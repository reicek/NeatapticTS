/**
 * @description Run a BM25 full-text search against `data/semantic-index.sqlite` and print
 * ranked results. Accepts a positional query argument or `--query`. Optionally restricts
 * results to a single corpus family (readme, skill, agent, plan, ts-source, demo, …).
 *
 * @param {string}  [--query <text>]   - Query text (also accepted as positional argument).
 * @param {number}  [--limit <n>]      - Maximum result count (default: 10).
 * @param {string}  [--family <name>]  - Restrict results to one document family.
 * @param {boolean} [--json]           - Emit JSON results array.
 * @param {string}  [--database <path>] - Path to the SQLite database file (default: `data/semantic-index.sqlite`).
 * @param {boolean} [--help]           - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error. JSON results written to stdout when `--json` is passed.
 */
import Database from 'better-sqlite3';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fail, parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import { defaultDatabasePath } from './init-schema.mjs';

export function querySemanticIndex(options = {}) {
  const query = String(options.query ?? '').trim();
  if (!query) throw new Error('A query is required. Pass text positionally or with --query.');

  const limit = Math.max(1, Number(options.limit ?? 10));
  const database = new Database(path.resolve(options.databasePath ?? defaultDatabasePath), { readonly: true, fileMustExist: true });
  const familyFilter = options.family ? 'AND d.doc_family = @family' : '';
  const statement = database.prepare(`
    SELECT d.file_path, d.doc_family, c.heading_path, c.body_text, bm25(chunks_fts) AS score
    FROM chunks_fts
    JOIN chunks c ON c.chunk_id = chunks_fts.rowid
    JOIN documents d ON d.doc_id = c.doc_id
    WHERE chunks_fts MATCH @query ${familyFilter}
    ORDER BY score
    LIMIT @limit
  `);
  const rows = statement.all({ query, family: options.family, limit });
  database.close();
  return rows;
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Semantic index query',
      usage: 'node scripts/semantic-index/query-index.mjs "Network activation" [--limit 10] [--family readme] [--json]',
      options: ['--query <text>    Query text (alternative to positional argument)', '--limit <n>      Maximum result count (default: 10)', '--family <name>  Restrict results to one document family (readme, skill, agent, plan, demo, ...)', '--json           Emit JSON results', '--database <path> Path to SQLite database file (default: data/semantic-index.sqlite)', '--help           Show this help'],
    });
    return;
  }

  try {
    const rows = querySemanticIndex({ query: args.query ?? args._.join(' '), limit: args.limit, family: args.family, databasePath: args.database });
    writeJsonOrText(rows, Boolean(args.json), (payload) => payload.map((row, resultIndex) => `${resultIndex + 1}. ${row.file_path} [${row.doc_family}] ${row.heading_path ?? ''}`).join('\n'));
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) await main();