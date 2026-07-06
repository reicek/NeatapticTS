import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const queriesPath = path.resolve('scripts/semantic-index/eval-queries.json');
const { searchCorpus } = await import(pathToFileURL(path.resolve('scripts/mcp-semantic/tools/search-corpus.mjs')).href);
const queries = JSON.parse(await readFile(queriesPath, 'utf8'));
const empty = [];
for (const q of queries) {
  const res = await searchCorpus({ query: q.query, limit: 5, use_dense: true });
  const results = Array.isArray(res) ? res : res.results;
  if (!Array.isArray(results) || results.length === 0) {
    empty.push(q.query_id);
  }
}
console.log(JSON.stringify({ query_count: queries.length, empty_queries: empty }));
