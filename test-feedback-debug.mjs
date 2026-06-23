import { createClient } from '@libsql/client';
import { splitSqlStatements, readCorpusSchema } from './scripts/mcp-semantic/__tests__/turso-test-helpers.mjs';

async function setupDb() {
  const client = createClient({ url: ':memory:' });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  return client;
}

async function insertDocumentAndChunk(client) {
  await client.execute({
    sql: `INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at) VALUES ('test.ts', 'src', 0, 0, 'abc', 0)`,
  });
  const docResult = await client.execute('SELECT doc_id FROM documents');
  const doc = docResult.rows[0];
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth) VALUES (?, 0, 'test body', 0, 9, 0)`,
    args: [doc.doc_id],
  });
  const chunkResult = await client.execute('SELECT chunk_id FROM chunks');
  return chunkResult.rows[0].chunk_id;
}

async function test() {
  const client = await setupDb();
  try {
    const chunkId = await insertDocumentAndChunk(client);
    
    const { recordFeedbackEventAsync, updateFeedbackScoresAsync } = await import('./scripts/mcp-semantic/tools/feedback-core.mjs');
    
    const t1 = Date.now();
    await recordFeedbackEventAsync(client, {
      chunk_id: chunkId,
      signal_type: 'positive',
      query_hash: 'hash123',
    });
    const t2 = Date.now();
    await recordFeedbackEventAsync(client, {
      chunk_id: chunkId,
      signal_type: 'negative',
      query_hash: 'hash123',
    });
    const t3 = Date.now();
    
    console.log('Time between events:', t2 - t1, 'ms');
    console.log('Time after events:', t3 - t2, 'ms');
    
    // Check created_at values
    const events = await client.execute({
      sql: 'SELECT event_id, signal_type, created_at FROM feedback_events WHERE chunk_id = ?',
      args: [chunkId],
    });
    for (const row of events.rows) {
      console.log('Event:', row.signal_type, 'created_at:', row.created_at, 'typeof:', typeof row.created_at);
    }
    
    const nowMs = Date.now();
    const score = await updateFeedbackScoresAsync(client, chunkId, nowMs);
    console.log('total_positive:', score.total_positive);
    console.log('total_negative:', score.total_negative);
    console.log('feedback_boost:', score.feedback_boost);
    console.log('boost < 0:', score.feedback_boost < 0);
  } finally {
    await client.close();
  }
}

test().catch(e => console.error(e));