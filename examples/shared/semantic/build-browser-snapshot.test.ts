import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

// Tests run from the repository root, so cwd is a stable anchor for repo-relative
// paths without needing import.meta.url (which currently breaks ts-jest for tests
// in this project).
const repositoryRoot = path.resolve();

interface BrowserSnapshot {
  generated_at: string;
  schema_version: string;
  families: string[];
  documents: Array<{ doc_id: number; file_path: string; family: string }>;
}

describe('semantic browser snapshot', () => {
  describe('build-browser-snapshot.mjs', () => {
    it('emits valid JSON from a seeded SQLite index', async () => {
      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'semantic-browser-snapshot-'),
      );
      const databasePath = path.join(fixtureDirectory, 'rag-index.sqlite');
      const outputPath = path.join(fixtureDirectory, 'semantic-snapshot.json');

      execFileSync(
        process.execPath,
        ['--input-type=module', '--eval', buildSeedScript(databasePath)],
        {
          cwd: repositoryRoot,
        },
      );
      execFileSync(
        process.execPath,
        [
          'rag-index/build-browser-snapshot.mjs',
          '--database',
          databasePath,
          '--output',
          outputPath,
          '--json',
        ],
        { cwd: repositoryRoot },
      );

      const snapshot = JSON.parse(
        await readFile(outputPath, 'utf8'),
      ) as BrowserSnapshot;
      await rm(fixtureDirectory, { recursive: true, force: true });

      expect({
        generatedAtIsIso: !Number.isNaN(Date.parse(snapshot.generated_at)),
        schemaVersion: snapshot.schema_version,
        families: snapshot.families,
        documentCount: snapshot.documents.length,
      }).toEqual({
        generatedAtIsIso: true,
        schemaVersion: '1',
        families: ['readme'],
        documentCount: 1,
      });
    });
  });
});

function buildSeedScript(databasePath: string): string {
  return `
    import { createClient } from '@libsql/client';
    import { readFile } from 'node:fs/promises';
    import path from 'node:path';
    const databasePath = ${JSON.stringify(databasePath)};
    const schema = await readFile(path.resolve('rag-index/schema-turso.sql'), 'utf8');
    const database = createClient({ url: 'file:' + databasePath });
    try {
      await database.executeMultiple(schema);
      await database.execute({
        sql: 'INSERT INTO documents(file_path, doc_family, mtime_ms, file_size, sha256, indexed_at) VALUES (?, ?, ?, ?, ?, ?)',
        args: ['README.md', 'readme', 1, 12, 'fixture-sha', 2],
      });
      await database.execute({
        sql: 'INSERT INTO chunks(doc_id, chunk_index, heading_path, body_text, char_start, char_end) VALUES (?, ?, ?, ?, ?, ?)',
        args: [1, 0, '# NEAT', 'NEAT browser snapshot fixture.', 0, 30],
      });
    } finally {
      await database.close();
    }
  `;
}
