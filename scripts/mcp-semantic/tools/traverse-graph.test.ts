/**
 * @module traverse-graph.test
 * @description Red tests for graph traversal serialization.
 *
 * The MCP tool currently emits malformed relationship objects when the graph
 * contains incomplete entity records. The symptom described in the plan is
 * `undefined --[undefined] --> undefined`. These tests pin down the contract
 * that every returned relationship must contain only well-formed, non-null
 * string fields.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';
import { createClient } from '@libsql/client';

const REPO_ROOT = path.resolve();

interface TraversalProbeResult {
  relationships: Array<{
    edge_id: number;
    source_entity_id: number;
    target_entity_id: number;
    source_qualified_name: string | null;
    target_qualified_name: string | null;
    source_entity_type: string | null;
    target_entity_type: string | null;
    relationship: string | null;
    confidence: string | null;
  }>;
  graph_available: boolean;
  total_discovered: number;
}

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 */
const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );
  const trimmed = output.trim();
  if (trimmed.length === 0) {
    throw new Error('Module evaluation produced empty output');
  }
  return JSON.parse(trimmed) as Result;
};

/**
 * Build a minimal entity/edge graph fixture with a deliberately malformed
 * entity record.
 *
 * The fixture has a valid seed function entity and a target function entity
 * whose `qualified_name` is NULL. An edge connects the two with a legal
 * relationship and confidence so the edge survives the current filters and
 * exposes the missing target qualified name in the serialized output.
 */
async function makeGraphFixture(): Promise<{ databasePath: string; tempDir: string }> {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'traverse-graph-red-'));
  const databasePath = path.join(tempDir, 'graph.sqlite');
  const db = createClient({ url: 'file:' + databasePath });
  try {
    await db.executeMultiple(`
      CREATE TABLE entities (
        entity_id INTEGER PRIMARY KEY,
        entity_type TEXT NOT NULL,
        name TEXT,
        qualified_name TEXT,
        doc_id INTEGER,
        chunk_id INTEGER,
        module_path TEXT,
        signature_text TEXT,
        file_path TEXT
      );
      CREATE TABLE edges (
        edge_id INTEGER PRIMARY KEY,
        source_entity_id INTEGER NOT NULL,
        target_entity_id INTEGER NOT NULL,
        relationship TEXT NOT NULL,
        confidence TEXT NOT NULL
      );
      INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, chunk_id, module_path, file_path)
        VALUES (1, 'function', 'activation', 'activationFunction', 1, 10, 'src/arch/network.ts', 'src/arch/network.ts');
      INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, chunk_id, module_path, file_path)
        VALUES (2, 'function', 'unnamedTarget', NULL, 1, 11, 'src/arch/network.ts', 'src/arch/network.ts');
      INSERT INTO edges (edge_id, source_entity_id, target_entity_id, relationship, confidence)
        VALUES (100, 1, 2, 'depends-on', 'high');
    `);
  } finally {
    await db.close();
  }
  return { databasePath, tempDir };
}

describe('traverse-graph serialization', () => {
  describe('relationship objects must be well-formed', () => {
    it('does not emit relationships with undefined or null qualified names', async () => {
      const { databasePath, tempDir } = await makeGraphFixture();
      try {
        const result = runModuleEvaluation<TraversalProbeResult>(`
          import { traverseGraph } from './scripts/mcp-semantic/tools/traverse-graph.mjs';
          const databasePath = ${JSON.stringify(databasePath)};
          const response = await traverseGraph({
            seed_names: ['activationFunction'],
            relationship_types: ['depends-on'],
            entity_types: ['function'],
            confidence_filter: ['high'],
            max_hops: 1,
            max_results: 10,
            databasePath,
          });
          console.log(JSON.stringify({
            relationships: response.relationships,
            graph_available: response.graph_available,
            total_discovered: response.total_discovered,
          }));
        `);

        const malformed = result.relationships.some(
          (relationship) =>
            relationship.source_qualified_name == null ||
            relationship.target_qualified_name == null ||
            relationship.source_entity_type == null ||
            relationship.target_entity_type == null ||
            relationship.relationship == null ||
            relationship.confidence == null,
        );

        expect(malformed).toBe(false);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('does not emit relationships with empty relationship or confidence strings', async () => {
      const { databasePath, tempDir } = await makeGraphFixture();
      try {
        const result = runModuleEvaluation<TraversalProbeResult>(`
          import { traverseGraph } from './scripts/mcp-semantic/tools/traverse-graph.mjs';
          const databasePath = ${JSON.stringify(databasePath)};
          const response = await traverseGraph({
            seed_names: ['activationFunction'],
            relationship_types: ['depends-on'],
            entity_types: ['function'],
            confidence_filter: ['high'],
            max_hops: 1,
            max_results: 10,
            databasePath,
          });
          console.log(JSON.stringify({
            relationships: response.relationships,
            graph_available: response.graph_available,
            total_discovered: response.total_discovered,
          }));
        `);

        const malformed = result.relationships.some(
          (relationship) =>
            typeof relationship.relationship !== 'string' ||
            relationship.relationship.length === 0 ||
            typeof relationship.confidence !== 'string' ||
            relationship.confidence.length === 0,
        );

        expect(malformed).toBe(false);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore.
        }
      }
    });
  });
});
