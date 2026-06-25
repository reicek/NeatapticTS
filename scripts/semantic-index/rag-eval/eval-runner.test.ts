/**
 * @module eval-runner.premium.test
 * @description Red tests for the premium primary search pipeline integration in the RAG eval runner.
 *
 * The eval runner must load and exercise the integrated `search_advanced` pipeline
 * with the premium LLM-facing defaults (compact, read_top_result, auto_fallback,
 * include_code_only for code_specific). Until Step 11 implementation lands, the
 * runner does not forward these defaults and the red contracts fail.
 */

import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

interface SearchCall {
  querySpec: Record<string, unknown>;
  conditionOptions: Record<string, unknown>;
}

const REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  '..',
);

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 *
 * Spawning a separate process avoids asking ts-jest to transform the ESM
 * implementation modules directly.
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

function makeQuery(overrides: Record<string, unknown> = {}) {
  return {
    query_id: 'sl-001',
    query: 'how does NEAT crossover work',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['readme'],
    expected_heading_contains: 'crossover',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: '',
    ...overrides,
  };
}

describe('eval-runner premium pipeline integration', () => {
  it('advanced_default passes compact, read_top_result, and auto_fallback defaults to the search function', () => {
    const query = makeQuery();
    const calls = runModuleEvaluation<SearchCall[]>(`
      import { runEval } from './scripts/semantic-index/eval-runner.mjs';
      const calls = [];
      const searchFn = (querySpec, conditionOptions) => {
        calls.push({ querySpec, conditionOptions });
        return [];
      };
      await runEval({ queries: ${JSON.stringify([query])}, condition: 'advanced_default', searchFn });
      console.log(JSON.stringify(calls));
    `);

    expect(calls).toHaveLength(1);
    expect(calls[0].conditionOptions).toEqual(
      expect.objectContaining({
        compact: true,
        read_top_result: true,
        auto_fallback: true,
      }),
    );
  });

  it('advanced_default passes include_code_only true for code_specific queries', () => {
    const query = makeQuery({
      query_id: 'cs-001',
      query: 'Network class architecture',
      class: 'code_specific',
      expected_doc_families: ['ts-source'],
    });
    const calls = runModuleEvaluation<SearchCall[]>(`
      import { runEval } from './scripts/semantic-index/eval-runner.mjs';
      const calls = [];
      const searchFn = (querySpec, conditionOptions) => {
        calls.push({ querySpec, conditionOptions });
        return [];
      };
      await runEval({ queries: ${JSON.stringify([query])}, condition: 'advanced_default', searchFn });
      console.log(JSON.stringify(calls));
    `);

    expect(calls).toHaveLength(1);
    expect(calls[0].conditionOptions).toEqual(
      expect.objectContaining({
        include_code_only: true,
      }),
    );
  });

  it('advanced_default result reports that premium defaults were applied', () => {
    const query = makeQuery();
    const result = runModuleEvaluation<Record<string, unknown>>(`
      import { runEval } from './scripts/semantic-index/eval-runner.mjs';
      const searchFn = () => [];
      const result = await runEval({ queries: ${JSON.stringify([query])}, condition: 'advanced_default', searchFn });
      console.log(JSON.stringify({ premium_defaults_applied: result.premium_defaults_applied }));
    `);

    expect(result.premium_defaults_applied).toBe(true);
  });
});
