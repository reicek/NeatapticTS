#!/usr/bin/env node

/**
 * @module eval-classification
 * @description Evaluation runner for query classification accuracy and routing effectiveness.
 *
 * Runs three evaluations:
 *
 * 1. **Classification accuracy** — runs `classifyQuery` on a curated eval set and
 *    compares each result to the expected class.
 * 2. **Routing effectiveness** — verifies per-class alpha defaults match the
 *    canonical routing table.
 * 3. **Latency** — measures classification time and confirms it is < 5 ms.
 *
 * Run: `node scripts/semantic-index/eval-classification.mjs --json`
 *
 * @example
 * ```bash
 * node scripts/semantic-index/eval-classification.mjs --json
 * ```
 */

import { classifyQuery, classifyForSearchCorpus } from './classify-query.mjs';
import { classifyAndRoute, DEFAULTS, ROUTING } from './routing-table.mjs';

// ---------------------------------------------------------------------------
// Eval query set — 12 queries, 2 per class, with ground-truth classifications
// ---------------------------------------------------------------------------

/** @type {ReadonlyArray<{query: string, expected_class: string, confidence_min?: number}>} */
const EVAL_QUERIES = Object.freeze([
  // simple_lookup (short, specific)
  { query: 'Network.activate', expected_class: 'simple_lookup' },
  { query: 'mutate add node', expected_class: 'simple_lookup' },

  // cross_boundary (cross-family concepts)
  {
    query: 'how does the NEAT evolve method use speciation',
    expected_class: 'cross_boundary',
  },
  {
    query: 'relationship between crossover and mutation',
    expected_class: 'cross_boundary',
  },

  // multi_hop (iterative retrieval — must contain multi-hop connective patterns)
  {
    query:
      'what functions call Network.activate that also use the slab fast path',
    expected_class: 'multi_hop',
  },
  {
    query:
      'which mutation operators that also affect connection genes and then change speciation',
    expected_class: 'multi_hop',
  },

  // exploratory (broad overview)
  {
    query: 'how does the training pipeline work',
    expected_class: 'exploratory',
  },
  { query: 'explain the NEAT algorithm', expected_class: 'exploratory' },

  // code_specific (implementation details)
  {
    query: 'implementation of crossover in NEAT',
    expected_class: 'code_specific',
  },
  {
    query: 'TypeScript source for Network.connect',
    expected_class: 'code_specific',
  },

  // plan_specific (design documents)
  {
    query: 'what is the checkpointing design',
    expected_class: 'plan_specific',
  },
  {
    query: 'architecture decisions for context assembly',
    expected_class: 'plan_specific',
  },
]);

// ---------------------------------------------------------------------------
// Classification accuracy evaluation
// ---------------------------------------------------------------------------

function evaluateClassificationAccuracy() {
  let correct = 0;
  let partialCredit = 0;
  const results = [];

  /** Adjacent classes that receive 0.5 partial credit */
  const ADJACENT_CLASSES = Object.freeze([
    ['cross_boundary', 'exploratory'],
    ['simple_lookup', 'code_specific'],
  ]);

  for (const { query, expected_class } of EVAL_QUERIES) {
    const classification = classifyQuery(query);
    const isCorrect = classification.query_class === expected_class;

    let isAdjacent = false;
    if (!isCorrect) {
      for (const [a, b] of ADJACENT_CLASSES) {
        if (
          (classification.query_class === a && expected_class === b) ||
          (classification.query_class === b && expected_class === a)
        ) {
          isAdjacent = true;
          break;
        }
      }
    }

    if (isCorrect) correct += 1;
    if (isAdjacent) partialCredit += 0.5;

    results.push({
      expected: expected_class,
      query,
      actual: classification.query_class,
      confidence: classification.confidence,
      correct: isCorrect,
      adjacent: isAdjacent,
    });
  }

  const accuracy = (correct + partialCredit) / EVAL_QUERIES.length;
  return {
    accuracy,
    correct,
    partial_credit: partialCredit,
    total: EVAL_QUERIES.length,
    results,
  };
}

// ---------------------------------------------------------------------------
// Routing effectiveness evaluation
// ---------------------------------------------------------------------------

function evaluateRoutingDefaults() {
  const mismatches = [];

  for (const [queryClass, expectedAlpha] of Object.entries(DEFAULTS)) {
    if (DEFAULTS[queryClass] !== expectedAlpha) {
      mismatches.push({
        queryClass,
        expected: expectedAlpha,
        actual: DEFAULTS[queryClass],
      });
    }
  }

  // Verify classifyAndRoute produces correct defaults for each class
  const representativeQueries = {
    simple_lookup: 'Network.activate',
    cross_boundary: 'relationship between crossover and mutation',
    multi_hop:
      'what functions call Network.activate that also use the slab fast path',
    exploratory: 'how does the training pipeline work',
    code_specific: 'implementation of crossover in NEAT',
    plan_specific: 'what is the checkpointing design',
  };

  for (const [queryClass, query] of Object.entries(representativeQueries)) {
    const result = classifyAndRoute(query);
    if (result.query_class !== queryClass) {
      mismatches.push({
        expected_class: queryClass,
        actual_class: result.query_class,
        query,
      });
    }
    if (result.alpha !== DEFAULTS[queryClass]) {
      mismatches.push({
        expected_alpha: DEFAULTS[queryClass],
        actual_alpha: result.alpha,
        queryClass,
      });
    }
    const expectedStrategy = ROUTING[queryClass];
    if (result.strategy.family !== expectedStrategy.family) {
      mismatches.push({
        expected_family: expectedStrategy.family,
        actual_family: result.strategy.family,
        queryClass,
      });
    }
  }

  return { passed: mismatches.length === 0, mismatches };
}

// ---------------------------------------------------------------------------
// Latency evaluation
// ---------------------------------------------------------------------------

function evaluateLatency() {
  const iterations = 100;
  const queries = EVAL_QUERIES.map((q) => q.query);
  const start = performance.now();

  for (let i = 0; i < iterations; i++) {
    for (const query of queries) {
      classifyQuery(query);
    }
  }

  const totalMs = performance.now() - start;
  const avgMs = totalMs / (iterations * queries.length);

  return {
    total_iterations: iterations * queries.length,
    total_ms: Number(totalMs.toFixed(2)),
    avg_ms_per_query: Number(avgMs.toFixed(4)),
    passed: avgMs < 5,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  const jsonMode = process.argv.includes('--json');

  const accuracyResult = evaluateClassificationAccuracy();
  const routingResult = evaluateRoutingDefaults();
  const latencyResult = evaluateLatency();

  const overall = {
    classification_accuracy: Number(accuracyResult.accuracy.toFixed(3)),
    routing_defaults: routingResult,
    latency: latencyResult,
    passed:
      accuracyResult.accuracy >= 0.833 &&
      routingResult.passed &&
      latencyResult.passed,
  };

  if (jsonMode) {
    console.log(JSON.stringify(overall, null, 2));
  } else {
    console.log('=== Classification Evaluation ===');
    console.log(
      `Accuracy: ${(accuracyResult.accuracy * 100).toFixed(1)}% (${accuracyResult.correct}/${accuracyResult.total} correct, ${accuracyResult.partial_credit} partial credit)`,
    );
    console.log(`Routing defaults: ${routingResult.passed ? 'PASS' : 'FAIL'}`);
    console.log(
      `Latency: ${latencyResult.avg_ms_per_query} ms/query (${latencyResult.passed ? 'PASS' : 'FAIL'})`,
    );
    console.log(`Overall: ${overall.passed ? 'PASS' : 'FAIL'}`);
  }

  process.exit(overall.passed ? 0 : 1);
}

main();
