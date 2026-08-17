import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const files = [
  'chunker-v2.mjs',
  'hybrid-rank.mjs',
  'init-turso.mjs',
  'build-entity-graph.mjs',
  'extract-cross-refs.mjs',
  'assemble-context.mjs',
  'eval-runner.mjs',
  'perf-benchmark.mjs',
  'parallel-search.mjs',
  'eval-classification.mjs',
  'build-index.mjs',
  'embed-readiness.mjs',
  'validate-turso-index.mjs',
  'session-start-index.mjs',
  'code-quality-scanner.mjs',
  'metadata-enrichment.mjs',
  'docs-quality/docs-quality.metrics.mjs',
  'docs-quality/docs-quality.compare.mjs',
  'expand-query.mjs',
  'eval-embeddings.mjs',
  'eval-compare.mjs',
  'eval-metrics.mjs',
  'reranker-readiness.mjs',
  '__tests__/turso-test-helpers.mjs',
];

const covPath = join(process.cwd(), 'coverage', 'coverage-final.json');
if (!existsSync(covPath)) {
  console.error('No coverage file found at', covPath);
  process.exit(1);
}

const cov = JSON.parse(readFileSync(covPath, 'utf8'));

for (const file of files) {
  const key = Object.keys(cov).find(k => k.replace(/\\/g, '/').endsWith(file.replace(/\\/g, '/')));
  if (!key) {
    console.log(`\n=== ${file}: NOT FOUND in coverage ===`);
    continue;
  }
  const data = cov[key];
  const totalBranches = Object.keys(data.branchMap).length;
  const uncovered = [];
  for (const [id, b] of Object.entries(data.branchMap)) {
    const hits = data.b[id] || [];
    const allHit = hits.every(h => h > 0);
    if (!allHit) {
      uncovered.push({ id, type: b.type, line: b.loc.start.line, hits });
    }
  }
  if (uncovered.length > 0) {
    console.log(`\n=== ${file} (${uncovered.length}/${totalBranches} branches uncovered) ===`);
    for (const u of uncovered) {
      console.log(`  line ${u.line}: ${u.type} hits=${JSON.stringify(u.hits)}`);
    }
  } else {
    console.log(`\n=== ${file}: ALL BRANCHES COVERED ===`);
  }
}