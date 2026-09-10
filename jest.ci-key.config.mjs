// Curated stable key test set for push CI.
// The full src-suite capability remains available via `npm run test:src`.
// Entries are grouped by domain with short `//` comments below.

import baseConfig from './jest.config.mjs';

const defaultProject = baseConfig.projects.find(
  (project) => project.displayName === 'default'
);
const { testMatch: _ignoredTestMatch, ...defaultProjectWithoutTestMatch } = defaultProject;

const ciKeyTestMatch = [
  // Architecture core
  '**/src/architecture/node/node.test.ts',
  '**/src/architecture/connection/connection.test.ts',
  '**/src/architecture/group/group.test.ts',
  '**/src/architecture/layer/layer.test.ts',
  '**/src/architecture/architect/architect.test.ts',
  '**/src/architecture/network/construct/network.construct.test.ts',
  '**/src/architecture/network/activate/network.activate.test.ts',
  '**/src/architecture/network/connect/network.connect.test.ts',
  '**/src/architecture/network/training/network.training.basic.test.ts',
  '**/src/architecture/network/evolve/network.evolve.test.ts',
  '**/src/architecture/network/genetic/network.genetic.test.ts',
  '**/src/architecture/network/topology/network.topology.test.ts',
  '**/src/architecture/network/serialize/network.serialize.test.ts',
  '**/src/architecture/network/slab/network.slab.fast-path.test.ts',
  '**/src/architecture/network/standalone/network.standalone.test.ts',
  '**/src/architecture/network/bootstrap/network.bootstrap.utils.direct.test.ts',
  '**/src/architecture/network/onnx/network.onnx.test.ts',
  '**/src/architecture/network/onnx/validate/network.onnx.validate.test.ts',
  '**/src/architecture/network/worker-payload/network.worker-payload.pool.test.ts',
  '**/src/architecture/network/evolve/network.evolve.multithread.test.ts',
  '**/src/architecture/activationArrayPool/activationArrayPool.test.ts',
  '**/src/architecture/nodePool/nodePool.test.ts',
  // NEAT evolution core
  '**/src/neat/genome/heredity/genome.heredity.test.ts',
  '**/src/neat/genome/genome.test.ts',
  '**/src/neat/mutation/mutation.test.ts',
  '**/src/neat/evolve/evolve.test.ts',
  '**/src/neat/evolve/offspring/evolve.offspring.test.ts',
  '**/src/neat/evolve/population/evolve.population.test.ts',
  '**/src/neat/evolve/speciation/evolve.speciation.test.ts',
  '**/src/neat/evolve/objectives/evolve.objectives.test.ts',
  '**/src/neat/evolve/adaptive/evolve.adaptive.test.ts',
  '**/src/neat/species/core/shared/species.core.shared.test.ts',
  '**/src/neat/species/stats/species.stats.test.ts',
  '**/src/neat/multiobjective/multiobjective.test.ts',
  '**/src/neat/diversity/core/diversity.core.test.ts',
  '**/src/neat/selection/core/selection.core.test.ts',
  '**/src/neat/lineage/lineage.test.ts',
  '**/src/neat/nge-dna/neat.nge-dna.test.ts',
  '**/src/neat/nge-dna/neat.nge-dna.schema.test.ts',
  '**/src/neat/nge-dna/neat.nge-dna.operator.test.ts',
  '**/src/neat/nge-evolution/neat.nge-evolution.test.ts',
  '**/src/neat/nge-evolution/neat.nge-evolution.reproduction-mode.test.ts',
  '**/src/neat/nge-evolution/neat.nge-evolution.reproduction.schema.test.ts',
  // Infrastructure core
  '**/src/neat/rng/rng.test.ts',
  '**/src/neat/rng/core/rng.utils.test.ts',
  '**/src/neat/cache/core/cache.core.test.ts',
  '**/src/neat/maintenance/facade/maintenance.facade.test.ts',
  '**/src/neat/telemetry/facade/lineage/telemetry.facade.lineage.test.ts',
  '**/src/neat/telemetry/facade/novelty/telemetry.facade.novelty.test.ts',
  '**/src/neat/telemetry/facade/runtime/telemetry.facade.runtime.test.ts',
  '**/src/methods/activation/activation.test.ts',
  '**/src/methods/mutation/mutation.test.ts',
  '**/src/methods/rate/rate.test.ts',
  '**/src/config.test.ts',
  '**/src/neataptic.test.ts',
];

/** @type {import('jest').Config} */
const config = {
  ...baseConfig,
  projects: [
    {
      ...defaultProjectWithoutTestMatch,
      displayName: 'ci-key',
      testMatch: ciKeyTestMatch,
    },
  ],
};

export default config;
