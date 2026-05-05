import { methods } from '../../../src/neataptic';
import { buildExampleArchitectureProfileNetwork } from '../../architectureProfiles';
import { createNeat } from './neatConfiguration';

const TEMPORAL_MUTATION_NAMES = [
  methods.mutation.ADD_BACK_CONN.name,
  methods.mutation.ADD_GATE.name,
  methods.mutation.ADD_GRU_NODE.name,
  methods.mutation.ADD_LSTM_NODE.name,
  methods.mutation.ADD_SELF_CONN.name,
].toSorted();

describe('createNeat', () => {
  it('adds the temporal mutation shelf when recurrent growth stays enabled', () => {
    const neatInstance = createNeat(6, 4, () => 0, {
      allowRecurrent: true,
    });
    const mutationNames = resolveConfiguredMutationNames(
      neatInstance.options.mutation,
    );

    expect(
      mutationNames
        .filter((mutationName) =>
          TEMPORAL_MUTATION_NAMES.includes(mutationName),
        )
        .toSorted(),
    ).toEqual(TEMPORAL_MUTATION_NAMES);
  });

  it('keeps the default shelf free of temporal operators when recurrence is disabled', () => {
    const neatInstance = createNeat(6, 4, () => 0, {
      allowRecurrent: false,
    });
    const mutationNames = resolveConfiguredMutationNames(
      neatInstance.options.mutation,
    );

    expect(
      mutationNames.filter((mutationName) =>
        TEMPORAL_MUTATION_NAMES.includes(mutationName),
      ),
    ).toEqual([]);
  });

  it('forwards an explicit builder-backed seed network into the NEAT runtime options', () => {
    const seedNetwork = buildExampleArchitectureProfileNetwork(
      'ascii-maze',
      'mlp',
    );
    const neatInstance = createNeat(6, 4, () => 0, {
      network: seedNetwork,
    });

    expect(neatInstance.options.network).toBe(seedNetwork);
  });

  it('leaves the global mutation rate unset so adaptive mutation can own per-genome pressure', () => {
    const neatInstance = createNeat(6, 4, () => 0);

    expect(neatInstance.options.mutationRate).toBeUndefined();
  });

  it('biases the default feed-forward shelf toward topology growth after generation-zero warm start', () => {
    const neatInstance = createNeat(6, 4, () => 0, {
      allowRecurrent: false,
    });
    const mutationCounts = countConfiguredMutationNames(
      neatInstance.options.mutation,
    );

    expect(mutationCounts).toEqual({
      ADD_CONN: 4,
      ADD_NODE: 3,
      MOD_ACTIVATION: 1,
      MOD_BIAS: 1,
      MOD_WEIGHT: 1,
      SUB_CONN: 1,
      SUB_NODE: 1,
    });
  });

  it('does not inject fresh provenance genomes after the generation-zero template copy pass', () => {
    const neatInstance = createNeat(6, 4, () => 0, {
      allowRecurrent: false,
    });

    expect(neatInstance.options.provenance).toBe(0);
  });

  it('defaults to multiple mutation attempts per admitted genome', () => {
    const neatInstance = createNeat(6, 4, () => 0, {
      allowRecurrent: false,
    });

    expect(neatInstance.options.mutationAmount).toBe(3);
  });

  it('drops the legacy hidden-node floor so sparse starters stay sparse at generation zero', () => {
    const neatInstance = createNeat(6, 4, () => 0, {
      allowRecurrent: false,
    });

    expect(neatInstance.options.minHidden).toBe(0);
  });

  it('allows at least one genome to gain connections under the default ASCII Maze mutation config', async () => {
    const seedNetwork = buildExampleArchitectureProfileNetwork(
      'ascii-maze',
      'random-sparse',
    );
    const neatInstance = createNeat(6, 4, () => 0, {
      allowRecurrent: false,
      network: seedNetwork,
      popSize: 32,
      seed: 42,
    });
    const maxConnectionCountBeforeMutation = Math.max(
      ...neatInstance.population.map((genome) => genome.connections.length),
    );

    await neatInstance.mutate();

    const maxConnectionCountAfterMutation = Math.max(
      ...neatInstance.population.map((genome) => genome.connections.length),
    );

    expect(maxConnectionCountAfterMutation).toBeGreaterThan(
      maxConnectionCountBeforeMutation,
    );
  });
});

function resolveConfiguredMutationNames(mutationShelf: unknown): string[] {
  if (!Array.isArray(mutationShelf)) {
    return [];
  }

  return mutationShelf.flatMap((mutationConfig) => {
    if (
      typeof mutationConfig === 'object' &&
      mutationConfig !== null &&
      'name' in mutationConfig &&
      typeof mutationConfig.name === 'string'
    ) {
      return [mutationConfig.name];
    }

    return [];
  });
}

function countConfiguredMutationNames(
  mutationShelf: unknown,
): Record<string, number> {
  return resolveConfiguredMutationNames(mutationShelf).reduce(
    (countsByName, mutationName) => ({
      ...countsByName,
      [mutationName]: (countsByName[mutationName] ?? 0) + 1,
    }),
    {} as Record<string, number>,
  );
}
