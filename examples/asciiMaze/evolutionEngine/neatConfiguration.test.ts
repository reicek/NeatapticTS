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
    const neatInstance = createNeat(6, 4, () => 0);
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
