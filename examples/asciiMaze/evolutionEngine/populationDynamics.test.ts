import {
  maybeApplyTopologyShakeupMutation,
  TOPOLOGY_SHAKEUP_STAGNATION_GENERATIONS,
} from './populationDynamics';

describe('maybeApplyTopologyShakeupMutation', () => {
  it('runs one structural-only mutation pass when stagnant progress reaches the shake-up threshold', async () => {
    const originalMutationShelf = [
      { name: 'MOD_WEIGHT' },
      { name: 'ADD_CONN' },
      { name: 'ADD_NODE' },
      { name: 'MOD_BIAS' },
    ];
    let capturedMutationNames: string[] = [];
    let capturedMutationRate = 0;
    let capturedMutationAmount = 0;
    const neat = {
      options: {
        mutation: originalMutationShelf,
        mutationAmount: 1,
        mutationRate: 0.25,
      },
      mutate: jest.fn(async function mutate() {
        capturedMutationNames = this.options.mutation.map(
          (mutationOperation: { name: string }) => mutationOperation.name,
        );
        capturedMutationRate = this.options.mutationRate;
        capturedMutationAmount = this.options.mutationAmount;
      }),
    };

    const didApply = await maybeApplyTopologyShakeupMutation(
      neat as never,
      TOPOLOGY_SHAKEUP_STAGNATION_GENERATIONS,
      jest.fn(),
      12,
    );

    expect({
      capturedMutationAmount,
      capturedMutationNames,
      capturedMutationRate,
      didApply,
      mutateCalls: neat.mutate.mock.calls.length,
      restoredMutationAmount: neat.options.mutationAmount,
      restoredMutationRate: neat.options.mutationRate,
      restoredMutationShelf: neat.options.mutation,
    }).toEqual({
      capturedMutationAmount: 2,
      capturedMutationNames: ['ADD_CONN', 'ADD_NODE'],
      capturedMutationRate: 1,
      didApply: true,
      mutateCalls: 1,
      restoredMutationAmount: 1,
      restoredMutationRate: 0.25,
      restoredMutationShelf: originalMutationShelf,
    });
  });

  it('skips the shake-up between threshold boundaries', async () => {
    const originalMutationShelf = [{ name: 'ADD_CONN' }, { name: 'ADD_NODE' }];
    const neat = {
      options: {
        mutation: originalMutationShelf,
        mutationAmount: 1,
        mutationRate: 0.25,
      },
      mutate: jest.fn(),
    };

    const didApply = await maybeApplyTopologyShakeupMutation(
      neat as never,
      TOPOLOGY_SHAKEUP_STAGNATION_GENERATIONS + 1,
      jest.fn(),
      13,
    );

    expect({
      didApply,
      mutateCalls: neat.mutate.mock.calls.length,
      mutationAmount: neat.options.mutationAmount,
      mutationRate: neat.options.mutationRate,
      mutationShelf: neat.options.mutation,
    }).toEqual({
      didApply: false,
      mutateCalls: 0,
      mutationAmount: 1,
      mutationRate: 0.25,
      mutationShelf: originalMutationShelf,
    });
  });
});
