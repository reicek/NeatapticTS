import Network from '../../../src/architecture/network';
import type { INetwork } from '../interfaces';
import { EvolutionEngine } from '../evolutionEngine';

describe('EvolutionEngine.printNetworkStructure', () => {
  it('prints activation scheduling and explicit IO role summaries when the runtime exposes them', () => {
    const network = new Network(2, 1, {
      seed: 713,
      enforceAcyclic: true,
    });
    network.activate([0.25, 0.75]);

    const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => undefined);

    try {
      EvolutionEngine.printNetworkStructure(network as unknown as INetwork);
      const loggedLines = consoleLogSpy.mock.calls.map((loggedCall) =>
        loggedCall.map((loggedValue) => String(loggedValue)).join(' '),
      );

      expect(loggedLines).toEqual(
        expect.arrayContaining([
          'Activation scheduling: acyclic via compiled-schedule (steps=2, recurrent components=0, issue=none)',
          'Explicit IO roles: 2 inputs, 1 outputs',
        ]),
      );
    } finally {
      consoleLogSpy.mockRestore();
    }
  });
});