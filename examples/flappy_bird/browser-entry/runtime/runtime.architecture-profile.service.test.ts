import {
  resolveAvailableRuntimeArchitectureProfiles,
  resolveRuntimeArchitectureSelectorItems,
  updateRuntimeArchitectureHistory,
} from './runtime.architecture-profile.service';

describe('resolveRuntimeArchitectureSelectorItems', () => {
  it('builds selector labels, leader markers, and score captions from the shared profile contract', () => {
    const selectorItems = resolveRuntimeArchitectureSelectorItems({
      availableProfiles: resolveAvailableRuntimeArchitectureProfiles(),
      selectedProfileId: 'narx',
      historyByProfileId: {
        mlp: { pipesPassed: 5, framesSurvived: 240 },
        narx: { pipesPassed: 7, framesSurvived: 210 },
      },
    });

    expect(
      selectorItems.map((selectorItem) => ({
        caption: selectorItem.caption,
        label: selectorItem.label,
        selected: selectorItem.selected,
        tooltipBodyLead: selectorItem.tooltipBodyLines[0],
        tooltipBodySecond: selectorItem.tooltipBodyLines[1],
        tooltipHeading: selectorItem.tooltipHeading,
      })),
    ).toEqual([
      {
        caption: 'Best 5 pipes',
        label: 'MLP',
        selected: false,
        tooltipBodyLead:
          'MLP is the plain feed-forward baseline: the current numbers go in, action scores come out, and nothing is remembered between frames.',
        tooltipBodySecond:
          'In simple words, it reacts only to what it sees right now. There is no built-in memory cell or delay shelf inside the network.',
        tooltipHeading: 'MLP · Multi-Layer Perceptron',
      },
      {
        caption: undefined,
        label: 'Sparse',
        selected: false,
        tooltipBodyLead:
          'Sparse starts from the same feed-forward idea as MLP, but it begins with many fewer wires already drawn.',
        tooltipBodySecond:
          'Think of it as a rough sketch instead of a finished diagram: evolution has to discover which connections deserve to exist.',
        tooltipHeading: 'Sparse · Sparse Feed-Forward Graph',
      },
      {
        caption: 'Best 7 pipes',
        label: 'NARX *',
        selected: true,
        tooltipBodyLead:
          'NARX is a recurrent network with an explicit short memory shelf for recent inputs and recent outputs.',
        tooltipBodySecond:
          'Instead of hiding memory behind gates, it keeps a visible rolling window of the recent past. You can picture it as a tiny notepad of what just happened.',
        tooltipHeading: 'NARX · Explicit Delay-Line Memory',
      },
      {
        caption: undefined,
        label: 'GRU',
        selected: false,
        tooltipBodyLead:
          'GRU is a gated recurrent network that learns what recent information to keep, refresh, or forget as the bird flies.',
        tooltipBodySecond:
          'In simple words, it builds its own short-term memory instead of relying on a fixed delay shelf. The gates act like small traffic lights for remembered state.',
        tooltipHeading: 'GRU · Gated Recurrent Unit',
      },
      {
        caption: undefined,
        label: 'LSTM',
        selected: false,
        tooltipBodyLead:
          'LSTM is a gated recurrent network designed to carry information forward over longer stretches of time.',
        tooltipBodySecond:
          'It uses a dedicated cell state plus gates that decide what to keep, write, and reveal. A simple picture is a memory lane with controlled entry and exit points.',
        tooltipHeading: 'LSTM · Long Short-Term Memory',
      },
    ]);
  });
});

describe('updateRuntimeArchitectureHistory', () => {
  it('keeps the higher-pipe record and uses frames survived only as a stable tiebreaker', () => {
    const updatedHistory = updateRuntimeArchitectureHistory(
      {
        narx: { pipesPassed: 4, framesSurvived: 300 },
      },
      'narx',
      { pipesPassed: 4, framesSurvived: 360 },
    );

    expect(updatedHistory).toEqual({
      narx: { pipesPassed: 4, framesSurvived: 360 },
    });
  });
});
