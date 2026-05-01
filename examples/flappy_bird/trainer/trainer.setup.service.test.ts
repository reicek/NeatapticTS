import {
  FLAPPY_NETWORK_INPUT_SIZE,
} from '../constants/constants';
import { DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID } from '../../architectureProfiles';
import {
  createNeatController,
  createTrainerSetup,
} from './trainer.setup.service';

interface TrainerControllerWithRecurrentPolicy {
  options: {
    allowRecurrent?: boolean;
    network?: {
      describeArchitecture: () => {
        hiddenLayerSizes: number[];
      };
      inputNodeIds: number[];
      outputNodeIds: number[];
    };
  };
}

describe('createNeatController', () => {
  it('defaults the trainer setup to the shared Flappy default profile', () => {
    expect({
      architectureProfileId: createTrainerSetup().architectureProfileId,
      inputSize: createTrainerSetup().inputSize,
      isRecurrent: createTrainerSetup().isRecurrent,
    }).toEqual({
      architectureProfileId: DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
      inputSize: FLAPPY_NETWORK_INPUT_SIZE,
      isRecurrent: false,
    });
  });

  it('uses the shared Flappy default profile dimensions for the default seed network', () => {
    const neatController = createNeatController(
      createTrainerSetup(),
    ) as TrainerControllerWithRecurrentPolicy;

    expect({
      hasHiddenNodes:
        (neatController.options.network?.describeArchitecture().hiddenLayerSizes
          .length ?? 0) > 0,
      inputNodeIds: neatController.options.network?.inputNodeIds.length,
      outputNodeIds: neatController.options.network?.outputNodeIds.length,
    }).toEqual({
      hasHiddenNodes: true,
      inputNodeIds: FLAPPY_NETWORK_INPUT_SIZE,
      outputNodeIds: 2,
    });
  });

  it('disables recurrent growth when the default feed-forward profile is selected', () => {
    const neatController = createNeatController(
      createTrainerSetup(),
    ) as TrainerControllerWithRecurrentPolicy;

    expect(neatController.options.allowRecurrent).toBe(false);
  });

  it('enables recurrent growth when a recurrent profile is selected', () => {
    const neatController = createNeatController({
      ...createTrainerSetup(),
      architectureProfileId: 'gru',
    }) as TrainerControllerWithRecurrentPolicy;

    expect(neatController.options.allowRecurrent).toBe(true);
  });
});
