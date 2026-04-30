import {
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
} from '../constants/constants';
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
  it('uses the shared Flappy MLP profile as the default seed network', () => {
    const neatController = createNeatController(
      createTrainerSetup(),
    ) as TrainerControllerWithRecurrentPolicy;

    expect({
      hiddenLayerSizes:
        neatController.options.network?.describeArchitecture().hiddenLayerSizes,
      inputNodeIds: neatController.options.network?.inputNodeIds.length,
      outputNodeIds: neatController.options.network?.outputNodeIds.length,
    }).toEqual({
      hiddenLayerSizes: FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
      inputNodeIds: FLAPPY_NETWORK_INPUT_SIZE,
      outputNodeIds: 2,
    });
  });

  it('disables recurrent growth when the default MLP profile is selected', () => {
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
