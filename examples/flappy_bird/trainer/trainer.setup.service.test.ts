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
  it('pins the trainer runtime to feed-forward growth', () => {
    const neatController = createNeatController(
      createTrainerSetup(),
    ) as TrainerControllerWithRecurrentPolicy;

    expect(neatController.options.allowRecurrent).toBe(false);
  });

  it('uses the shared Flappy MLP profile as the default seed network', () => {
    const neatController = createNeatController(
      createTrainerSetup(),
    ) as TrainerControllerWithRecurrentPolicy;

    expect({
      hiddenLayerSizes: neatController.options.network?.describeArchitecture()
        .hiddenLayerSizes,
      inputNodeIds: neatController.options.network?.inputNodeIds.length,
      outputNodeIds: neatController.options.network?.outputNodeIds.length,
    }).toEqual({
      hiddenLayerSizes: [16, 8, 4],
      inputNodeIds: 38,
      outputNodeIds: 2,
    });
  });
});