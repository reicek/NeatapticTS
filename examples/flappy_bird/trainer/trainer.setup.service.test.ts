import {
  createNeatController,
  createTrainerSetup,
} from './trainer.setup.service';

interface TrainerControllerWithRecurrentPolicy {
  options: {
    allowRecurrent?: boolean;
  };
}

describe('createNeatController', () => {
  it('pins the trainer runtime to feed-forward growth', () => {
    const neatController = createNeatController(
      createTrainerSetup(),
    ) as TrainerControllerWithRecurrentPolicy;

    expect(neatController.options.allowRecurrent).toBe(false);
  });
});