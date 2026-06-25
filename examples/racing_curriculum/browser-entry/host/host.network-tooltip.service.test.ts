/** @jest-environment jsdom */

import {
  resolveRacingNetworkTooltipScene,
  type NetworkVisualizationPositionedScene,
} from './host.network-tooltip.service';

describe('resolveRacingNetworkTooltipScene', () => {
  it('returns an input tooltip when the pointer is over an input node', () => {
    const positionedScene: NetworkVisualizationPositionedScene = {
      positionedNodes: [
        {
          node: { index: 0, type: 'input', bias: 0 },
          xPx: 20,
          yPx: 20,
        },
      ],
      nodeDimensions: { widthPx: 20, heightPx: 20 },
      inputDescriptionScenes: [
        {
          nodeIndex: 0,
          leftPx: 10,
          topPx: 10,
          widthPx: 20,
          heightPx: 20,
          tooltipHeading: 'Lateral Offset',
          tooltipBodyParagraphs: [
            'Distance from the optimal racing line.',
            'Used by the controller to center the car.',
          ],
        },
      ],
      inputGroupLabelBandScenes: [],
      hiddenColumnLabelScenes: [],
    };

    const tooltipScene = resolveRacingNetworkTooltipScene(
      { xPx: 20, yPx: 20 },
      positionedScene,
    );

    expect(tooltipScene?.heading).toBe('Lateral Offset');
  });
});
