import {
  getApprovedExampleArchitectureProfiles,
  type ExampleArchitectureProfile,
  type ExampleArchitectureProfileId,
} from '../../architectureProfiles';

/** Monospace font stack shared with the host UI palette. */
const MAZE_ARCH_MONOSPACE_FONT = 'Consolas, Menlo, Monaco, monospace';

/** Dim border color for unselected buttons. */
const MAZE_ARCH_BUTTON_BORDER_IDLE = 'rgba(15, 181, 255, 0.5)';

/** Active border color matching the neon panel outline. */
const MAZE_ARCH_BUTTON_BORDER_ACTIVE = 'rgba(15, 181, 255, 0.92)';

/** Subtle tinted fill for the selected button. */
const MAZE_ARCH_BUTTON_BG_ACTIVE = 'rgba(15, 181, 255, 0.12)';

/** Warm accent used for tooltip headings. */
const MAZE_ARCH_TOOLTIP_HEADING_COLOR = '#ff9a2e';

/** Border color for the floating tooltip card. */
const MAZE_ARCH_TOOLTIP_BORDER_COLOR = '#0fb5ff';

/** Text color used for button labels and dim states. */
const MAZE_ARCH_BUTTON_COLOR_IDLE = '#9fdcff';

/** Text color used for the selected button label. */
const MAZE_ARCH_BUTTON_COLOR_ACTIVE = '#dff6ff';

/** Default unselected button background. */
const MAZE_ARCH_BUTTON_BG_IDLE = 'rgba(6, 14, 28, 0.8)';

/** Label for the Flappy-style reset action beneath the selector grid. */
const MAZE_ARCH_RESET_BUTTON_TEXT = 'Reset Simulation';

/**
 * Controller returned by {@link createMazeArchitectureSelector} for updating
 * selector state after the initial render.
 */
export interface MazeArchitectureSelectorController {
  /**
   * Update the visually active button without triggering a restart.
   *
   * @param profileId - Profile id to mark as selected.
   */
  setSelectedProfile: (profileId: ExampleArchitectureProfileId) => void;

  /**
   * Enable or disable all selector buttons.
   *
   * Buttons should be disabled while a restart is pending to prevent
   * duplicate selection events.
   *
   * @param disabled - `true` to grey-out and lock all buttons.
   */
  setDisabled: (disabled: boolean) => void;
}

/**
 * Render a Flappy-style architecture selector into the given container element.
 *
 * Each button shows an educational hover tooltip describing the architecture
 * family. Clicking a button that is not currently selected calls `onSelect`
 * with the chosen profile id.
 *
 * @param containerElement - Element whose children will be replaced with the selector buttons.
 * @param selectedProfileId - Initially selected architecture profile id.
 * @param onSelect - Callback invoked with the newly selected profile id on click.
 * @param onReset - Optional callback invoked when the current architecture should restart from scratch.
 * @returns Controller for updating selected state and disabling the selector.
 *
 * @example
 * ```ts
 * const controller = createMazeArchitectureSelector(
 *   document.querySelector('.arch-buttons'),
 *   'random-sparse',
 *   (profileId) => console.log('selected', profileId),
 * );
 * controller.setDisabled(true);
 * ```
 */
export function createMazeArchitectureSelector(
  containerElement: HTMLElement,
  selectedProfileId: ExampleArchitectureProfileId,
  onSelect: (profileId: ExampleArchitectureProfileId) => void,
  onReset?: () => void,
): MazeArchitectureSelectorController {
  // Step 1: Collect approved profiles for the maze demo.
  const approvedProfiles = getApprovedExampleArchitectureProfiles('ascii-maze');

  // Step 2: Track mutable state shared across buttons.
  let currentSelectedId = selectedProfileId;
  let isDisabled = false;

  // Step 3: Build button item records for later state updates.
  const itemRecords: {
    profileId: ExampleArchitectureProfileId;
    buttonElement: HTMLButtonElement;
  }[] = [];
  const gridElement = document.createElement('div');

  // Step 4: Populate the container.
  containerElement.innerHTML = '';
  containerElement.style.display = 'flex';
  containerElement.style.flexDirection = 'column';
  containerElement.style.gap = '8px';
  containerElement.style.width = '100%';
  containerElement.style.boxSizing = 'border-box';
  containerElement.style.overflow = 'visible';

  gridElement.style.display = 'grid';
  gridElement.style.gridTemplateColumns = 'repeat(2, minmax(0, 1fr))';
  gridElement.style.gap = '6px';
  gridElement.style.width = '100%';

  for (const profile of approvedProfiles) {
    const { itemElement, buttonElement } = buildButtonItem(profile);
    gridElement.appendChild(itemElement);
    itemRecords.push({ profileId: profile.id, buttonElement });
  }
  containerElement.appendChild(gridElement);

  if (onReset) {
    containerElement.appendChild(buildResetSimulationButtonElement(onReset));
  }

  return {
    setSelectedProfile: (profileId: ExampleArchitectureProfileId): void => {
      currentSelectedId = profileId;
      for (const record of itemRecords) {
        applyButtonPresentation(
          record.buttonElement,
          record.profileId === profileId,
          isDisabled,
          false,
        );
      }
    },

    setDisabled: (disabled: boolean): void => {
      isDisabled = disabled;
      for (const record of itemRecords) {
        applyButtonPresentation(
          record.buttonElement,
          record.profileId === currentSelectedId,
          disabled,
          false,
        );
      }
    },
  };

  /**
   * Build a wrapper div containing the tooltip and the selector button.
   *
   * @param profile - Architecture profile to render.
   * @returns Wrapper and button elements for DOM insertion and state tracking.
   */
  function buildButtonItem(profile: ExampleArchitectureProfile): {
    itemElement: HTMLDivElement;
    buttonElement: HTMLButtonElement;
  } {
    const itemElement = document.createElement('div');
    itemElement.style.display = 'flex';
    itemElement.style.flexDirection = 'column';
    itemElement.style.gap = '4px';
    itemElement.style.minWidth = '0';
    itemElement.style.overflow = 'visible';
    itemElement.style.position = 'relative';
    itemElement.style.zIndex = '0';

    const tooltipElement = buildTooltipElement(profile);
    const buttonElement = buildButtonElement(
      profile,
      tooltipElement,
      itemElement,
    );

    itemElement.appendChild(tooltipElement);
    itemElement.appendChild(buttonElement);

    return { itemElement, buttonElement };
  }

  /**
   * Build the reset button shown below the selector grid.
   *
   * @param onResetSimulation - Callback invoked when the user requests a fresh run.
   * @returns Styled reset button matching the Flappy selector tray.
   */
  function buildResetSimulationButtonElement(
    onResetSimulation: () => void,
  ): HTMLButtonElement {
    const resetButtonElement = document.createElement('button');
    resetButtonElement.type = 'button';
    resetButtonElement.textContent = MAZE_ARCH_RESET_BUTTON_TEXT;
    resetButtonElement.style.width = '100%';
    resetButtonElement.style.boxSizing = 'border-box';
    resetButtonElement.style.padding = '5px 8px';
    resetButtonElement.style.borderRadius = '10px';
    resetButtonElement.style.borderStyle = 'solid';
    resetButtonElement.style.borderWidth = '1px';
    resetButtonElement.style.borderColor = MAZE_ARCH_BUTTON_BORDER_IDLE;
    resetButtonElement.style.background = 'transparent';
    resetButtonElement.style.fontFamily = MAZE_ARCH_MONOSPACE_FONT;
    resetButtonElement.style.fontSize = '11px';
    resetButtonElement.style.fontWeight = '700';
    resetButtonElement.style.textTransform = 'uppercase';
    resetButtonElement.style.letterSpacing = '0.08em';
    resetButtonElement.style.color = MAZE_ARCH_BUTTON_COLOR_IDLE;
    resetButtonElement.style.opacity = '0.65';
    resetButtonElement.style.cursor = 'pointer';
    resetButtonElement.style.transition =
      'border-color 120ms ease-out, box-shadow 120ms ease-out, color 120ms ease-out, background 120ms ease-out, transform 120ms ease-out';

    resetButtonElement.addEventListener('mouseenter', () => {
      if (isDisabled) {
        return;
      }

      resetButtonElement.style.opacity = '1';
      resetButtonElement.style.borderColor = MAZE_ARCH_TOOLTIP_HEADING_COLOR;
      resetButtonElement.style.color = MAZE_ARCH_TOOLTIP_HEADING_COLOR;
      resetButtonElement.style.transform = 'translateY(-1px)';
      resetButtonElement.style.boxShadow =
        '0 0 8px rgba(255, 154, 46, 0.55), inset 0 0 10px rgba(255, 154, 46, 0.08)';
    });

    resetButtonElement.addEventListener('mouseleave', () => {
      resetButtonElement.style.opacity = isDisabled ? '0.4' : '0.65';
      resetButtonElement.style.borderColor = MAZE_ARCH_BUTTON_BORDER_IDLE;
      resetButtonElement.style.color = MAZE_ARCH_BUTTON_COLOR_IDLE;
      resetButtonElement.style.transform = 'translateY(0)';
      resetButtonElement.style.boxShadow = '';
    });

    resetButtonElement.addEventListener('click', () => {
      if (!isDisabled) {
        onResetSimulation();
      }
    });

    return resetButtonElement;
  }

  /**
   * Build and style the clickable selector button.
   *
   * @param profile - Architecture profile for label and identity.
   * @param tooltipElement - Tooltip to show on hover.
   * @param itemElement - Wrapper element whose z-index is lifted on hover.
   * @returns Configured button element ready for DOM insertion.
   */
  function buildButtonElement(
    profile: ExampleArchitectureProfile,
    tooltipElement: HTMLDivElement,
    itemElement: HTMLDivElement,
  ): HTMLButtonElement {
    const buttonElement = document.createElement('button');
    buttonElement.type = 'button';
    buttonElement.textContent = profile.label;
    buttonElement.style.fontFamily = MAZE_ARCH_MONOSPACE_FONT;
    buttonElement.style.fontSize = '11px';
    buttonElement.style.fontWeight = '600';
    buttonElement.style.letterSpacing = '0.06em';
    buttonElement.style.textTransform = 'uppercase';
    buttonElement.style.padding = '5px 8px';
    buttonElement.style.borderRadius = '10px';
    buttonElement.style.borderStyle = 'solid';
    buttonElement.style.borderWidth = '1px';
    buttonElement.style.width = '100%';
    buttonElement.style.boxSizing = 'border-box';
    buttonElement.style.whiteSpace = 'nowrap';
    buttonElement.style.overflow = 'hidden';
    buttonElement.style.textOverflow = 'ellipsis';
    buttonElement.style.transition =
      'border-color 120ms ease-out, box-shadow 120ms ease-out, color 120ms ease-out, background 120ms ease-out, transform 120ms ease-out';

    // Set initial presentation.
    applyButtonPresentation(
      buttonElement,
      profile.id === currentSelectedId,
      isDisabled,
      false,
    );

    buttonElement.addEventListener('mouseenter', () => {
      if (!isDisabled) {
        tooltipElement.style.opacity = '1';
        tooltipElement.style.transform = 'translate(-50%, 0)';
        itemElement.style.zIndex = '3';
        applyButtonPresentation(
          buttonElement,
          profile.id === currentSelectedId,
          false,
          true,
        );
      }
    });

    buttonElement.addEventListener('mouseleave', () => {
      tooltipElement.style.opacity = '0';
      tooltipElement.style.transform = 'translate(-50%, 6px)';
      itemElement.style.zIndex = '0';
      applyButtonPresentation(
        buttonElement,
        profile.id === currentSelectedId,
        isDisabled,
        false,
      );
    });

    buttonElement.addEventListener('click', () => {
      if (!isDisabled && profile.id !== currentSelectedId) {
        onSelect(profile.id);
      }
    });

    return buttonElement;
  }

  /**
   * Apply visual state to a button based on selected, disabled, and hover flags.
   *
   * @param buttonElement - Button to style.
   * @param selected - Whether this button represents the active profile.
   * @param disabled - Whether the selector is locked during a restart.
   * @param hovered - Whether the cursor is currently over the button.
   */
  function applyButtonPresentation(
    buttonElement: HTMLButtonElement,
    selected: boolean,
    disabled: boolean,
    hovered: boolean,
  ): void {
    buttonElement.style.borderColor = selected
      ? MAZE_ARCH_BUTTON_BORDER_ACTIVE
      : MAZE_ARCH_BUTTON_BORDER_IDLE;
    buttonElement.style.color = selected
      ? MAZE_ARCH_BUTTON_COLOR_ACTIVE
      : MAZE_ARCH_BUTTON_COLOR_IDLE;
    buttonElement.style.background = selected
      ? MAZE_ARCH_BUTTON_BG_ACTIVE
      : MAZE_ARCH_BUTTON_BG_IDLE;
    buttonElement.style.opacity = disabled ? '0.55' : '1';
    buttonElement.style.cursor = disabled ? 'default' : 'pointer';
    buttonElement.style.transform =
      hovered && !disabled ? 'translateY(-1px)' : 'translateY(0)';

    if (hovered && !disabled) {
      buttonElement.style.boxShadow = selected
        ? '0 0 8px rgba(255, 92, 255, 0.7), inset 0 0 10px rgba(0, 229, 255, 0.16)'
        : '0 0 8px rgba(15, 181, 255, 0.72), inset 0 0 10px rgba(0, 229, 255, 0.16)';
    } else if (selected) {
      buttonElement.style.boxShadow =
        '0 0 10px rgba(15, 181, 255, 0.28), inset 0 0 8px rgba(15, 181, 255, 0.08)';
    } else {
      buttonElement.style.boxShadow = 'inset 0 0 6px rgba(15, 181, 255, 0.08)';
    }
  }

  /**
   * Build the floating tooltip card shown above a button on hover.
   *
   * @param profile - Architecture profile for tooltip content.
   * @returns Configured tooltip div ready for DOM insertion.
   */
  function buildTooltipElement(
    profile: ExampleArchitectureProfile,
  ): HTMLDivElement {
    const tooltipElement = document.createElement('div');
    tooltipElement.setAttribute('role', 'tooltip');
    tooltipElement.style.position = 'absolute';
    tooltipElement.style.left = '50%';
    tooltipElement.style.bottom = 'calc(100% + 10px)';
    tooltipElement.style.transform = 'translate(-50%, 6px)';
    tooltipElement.style.opacity = '0';
    tooltipElement.style.pointerEvents = 'none';
    tooltipElement.style.width = 'max-content';
    tooltipElement.style.maxWidth = '260px';
    tooltipElement.style.padding = '10px 12px';
    tooltipElement.style.border = `1px solid ${MAZE_ARCH_TOOLTIP_BORDER_COLOR}`;
    tooltipElement.style.borderRadius = '12px';
    tooltipElement.style.background =
      'linear-gradient(180deg, rgba(0, 21, 34, 0.96), rgba(4, 11, 19, 0.99))';
    tooltipElement.style.boxShadow =
      '0 0 14px rgba(15, 181, 255, 0.34), inset 0 0 12px rgba(15, 181, 255, 0.08)';
    tooltipElement.style.backdropFilter = 'blur(6px)';
    tooltipElement.style.zIndex = '20';
    tooltipElement.style.transition =
      'opacity 120ms ease-out, transform 120ms ease-out';

    // Heading row.
    const headingElement = document.createElement('div');
    headingElement.textContent = resolveTooltipHeading(profile);
    headingElement.style.fontFamily = MAZE_ARCH_MONOSPACE_FONT;
    headingElement.style.fontSize = '11px';
    headingElement.style.fontWeight = '700';
    headingElement.style.letterSpacing = '0.08em';
    headingElement.style.textTransform = 'uppercase';
    headingElement.style.color = MAZE_ARCH_TOOLTIP_HEADING_COLOR;
    headingElement.style.marginBottom = '6px';
    headingElement.style.textShadow = '0 0 8px rgba(255, 154, 46, 0.35)';

    // Body lines.
    const bodyElement = document.createElement('div');
    bodyElement.style.display = 'flex';
    bodyElement.style.flexDirection = 'column';
    bodyElement.style.gap = '4px';
    bodyElement.style.fontFamily = MAZE_ARCH_MONOSPACE_FONT;
    bodyElement.style.fontSize = '10px';
    bodyElement.style.lineHeight = '1.35';
    bodyElement.style.color = MAZE_ARCH_BUTTON_COLOR_IDLE;

    for (const bodyLine of resolveTooltipBodyLines(profile)) {
      const lineElement = document.createElement('div');
      lineElement.textContent = bodyLine;
      lineElement.style.whiteSpace = 'normal';
      bodyElement.appendChild(lineElement);
    }

    // Arrow pointing down from the tooltip card.
    const arrowElement = document.createElement('div');
    arrowElement.style.position = 'absolute';
    arrowElement.style.left = '50%';
    arrowElement.style.bottom = '-6px';
    arrowElement.style.width = '12px';
    arrowElement.style.height = '12px';
    arrowElement.style.transform = 'translateX(-50%) rotate(45deg)';
    arrowElement.style.background = 'rgba(0, 21, 34, 0.98)';
    arrowElement.style.borderRight = `1px solid ${MAZE_ARCH_TOOLTIP_BORDER_COLOR}`;
    arrowElement.style.borderBottom = `1px solid ${MAZE_ARCH_TOOLTIP_BORDER_COLOR}`;

    tooltipElement.appendChild(headingElement);
    tooltipElement.appendChild(bodyElement);
    tooltipElement.appendChild(arrowElement);

    return tooltipElement;
  }
}

/**
 * Resolve the punchy tooltip heading for one ASCII Maze architecture profile.
 *
 * @param profile - Architecture profile to describe.
 * @returns Short heading string shown above the tooltip body.
 */
function resolveTooltipHeading(profile: ExampleArchitectureProfile): string {
  switch (profile.id) {
    case 'mlp':
      return 'MLP · Multi-Layer Perceptron';
    case 'random-sparse':
      return 'Sparse · Sparse Feed-Forward Graph';
    case 'narx':
      return 'NARX · Explicit Delay-Line Memory';
    case 'gru':
      return 'GRU · Gated Recurrent Unit';
    case 'lstm':
      return 'LSTM · Long Short-Term Memory';
  }
}

/**
 * Resolve the educational tooltip body lines for one ASCII Maze architecture profile.
 *
 * @param profile - Architecture profile to describe.
 * @returns Array of short explanatory sentences shown below the heading.
 */
function resolveTooltipBodyLines(
  profile: ExampleArchitectureProfile,
): string[] {
  switch (profile.id) {
    case 'mlp':
      return [
        'MLP is the plain feed-forward baseline: maze observations go in, direction scores come out, nothing is remembered between steps.',
        'It reacts only to what it sees right now. No built-in memory cell or delay shelf.',
        'Great reference point for comparing the memory-based families.',
      ];
    case 'random-sparse':
      return [
        'Sparse starts from the same feed-forward idea as MLP, but begins with fewer wires already drawn.',
        'Evolution must discover which connections deserve to exist.',
        'Great for watching useful structure emerge over time.',
      ];
    case 'narx':
      return [
        'NARX adds a short delay-line memory by feeding recent inputs and outputs back into the next step.',
        'Useful for tasks where knowing the last move helps predict the next best action.',
        'Carries recent state without full gating overhead.',
      ];
    case 'gru':
      return [
        'GRU is a gated recurrent block that learns what to carry forward and what to forget.',
        'Update and reset gates let the network hold relevant path history longer.',
        'More expressive than NARX for mazes that need longer-range memory.',
      ];
    case 'lstm':
      return [
        'LSTM adds a dedicated memory cell with separate input, forget, and output gates.',
        'Good at remembering key maze features over many steps without them fading.',
        'The most parameter-rich family — gives NEAT more structure to evolve but also more to tune.',
      ];
  }
}
