import {
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
} from '../../constants/constants';
import { FLAPPY_HOST_TABLE_FONT_SIZE } from './host.constants';
import type {
  CanvasHostOptions,
  HostArchitectureSelectorController,
  HostArchitectureSelectorItem,
} from './host.types';

const FLAPPY_ARCHITECTURE_SELECTOR_TITLE_TEXT = 'Architectures';
const FLAPPY_ARCHITECTURE_SELECTOR_RESET_TEXT = 'Reset Scores';
const FLAPPY_ARCHITECTURE_SELECTOR_GRID_COLUMNS = 'repeat(2, minmax(0, 1fr))';
const FLAPPY_ARCHITECTURE_SELECTOR_ITEM_GAP_PX = 4;
const FLAPPY_ARCHITECTURE_SELECTOR_GRID_GAP_PX = 6;
const FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_PADDING = '6px 8px';
const FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_RADIUS_PX = 10;
const FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_FONT_SIZE = '11px';
const FLAPPY_ARCHITECTURE_SELECTOR_CAPTION_FONT_SIZE = '10px';
const FLAPPY_ARCHITECTURE_TOOLTIP_OFFSET_PX = 10;
const FLAPPY_ARCHITECTURE_TOOLTIP_MAX_WIDTH_PX = 260;
const FLAPPY_ARCHITECTURE_TOOLTIP_PADDING = '10px 12px';
const FLAPPY_ARCHITECTURE_TOOLTIP_RADIUS_PX = 12;
const FLAPPY_ARCHITECTURE_TOOLTIP_HEADING_FONT_SIZE = '11px';
const FLAPPY_ARCHITECTURE_TOOLTIP_BODY_FONT_SIZE = '10px';
const FLAPPY_ARCHITECTURE_SELECTOR_TRANSITION =
  'border-color 120ms ease-out, box-shadow 120ms ease-out, color 120ms ease-out, background 120ms ease-out, transform 120ms ease-out';
const FLAPPY_ARCHITECTURE_TOOLTIP_TRANSITION =
  'opacity 120ms ease-out, transform 120ms ease-out';

/**
 * Creates the Flappy architecture selector control group used in the browser HUD.
 *
 * The selector is intentionally presentation-only. It renders the current shared
 * architecture profiles, exposes a narrow click callback, and lets the runtime
 * update local-record captions without rebuilding the whole host tree.
 *
 * @param options - Initial selector items and restart callback.
 * @returns Imperative controller for selector item and disabled-state updates.
 */
export function createHostArchitectureSelector(
  options: CanvasHostOptions,
): HostArchitectureSelectorController {
  const rootElement = document.createElement('div');
  const titleElement = document.createElement('div');
  const gridElement = document.createElement('div');
  let currentItems = options.architectureSelectorItems;
  let disabled = false;

  // Step 1: Style the host container and static section title.
  rootElement.style.display = 'flex';
  rootElement.style.flexDirection = 'column';
  rootElement.style.gap = `${FLAPPY_ARCHITECTURE_SELECTOR_GRID_GAP_PX}px`;
  rootElement.style.width = '100%';
  rootElement.style.boxSizing = 'border-box';

  titleElement.textContent = FLAPPY_ARCHITECTURE_SELECTOR_TITLE_TEXT;
  titleElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  titleElement.style.fontSize = FLAPPY_HOST_TABLE_FONT_SIZE;
  titleElement.style.fontWeight = '700';
  titleElement.style.textTransform = 'uppercase';
  titleElement.style.letterSpacing = '0.08em';
  titleElement.style.color = FLAPPY_NEON_PALETTE.hudAccent;

  gridElement.style.display = 'grid';
  gridElement.style.gridTemplateColumns =
    FLAPPY_ARCHITECTURE_SELECTOR_GRID_COLUMNS;
  gridElement.style.gap = `${FLAPPY_ARCHITECTURE_SELECTOR_GRID_GAP_PX}px`;
  gridElement.style.width = '100%';

  rootElement.appendChild(titleElement);
  rootElement.appendChild(gridElement);

  // Step 1b: Add the Reset Scores button below the architecture grid when a callback is provided.
  if (options.onResetScores) {
    const resetButtonElement = createResetScoresButtonElement(
      options.onResetScores,
    );
    rootElement.appendChild(resetButtonElement);
  }

  const updateItems = (nextItems: HostArchitectureSelectorItem[]): void => {
    currentItems = nextItems;
    gridElement.replaceChildren(
      ...nextItems.map((selectorItem) =>
        createArchitectureSelectorItemElement(selectorItem, disabled, () => {
          options.onSelectArchitectureProfile?.(selectorItem.id);
        }),
      ),
    );
  };

  // Step 2: Render the initial selector item set.
  updateItems(currentItems);

  return {
    element: rootElement,
    setDisabled: (nextDisabled: boolean): void => {
      disabled = nextDisabled;
      updateItems(currentItems);
    },
    updateItems,
  };
}

/**
 * Creates one selector item element with hover glow and optional best-score caption.
 *
 * @param selectorItem - Render-ready button state.
 * @param disabled - Whether the selector is temporarily disabled.
 * @param onClick - Click callback invoked for fresh-run selection.
 * @returns Rendered selector item wrapper.
 */
function createArchitectureSelectorItemElement(
  selectorItem: HostArchitectureSelectorItem,
  disabled: boolean,
  onClick: () => void,
): HTMLDivElement {
  const itemElement = document.createElement('div');
  const buttonElement = document.createElement('button');
  const captionElement = document.createElement('div');
  const tooltipElement = createArchitectureSelectorTooltipElement(selectorItem);
  const tooltipId = `flappy-architecture-tooltip-${selectorItem.id}`;

  // Step 1: Build the item wrapper and action button structure.
  itemElement.style.display = 'flex';
  itemElement.style.flexDirection = 'column';
  itemElement.style.gap = `${FLAPPY_ARCHITECTURE_SELECTOR_ITEM_GAP_PX}px`;
  itemElement.style.minWidth = '0';
  itemElement.style.position = 'relative';
  itemElement.style.overflow = 'visible';
  itemElement.style.zIndex = '0';

  buttonElement.type = 'button';
  buttonElement.textContent = selectorItem.label;
  buttonElement.disabled = disabled;
  buttonElement.setAttribute('aria-describedby', tooltipId);
  buttonElement.style.width = '100%';
  buttonElement.style.boxSizing = 'border-box';
  buttonElement.style.padding = FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_PADDING;
  buttonElement.style.borderRadius = `${FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_RADIUS_PX}px`;
  buttonElement.style.borderStyle = 'solid';
  buttonElement.style.borderWidth = '1px';
  buttonElement.style.background = 'transparent';
  buttonElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  buttonElement.style.fontSize = FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_FONT_SIZE;
  buttonElement.style.fontWeight = '700';
  buttonElement.style.textTransform = 'uppercase';
  buttonElement.style.letterSpacing = '0.08em';
  buttonElement.style.transition = FLAPPY_ARCHITECTURE_SELECTOR_TRANSITION;
  buttonElement.style.cursor = disabled ? 'default' : 'pointer';

  tooltipElement.id = tooltipId;

  applyArchitectureSelectorButtonPresentation(
    buttonElement,
    selectorItem,
    disabled,
    false,
  );

  const setTooltipVisible = (tooltipVisible: boolean): void => {
    tooltipElement.style.opacity = tooltipVisible ? '1' : '0';
    tooltipElement.style.transform = tooltipVisible
      ? 'translate(-50%, 0)'
      : 'translate(-50%, 6px)';
    itemElement.style.zIndex = tooltipVisible ? '3' : '0';
  };

  if (!disabled) {
    buttonElement.addEventListener('mouseenter', () => {
      setTooltipVisible(true);
      applyArchitectureSelectorButtonPresentation(
        buttonElement,
        selectorItem,
        disabled,
        true,
      );
    });
    buttonElement.addEventListener('mouseleave', () => {
      setTooltipVisible(false);
      applyArchitectureSelectorButtonPresentation(
        buttonElement,
        selectorItem,
        disabled,
        false,
      );
    });
    buttonElement.addEventListener('focus', () => {
      setTooltipVisible(true);
    });
    buttonElement.addEventListener('blur', () => {
      setTooltipVisible(false);
    });
    buttonElement.addEventListener('click', onClick);
  }

  // Step 2: Render the optional historical-best caption below the button.
  captionElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  captionElement.style.fontSize =
    FLAPPY_ARCHITECTURE_SELECTOR_CAPTION_FONT_SIZE;
  captionElement.style.letterSpacing = '0.06em';
  captionElement.style.textTransform = 'uppercase';
  captionElement.style.color = FLAPPY_NEON_PALETTE.bestRunText;
  captionElement.style.minHeight = '12px';
  captionElement.style.opacity = disabled ? '0.55' : '0.85';
  captionElement.style.visibility = selectorItem.caption ? 'visible' : 'hidden';
  captionElement.textContent = selectorItem.caption ?? '';

  itemElement.appendChild(tooltipElement);
  itemElement.appendChild(buttonElement);
  itemElement.appendChild(captionElement);
  return itemElement;
}

/**
 * Creates the neon tooltip shown above one architecture selector button.
 *
 * @param selectorItem - Render-ready selector item with heading and teaching copy.
 * @returns Tooltip element positioned above the button.
 */
function createArchitectureSelectorTooltipElement(
  selectorItem: HostArchitectureSelectorItem,
): HTMLDivElement {
  const tooltipElement = document.createElement('div');
  const tooltipHeadingElement = document.createElement('div');
  const tooltipBodyElement = document.createElement('div');
  const tooltipArrowElement = document.createElement('div');

  // Step 1: Style the floating tooltip shell and arrow.
  tooltipElement.setAttribute('role', 'tooltip');
  tooltipElement.style.position = 'absolute';
  tooltipElement.style.left = '50%';
  tooltipElement.style.bottom = `calc(100% + ${FLAPPY_ARCHITECTURE_TOOLTIP_OFFSET_PX}px)`;
  tooltipElement.style.transform = 'translate(-50%, 6px)';
  tooltipElement.style.opacity = '0';
  tooltipElement.style.pointerEvents = 'none';
  tooltipElement.style.width = 'max-content';
  tooltipElement.style.maxWidth = `${FLAPPY_ARCHITECTURE_TOOLTIP_MAX_WIDTH_PX}px`;
  tooltipElement.style.padding = FLAPPY_ARCHITECTURE_TOOLTIP_PADDING;
  tooltipElement.style.border = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
  tooltipElement.style.borderRadius = `${FLAPPY_ARCHITECTURE_TOOLTIP_RADIUS_PX}px`;
  tooltipElement.style.background =
    'linear-gradient(180deg, rgba(0, 21, 34, 0.96), rgba(4, 11, 19, 0.99))';
  tooltipElement.style.boxShadow =
    '0 0 14px rgba(15, 181, 255, 0.34), inset 0 0 12px rgba(15, 181, 255, 0.08)';
  tooltipElement.style.backdropFilter = 'blur(6px)';
  tooltipElement.style.zIndex = '20';
  tooltipElement.style.transition = FLAPPY_ARCHITECTURE_TOOLTIP_TRANSITION;

  tooltipArrowElement.style.position = 'absolute';
  tooltipArrowElement.style.left = '50%';
  tooltipArrowElement.style.bottom = '-6px';
  tooltipArrowElement.style.width = '12px';
  tooltipArrowElement.style.height = '12px';
  tooltipArrowElement.style.transform = 'translateX(-50%) rotate(45deg)';
  tooltipArrowElement.style.background = 'rgba(0, 21, 34, 0.98)';
  tooltipArrowElement.style.borderRight = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
  tooltipArrowElement.style.borderBottom = `1px solid ${FLAPPY_NEON_PALETTE.hudPanelBorder}`;
  tooltipArrowElement.style.boxShadow = '0 0 10px rgba(15, 181, 255, 0.26)';

  // Step 2: Render the heading and the educational body copy.
  tooltipHeadingElement.textContent = selectorItem.tooltipHeading;
  tooltipHeadingElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  tooltipHeadingElement.style.fontSize =
    FLAPPY_ARCHITECTURE_TOOLTIP_HEADING_FONT_SIZE;
  tooltipHeadingElement.style.fontWeight = '700';
  tooltipHeadingElement.style.letterSpacing = '0.08em';
  tooltipHeadingElement.style.textTransform = 'uppercase';
  tooltipHeadingElement.style.color = FLAPPY_NEON_PALETTE.hudAccent;
  tooltipHeadingElement.style.marginBottom = '6px';
  tooltipHeadingElement.style.textShadow = '0 0 8px rgba(255, 154, 46, 0.35)';

  tooltipBodyElement.style.display = 'flex';
  tooltipBodyElement.style.flexDirection = 'column';
  tooltipBodyElement.style.gap = '4px';
  tooltipBodyElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  tooltipBodyElement.style.fontSize =
    FLAPPY_ARCHITECTURE_TOOLTIP_BODY_FONT_SIZE;
  tooltipBodyElement.style.lineHeight = '1.35';
  tooltipBodyElement.style.color = FLAPPY_NEON_PALETTE.hudText;

  selectorItem.tooltipBodyLines.forEach((tooltipBodyLine) => {
    const tooltipLineElement = document.createElement('div');
    tooltipLineElement.textContent = tooltipBodyLine;
    tooltipLineElement.style.whiteSpace = 'normal';
    tooltipBodyElement.appendChild(tooltipLineElement);
  });

  tooltipElement.appendChild(tooltipHeadingElement);
  tooltipElement.appendChild(tooltipBodyElement);
  tooltipElement.appendChild(tooltipArrowElement);
  return tooltipElement;
}

/**
 * Applies the neon-outline presentation for one architecture selector button.
 *
 * @param buttonElement - Target button element.
 * @param selectorItem - Render-ready selector state.
 * @param disabled - Whether the selector is disabled.
 * @param hovered - Whether the pointer is currently hovering the button.
 * @returns Nothing.
 */
function applyArchitectureSelectorButtonPresentation(
  buttonElement: HTMLButtonElement,
  selectorItem: HostArchitectureSelectorItem,
  disabled: boolean,
  hovered: boolean,
): void {
  const activeBorderColor = selectorItem.selected
    ? FLAPPY_NEON_PALETTE.statusText
    : FLAPPY_NEON_PALETTE.hudPanelBorder;
  const activeTextColor = selectorItem.selected
    ? FLAPPY_NEON_PALETTE.statusText
    : FLAPPY_NEON_PALETTE.hudText;
  const hoverGlowColor = selectorItem.selected
    ? 'rgba(255, 92, 255, 0.7)'
    : 'rgba(15, 181, 255, 0.72)';

  buttonElement.style.borderColor = activeBorderColor;
  buttonElement.style.color = activeTextColor;
  buttonElement.style.background = selectorItem.selected
    ? 'rgba(255, 92, 255, 0.08)'
    : 'rgba(6, 11, 20, 0.55)';
  buttonElement.style.opacity = disabled ? '0.55' : '1';
  buttonElement.style.transform =
    hovered && !disabled ? 'translateY(-1px)' : 'translateY(0)';
  buttonElement.style.boxShadow = resolveArchitectureSelectorBoxShadow(
    selectorItem.selected,
    hovered && !disabled,
    hoverGlowColor,
  );
}

/**
 * Creates a "Reset Scores" button that clears all architecture best-score captions.
 *
 * The button sits below the architecture grid and matches the selector's visual
 * language while using a muted amber accent to signal that it is a clearing
 * action rather than a selection.
 *
 * @param onResetScores - Callback invoked when the button is clicked.
 * @returns Styled reset button element.
 */
function createResetScoresButtonElement(
  onResetScores: () => void,
): HTMLButtonElement {
  const resetButtonElement = document.createElement('button');

  resetButtonElement.type = 'button';
  resetButtonElement.textContent = FLAPPY_ARCHITECTURE_SELECTOR_RESET_TEXT;
  resetButtonElement.style.width = '100%';
  resetButtonElement.style.boxSizing = 'border-box';
  resetButtonElement.style.padding =
    FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_PADDING;
  resetButtonElement.style.borderRadius = `${FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_RADIUS_PX}px`;
  resetButtonElement.style.borderStyle = 'solid';
  resetButtonElement.style.borderWidth = '1px';
  resetButtonElement.style.borderColor = FLAPPY_NEON_PALETTE.hudPanelBorder;
  resetButtonElement.style.background = 'transparent';
  resetButtonElement.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  resetButtonElement.style.fontSize =
    FLAPPY_ARCHITECTURE_SELECTOR_BUTTON_FONT_SIZE;
  resetButtonElement.style.fontWeight = '700';
  resetButtonElement.style.textTransform = 'uppercase';
  resetButtonElement.style.letterSpacing = '0.08em';
  resetButtonElement.style.color = FLAPPY_NEON_PALETTE.hudText;
  resetButtonElement.style.opacity = '0.65';
  resetButtonElement.style.cursor = 'pointer';
  resetButtonElement.style.transition = FLAPPY_ARCHITECTURE_SELECTOR_TRANSITION;

  resetButtonElement.addEventListener('mouseenter', () => {
    resetButtonElement.style.opacity = '1';
    resetButtonElement.style.borderColor = FLAPPY_NEON_PALETTE.hudAccent;
    resetButtonElement.style.color = FLAPPY_NEON_PALETTE.hudAccent;
    resetButtonElement.style.transform = 'translateY(-1px)';
    resetButtonElement.style.boxShadow = `0 0 8px rgba(255, 154, 46, 0.55), inset 0 0 10px rgba(255, 154, 46, 0.08)`;
  });

  resetButtonElement.addEventListener('mouseleave', () => {
    resetButtonElement.style.opacity = '0.65';
    resetButtonElement.style.borderColor = FLAPPY_NEON_PALETTE.hudPanelBorder;
    resetButtonElement.style.color = FLAPPY_NEON_PALETTE.hudText;
    resetButtonElement.style.transform = 'translateY(0)';
    resetButtonElement.style.boxShadow = '';
  });

  resetButtonElement.addEventListener('click', onResetScores);

  return resetButtonElement;
}

/**
 * Resolves the neon glow stack for the architecture selector button state.
 *
 * @param selected - Whether the button is the active profile.
 * @param hovered - Whether the button is hovered.
 * @param hoverGlowColor - Primary glow color for the current button state.
 * @returns CSS box-shadow string.
 */
function resolveArchitectureSelectorBoxShadow(
  selected: boolean,
  hovered: boolean,
  hoverGlowColor: string,
): string {
  if (hovered) {
    return `0 0 8px ${hoverGlowColor}, inset 0 0 10px rgba(0, 229, 255, 0.16)`;
  }

  if (selected) {
    return '0 0 6px rgba(255, 92, 255, 0.48), inset 0 0 8px rgba(255, 92, 255, 0.12)';
  }

  return 'inset 0 0 6px rgba(15, 181, 255, 0.08)';
}
