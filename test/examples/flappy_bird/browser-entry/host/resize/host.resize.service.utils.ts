import { clamp } from '../../browser-entry.math.utils';
import {
  FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX,
  FLAPPY_VIEWPORT_MOBILE_MINIMAL_UI_BREAKPOINT_PX,
  FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_PX,
  FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_RATIO,
  FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
  FLAPPY_VIEWPORT_NETWORK_ONLY_BREAKPOINT_PX,
  FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
  FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
} from '../../../constants/constants';
import {
  FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
  FLAPPY_HOST_RESIZE_SPLIT_RATIO,
} from './host.resize.service.constants';
import type {
  ResponsiveViewportLayoutContext,
  SimulationCanvasBounds,
  StatsPanelDimensions,
} from './host.resize.service.types';

/**
 * Resolves responsive layout measurements and mode flags from the host DOM.
 *
 * @param containerElement - Width and height source.
 * @param statsContainer - Stats host element.
 * @param networkCanvasHost - Network host element.
 * @returns Responsive layout context.
 */
export function resolveResponsiveViewportLayoutContext(
  containerElement: HTMLElement,
  statsContainer: HTMLElement,
  networkCanvasHost: HTMLElement,
): ResponsiveViewportLayoutContext {
  // Step 1: Measure header height and viewport budgets.
  const headerHeightPx = resolveHeaderHeightPx(containerElement);
  const totalCanvasBudgetPx = Math.max(
    FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
    containerElement.clientHeight - headerHeightPx,
  );
  const viewportWidthPx = Math.max(
    FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
    containerElement.clientWidth,
  );
  const viewportHeightPx = Math.max(
    FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
    containerElement.clientHeight,
  );

  // Step 2: Measure panel-specific height constraints.
  const nonNetworkStatsHeightPx = Math.max(
    0,
    statsContainer.scrollHeight - networkCanvasHost.offsetHeight,
  );
  const hardMinimumSimulationHeightPx = Math.max(
    FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_PX,
    Math.floor(
      window.innerHeight * FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_RATIO,
    ),
  );

  // Step 3: Resolve the layout-mode flags used by responsive branching.
  const useMinimalMobileLayout =
    viewportWidthPx < FLAPPY_VIEWPORT_MOBILE_MINIMAL_UI_BREAKPOINT_PX;
  const useLandscapeSplitLayout = viewportWidthPx > viewportHeightPx;
  const useNetworkOnlyPanel =
    !useLandscapeSplitLayout &&
    viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_ONLY_BREAKPOINT_PX;

  return {
    headerHeightPx,
    nonNetworkStatsHeightPx,
    hardMinimumSimulationHeightPx,
    totalCanvasBudgetPx,
    viewportWidthPx,
    viewportHeightPx,
    useMinimalMobileLayout,
    useLandscapeSplitLayout,
    useNetworkOnlyPanel,
  };
}

/**
 * Resolves the simulation canvas bounds for the minimal mobile layout.
 *
 * @param containerElement - Width and height source.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @returns Simulation canvas bounds.
 */
export function resolveMinimalMobileCanvasBounds(
  containerElement: HTMLElement,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
): SimulationCanvasBounds {
  // Step 1: Use the full container width for the simulation canvas.
  const availableWidthPx = Math.max(
    FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
    Math.floor(containerElement.clientWidth),
  );

  // Step 2: Subtract the bottom breathing room while keeping a positive height.
  const availableHeightPx = Math.max(
    FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
    Math.floor(
      responsiveViewportLayoutContext.totalCanvasBudgetPx -
        FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
    ),
  );

  return {
    availableWidthPx,
    availableHeightPx,
  };
}

/**
 * Resolves the stats panel height and width budgets.
 *
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @returns Stats panel dimensions.
 */
export function resolveStatsPanelDimensions(
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
): StatsPanelDimensions {
  // Step 1: Resolve the preferred stats panel height from the vertical split budget.
  const halfSplitTargetHeightPx = Math.max(
    FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
    Math.floor(
      (responsiveViewportLayoutContext.totalCanvasBudgetPx -
        FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX) *
        FLAPPY_HOST_RESIZE_SPLIT_RATIO,
    ),
  );
  const maximumStatsHeightPx = Math.max(
    FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
    responsiveViewportLayoutContext.totalCanvasBudgetPx -
      responsiveViewportLayoutContext.hardMinimumSimulationHeightPx -
      FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
  );
  const resolvedStatsPanelHeightPx =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? responsiveViewportLayoutContext.totalCanvasBudgetPx
      : Math.floor(
          clamp(
            halfSplitTargetHeightPx,
            FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
            maximumStatsHeightPx,
          ),
        );
  const minimumStatsPanelHeightPx =
    responsiveViewportLayoutContext.nonNetworkStatsHeightPx +
    FLAPPY_UI_NETWORK_HOST_FIXED_HEIGHT_PX;

  // Step 2: Resolve the stats panel width budget for landscape layouts.
  const halfSplitTargetWidthPx = Math.max(
    FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
    Math.floor(
      (responsiveViewportLayoutContext.viewportWidthPx -
        FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX) *
        FLAPPY_HOST_RESIZE_SPLIT_RATIO,
    ),
  );
  const maximumStatsWidthPx = Math.max(
    FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
    responsiveViewportLayoutContext.viewportWidthPx -
      FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX -
      FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
  );

  return {
    adjustedStatsPanelHeightPx:
      responsiveViewportLayoutContext.useLandscapeSplitLayout
        ? resolvedStatsPanelHeightPx
        : Math.max(resolvedStatsPanelHeightPx, minimumStatsPanelHeightPx),
    resolvedStatsPanelWidthPx: Math.floor(
      clamp(
        halfSplitTargetWidthPx,
        FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX,
        maximumStatsWidthPx,
      ),
    ),
  };
}

/**
 * Resolves simulation canvas bounds from the current viewport layout.
 *
 * @param containerElement - Width and height source.
 * @param responsiveViewportLayoutContext - Responsive layout context.
 * @param statsPanelDimensions - Resolved stats panel dimensions.
 * @returns Simulation canvas bounds.
 */
export function resolveSimulationCanvasBounds(
  containerElement: HTMLElement,
  responsiveViewportLayoutContext: ResponsiveViewportLayoutContext,
  statsPanelDimensions: StatsPanelDimensions,
): SimulationCanvasBounds {
  // Step 1: Resolve the simulation width from the active split orientation.
  const availableWidthPx =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? Math.max(
          FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
          containerElement.clientWidth -
            statsPanelDimensions.resolvedStatsPanelWidthPx -
            FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX,
        )
      : Math.max(
          FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
          containerElement.clientWidth,
        );

  // Step 2: Resolve the simulation height from the remaining vertical budget.
  const availableHeightPx =
    responsiveViewportLayoutContext.useLandscapeSplitLayout
      ? Math.max(
          FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
          responsiveViewportLayoutContext.totalCanvasBudgetPx -
            FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
        )
      : Math.max(
          FLAPPY_HOST_RESIZE_MIN_DIMENSION_PX,
          responsiveViewportLayoutContext.totalCanvasBudgetPx -
            statsPanelDimensions.adjustedStatsPanelHeightPx -
            FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX -
            FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX,
        );

  return {
    availableWidthPx,
    availableHeightPx,
  };
}

/**
 * Resolves the header canvas height, if present.
 *
 * @param containerElement - Width and height source.
 * @returns Header height in pixels.
 */
function resolveHeaderHeightPx(containerElement: HTMLElement): number {
  // Step 1: Locate the header canvas hosted above the simulation viewport.
  const headerCanvasElement = containerElement.querySelector('canvas');

  // Step 2: Return the canvas height when found, otherwise fall back to zero.
  return Math.max(
    0,
    headerCanvasElement instanceof HTMLCanvasElement
      ? headerCanvasElement.offsetHeight
      : 0,
  );
}
