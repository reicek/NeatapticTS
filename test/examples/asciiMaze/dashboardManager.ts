/**
 * Compatibility entrypoint for the dedicated dashboardManager module.
 *
 * The real implementation now lives under `dashboardManager/dashboardManager.ts`.
 * This file remains so existing imports such as `./dashboardManager` continue
 * to resolve without changes.
 */

export { DashboardManager } from './dashboardManager/dashboardManager';
export type {
  AsciiMazeDetailedStats,
  AsciiMazeTelemetrySnapshot,
  DashboardTelemetry,
  DashboardTelemetryPayload,
  RuntimeDashboardManager,
} from './dashboardManager/dashboardManager.types';
