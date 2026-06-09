export const LEARNING_LOG_PATH: string;

export function clearPreparedRuntimeContext(
  sessionId: string,
  actionId?: string | null,
): Promise<unknown>;

export function countTrailingGateFailures(
  events: Array<Record<string, unknown>>,
  sessionId: string,
): number;

export function getRuntimeContextPath(sessionId: string): string;

export function prepareRuntimeContext(
  options: Record<string, unknown>,
): Promise<any>;

export function readRuntimeContext(sessionId: string): Promise<any>;

export function diagnosePreparedRuntimeContext(
  options: Record<string, unknown>,
): {
  ok: boolean;
  reason: string;
  recoveryHint: string;
  actionClass: 'write' | 'execute' | null;
  preparedAction: any;
};

export function validatePreparedRuntimeContext(
  options: Record<string, unknown>,
): {
  ok: boolean;
  reason: string;
  recoveryHint: string;
  actionClass: 'write' | 'execute' | null;
  preparedAction: any;
};
