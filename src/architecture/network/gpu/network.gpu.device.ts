export async function requestGPUDevice(): Promise<GPUDevice | null> {
  throw new Error('requestGPUDevice not implemented');
}

export function isDeviceReady(device: GPUDevice | null): boolean {
  throw new Error('isDeviceReady not implemented');
}
