/**
 * Root compatibility facade for the ONNX-like import/export chapter.
 *
 * The implementation now lives under `network/onnx/`, where export, import,
 * and schema concerns are split into smaller chapters. This root file remains
 * so existing architecture-level imports keep working.
 */
// Backward compatibility shim: logic moved to network/network.onnx.ts
export * from './network/onnx/network.onnx';
export { default } from './network/onnx/network.onnx';
