#!/usr/bin/env node
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { access, constants, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { after, before, describe, it } from 'node:test';

import { MCP_PROTOCOL_VERSION } from '../mcp-utils.mjs';

const SERVER_ENTRYPOINT =
  'scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs';
const REPO_ROOT = path.resolve(import.meta.dirname, '../../../../');
const SYNTHETIC_AGENT_RELATIVE_PATH =
  '.github/agents/__test-synthetic-userinvocable.agent.md';
const EXPECTED_TOOL_NAMES = [
  'build_dispatch_packet',
  'get_dispatch_policy',
  'list_dispatchable_agents',
];

const SYNTHETIC_AGENT_NAME = '__test-synthetic-userinvocable';

before(async () => {
  await removeSyntheticAgentFixture();
});

after(async () => {
  await removeSyntheticAgentFixture();
});

describe('neataptic-dispatch-mcp red contracts', () => {
  it('echoes the MCP protocol version during the initialize handshake', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.initialize();
      assert.strictEqual(result.protocolVersion, MCP_PROTOCOL_VERSION);
    } finally {
      client.close();
    }
  });

  it('exposes exactly the planned dispatch tools', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.request('tools/list');
      const names = result.tools.map((tool) => tool.name).sort();
      assert.deepStrictEqual(names, EXPECTED_TOOL_NAMES.toSorted());
    } finally {
      client.close();
    }
  });

  it('returns a sorted agent list that includes Tier 1 and hidden specialist agents', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('list_dispatchable_agents');
      const payload = result.structuredContent;
      const names = payload.agents.map((agent) => agent.name);
      const isSorted = names.every(
        (name, index) => index === 0 || names[index - 1] <= name,
      );
      assert.ok(
        Array.isArray(names) &&
          names.length > 0 &&
          names.includes('01-planning') &&
          names.includes('plan-scout') &&
          isSorted,
        `expected non-empty sorted list including 01-planning and plan-scout, got ${names.join(', ')}`,
      );
    } finally {
      client.close();
    }
  });

  it('allows a valid downward delegation and returns a dispatch packet', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('build_dispatch_packet', {
        target_agent: 'plan-scout',
        caller_tier: 1,
        prompt: 'Continue from the active plan.',
        context_tier: 'default',
      });
      const payload = result.structuredContent;
      assert.strictEqual(payload.ok, true);
      assert.strictEqual(payload.dispatch_allowed, true);
      const packetFields = Object.keys(payload.dispatch_packet ?? {}).sort();
      for (const field of [
        'agent_type',
        'context_tier',
        'description',
        'model',
        'name',
        'prompt',
        'skills',
      ]) {
        assert.ok(
          packetFields.includes(field),
          `expected dispatch_packet to include ${field}, got ${packetFields.join(', ')}`,
        );
      }
    } finally {
      client.close();
    }
  });

  it('allows Tier 0 to dispatch to a Tier 1 agent even when inventory returns string tiers', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('build_dispatch_packet', {
        target_agent: '01-planning',
        caller_tier: 0,
        prompt: 'Plan the next implementation step.',
        context_tier: 'default',
      });
      const payload = result.structuredContent;
      assert.deepStrictEqual(
        {
          ok: payload.ok,
          dispatchAllowed: payload.dispatch_allowed,
          targetTier: payload.agent?.tier,
        },
        {
          ok: true,
          dispatchAllowed: true,
          targetTier: 1,
        },
      );
    } finally {
      client.close();
    }
  });

  it('rejects an upward delegation from Tier 3 to Tier 1 with a clear reason', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('build_dispatch_packet', {
        target_agent: '01-planning',
        caller_tier: 3,
      });
      const payload = result.structuredContent;
      assert.deepStrictEqual(
        {
          ok: payload.ok,
          dispatchAllowed: payload.dispatch_allowed,
          hasReason: typeof payload.reason === 'string',
        },
        { ok: false, dispatchAllowed: false, hasReason: true },
      );
      assert.match(
        payload.reason,
        /target tier must be greater than the caller tier/i,
      );
    } finally {
      client.close();
    }
  });

  it('rejects an unknown target agent', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('build_dispatch_packet', {
        target_agent: '__nonexistent-agent__',
        caller_tier: 1,
      });
      const payload = result.structuredContent;
      assert.deepStrictEqual(
        {
          ok: payload.ok,
          dispatchAllowed: payload.dispatch_allowed,
          hasReason: typeof payload.reason === 'string',
        },
        { ok: false, dispatchAllowed: false, hasReason: true },
      );
    } finally {
      client.close();
    }
  });

  it('rejects a Tier 3 agent marked userInvocable because only Tier 1 agents may be user-invocable', async () => {
    await writeSyntheticAgentFixture();
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('build_dispatch_packet', {
        target_agent: SYNTHETIC_AGENT_NAME,
        caller_tier: 1,
      });
      const payload = result.structuredContent;
      assert.match(
        payload.reason,
        /userInvocable is only valid for Tier 1 agents/i,
      );
    } finally {
      client.close();
      await removeSyntheticAgentFixture();
    }
  });

  it('returns the allowed delegation edges and user-invocable rule', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('get_dispatch_policy');
      const policy = result.structuredContent;
      assert.deepStrictEqual(policy.allowed_edges, [
        { from: 0, to: 1 },
        { from: 1, to: 2 },
        { from: 1, to: 3 },
        { from: 1, to: 4 },
        { from: 2, to: 3 },
        { from: 2, to: 4 },
        { from: 3, to: 4 },
      ]);
      assert.strictEqual(
        policy.user_invocable_rule,
        'Only Tier 1 agents may be userInvocable',
      );
    } finally {
      client.close();
    }
  });

  it('rejects a prompt that exceeds the maximum allowed length', async () => {
    const client = await startDispatchServer();
    try {
      const longPrompt = 'x'.repeat(501);
      const result = await client.callTool('build_dispatch_packet', {
        target_agent: 'plan-scout',
        caller_tier: 1,
        prompt: longPrompt,
        context_tier: 'default',
      });
      const payload = result.structuredContent;
      assert.deepStrictEqual(
        {
          ok: payload.ok,
          dispatchAllowed: payload.dispatch_allowed,
          hasReason: typeof payload.reason === 'string',
        },
        { ok: false, dispatchAllowed: false, hasReason: true },
      );
      assert.match(
        payload.reason,
        /prompt length|prompt too long|prompt exceeds/i,
      );
    } finally {
      client.close();
    }
  });

  it('exposes prompt length rule and maximum through dispatch policy', async () => {
    const client = await startDispatchServer();
    try {
      const result = await client.callTool('get_dispatch_policy');
      const policy = result.structuredContent;
      assert.ok(
        typeof policy.prompt_length_rule === 'string',
        'expected prompt_length_rule to be a string',
      );
      assert.ok(
        typeof policy.prompt_length_max === 'number',
        'expected prompt_length_max to be a number',
      );
    } finally {
      client.close();
    }
  });
});

async function startDispatchServer() {
  const entrypointAbsolute = path.resolve(REPO_ROOT, SERVER_ENTRYPOINT);
  await access(entrypointAbsolute, constants.F_OK);
  const child = spawn(process.execPath, [entrypointAbsolute], {
    cwd: REPO_ROOT,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  return createStdioClient(child);
}

function createStdioClient(child) {
  let nextId = 1;
  const pending = new Map();
  let buffer = '';
  let stderrBuffer = '';

  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    buffer += chunk;
    let newlineIndex;
    while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      if (!line.trim()) continue;

      try {
        const message = JSON.parse(line);
        const resolver = pending.get(message.id);
        if (!resolver) continue;
        pending.delete(message.id);
        clearTimeout(resolver.timer);
        if (message.error) {
          resolver.reject(new Error(String(message.error.message)));
        } else {
          resolver.resolve(message.result);
        }
      } catch {
        // Ignore non-JSON diagnostic lines.
      }
    }
  });

  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    stderrBuffer += chunk;
  });

  child.on('error', (error) => {
    for (const resolver of pending.values()) {
      clearTimeout(resolver.timer);
      resolver.reject(error);
    }
    pending.clear();
  });

  child.on('close', (exitCode) => {
    if (pending.size === 0) return;
    const error = new Error(
      `MCP server exited with code ${exitCode}. stderr: ${stderrBuffer.slice(-500)}`,
    );
    for (const resolver of pending.values()) {
      clearTimeout(resolver.timer);
      resolver.reject(error);
    }
    pending.clear();
  });

  return {
    request(method, params) {
      const id = nextId++;
      const envelope = { jsonrpc: '2.0', id, method };
      if (params !== undefined) {
        envelope.params = params;
      }
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(
            new Error(
              `MCP request ${method} timed out. stderr: ${stderrBuffer.slice(-500)}`,
            ),
          );
        }, 10_000);
        pending.set(id, { resolve, reject, timer });
        child.stdin.write(`${JSON.stringify(envelope)}\n`);
      });
    },
    initialize() {
      return this.request('initialize', {
        protocolVersion: MCP_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: {
          name: 'neataptic-dispatch-red-test',
          version: '0.1.0',
        },
      });
    },
    callTool(name, args = {}) {
      return this.request('tools/call', { name, arguments: args });
    },
    close() {
      child.stdin.end();
      child.kill();
      for (const resolver of pending.values()) {
        clearTimeout(resolver.timer);
        resolver.reject(new Error('MCP client closed before response'));
      }
      pending.clear();
    },
  };
}

async function writeSyntheticAgentFixture() {
  const fixturePath = path.resolve(REPO_ROOT, SYNTHETIC_AGENT_RELATIVE_PATH);
  const fixtureText = `---
name: ${SYNTHETIC_AGENT_NAME}
tier: 3
user-invocable: true
model: test-model
skills: [test-skill]
description: Synthetic fixture for userInvocable guard red test.
---

# ${SYNTHETIC_AGENT_NAME}

This agent exists only to verify the userInvocable guard.
`;
  await writeFile(fixturePath, fixtureText);
}

async function removeSyntheticAgentFixture() {
  try {
    await unlink(path.resolve(REPO_ROOT, SYNTHETIC_AGENT_RELATIVE_PATH));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
}
