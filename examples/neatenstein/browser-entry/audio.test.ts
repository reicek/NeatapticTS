import { describe, expect, it } from '@jest/globals';

const loadModule = (path: string): Promise<any> => import(path);

interface MockAudioNode {
  connect: jest.Mock;
}

interface MockAudioParam {
  value: number;
  setValueAtTime: jest.Mock;
  exponentialRampToValueAtTime: jest.Mock;
  cancelScheduledValues?: jest.Mock;
}

interface MockStereoPanner extends MockAudioNode {
  pan: MockAudioParam;
}

interface MockGain extends MockAudioNode {
  gain: MockAudioParam;
}

function installMockAudioContext(): {
  instances: Array<{
    resume: jest.Mock;
    createStereoPanner: jest.Mock;
    createGain: jest.Mock;
    createdNodes: Array<{ type: string; node: MockStereoPanner | MockGain }>;
  }>;
} {
  const instances: Array<
    ReturnType<typeof installMockAudioContext>['instances'][number]
  > = [];

  function createMockNode(): MockAudioNode {
    return { connect: jest.fn() };
  }

  (globalThis as unknown as Record<string, unknown>).AudioContext = class {
    resume = jest.fn();
    currentTime = 0;
    destination = createMockNode();
    createOscillator = jest.fn(() => ({
      ...createMockNode(),
      frequency: {
        value: 0,
        setValueAtTime: jest.fn(),
        exponentialRampToValueAtTime: jest.fn(),
      },
      start: jest.fn(),
      stop: jest.fn(),
      type: 'sine',
    }));
    createBiquadFilter = jest.fn(() => ({
      ...createMockNode(),
      frequency: { value: 0, setValueAtTime: jest.fn() },
      type: 'lowpass',
    }));
    createDynamicsCompressor = jest.fn(createMockNode);
    createdNodes: Array<{ type: string; node: MockStereoPanner | MockGain }> =
      [];
    createStereoPanner = jest.fn(() => {
      const node: MockStereoPanner = {
        ...createMockNode(),
        pan: {
          value: 0,
          setValueAtTime: jest.fn(),
          exponentialRampToValueAtTime: jest.fn(),
        },
      };
      this.createdNodes.push({ type: 'panner', node });
      return node;
    });
    createGain = jest.fn(() => {
      const node: MockGain = {
        ...createMockNode(),
        gain: {
          value: 1,
          setValueAtTime: jest.fn(),
          exponentialRampToValueAtTime: jest.fn(),
          cancelScheduledValues: jest.fn(),
        },
      };
      this.createdNodes.push({ type: 'gain', node });
      return node;
    });

    constructor() {
      instances.push(this);
    }
  };

  return { instances };
}

describe('Neatenstein audio engine', () => {
  it('exports the sound-name list or a factory function', async () => {
    const audio = await loadModule('./audio.ts');
    const hasNames = Array.isArray(audio.NEATENSTEIN_AUDIO_SOUND_NAMES);
    const hasFactory = typeof audio.createNeatensteinAudioEngine === 'function';
    expect(hasNames || hasFactory).toBe(true);
  });

  it('resumes the AudioContext on engine resume', async () => {
    const { instances } = installMockAudioContext();
    const audio = await loadModule('./audio.ts');
    const engine = audio.createNeatensteinAudioEngine();
    engine.resume();
    expect(instances[0].resume).toHaveBeenCalled();
  });

  it('accepts the six known sound names and rejects unknown names', async () => {
    installMockAudioContext();
    const audio = await loadModule('./audio.ts');
    const engine = audio.createNeatensteinAudioEngine();
    const known = [
      'fire',
      'enemy-hit',
      'player-damage',
      'dash',
      'kill',
      'generation-up',
    ];
    const acceptsAll = known.every((name) => {
      try {
        engine.playSound(name);
        return true;
      } catch {
        return false;
      }
    });
    let throwsForUnknown = false;
    try {
      engine.playSound('unknown-sound');
    } catch {
      throwsForUnknown = true;
    }
    expect({ acceptsAll, throwsForUnknown }).toEqual({
      acceptsAll: true,
      throwsForUnknown: true,
    });
  });

  it('applies positional pan and distance attenuation', async () => {
    const { instances } = installMockAudioContext();
    const audio = await loadModule('./audio.ts');
    const engine = audio.createNeatensteinAudioEngine();
    engine.playSound('enemy-hit', {
      angleRad: Math.PI / 4,
      distance: 5,
    });
    const panner = instances[0].createdNodes.find((n) => n.type === 'panner')
      ?.node as MockStereoPanner | undefined;
    const gainNode = instances[0].createdNodes.find((n) => n.type === 'gain')
      ?.node as MockGain | undefined;
    expect({
      panPositive: (panner?.pan.value ?? 0) > 0,
      gainAttenuated: (gainNode?.gain.value ?? 1) < 1,
    }).toEqual({ panPositive: true, gainAttenuated: true });
  });

  it('schedules generation-up audio on the same tick as the pulse', async () => {
    const audio = await loadModule('./audio.ts');
    const pulse = await loadModule('./renderer/pulse.ts');
    const simTick = 42;
    const audioScheduled = audio.scheduleNeatensteinGenerationUpAudio(simTick);
    const pulseScheduled = pulse.emitNeatensteinGenerationUpPulse(simTick);
    expect(audioScheduled && pulseScheduled).toBe(true);
  });

  it('does not instantiate an AudioContext at module load time in Node', async () => {
    const { instances } = installMockAudioContext();
    await loadModule('./audio.ts');
    expect(instances.length).toBe(0);
  });
});
