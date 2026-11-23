import { prefixScanPrepareWgsl, prefixScanWgsl } from "../shaders";
import { createComputePass, encodeComputePass, type ComputePass } from "./ComputePass";

export default class PrefixScan {
  private readonly levelInfo: GPUBuffer;
  private readonly prepare: ComputePass;
  private readonly scanPasses: ComputePass[] = [];
  private readonly offsetPasses: ComputePass[] = [];
  private readonly scratchBuffers: GPUBuffer[] = [];

  constructor(device: GPUDevice, input: GPUBuffer, output: GPUBuffer, count: GPUBuffer) {
    const u32Size = Uint32Array.BYTES_PER_ELEMENT;
    const createScratch = (label: string, size: number, extraUsage = 0): GPUBuffer => {
      const buffer = device.createBuffer({
        label,
        size,
        usage: GPUBufferUsage.STORAGE | extraUsage,
      });
      this.scratchBuffers.push(buffer);
      return buffer;
    };

    const levels: { input: GPUBuffer; output: GPUBuffer; sums: GPUBuffer }[] = [];
    let capacity = input.size / u32Size;
    let levelInput = input;
    let levelOutput = output;
    while (true) {
      const blockCount = Math.ceil(capacity / 256);
      const sums = createScratch(`scan block sums ${levels.length}`, blockCount * u32Size);
      levels.push({ input: levelInput, output: levelOutput, sums });
      if (blockCount === 1) break;
      capacity = blockCount;
      levelInput = sums;
      levelOutput = createScratch(`scan prefixes ${levels.length}`, capacity * u32Size);
    }

    this.levelInfo = createScratch(
      "scan level counts and dispatch args", levels.length * 16, GPUBufferUsage.INDIRECT,
    );
    this.prepare = createComputePass(device, {
      label: "scan prepare pipeline",
      passLabel: "scan prepare pass",
      compute: {
        module: device.createShaderModule({ code: prefixScanPrepareWgsl }),
        entryPoint: "prepareScan",
      },
      buffers: { 0: count, 1: this.levelInfo },
    });

    const module = device.createShaderModule({ label: "exclusive scan shader", code: prefixScanWgsl });
    for (let i = 0; i < levels.length; i++) {
      const level = levels[i];
      this.scanPasses.push(createComputePass(device, {
        label: `scan blocks pipeline ${i}`,
        passLabel: `scan/blocks/${i} pass`,
        compute: { module, entryPoint: "scanBlocks", constants: { SCAN_LEVEL: i } },
        buffers: { 0: level.input, 1: level.output, 2: level.sums, 4: this.levelInfo },
      }));

      if (i + 1 < levels.length) {
        this.offsetPasses.push(createComputePass(device, {
          label: `scan offsets pipeline ${i}`,
          passLabel: `scan/offsets/${i} pass`,
          compute: { module, entryPoint: "addBlockOffsets", constants: { SCAN_LEVEL: i } },
          buffers: { 1: level.output, 3: levels[i + 1].output, 4: this.levelInfo },
        }));
      }
    }
  }

  public encode(encoder: GPUCommandEncoder): void {
    encodeComputePass(encoder, this.prepare, 1);

    for (let i = 0; i < this.scanPasses.length; i++) {
      encodeComputePass(encoder, this.scanPasses[i], this.levelInfo, i * 16 + 4);
    }

    for (let i = this.offsetPasses.length - 1; i >= 0; i--) {
      encodeComputePass(encoder, this.offsetPasses[i], this.levelInfo, i * 16 + 4);
    }
  }

  public destroy(): void {
    for (const buffer of this.scratchBuffers) buffer.destroy();
  }
}
