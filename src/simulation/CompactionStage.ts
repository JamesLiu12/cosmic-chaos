import BufferOwner from "../gpu/BufferOwner";
import { createComputePass, encodeComputePass, type ComputePass } from "../gpu/ComputePass";
import PrefixScan from "../gpu/PrefixScan";
import { compactScanWgsl, compactSerialWgsl } from "../shaders";
import { BODY_DISPATCH_CONSTANTS } from "./dispatch";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

export default class CompactionStage {
  private readonly serial: ComputePass;
  private readonly params: GPUBuffer;
  private readonly dispatchArgs: GPUBuffer;
  private readonly scanCount: GPUBuffer;
  private readonly scan: PrefixScan;
  private readonly markPass: ComputePass;
  private readonly scatterPass: ComputePass;
  private readonly publishPass: ComputePass;
  private readonly bufferOwner: BufferOwner;

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.bufferOwner = new BufferOwner(device);
    this.params = buffers.params;
    this.dispatchArgs = buffers.dispatchArgs;
    const { bodiesA, bodiesB, collisionCandidates, params } = buffers;
    const bodyArrayBytes = collisionCandidates.size;
    const oldToNew = this.bufferOwner.createBuffer({
      label: "compaction slot mapping",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.serial = createComputePass(device, {
      label: "compaction pipeline",
      bindGroupLabel: "compaction bind group",
      passLabel: "compaction pass",
      compute: {
        module: device.createShaderModule({
          label: "compaction shader",
          code: compactSerialWgsl,
        }),
        entryPoint: "compactBodies",
      },
      buffers: { 0: bodiesB, 1: bodiesA, 2: collisionCandidates, 3: params, 4: oldToNew },
    });

    const alive = this.bufferOwner.createBuffer({
      label: "compaction alive flags",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const prefixes = this.bufferOwner.createBuffer({
      label: "compaction exclusive prefixes",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const candidateScratch = this.bufferOwner.createBuffer({
      label: "compaction remapped candidates",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.scanCount = this.bufferOwner.createBuffer({
      label: "compaction input count",
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.scan = new PrefixScan(device, alive, prefixes, this.scanCount);

    const module = device.createShaderModule({
      label: "scan compaction shader",
      code: compactScanWgsl,
    });
    this.markPass = createComputePass(device, {
      label: "compaction/scan/markAlive pass",
      compute: { module, entryPoint: "markAlive", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 0: bodiesB, 3: params, 5: alive },
    });
    this.scatterPass = createComputePass(device, {
      label: "compaction/scan/scatterBodies pass",
      compute: { module, entryPoint: "scatterBodies", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 0: bodiesB, 1: bodiesA, 2: collisionCandidates, 3: params, 5: alive, 6: prefixes, 7: candidateScratch },
    });
    this.publishPass = createComputePass(device, {
      label: "compaction/scan/publishCandidates pass",
      compute: { module, entryPoint: "publishCandidates", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 2: collisionCandidates, 3: params, 4: this.scanCount, 5: alive, 6: prefixes, 7: candidateScratch },
    });
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["compaction"],
  ): void {
    switch (algorithm) {
      case "serial": {
        encodeComputePass(encoder, this.serial, 1);
        break;
      }
      case "scan": {
        encoder.copyBufferToBuffer(this.params, 12, this.scanCount, 0, 4);
        encodeComputePass(encoder, this.markPass, this.dispatchArgs);
        this.scan.encode(encoder);
        encodeComputePass(encoder, this.scatterPass, this.dispatchArgs);
        encodeComputePass(encoder, this.publishPass, this.dispatchArgs);
        break;
      }
    }
  }

  public destroy(): void {
    this.scan.destroy();
    this.bufferOwner.destroy();
  }
}