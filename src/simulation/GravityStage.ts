import { BODY_STRIDE } from "../Body";
import BufferOwner from "../gpu/BufferOwner";
import { createComputePass, encodeComputePass, type ComputePass } from "../gpu/ComputePass";
import { gravityNaiveWgsl, gravitySourceSplitWgsl } from "../shaders";
import { BODY_DISPATCH_CONSTANTS } from "./dispatch";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

const SOURCE_SEGMENTS = 8;
const PARTIAL_ACCELERATION_STRIDE = 16;

export default class GravityStage {
  private readonly drift: ComputePass;
  private readonly gravityPasses: Record<Exclude<SimulationAlgorithms["gravity"], "source-split">, ComputePass>;
  private readonly dispatchArgs: GPUBuffer;
  private readonly splitDispatch: GPUBuffer;
  private readonly prepareSplit: ComputePass;
  private readonly accumulateSplit: ComputePass;
  private readonly integrateSplit: ComputePass;
  private readonly bufferOwner: BufferOwner;

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.dispatchArgs = buffers.dispatchArgs;
    this.bufferOwner = new BufferOwner(device);

    const module = device.createShaderModule({
      label: "naive gravity shader",
      code: gravityNaiveWgsl,
    });

    this.drift = createComputePass(device, {
      label: "drift half pipeline",
      bindGroupLabel: "drift half bind group",
      passLabel: "drift half pass",
      compute: { module, entryPoint: "driftHalf", constants: BODY_DISPATCH_CONSTANTS },
      buffers: [buffers.bodiesA, buffers.bodiesB, buffers.params],
    });
    const gravityBuffers = [buffers.bodiesB, buffers.bodiesA, buffers.params];
    const tiledModule = device.createShaderModule({
      label: "tiled gravity shader",
      code: gravitySourceSplitWgsl,
    });
    this.gravityPasses = {
      naive: createComputePass(device, {
        label: "naive gravity pipeline",
        bindGroupLabel: "naive gravity bind group",
        passLabel: "naive gravity pass",
        compute: { module, entryPoint: "gravityNaive", constants: BODY_DISPATCH_CONSTANTS },
        buffers: gravityBuffers,
      }),
      tiled: createComputePass(device, {
        label: "tiled gravity pipeline",
        bindGroupLabel: "tiled gravity bind group",
        passLabel: "tiled gravity pass",
        compute: {
          module: tiledModule,
          entryPoint: "gravityTiled",
          constants: BODY_DISPATCH_CONSTANTS,
        },
        buffers: gravityBuffers,
      }),
    };

    const capacity = buffers.bodiesA.size / BODY_STRIDE;
    const partials = this.bufferOwner.createBuffer({
      label: "gravity partial accelerations",
      size: capacity * SOURCE_SEGMENTS * PARTIAL_ACCELERATION_STRIDE,
      usage: GPUBufferUsage.STORAGE,
    });
    this.splitDispatch = this.bufferOwner.createBuffer({
      label: "gravity source-split dispatch args",
      size: 3 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });
    const constants = { ...BODY_DISPATCH_CONSTANTS, SOURCE_SEGMENTS };
    this.prepareSplit = createComputePass(device, {
      label: "gravity/source-split/prepare pass",
      compute: { module: tiledModule, entryPoint: "prepareGravitySplit", constants },
      buffers: { 2: buffers.params, 4: this.splitDispatch },
    });
    this.accumulateSplit = createComputePass(device, {
      label: "gravity/source-split/partials pass",
      compute: { module: tiledModule, entryPoint: "accumulateGravitySplit", constants },
      buffers: { 0: buffers.bodiesB, 2: buffers.params, 3: partials },
    });
    this.integrateSplit = createComputePass(device, {
      label: "gravity/source-split/integrate pass",
      compute: { module: tiledModule, entryPoint: "integrateGravitySplit", constants },
      buffers: { 0: buffers.bodiesB, 1: buffers.bodiesA, 2: buffers.params, 3: partials },
    });
  }

  public encodeDrift(encoder: GPUCommandEncoder): void {
    encodeComputePass(encoder, this.drift, this.dispatchArgs);
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["gravity"],
  ): void {
    if (algorithm === "source-split") {
      encodeComputePass(encoder, this.prepareSplit, 1);
      encodeComputePass(encoder, this.accumulateSplit, this.splitDispatch);
      encodeComputePass(encoder, this.integrateSplit, this.dispatchArgs);
      return;
    }
    encodeComputePass(encoder, this.gravityPasses[algorithm], this.dispatchArgs);
  }

  public destroy(): void {
    this.bufferOwner.destroy();
  }
}