import { createComputePass, encodeComputePass, type ComputePass } from "../gpu/ComputePass";
import { gravityNaiveWgsl, gravityTiledWgsl } from "../shaders";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

export default class GravityStage {
  private readonly drift: ComputePass;
  private readonly gravityPasses: Record<SimulationAlgorithms["gravity"], ComputePass>;
  private readonly dispatchArgs: GPUBuffer;

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.dispatchArgs = buffers.dispatchArgs;

    const module = device.createShaderModule({
      label: "naive gravity shader",
      code: gravityNaiveWgsl,
    });

    this.drift = createComputePass(device, {
      label: "drift half pipeline",
      bindGroupLabel: "drift half bind group",
      passLabel: "drift half pass",
      compute: { module, entryPoint: "driftHalf" },
      buffers: [buffers.bodiesA, buffers.bodiesB, buffers.params],
    });
    const gravityBuffers = [buffers.bodiesB, buffers.bodiesA, buffers.params];
    this.gravityPasses = {
      naive: createComputePass(device, {
        label: "naive gravity pipeline",
        bindGroupLabel: "naive gravity bind group",
        passLabel: "naive gravity pass",
        compute: { module, entryPoint: "gravityNaive" },
        buffers: gravityBuffers,
      }),
      tiled: createComputePass(device, {
        label: "tiled gravity pipeline",
        bindGroupLabel: "tiled gravity bind group",
        passLabel: "tiled gravity pass",
        compute: {
          module: device.createShaderModule({
            label: "tiled gravity shader",
            code: gravityTiledWgsl,
          }),
          entryPoint: "gravityTiled",
        },
        buffers: gravityBuffers,
      }),
    };
  }

  public encodeDrift(encoder: GPUCommandEncoder): void {
    encodeComputePass(encoder, this.drift, this.dispatchArgs);
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["gravity"],
  ): void {
    encodeComputePass(encoder, this.gravityPasses[algorithm], this.dispatchArgs);
  }
}
