import PrefixScan from "../gpu/PrefixScan";
import { absorptionDonorListsWgsl, absorptionFullScanWgsl } from "../shaders";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

type AbsorptionPass = {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup
};

export default class AbsorptionStage {
  private readonly fullScan: AbsorptionPass;
  private readonly countDonors: AbsorptionPass;
  private readonly scatterDonors: AbsorptionPass;
  private readonly resolveReceivers: AbsorptionPass;
  private readonly updateDonors: AbsorptionPass;
  private readonly dispatchArgs: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly donorCounts: GPUBuffer;
  private readonly scanCount: GPUBuffer;
  private readonly scan: PrefixScan;
  private readonly scratchBuffers: GPUBuffer[] = [];

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.dispatchArgs = buffers.dispatchArgs;
    this.params = buffers.params;
    const createScratch = (label: string, size = buffers.collisionCandidates.size, extraUsage = 0) => {
      const buffer = device.createBuffer({
        label: `absorption ${label}`, size, usage: GPUBufferUsage.STORAGE | extraUsage,
      });
      this.scratchBuffers.push(buffer);
      return buffer;
    };
    this.donorCounts = createScratch("donor counts", buffers.collisionCandidates.size, GPUBufferUsage.COPY_DST);
    this.scanCount = createScratch("scan input count", 4, GPUBufferUsage.COPY_DST);
    const listOffsets = createScratch("donor list offsets");
    const donorRanks = createScratch("donor ranks");
    const donorIndices = createScratch("donor indices");
    const receiverAccepted = createScratch("receiver decisions");
    this.scan = new PrefixScan(device, this.donorCounts, listOffsets, this.scanCount);

    const resources = [
      buffers.bodiesA, buffers.bodiesB, buffers.collisionCandidates, buffers.params,
      this.donorCounts, listOffsets, donorRanks, donorIndices, receiverAccepted,
    ];
    const createPass = (
      module: GPUShaderModule, entryPoint: string, label: string, bindings: number[],
    ): AbsorptionPass => {
      const pipeline = device.createComputePipeline({
        label, layout: "auto", compute: { module, entryPoint },
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({
          binding, resource: { buffer: resources[binding] },
        })),
      });
      return { pipeline, bindGroup };
    };

    const fullScanModule = device.createShaderModule({
      label: "absorption full scan shader", code: absorptionFullScanWgsl,
    });
    const donorModule = device.createShaderModule({
      label: "absorption donor lists shader", code: absorptionDonorListsWgsl,
    });
    this.fullScan = createPass(fullScanModule, "applyAbsorption", "absorption pass", [0, 1, 2, 3]);
    const donorPass = (entryPoint: string, bindings: number[]) =>
      createPass(donorModule, entryPoint, `absorption/donor-lists/${entryPoint}`, bindings);
    this.countDonors = donorPass("countDonors", [2, 3, 4, 6]);
    this.scatterDonors = donorPass("scatterDonors", [2, 3, 5, 6, 7]);
    this.resolveReceivers = donorPass("resolveReceivers", [0, 1, 2, 3, 4, 5, 7, 8]);
    this.updateDonors = donorPass("updateDonors", [0, 1, 2, 3, 8]);
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["absorption"],
  ): void {
    switch (algorithm) {
      case "full-scan": {
        this.encodePass(encoder, this.fullScan);
        break;
      }
      case "donor-lists": {
        encoder.clearBuffer(this.donorCounts);
        encoder.copyBufferToBuffer(this.params, 12, this.scanCount, 0, 4);
        this.encodePass(encoder, this.countDonors);
        this.scan.encode(encoder);
        this.encodePass(encoder, this.scatterDonors);
        this.encodePass(encoder, this.resolveReceivers);
        this.encodePass(encoder, this.updateDonors);
        break;
      }
    }
  }

  private encodePass(encoder: GPUCommandEncoder, step: AbsorptionPass): void {
    const pass = encoder.beginComputePass({ label: step.pipeline.label });
    pass.setPipeline(step.pipeline);
    pass.setBindGroup(0, step.bindGroup);
    pass.dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
    pass.end();
  }

  public destroy(): void {
    this.scan.destroy();
    for (const buffer of this.scratchBuffers) buffer.destroy();
  }
}