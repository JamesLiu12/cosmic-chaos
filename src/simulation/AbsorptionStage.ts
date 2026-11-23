import BufferOwner from "../gpu/BufferOwner";
import { createComputePass, encodeComputePass, type ComputePass } from "../gpu/ComputePass";
import PrefixScan from "../gpu/PrefixScan";
import { absorptionDonorListsWgsl, absorptionFullScanWgsl } from "../shaders";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

export default class AbsorptionStage {
  private readonly fullScan: ComputePass;
  private readonly countDonors: ComputePass;
  private readonly scatterDonors: ComputePass;
  private readonly resolveReceivers: ComputePass;
  private readonly updateDonors: ComputePass;
  private readonly dispatchArgs: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly donorCounts: GPUBuffer;
  private readonly scanCount: GPUBuffer;
  private readonly scan: PrefixScan;
  private readonly bufferOwner: BufferOwner;

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.bufferOwner = new BufferOwner(device);
    this.dispatchArgs = buffers.dispatchArgs;
    this.params = buffers.params;
    const { bodiesA, bodiesB, collisionCandidates, params } = buffers;
    const bodyArrayBytes = collisionCandidates.size;
    this.donorCounts = this.bufferOwner.createBuffer({
      label: "absorption donor counts",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.scanCount = this.bufferOwner.createBuffer({
      label: "absorption scan input count",
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const listOffsets = this.bufferOwner.createBuffer({
      label: "absorption donor list offsets",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const donorRanks = this.bufferOwner.createBuffer({
      label: "absorption donor ranks",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const donorIndices = this.bufferOwner.createBuffer({
      label: "absorption donor indices",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const receiverAccepted = this.bufferOwner.createBuffer({
      label: "absorption receiver decisions",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.scan = new PrefixScan(device, this.donorCounts, listOffsets, this.scanCount);

    const fullScanModule = device.createShaderModule({
      label: "absorption full scan shader", code: absorptionFullScanWgsl,
    });
    const donorModule = device.createShaderModule({
      label: "absorption donor lists shader", code: absorptionDonorListsWgsl,
    });
    this.fullScan = createComputePass(device, {
      label: "absorption pass",
      compute: { module: fullScanModule, entryPoint: "applyAbsorption" },
      buffers: { 0: bodiesA, 1: bodiesB, 2: collisionCandidates, 3: params },
    });
    this.countDonors = createComputePass(device, {
      label: "absorption/donor-lists/countDonors",
      compute: { module: donorModule, entryPoint: "countDonors" },
      buffers: { 2: collisionCandidates, 3: params, 4: this.donorCounts, 6: donorRanks },
    });
    this.scatterDonors = createComputePass(device, {
      label: "absorption/donor-lists/scatterDonors",
      compute: { module: donorModule, entryPoint: "scatterDonors" },
      buffers: { 2: collisionCandidates, 3: params, 5: listOffsets, 6: donorRanks, 7: donorIndices },
    });
    this.resolveReceivers = createComputePass(device, {
      label: "absorption/donor-lists/resolveReceivers",
      compute: { module: donorModule, entryPoint: "resolveReceivers" },
      buffers: {
        0: bodiesA,
        1: bodiesB,
        2: collisionCandidates,
        3: params,
        4: this.donorCounts,
        5: listOffsets,
        7: donorIndices,
        8: receiverAccepted,
      },
    });
    this.updateDonors = createComputePass(device, {
      label: "absorption/donor-lists/updateDonors",
      compute: { module: donorModule, entryPoint: "updateDonors" },
      buffers: { 0: bodiesA, 1: bodiesB, 2: collisionCandidates, 3: params, 8: receiverAccepted },
    });
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["absorption"],
  ): void {
    switch (algorithm) {
      case "full-scan": {
        encodeComputePass(encoder, this.fullScan, this.dispatchArgs);
        break;
      }
      case "donor-lists": {
        encoder.clearBuffer(this.donorCounts);
        encoder.copyBufferToBuffer(this.params, 12, this.scanCount, 0, 4);
        encodeComputePass(encoder, this.countDonors, this.dispatchArgs);
        this.scan.encode(encoder);
        encodeComputePass(encoder, this.scatterDonors, this.dispatchArgs);
        encodeComputePass(encoder, this.resolveReceivers, this.dispatchArgs);
        encodeComputePass(encoder, this.updateDonors, this.dispatchArgs);
        break;
      }
    }
  }

  public destroy(): void {
    this.scan.destroy();
    this.bufferOwner.destroy();
  }
}
