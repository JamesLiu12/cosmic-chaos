import BufferOwner from "../gpu/BufferOwner";
import { createComputePass, encodeComputePass, type ComputePass } from "../gpu/ComputePass";
import PrefixScan from "../gpu/PrefixScan";
import { explosionParallelWgsl, explosionSerialWgsl } from "../shaders";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

export default class ExplosionStage {
  private readonly buffers: SimulationBuffers;
  private readonly serial: ComputePass;
  private readonly bufferOwner: BufferOwner;
  private readonly bodyCount: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly dispatches: GPUBuffer;
  private readonly eventScan: PrefixScan;
  private readonly budgetScan: PrefixScan;
  private readonly classify: ComputePass;
  private readonly pack: ComputePass;
  private readonly allocate: ComputePass;
  private readonly respond: ComputePass;
  private readonly emit: ComputePass;
  private readonly publish: ComputePass;

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.bufferOwner = new BufferOwner(device);
    this.buffers = buffers;
    const { bodiesA, bodiesB, params, collisionCandidates } = buffers;
    this.serial = createComputePass(device, {
      label: "explosion pipeline",
      bindGroupLabel: "explosion bind group",
      passLabel: "explosion pass",
      compute: {
        module: device.createShaderModule({
          label: "explosion shader",
          code: explosionSerialWgsl,
        }),
        entryPoint: "explodeBodies",
      },
      buffers: { 0: bodiesA, 1: collisionCandidates, 2: params },
    });
    const bodyArrayBytes = collisionCandidates.size;
    this.bodyCount = this.bufferOwner.createBuffer({
      label: "explosion input count",
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const flags = this.bufferOwner.createBuffer({
      label: "explosion event flags",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const ranks = this.bufferOwner.createBuffer({
      label: "explosion event ranks",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const parents = this.bufferOwner.createBuffer({
      label: "explosion event parents",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const requests = this.bufferOwner.createBuffer({
      label: "explosion append requests",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const offsets = this.bufferOwner.createBuffer({
      label: "explosion append offsets",
      size: bodyArrayBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.state = this.bufferOwner.createBuffer({
      label: "explosion event count, approved end, final count",
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.dispatches = this.bufferOwner.createBuffer({
      label: "explosion response, emission, publish args",
      size: 36,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });
    this.eventScan = new PrefixScan(device, flags, ranks, this.bodyCount);
    this.budgetScan = new PrefixScan(device, requests, offsets, this.state);

    const module = device.createShaderModule({ label: "parallel explosion shader", code: explosionParallelWgsl });
    this.classify = createComputePass(device, {
      label: "explosion/parallel/classify",
      compute: { module, entryPoint: "classify" },
      buffers: { 0: bodiesA, 2: params, 4: flags },
    });
    this.pack = createComputePass(device, {
      label: "explosion/parallel/packEvents",
      compute: { module, entryPoint: "packEvents" },
      buffers: { 0: bodiesA, 2: params, 4: flags, 5: ranks, 6: parents, 7: requests, 9: this.state },
    });
    this.allocate = createComputePass(device, {
      label: "explosion/parallel/allocateEvents",
      compute: { module, entryPoint: "allocateEvents" },
      buffers: { 0: bodiesA, 2: params, 7: requests, 8: offsets, 9: this.state, 10: this.dispatches },
    });
    this.respond = createComputePass(device, {
      label: "explosion/parallel/applyResponses",
      compute: { module, entryPoint: "applyResponses" },
      buffers: { 0: bodiesA, 1: bodiesB, 2: params, 6: parents, 7: requests, 9: this.state },
    });
    this.emit = createComputePass(device, {
      label: "explosion/parallel/emitFragments",
      compute: { module, entryPoint: "emitFragments" },
      buffers: { 0: bodiesA, 1: bodiesB, 2: params, 6: parents, 7: requests, 8: offsets, 9: this.state },
    });
    this.publish = createComputePass(device, {
      label: "explosion/parallel/publishBodies",
      compute: { module, entryPoint: "publishBodies" },
      buffers: { 0: bodiesA, 1: bodiesB, 2: params, 3: collisionCandidates, 9: this.state, 11: this.bodyCount },
    });
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["explosion"],
  ): void {
    switch (algorithm) {
      case "serial": {
        encodeComputePass(encoder, this.serial, 1);
        break;
      }
      case "parallel": {
        encoder.copyBufferToBuffer(this.buffers.params, 12, this.bodyCount, 0, 4);
        encodeComputePass(encoder, this.classify, this.buffers.dispatchArgs);
        this.eventScan.encode(encoder);
        encoder.clearBuffer(this.state, 0, 4);
        encodeComputePass(encoder, this.pack, this.buffers.dispatchArgs);
        this.budgetScan.encode(encoder);
        encodeComputePass(encoder, this.allocate, 1);
        encodeComputePass(encoder, this.respond, this.dispatches);
        encodeComputePass(encoder, this.emit, this.dispatches, 12);
        encodeComputePass(encoder, this.publish, this.dispatches, 24);
        break;
      }
    }
  }

  public destroy(): void {
    this.eventScan.destroy();
    this.budgetScan.destroy();
    this.bufferOwner.destroy();
  }
}
