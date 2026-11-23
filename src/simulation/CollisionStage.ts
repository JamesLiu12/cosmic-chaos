import BufferOwner from "../gpu/BufferOwner";
import { createComputePass, encodeComputePass, type ComputePass } from "../gpu/ComputePass";
import PrefixScan from "../gpu/PrefixScan";
import { collisionGridWgsl, collisionAllPairsWgsl } from "../shaders";
import { BODY_DISPATCH_CONSTANTS, BODY_WORKGROUP_SIZE } from "./dispatch";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

const GRID_SIDE = 32;
const CELL_ENTRY_STRIDE = 32;
const GRID_BOUNDS_STRIDE = 32;

export default class CollisionStage {
  private readonly allPairs: ComputePass;
  private readonly reduceBodyBounds: ComputePass;
  private readonly finishGridBounds: ComputePass;
  private readonly countCells: ComputePass;
  private readonly scatterCells: ComputePass;
  private readonly queryGrid: ComputePass;
  private readonly dispatchArgs: GPUBuffer;
  private readonly cellCounts: GPUBuffer;
  private readonly scan: PrefixScan;
  private readonly bufferOwner: BufferOwner;

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.bufferOwner = new BufferOwner(device);
    this.dispatchArgs = buffers.dispatchArgs;
    const u32Size = Uint32Array.BYTES_PER_ELEMENT;
    const capacity = buffers.collisionCandidates.size / u32Size;
    const scanLength = GRID_SIDE ** 3 + 1;

    this.cellCounts = this.bufferOwner.createBuffer({
      label: "collision cell counts",
      size: scanLength * u32Size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const cellOffsets = this.bufferOwner.createBuffer({
      label: "collision cell offsets",
      size: scanLength * u32Size,
      usage: GPUBufferUsage.STORAGE,
    });
    const cellRanks = this.bufferOwner.createBuffer({
      label: "collision body cells and ranks",
      size: capacity * 2 * u32Size,
      usage: GPUBufferUsage.STORAGE,
    });
    const cellEntries = this.bufferOwner.createBuffer({
      label: "collision cell entries",
      size: capacity * CELL_ENTRY_STRIDE,
      usage: GPUBufferUsage.STORAGE,
    });
    const geometry = this.bufferOwner.createBuffer({
      label: "collision grid geometry",
      size: GRID_BOUNDS_STRIDE,
      usage: GPUBufferUsage.STORAGE,
    });
    const groupBounds = this.bufferOwner.createBuffer({
      label: "collision bounds per workgroup",
      size: Math.ceil(capacity / BODY_WORKGROUP_SIZE) * GRID_BOUNDS_STRIDE,
      usage: GPUBufferUsage.STORAGE,
    });
    const scanCount = this.bufferOwner.createBuffer({
      label: "collision grid scan length",
      size: u32Size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(scanCount, 0, new Uint32Array([scanLength]));
    this.scan = new PrefixScan(device, this.cellCounts, cellOffsets, scanCount);

    const { bodiesA, collisionCandidates, params } = buffers;
    const allPairsModule = device.createShaderModule({ label: "collision detection shader", code: collisionAllPairsWgsl });
    this.allPairs = createComputePass(device, {
      label: "collision detection pass",
      compute: { module: allPairsModule, entryPoint: "detectCollisions", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 0: bodiesA, 1: collisionCandidates, 2: params },
    });

    const gridModule = device.createShaderModule({ label: "collision grid shader", code: collisionGridWgsl });
    const constants = { ...BODY_DISPATCH_CONSTANTS, GRID_SIDE };
    this.reduceBodyBounds = createComputePass(device, {
      label: "collision/grid/reduce bounds pass",
      compute: { module: gridModule, entryPoint: "reduceBodyBounds", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 0: bodiesA, 2: params, 8: groupBounds },
    });
    this.finishGridBounds = createComputePass(device, {
      label: "collision/grid/finish bounds pass",
      compute: { module: gridModule, entryPoint: "finishGridBounds", constants },
      buffers: { 2: params, 7: geometry, 8: groupBounds },
    });
    this.countCells = createComputePass(device, {
      label: "collision/grid/count pass",
      compute: { module: gridModule, entryPoint: "countCells", constants },
      buffers: { 0: bodiesA, 2: params, 3: this.cellCounts, 5: cellRanks, 7: geometry },
    });
    this.scatterCells = createComputePass(device, {
      label: "collision/grid/scatter pass",
      compute: { module: gridModule, entryPoint: "scatterCells", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 0: bodiesA, 2: params, 4: cellOffsets, 5: cellRanks, 6: cellEntries },
    });
    this.queryGrid = createComputePass(device, {
      label: "collision/grid/query pass",
      compute: { module: gridModule, entryPoint: "detectGridCollisions", constants },
      buffers: { 0: bodiesA, 1: collisionCandidates, 2: params, 4: cellOffsets, 6: cellEntries, 7: geometry },
    });
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["collision"],
  ): void {
    if (algorithm === "grid") {
      encodeComputePass(encoder, this.reduceBodyBounds, this.dispatchArgs);
      encodeComputePass(encoder, this.finishGridBounds, 1);
      encoder.clearBuffer(this.cellCounts);
      encodeComputePass(encoder, this.countCells, this.dispatchArgs);
      this.scan.encode(encoder);
      encodeComputePass(encoder, this.scatterCells, this.dispatchArgs);
      encodeComputePass(encoder, this.queryGrid, this.dispatchArgs);
    } else {
      encodeComputePass(encoder, this.allPairs, this.dispatchArgs);
    }
  }

  public destroy(): void {
    this.scan.destroy();
    this.bufferOwner.destroy();
  }
}