import BufferOwner from "../gpu/BufferOwner";
import { createComputePass, encodeComputePass, type ComputePass } from "../gpu/ComputePass";
import PrefixScan from "../gpu/PrefixScan";
import { collisionGridWgsl, collisionAllPairsWgsl } from "../shaders";
import { BODY_DISPATCH_CONSTANTS, BODY_WORKGROUP_SIZE } from "./dispatch";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

const GRID_SIDE = 32;
const CELL_ENTRY_STRIDE = 32;
const GRID_BOUNDS_STRIDE = 32;
const MIN_PAIR_CELL_SIZE = 64;
const CLUSTER_PAIR_STRIDE = 16;

type GridPasses = {
  count: ComputePass;
  query: ComputePass;
};

export default class CollisionStage {
  private readonly allPairs: ComputePass;
  private readonly reduceBodyBounds: ComputePass;
  private readonly finishGridBounds: ComputePass;
  private readonly grid: GridPasses;
  private readonly clusteredGrid: GridPasses;
  private readonly scatterCells: ComputePass;
  private readonly buildClusterPairs: ComputePass;
  private readonly queryClusterPairs: ComputePass;
  private readonly dispatchArgs: GPUBuffer;
  private readonly hotCellDispatch: GPUBuffer;
  private readonly pairDispatch: GPUBuffer;
  private readonly reservedPairs: GPUBuffer;
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
    const pairCapacity = Math.min(capacity * 2, device.limits.maxComputeWorkgroupsPerDimension);
    const clusterPairs = this.bufferOwner.createBuffer({
      label: "collision cluster pairs",
      size: pairCapacity * CLUSTER_PAIR_STRIDE,
      usage: GPUBufferUsage.STORAGE,
    });
    const hotCells = this.bufferOwner.createBuffer({
      label: "collision dense cells",
      size: Math.max(1, Math.floor(capacity / MIN_PAIR_CELL_SIZE)) * u32Size,
      usage: GPUBufferUsage.STORAGE,
    });
    const createDispatch = (label: string): GPUBuffer => {
      const args = this.bufferOwner.createBuffer({
        label,
        size: 3 * u32Size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(args, 0, new Uint32Array([0, 1, 1]));
      return args;
    };
    this.hotCellDispatch = createDispatch("collision dense cell dispatch");
    this.pairDispatch = createDispatch("collision cluster pair dispatch");
    this.reservedPairs = this.bufferOwner.createBuffer({
      label: "collision cluster pair reservations",
      size: u32Size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
    const createGridPasses = (enableClusterPairs: boolean): GridPasses => {
      const gridConstants = { ...constants, MIN_PAIR_CELL_SIZE, CLUSTER_PAIRS_ENABLED: Number(enableClusterPairs) };
      return {
        count: createComputePass(device, {
          label: "collision/grid/count pass",
          compute: { module: gridModule, entryPoint: "countCells", constants: gridConstants },
          buffers: {
            0: bodiesA, 2: params, 3: this.cellCounts, 5: cellRanks, 7: geometry,
            11: hotCells, 12: this.hotCellDispatch,
          },
        }),
        query: createComputePass(device, {
          label: "collision/grid/query pass",
          compute: { module: gridModule, entryPoint: "detectGridCollisions", constants: gridConstants },
          buffers: {
            1: collisionCandidates, 4: cellOffsets, 6: cellEntries, 7: geometry,
            9: clusterPairs, 10: this.reservedPairs,
          },
        }),
      };
    };
    this.grid = createGridPasses(false);
    this.clusteredGrid = createGridPasses(true);
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
    this.scatterCells = createComputePass(device, {
      label: "collision/grid/scatter pass",
      compute: { module: gridModule, entryPoint: "scatterCells", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 0: bodiesA, 1: collisionCandidates, 2: params, 4: cellOffsets, 5: cellRanks, 6: cellEntries },
    });
    this.buildClusterPairs = createComputePass(device, {
      label: "collision/grid/build cluster pairs pass",
      compute: { module: gridModule, entryPoint: "buildClusterPairs", constants },
      buffers: {
        4: cellOffsets, 7: geometry, 9: clusterPairs, 10: this.reservedPairs,
        11: hotCells, 13: this.pairDispatch,
      },
    });
    this.queryClusterPairs = createComputePass(device, {
      label: "collision/grid/cluster pairs pass",
      compute: { module: gridModule, entryPoint: "queryClusterPairs", constants: BODY_DISPATCH_CONSTANTS },
      buffers: { 0: bodiesA, 1: collisionCandidates, 6: cellEntries, 9: clusterPairs, 10: this.reservedPairs },
    });
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["collision"],
  ): void {
    if (algorithm === "all-pairs") {
      encodeComputePass(encoder, this.allPairs, this.dispatchArgs);
      return;
    }

    const clustered = algorithm === "grid-clusters";
    const passes = clustered ? this.clusteredGrid : this.grid;
    encodeComputePass(encoder, this.reduceBodyBounds, this.dispatchArgs);
    encodeComputePass(encoder, this.finishGridBounds, 1);
    encoder.clearBuffer(this.cellCounts);
    if (clustered) {
      encoder.clearBuffer(this.hotCellDispatch, 0, Uint32Array.BYTES_PER_ELEMENT);
      encoder.clearBuffer(this.pairDispatch, 0, Uint32Array.BYTES_PER_ELEMENT);
      encoder.clearBuffer(this.reservedPairs);
    }
    encodeComputePass(encoder, passes.count, this.dispatchArgs);
    this.scan.encode(encoder);
    encodeComputePass(encoder, this.scatterCells, this.dispatchArgs);
    if (clustered) {
      encodeComputePass(encoder, this.buildClusterPairs, this.hotCellDispatch);
    }
    encodeComputePass(encoder, passes.query, this.dispatchArgs);
    if (clustered) {
      encodeComputePass(encoder, this.queryClusterPairs, this.pairDispatch);
    }
  }

  public destroy(): void {
    this.scan.destroy();
    this.bufferOwner.destroy();
  }
}