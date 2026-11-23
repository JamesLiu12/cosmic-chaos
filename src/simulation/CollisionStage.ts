import PrefixScan from "../gpu/PrefixScan";
import { collisionGridWgsl, collisionAllPairsWgsl } from "../shaders";
import type { SimulationAlgorithms, SimulationBuffers } from "./types";

const GRID_SIDE = 32;
const BODY_WORKGROUP_SIZE = 64;
const CELL_ENTRY_STRIDE = 32;
const GRID_BOUNDS_STRIDE = 32;

type CollisionPass = {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
};

export default class CollisionStage {
  private readonly allPairs: CollisionPass;
  private readonly reduceBodyBounds: CollisionPass;
  private readonly finishGridBounds: CollisionPass;
  private readonly countCells: CollisionPass;
  private readonly scatterCells: CollisionPass;
  private readonly queryGrid: CollisionPass;
  private readonly dispatchArgs: GPUBuffer;
  private readonly cellCounts: GPUBuffer;
  private readonly scan: PrefixScan;
  private readonly scratchBuffers: GPUBuffer[] = [];

  constructor(device: GPUDevice, buffers: SimulationBuffers) {
    this.dispatchArgs = buffers.dispatchArgs;
    const u32Size = Uint32Array.BYTES_PER_ELEMENT;
    const capacity = buffers.collisionCandidates.size / u32Size;
    const scanLength = GRID_SIDE ** 3 + 1;

    const createScratch = (label: string, size: number, extraUsage = 0): GPUBuffer => {
      const buffer = device.createBuffer({
        label,
        size,
        usage: GPUBufferUsage.STORAGE | extraUsage,
      });
      this.scratchBuffers.push(buffer);
      return buffer;
    };
    this.cellCounts = createScratch("collision cell counts", scanLength * u32Size, GPUBufferUsage.COPY_DST);
    const cellOffsets = createScratch("collision cell offsets", scanLength * u32Size);
    const cellRanks = createScratch("collision body cells and ranks", capacity * 2 * u32Size);
    const cellEntries = createScratch("collision cell entries", capacity * CELL_ENTRY_STRIDE);
    const geometry = createScratch("collision grid geometry", GRID_BOUNDS_STRIDE);
    const groupBounds = createScratch(
      "collision bounds per workgroup",
      Math.ceil(capacity / BODY_WORKGROUP_SIZE) * GRID_BOUNDS_STRIDE,
    );
    const scanCount = createScratch("collision grid scan length", u32Size, GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(scanCount, 0, new Uint32Array([scanLength]));
    this.scan = new PrefixScan(device, this.cellCounts, cellOffsets, scanCount);

    const resources = [
      buffers.bodiesA,
      buffers.collisionCandidates,
      buffers.params,
      this.cellCounts,
      cellOffsets,
      cellRanks,
      cellEntries,
      geometry,
      groupBounds,
    ];
    const createPass = (
      compute: GPUProgrammableStage,
      label: string,
      bindings: number[],
    ): CollisionPass => {
      const pipeline = device.createComputePipeline({ label, layout: "auto", compute });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({
          binding,
          resource: { buffer: resources[binding] },
        })),
      });
      return { pipeline, bindGroup };
    };

    const allPairsModule = device.createShaderModule({ label: "collision detection shader", code: collisionAllPairsWgsl });
    this.allPairs = createPass(
      { module: allPairsModule, entryPoint: "detectCollisions" },
      "collision detection pass",
      [0, 1, 2],
    );

    const gridModule = device.createShaderModule({ label: "collision grid shader", code: collisionGridWgsl });
    const constants = { GRID_SIDE };
    this.reduceBodyBounds = createPass(
      { module: gridModule, entryPoint: "reduceBodyBounds" },
      "collision/grid/reduce bounds pass",
      [0, 2, 8],
    );
    this.finishGridBounds = createPass(
      { module: gridModule, entryPoint: "finishGridBounds", constants },
      "collision/grid/finish bounds pass",
      [2, 7, 8],
    );
    this.countCells = createPass(
      { module: gridModule, entryPoint: "countCells", constants },
      "collision/grid/count pass",
      [0, 2, 3, 5, 7],
    );
    this.scatterCells = createPass(
      { module: gridModule, entryPoint: "scatterCells" },
      "collision/grid/scatter pass",
      [0, 2, 4, 5, 6],
    );
    this.queryGrid = createPass(
      { module: gridModule, entryPoint: "detectGridCollisions", constants },
      "collision/grid/query pass",
      [0, 1, 2, 4, 6, 7],
    );
  }

  public encode(
    encoder: GPUCommandEncoder,
    algorithm: SimulationAlgorithms["collision"],
  ): void {
    if (algorithm === "grid") {
      this.encodePass(encoder, this.reduceBodyBounds);
      this.encodePass(encoder, this.finishGridBounds, 1);
      encoder.clearBuffer(this.cellCounts);
      this.encodePass(encoder, this.countCells);
      this.scan.encode(encoder);
      this.encodePass(encoder, this.scatterCells);
      this.encodePass(encoder, this.queryGrid);
    } else {
      this.encodePass(encoder, this.allPairs);
    }
  }

  private encodePass(encoder: GPUCommandEncoder, step: CollisionPass, workgroupCount?: number): void {
    const pass = encoder.beginComputePass({ label: step.pipeline.label });
    pass.setPipeline(step.pipeline);
    pass.setBindGroup(0, step.bindGroup);
    if (workgroupCount === undefined) {
      pass.dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
    } else {
      pass.dispatchWorkgroups(workgroupCount);
    }
    pass.end();
  }

  public destroy(): void {
    this.scan.destroy();
    for (const buffer of this.scratchBuffers) buffer.destroy();
  }
}