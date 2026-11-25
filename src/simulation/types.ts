export interface SimulationBuffers {
  readonly bodiesA: GPUBuffer;
  readonly bodiesB: GPUBuffer;
  readonly params: GPUBuffer;
  readonly dispatchArgs: GPUBuffer;
  readonly collisionCandidates: GPUBuffer;
}

export interface SimulationAlgorithms {
  gravity: "naive" | "tiled";
  collision: "all-pairs" | "grid" | "grid-clusters";
  absorption: "full-scan" | "donor-lists";
  compaction: "serial" | "scan";
  explosion: "serial" | "parallel";
}

export const BASELINE_ALGORITHMS: Readonly<SimulationAlgorithms> = {
  gravity: "naive",
  collision: "all-pairs",
  absorption: "full-scan",
  compaction: "serial",
  explosion: "serial",
};

export const OPTIMIZED_ALGORITHMS: Readonly<SimulationAlgorithms> = {
  gravity: "tiled",
  collision: "grid",
  absorption: "donor-lists",
  compaction: "scan",
  explosion: "parallel",
};