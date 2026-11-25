override GRID_SIDE: u32;

const MAX_POSITION: f32 = 3.4e+38;
const MIN_GRID_SPAN: f32 = 0.0001;
override MIN_PAIR_CELL_SIZE: u32;
override CLUSTER_PAIRS_ENABLED: bool = false;

struct CellEntry {
    position: vec3<f32>,
    radius: f32,
    mass: f32,
    bodyIndex: u32,
}

struct BodyBounds {
    minPosition: vec3<f32>,
    maxRadius: f32,
    maxPosition: vec3<f32>,
}

struct GridGeometry {
    origin: vec3<f32>,
    maxRadius: f32,
    cellScale: vec3<f32>,
}

struct ClusterPair {
    queryStart: u32,
    queryEnd: u32,
    candidateStart: u32,
    candidateEnd: u32,
}

struct DispatchArgs {
    x: atomic<u32>,
    y: u32,
    z: u32,
}

const EMPTY_BOUNDS = BodyBounds(vec3<f32>(MAX_POSITION), 0.0, vec3<f32>(-MAX_POSITION));

@group(0) @binding(1)
var<storage, read_write> collisionCandidates: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> cellCounts: array<atomic<u32>>;

@group(0) @binding(4)
var<storage, read> cellOffsets: array<u32>;

@group(0) @binding(5)
var<storage, read_write> cellRanks: array<vec2<u32>>;

@group(0) @binding(6)
var<storage, read_write> cellEntries: array<CellEntry>;

@group(0) @binding(7)
var<storage, read_write> gridGeometry: GridGeometry;

@group(0) @binding(8)
var<storage, read_write> groupBounds: array<BodyBounds>;

@group(0) @binding(9)
var<storage, read_write> clusterPairs: array<ClusterPair>;

@group(0) @binding(10)
var<storage, read_write> reservedPairs: atomic<u32>;

@group(0) @binding(11)
var<storage, read_write> hotCells: array<u32>;

@group(0) @binding(12)
var<storage, read_write> hotCellDispatch: DispatchArgs;

@group(0) @binding(13)
var<storage, read_write> pairDispatch: DispatchArgs;

var<workgroup> boundsScratch: array<BodyBounds, BODY_WORKGROUP_SIZE>;
var<workgroup> candidateTile: array<CellEntry, BODY_WORKGROUP_SIZE>;

fn mergeBounds(left: BodyBounds, right: BodyBounds) -> BodyBounds {
    return BodyBounds(
        min(left.minPosition, right.minPosition),
        max(left.maxRadius, right.maxRadius),
        max(left.maxPosition, right.maxPosition),
    );
}

fn reduceBounds(bounds: BodyBounds, localIndex: u32) -> BodyBounds {
    boundsScratch[localIndex] = bounds;
    for (var stride = BODY_WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        workgroupBarrier();
        if (localIndex < stride) {
            boundsScratch[localIndex] = mergeBounds(
                boundsScratch[localIndex], boundsScratch[localIndex + stride],
            );
        }
    }
    workgroupBarrier();
    return boundsScratch[0];
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn reduceBodyBounds(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(local_invocation_index) localIndex: u32,
) {
    var bounds = EMPTY_BOUNDS;
    if (id.x < params.bodyCount) {
        let body = bodies[id.x];
        if (body.absorptionCooldown == 0u) {
            bounds = BodyBounds(body.position, body.radius, body.position);
        }
    }

    let reduced = reduceBounds(bounds, localIndex);
    if (localIndex == 0u) {
        groupBounds[groupId.x] = reduced;
    }
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn finishGridBounds(@builtin(local_invocation_index) localIndex: u32) {
    let groupCount = bodyWorkgroupCount(params.bodyCount);
    var bounds = EMPTY_BOUNDS;
    for (var groupIndex = localIndex; groupIndex < groupCount; groupIndex += BODY_WORKGROUP_SIZE) {
        bounds = mergeBounds(bounds, groupBounds[groupIndex]);
    }

    let reduced = reduceBounds(bounds, localIndex);
    if (localIndex == 0u) {
        if (any(reduced.minPosition > reduced.maxPosition)) {
            gridGeometry = GridGeometry();
        } else {
            let span = max(reduced.maxPosition - reduced.minPosition, vec3<f32>(MIN_GRID_SPAN));
            let cellScale = vec3<f32>(f32(GRID_SIDE)) / span;
            gridGeometry = GridGeometry(reduced.minPosition, reduced.maxRadius, cellScale);
        }
    }
}

fn gridCell(position: vec3<f32>, geometry: GridGeometry) -> vec3<u32> {
    let cell = (position - geometry.origin) * geometry.cellScale;
    return vec3<u32>(clamp(cell, vec3<f32>(0.0), vec3<f32>(f32(GRID_SIDE - 1u))));
}

fn cellIndex(cell: vec3<u32>) -> u32 {
    return cell.x + GRID_SIDE * (cell.y + GRID_SIDE * cell.z);
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn countCells(@builtin(global_invocation_id) id: vec3<u32>) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let body = bodies[bodyIndex];
    if (body.absorptionCooldown > 0u) {
        return;
    }

    let bucket = cellIndex(gridCell(body.position, gridGeometry));
    let localRank = atomicAdd(&cellCounts[bucket], 1u);
    cellRanks[bodyIndex] = vec2<u32>(bucket, localRank);
    if (CLUSTER_PAIRS_ENABLED && localRank + 1u == MIN_PAIR_CELL_SIZE) {
        let hotIndex = atomicAdd(&hotCellDispatch.x, 1u);
        hotCells[hotIndex] = bucket;
    }
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn scatterCells(@builtin(global_invocation_id) id: vec3<u32>) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let body = bodies[bodyIndex];
    if (body.absorptionCooldown > 0u) {
        atomicStore(&collisionCandidates[bodyIndex], bodyIndex);
        return;
    }

    let cellRank = cellRanks[bodyIndex];
    let slot = cellOffsets[cellRank.x] + cellRank.y;
    cellEntries[slot] = CellEntry(body.position, body.radius, body.mass, bodyIndex);
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn buildClusterPairs(
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let candidateBucket = hotCells[groupId.x];
    let candidateStart = cellOffsets[candidateBucket];
    let candidateEnd = cellOffsets[candidateBucket + 1u];
    let candidateTileCount = bodyWorkgroupCount(candidateEnd - candidateStart);
    let cell = vec3<u32>(
        candidateBucket % GRID_SIDE,
        (candidateBucket / GRID_SIDE) % GRID_SIDE,
        candidateBucket / (GRID_SIDE * GRID_SIDE),
    );

    let geometry = gridGeometry;
    let gridMax = geometry.origin + vec3<f32>(f32(GRID_SIDE)) / geometry.cellScale;
    let reach = vec3<f32>(2.0 * geometry.maxRadius);
    let padding = (max(abs(geometry.origin), abs(gridMax)) + reach + vec3<f32>(1.0)) * 0.000001;
    let maxCell = vec3<u32>(GRID_SIDE - 1u);
    let cellReach = vec3<u32>(min(
        ceil((reach + padding) * geometry.cellScale),
        vec3<f32>(maxCell),
    ));
    let firstCell = cell - min(cell, cellReach);
    let lastCell = min(cell + cellReach, maxCell);
    let span = lastCell - firstCell + vec3<u32>(1u);
    let cellCount = span.x * span.y * span.z;
    for (var offset = localIndex; offset < cellCount; offset += BODY_WORKGROUP_SIZE) {
        let queryCell = firstCell + vec3<u32>(
            offset % span.x,
            (offset / span.x) % span.y,
            offset / (span.x * span.y),
        );
        let queryBucket = cellIndex(queryCell);
        let queryStart = cellOffsets[queryBucket];
        let queryEnd = cellOffsets[queryBucket + 1u];
        if (queryStart == queryEnd) {
            continue;
        }

        let pairCount = bodyWorkgroupCount(queryEnd - queryStart) * candidateTileCount;
        let firstPair = atomicAdd(&reservedPairs, pairCount);
        let pairEnd = firstPair + pairCount;
        if (pairEnd > arrayLength(&clusterPairs)) {
            continue;
        }

        var pairIndex = firstPair;
        for (var querySlot = queryStart; querySlot < queryEnd; querySlot += BODY_WORKGROUP_SIZE) {
            for (var candidateSlot = candidateStart; candidateSlot < candidateEnd; candidateSlot += BODY_WORKGROUP_SIZE) {
                clusterPairs[pairIndex] = ClusterPair(
                    querySlot, min(querySlot + BODY_WORKGROUP_SIZE, queryEnd),
                    candidateSlot, min(candidateSlot + BODY_WORKGROUP_SIZE, candidateEnd),
                );
                pairIndex++;
            }
        }
        atomicMax(&pairDispatch.x, pairEnd);
    }
}

fn clusterPairOverflow() -> bool {
    return atomicLoad(&reservedPairs) > arrayLength(&clusterPairs);
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn detectGridCollisions(@builtin(global_invocation_id) id: vec3<u32>) {
    let entryIndex = id.x;
    let entryCount = cellOffsets[GRID_SIDE * GRID_SIDE * GRID_SIDE];
    if (entryIndex >= entryCount) {
        return;
    }

    let splitPairs = CLUSTER_PAIRS_ENABLED && !clusterPairOverflow();
    let geometry = gridGeometry;
    let body = cellEntries[entryIndex];
    let reach = vec3<f32>(body.radius + geometry.maxRadius);
    let padding = (abs(body.position) + reach + vec3<f32>(1.0)) * 0.000001;
    let firstCell = gridCell(body.position - reach - padding, geometry);
    let lastCell = gridCell(body.position + reach + padding, geometry);

    var candidate = body.bodyIndex;
    var candidateMass = body.mass;
    for (var z = firstCell.z; z <= lastCell.z; z++) {
        for (var y = firstCell.y; y <= lastCell.y; y++) {
            for (var x = firstCell.x; x <= lastCell.x; x++) {
                let bucket = cellIndex(vec3<u32>(x, y, z));
                let start = cellOffsets[bucket];
                let end = cellOffsets[bucket + 1u];
                if (splitPairs && end - start >= MIN_PAIR_CELL_SIZE) {
                    continue;
                }
                for (var slot = start; slot < end; slot++) {
                    let other = cellEntries[slot];
                    if (hasHigherCollisionPriority(other.mass, other.bodyIndex, candidateMass, candidate) &&
                        spheresOverlap(body.position, body.radius, other.position, other.radius)) {
                        candidate = other.bodyIndex;
                        candidateMass = other.mass;
                    }
                }
            }
        }
    }

    atomicStore(&collisionCandidates[body.bodyIndex], candidate);
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn queryClusterPairs(
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let pair = clusterPairs[groupId.x];
    let candidateSlot = pair.candidateStart + localIndex;
    if (candidateSlot < pair.candidateEnd) {
        candidateTile[localIndex] = cellEntries[candidateSlot];
    }
    workgroupBarrier();

    let entryIndex = pair.queryStart + localIndex;
    if (entryIndex >= pair.queryEnd || clusterPairOverflow()) {
        return;
    }

    let body = cellEntries[entryIndex];
    var candidate = body.bodyIndex;
    var candidateMass = body.mass;
    let tileCount = pair.candidateEnd - pair.candidateStart;
    for (var slot = 0u; slot < tileCount; slot++) {
        let other = candidateTile[slot];
        if (hasHigherCollisionPriority(other.mass, other.bodyIndex, candidateMass, candidate) &&
            spheresOverlap(body.position, body.radius, other.position, other.radius)) {
            candidate = other.bodyIndex;
            candidateMass = other.mass;
        }
    }

    if (candidate == body.bodyIndex) {
        return;
    }

    var current = atomicLoad(&collisionCandidates[body.bodyIndex]);
    while (hasHigherCollisionPriority(candidateMass, candidate, bodies[current].mass, current)) {
        let result = atomicCompareExchangeWeak(&collisionCandidates[body.bodyIndex], current, candidate);
        if (result.exchanged) {
            break;
        }
        current = result.old_value;
    }
}
