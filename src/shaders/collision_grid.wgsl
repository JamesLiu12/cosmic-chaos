override GRID_SIDE: u32;

const BODY_WORKGROUP_SIZE: u32 = 64u;
const MAX_POSITION: f32 = 3.4e+38;
const MIN_GRID_SPAN: f32 = 0.0001;

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

const EMPTY_BOUNDS = BodyBounds(vec3<f32>(MAX_POSITION), 0.0, vec3<f32>(-MAX_POSITION));

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

var<workgroup> boundsScratch: array<BodyBounds, BODY_WORKGROUP_SIZE>;

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
    let groupCount = (params.bodyCount + BODY_WORKGROUP_SIZE - 1u) / BODY_WORKGROUP_SIZE;
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

@compute @workgroup_size(64)
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
}

@compute @workgroup_size(64)
fn scatterCells(@builtin(global_invocation_id) id: vec3<u32>) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let body = bodies[bodyIndex];
    if (body.absorptionCooldown > 0u) {
        return;
    }

    let cellRank = cellRanks[bodyIndex];
    let slot = cellOffsets[cellRank.x] + cellRank.y;
    cellEntries[slot] = CellEntry(body.position, body.radius, body.mass, bodyIndex);
}

@compute @workgroup_size(64)
fn detectGridCollisions(@builtin(global_invocation_id) id: vec3<u32>) {
    let bodyIndex = id.x;
    if (bodyIndex >= params.bodyCount) {
        return;
    }

    let body = bodies[bodyIndex];
    if (body.absorptionCooldown > 0u) {
        collisionCandidates[bodyIndex] = bodyIndex;
        return;
    }

    let geometry = gridGeometry;
    let reach = vec3<f32>(body.radius + geometry.maxRadius);
    let padding = (abs(body.position) + reach + vec3<f32>(1.0)) * 0.000001;
    let firstCell = gridCell(body.position - reach - padding, geometry);
    let lastCell = gridCell(body.position + reach + padding, geometry);

    var candidate = bodyIndex;
    var candidateMass = body.mass;
    for (var z = firstCell.z; z <= lastCell.z; z++) {
        for (var y = firstCell.y; y <= lastCell.y; y++) {
            for (var x = firstCell.x; x <= lastCell.x; x++) {
                let bucket = cellIndex(vec3<u32>(x, y, z));
                let end = cellOffsets[bucket + 1u];
                for (var slot = cellOffsets[bucket]; slot < end; slot++) {
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

    collisionCandidates[bodyIndex] = candidate;
}
