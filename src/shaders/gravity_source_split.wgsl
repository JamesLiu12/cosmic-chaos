override SOURCE_SEGMENTS: u32;

@group(0) @binding(3)
var<storage, read_write> partialAccelerations: array<vec3<f32>>;

@group(0) @binding(4)
var<storage, read_write> splitDispatch: array<u32, 3>;

@compute @workgroup_size(1)
fn prepareGravitySplit() {
    splitDispatch = array<u32, 3>(bodyWorkgroupCount(params.bodyCount), SOURCE_SEGMENTS, 1u);
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn accumulateGravitySplit(
    @builtin(global_invocation_id) invocationId: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let i = invocationId.x;
    let segment = groupId.y;
    let count = params.bodyCount;
    let isValidBody = i < count;

    var body = Body();
    if (isValidBody) {
        body = inputBodies[i];
    }

    let tileCount = bodyWorkgroupCount(count);
    let firstSource = (tileCount * segment / SOURCE_SEGMENTS) * BODY_WORKGROUP_SIZE;
    let sourceEnd = min((tileCount * (segment + 1u) / SOURCE_SEGMENTS) * BODY_WORKGROUP_SIZE, count);
    let acceleration = accumulateGravityTiles(body, isValidBody, localIndex, firstSource, sourceEnd);
    if (isValidBody) {
        partialAccelerations[segment * count + i] = acceleration;
    }
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn integrateGravitySplit(@builtin(global_invocation_id) invocationId: vec3<u32>) {
    let i = invocationId.x;
    let count = params.bodyCount;
    if (i >= count) {
        return;
    }

    var acceleration = vec3<f32>(0.0);
    for (var segment = 0u; segment < SOURCE_SEGMENTS; segment++) {
        acceleration += partialAccelerations[segment * count + i];
    }
    outputBodies[i] = integrateGravity(inputBodies[i], acceleration);
}