@group(0) @binding(0)
var<storage, read> inputBodies: array<Body>;

@group(0) @binding(1)
var<storage, read_write> outputBodies: array<Body>;

@group(0) @binding(2)
var<uniform> params: SimulationUniforms;

var<workgroup> tilePositionMass: array<vec4<f32>, BODY_WORKGROUP_SIZE>;
var<workgroup> tileRadius: array<f32, BODY_WORKGROUP_SIZE>;

fn accumulateGravityTiles(
    body: Body,
    isValidBody: bool,
    localIndex: u32,
    firstSource: u32,
    sourceEnd: u32,
) -> vec3<f32> {
    var acceleration = vec3<f32>(0.0);

    for (var base = firstSource; base < sourceEnd; base += BODY_WORKGROUP_SIZE) {
        let sourceIndex = base + localIndex;
        if (sourceIndex < sourceEnd) {
            let other = inputBodies[sourceIndex];
            tilePositionMass[localIndex] = vec4<f32>(other.position, other.mass);
            tileRadius[localIndex] = other.radius;
        } else {
            tilePositionMass[localIndex] = vec4<f32>(0.0);
            tileRadius[localIndex] = 0.0;
        }
        workgroupBarrier();

        if (isValidBody) {
            let tileCount = min(BODY_WORKGROUP_SIZE, sourceEnd - base);
            for (var j = 0u; j < tileCount; j++) {
                let other = tilePositionMass[j];
                let r = other.xyz - body.position;
                let radiusSum = body.radius + tileRadius[j];
                let softening = max(params.softeningLength, 0.5 * radiusSum);
                let inverseDistance = inverseSqrt(dot(r, r) + softening * softening);

                acceleration += other.w
                    * r
                    * inverseDistance
                    * inverseDistance
                    * inverseDistance;
            }
        }
        workgroupBarrier();
    }

    return acceleration;
}

@compute @workgroup_size(BODY_WORKGROUP_SIZE)
fn gravityTiled(
    @builtin(global_invocation_id) invocationId: vec3<u32>,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let i = invocationId.x;
    let count = params.bodyCount;
    let isValidBody = i < count;

    var body = Body();
    if (isValidBody) {
        body = inputBodies[i];
    }

    let acceleration = accumulateGravityTiles(body, isValidBody, localIndex, 0u, count);
    if (isValidBody) {
        outputBodies[i] = integrateGravity(body, acceleration);
    }
}