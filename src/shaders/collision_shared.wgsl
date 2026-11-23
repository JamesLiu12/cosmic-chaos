@group(0) @binding(0)
var<storage, read> bodies: array<Body>;

@group(0) @binding(1)
var<storage, read_write> collisionCandidates: array<u32>;

@group(0) @binding(2)
var<uniform> params: SimulationUniforms;

fn hasHigherCollisionPriority(mass: f32, index: u32, candidateMass: f32, candidate: u32) -> bool {
    return mass > candidateMass || (mass == candidateMass && index < candidate);
}

fn spheresOverlap(
    position: vec3<f32>,
    radius: f32,
    otherPosition: vec3<f32>,
    otherRadius: f32,
) -> bool {
    let distance = otherPosition - position;
    let radiusSum = radius + otherRadius;
    return dot(distance, distance) <= radiusSum * radiusSum;
}
